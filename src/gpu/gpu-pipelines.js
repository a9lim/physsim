/**
 * @fileoverview GPU compute pipeline creation for Phase 2 shaders.
 *
 * Each function creates a pipeline + bind group layout for one shader.
 * Shaders are loaded via fetch() and prepended with common.wgsl.
 */

/** Fetch a WGSL shader file relative to src/gpu/shaders/ */
async function fetchShader(filename) {
    const resp = await fetch(`src/gpu/shaders/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    return resp.text();
}

/**
 * Create all Phase 2 compute pipelines.
 * Returns an object with pipeline + bindGroupLayout for each shader.
 */
export async function createPhase2Pipelines(device) {
    const commonWGSL = await fetchShader('common.wgsl');

    async function makePipeline(label, filename, layouts) {
        const code = commonWGSL + '\n' + await fetchShader(filename);
        const module = device.createShaderModule({ label, code });

        const bindGroupLayouts = layouts.map((entries, groupIdx) =>
            device.createBindGroupLayout({
                label: `${label}_group${groupIdx}`,
                entries: entries.map((entry, i) => ({
                    binding: i,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: entry },
                })),
            })
        );

        const pipeline = device.createComputePipeline({
            label,
            layout: device.createPipelineLayout({ bindGroupLayouts }),
            compute: { module, entryPoint: 'main' },
        });

        return { pipeline, bindGroupLayouts };
    }

    // --- resetForces ---
    const resetForces = await makePipeline('resetForces', 'reset-forces.wgsl', [
        ['uniform', 'storage', 'storage', 'storage', 'storage', 'storage',
         'storage', 'storage', 'storage', 'storage', 'storage', 'storage'],
    ]);

    // --- cacheDerived ---
    const cacheDerived = await makePipeline('cacheDerived', 'cache-derived.wgsl', [
        ['uniform', 'read-only-storage', 'read-only-storage', 'read-only-storage',
         'read-only-storage', 'read-only-storage',
         'storage', 'storage', 'storage', 'storage', 'storage', 'storage',
         'storage', 'storage', 'storage', 'read-only-storage'],
    ]);

    // --- pairForce (3 bind groups) ---
    const pairForce = await makePipeline('pairForce', 'pair-force.wgsl', [
        // Group 0: uniforms
        ['uniform'],
        // Group 1: particle state (read-only)
        ['read-only-storage', 'read-only-storage', 'read-only-storage', 'read-only-storage',
         'read-only-storage', 'read-only-storage', 'read-only-storage', 'read-only-storage',
         'read-only-storage', 'read-only-storage', 'read-only-storage', 'read-only-storage',
         'read-only-storage', 'read-only-storage', 'read-only-storage'],
        // Group 2: force accumulators (read-write)
        ['storage', 'storage', 'storage', 'storage', 'storage', 'storage',
         'storage', 'storage', 'storage'],
    ]);

    // --- externalFields ---
    const externalFields = await makePipeline('externalFields', 'external-fields.wgsl', [
        ['uniform', 'read-only-storage', 'read-only-storage', 'read-only-storage',
         'storage', 'storage', 'storage', 'storage'],
    ]);

    // --- borisHalfKick ---
    const borisHalfKick = await makePipeline('borisHalfKick', 'boris-half-kick.wgsl', [
        ['uniform', 'storage', 'storage', 'read-only-storage',
         'read-only-storage', 'read-only-storage', 'read-only-storage'],
    ]);

    // --- borisRotate ---
    const borisRotate = await makePipeline('borisRotate', 'boris-rotate.wgsl', [
        ['uniform', 'storage', 'storage', 'read-only-storage',
         'read-only-storage', 'read-only-storage', 'read-only-storage'],
    ]);

    // --- borisDrift ---
    const borisDrift = await makePipeline('borisDrift', 'boris.wgsl', [
        ['uniform', 'storage', 'storage', 'storage', 'storage',
         'read-only-storage', 'storage', 'storage'],
    ]);

    // --- spinOrbit ---
    const spinOrbit = await makePipeline('spinOrbit', 'spin-orbit.wgsl', [
        ['uniform', 'storage', 'storage', 'storage', 'read-only-storage',
         'read-only-storage', 'read-only-storage', 'read-only-storage',
         'read-only-storage', 'read-only-storage', 'read-only-storage',
         'read-only-storage', 'read-only-storage', 'storage', 'storage'],
    ]);

    // --- applyTorques ---
    const applyTorques = await makePipeline('applyTorques', 'apply-torques.wgsl', [
        ['uniform', 'storage', 'read-only-storage', 'read-only-storage',
         'read-only-storage', 'read-only-storage', 'storage'],
    ]);

    return {
        resetForces, cacheDerived, pairForce, externalFields,
        borisHalfKick, borisRotate, borisDrift, spinOrbit, applyTorques,
    };
}

/**
 * Create tree build compute pipelines (Phase 3: GPU Barnes-Hut).
 * 4 entry points from tree-build.wgsl:
 *   computeBounds (workgroup_size 256)
 *   initRoot (workgroup_size 1)
 *   insertParticles (workgroup_size 64)
 *   computeAggregates (workgroup_size 64)
 *
 * All share the same bind group layouts:
 *   Group 0: tree state (nodes, nodeCounter, bounds, visitorFlags) — read-write
 *   Group 1: particle SoA (9 bindings) — read-only
 *   Group 2: uniforms — uniform
 */
export async function createTreeBuildPipelines(device) {
    const code = await fetchShader('tree-build.wgsl');
    const module = device.createShaderModule({ label: 'treeBuild', code });

    // Group 0: tree node buffer + counter + bounds + visitor flags
    const group0Layout = device.createBindGroupLayout({
        label: 'treeBuild_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });

    // Group 1: particle SoA inputs (read-only)
    const group1Layout = device.createBindGroupLayout({
        label: 'treeBuild_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });

    // Group 2: uniforms
    const group2Layout = device.createBindGroupLayout({
        label: 'treeBuild_group2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    const bindGroupLayouts = [group0Layout, group1Layout, group2Layout];
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts });

    const computeBounds = device.createComputePipeline({
        label: 'computeBounds',
        layout: pipelineLayout,
        compute: { module, entryPoint: 'computeBounds' },
    });

    const initRoot = device.createComputePipeline({
        label: 'initRoot',
        layout: pipelineLayout,
        compute: { module, entryPoint: 'initRoot' },
    });

    const insertParticles = device.createComputePipeline({
        label: 'insertParticles',
        layout: pipelineLayout,
        compute: { module, entryPoint: 'insertParticles' },
    });

    const computeAggregates = device.createComputePipeline({
        label: 'computeAggregates',
        layout: pipelineLayout,
        compute: { module, entryPoint: 'computeAggregates' },
    });

    return {
        computeBounds,
        initRoot,
        insertParticles,
        computeAggregates,
        bindGroupLayouts,
    };
}

/**
 * Create tree force (Barnes-Hut walk) compute pipeline.
 * Standalone shader — defines its own SimUniforms and node accessors.
 * Bind groups:
 *   Group 0: nodes (read-only storage) + uniforms
 *   Group 1: particle SoA (15 read-only bindings) + ghostOriginalIdx
 *   Group 2: force accumulators (5 read-write bindings)
 */
export async function createTreeForcePipeline(device) {
    const code = await fetchShader('forces-tree.wgsl');
    const module = device.createShaderModule({ label: 'treeForce', code });

    // Group 0: tree nodes (read-only) + uniforms
    const group0Layout = device.createBindGroupLayout({
        label: 'treeForce_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    // Group 1: particle SoA inputs (16 read-only bindings, including deathMass)
    const group1Entries = [];
    for (let i = 0; i < 16; i++) {
        group1Entries.push({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' },
        });
    }
    const group1Layout = device.createBindGroupLayout({
        label: 'treeForce_group1',
        entries: group1Entries,
    });

    // Group 2: force accumulators (5 read-write bindings)
    const group2Layout = device.createBindGroupLayout({
        label: 'treeForce_group2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });

    const bindGroupLayouts = [group0Layout, group1Layout, group2Layout];
    const pipeline = device.createComputePipeline({
        label: 'treeForce',
        layout: device.createPipelineLayout({ bindGroupLayouts }),
        compute: { module, entryPoint: 'main' },
    });

    return { pipeline, bindGroupLayouts };
}

/**
 * Create collision detection/resolution pipelines (Phase 3).
 * Two entry points from collision.wgsl:
 *   detectCollisions — tree broadphase overlap query
 *   resolveCollisions — process detected pairs (merge/annihilation)
 *
 * Bind groups:
 *   Group 0: tree nodes (read-only) + uniforms
 *   Group 1: particle SoA (read-write for resolve) + ghost mapping
 *   Group 2: collision pair buffer + counters + merge results
 */
export async function createCollisionPipelines(device) {
    const code = await fetchShader('collision.wgsl');
    const module = device.createShaderModule({ label: 'collision', code });

    // Group 0: tree nodes (read-only) + uniforms
    const group0Layout = device.createBindGroupLayout({
        label: 'collision_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    // Group 1: particle SoA (12 bindings)
    // bindings 0-7: read-write (posX, posY, velWX, velWY, angW, mass, baseMass, charge)
    // binding 8: flags (read-write, atomic)
    // binding 9: radius (read-only)
    // binding 10: particleId (read-only)
    // binding 11: ghostOriginalIdx (read-only)
    const group1Layout = device.createBindGroupLayout({
        label: 'collision_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });

    // Group 2: collision pairs + counters + merge results + death metadata
    const group2Layout = device.createBindGroupLayout({
        label: 'collision_group2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });

    const bindGroupLayouts = [group0Layout, group1Layout, group2Layout];
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts });

    const detectCollisions = device.createComputePipeline({
        label: 'detectCollisions',
        layout: pipelineLayout,
        compute: { module, entryPoint: 'detectCollisions' },
    });

    const resolveCollisions = device.createComputePipeline({
        label: 'resolveCollisions',
        layout: pipelineLayout,
        compute: { module, entryPoint: 'resolveCollisions' },
    });

    return { detectCollisions, resolveCollisions, bindGroupLayouts };
}

/**
 * Create dead particle GC compute pipeline (Phase 3).
 * Transitions RETIRED particles to FREE when their signal delay history expires.
 * Single compute pipeline, dispatched once per frame.
 * Bind group:
 *   Group 0: flags (read-write), deathTime (read-only), uniforms, freeStack (read-write), freeTop (read-write)
 */
export async function createDeadGCPipeline(device) {
    const code = await fetchShader('dead-gc.wgsl');
    const module = device.createShaderModule({ label: 'deadGC', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'deadGC_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });

    const pipeline = device.createComputePipeline({
        label: 'deadGC',
        layout: device.createPipelineLayout({ bindGroupLayouts: [group0Layout] }),
        compute: { module, entryPoint: 'main' },
    });

    return { pipeline, bindGroupLayouts: [group0Layout] };
}

/**
 * Create Phase 4 compute pipelines: history, 1PN, radiation, bosons, boson-tree.
 * Each pipeline has its own bind group layouts matching the shader declarations.
 */
export async function createPhase4Pipelines(device) {
    // ── recordHistory (history.wgsl, entry: recordHistory) ──
    // Group 0: uniform + particle state (7 bindings)
    // Group 1: history ring buffers + meta (7 bindings)
    const historyCode = await fetchShader('history.wgsl');
    const historyModule = device.createShaderModule({ label: 'history', code: historyCode });

    const historyG0 = device.createBindGroupLayout({
        label: 'history_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });
    const historyG1 = device.createBindGroupLayout({
        label: 'history_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });
    const historyLayouts = [historyG0, historyG1];
    const recordHistory = {
        pipeline: device.createComputePipeline({
            label: 'recordHistory',
            layout: device.createPipelineLayout({ bindGroupLayouts: historyLayouts }),
            compute: { module: historyModule, entryPoint: 'recordHistory' },
        }),
        bindGroupLayouts: historyLayouts,
    };

    // ── 1PN (onePN.wgsl, entries: compute1PN, vvKick1PN) ──
    // Group 0: uniforms (1 binding)
    // Group 1: particle state (9 read-only)
    // Group 2: forces2 (rw) + f1pnOld (ro) + velWX_rw (rw) + velWY_rw (rw) = 4 bindings
    const onePNCode = await fetchShader('onePN.wgsl');
    const onePNModule = device.createShaderModule({ label: 'onePN', code: onePNCode });

    const onePNG0 = device.createBindGroupLayout({
        label: 'onePN_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const onePNG1 = device.createBindGroupLayout({
        label: 'onePN_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });
    const onePNG2 = device.createBindGroupLayout({
        label: 'onePN_g2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },     // forces2
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // f1pnOld
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },     // velWX_rw
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },     // velWY_rw
        ],
    });
    const onePNLayouts = [onePNG0, onePNG1, onePNG2];
    const onePNPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: onePNLayouts });

    const compute1PN = {
        pipeline: device.createComputePipeline({
            label: 'compute1PN',
            layout: onePNPipelineLayout,
            compute: { module: onePNModule, entryPoint: 'compute1PN' },
        }),
        bindGroupLayouts: onePNLayouts,
    };
    const vvKick1PN = {
        pipeline: device.createComputePipeline({
            label: 'vvKick1PN',
            layout: onePNPipelineLayout,
            compute: { module: onePNModule, entryPoint: 'vvKick1PN' },
        }),
        bindGroupLayouts: onePNLayouts,
    };

    // ── Radiation (radiation.wgsl, entries: lamrorRadiation, hawkingRadiation, pionEmission) ──
    // Group 0: uniforms (1 binding)
    // Group 1: particle state (17 bindings: 0-16)
    // Group 2: radiation accumulators + display (5 bindings)
    // Group 3: photon pool (9) + pion pool (11) + charge_rw (1) = 21 bindings
    const radiationCode = await fetchShader('radiation.wgsl');
    const radiationModule = device.createShaderModule({ label: 'radiation', code: radiationCode });

    const radG0 = device.createBindGroupLayout({
        label: 'radiation_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const radG1Entries = [];
    // bindings 0-1: posX, posY (read)
    // bindings 2-3: velWX, velWY (read_write)
    // binding 4: mass (read_write)
    // binding 5: charge_buf (read)
    // binding 6: flags (read)
    // binding 7: invMass (read_write)
    // binding 8: baseMass (read_write)
    // binding 9: radius (read)
    // binding 10: angW_buf (read)
    // binding 11: particleId (read)
    // bindings 12-13: force_totalX/Y (read)
    // binding 14: jerk_buf (read)
    // bindings 15-16: yukForceX/Y (read)
    const radG1Types = [
        'read-only-storage', 'read-only-storage',
        'storage', 'storage',
        'storage',
        'read-only-storage', 'read-only-storage',
        'storage', 'storage',
        'read-only-storage', 'read-only-storage', 'read-only-storage',
        'read-only-storage', 'read-only-storage', 'read-only-storage',
        'read-only-storage', 'read-only-storage',
    ];
    for (let i = 0; i < radG1Types.length; i++) {
        radG1Entries.push({
            binding: i, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: radG1Types[i] },
        });
    }
    const radG1 = device.createBindGroupLayout({ label: 'radiation_g1', entries: radG1Entries });

    const radG2 = device.createBindGroupLayout({
        label: 'radiation_g2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // radAccum
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // hawkAccum
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // yukawaRadAccum
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // radDisplayX
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // radDisplayY
        ],
    });

    // Group 3: photon pool (0-8) + pion pool (9-19) + charge_rw (20) = 21 bindings
    const radG3Entries = [];
    for (let i = 0; i < 21; i++) {
        radG3Entries.push({
            binding: i, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' },
        });
    }
    const radG3 = device.createBindGroupLayout({ label: 'radiation_g3', entries: radG3Entries });

    const radLayouts = [radG0, radG1, radG2, radG3];
    const radPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: radLayouts });

    const lamrorRadiation = {
        pipeline: device.createComputePipeline({
            label: 'lamrorRadiation',
            layout: radPipelineLayout,
            compute: { module: radiationModule, entryPoint: 'lamrorRadiation' },
        }),
        bindGroupLayouts: radLayouts,
    };
    const hawkingRadiation = {
        pipeline: device.createComputePipeline({
            label: 'hawkingRadiation',
            layout: radPipelineLayout,
            compute: { module: radiationModule, entryPoint: 'hawkingRadiation' },
        }),
        bindGroupLayouts: radLayouts,
    };
    const pionEmission = {
        pipeline: device.createComputePipeline({
            label: 'pionEmission',
            layout: radPipelineLayout,
            compute: { module: radiationModule, entryPoint: 'pionEmission' },
        }),
        bindGroupLayouts: radLayouts,
    };

    // ── Bosons (bosons.wgsl, entries: updatePhotons, updatePions, absorbPhotons, absorbPions, decayPions) ──
    // Group 0: uniforms + aliveCountAtomic (2 bindings)
    // Group 1: particle SoA (11 bindings, read_write)
    // Group 2: photon pool (9 bindings)
    // Group 3: pion pool (11 bindings)
    const bosonsCode = await fetchShader('bosons.wgsl');
    const bosonsModule = device.createShaderModule({ label: 'bosons', code: bosonsCode });

    const bosG0 = device.createBindGroupLayout({
        label: 'bosons_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // aliveCountAtomic
        ],
    });
    // Group 1: 11 particle bindings — mix of read and read_write
    const bosG1Types = [
        'storage', 'storage',       // posX, posY (rw for decay spawn)
        'storage',                   // mass (rw)
        'read-only-storage',         // radius (ro)
        'storage',                   // flags (rw)
        'read-only-storage',         // particleId (ro)
        'storage', 'storage',       // velWX, velWY (rw for absorption)
        'storage',                   // charge_buf (rw)
        'storage',                   // baseMass (rw)
        'storage',                   // angW_buf (rw)
    ];
    const bosG1Entries = bosG1Types.map((type, i) => ({
        binding: i, visibility: GPUShaderStage.COMPUTE,
        buffer: { type },
    }));
    const bosG1 = device.createBindGroupLayout({ label: 'bosons_g1', entries: bosG1Entries });

    // Group 2: photon pool (9 bindings, all storage)
    const bosG2Entries = [];
    for (let i = 0; i < 9; i++) {
        bosG2Entries.push({
            binding: i, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' },
        });
    }
    const bosG2 = device.createBindGroupLayout({ label: 'bosons_g2', entries: bosG2Entries });

    // Group 3: pion pool (11 bindings, all storage)
    const bosG3Entries = [];
    for (let i = 0; i < 11; i++) {
        bosG3Entries.push({
            binding: i, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' },
        });
    }
    const bosG3 = device.createBindGroupLayout({ label: 'bosons_g3', entries: bosG3Entries });

    const bosLayouts = [bosG0, bosG1, bosG2, bosG3];
    const bosPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: bosLayouts });

    const bosonEntries = ['updatePhotons', 'updatePions', 'absorbPhotons', 'absorbPions', 'decayPions'];
    const bosonPipelines = {};
    for (const entry of bosonEntries) {
        bosonPipelines[entry] = {
            pipeline: device.createComputePipeline({
                label: entry,
                layout: bosPipelineLayout,
                compute: { module: bosonsModule, entryPoint: entry },
            }),
            bindGroupLayouts: bosLayouts,
        };
    }

    // ── Boson Tree (boson-tree.wgsl, entries: insertBosonsIntoTree, computeBosonAggregates,
    //    computeBosonGravity, applyBosonBosonGravity) ──
    // Group 0: uniforms (1 binding)
    // Group 1: boson tree nodes + counter (2 bindings)
    // Group 2: photon pool (7 bindings)
    // Group 3: pion pool (7 bindings)
    // Group 4: particle SoA (5 bindings) — for computeBosonGravity
    const bosonTreeCode = await fetchShader('boson-tree.wgsl');
    const bosonTreeModule = device.createShaderModule({ label: 'bosonTree', code: bosonTreeCode });

    const btG0 = device.createBindGroupLayout({
        label: 'bosonTree_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const btG1 = device.createBindGroupLayout({
        label: 'bosonTree_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // bosonTree
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // bosonNodeCounter
        ],
    });
    // Group 2: photon pool (7 bindings): posX, posY, velX, velY (rw), energy, flags, count
    const btG2Types = [
        'read-only-storage', 'read-only-storage',
        'storage', 'storage',  // velX, velY (rw for boson-boson gravity)
        'read-only-storage', 'read-only-storage',
        'storage',  // phCount (atomic)
    ];
    const btG2 = device.createBindGroupLayout({
        label: 'bosonTree_g2',
        entries: btG2Types.map((type, i) => ({
            binding: i, visibility: GPUShaderStage.COMPUTE,
            buffer: { type },
        })),
    });
    // Group 3: pion pool (7 bindings): posX, posY, wX(rw), wY(rw), mass, flags, count
    const btG3Types = [
        'read-only-storage', 'read-only-storage',
        'storage', 'storage',  // wX, wY (rw)
        'read-only-storage', 'read-only-storage',
        'storage',  // piCount (atomic)
    ];
    const btG3 = device.createBindGroupLayout({
        label: 'bosonTree_g3',
        entries: btG3Types.map((type, i) => ({
            binding: i, visibility: GPUShaderStage.COMPUTE,
            buffer: { type },
        })),
    });
    // Group 4: particle SoA (5 bindings) for computeBosonGravity
    const btG4 = device.createBindGroupLayout({
        label: 'bosonTree_g4',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // posX
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // posY
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // mass
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // flags
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // forces0
        ],
    });
    const btLayouts = [btG0, btG1, btG2, btG3, btG4];
    const btPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: btLayouts });

    const bosonTreeEntries = [
        'insertBosonsIntoTree', 'computeBosonAggregates',
        'computeBosonGravity', 'applyBosonBosonGravity',
    ];
    const bosonTreePipelines = {};
    for (const entry of bosonTreeEntries) {
        bosonTreePipelines[entry] = {
            pipeline: device.createComputePipeline({
                label: entry,
                layout: btPipelineLayout,
                compute: { module: bosonTreeModule, entryPoint: entry },
            }),
            bindGroupLayouts: btLayouts,
        };
    }

    return {
        recordHistory,
        compute1PN, vvKick1PN,
        lamrorRadiation, hawkingRadiation, pionEmission,
        ...bosonPipelines,
        ...bosonTreePipelines,
    };
}

/**
 * Create boson render pipelines (boson-render.wgsl).
 * Two render pipelines: photon rendering + pion rendering.
 * Returns { photonPipeline, pionPipeline, bindGroupLayouts }.
 */
export async function createBosonRenderPipelines(device, format, isLight) {
    const code = await fetchShader('boson-render.wgsl');
    const module = device.createShaderModule({ label: 'bosonRender', code });

    // Group 0: camera uniforms
    const g0 = device.createBindGroupLayout({
        label: 'bosonRender_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        ],
    });
    // Group 1: photon pool (5 read-only bindings)
    const g1 = device.createBindGroupLayout({
        label: 'bosonRender_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        ],
    });
    // Group 2: pion pool (5 read-only bindings)
    const g2 = device.createBindGroupLayout({
        label: 'bosonRender_g2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        ],
    });

    const bindGroupLayouts = [g0, g1, g2];
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts });

    const blendState = isLight
        ? {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        }
        : {
            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        };

    const photonPipeline = device.createRenderPipeline({
        label: 'photonRender',
        layout: pipelineLayout,
        vertex: { module, entryPoint: 'vertexPhoton' },
        fragment: {
            module, entryPoint: 'fragmentBoson',
            targets: [{ format, blend: blendState }],
        },
        primitive: { topology: 'triangle-strip' },
    });

    const pionPipeline = device.createRenderPipeline({
        label: 'pionRender',
        layout: pipelineLayout,
        vertex: { module, entryPoint: 'vertexPion' },
        fragment: {
            module, entryPoint: 'fragmentBoson',
            targets: [{ format, blend: blendState }],
        },
        primitive: { topology: 'triangle-strip' },
    });

    return { photonPipeline, pionPipeline, bindGroupLayouts };
}

/**
 * Create ghost generation compute pipeline.
 * Standalone shader (not prepended with common.wgsl) — defines its own SimUniforms.
 * Bind groups:
 *   Group 0: read-only particle SoA (posX, posY, velWX, velWY, angW, mass, charge, flags)
 *   Group 1: read-write ghost output SoA + read-only derived + ghost versions + particleId
 *   Group 2: ghostCounter atomic + uniforms + ghostOriginalIdx
 */
export async function createGhostGenPipeline(device) {
    const code = await fetchShader('ghost-gen.wgsl');
    const module = device.createShaderModule({ label: 'ghostGen', code });

    // Group 0: read-only particle inputs (8 bindings)
    const group0Layout = device.createBindGroupLayout({
        label: 'ghostGen_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });

    // Group 1: ghost output (read-write) + derived inputs (read-only) (16 bindings)
    const group1Layout = device.createBindGroupLayout({
        label: 'ghostGen_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });

    // Group 2: ghostCounter + uniforms + ghostOriginalIdx (3 bindings)
    const group2Layout = device.createBindGroupLayout({
        label: 'ghostGen_group2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });

    const pipeline = device.createComputePipeline({
        label: 'ghostGen',
        layout: device.createPipelineLayout({
            bindGroupLayouts: [group0Layout, group1Layout, group2Layout],
        }),
        compute: { module, entryPoint: 'main' },
    });

    return { pipeline, bindGroupLayouts: [group0Layout, group1Layout, group2Layout] };
}

/**
 * Create field deposition pipelines (Phase 5: scalar field two-pass PQS).
 * Entry points from field-deposit.wgsl (prepended with field-common.wgsl):
 *   scatterDeposit, scatterDepositAxion, scatterDepositThermal, gatherDeposit, clearGrid
 *
 * Bind groups:
 *   Group 0: particle SoA (8 read-only bindings)
 *   Group 1: scratch + target grid + uniforms (4 bindings)
 */
export async function createFieldDepositPipelines(device) {
    const fieldCommonWGSL = await fetchShader('field-common.wgsl');
    const depositWGSL = await fetchShader('field-deposit.wgsl');
    const code = fieldCommonWGSL + '\n' + depositWGSL;
    const module = device.createShaderModule({ label: 'fieldDeposit', code });

    // Group 0: particle SoA (8 read-only bindings)
    const group0Layout = device.createBindGroupLayout({
        label: 'fieldDeposit_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // posX
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // posY
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // mass
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // baseMass
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // charge
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // flags
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // velWX
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // velWY
        ],
    });

    // Group 1: scratch weights + scratch indices + target grid + uniforms
    const group1Layout = device.createBindGroupLayout({
        label: 'fieldDeposit_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // scratchWeights
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // scratchIndices
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // targetGrid
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // FieldUniforms
        ],
    });

    const bindGroupLayouts = [group0Layout, group1Layout];
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts });

    const entryPoints = ['scatterDeposit', 'scatterDepositAxion', 'scatterDepositThermal', 'gatherDeposit', 'clearGrid'];
    const pipelines = {};
    for (const entry of entryPoints) {
        pipelines[entry] = device.createComputePipeline({
            label: entry,
            layout: pipelineLayout,
            compute: { module, entryPoint: entry },
        });
    }

    return { ...pipelines, bindGroupLayouts };
}
