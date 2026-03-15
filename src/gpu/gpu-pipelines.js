/**
 * @fileoverview GPU compute pipeline creation for all shader phases.
 *
 * Each function creates a pipeline + bind group layout for one shader.
 * Shaders are loaded via fetch() and prepended with common.wgsl.
 *
 * Buffer packing: ParticleState (36B), ParticleAux (20B), RadiationState (32B),
 * Photon (32B), Pion (48B) packed structs reduce storage buffer count per stage to ≤10.
 */

/** Shader version — bump to invalidate browser cache after shader edits */
const SHADER_VERSION = 17;

/** Fetch a WGSL shader file relative to src/gpu/shaders/ */
async function fetchShader(filename, prepend = '') {
    const resp = await fetch(`src/gpu/shaders/${filename}?v=${SHADER_VERSION}`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    const source = await resp.text();
    return prepend ? prepend + '\n' + source : source;
}

/**
 * Create all Phase 2 compute pipelines.
 * Returns an object with pipeline + bindGroupLayout for each shader.
 */
export async function createPhase2Pipelines(device, wgslConstants = '') {
    const commonWGSL = wgslConstants + '\n' + await fetchShader('common.wgsl');

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
    // 1 uniform + 1 storage (allForces) = 1 storage buffer per stage
    const resetForces = await makePipeline('resetForces', 'reset-forces.wgsl', [
        ['uniform', 'storage'],
    ]);

    // --- cacheDerived ---
    // uniforms + particleState (rw) + derived (rw) + particleAux (rw) + axYukMod (rw) = 4 storage
    const cacheDerived = await makePipeline('cacheDerived', 'cache-derived.wgsl', [
        ['uniform', 'storage', 'storage', 'storage', 'storage'],
    ]);

    // --- pairForce (4 bind groups) ---
    // Group 0: uniforms
    // Group 1: particleState (rw) + derived (rw) + axYukMod (rw) = 3 storage
    // Group 2: allForces (rw) = 1 storage
    // Group 3: radiationState (rw) + maxAccel (rw) = 2 storage
    // Total: 6 storage buffers per stage
    const pairForce = await makePipeline('pairForce', 'pair-force.wgsl', [
        ['uniform'],
        ['storage', 'storage', 'storage'],
        ['storage'],
        ['storage', 'storage'],
    ]);

    // --- externalFields ---
    // uniforms + particleState (rw) + allForces (rw) = 2 storage
    const externalFields = await makePipeline('externalFields', 'external-fields.wgsl', [
        ['uniform', 'storage', 'storage'],
    ]);

    // --- borisHalfKick ---
    // uniforms + particleState (rw) + allForces (rw) = 2 storage
    const borisHalfKick = await makePipeline('borisHalfKick', 'boris-half-kick.wgsl', [
        ['uniform', 'storage', 'storage'],
    ]);

    // --- borisRotate ---
    // uniforms + particleState (rw) + allForces (rw) = 2 storage
    const borisRotate = await makePipeline('borisRotate', 'boris-rotate.wgsl', [
        ['uniform', 'storage', 'storage'],
    ]);

    // --- borisDrift ---
    // uniforms + particleState (rw) + derived (rw) + allForces (rw) = 3 storage
    const borisDrift = await makePipeline('borisDrift', 'boris.wgsl', [
        ['uniform', 'storage', 'storage', 'storage'],
    ]);

    // --- spinOrbit ---
    // uniforms + particleState (rw) + derived (rw) + allForces (rw) = 3 storage
    const spinOrbit = await makePipeline('spinOrbit', 'spin-orbit.wgsl', [
        ['uniform', 'storage', 'storage', 'storage'],
    ]);

    // --- applyTorques ---
    // uniforms + particleState (rw) + allForces (rw) + derived (rw) = 3 storage
    const applyTorques = await makePipeline('applyTorques', 'apply-torques.wgsl', [
        ['uniform', 'storage', 'storage', 'storage'],
    ]);

    // --- saveF1pn ---
    // Save 1PN forces before Boris kick for velocity-Verlet correction.
    // uniforms + allForces (rw) + f1pnOld (rw) + particleState (rw) = 3 storage
    const saveF1pn = await makePipeline('saveF1pn', 'save-f1pn.wgsl', [
        ['uniform', 'storage', 'storage', 'storage'],
    ]);

    // --- borisFused ---
    // Fused halfKick1 + borisRotate + halfKick2 in one pass.
    // Eliminates 2 extra dispatches + barrier overhead + redundant global memory loads.
    // uniforms + particleState (rw) + allForces (rw) = 2 storage (same as individual shaders)
    const borisFused = await makePipeline('borisFused', 'boris-fused.wgsl', [
        ['uniform', 'storage', 'storage'],
    ]);

    return {
        resetForces, cacheDerived, pairForce, externalFields,
        borisHalfKick, borisRotate, borisDrift, spinOrbit, applyTorques,
        saveF1pn, borisFused,
    };
}

/**
 * Create tree build compute pipelines (Phase 3: GPU Barnes-Hut).
 * 4 entry points from tree-build.wgsl.
 *
 * Bind groups:
 *   Group 0: tree state (nodes, nodeCounter, bounds, visitorFlags) — read-write
 *   Group 1: particleState (ro) + derived (ro) = 2 bindings
 *   Group 2: uniforms
 */
export async function createTreeBuildPipelines(device, wgslConstants = '') {
    const code = await fetchShader('tree-build.wgsl', wgslConstants);
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

    // Group 1: packed particle state + derived = 2 bindings (rw for encoder compat)
    const group1Layout = device.createBindGroupLayout({
        label: 'treeBuild_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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
 * Bind groups:
 *   Group 0: nodes (ro) + uniforms = 2
 *   Group 1: particleState (ro) + particleAux (ro) + derived (ro) + axYukMod (ro) + ghostOriginalIdx (ro) = 5
 *   Group 2: allForces (rw) + radiationState (rw) + maxAccel (rw) = 3
 *   Total: 8 storage buffers per stage
 */
export async function createTreeForcePipeline(device, wgslConstants = '') {
    const code = await fetchShader('forces-tree.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'treeForce', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'treeForce_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // nodes (rw for encoder compat)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    // Group 1: packed particle structs (rw for encoder compat — shared buffers written by other passes)
    const group1Layout = device.createBindGroupLayout({
        label: 'treeForce_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleState
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleAux
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // derived
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // axYukMod
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // ghostOriginalIdx
        ],
    });

    // Group 2: force accumulators
    const group2Layout = device.createBindGroupLayout({
        label: 'treeForce_group2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // allForces
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // radiationState
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // maxAccel
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
 * Bind groups:
 *   Group 0: nodes (ro) + uniforms = 2
 *   Group 1: particleState (rw) + particleAux (rw) + ghostOriginalIdx (ro) = 3
 *   Group 2: collisionPairs + pairCounter + mergeResults + mergeCounter = 4
 *   Total: 8 storage buffers per stage
 */
export async function createCollisionPipelines(device, wgslConstants = '') {
    const code = await fetchShader('collision.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'collision', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'collision_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // nodes (rw for encoder compat)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    // Group 1: packed particle structs (rw for resolve + encoder compat)
    const group1Layout = device.createBindGroupLayout({
        label: 'collision_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleState
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleAux
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // ghostOriginalIdx (rw for encoder compat)
        ],
    });

    // Group 2: collision pairs + counters + merge results (death metadata now in particleAux)
    const group2Layout = device.createBindGroupLayout({
        label: 'collision_group2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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

    const detectCollisionsPairwise = device.createComputePipeline({
        label: 'detectCollisionsPairwise',
        layout: pipelineLayout,
        compute: { module, entryPoint: 'detectCollisionsPairwise' },
    });

    const resolveBouncePairwise = device.createComputePipeline({
        label: 'resolveBouncePairwise',
        layout: pipelineLayout,
        compute: { module, entryPoint: 'resolveBouncePairwise' },
    });

    return { detectCollisions, resolveCollisions, detectCollisionsPairwise, resolveBouncePairwise, bindGroupLayouts };
}

/**
 * Create dead particle GC compute pipeline (Phase 3).
 * Bind group:
 *   Group 0: particleState (rw) + particleAux (ro) + uniforms + freeStack (rw) + freeTop (rw) = 4 storage
 */
export async function createDeadGCPipeline(device, wgslConstants = '') {
    const code = await fetchShader('dead-gc.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'deadGC', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'deadGC_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleState (flags)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleAux (rw for encoder compat)
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
 */
export async function createPhase4Pipelines(device, wgslConstants = '') {
    // ── recordHistory (history.wgsl, entry: recordHistory) ──
    // Group 0: uniform + particleState (ro) = 1 storage
    // Group 1: history ring buffers + meta (7 bindings)
    const historyCode = await fetchShader('history.wgsl', wgslConstants);
    const historyModule = device.createShaderModule({ label: 'history', code: historyCode });

    const historyG0 = device.createBindGroupLayout({
        label: 'history_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleState (rw for encoder compat)
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
    // Group 0: uniforms
    // Group 1: particleState (ro) + derived (ro) + axYukMod (ro) = 3
    // Group 2: allForces (rw) + f1pnOld (ro) + particleState_rw (rw) = 3
    const onePNCode = await fetchShader('onePN.wgsl', wgslConstants);
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
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleState (rw for encoder compat)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // derived (rw for encoder compat)
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // axYukMod (rw for encoder compat)
        ],
    });
    const onePNG2 = device.createBindGroupLayout({
        label: 'onePN_g2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // allForces
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // f1pnOld (rw for encoder compat)
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

    // ── Radiation (radiation.wgsl) ──
    // Group 0: uniforms
    // Group 1: particleState (rw) + particleAux (ro) + derived (rw) + allForces (ro) + radiationState (rw) + axYukMod (ro) = 6
    // Group 2: photonPool (rw) + phCount (rw) = 2
    // Group 3: pionPool (rw) + piCount (rw) = 2
    // Total: 10 storage buffers per stage
    const radiationCode = await fetchShader('radiation.wgsl', wgslConstants);
    const radiationModule = device.createShaderModule({ label: 'radiation', code: radiationCode });

    const radG0 = device.createBindGroupLayout({
        label: 'radiation_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const radG1 = device.createBindGroupLayout({
        label: 'radiation_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleState (rw)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleAux (rw for encoder compat)
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // derived (rw)
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // allForces (rw for encoder compat)
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // radiationState (rw)
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // axYukMod (rw for encoder compat)
        ],
    });
    const radG2 = device.createBindGroupLayout({
        label: 'radiation_g2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // photonPool
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // phCount
        ],
    });
    const radG3 = device.createBindGroupLayout({
        label: 'radiation_g3',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // pionPool
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // piCount
        ],
    });

    const radLayouts = [radG0, radG1, radG2, radG3];
    const radPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: radLayouts });

    const larmorRadiation = {
        pipeline: device.createComputePipeline({
            label: 'larmorRadiation', layout: radPipelineLayout,
            compute: { module: radiationModule, entryPoint: 'larmorRadiation' },
        }),
        bindGroupLayouts: radLayouts,
    };
    const hawkingRadiation = {
        pipeline: device.createComputePipeline({
            label: 'hawkingRadiation', layout: radPipelineLayout,
            compute: { module: radiationModule, entryPoint: 'hawkingRadiation' },
        }),
        bindGroupLayouts: radLayouts,
    };
    const pionEmission = {
        pipeline: device.createComputePipeline({
            label: 'pionEmission', layout: radPipelineLayout,
            compute: { module: radiationModule, entryPoint: 'pionEmission' },
        }),
        bindGroupLayouts: radLayouts,
    };

    // ── Bosons (bosons.wgsl) ──
    // Group 0: uniforms + poolMgmt = 2
    // Group 1: particleState (rw) + particleAux (ro) = 2
    // Group 2: photonPool (rw) + phCount (rw) = 2
    // Group 3: pionPool (rw) + piCount (rw) = 2
    // Total: 8 storage buffers per stage
    const bosonsCode = await fetchShader('bosons.wgsl', wgslConstants);
    const bosonsModule = device.createShaderModule({ label: 'bosons', code: bosonsCode });

    const bosG0 = device.createBindGroupLayout({
        label: 'bosons_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // poolMgmt
        ],
    });
    const bosG1 = device.createBindGroupLayout({
        label: 'bosons_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleState (rw)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleAux (rw for encoder compat)
        ],
    });
    const bosG2 = device.createBindGroupLayout({
        label: 'bosons_g2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // photonPool
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // phCount
        ],
    });
    const bosG3 = device.createBindGroupLayout({
        label: 'bosons_g3',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // pionPool
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // piCount
        ],
    });

    const bosLayouts = [bosG0, bosG1, bosG2, bosG3];
    const bosPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: bosLayouts });

    const bosonEntries = ['updatePhotons', 'updatePions', 'absorbPhotons', 'absorbPions', 'decayPions'];
    const bosonPipelines = {};
    for (const entry of bosonEntries) {
        bosonPipelines[entry] = {
            pipeline: device.createComputePipeline({
                label: entry, layout: bosPipelineLayout,
                compute: { module: bosonsModule, entryPoint: entry },
            }),
            bindGroupLayouts: bosLayouts,
        };
    }

    // ── Boson Tree (boson-tree.wgsl) ──
    // Group 0: uniforms + bosonTreeNodes + bosonTreeCounter = 3 (1 uniform + 2 storage)
    // Group 1: photonPool (rw) + phCount (rw) = 2
    // Group 2: pionPool (rw) + piCount (rw) = 2
    // Group 3: particleState (ro) + allForces (rw) = 2
    // Total: 4 groups, 8 storage buffers per stage
    const bosonTreeCode = await fetchShader('boson-tree.wgsl', wgslConstants);
    const bosonTreeModule = device.createShaderModule({ label: 'bosonTree', code: bosonTreeCode });

    const btG0 = device.createBindGroupLayout({
        label: 'bosonTree_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // bosonTreeNodes
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // bosonTreeCounter
        ],
    });
    const btG1 = device.createBindGroupLayout({
        label: 'bosonTree_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // photonPool
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // phCount
        ],
    });
    const btG2 = device.createBindGroupLayout({
        label: 'bosonTree_g2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // pionPool
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // piCount
        ],
    });
    const btG3 = device.createBindGroupLayout({
        label: 'bosonTree_g3',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleState (rw for encoder compat)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // allForces
        ],
    });
    const btLayouts = [btG0, btG1, btG2, btG3];
    const btPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: btLayouts });

    const bosonTreeEntries = [
        'insertBosonsIntoTree', 'computeBosonAggregates',
        'computeBosonGravity', 'applyBosonBosonGravity',
    ];
    const bosonTreePipelines = {};
    for (const entry of bosonTreeEntries) {
        bosonTreePipelines[entry] = {
            pipeline: device.createComputePipeline({
                label: entry, layout: btPipelineLayout,
                compute: { module: bosonTreeModule, entryPoint: entry },
            }),
            bindGroupLayouts: btLayouts,
        };
    }

    return {
        recordHistory,
        compute1PN, vvKick1PN,
        larmorRadiation, hawkingRadiation, pionEmission,
        ...bosonPipelines,
        ...bosonTreePipelines,
    };
}

/**
 * Create boson render pipelines (boson-render.wgsl).
 * Two render pipelines: photon rendering + pion rendering.
 * Group 0: camera uniforms
 * Group 1: photonPool (ro) + phCount (ro) = 2 (was 5)
 * Group 2: pionPool (ro) + piCount (ro) = 2 (was 5)
 */
export async function createBosonRenderPipelines(device, format, isLight, wgslConstants = '') {
    const code = await fetchShader('boson-render.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'bosonRender', code });

    const g0 = device.createBindGroupLayout({
        label: 'bosonRender_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        ],
    });
    const g1 = device.createBindGroupLayout({
        label: 'bosonRender_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // photonPool
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // phCount
        ],
    });
    const g2 = device.createBindGroupLayout({
        label: 'bosonRender_g2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // pionPool
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // piCount
        ],
    });

    const bindGroupLayouts = [g0, g1, g2];
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts });

    // Boson fragment shader outputs premultiplied alpha: use 'one' for srcFactor
    const blendState = isLight
        ? {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        }
        : {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
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
 * Bind groups:
 *   Group 0: particleState (ro) = 1
 *   Group 1: ghostState (rw) + ghostAux (rw) + derived (ro) + ghostDerived (rw) + particleAux (ro) = 5
 *   Group 2: ghostCounter (rw) + uniforms + ghostOriginalIdx (rw) = 3
 *   Total: 8 storage buffers per stage
 */
export async function createGhostGenPipeline(device, wgslConstants = '') {
    const code = await fetchShader('ghost-gen.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'ghostGen', code });

    // Group 0: packed particle state (rw for encoder compat)
    const group0Layout = device.createBindGroupLayout({
        label: 'ghostGen_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleState (rw for encoder compat)
        ],
    });

    // Group 1: ghost output + derived + aux (rw for encoder compat on shared buffers)
    const group1Layout = device.createBindGroupLayout({
        label: 'ghostGen_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // ghostState
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // ghostAux
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // derived_in (rw for encoder compat)
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // ghostDerived
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleAux_in (rw for encoder compat)
        ],
    });

    // Group 2: ghostCounter + uniforms + ghostOriginalIdx
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
 * Bind groups:
 *   Group 0: particleState (ro) = 1 (was 8 separate SoA buffers)
 *   Group 1: scratch + scratchIndices + targetGrid + fieldUniforms = 4
 *   Total: 4 storage buffers per stage
 */
export async function createFieldDepositPipelines(device, wgslConstants = '') {
    const fieldCommonWGSL = wgslConstants + '\n' + await fetchShader('field-common.wgsl');
    const depositWGSL = await fetchShader('field-deposit.wgsl');
    const code = fieldCommonWGSL + '\n' + depositWGSL;
    const module = device.createShaderModule({ label: 'fieldDeposit', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'fieldDeposit_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleState (rw for encoder compat)
        ],
    });

    const group1Layout = device.createBindGroupLayout({
        label: 'fieldDeposit_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    const bindGroupLayouts = [group0Layout, group1Layout];
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts });

    const entryPoints = ['scatterDeposit', 'scatterDepositAxion', 'scatterDepositThermal', 'gatherDeposit', 'clearGrid'];
    const pipelines = {};
    for (const entry of entryPoints) {
        pipelines[entry] = device.createComputePipeline({
            label: entry, layout: pipelineLayout,
            compute: { module, entryPoint: entry },
        });
    }

    return { ...pipelines, bindGroupLayouts };
}

/**
 * Create field evolution pipelines (Phase 5: Störmer-Verlet KDK).
 * Same as before — no particle buffers involved, just field grids.
 */
export async function createFieldEvolvePipelines(device, wgslConstants = '') {
    const fieldCommonWGSL = wgslConstants + '\n' + await fetchShader('field-common.wgsl');
    const evolveWGSL = await fetchShader('field-evolve.wgsl');
    const code = fieldCommonWGSL + '\n' + evolveWGSL;
    const module = device.createShaderModule({ label: 'fieldEvolve', code });

    const evolveGroup0Layout = device.createBindGroupLayout({
        label: 'fieldEvolve_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // source (rw for encoder compat)
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // thermal (rw for encoder compat)
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // sgPhiFull (rw for encoder compat)
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // sgGradX (rw for encoder compat)
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // sgGradY (rw for encoder compat)
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // fieldGradX (rw for computeGridGradients)
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // fieldGradY (rw for computeGridGradients)
            { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const evolveBindGroupLayouts = [evolveGroup0Layout];
    const evolvePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: evolveBindGroupLayouts });

    const gradGroup0Layout = device.createBindGroupLayout({
        label: 'fieldGrad_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // source (rw for encoder compat)
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // thermal (rw for encoder compat)
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // sgPhiFull (rw for encoder compat)
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // sgGradX (rw for encoder compat)
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // sgGradY (rw for encoder compat)
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const gradBindGroupLayouts = [gradGroup0Layout];
    const gradPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: gradBindGroupLayouts });

    const evolveEntries = ['computeLaplacian', 'higgsHalfKick', 'axionHalfKick', 'fieldDrift', 'nanFixupHiggs', 'nanFixupAxion'];
    const pipelines = {};
    for (const entry of evolveEntries) {
        pipelines[entry] = device.createComputePipeline({
            label: entry, layout: evolvePipelineLayout,
            compute: { module, entryPoint: entry },
        });
    }

    pipelines.computeGridGradients = device.createComputePipeline({
        label: 'computeGridGradients', layout: gradPipelineLayout,
        compute: { module, entryPoint: 'computeGridGradients' },
    });

    return { ...pipelines, evolveBindGroupLayouts, gradBindGroupLayouts };
}

/**
 * Create field force application pipelines (Phase 5: field → particle forces).
 * Bind groups:
 *   Group 0: particleState (rw) + particleAux (ro) + derived (rw) = 3
 *   Group 1: field arrays (6 ro)
 *   Group 2: allForces (rw) + axYukMod (rw) = 2
 *   Group 3: fieldUniforms = uniform
 *   Total: 9 storage buffers per stage
 */
export async function createFieldForcesPipelines(device, wgslConstants = '') {
    const fieldCommonWGSL = wgslConstants + '\n' + await fetchShader('field-common.wgsl');
    const forcesWGSL = await fetchShader('field-forces.wgsl');
    const code = fieldCommonWGSL + '\n' + forcesWGSL;
    const module = device.createShaderModule({ label: 'fieldForces', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'fieldForces_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleState (rw)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // derived (rw)
        ],
    });

    const group1Layout = device.createBindGroupLayout({
        label: 'fieldForces_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // higgsField (rw for encoder compat)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // higgsGradX (rw for encoder compat)
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // higgsGradY (rw for encoder compat)
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // axionField (rw for encoder compat)
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // axionGradX (rw for encoder compat)
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // axionGradY (rw for encoder compat)
        ],
    });

    const group2Layout = device.createBindGroupLayout({
        label: 'fieldForces_group2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });

    const group3Layout = device.createBindGroupLayout({
        label: 'fieldForces_group3',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    const bindGroupLayouts = [group0Layout, group1Layout, group2Layout, group3Layout];
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts });

    const applyHiggsForces = device.createComputePipeline({
        label: 'applyHiggsForces', layout: pipelineLayout,
        compute: { module, entryPoint: 'applyHiggsForces' },
    });
    const applyAxionForces = device.createComputePipeline({
        label: 'applyAxionForces', layout: pipelineLayout,
        compute: { module, entryPoint: 'applyAxionForces' },
    });

    return { applyHiggsForces, applyAxionForces, bindGroupLayouts };
}

/**
 * Create field self-gravity pipelines (Phase 5).
 * No particle buffers — just field grids. Unchanged.
 */
export async function createFieldSelfGravPipelines(device, wgslConstants = '') {
    const fieldCommonWGSL = wgslConstants + '\n' + await fetchShader('field-common.wgsl');
    const sgWGSL = await fetchShader('field-selfgrav.wgsl');
    const code = fieldCommonWGSL + '\n' + sgWGSL;
    const module = device.createShaderModule({ label: 'fieldSelfGrav', code });

    // Group 0: core field arrays + uniform (6 storage + 1 uniform)
    const group0Layout = device.createBindGroupLayout({
        label: 'fieldSelfGrav_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // field
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // fieldDot
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // gradX
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // gradY
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // energyDensity
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // coarseRho
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },   // FieldUniforms
        ],
    });
    // Group 1: self-gravity arrays (4 storage — sgInvR removed, computed inline)
    const group1Layout = device.createBindGroupLayout({
        label: 'fieldSelfGrav_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // coarsePhi
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // sgPhiFull
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // sgGradX
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },   // sgGradY
        ],
    });

    const bindGroupLayouts = [group0Layout, group1Layout];
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts });

    const entryPoints = [
        'computeEnergyDensityHiggs', 'computeEnergyDensityAxion',
        'downsampleRho', 'computeCoarsePotential', 'upsamplePhi', 'computeSelfGravGradients',
    ];
    const pipelines = {};
    for (const entry of entryPoints) {
        pipelines[entry] = device.createComputePipeline({
            label: entry, layout: pipelineLayout,
            compute: { module, entryPoint: entry },
        });
    }

    return { ...pipelines, bindGroupLayouts };
}

/**
 * Create field excitation pipeline (Phase 5). Unchanged — no particle buffers.
 */
export async function createFieldExcitationPipeline(device, wgslConstants = '') {
    const fieldCommonWGSL = wgslConstants + '\n' + await fetchShader('field-common.wgsl');
    const excWGSL = await fetchShader('field-excitation.wgsl');
    const code = fieldCommonWGSL + '\n' + excWGSL;
    const module = device.createShaderModule({ label: 'fieldExcitation', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'fieldExcitation_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });

    const bindGroupLayouts = [group0Layout];
    const pipeline = device.createComputePipeline({
        label: 'depositExcitations',
        layout: device.createPipelineLayout({ bindGroupLayouts }),
        compute: { module, entryPoint: 'depositExcitations' },
    });

    return { pipeline, bindGroupLayouts };
}

/**
 * Create heatmap compute pipelines (Phase 5).
 * Bind groups:
 *   heatmapLayout:
 *     Group 0: particleState (ro) = 1 (was 5 separate buffers)
 *     Group 1: potential grids (3 rw) + HeatmapUniforms = 4
 *     Group 2: signal delay history (histPosX, histPosY, histTime, histMeta) = 4
 *   blurLayout: unchanged
 */
export async function createHeatmapPipelines(device, wgslConstants = '') {
    const code = await fetchShader('heatmap.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'heatmap', code });

    const hmG0 = device.createBindGroupLayout({
        label: 'heatmap_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleState (rw for encoder compat)
        ],
    });
    const hmG1 = device.createBindGroupLayout({
        label: 'heatmap_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    // Group 2: signal delay history buffers (histPosX, histPosY, histTime, histMeta)
    const hmG2 = device.createBindGroupLayout({
        label: 'heatmap_g2_history',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // histPosX
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // histPosY
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // histTime
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // histMeta
        ],
    });
    const heatmapLayouts = [hmG0, hmG1, hmG2];

    const computeHeatmap = device.createComputePipeline({
        label: 'computeHeatmap',
        layout: device.createPipelineLayout({ bindGroupLayouts: heatmapLayouts }),
        compute: { module, entryPoint: 'computeHeatmap' },
    });

    const blurG0 = device.createBindGroupLayout({
        label: 'heatmapBlur_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });
    const blurLayouts = [blurG0];
    const blurPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: blurLayouts });

    const blurHorizontal = device.createComputePipeline({
        label: 'blurHorizontal', layout: blurPipelineLayout,
        compute: { module, entryPoint: 'blurHorizontal' },
    });
    const blurVertical = device.createComputePipeline({
        label: 'blurVertical', layout: blurPipelineLayout,
        compute: { module, entryPoint: 'blurVertical' },
    });

    return { computeHeatmap, blurHorizontal, blurVertical, heatmapLayouts, blurLayouts };
}

/**
 * Create expansion compute pipeline (Phase 5).
 * Bind group: particleState (rw) + ExpansionUniforms = 1 storage + 1 uniform
 */
export async function createExpansionPipeline(device, wgslConstants = '') {
    const code = await fetchShader('expansion.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'expansion', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'expansion_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleState
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // ExpansionUniforms
        ],
    });

    const bindGroupLayouts = [group0Layout];
    const pipeline = device.createComputePipeline({
        label: 'applyExpansion',
        layout: device.createPipelineLayout({ bindGroupLayouts }),
        compute: { module, entryPoint: 'applyExpansion' },
    });

    return { pipeline, bindGroupLayouts };
}

/**
 * Create disintegration compute pipeline (Phase 5).
 * Bind groups:
 *   Group 0: particleState (ro) + particleAux (ro) + derived (ro) = 3
 *   Group 1: events + eventCounter + DisintUniforms = 3
 */
export async function createDisintegrationPipeline(device, wgslConstants = '') {
    const code = await fetchShader('disintegration.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'disintegration', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'disint_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleState (rw for encoder compat)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleAux (rw for encoder compat)
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // derived (rw for encoder compat)
        ],
    });

    const group1Layout = device.createBindGroupLayout({
        label: 'disint_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    const bindGroupLayouts = [group0Layout, group1Layout];
    const pipeline = device.createComputePipeline({
        label: 'checkDisintegration',
        layout: device.createPipelineLayout({ bindGroupLayouts }),
        compute: { module, entryPoint: 'checkDisintegration' },
    });

    return { pipeline, bindGroupLayouts };
}

/**
 * Create pair production compute pipeline (Phase 5).
 * Bind groups:
 *   Group 0: photonPool (ro) + phCount (ro) = 2 (was 7 separate)
 *   Group 1: particleState (ro) = 1 (was 4 separate)
 *   Group 2: pairEvents + pairCounter + PairProdUniforms = 3
 */
export async function createPairProductionPipeline(device, wgslConstants = '') {
    const code = await fetchShader('pair-production.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'pairProduction', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'pairProd_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // photonPool (rw for encoder compat)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // phCount (rw for encoder compat)
        ],
    });

    const group1Layout = device.createBindGroupLayout({
        label: 'pairProd_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleState (rw for encoder compat)
        ],
    });

    const group2Layout = device.createBindGroupLayout({
        label: 'pairProd_g2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    const bindGroupLayouts = [group0Layout, group1Layout, group2Layout];
    const pipeline = device.createComputePipeline({
        label: 'checkPairProduction',
        layout: device.createPipelineLayout({ bindGroupLayouts }),
        compute: { module, entryPoint: 'checkPairProduction' },
    });

    return { pipeline, bindGroupLayouts };
}

/**
 * Create update-colors compute pipeline.
 * Standalone shader (defines own structs).
 * Bindings: ColorUniforms (uniform), particleState (ro), color (rw).
 */
export async function createUpdateColorsPipeline(device, wgslConstants = '') {
    const code = await fetchShader('update-colors.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'updateColors', code });

    const bindGroupLayout = device.createBindGroupLayout({
        label: 'updateColors_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });

    const pipeline = device.createComputePipeline({
        label: 'updateColors',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module, entryPoint: 'main' },
    });

    return { pipeline, bindGroupLayout };
}

/**
 * Create spin ring render pipelines (arc line-strip + arrowhead triangle-list).
 * Standalone shader (defines own structs).
 * Bindings: camera (uniform), particleState (ro), particleAux (ro), derived (ro).
 * Returns { arcPipeline, arrowPipeline, bindGroupLayout }.
 */
export async function createSpinRenderPipeline(device, format, isLight, wgslConstants = '') {
    const code = await fetchShader('spin-render.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'spinRender', code });

    const bindGroupLayout = device.createBindGroupLayout({
        label: 'spinRender_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        ],
    });

    // Premultiplied alpha: shader outputs rgb*a, use srcFactor='one' (not 'src-alpha')
    const blendState = isLight
        ? {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        }
        : {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        };

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    // Arc pipeline (triangle-strip ribbon, ARC_SEGMENTS * 2 vertices per instance)
    const arcPipeline = device.createRenderPipeline({
        label: 'spinRender_arc',
        layout: pipelineLayout,
        vertex: { module, entryPoint: 'vs_main' },
        fragment: {
            module, entryPoint: 'fs_main',
            targets: [{ format, blend: blendState }],
        },
        primitive: { topology: 'triangle-strip' },
    });

    // Arrowhead pipeline (triangle-list, 3 vertices per instance)
    const arrowPipeline = device.createRenderPipeline({
        label: 'spinRender_arrow',
        layout: pipelineLayout,
        vertex: { module, entryPoint: 'vs_arrow' },
        fragment: {
            module, entryPoint: 'fs_arrow',
            targets: [{ format, blend: blendState }],
        },
        primitive: { topology: 'triangle-list' },
    });

    return { arcPipeline, arrowPipeline, bindGroupLayout };
}

/**
 * Create trail recording compute pipeline.
 * Standalone shader (defines own structs).
 * Bindings: particleState (ro), trailX/Y (rw), trailWriteIdx/Count (rw).
 */
export async function createTrailRecordPipeline(device, wgslConstants = '') {
    const code = await fetchShader('trails.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'trailRecord', code });

    const bindGroupLayout = device.createBindGroupLayout({
        label: 'trailRecord_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });

    const pipeline = device.createComputePipeline({
        label: 'trailRecord',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module, entryPoint: 'main' },
    });

    return { pipeline, bindGroupLayout };
}

/**
 * Create trail render pipeline (triangle-strip ribbon per particle instance).
 * Standalone shader. Width scales with particle radius (0.5 * radius), matching CPU.
 * Bindings: camera (uniform), trailParams (uniform), trailX/Y (ro), trailWriteIdx/Count (ro), color (ro), particles (ro), particleAux (ro).
 */
export async function createTrailRenderPipeline(device, format, isLight, wgslConstants = '') {
    const code = await fetchShader('trail-render.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'trailRender', code });

    const bindGroupLayout = device.createBindGroupLayout({
        label: 'trailRender_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
            { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 6, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
            { binding: 7, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 8, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // particleAux (for radius)
        ],
    });

    // Premultiplied alpha: shader outputs rgb*a, use srcFactor='one' (not 'src-alpha')
    const blendState = isLight
        ? {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        }
        : {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        };

    const pipeline = device.createRenderPipeline({
        label: 'trailRender',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        vertex: { module, entryPoint: 'vs_main' },
        fragment: {
            module, entryPoint: 'fs_main',
            targets: [{ format, blend: blendState }],
        },
        primitive: { topology: 'triangle-strip' },
    });

    return { pipeline, bindGroupLayout };
}

/**
 * Create arrow render pipeline for force/velocity vector visualization.
 * Standalone shader (no common.wgsl prepend).
 * Bindings: camera (uniform), arrowParams (uniform), particleState/Aux/allForces (read-only-storage).
 */
export async function createArrowRenderPipeline(device, format, isLight, wgslConstants = '') {
    const code = await fetchShader('arrow-render.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'arrowRender', code });

    const bindGroupLayout = device.createBindGroupLayout({
        label: 'arrowRender_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },        // camera
            { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },        // arrowParams
            { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // particleState
            { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // particleAux
            { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // allForces
            { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // derived (for velocity vectors)
        ],
    });

    // Premultiplied alpha: shader outputs rgb*a, use srcFactor='one' (not 'src-alpha')
    const blendState = isLight
        ? {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        }
        : {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        };

    const pipeline = device.createRenderPipeline({
        label: 'arrowRender',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        vertex: { module, entryPoint: 'vs_main' },
        fragment: {
            module, entryPoint: 'fs_main',
            targets: [{ format, blend: blendState }],
        },
        primitive: { topology: 'triangle-list' },
    });

    return { pipeline, bindGroupLayout };
}

/**
 * Create field overlay render pipeline (Phase 5). Unchanged — no particle buffers.
 */
export async function createFieldRenderPipeline(device, format, isLight, wgslConstants = '') {
    const fieldCommonWGSL = wgslConstants + '\n' + await fetchShader('field-common.wgsl');
    const renderWGSL = await fetchShader('field-render.wgsl');
    const code = fieldCommonWGSL + '\n' + renderWGSL;
    const module = device.createShaderModule({ label: 'fieldRender', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'fieldRender_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ],
    });

    const bindGroupLayouts = [group0Layout];
    const blendState = {
        color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };

    const pipeline = device.createRenderPipeline({
        label: 'fieldRender',
        layout: device.createPipelineLayout({ bindGroupLayouts }),
        vertex: { module, entryPoint: 'vsFullscreen' },
        fragment: {
            module, entryPoint: 'fsFieldOverlay',
            targets: [{ format, blend: blendState }],
        },
        primitive: { topology: 'triangle-list' },
    });

    return { pipeline, bindGroupLayouts };
}

/**
 * Create heatmap overlay render pipeline (Phase 5). Unchanged — no particle buffers.
 */
export async function createHeatmapRenderPipeline(device, format, isLight, wgslConstants = '') {
    const code = await fetchShader('heatmap-render.wgsl', wgslConstants);
    const module = device.createShaderModule({ label: 'heatmapRender', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'heatmapRender_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ],
    });

    const bindGroupLayouts = [group0Layout];
    const blendState = {
        color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };

    const pipeline = device.createRenderPipeline({
        label: 'heatmapRender',
        layout: device.createPipelineLayout({ bindGroupLayouts }),
        vertex: { module, entryPoint: 'vsFullscreen' },
        fragment: {
            module, entryPoint: 'fsHeatmapOverlay',
            targets: [{ format, blend: blendState }],
        },
        primitive: { topology: 'triangle-list' },
    });

    return { pipeline, bindGroupLayouts };
}
