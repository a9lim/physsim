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
const SHADER_VERSION = 2;

/** Fetch a WGSL shader file relative to src/gpu/shaders/ */
async function fetchShader(filename) {
    const resp = await fetch(`src/gpu/shaders/${filename}?v=${SHADER_VERSION}`);
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
    // 1 uniform + 1 storage (allForces) = 1 storage buffer per stage
    const resetForces = await makePipeline('resetForces', 'reset-forces.wgsl', [
        ['uniform', 'storage'],
    ]);

    // --- cacheDerived ---
    // uniforms + particleState (ro) + derived (rw) + particleAux (rw) = 3 storage
    const cacheDerived = await makePipeline('cacheDerived', 'cache-derived.wgsl', [
        ['uniform', 'read-only-storage', 'storage', 'storage'],
    ]);

    // --- pairForce (4 bind groups) ---
    // Group 0: uniforms
    // Group 1: particleState (ro) + derived (ro) + axYukMod (ro) = 3 storage
    // Group 2: allForces (rw) = 1 storage
    // Group 3: radiationState (rw) + maxAccel (rw) = 2 storage
    // Total: 6 storage buffers per stage
    const pairForce = await makePipeline('pairForce', 'pair-force.wgsl', [
        ['uniform'],
        ['read-only-storage', 'read-only-storage', 'read-only-storage'],
        ['storage'],
        ['storage', 'storage'],
    ]);

    // --- externalFields ---
    // uniforms + particleState (ro) + allForces (rw) = 2 storage
    const externalFields = await makePipeline('externalFields', 'external-fields.wgsl', [
        ['uniform', 'read-only-storage', 'storage'],
    ]);

    // --- borisHalfKick ---
    // uniforms + particleState (rw) + allForces (ro) = 2 storage
    const borisHalfKick = await makePipeline('borisHalfKick', 'boris-half-kick.wgsl', [
        ['uniform', 'storage', 'read-only-storage'],
    ]);

    // --- borisRotate ---
    // uniforms + particleState (rw) + allForces (ro) = 2 storage
    const borisRotate = await makePipeline('borisRotate', 'boris-rotate.wgsl', [
        ['uniform', 'storage', 'read-only-storage'],
    ]);

    // --- borisDrift ---
    // uniforms + particleState (rw) + derived (rw) = 2 storage
    const borisDrift = await makePipeline('borisDrift', 'boris.wgsl', [
        ['uniform', 'storage', 'storage'],
    ]);

    // --- spinOrbit ---
    // uniforms + particleState (rw) + derived (rw) + allForces (rw) = 3 storage
    const spinOrbit = await makePipeline('spinOrbit', 'spin-orbit.wgsl', [
        ['uniform', 'storage', 'storage', 'storage'],
    ]);

    // --- applyTorques ---
    // uniforms + particleState (ro) + allForces (ro) + derived (rw) = 3 storage
    const applyTorques = await makePipeline('applyTorques', 'apply-torques.wgsl', [
        ['uniform', 'read-only-storage', 'read-only-storage', 'storage'],
    ]);

    return {
        resetForces, cacheDerived, pairForce, externalFields,
        borisHalfKick, borisRotate, borisDrift, spinOrbit, applyTorques,
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

    // Group 1: packed particle state + derived = 2 bindings
    const group1Layout = device.createBindGroupLayout({
        label: 'treeBuild_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
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
export async function createTreeForcePipeline(device) {
    const code = await fetchShader('forces-tree.wgsl');
    const module = device.createShaderModule({ label: 'treeForce', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'treeForce_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    // Group 1: packed particle structs
    const group1Layout = device.createBindGroupLayout({
        label: 'treeForce_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleState
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleAux
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // derived
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // axYukMod
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // ghostOriginalIdx
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
export async function createCollisionPipelines(device) {
    const code = await fetchShader('collision.wgsl');
    const module = device.createShaderModule({ label: 'collision', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'collision_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    // Group 1: packed particle structs (rw for resolve)
    const group1Layout = device.createBindGroupLayout({
        label: 'collision_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleState
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleAux
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // ghostOriginalIdx
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

    return { detectCollisions, resolveCollisions, bindGroupLayouts };
}

/**
 * Create dead particle GC compute pipeline (Phase 3).
 * Bind group:
 *   Group 0: particleState (rw) + particleAux (ro) + uniforms + freeStack (rw) + freeTop (rw) = 4 storage
 */
export async function createDeadGCPipeline(device) {
    const code = await fetchShader('dead-gc.wgsl');
    const module = device.createShaderModule({ label: 'deadGC', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'deadGC_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleState (flags)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleAux (deathTime)
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
export async function createPhase4Pipelines(device) {
    // ── recordHistory (history.wgsl, entry: recordHistory) ──
    // Group 0: uniform + particleState (ro) = 1 storage
    // Group 1: history ring buffers + meta (7 bindings)
    const historyCode = await fetchShader('history.wgsl');
    const historyModule = device.createShaderModule({ label: 'history', code: historyCode });

    const historyG0 = device.createBindGroupLayout({
        label: 'history_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleState
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
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleState
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // derived
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // axYukMod
        ],
    });
    const onePNG2 = device.createBindGroupLayout({
        label: 'onePN_g2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // allForces
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // f1pnOld
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleState (rw for VV kick)
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
    const radiationCode = await fetchShader('radiation.wgsl');
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
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleAux
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // derived (rw)
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // allForces
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // radiationState (rw)
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // axYukMod
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

    const lamrorRadiation = {
        pipeline: device.createComputePipeline({
            label: 'lamrorRadiation', layout: radPipelineLayout,
            compute: { module: radiationModule, entryPoint: 'lamrorRadiation' },
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
    const bosonsCode = await fetchShader('bosons.wgsl');
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
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleAux (ro)
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
    // Group 0: uniforms
    // Group 1: bosonTreeNodes + bosonTreeCounter = 2
    // Group 2: photonPool (rw) + phCount (rw) = 2
    // Group 3: pionPool (rw) + piCount (rw) = 2
    // Group 4: particleState (ro) + allForces (rw) = 2
    // Total: 8 storage buffers per stage
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
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });
    const btG2 = device.createBindGroupLayout({
        label: 'bosonTree_g2',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // photonPool
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // phCount
        ],
    });
    const btG3 = device.createBindGroupLayout({
        label: 'bosonTree_g3',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // pionPool
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // piCount
        ],
    });
    const btG4 = device.createBindGroupLayout({
        label: 'bosonTree_g4',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleState
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // allForces
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
                label: entry, layout: btPipelineLayout,
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
 * Group 0: camera uniforms
 * Group 1: photonPool (ro) + phCount (ro) = 2 (was 5)
 * Group 2: pionPool (ro) + piCount (ro) = 2 (was 5)
 */
export async function createBosonRenderPipelines(device, format, isLight) {
    const code = await fetchShader('boson-render.wgsl');
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
 * Bind groups:
 *   Group 0: particleState (ro) = 1
 *   Group 1: ghostState (rw) + ghostAux (rw) + derived (ro) + ghostDerived (rw) + particleAux (ro) = 5
 *   Group 2: ghostCounter (rw) + uniforms + ghostOriginalIdx (rw) = 3
 *   Total: 8 storage buffers per stage
 */
export async function createGhostGenPipeline(device) {
    const code = await fetchShader('ghost-gen.wgsl');
    const module = device.createShaderModule({ label: 'ghostGen', code });

    // Group 0: packed particle state (read-only)
    const group0Layout = device.createBindGroupLayout({
        label: 'ghostGen_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleState
        ],
    });

    // Group 1: ghost output + derived + aux
    const group1Layout = device.createBindGroupLayout({
        label: 'ghostGen_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // ghostState
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // ghostAux
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // derived_in
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // ghostDerived
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleAux_in
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
export async function createFieldDepositPipelines(device) {
    const fieldCommonWGSL = await fetchShader('field-common.wgsl');
    const depositWGSL = await fetchShader('field-deposit.wgsl');
    const code = fieldCommonWGSL + '\n' + depositWGSL;
    const module = device.createShaderModule({ label: 'fieldDeposit', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'fieldDeposit_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleState
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
export async function createFieldEvolvePipelines(device) {
    const fieldCommonWGSL = await fetchShader('field-common.wgsl');
    const evolveWGSL = await fetchShader('field-evolve.wgsl');
    const code = fieldCommonWGSL + '\n' + evolveWGSL;
    const module = device.createShaderModule({ label: 'fieldEvolve', code });

    const evolveGroup0Layout = device.createBindGroupLayout({
        label: 'fieldEvolve_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
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
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
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
export async function createFieldForcesPipelines(device) {
    const fieldCommonWGSL = await fetchShader('field-common.wgsl');
    const forcesWGSL = await fetchShader('field-forces.wgsl');
    const code = fieldCommonWGSL + '\n' + forcesWGSL;
    const module = device.createShaderModule({ label: 'fieldForces', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'fieldForces_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // particleState (rw)
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleAux (ro)
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // derived (rw)
        ],
    });

    const group1Layout = device.createBindGroupLayout({
        label: 'fieldForces_group1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
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
export async function createFieldSelfGravPipelines(device) {
    const fieldCommonWGSL = await fetchShader('field-common.wgsl');
    const sgWGSL = await fetchShader('field-selfgrav.wgsl');
    const code = fieldCommonWGSL + '\n' + sgWGSL;
    const module = device.createShaderModule({ label: 'fieldSelfGrav', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'fieldSelfGrav_group0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });

    const bindGroupLayouts = [group0Layout];
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
export async function createFieldExcitationPipeline(device) {
    const fieldCommonWGSL = await fetchShader('field-common.wgsl');
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
 *   blurLayout: unchanged
 */
export async function createHeatmapPipelines(device) {
    const code = await fetchShader('heatmap.wgsl');
    const module = device.createShaderModule({ label: 'heatmap', code });

    const hmG0 = device.createBindGroupLayout({
        label: 'heatmap_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleState
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
    const heatmapLayouts = [hmG0, hmG1];

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
export async function createExpansionPipeline(device) {
    const code = await fetchShader('expansion.wgsl');
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
export async function createDisintegrationPipeline(device) {
    const code = await fetchShader('disintegration.wgsl');
    const module = device.createShaderModule({ label: 'disintegration', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'disint_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleState
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleAux
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // derived
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
export async function createPairProductionPipeline(device) {
    const code = await fetchShader('pair-production.wgsl');
    const module = device.createShaderModule({ label: 'pairProduction', code });

    const group0Layout = device.createBindGroupLayout({
        label: 'pairProd_g0',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // photonPool
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // phCount
        ],
    });

    const group1Layout = device.createBindGroupLayout({
        label: 'pairProd_g1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleState
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
 * Create field overlay render pipeline (Phase 5). Unchanged — no particle buffers.
 */
export async function createFieldRenderPipeline(device, format, isLight) {
    const fieldCommonWGSL = await fetchShader('field-common.wgsl');
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
export async function createHeatmapRenderPipeline(device, format, isLight) {
    const code = await fetchShader('heatmap-render.wgsl');
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
