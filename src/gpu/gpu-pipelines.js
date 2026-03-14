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

    // Group 1: particle SoA inputs (15 read-only bindings)
    const group1Entries = [];
    for (let i = 0; i < 15; i++) {
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
