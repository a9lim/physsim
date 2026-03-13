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
