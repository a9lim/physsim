/**
 * @fileoverview GPU buffer allocation for SoA particle state.
 *
 * Creates and manages all GPUBuffer instances for the particle system.
 * Buffers are fixed-capacity (MAX_PARTICLES), indexed by particle slot.
 */

/** @param {GPUDevice} device */
export function createParticleBuffers(device, maxParticles) {
    const FLOAT_SIZE = 4;
    const UINT_SIZE = 4;

    /**
     * Helper: create a storage buffer with COPY_SRC for readback.
     * @param {string} label
     * @param {number} elementSize - bytes per element
     * @param {number} count
     */
    function storageBuffer(label, elementSize, count) {
        return device.createBuffer({
            label,
            size: elementSize * count,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
    }

    // Core state (f32 per particle)
    const posX = storageBuffer('posX', FLOAT_SIZE, maxParticles);
    const posY = storageBuffer('posY', FLOAT_SIZE, maxParticles);
    const velWX = storageBuffer('velWX', FLOAT_SIZE, maxParticles);
    const velWY = storageBuffer('velWY', FLOAT_SIZE, maxParticles);
    const angW = storageBuffer('angW', FLOAT_SIZE, maxParticles);
    const mass = storageBuffer('mass', FLOAT_SIZE, maxParticles);
    const baseMass = storageBuffer('baseMass', FLOAT_SIZE, maxParticles);
    const charge = storageBuffer('charge', FLOAT_SIZE, maxParticles);

    // Derived/cached
    const radius = storageBuffer('radius', FLOAT_SIZE, maxParticles);
    const gamma = storageBuffer('gamma', FLOAT_SIZE, maxParticles);

    // Particle metadata
    const flags = storageBuffer('flags', UINT_SIZE, maxParticles);
    const color = storageBuffer('color', UINT_SIZE, maxParticles);

    // Pool management: aliveCount + freeStack + freeTop
    // Packed into one buffer: [aliveCount: u32, freeTop: u32, freeStack: u32[maxParticles]]
    const poolMgmt = storageBuffer('poolMgmt', UINT_SIZE, maxParticles + 2);

    // Stats readback buffer (small, double-buffered)
    const statsBuffer = device.createBuffer({
        label: 'statsBuffer',
        size: 64, // 16 floats
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const statsStagingA = device.createBuffer({
        label: 'statsStagingA',
        size: 64,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const statsStagingB = device.createBuffer({
        label: 'statsStagingB',
        size: 64,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    return {
        maxParticles,
        // Core state
        posX, posY, velWX, velWY, angW, mass, baseMass, charge,
        // Derived
        radius, gamma,
        // Metadata
        flags, color,
        // Pool
        poolMgmt,
        // Stats
        statsBuffer, statsStagingA, statsStagingB,

        /** Destroy all buffers */
        destroy() {
            for (const key of Object.keys(this)) {
                if (this[key] && typeof this[key].destroy === 'function') {
                    this[key].destroy();
                }
            }
        }
    };
}

/**
 * Create the uniform buffer for simulation parameters.
 * Layout matches SimUniforms struct in common.wgsl.
 */
export function createUniformBuffer(device) {
    // 256 bytes is enough for all uniforms with padding
    return device.createBuffer({
        label: 'simUniforms',
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
}

/**
 * Write simulation uniforms to the GPU buffer.
 * @param {GPUDevice} device
 * @param {GPUBuffer} buffer
 * @param {Object} params - Simulation parameters
 */
export function writeUniforms(device, buffer, params) {
    // Pack into Float32Array matching WGSL struct layout
    const data = new ArrayBuffer(256);
    const f = new Float32Array(data);
    const u = new Uint32Array(data);

    f[0] = params.dt || 0;
    f[1] = params.simTime || 0;
    f[2] = params.domainW || 1;
    f[3] = params.domainH || 1;
    f[4] = params.speedScale || 1;
    f[5] = params.softening || 8;
    f[6] = params.softeningSq || 64;
    u[7] = params.toggles0 || 0;     // toggle bitfield 0
    u[8] = params.toggles1 || 0;     // toggle bitfield 1
    f[9] = params.yukawaCoupling || 14;
    f[10] = params.yukawaMu || 0.15;
    f[11] = params.higgsMass || 0.5;
    f[12] = params.axionMass || 0.05;
    u[13] = params.boundaryMode || 0;
    u[14] = params.topologyMode || 0;
    u[15] = params.collisionMode || 0;
    u[16] = params.maxParticles || 4096;
    u[17] = params.aliveCount || 0;

    device.queue.writeBuffer(buffer, 0, data);
}
