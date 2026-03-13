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

    // Derived/cached (Phase 2)
    const magMoment = storageBuffer('magMoment', FLOAT_SIZE, maxParticles);
    const angMomentum = storageBuffer('angMomentum', FLOAT_SIZE, maxParticles);
    const axMod = storageBuffer('axMod', FLOAT_SIZE, maxParticles);
    const yukMod = storageBuffer('yukMod', FLOAT_SIZE, maxParticles);
    const velX = storageBuffer('velX', FLOAT_SIZE, maxParticles);  // coordinate velocity
    const velY = storageBuffer('velY', FLOAT_SIZE, maxParticles);
    const angVel = storageBuffer('angVel', FLOAT_SIZE, maxParticles);
    const invMass = storageBuffer('invMass', FLOAT_SIZE, maxParticles);
    const radiusSq = storageBuffer('radiusSq', FLOAT_SIZE, maxParticles);

    // Force accumulators (vec4 = 16 bytes each, packed pairs)
    const VEC4_SIZE = 16;
    const forces0 = storageBuffer('forces0', VEC4_SIZE, maxParticles); // gravity.xy, coulomb.xy
    const forces1 = storageBuffer('forces1', VEC4_SIZE, maxParticles); // magnetic.xy, gravitomag.xy
    const forces2 = storageBuffer('forces2', VEC4_SIZE, maxParticles); // f1pn.xy, spinCurv.xy
    const forces3 = storageBuffer('forces3', VEC4_SIZE, maxParticles); // radiation.xy, yukawa.xy
    const forces4 = storageBuffer('forces4', VEC4_SIZE, maxParticles); // external.xy, higgs.xy
    const forces5 = storageBuffer('forces5', VEC4_SIZE, maxParticles); // axion.xy, pad, pad
    const torques = storageBuffer('torques', VEC4_SIZE, maxParticles); // spinOrbit, frameDrag, tidal, contact
    const bFields = storageBuffer('bFields', VEC4_SIZE, maxParticles); // Bz, Bgz, extBz, pad
    const bFieldGrads = storageBuffer('bFieldGrads', VEC4_SIZE, maxParticles); // dBzdx, dBzdy, dBgzdx, dBgzdy

    // Total force accumulator (sum of all force types)
    const totalForceX = storageBuffer('totalForceX', FLOAT_SIZE, maxParticles);
    const totalForceY = storageBuffer('totalForceY', FLOAT_SIZE, maxParticles);

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
        radius, gamma, magMoment, angMomentum, axMod, yukMod,
        velX, velY, angVel, invMass, radiusSq,
        // Metadata
        flags, color,
        // Forces
        forces0, forces1, forces2, forces3, forces4, forces5,
        torques, bFields, bFieldGrads,
        totalForceX, totalForceY,
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
    // External fields (Phase 2)
    f[18] = params.extGravity || 0;
    f[19] = params.extGravityAngle || 0;
    f[20] = params.extElectric || 0;
    f[21] = params.extElectricAngle || 0;
    f[22] = params.extBz || 0;
    f[23] = params.bounceFriction || 0.4;
    // Precomputed external field directions
    f[24] = (params.extGravity || 0) * Math.cos(params.extGravityAngle || 0); // extGx
    f[25] = (params.extGravity || 0) * Math.sin(params.extGravityAngle || 0); // extGy
    f[26] = (params.extElectric || 0) * Math.cos(params.extElectricAngle || 0); // extEx
    f[27] = (params.extElectric || 0) * Math.sin(params.extElectricAngle || 0); // extEy
    f[28] = params.axionCoupling || 0.05;
    f[29] = params.higgsCoupling || 1.0;
    u[30] = params.particleCount || 0;  // actual alive particle count for dispatch sizing

    device.queue.writeBuffer(buffer, 0, data);
}
