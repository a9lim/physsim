/**
 * @fileoverview GPU buffer allocation for SoA particle state.
 *
 * Creates and manages all GPUBuffer instances for the particle system.
 * Buffers are fixed-capacity (MAX_PARTICLES), indexed by particle slot.
 */

// Signal delay history constants
const HISTORY_LEN = 256;

// Boson pool constants
const MAX_PHOTONS = 512;
const MAX_PIONS = 256;

// Quadtree node size in bytes (20 u32 words = 80 bytes, must match tree-build.wgsl)
const QTNODE_SIZE_BYTES = 80;

/** @param {GPUDevice} device */
export function createParticleBuffers(device, maxParticles) {
    const FLOAT_SIZE = 4;
    const UINT_SIZE = 4;

    // Total SoA capacity: MAX_PARTICLES * 2 to accommodate ghost entries appended after alive particles
    const soaCapacity = maxParticles * 2;

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

    // Core state (f32 per particle) — sized for particles + ghosts
    const posX = storageBuffer('posX', FLOAT_SIZE, soaCapacity);
    const posY = storageBuffer('posY', FLOAT_SIZE, soaCapacity);
    const velWX = storageBuffer('velWX', FLOAT_SIZE, soaCapacity);
    const velWY = storageBuffer('velWY', FLOAT_SIZE, soaCapacity);
    const angW = storageBuffer('angW', FLOAT_SIZE, soaCapacity);
    const mass = storageBuffer('mass', FLOAT_SIZE, soaCapacity);
    const baseMass = storageBuffer('baseMass', FLOAT_SIZE, maxParticles);
    const charge = storageBuffer('charge', FLOAT_SIZE, soaCapacity);

    // Derived/cached — sized for particles + ghosts
    const radius = storageBuffer('radius', FLOAT_SIZE, soaCapacity);
    const gamma = storageBuffer('gamma', FLOAT_SIZE, soaCapacity);

    // Derived/cached (Phase 2) — sized for particles + ghosts
    const magMoment = storageBuffer('magMoment', FLOAT_SIZE, soaCapacity);
    const angMomentum = storageBuffer('angMomentum', FLOAT_SIZE, soaCapacity);
    const axMod = storageBuffer('axMod', FLOAT_SIZE, soaCapacity);
    const yukMod = storageBuffer('yukMod', FLOAT_SIZE, soaCapacity);
    const velX = storageBuffer('velX', FLOAT_SIZE, soaCapacity);  // coordinate velocity
    const velY = storageBuffer('velY', FLOAT_SIZE, soaCapacity);
    const angVel = storageBuffer('angVel', FLOAT_SIZE, soaCapacity);
    const invMass = storageBuffer('invMass', FLOAT_SIZE, soaCapacity);
    const radiusSq = storageBuffer('radiusSq', FLOAT_SIZE, soaCapacity);

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

    // Particle metadata — flags sized for particles + ghosts
    const flags = storageBuffer('flags', UINT_SIZE, soaCapacity);
    const color = storageBuffer('color', UINT_SIZE, maxParticles);

    // Particle ID for ghost->original mapping
    const particleId = storageBuffer('particleId', UINT_SIZE, soaCapacity);

    // Ghost particle counter (single atomic u32)
    const ghostCounter = device.createBuffer({
        label: 'ghostCounter',
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Ghost output SoA buffers (separate from particle buffers to avoid aliasing)
    // After dispatch, ghost data is copied into the main SoA arrays at offset aliveCount.
    const ghostPosX = storageBuffer('ghostPosX', FLOAT_SIZE, maxParticles);
    const ghostPosY = storageBuffer('ghostPosY', FLOAT_SIZE, maxParticles);
    const ghostVelWX = storageBuffer('ghostVelWX', FLOAT_SIZE, maxParticles);
    const ghostVelWY = storageBuffer('ghostVelWY', FLOAT_SIZE, maxParticles);
    const ghostAngW = storageBuffer('ghostAngW', FLOAT_SIZE, maxParticles);
    const ghostMass = storageBuffer('ghostMass', FLOAT_SIZE, maxParticles);
    const ghostCharge = storageBuffer('ghostCharge', FLOAT_SIZE, maxParticles);
    const ghostFlags = storageBuffer('ghostFlags', UINT_SIZE, maxParticles);
    const ghostRadius = storageBuffer('ghostRadius', FLOAT_SIZE, maxParticles);
    const ghostMagMoment = storageBuffer('ghostMagMoment', FLOAT_SIZE, maxParticles);
    const ghostAngMomentum = storageBuffer('ghostAngMomentum', FLOAT_SIZE, maxParticles);
    const ghostParticleId = storageBuffer('ghostParticleId', UINT_SIZE, maxParticles);

    // Ghost original index mapping (which alive particle each ghost copies)
    const ghostOriginalIdx = storageBuffer('ghostOriginalIdx', UINT_SIZE, maxParticles);

    // Staging buffer for reading back ghost count
    const ghostCountStaging = device.createBuffer({
        label: 'ghostCountStaging',
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── Quadtree buffers (Phase 3: GPU Barnes-Hut) ──
    // Each QTNode is 20 u32 words = 80 bytes
    const QT_NODE_SIZE = 80; // bytes (20 x u32)
    const QT_MAX_NODES = maxParticles * 6;

    // Node buffer: flat array of u32, accessed atomically for CAS insertion
    const qtNodeBuffer = device.createBuffer({
        label: 'qtNodes',
        size: QT_NODE_SIZE * QT_MAX_NODES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Node counter (atomic u32) — next free slot index
    const qtNodeCounter = device.createBuffer({
        label: 'qtNodeCounter',
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Bounds reduction output (minX, minY, maxX, maxY as 4 atomic i32s for fixed-point)
    const qtBoundsBuffer = device.createBuffer({
        label: 'qtBounds',
        size: 16, // 4 x i32 (fixed-point for atomicMin/atomicMax)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Visitor flags for bottom-up aggregate (one u32 per node)
    const qtVisitorFlags = device.createBuffer({
        label: 'qtVisitorFlags',
        size: 4 * QT_MAX_NODES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

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

    // ── Collision buffers (Phase 3: collision detection/resolution) ──
    // Collision pair append buffer: stores (idx1, idx2) pairs found by broadphase
    // Max pairs = MAX_PARTICLES (generous upper bound)
    const collisionPairBuffer = device.createBuffer({
        label: 'collisionPairs',
        size: 8 * maxParticles, // u32 pairs: (idx1, idx2) = 8 bytes each
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Collision pair counter (atomic u32)
    const collisionPairCounter = device.createBuffer({
        label: 'collisionPairCounter',
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Merge results buffer: stores merge/annihilation events for post-processing
    // Each event: { x, y, energy, type } = 16 bytes
    const mergeResultBuffer = device.createBuffer({
        label: 'mergeResults',
        size: 16 * maxParticles,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const mergeResultCounter = device.createBuffer({
        label: 'mergeResultCounter',
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Staging buffers for readback of merge results
    const mergeCountStaging = device.createBuffer({
        label: 'mergeCountStaging',
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const mergeResultStaging = device.createBuffer({
        label: 'mergeResultStaging',
        size: 16 * maxParticles,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── Death metadata buffers (Phase 3: dead particle GC) ──
    // Written when a particle is retired (merged/annihilated/despawned)
    const deathTime = storageBuffer('deathTime', FLOAT_SIZE, soaCapacity);
    const deathMass = storageBuffer('deathMass', FLOAT_SIZE, soaCapacity);
    const deathAngVel = storageBuffer('deathAngVel', FLOAT_SIZE, soaCapacity);

    // Free stack for slot reuse (managed by dead GC shader)
    const freeStack = storageBuffer('freeStack', UINT_SIZE, maxParticles);
    const freeTop = device.createBuffer({
        label: 'freeTop',
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // ── Phase 4: 1PN velocity-Verlet correction ──
    // Old 1PN forces for VV correction: vec2<f32> per particle (stored as f32[maxParticles * 2])
    const f1pnOld = storageBuffer('f1pnOld', FLOAT_SIZE, maxParticles * 2);

    // ── Phase 4: Jerk accumulators for radiation ──
    // Analytical jerk accumulated in pair-force pass, consumed by radiation shader
    const jerkX = storageBuffer('jerkX', FLOAT_SIZE, maxParticles);
    const jerkY = storageBuffer('jerkY', FLOAT_SIZE, maxParticles);

    // ── Phase 4: Radiation accumulators ──
    // Per-particle energy accumulators for photon/pion emission thresholds
    const radAccum = storageBuffer('radAccum', FLOAT_SIZE, maxParticles);
    const hawkAccum = storageBuffer('hawkAccum', FLOAT_SIZE, maxParticles);
    const yukawaRadAccum = storageBuffer('yukawaRadAccum', FLOAT_SIZE, maxParticles);

    // Radiation display forces (for renderer force arrows)
    const radDisplayX = storageBuffer('radDisplayX', FLOAT_SIZE, maxParticles);
    const radDisplayY = storageBuffer('radDisplayY', FLOAT_SIZE, maxParticles);

    // Max acceleration for adaptive substepping (single u32, atomicMax in force shader)
    const maxAccelBuffer = storageBuffer('maxAccel', UINT_SIZE, 1);
    const maxAccelStaging = device.createBuffer({
        label: 'maxAccelStaging',
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── Boson pool buffers (Phase 4: photon/pion SoA) ──
    // Photon pool (SoA)
    const phSize = MAX_PHOTONS * FLOAT_SIZE;
    const phPosX = device.createBuffer({ size: phSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'phPosX' });
    const phPosY = device.createBuffer({ size: phSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'phPosY' });
    const phVelX = device.createBuffer({ size: phSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'phVelX' });
    const phVelY = device.createBuffer({ size: phSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'phVelY' });
    const phEnergy = device.createBuffer({ size: phSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'phEnergy' });
    const phEmitterId = device.createBuffer({ size: MAX_PHOTONS * UINT_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'phEmitterId' });
    const phAge = device.createBuffer({ size: MAX_PHOTONS * UINT_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'phAge' }); // u32
    const phFlags = device.createBuffer({ size: MAX_PHOTONS * UINT_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'phFlags' }); // u32: alive, type (em/grav)
    const phCount = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: 'phCount' }); // atomic<u32>

    // Pion pool (SoA)
    const piSize = MAX_PIONS * FLOAT_SIZE;
    const piPosX = device.createBuffer({ size: piSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'piPosX' });
    const piPosY = device.createBuffer({ size: piSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'piPosY' });
    const piWX = device.createBuffer({ size: piSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'piWX' }); // proper velocity
    const piWY = device.createBuffer({ size: piSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'piWY' });
    const piMass = device.createBuffer({ size: piSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'piMass' });
    const piCharge = device.createBuffer({ size: MAX_PIONS * UINT_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'piCharge' }); // i32: +1, -1, 0
    const piEnergy = device.createBuffer({ size: piSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'piEnergy' });
    const piEmitterId = device.createBuffer({ size: MAX_PIONS * UINT_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'piEmitterId' });
    const piAge = device.createBuffer({ size: MAX_PIONS * UINT_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'piAge' }); // u32
    const piFlags = device.createBuffer({ size: MAX_PIONS * UINT_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'piFlags' }); // u32: alive
    const piCount = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: 'piCount' }); // atomic<u32>

    // ── Boson tree buffers (Phase 4: boson gravity BH tree) ──
    const MAX_BOSON_NODES = (MAX_PHOTONS + MAX_PIONS) * 6;
    const bosonTreeNodes = device.createBuffer({
        size: MAX_BOSON_NODES * QTNODE_SIZE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'bosonTreeNodes'
    });
    const bosonTreeCounter = device.createBuffer({
        size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'bosonTreeCounter'
    });

    return {
        maxParticles,
        soaCapacity,
        // Core state
        posX, posY, velWX, velWY, angW, mass, baseMass, charge,
        // Derived
        radius, gamma, magMoment, angMomentum, axMod, yukMod,
        velX, velY, angVel, invMass, radiusSq,
        // Metadata
        flags, color, particleId,
        // Forces
        forces0, forces1, forces2, forces3, forces4, forces5,
        torques, bFields, bFieldGrads,
        totalForceX, totalForceY,
        // Pool
        poolMgmt,
        // Stats
        statsBuffer, statsStagingA, statsStagingB,
        // Adaptive substepping
        maxAccelBuffer, maxAccelStaging,
        // Ghost generation
        ghostCounter, ghostOriginalIdx, ghostCountStaging,
        ghostPosX, ghostPosY, ghostVelWX, ghostVelWY, ghostAngW,
        ghostMass, ghostCharge, ghostFlags,
        ghostRadius, ghostMagMoment, ghostAngMomentum, ghostParticleId,
        // Quadtree (Phase 3)
        qtNodeBuffer, qtNodeCounter, qtBoundsBuffer, qtVisitorFlags,
        QT_MAX_NODES,
        // Collision (Phase 3)
        collisionPairBuffer, collisionPairCounter,
        mergeResultBuffer, mergeResultCounter,
        mergeCountStaging, mergeResultStaging,
        // Death metadata (Phase 3: dead particle GC)
        deathTime, deathMass, deathAngVel,
        freeStack, freeTop,
        // 1PN VV correction (Phase 4)
        f1pnOld,
        // Jerk + radiation accumulators (Phase 4)
        jerkX, jerkY,
        radAccum, hawkAccum, yukawaRadAccum,
        radDisplayX, radDisplayY,
        // Photon pool (Phase 4)
        phPosX, phPosY, phVelX, phVelY, phEnergy, phEmitterId, phAge, phFlags, phCount,
        MAX_PHOTONS,
        // Pion pool (Phase 4)
        piPosX, piPosY, piWX, piWY, piMass, piCharge, piEnergy, piEmitterId, piAge, piFlags, piCount,
        MAX_PIONS,
        // Boson tree (Phase 4)
        bosonTreeNodes, bosonTreeCounter, MAX_BOSON_NODES,

        // Signal delay history (lazy-allocated)
        historyAllocated: false,
        histPosX: null,
        histPosY: null,
        histVelWX: null,
        histVelWY: null,
        histAngW: null,
        histTime: null,
        histMeta: null,

        /**
         * Lazily allocate signal delay history buffers.
         * Called when relativity is first enabled.
         * Total: 6 f32 arrays × 256 × maxParticles × 4 bytes = ~24 MB at 4096 particles
         */
        allocateHistoryBuffers(dev) {
            if (this.historyAllocated) return;
            const size = maxParticles * HISTORY_LEN * 4; // f32 bytes

            this.histPosX = dev.createBuffer({
                size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'histPosX'
            });
            this.histPosY = dev.createBuffer({
                size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'histPosY'
            });
            this.histVelWX = dev.createBuffer({
                size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'histVelWX'
            });
            this.histVelWY = dev.createBuffer({
                size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'histVelWY'
            });
            this.histAngW = dev.createBuffer({
                size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'histAngW'
            });
            this.histTime = dev.createBuffer({
                size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'histTime'
            });
            // Per-particle: write index (u32) and count (u32), packed as u32[maxParticles * 2]
            this.histMeta = dev.createBuffer({
                size: maxParticles * 2 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'histMeta'
            });

            this.historyAllocated = true;
        },

        /** Destroy all buffers */
        destroy() {
            for (const key of Object.keys(this)) {
                if (this[key] && typeof this[key].destroy === 'function') {
                    this[key].destroy();
                }
            }
            this.historyAllocated = false;
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
    f[31] = params.bhTheta || 0.5;     // Barnes-Hut opening angle

    device.queue.writeBuffer(buffer, 0, data);
}
