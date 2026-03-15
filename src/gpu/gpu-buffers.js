/**
 * @fileoverview GPU buffer allocation for particle state.
 *
 * Creates and manages all GPUBuffer instances for the particle system.
 * Buffers are fixed-capacity (MAX_PARTICLES), indexed by particle slot.
 *
 * Packed struct buffers reduce storage buffer count per shader stage:
 *   ParticleState  (36 bytes) — posX,posY,velWX,velWY,mass,charge,angW,baseMass,flags
 *   ParticleAux    (20 bytes) — radius,particleId,deathTime,deathMass,deathAngVel
 *   RadiationState (96 bytes) — jerk,accumulators,display,quadrupole history,Larmor backward-diff
 *   Photon         (32 bytes) — pos,vel,energy,emitterId,lifetime,flags
 *   Pion           (48 bytes) — pos,w,mass,charge,energy,emitterId,age,flags,pad
 */

import {
    HISTORY_SIZE, MAX_PHOTONS, MAX_PIONS, MAX_TRAIL_LENGTH,
    GPU_SCALAR_GRID, GPU_SELFGRAV_GRID,
} from '../config.js';

// Signal delay history constants
const HISTORY_LEN = HISTORY_SIZE;

// Quadtree node size in bytes (20 u32 words = 80 bytes, must match tree-build.wgsl)
const QTNODE_SIZE_BYTES = 80;

// Packed struct sizes (must match common.wgsl struct definitions)
const PARTICLE_STATE_SIZE = 36;  // 9 × 4 bytes
const PARTICLE_AUX_SIZE = 20;   // 5 × 4 bytes
const RADIATION_STATE_SIZE = 96; // 24 × 4 bytes (was 64 = 16 fields; added 5 Larmor backward-diff history + 3 pad)
const PHOTON_SIZE = 32;          // 8 × 4 bytes
const PION_SIZE = 48;            // 12 × 4 bytes
const DERIVED_SIZE = 32;         // 8 × f32 (ParticleDerived)
const VEC2_SIZE = 8;             // 2 × f32
const ALLFORCES_SIZE = 160;      // 10 × vec4

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

    // ── Packed particle struct buffers ──

    // ParticleState (36 bytes): posX,posY,velWX,velWY,mass,charge,angW,baseMass,flags
    // Sized for particles + ghosts
    const particleState = storageBuffer('particleState', PARTICLE_STATE_SIZE, soaCapacity);

    // ParticleAux (20 bytes): radius,particleId,deathTime,deathMass,deathAngVel
    // Sized for particles + ghosts
    const particleAux = storageBuffer('particleAux', PARTICLE_AUX_SIZE, soaCapacity);

    // Packed derived state: ParticleDerived struct (32 bytes per particle)
    // Replaces: magAngMom, invMassRadSq, vel, angVel (4 buffers → 1)
    const derived = storageBuffer('derived', DERIVED_SIZE, soaCapacity);

    // Packed axMod + yukMod (vec2, 8 bytes per particle)
    const axYukMod = storageBuffer('axYukMod', VEC2_SIZE, soaCapacity);

    // Packed AllForces struct (160 bytes per particle)
    // Replaces: forces0-5, torques, bFields, bFieldGrads, totalForce (10 buffers → 1)
    const allForces = storageBuffer('allForces', ALLFORCES_SIZE, maxParticles);

    // Particle color (u32 per particle, not packed — only used by renderer)
    const color = storageBuffer('color', UINT_SIZE, maxParticles);

    // ── Radiation state (packed struct) ──
    // RadiationState (96 bytes): jerk, accumulators, display, quadrupole history, Larmor backward-diff history
    const radiationState = storageBuffer('radiationState', RADIATION_STATE_SIZE, maxParticles);

    // ── Quadrupole radiation reduction buffer ──
    // Workgroup partial sums for 2-pass reduction. Layout:
    //   [0 .. MAX_WG*4): CoM pass: {comXw, comYw, totalMass, totalKE} per workgroup
    //   [MAX_WG*4 .. MAX_WG*12): Contrib pass: {d3Ixx,d3Ixy,d3Iyy, d3Qxx,d3Qxy,d3Qyy, totalD3I,totalD3Q} per wg
    const MAX_QUAD_WG = Math.ceil(maxParticles / 64);
    const quadReductionBuf = storageBuffer('quadReduction', 4, MAX_QUAD_WG * 12); // f32 elements

    // ── Ghost generation ──
    // Ghost particle counter (single atomic u32)
    const ghostCounter = device.createBuffer({
        label: 'ghostCounter',
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Ghost output struct buffers (separate from particle buffers to avoid aliasing)
    // After dispatch, ghost data is copied into the main arrays at offset aliveCount.
    const ghostState = storageBuffer('ghostState', PARTICLE_STATE_SIZE, maxParticles);
    const ghostAux = storageBuffer('ghostAux', PARTICLE_AUX_SIZE, maxParticles);
    const ghostDerived = storageBuffer('ghostDerived', DERIVED_SIZE, maxParticles);

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

    // Stats readback buffer (double-buffered: aggregates + selected particle data)
    const STATS_BUFFER_SIZE = 256; // 64 f32: 16 aggregates + 48 selected particle
    const statsBuffer = device.createBuffer({
        label: 'statsBuffer',
        size: STATS_BUFFER_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const statsStagingA = device.createBuffer({
        label: 'statsStagingA',
        size: STATS_BUFFER_SIZE,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const statsStagingB = device.createBuffer({
        label: 'statsStagingB',
        size: STATS_BUFFER_SIZE,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── Collision buffers (Phase 3: collision detection/resolution) ──
    const collisionPairBuffer = device.createBuffer({
        label: 'collisionPairs',
        size: 8 * maxParticles,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const collisionPairCounter = device.createBuffer({
        label: 'collisionPairCounter',
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

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

    // Free stack for slot reuse (managed by dead GC shader)
    const freeStack = storageBuffer('freeStack', UINT_SIZE, maxParticles);
    const freeTop = device.createBuffer({
        label: 'freeTop',
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const freeTopStaging = device.createBuffer({
        label: 'freeTopStaging',
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const freeStackStaging = device.createBuffer({
        label: 'freeStackStaging',
        size: maxParticles * UINT_SIZE,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── Phase 4: 1PN velocity-Verlet correction ──
    const f1pnOld = storageBuffer('f1pnOld', FLOAT_SIZE, maxParticles * 2);

    // Max acceleration for adaptive substepping (single u32, atomicMax in force shader)
    const maxAccelBuffer = storageBuffer('maxAccel', UINT_SIZE, 1);
    const maxAccelStaging = device.createBuffer({
        label: 'maxAccelStaging',
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── Boson pool buffers (Phase 4: packed photon/pion structs) ──
    // Photon pool: array<Photon, MAX_PHOTONS> (32 bytes each)
    const photonPool = storageBuffer('photonPool', PHOTON_SIZE, MAX_PHOTONS);
    const phCount = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: 'phCount',
    }); // atomic<u32>

    // Pion pool: array<Pion, MAX_PIONS> (48 bytes each)
    const pionPool = storageBuffer('pionPool', PION_SIZE, MAX_PIONS);
    const piCount = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: 'piCount',
    }); // atomic<u32>

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
        // Packed particle structs
        particleState, particleAux,
        // Derived (packed struct)
        derived, axYukMod,
        // Color (separate — renderer only)
        color,
        // Forces (packed AllForces struct)
        allForces,
        // Radiation (packed RadiationState struct)
        radiationState,
        // Quadrupole reduction (workgroup partial sums)
        quadReductionBuf, MAX_QUAD_WG,
        // Pool
        poolMgmt,
        // Stats
        statsBuffer, statsStagingA, statsStagingB,
        // Adaptive substepping
        maxAccelBuffer, maxAccelStaging,
        // Ghost generation
        ghostCounter, ghostOriginalIdx, ghostCountStaging,
        ghostState, ghostAux, ghostDerived,
        // Quadtree (Phase 3)
        qtNodeBuffer, qtNodeCounter, qtBoundsBuffer, qtVisitorFlags,
        QT_MAX_NODES,
        // Collision (Phase 3)
        collisionPairBuffer, collisionPairCounter,
        mergeResultBuffer, mergeResultCounter,
        mergeCountStaging, mergeResultStaging,
        // Free stack (Phase 3: dead particle GC)
        freeStack, freeTop, freeTopStaging, freeStackStaging,
        // 1PN VV correction (Phase 4)
        f1pnOld,
        // Photon pool (Phase 4, packed struct)
        photonPool, phCount,
        MAX_PHOTONS,
        // Pion pool (Phase 4, packed struct)
        pionPool, piCount,
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

// ── Scalar Field Grid Buffers ──
// Default GRID_RES = 64 (matches CPU SCALAR_GRID), configurable as power-of-2
const FIELD_GRID_RES = GPU_SCALAR_GRID;
const FIELD_GRID_SQ = FIELD_GRID_RES * FIELD_GRID_RES;
const COARSE_RES = GPU_SELFGRAV_GRID;
const COARSE_SQ = COARSE_RES * COARSE_RES;
const PQS_STENCIL_SIZE = 16; // 4x4 stencil per particle

/**
 * Allocate GPU buffers for one scalar field instance.
 * @param {GPUDevice} device
 * @param {string} label - 'higgs' or 'axion'
 * @param {number} maxParticles
 * @returns {Object} Buffer set for one field
 */
export function createFieldBuffers(device, label, maxParticles) {
    const gridBytes = FIELD_GRID_SQ * 4; // f32
    const coarseBytes = COARSE_SQ * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    return {
        field:         device.createBuffer({ label: `${label}-field`,         size: gridBytes, usage }),
        fieldDot:      device.createBuffer({ label: `${label}-fieldDot`,      size: gridBytes, usage }),
        laplacian:     device.createBuffer({ label: `${label}-laplacian`,     size: gridBytes, usage }),
        source:        device.createBuffer({ label: `${label}-source`,        size: gridBytes, usage }),
        gradX:         device.createBuffer({ label: `${label}-gradX`,         size: gridBytes, usage }),
        gradY:         device.createBuffer({ label: `${label}-gradY`,         size: gridBytes, usage }),
        energyDensity: device.createBuffer({ label: `${label}-energyDensity`, size: gridBytes, usage }),
        // Thermal grid (Higgs only, but allocated for both to keep layout uniform)
        thermal:       device.createBuffer({ label: `${label}-thermal`,       size: gridBytes, usage }),
        // Self-gravity coarse grid
        coarseRho:     device.createBuffer({ label: `${label}-sgRho`,         size: coarseBytes, usage }),
        coarsePhi:     device.createBuffer({ label: `${label}-sgPhi`,         size: coarseBytes, usage }),
        sgPhiFull:     device.createBuffer({ label: `${label}-sgPhiFull`,     size: gridBytes, usage }),
        sgGradX:       device.createBuffer({ label: `${label}-sgGradX`,       size: gridBytes, usage }),
        sgGradY:       device.createBuffer({ label: `${label}-sgGradY`,       size: gridBytes, usage }),
        // sgInvR removed — computed inline in field-selfgrav.wgsl
    };
}

/**
 * Allocate shared PQS scratch buffer for two-pass scatter/gather deposition.
 * Layout: f32[maxParticles * PQS_STENCIL_SIZE] — each particle writes 16 weights.
 * @param {GPUDevice} device
 * @param {number} maxParticles
 */
export function createPQSScratchBuffer(device, maxParticles) {
    return device.createBuffer({
        label: 'pqs-scratch',
        size: maxParticles * PQS_STENCIL_SIZE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
}

/**
 * Allocate shared PQS index buffer: stores (baseIx, baseIy) per particle for gather pass.
 * Layout: u32[maxParticles * 2]
 */
export function createPQSIndexBuffer(device, maxParticles) {
    return device.createBuffer({
        label: 'pqs-indices',
        size: maxParticles * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
}

/**
 * Allocate heatmap buffers.
 * @param {GPUDevice} device
 * @param {number} gridSize - default 64 (HEATMAP_GRID)
 */
export function createHeatmapBuffers(device, gridSize = 64) {
    const bytes = gridSize * gridSize * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    return {
        gravPotential:   device.createBuffer({ label: 'heatmap-grav',   size: bytes, usage }),
        elecPotential:   device.createBuffer({ label: 'heatmap-elec',   size: bytes, usage }),
        yukawaPotential: device.createBuffer({ label: 'heatmap-yukawa', size: bytes, usage }),
        blurTemp:        device.createBuffer({ label: 'heatmap-blur',   size: bytes, usage }),
        output:          device.createBuffer({ label: 'heatmap-output', size: gridSize * gridSize * 4 * 4, usage }), // RGBA u8 packed as u32
    };
}

/**
 * Allocate excitation event buffers for field excitation deposits.
 * @param {GPUDevice} device
 * @param {number} maxEvents - max events per frame (default 64)
 */
export function createExcitationBuffers(device, maxEvents = 64) {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    return {
        events: device.createBuffer({ label: 'excitation-events', size: maxEvents * 16, usage }), // ExcitationEvent = 16 bytes
        counter: device.createBuffer({ label: 'excitation-counter', size: 4, usage }),
    };
}

/**
 * Allocate disintegration event buffers for tidal fragmentation / Roche transfer.
 * @param {GPUDevice} device
 * @param {number} maxEvents - max events per frame (default 64)
 */
export function createDisintegrationBuffers(device, maxEvents = 64) {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    return {
        events: device.createBuffer({ label: 'disint-events', size: maxEvents * 32, usage }), // DisintEvent = 32 bytes
        counter: device.createBuffer({ label: 'disint-counter', size: 4, usage }),
        staging: device.createBuffer({ label: 'disint-staging', size: maxEvents * 32,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        counterStaging: device.createBuffer({ label: 'disint-count-staging', size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    };
}

/**
 * Allocate pair production event buffers.
 * @param {GPUDevice} device
 * @param {number} maxEvents - max events per frame (default 32)
 */
export function createPairProductionBuffers(device, maxEvents = 32) {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    return {
        events: device.createBuffer({ label: 'pairprod-events', size: maxEvents * 32, usage }), // PairEvent = 32 bytes
        counter: device.createBuffer({ label: 'pairprod-counter', size: 4, usage }),
        staging: device.createBuffer({ label: 'pairprod-staging', size: maxEvents * 32,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        counterStaging: device.createBuffer({ label: 'pairprod-count-staging', size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    };
}

// Trail rendering constants
const TRAIL_LEN = MAX_TRAIL_LENGTH;

/**
 * Allocate trail ring buffer storage for GPU trail rendering.
 * @param {GPUDevice} device
 * @param {number} maxParticles
 */
export function createTrailBuffers(device, maxParticles) {
    const posBytes = maxParticles * TRAIL_LEN * 4; // f32
    const metaBytes = maxParticles * 4; // u32 per particle
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    return {
        trailX:        device.createBuffer({ label: 'trailX',        size: posBytes, usage }),
        trailY:        device.createBuffer({ label: 'trailY',        size: posBytes, usage }),
        trailWriteIdx: device.createBuffer({ label: 'trailWriteIdx', size: metaBytes, usage }),
        trailCount:    device.createBuffer({ label: 'trailCount',    size: metaBytes, usage }),
    };
}

export {
    FIELD_GRID_RES, FIELD_GRID_SQ, COARSE_RES, COARSE_SQ, PQS_STENCIL_SIZE,
    PARTICLE_STATE_SIZE, PARTICLE_AUX_SIZE, RADIATION_STATE_SIZE,
    PHOTON_SIZE, PION_SIZE, DERIVED_SIZE, VEC2_SIZE, ALLFORCES_SIZE, TRAIL_LEN,
};

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
// Pre-allocated uniform data buffer (avoids GC pressure from per-substep allocation)
const _uniformData = new ArrayBuffer(256);
const _uniformF32 = new Float32Array(_uniformData);
const _uniformU32 = new Uint32Array(_uniformData);

export function writeUniforms(device, buffer, params) {
    // Pack into pre-allocated Float32Array matching WGSL struct layout
    const f = _uniformF32;
    const u = _uniformU32;

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
    f[23] = params.bounceFriction ?? 0.4;
    // Precomputed external field directions
    f[24] = (params.extGravity || 0) * Math.cos(params.extGravityAngle || 0); // extGx
    f[25] = (params.extGravity || 0) * Math.sin(params.extGravityAngle || 0); // extGy
    f[26] = (params.extElectric || 0) * Math.cos(params.extElectricAngle || 0); // extEx
    f[27] = (params.extElectric || 0) * Math.sin(params.extElectricAngle || 0); // extEy
    f[28] = params.axionCoupling || 0.05;
    f[29] = params.higgsCoupling || 1.0;
    u[30] = params.particleCount || 0;  // actual alive particle count for dispatch sizing
    f[31] = params.bhTheta || 0.5;     // Barnes-Hut opening angle
    // _pad3 at index 32, _pad4 at index 33 in common.wgsl — reuse for Phase 4
    u[32] = params.frameCount || 0;    // frame counter for RNG seed

    device.queue.writeBuffer(buffer, 0, _uniformData);
}
