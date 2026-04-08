/**
 * @fileoverview GPU buffer allocation for particle state.
 *
 * Creates and manages all GPUBuffer instances for the particle system.
 * Buffers are fixed-capacity (MAX_PARTICLES), indexed by particle slot.
 *
 * Packed struct buffers reduce storage buffer count per shader stage:
 *   ParticleState  (36 bytes) — posX,posY,velWX,velWY,mass,charge,angW,baseMass,flags
 *   ParticleAux    (20 bytes) — radius,particleId,deathTime,deathMass,deathAngVel
 *   RadiationState (48 bytes) — accumulators,display,quadrupole contrib
 *   Photon         (32 bytes) — pos,vel,energy,emitterId,lifetime,flags
 *   Pion           (48 bytes) — pos,w,mass,charge,energy,emitterId,age,flags,pad
 */

import {
    HISTORY_SIZE, GPU_MAX_PHOTONS, GPU_MAX_PIONS, MAX_TRAIL_LENGTH,
    GPU_SCALAR_GRID, GPU_MAX_LEPTONS,
} from '../config.js';
import { HIST_STRIDE as HIST_STRIDE_CONST } from './gpu-constants.js';

// Signal delay history constants
const HISTORY_LEN = HISTORY_SIZE;

// Quadtree node size in bytes (20 u32 words = 80 bytes, must match tree-build.wgsl)
const QTNODE_SIZE_BYTES = 80;

// Packed struct sizes (must match common.wgsl struct definitions)
const PARTICLE_STATE_SIZE = 36;  // 9 × 4 bytes
const PARTICLE_AUX_SIZE = 20;   // 5 × 4 bytes
const RADIATION_STATE_SIZE = 48; // 12 × 4 bytes (accumulators + display + quadrupole scratch)
const PHOTON_SIZE = 32;          // 8 × 4 bytes
const PION_SIZE = 48;            // 12 × 4 bytes
const DERIVED_SIZE = 32;         // 8 × f32 (ParticleDerived)
const VEC2_SIZE = 8;             // 2 × f32
const VEC4_SIZE = 16;            // 4 × f32
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

    // Packed axMod + yukMod + higgsMod (vec4, 16 bytes per particle)
    const axYukMod = storageBuffer('axYukMod', VEC4_SIZE, soaCapacity);

    // Packed AllForces struct (160 bytes per particle)
    // Replaces: forces0-5, torques, bFields, bFieldGrads, totalForce (10 buffers → 1)
    const allForces = storageBuffer('allForces', ALLFORCES_SIZE, maxParticles);

    // Particle color (u32 per particle, not packed — only used by renderer)
    const color = storageBuffer('color', UINT_SIZE, maxParticles);

    // ── Radiation state (packed struct) ──
    // RadiationState (48 bytes): accumulators, display, quadrupole contrib
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
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Bounds reduction output (minX, minY, maxX, maxY as 4 atomic i32s for fixed-point)
    const qtBoundsBuffer = device.createBuffer({
        label: 'qtBounds',
        size: 16, // 4 x i32 (fixed-point for atomicMin/atomicMax)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Visitor flags for bottom-up aggregate (one u32 per node)
    const qtVisitorFlags = device.createBuffer({
        label: 'qtVisitorFlags',
        size: 4 * QT_MAX_NODES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Pool management: aliveCount + freeStack + freeTop
    // Packed into one buffer: [aliveCount: u32, freeTop: u32, freeStack: u32[maxParticles]]
    const poolMgmt = storageBuffer('poolMgmt', UINT_SIZE, maxParticles + 2);

    // Stats readback buffer (double-buffered: aggregates + PE + field energy + selected particle)
    const STATS_BUFFER_SIZE = 512; // 128 f32
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

    const collisionClaims = device.createBuffer({
        label: 'collisionClaims',
        size: 4 * maxParticles,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
    // Photon pool: array<Photon, GPU_MAX_PHOTONS> (32 bytes each)
    const photonPool = storageBuffer('photonPool', PHOTON_SIZE, GPU_MAX_PHOTONS);
    const phCount = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: 'phCount',
    }); // atomic<u32>

    // Pion pool: shared buffer for pions + leptons (48 bytes each)
    const PION_POOL_CAP = GPU_MAX_PIONS + GPU_MAX_LEPTONS;
    const pionPool = storageBuffer('pionPool', PION_SIZE, PION_POOL_CAP);
    const piCount = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: 'piCount',
    }); // atomic<u32>

    // Pion annihilation claim buffer: atomic CAS to prevent double-annihilation races
    const pionClaims = device.createBuffer({
        size: 4 * PION_POOL_CAP,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'pionClaims',
    });

    // ── Boson tree buffers (Phase 4: boson gravity BH tree) ──
    const MAX_BOSON_NODES = (GPU_MAX_PHOTONS + PION_POOL_CAP) * 6;
    const bosonTreeNodes = device.createBuffer({
        size: MAX_BOSON_NODES * QTNODE_SIZE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'bosonTreeNodes'
    });
    const bosonTreeCounter = device.createBuffer({
        size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'bosonTreeCounter'
    });
    const bosonVisitorFlags = device.createBuffer({
        label: 'bosonVisitorFlags',
        size: MAX_BOSON_NODES * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
        collisionClaims,
        mergeCountStaging, mergeResultStaging,
        // Free stack (Phase 3: dead particle GC)
        freeStack, freeTop, freeTopStaging, freeStackStaging,
        // 1PN VV correction (Phase 4)
        f1pnOld,
        // Photon pool (Phase 4, packed struct)
        photonPool, phCount,
        GPU_MAX_PHOTONS,
        // Pion pool (Phase 4, packed struct)
        pionPool, piCount, pionClaims,
        GPU_MAX_PIONS, PION_POOL_CAP,
        // Boson tree (Phase 4)
        bosonTreeNodes, bosonTreeCounter, bosonVisitorFlags, MAX_BOSON_NODES,

        // Signal delay history (lazy-allocated)
        historyAllocated: false,
        histData: null,    // interleaved [posX, posY, velX, velY, angW, time] per sample (stride 6)
        histMeta: null,    // [writeIdx, count, creationTimeBits, _pad] per particle (stride 4)

        /**
         * Lazily allocate signal delay history buffers.
         * Called when relativity is first enabled.
         * histData: interleaved f32, stride 6 per sample, HISTORY_LEN samples per particle
         * histMeta: 4 u32 per particle (writeIdx, count, creationTimeBits, _pad)
         */
        allocateHistoryBuffers(dev) {
            if (this.historyAllocated) return;
            const HIST_STRIDE = HIST_STRIDE_CONST;
            const dataSize = maxParticles * HISTORY_LEN * HIST_STRIDE * 4; // f32 each
            this.histData = dev.createBuffer({
                label: 'histData',
                size: dataSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            // histMeta: 4 u32 per particle (writeIdx, count, creationTimeBits, _pad)
            const metaSize = maxParticles * 4 * 4; // 4 u32 per particle
            this.histMeta = dev.createBuffer({
                label: 'histMeta',
                size: metaSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
/**
 * Allocate GPU buffers for one scalar field instance.
 * @param {GPUDevice} device
 * @param {string} label - 'higgs' or 'axion'
 * @param {number} maxParticles
 * @returns {Object} Buffer set for one field
 */
export function createFieldBuffers(device, label, maxParticles) {
    const gridBytes = FIELD_GRID_SQ * 4; // f32
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    const complexBytes = FIELD_GRID_SQ * 2 * 4; // interleaved complex f32 (re,im pairs)

    return {
        field:         device.createBuffer({ label: `${label}-field`,         size: gridBytes, usage }),
        fieldDot:      device.createBuffer({ label: `${label}-fieldDot`,      size: gridBytes, usage }),
        source:        device.createBuffer({ label: `${label}-source`,        size: gridBytes, usage }),
        gradX:         device.createBuffer({ label: `${label}-gradX`,         size: gridBytes, usage }),
        gradY:         device.createBuffer({ label: `${label}-gradY`,         size: gridBytes, usage }),
        energyDensity: device.createBuffer({ label: `${label}-energyDensity`, size: gridBytes, usage }),
        // Thermal grid (Higgs only, but allocated for both to keep layout uniform)
        thermal:       device.createBuffer({ label: `${label}-thermal`,       size: gridBytes, usage }),
        // Self-gravity: FFT convolution on full grid
        sgPhiFull:     device.createBuffer({ label: `${label}-sgPhiFull`,     size: gridBytes, usage }),
        sgGradX:       device.createBuffer({ label: `${label}-sgGradX`,       size: gridBytes, usage }),
        sgGradY:       device.createBuffer({ label: `${label}-sgGradY`,       size: gridBytes, usage }),
        // FFT ping-pong buffers (interleaved complex: GRID*GRID*2 f32)
        fftA:          device.createBuffer({ label: `${label}-fftA`,          size: complexBytes, usage }),
        fftB:          device.createBuffer({ label: `${label}-fftB`,          size: complexBytes, usage }),
        // Precomputed Green's function in Fourier space (uploaded from CPU)
        greenHat:      device.createBuffer({ label: `${label}-greenHat`,      size: complexBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    };
}

/**
 * Allocate atomic grid buffer for single-pass PQS deposition.
 * Layout: atomic<i32>[FIELD_GRID_SQ] — fixed-point accumulator, cleared by finalizeDeposit.
 * @param {GPUDevice} device
 */
export function createAtomicGridBuffer(device) {
    return device.createBuffer({
        label: 'atomic-deposit-grid',
        size: FIELD_GRID_SQ * 4,
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
        events: device.createBuffer({ label: 'disint-events', size: maxEvents * 48, usage }), // DisintEvent = 48 bytes
        counter: device.createBuffer({ label: 'disint-counter', size: 4, usage }),
        staging: device.createBuffer({ label: 'disint-staging', size: maxEvents * 48,
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

/**
 * Allocate kugelblitz collapse event buffers.
 * KugelblitzEvent = 32 bytes (8 × f32): x, y, px, py, energy, charge, angL, count.
 * Max 1 event per substep.
 */
export function createKugelblitzBuffers(device) {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    return {
        events: device.createBuffer({ label: 'kugelblitz-events', size: 32, usage }),
        counter: device.createBuffer({ label: 'kugelblitz-counter', size: 4, usage }),
        staging: device.createBuffer({ label: 'kugelblitz-staging', size: 32,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        counterStaging: device.createBuffer({ label: 'kugelblitz-count-staging', size: 4,
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
    FIELD_GRID_RES, FIELD_GRID_SQ,
    PARTICLE_STATE_SIZE, PARTICLE_AUX_SIZE, RADIATION_STATE_SIZE,
    PHOTON_SIZE, PION_SIZE, DERIVED_SIZE, VEC2_SIZE, VEC4_SIZE, ALLFORCES_SIZE, TRAIL_LEN,
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

export function writeFrameUniforms(device, buffer, params) {
    // Pack into pre-allocated Float32Array matching WGSL struct layout.
    // Writes the FULL 256-byte buffer — call once per frame for slow-changing fields.
    // Substep-varying fields (dt, simTime, aliveCount, particleCount, frameCount)
    // are overwritten per substep by writeSubstepUniforms().
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
    // Precomputed external field directions (use cached trig from setToggles)
    f[24] = params.extGx ?? ((params.extGravity || 0) * Math.cos(params.extGravityAngle || 0));
    f[25] = params.extGy ?? ((params.extGravity || 0) * Math.sin(params.extGravityAngle || 0));
    f[26] = params.extEx ?? ((params.extElectric || 0) * Math.cos(params.extElectricAngle || 0));
    f[27] = params.extEy ?? ((params.extElectric || 0) * Math.sin(params.extElectricAngle || 0));
    f[28] = params.axionCoupling || 0.05;
    f[29] = params.higgsCoupling || 1.0;
    u[30] = params.particleCount || 0;  // actual alive particle count for dispatch sizing
    f[31] = params.bhTheta || 0.5;     // Barnes-Hut opening angle
    u[32] = params.frameCount || 0;    // frameCount field in SimUniforms

    device.queue.writeBuffer(buffer, 0, _uniformData);
}

// Pre-allocated scratch for per-substep partial uniform writes (avoids GC)
const _substepF32 = new Float32Array(1);
const _substepU32 = new Uint32Array(1);

// Byte offsets of substep-varying fields within SimUniforms (index * 4)
const UNIFORM_DT_OFFSET = 0;           // f[0]
const UNIFORM_SIMTIME_OFFSET = 4;      // f[1]
const UNIFORM_ALIVE_OFFSET = 68;       // u[17]
const UNIFORM_PCOUNT_OFFSET = 120;     // u[30]
const UNIFORM_FRAME_OFFSET = 128;      // u[32]

/**
 * Write only the 5 substep-varying fields at their byte offsets within the
 * existing SimUniforms buffer. Avoids re-uploading the full 256 bytes per substep.
 */
export function writeSubstepUniforms(device, buffer, dt, simTime, aliveCount, particleCount, frameCount) {
    _substepF32[0] = dt;
    device.queue.writeBuffer(buffer, UNIFORM_DT_OFFSET, _substepF32);
    _substepF32[0] = simTime;
    device.queue.writeBuffer(buffer, UNIFORM_SIMTIME_OFFSET, _substepF32);
    _substepU32[0] = aliveCount;
    device.queue.writeBuffer(buffer, UNIFORM_ALIVE_OFFSET, _substepU32);
    _substepU32[0] = particleCount;
    device.queue.writeBuffer(buffer, UNIFORM_PCOUNT_OFFSET, _substepU32);
    _substepU32[0] = frameCount;
    device.queue.writeBuffer(buffer, UNIFORM_FRAME_OFFSET, _substepU32);
}
