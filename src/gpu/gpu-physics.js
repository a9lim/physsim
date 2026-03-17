/**
 * @fileoverview GPUPhysics — WebGPU compute pipeline orchestrator.
 *
 * Phase 2+3+4+5: Full force computation, Boris integrator, tree build, collisions,
 * dead GC, radiation, 1PN VV, boson lifecycle, boson gravity, signal delay history,
 * scalar fields, heatmap, expansion, disintegration, pair production.
 *
 * Dispatch sequence per substep:
 *   1. resetForces (DMA clear)
 *   2. cacheDerived
 *   3. generateGhosts        (Phase 3 — if periodic boundary)
 *   4a-d. treeBuild           (Phase 3 — if BH enabled)
 *   5. computeForces          (Phase 2 pairwise OR Phase 3 tree walk)
 *   5b. externalFields
 *   5c. scalarFieldForces      (Phase 5 — gradient forces + mass/axMod modulation, before Boris)
 *   6-8. borisFused            (halfKick + rotate + halfKick in one pass)
 *   9. spinOrbit
 *  10. applyTorques
 *  11. radiationReaction      (Phase 4 — Larmor, Hawking, pion emission)
 *  12. borisDrift
 *  13. cosmologicalExpansion  (Phase 5 — if expansionEnabled)
 *  14. compute1PN_VV          (Phase 4 — 1PN recompute + VV correction kick)
 *  15. scalarFieldEvolve      (Phase 5 — deposit → [self-grav] → KDK → gradients)
 *  17-18. collisions          (Phase 3 — detect + resolve)
 *  19. fieldExcitations       (Phase 5 — merge KE → wave packets)
 *  20. disintegrationCheck    (Phase 5 — tidal + Roche)
 *  21. bosonUpdate            (Phase 4 — photon/pion drift, absorption, decay)
 *  22. pairProduction         (Phase 5 — photon → particle pair)
 *  24. boundary
 *
 * Post-substep (once per frame):
 *  - heatmap compute + blur   (Phase 5 — every HEATMAP_INTERVAL frames)
 *  - bosonInteraction         (Phase 4 — build boson tree + particle/boson gravity + pion Coulomb)
 *  - deadParticleGC           (Phase 3)
 *  - recordHistory            (Phase 4 — every HISTORY_STRIDE frames)
 */
import { createParticleBuffers, createUniformBuffer, writeUniforms, createFieldBuffers, createAtomicGridBuffer, createHeatmapBuffers, createExcitationBuffers, createDisintegrationBuffers, createPairProductionBuffers, createTrailBuffers, FIELD_GRID_RES, PARTICLE_STATE_SIZE, PARTICLE_AUX_SIZE, RADIATION_STATE_SIZE, PHOTON_SIZE, PION_SIZE, DERIVED_SIZE } from './gpu-buffers.js';
import { fetchShader, createPhase2Pipelines, createGhostGenPipeline, createTreeBuildPipelines, createTreeForcePipeline, createCollisionPipelines, createDeadGCPipeline, createPhase4Pipelines, createFieldDepositPipelines, createFieldEvolvePipelines, createFieldForcesPipelines, createFieldParticleGravPipeline, createFieldSelfGravPipelines, createFFTPipelines, createFieldExcitationPipeline, createHeatmapPipelines, createExpansionPipeline, createDisintegrationPipeline, createPairProductionPipeline, createUpdateColorsPipeline, createTrailRecordPipeline, createHitTestPipeline, createComputeStatsPipeline } from './gpu-pipelines.js';
import { buildWGSLConstants, GRAVITY_BIT, COULOMB_BIT, MAGNETIC_BIT, GRAVITOMAG_BIT, ONE_PN_BIT, RELATIVITY_BIT, SPIN_ORBIT_BIT, RADIATION_BIT, BLACK_HOLE_BIT, DISINTEGRATION_BIT, EXPANSION_BIT, YUKAWA_BIT, HIGGS_BIT, AXION_BIT, BARNES_HUT_BIT, BOSON_INTER_BIT, HIST_META_STRIDE } from './gpu-constants.js';
import {
    HISTORY_STRIDE, GPU_MAX_PHOTONS, GPU_MAX_PIONS,
    GPU_MAX_PARTICLES, GPU_HEATMAP_GRID, PHYSICS_DT,
    COL_MERGE, COL_BOUNCE, BOUND_LOOP,
    COL_NAMES, BOUND_NAMES, TOPO_NAMES,
    SOFTENING_SQ, BH_SOFTENING_SQ,
} from '../config.js';
import { fft2d } from '../fft.js';

const MAX_PARTICLES = GPU_MAX_PARTICLES;

// Pre-allocated typed arrays for per-frame writeBuffer calls (avoid GC pressure)
const _qtNodeCounterData = new Uint32Array([1]);
const _qtBoundsResetData = new Int32Array([2147483647, 2147483647, -2147483647, -2147483647]);
const _postSubDt = new Float32Array(1); // scratch for writing total dt to uniforms post-substep
const _bosonRootData = new Uint32Array(20);
const _bosonRootF32 = new Float32Array(_bosonRootData.buffer);

// Pre-allocated field uniform data (avoids per-substep GC from _writeFieldUniforms)
const _fieldUniformData = new ArrayBuffer(256);
const _fieldUniformF32 = new Float32Array(_fieldUniformData);
const _fieldUniformU32 = new Uint32Array(_fieldUniformData);

// Pre-allocated dispatch uniform buffers (avoids per-frame GC from dispatch methods)
const _disintUniformData = new ArrayBuffer(48);
const _disintUniformF32 = new Float32Array(_disintUniformData);
const _disintUniformU32 = new Uint32Array(_disintUniformData);
const _pairProdUniformData = new ArrayBuffer(48);
const _pairProdUniformF32 = new Float32Array(_pairProdUniformData);
const _pairProdUniformU32 = new Uint32Array(_pairProdUniformData);
const _heatmapUniformData = new ArrayBuffer(96);
const _heatmapUniformF32 = new Float32Array(_heatmapUniformData);
const _heatmapUniformU32 = new Uint32Array(_heatmapUniformData);
const _colorUniformData = new Uint32Array(4);
const _fgUniformData = new ArrayBuffer(16);  // FGUniforms: domainW, domainH, aliveCount, _pad
const _fgUniformF32 = new Float32Array(_fgUniformData);
const _fgUniformU32 = new Uint32Array(_fgUniformData);

// Pre-allocated addParticle buffers (avoids per-call allocation)
const _addParticleStateData = new ArrayBuffer(36);  // PARTICLE_STATE_SIZE
const _addParticleStateF32 = new Float32Array(_addParticleStateData);
const _addParticleStateU32 = new Uint32Array(_addParticleStateData);
const _addParticleAuxData = new ArrayBuffer(20);     // PARTICLE_AUX_SIZE
const _addParticleAuxF32 = new Float32Array(_addParticleAuxData);
const _addParticleAuxU32 = new Uint32Array(_addParticleAuxData);
const _addParticleColorData = new Uint32Array(1);
const _addParticleModData = new Float32Array(2);
const _addParticleRadData = new Float32Array(12);    // RADIATION_STATE_SIZE / 4
const _addParticleMetaBuf = new ArrayBuffer(16);       // 4 u32 for histMeta (stride 4)
const _addParticleMetaU32 = new Uint32Array(_addParticleMetaBuf);
const _addParticleMetaF32 = new Float32Array(_addParticleMetaBuf);
const _zeroU32 = new Uint32Array([0]);                // reusable zero for counter resets
const _retiredFlagU32 = new Uint32Array([2]);         // FLAG_RETIRED for removeParticle
const _deathMetaData = new ArrayBuffer(12);           // deathTime(f32), deathMass(f32), deathAngVel(f32)
const _deathMetaF32 = new Float32Array(_deathMetaData);

// Pre-allocated hitTest uniform data (avoids per-call allocation)
const _hitUniformData = new ArrayBuffer(16);
const _hitUniformF32 = new Float32Array(_hitUniformData);
const _hitUniformU32 = new Uint32Array(_hitUniformData);

// Pre-allocated requestStats uniform data (avoids per-call allocation)
const _statsUniformData = new ArrayBuffer(48);
const _statsUniformF32 = new Float32Array(_statsUniformData);
const _statsUniformU32 = new Uint32Array(_statsUniformData);
const _statsUniformI32 = new Int32Array(_statsUniformData);

// Pre-allocated FFT params data (avoids per-butterfly-stage allocation in self-gravity)
const _fftParamsData = new ArrayBuffer(32);
const _fftParamsF32 = new Float32Array(_fftParamsData);
const _fftParamsU32 = new Uint32Array(_fftParamsData);
const _fftParamsI32 = new Int32Array(_fftParamsData);

// Pre-allocated field excitation event data (max 64 events × 4 floats)
const _excitationEventData = new Float32Array(64 * 4);
const _excitationCountData = new Uint32Array(1);

// Pre-allocated field vacuum data (reused across _initFieldToVacuum calls)
const _FIELD_GRID_SQ = 128 * 128; // GPU_SCALAR_GRID² (max supported)
const _vacuumFieldData = new Float32Array(_FIELD_GRID_SQ);
const _zeroFieldData = new Float32Array(_FIELD_GRID_SQ);

// Pre-allocated Green's hat upload buffer
const _greenHatUploadData = new Float32Array(_FIELD_GRID_SQ * 2);

export default class GPUPhysics {
    /**
     * @param {GPUDevice} device
     * @param {number} domainW
     * @param {number} domainH
     */
    constructor(device, domainW, domainH) {
        this.device = device;
        this.domainW = domainW;
        this.domainH = domainH;
        this.simTime = 0;
        this.aliveCount = 0;

        // Defaults matching CPU path
        this.boundaryMode = 0;  // despawn
        this.topologyMode = 0;  // torus

        this.buffers = createParticleBuffers(device, MAX_PARTICLES);
        this.uniformBuffer = createUniformBuffer(device);

        // Phase 1 pipelines
        this._boundaryPipeline = null;
        this._boundaryBindGroup = null;

        // Shared 4-byte dummy storage buffer (placeholder for unallocated history bind groups)
        this._dummyHistBuf = device.createBuffer({
            label: 'dummy-hist',
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Encoder-side staging buffers for tree build resets.
        // queue.writeBuffer() happens at queue time (before encoder start), so it
        // can't reset state between two tree builds in the same command buffer.
        // These COPY_SRC buffers let us use encoder.copyBufferToBuffer() instead,
        // which executes in-order within the command buffer.
        this._qtCounterSrc = device.createBuffer({
            label: 'qt-counter-src',
            size: 4,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Uint32Array(this._qtCounterSrc.getMappedRange()).set([1]);
        this._qtCounterSrc.unmap();

        this._qtBoundsSrc = device.createBuffer({
            label: 'qt-bounds-src',
            size: 16,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Int32Array(this._qtBoundsSrc.getMappedRange()).set([2147483647, 2147483647, -2147483647, -2147483647]);
        this._qtBoundsSrc.unmap();

        // Phase 2 pipelines + bind groups
        this._phase2 = null;

        // Phase 3: Ghost generation
        this._ghostGenPipeline = null;
        this._ghostGenBindGroups = null;
        this._ghostCount = 0;
        this._ghostCountPending = false;

        // Phase 3: Tree build (GPU Barnes-Hut)
        this._treeBuild = null;
        this._treeBuildBindGroups = null;
        this._barnesHutEnabled = false;

        // Phase 3: Tree force (BH tree walk)
        this._treeForcePipeline = null;
        this._treeForceBindGroups = null;

        // Phase 3: Collision detection/resolution
        this._collisionPipelines = null;
        this._collisionBindGroups = null;
        this._mergeResultsPending = false;
        this._pendingMergeEvents = [];

        // Phase 3: Dead particle GC
        this._deadGCPipeline = null;
        this._deadGCBindGroup = null;

        // Phase 4: Advanced physics pipelines
        this._phase4 = null;
        this._phase4BindGroups = {};

        // Phase 4: History stride counter
        this._histStride = 0;
        this._frameCount = 0;

        // Phase 4: Toggle state for advanced passes
        this._relativityEnabled = false;
        this._onePNEnabled = false;
        this._radiationEnabled = false;
        this._yukawaEnabled = false;
        this._bosonInterEnabled = false;

        // Toggle state
        this._toggles0 = 0;
        this._toggles1 = 0;
        this._gravityEnabled = false;
        this._coulombEnabled = false;
        this._magneticEnabled = false;
        this._gravitomagEnabled = false;
        this._spinOrbitEnabled = false;
        this._blackHoleEnabled = false;
        this._yukawaCoupling = 14;
        this._yukawaMu = 0.15;
        this._higgsMass = 0.5;
        this._axionMass = 0.05;
        this._extGravity = 0;
        this._extGravityAngle = 0;
        this._extElectric = 0;
        this._extElectricAngle = 0;
        this._extBz = 0;
        this._bounceFriction = 0.4;
        this._collisionMode = 0;
        this._axionCoupling = 0.05;
        this._higgsCoupling = 1.0;

        // Phase 5: Scalar field buffers (lazy-allocated on first toggle-on)
        this._higgsBuffers = null;
        this._axionBuffers = null;
        this._atomicGrid = null;
        this._heatmapBuffers = null;
        this._excitationBuffers = null;
        this._disintBuffers = null;
        this._pairProdBuffers = null;
        this._higgsEnabled = false;
        this._axionEnabled = false;
        this._fieldResolution = FIELD_GRID_RES; // default: 64, configurable to 128/256
        this._gravityEnabled = false;
        this._expansionEnabled = false;
        this._disintegrationEnabled = false;
        this._hubbleParam = 0.001;
        this._heatmapEnabled = false;
        this._heatmapMode = 'all';
        this._heatmapFrame = 0;

        // Phase 5: Pipelines (lazy-initialized)
        this._fieldDeposit = null;
        this._fieldEvolve = null;
        this._fieldForces = null;
        this._fieldParticleGrav = null;
        this._fieldSelfGrav = null;
        this._fftPipelines = null;
        this._fftParamsBuffer = null;
        this._fftBGs = {};
        this._greenHatUploaded = {};
        this._fieldExcitation = null;
        this._heatmapPipelines = null;
        this._expansionPipeline = null;
        this._disintPipeline = null;
        this._pairProdPipeline = null;

        // Phase 5: Bind groups (lazy-created per field)
        this._fieldDepositBGs = {};
        this._fieldEvolveBGs = {};
        this._fieldGradBGs = {};
        this._fieldForcesBGs = null;
        this._fieldParticleGravBGs = {};
        this._fieldParticleGravUniform = null;
        this._fieldSelfGravBGs = {};
        this._fieldExcitationBGs = {};
        this._heatmapBGs = null;
        this._heatmapBlurBGs = {};
        this._expansionBG = null;
        this._disintBGs = null;
        this._pairProdBGs = null;

        // Phase 5: FieldUniforms buffer (shared)
        this._fieldUniformBuffer = null;
        this._heatmapUniformBuffer = null;
        this._expansionUniformBuffer = null;
        this._disintUniformBuffer = null;
        this._pairProdUniformBuffer = null;

        // (sgInvR table removed — computed inline in field-selfgrav.wgsl)

        // Update colors pipeline
        this._updateColorsPipeline = null;
        this._updateColorsBindGroup = null;
        this._colorUniformBuffer = null;

        // Trail recording pipeline + buffers (lazy-allocated)
        this._trailRecordPipeline = null;
        this._trailRecordBindGroup = null;
        this._trailBuffers = null;
        this._trailsEnabled = true; // default on (matches CPU)

        // Hit test pipeline + buffers
        this._hitTestPipeline = null;
        this._hitTestBindGroup = null;
        this._hitUniformBuffer = null;
        this._hitResultBuffer = null;
        this._hitResultStaging = null;
        this._hitPending = false;
        this._hitResultReady = false;
        this._hitStagingI32 = null;
        this._hitStagingF32 = null;

        // Stats readback (compute-stats.wgsl)
        this._statsPipeline = null;
        this._statsBindGroup0 = null;
        this._statsBindGroup1 = null;
        this._statsGroup0Layout = null;
        this._statsGroup1Layout = null;
        this._statsUniformBuffer = null;
        this._statsDummyBuffer = null;
        this._statsPending = false;
        this._statsResultReady = false;
        this._statsData = null; // Float32Array(128) from readback
        this._statsStagingFlip = false; // double-buffer flip
        this._statsFieldHasHiggs = false;
        this._statsFieldHasAxion = false;

        // Free slot management: CPU-side mirror of GPU free stack
        this._cpuFreeSlots = [];
        this._freeTopPending = false;

        // Adaptive substepping state
        this._maxAccel = 0;
        this._maxAccelPending = false;

        this._ready = false;
    }

    /** Load WGSL shaders and create compute pipelines. Must be called before update(). */
    async init() {
        const wgslConstants = buildWGSLConstants();
        // Fetch shared includes + common.wgsl
        const [sharedStructs, sharedTopo, sharedRng, commonSrc, boundaryWGSL] = await Promise.all([
            fetchShader('shared-structs.wgsl'),
            fetchShader('shared-topology.wgsl'),
            fetchShader('shared-rng.wgsl'),
            fetchShader('common.wgsl'),
            fetchShader('boundary.wgsl'),
        ]);
        const commonWGSL = wgslConstants + '\n' + sharedStructs + '\n' + sharedTopo + '\n' + sharedRng + '\n' + commonSrc;

        // --- Boundary pipeline ---
        const boundaryModule = this.device.createShaderModule({
            label: 'boundary',
            code: commonWGSL + '\n' + boundaryWGSL,
        });

        const boundaryBindGroupLayout = this.device.createBindGroupLayout({
            label: 'boundary',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleAux
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // allForces
            ],
        });

        this._boundaryPipeline = this.device.createComputePipeline({
            label: 'boundary',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [boundaryBindGroupLayout] }),
            compute: { module: boundaryModule, entryPoint: 'main' },
        });

        this._boundaryBindGroup = this.device.createBindGroup({
            label: 'boundary',
            layout: boundaryBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.buffers.particleState } },
                { binding: 2, resource: { buffer: this.buffers.particleAux } },
                { binding: 3, resource: { buffer: this.buffers.allForces } },
            ],
        });

        // --- Phase 2 pipelines ---
        this._phase2 = await createPhase2Pipelines(this.device, wgslConstants);
        this._createPhase2BindGroups();

        // --- Phase 3: Ghost generation pipeline ---
        const ghostGen = await createGhostGenPipeline(this.device, wgslConstants);
        this._ghostGenPipeline = ghostGen.pipeline;
        this._createGhostGenBindGroups(ghostGen.bindGroupLayouts);

        // --- Phase 3: Tree build pipelines ---
        this._treeBuild = await createTreeBuildPipelines(this.device, wgslConstants);
        this._createTreeBuildBindGroups(this._treeBuild.bindGroupLayouts);

        // --- Phase 3: Tree force pipeline ---
        const treeForce = await createTreeForcePipeline(this.device, wgslConstants);
        this._treeForcePipeline = treeForce.pipeline;
        this._createTreeForceBindGroups(treeForce.bindGroupLayouts);

        // --- Phase 3: Collision detection/resolution pipelines ---
        this._collisionPipelines = await createCollisionPipelines(this.device, wgslConstants);
        this._createCollisionBindGroups(this._collisionPipelines.bindGroupLayouts);

        // --- Phase 3: Dead particle GC pipeline ---
        const deadGC = await createDeadGCPipeline(this.device, wgslConstants);
        this._deadGCPipeline = deadGC.pipeline;
        this._createDeadGCBindGroup(deadGC.bindGroupLayouts);

        // --- Phase 4: Advanced physics pipelines ---
        this._phase4 = await createPhase4Pipelines(this.device, wgslConstants);
        this._createPhase4BindGroups();

        // --- Update colors compute pipeline ---
        {
            const uc = await createUpdateColorsPipeline(this.device, wgslConstants);
            this._updateColorsPipeline = uc.pipeline;
            this._colorUniformBuffer = this.device.createBuffer({
                label: 'colorUniforms', size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this._updateColorsBindGroup = this.device.createBindGroup({
                label: 'updateColors',
                layout: uc.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this._colorUniformBuffer } },
                    { binding: 1, resource: { buffer: this.buffers.particleState } },
                    { binding: 2, resource: { buffer: this.buffers.color } },
                ],
            });
        }

        // --- Trail recording compute pipeline ---
        {
            const tr = await createTrailRecordPipeline(this.device, wgslConstants);
            this._trailRecordPipeline = tr.pipeline;
            this._trailRecordLayout = tr.bindGroupLayout;
        }

        // --- Hit test compute pipeline ---
        {
            const ht = await createHitTestPipeline(this.device, wgslConstants);
            this._hitTestPipeline = ht.pipeline;
            this._hitUniformBuffer = this.device.createBuffer({
                label: 'hitUniforms', size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            // 12 u32s = 48 bytes: index + mass/charge/radius/vel/angVel/pos + reserved
            this._hitResultBuffer = this.device.createBuffer({
                label: 'hitResult', size: 48,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            this._hitResultStaging = this.device.createBuffer({
                label: 'hitResultStaging', size: 48,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
            this._hitTestBindGroup = this.device.createBindGroup({
                label: 'hitTest',
                layout: ht.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this._hitUniformBuffer } },
                    { binding: 1, resource: { buffer: this.buffers.qtNodeBuffer } },
                    { binding: 2, resource: { buffer: this.buffers.particleState } },
                    { binding: 3, resource: { buffer: this.buffers.particleAux } },
                    { binding: 4, resource: { buffer: this._hitResultBuffer } },
                    { binding: 5, resource: { buffer: this.buffers.derived } },
                ],
            });
        }

        // --- Compute stats pipeline ---
        {
            const cs = await createComputeStatsPipeline(this.device, wgslConstants);
            this._statsPipeline = cs.pipeline;
            this._statsGroup0Layout = cs.group0Layout;
            this._statsGroup1Layout = cs.group1Layout;
            this._statsUniformBuffer = this.device.createBuffer({
                label: 'statsUniforms', size: 48, // StatsUniforms struct
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            // Dummy 4-byte buffer for inactive field bindings
            this._statsDummyBuffer = this.device.createBuffer({
                label: 'statsDummy', size: 4,
                usage: GPUBufferUsage.STORAGE,
            });
            this._statsBindGroup0 = this.device.createBindGroup({
                label: 'computeStats_g0',
                layout: cs.group0Layout,
                entries: [
                    { binding: 0, resource: { buffer: this._statsUniformBuffer } },
                    { binding: 1, resource: { buffer: this.buffers.particleState } },
                    { binding: 2, resource: { buffer: this.buffers.derived } },
                    { binding: 3, resource: { buffer: this.buffers.allForces } },
                    { binding: 4, resource: { buffer: this.buffers.statsBuffer } },
                    { binding: 5, resource: { buffer: this.buffers.axYukMod } },
                ],
            });
            // Field bind group rebuilt when fields toggle on/off
            this._statsBindGroup1 = null;
            this._rebuildStatsFieldBindGroup();
        }

        this._ready = true;
    }

    _createPhase2BindGroups() {
        const b = this.buffers;
        const p2 = this._phase2;

        // Helper to create a bind group from layout + buffer list
        const bg = (label, layout, buffers) =>
            this.device.createBindGroup({
                label,
                layout,
                entries: buffers.map((buf, i) => ({ binding: i, resource: { buffer: buf } })),
            });

        // cacheDerived: uniforms + particleState (packed) + derived + particleAux + axYukMod
        this._bg_cacheDerived = bg('cacheDerived', p2.cacheDerived.bindGroupLayouts[0],
            [this.uniformBuffer, b.particleState, b.derived, b.particleAux, b.axYukMod]);

        // pairForce: 4 bind groups (packed structs)
        // Group 3 (history) requires lazy creation — use dummy buffers until history allocated
        this._bg_pairForce0 = bg('pairForce_g0', p2.pairForce.bindGroupLayouts[0],
            [this.uniformBuffer]);
        this._bg_pairForce1 = bg('pairForce_g1', p2.pairForce.bindGroupLayouts[1],
            [b.particleState, b.derived, b.axYukMod, b.particleAux]);
        this._bg_pairForce2 = bg('pairForce_g2', p2.pairForce.bindGroupLayouts[2],
            [b.allForces, b.maxAccelBuffer]);
        // Group 3: history buffers — created lazily in _ensurePairForceHistoryBG()
        this._bg_pairForce3 = null;

        // externalFields (packed particleState + allForces)
        this._bg_extFields = bg('extFields', p2.externalFields.bindGroupLayouts[0],
            [this.uniformBuffer, b.particleState, b.allForces]);

        // borisFused (halfKick + rotate + halfKick in one pass)
        this._bg_borisFused = bg('borisFused', p2.borisFused.bindGroupLayouts[0],
            [this.uniformBuffer, b.particleState, b.allForces]);

        // borisDrift (writes vel to derived, reads/writes particleState + allForces for display force reconstruction)
        this._bg_drift = bg('drift', p2.borisDrift.bindGroupLayouts[0],
            [this.uniformBuffer, b.particleState, b.derived, b.allForces]);

        // spinOrbit (packed particleState, derived, allForces)
        this._bg_spinOrbit = bg('spinOrbit', p2.spinOrbit.bindGroupLayouts[0],
            [this.uniformBuffer, b.particleState, b.derived, b.allForces]);

        // applyTorques (reads from particleState, allForces, writes derived)
        this._bg_torques = bg('torques', p2.applyTorques.bindGroupLayouts[0],
            [this.uniformBuffer, b.particleState, b.allForces, b.derived]);

        // saveF1pn (save 1PN forces before Boris kick for VV correction)
        this._bg_saveF1pn = bg('saveF1pn', p2.saveF1pn.bindGroupLayouts[0],
            [this.uniformBuffer, b.allForces, b.f1pnOld, b.particleState]);
    }

    /**
     * Create bind groups for ghost generation pipeline.
     * Ghost output arrays are views into the same SoA buffers at offset = aliveCount.
     * Since ghost writes use atomicAdd for slot allocation, the output buffers are the
     * SAME buffers as input but the shader writes starting at the ghost slot offset.
     */
    _createGhostGenBindGroups(layouts) {
        const b = this.buffers;
        const bg = (label, layout, entries) =>
            this.device.createBindGroup({
                label,
                layout,
                entries: entries.map((buf, i) => ({ binding: i, resource: { buffer: buf } })),
            });

        // Group 0: read-only particle state (packed struct)
        const group0 = bg('ghostGen_g0', layouts[0],
            [b.particleState]);

        // Group 1: ghost output + derived inputs/outputs + aux
        const group1 = bg('ghostGen_g1', layouts[1],
            [b.ghostState, b.ghostAux, b.derived, b.ghostDerived, b.particleAux]);

        // Group 2: ghostCounter + uniforms + ghostOriginalIdx
        const group2 = bg('ghostGen_g2', layouts[2],
            [b.ghostCounter, this.uniformBuffer, b.ghostOriginalIdx]);

        this._ghostGenBindGroups = [group0, group1, group2];
    }

    /**
     * Dispatch ghost generation before tree build (Phase 3).
     * Only runs when boundary is LOOP (periodic).
     * Resets ghost counter, dispatches shader, then copies ghost data into main SoA arrays.
     */
    _dispatchGhostGen(encoder) {
        if (this.boundaryMode !== BOUND_LOOP) {
            this._ghostCount = 0;
            return;
        }

        // Reset ghost counter to 0
        encoder.clearBuffer(this.buffers.ghostCounter, 0, 4);

        const pass = encoder.beginComputePass({ label: 'ghostGen' });
        pass.setPipeline(this._ghostGenPipeline);
        pass.setBindGroup(0, this._ghostGenBindGroups[0]);
        pass.setBindGroup(1, this._ghostGenBindGroups[1]);
        pass.setBindGroup(2, this._ghostGenBindGroups[2]);
        pass.dispatchWorkgroups(Math.ceil(this.aliveCount / 64));
        pass.end();

        // Copy ghost data from dedicated buffers into main arrays at offset aliveCount.
        // Uses previous frame's ghost count for copy size (1-frame latency, safe for tree build).
        const ghostCount = this._ghostCount;
        if (ghostCount > 0) {
            const b = this.buffers;
            // Copy packed ParticleState structs (36 bytes per element)
            const stateOffset = this.aliveCount * PARTICLE_STATE_SIZE;
            const stateBytes = ghostCount * PARTICLE_STATE_SIZE;
            encoder.copyBufferToBuffer(b.ghostState, 0, b.particleState, stateOffset, stateBytes);
            // Copy packed ParticleAux structs (20 bytes per element)
            const auxOffset = this.aliveCount * PARTICLE_AUX_SIZE;
            const auxBytes = ghostCount * PARTICLE_AUX_SIZE;
            encoder.copyBufferToBuffer(b.ghostAux, 0, b.particleAux, auxOffset, auxBytes);
            // Copy ParticleDerived structs (32 bytes per element)
            const derivedOffset = this.aliveCount * DERIVED_SIZE;
            const derivedBytes = ghostCount * DERIVED_SIZE;
            encoder.copyBufferToBuffer(b.ghostDerived, 0, b.derived, derivedOffset, derivedBytes);
        }
    }

    /**
     * Create bind groups for tree build pipelines.
     * All 4 entry points share the same bind group layouts.
     */
    _createTreeBuildBindGroups(layouts) {
        const b = this.buffers;

        // Group 0: tree state buffers
        this._treeBuildBG0 = this.device.createBindGroup({
            label: 'treeBuild_g0',
            layout: layouts[0],
            entries: [
                { binding: 0, resource: { buffer: b.qtNodeBuffer } },
                { binding: 1, resource: { buffer: b.qtNodeCounter } },
                { binding: 2, resource: { buffer: b.qtBoundsBuffer } },
                { binding: 3, resource: { buffer: b.qtVisitorFlags } },
            ],
        });

        // Group 1: particle state (packed struct) + derived + aux
        this._treeBuildBG1 = this.device.createBindGroup({
            label: 'treeBuild_g1',
            layout: layouts[1],
            entries: [
                { binding: 0, resource: { buffer: b.particleState } },
                { binding: 1, resource: { buffer: b.derived } },
                { binding: 2, resource: { buffer: b.particleAux } },
            ],
        });

        // Group 2: uniforms
        this._treeBuildBG2 = this.device.createBindGroup({
            label: 'treeBuild_g2',
            layout: layouts[2],
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
            ],
        });
    }

    /**
     * Create bind groups for tree force (BH walk) pipeline.
     * Group 0: nodes (read-only) + uniforms
     * Group 1: particle SoA (15 read-only) + ghostOriginalIdx
     * Group 2: force accumulators (5 read-write)
     */
    _createTreeForceBindGroups(layouts) {
        const b = this.buffers;

        // Group 0: tree nodes + uniforms
        this._treeForceGroup0 = this.device.createBindGroup({
            label: 'treeForce_g0',
            layout: layouts[0],
            entries: [
                { binding: 0, resource: { buffer: b.qtNodeBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } },
            ],
        });

        // Group 1: packed particle structs + derived + axYukMod
        this._treeForceGroup1 = this.device.createBindGroup({
            label: 'treeForce_g1',
            layout: layouts[1],
            entries: [
                { binding: 0, resource: { buffer: b.particleState } },
                { binding: 1, resource: { buffer: b.particleAux } },
                { binding: 2, resource: { buffer: b.derived } },
                { binding: 3, resource: { buffer: b.axYukMod } },
                { binding: 4, resource: { buffer: b.ghostOriginalIdx } },
            ],
        });

        // Group 2: allForces + maxAccel (radiationState removed — jerk now in AllForces)
        this._treeForceGroup2 = this.device.createBindGroup({
            label: 'treeForce_g2',
            layout: layouts[2],
            entries: [
                { binding: 0, resource: { buffer: b.allForces } },
                { binding: 1, resource: { buffer: b.maxAccelBuffer } },
            ],
        });

        // Group 3: history buffers — created lazily in _ensureTreeForceHistoryBG()
        this._treeForceGroup3 = null;
        this._treeForceLayouts = layouts;
    }

    /**
     * Ensure pair-force history bind group exists (lazy allocation).
     * Uses dummy buffers when history not yet allocated.
     */
    _ensurePairForceHistoryBG() {
        if (this._bg_pairForce3) return;
        const b = this.buffers;
        const p2 = this._phase2;
        if (b.historyAllocated) {
            this._bg_pairForce3 = this.device.createBindGroup({
                label: 'pairForce_g3_history',
                layout: p2.pairForce.bindGroupLayouts[3],
                entries: [
                    { binding: 0, resource: { buffer: b.histData } },
                    { binding: 1, resource: { buffer: b.histMeta } },
                ],
            });
        } else {
            this._bg_pairForce3 = this.device.createBindGroup({
                label: 'pairForce_g3_dummy',
                layout: p2.pairForce.bindGroupLayouts[3],
                entries: [
                    { binding: 0, resource: { buffer: this._dummyHistBuf } },
                    { binding: 1, resource: { buffer: this._dummyHistBuf } },
                ],
            });
            this._bg_pairForce3_isDummy = true;
        }
    }

    /**
     * Ensure tree-force history bind group exists (lazy allocation).
     * Uses shared dummy buffer when history not yet allocated.
     */
    _ensureTreeForceHistoryBG() {
        if (this._treeForceGroup3) return;
        const b = this.buffers;
        const layouts = this._treeForceLayouts;
        if (b.historyAllocated) {
            this._treeForceGroup3 = this.device.createBindGroup({
                label: 'treeForce_g3_history',
                layout: layouts[3],
                entries: [
                    { binding: 0, resource: { buffer: b.histData } },
                    { binding: 1, resource: { buffer: b.histMeta } },
                ],
            });
        } else {
            this._treeForceGroup3 = this.device.createBindGroup({
                label: 'treeForce_g3_dummy',
                layout: layouts[3],
                entries: [
                    { binding: 0, resource: { buffer: this._dummyHistBuf } },
                    { binding: 1, resource: { buffer: this._dummyHistBuf } },
                ],
            });
            this._treeForceGroup3_isDummy = true;
        }
    }

    /**
     * Upgrade dummy history bind groups to real ones when history buffers become available.
     * Called from _ensureHistoryBindGroups.
     */
    _upgradeForceHistoryBGs() {
        const b = this.buffers;
        if (!b.historyAllocated) return;
        if (this._bg_pairForce3_isDummy) {
            const p2 = this._phase2;
            this._bg_pairForce3 = this.device.createBindGroup({
                label: 'pairForce_g3_history',
                layout: p2.pairForce.bindGroupLayouts[3],
                entries: [
                    { binding: 0, resource: { buffer: b.histData } },
                    { binding: 1, resource: { buffer: b.histMeta } },
                ],
            });
            this._bg_pairForce3_isDummy = false;
        }
        if (this._treeForceGroup3_isDummy && this._treeForceLayouts) {
            this._treeForceGroup3 = this.device.createBindGroup({
                label: 'treeForce_g3_history',
                layout: this._treeForceLayouts[3],
                entries: [
                    { binding: 0, resource: { buffer: b.histData } },
                    { binding: 1, resource: { buffer: b.histMeta } },
                ],
            });
            this._treeForceGroup3_isDummy = false;
        }
        if (this._onePNG3_isDummy && this._phase4) {
            const p4 = this._phase4;
            this._phase4BindGroups.onePNG3 = this.device.createBindGroup({
                label: 'onePN_g3_history',
                layout: p4.compute1PN.bindGroupLayouts[3],
                entries: [
                    { binding: 0, resource: { buffer: b.histData } },
                    { binding: 1, resource: { buffer: b.histMeta } },
                ],
            });
            this._onePNG3_isDummy = false;
        }
        if (this._onePNTreeG3_isDummy && this._phase4) {
            const p4 = this._phase4;
            this._phase4BindGroups.onePNTreeG3 = this.device.createBindGroup({
                label: 'onePNTree_g3_history',
                layout: p4.compute1PNTree.bindGroupLayouts[3],
                entries: [
                    { binding: 0, resource: { buffer: b.histData } },
                    { binding: 1, resource: { buffer: b.histMeta } },
                    { binding: 2, resource: { buffer: b.qtNodeBuffer } },
                ],
            });
            this._onePNTreeG3_isDummy = false;
        }
    }

    /**
     * Create bind groups for collision detection/resolution pipelines.
     * Both detectCollisions and resolveCollisions share the same layout.
     */
    _createCollisionBindGroups(layouts) {
        const b = this.buffers;

        // Group 0: tree nodes (read-only) + uniforms
        this._collisionBG0 = this.device.createBindGroup({
            label: 'collision_g0',
            layout: layouts[0],
            entries: [
                { binding: 0, resource: { buffer: b.qtNodeBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } },
            ],
        });

        // Group 1: packed particle structs + ghost mapping + allForces (contact torque)
        this._collisionBG1 = this.device.createBindGroup({
            label: 'collision_g1',
            layout: layouts[1],
            entries: [
                { binding: 0, resource: { buffer: b.particleState } },
                { binding: 1, resource: { buffer: b.particleAux } },
                { binding: 2, resource: { buffer: b.ghostOriginalIdx } },
                { binding: 3, resource: { buffer: b.allForces } },
            ],
        });

        // Group 2: collision pairs + counters + merge results
        this._collisionBG2 = this.device.createBindGroup({
            label: 'collision_g2',
            layout: layouts[2],
            entries: [
                { binding: 0, resource: { buffer: b.collisionPairBuffer } },
                { binding: 1, resource: { buffer: b.collisionPairCounter } },
                { binding: 2, resource: { buffer: b.mergeResultBuffer } },
                { binding: 3, resource: { buffer: b.mergeResultCounter } },
            ],
        });
    }

    /**
     * Create bind group for dead particle GC pipeline.
     */
    _createDeadGCBindGroup(layouts) {
        const b = this.buffers;
        this._deadGCBindGroup = this.device.createBindGroup({
            label: 'deadGC_g0',
            layout: layouts[0],
            entries: [
                { binding: 0, resource: { buffer: b.particleState } },
                { binding: 1, resource: { buffer: b.particleAux } },
                { binding: 2, resource: { buffer: this.uniformBuffer } },
                { binding: 3, resource: { buffer: b.freeStack } },
                { binding: 4, resource: { buffer: b.freeTop } },
            ],
        });
    }

    /**
     * Create bind groups for all Phase 4 pipelines.
     */
    _createPhase4BindGroups() {
        const b = this.buffers;
        const p4 = this._phase4;
        const bg = (label, layout, buffers) =>
            this.device.createBindGroup({
                label,
                layout,
                entries: buffers.map((buf, i) => ({ binding: i, resource: { buffer: buf } })),
            });

        // ── recordHistory ──
        // Bind groups created lazily when history buffers are allocated
        this._phase4BindGroups.historyG0Buffers = [
            this.uniformBuffer, b.particleState,
        ];
        this._phase4BindGroups.historyG0 = null; // lazy
        this._phase4BindGroups.historyG1 = null; // lazy

        // ── 1PN (compute1PN + vvKick1PN share bind groups) ──
        this._phase4BindGroups.onePNG0 = bg('onePN_g0', p4.compute1PN.bindGroupLayouts[0],
            [this.uniformBuffer]);
        this._phase4BindGroups.onePNG1 = bg('onePN_g1', p4.compute1PN.bindGroupLayouts[1],
            [b.particleState, b.derived, b.axYukMod]);
        this._phase4BindGroups.onePNG2 = bg('onePN_g2', p4.compute1PN.bindGroupLayouts[2],
            [b.allForces, b.f1pnOld]);
        // Group 3 (history) — dummy until history allocated (pipeline layout requires all groups)
        this._phase4BindGroups.onePNG3 = this.device.createBindGroup({
            label: 'onePN_g3_dummy',
            layout: p4.compute1PN.bindGroupLayouts[3],
            entries: [
                { binding: 0, resource: { buffer: this._dummyHistBuf } },
                { binding: 1, resource: { buffer: this._dummyHistBuf } },
            ],
        });
        this._onePNG3_isDummy = true;

        // ── 1PN Tree Walk (compute1PNTree — shares G0, G2 with pairwise) ──
        this._phase4BindGroups.onePNTreeG1 = bg('onePNTree_g1', p4.compute1PNTree.bindGroupLayouts[1],
            [b.particleState, b.derived, b.axYukMod, b.ghostOriginalIdx]);
        this._phase4BindGroups.onePNTreeG3 = this.device.createBindGroup({
            label: 'onePNTree_g3_dummy',
            layout: p4.compute1PNTree.bindGroupLayouts[3],
            entries: [
                { binding: 0, resource: { buffer: this._dummyHistBuf } },
                { binding: 1, resource: { buffer: this._dummyHistBuf } },
                { binding: 2, resource: { buffer: b.qtNodeBuffer } },
            ],
        });
        this._onePNTreeG3_isDummy = true;

        // ── Radiation (larmorRadiation, hawkingRadiation, pionEmission share bind groups) ──
        this._phase4BindGroups.radG0 = bg('radiation_g0', p4.larmorRadiation.bindGroupLayouts[0],
            [this.uniformBuffer]);
        // Group 1: packed particle state + aux + derived + allForces + radiationState + axYukMod
        this._phase4BindGroups.radG1 = bg('radiation_g1', p4.larmorRadiation.bindGroupLayouts[1],
            [b.particleState, b.particleAux, b.derived, b.allForces, b.radiationState, b.axYukMod]);
        // Group 2: photon pool (packed) + phCount
        this._phase4BindGroups.radG2 = bg('radiation_g2', p4.larmorRadiation.bindGroupLayouts[2],
            [b.photonPool, b.phCount]);
        // Group 3: pion pool (packed) + piCount
        this._phase4BindGroups.radG3 = bg('radiation_g3', p4.larmorRadiation.bindGroupLayouts[3],
            [b.pionPool, b.piCount]);

        // ── Quadrupole radiation (quadrupoleCoM, quadrupoleContrib, quadrupoleApply share bind groups) ──
        this._phase4BindGroups.quadG0 = bg('quadrupole_g0', p4.quadrupoleCoM.bindGroupLayouts[0],
            [this.uniformBuffer]);
        this._phase4BindGroups.quadG1 = bg('quadrupole_g1', p4.quadrupoleCoM.bindGroupLayouts[1],
            [b.particleState, b.particleAux, b.derived, b.allForces, b.radiationState]);
        this._phase4BindGroups.quadG2 = bg('quadrupole_g2', p4.quadrupoleCoM.bindGroupLayouts[2],
            [b.photonPool, b.phCount]);
        this._phase4BindGroups.quadG3 = bg('quadrupole_g3', p4.quadrupoleCoM.bindGroupLayouts[3],
            [b.quadReductionBuf]);

        // ── Bosons (updatePhotons, updatePions, absorbPhotons, absorbPions, decayPions) ──
        this._phase4BindGroups.bosG0 = bg('bosons_g0', p4.updatePhotons.bindGroupLayouts[0],
            [this.uniformBuffer, b.poolMgmt]);
        this._phase4BindGroups.bosG1 = bg('bosons_g1', p4.updatePhotons.bindGroupLayouts[1],
            [b.particleState, b.particleAux]);
        this._phase4BindGroups.bosG2 = bg('bosons_g2', p4.updatePhotons.bindGroupLayouts[2],
            [b.photonPool, b.phCount]);
        this._phase4BindGroups.bosG3 = bg('bosons_g3', p4.updatePhotons.bindGroupLayouts[3],
            [b.pionPool, b.piCount]);

        // ── Boson Tree Walk (updatePhotonsTree, updatePionsTree, absorbPhotonsTree, absorbPionsTree) ──
        // Group 1 adds tree nodes buffer at binding 2; groups 0,2,3 shared with non-tree variants.
        this._phase4BindGroups.bosTreeG1 = bg('bosonsTree_g1', p4.updatePhotonsTree.bindGroupLayouts[1],
            [b.particleState, b.particleAux, b.qtNodeBuffer]);

        // ── Boson Tree (insertBosonsIntoTree, computeBosonAggregates, computeBosonGravity, applyBosonBosonGravity) ──
        // Group 0: uniforms + tree nodes (atomic) + counter + visitor flags
        this._phase4BindGroups.btG0 = bg('bosonTree_g0', p4.insertBosonsIntoTree.bindGroupLayouts[0],
            [this.uniformBuffer, b.bosonTreeNodes, b.bosonTreeCounter, b.bosonVisitorFlags]);
        this._phase4BindGroups.btG1 = bg('bosonTree_g1', p4.insertBosonsIntoTree.bindGroupLayouts[1],
            [b.photonPool, b.phCount]);
        this._phase4BindGroups.btG2 = bg('bosonTree_g2', p4.insertBosonsIntoTree.bindGroupLayouts[2],
            [b.pionPool, b.piCount]);
        this._phase4BindGroups.btG3 = bg('bosonTree_g3', p4.insertBosonsIntoTree.bindGroupLayouts[3],
            [b.particleState, b.allForces]);
    }

    /**
     * Ensure history bind groups exist (lazy allocation).
     * Called when relativity is first enabled.
     */
    _ensureHistoryBindGroups() {
        if (this._phase4BindGroups.historyG0) return;
        if (!this.buffers.historyAllocated) {
            this.buffers.allocateHistoryBuffers(this.device);
        }
        const b = this.buffers;
        const p4 = this._phase4;
        const bg = (label, layout, buffers) =>
            this.device.createBindGroup({
                label, layout,
                entries: buffers.map((buf, i) => ({ binding: i, resource: { buffer: buf } })),
            });

        this._phase4BindGroups.historyG0 = bg('history_g0', p4.recordHistory.bindGroupLayouts[0],
            [this.uniformBuffer, b.particleState]);
        this._phase4BindGroups.historyG1 = bg('history_g1', p4.recordHistory.bindGroupLayouts[1],
            [b.histData, b.histMeta]);

        // Upgrade dummy force history bind groups (pair-force, tree-force, 1PN) to real ones
        this._upgradeForceHistoryBGs();
    }

    /**
     * Dispatch radiation reaction passes (Phase 4, Pass 11).
     * Runs after Boris half-kick 2 and torques, before drift.
     */
    _dispatchRadiation(encoder) {
        if (!this._radiationEnabled) return;

        const workgroups = Math.ceil(this.aliveCount / 64);
        const bgs = this._phase4BindGroups;
        const p4 = this._phase4;

        // Larmor radiation (requires Coulomb + Radiation)
        const passLarmor = encoder.beginComputePass({ label: 'larmorRadiation' });
        passLarmor.setPipeline(p4.larmorRadiation.pipeline);
        passLarmor.setBindGroup(0, bgs.radG0);
        passLarmor.setBindGroup(1, bgs.radG1);
        passLarmor.setBindGroup(2, bgs.radG2);
        passLarmor.setBindGroup(3, bgs.radG3);
        passLarmor.dispatchWorkgroups(workgroups);
        passLarmor.end();

        // Hawking radiation (requires Black Hole + Radiation)
        if (this._blackHoleEnabled) {
            const passHawking = encoder.beginComputePass({ label: 'hawkingRadiation' });
            passHawking.setPipeline(p4.hawkingRadiation.pipeline);
            passHawking.setBindGroup(0, bgs.radG0);
            passHawking.setBindGroup(1, bgs.radG1);
            passHawking.setBindGroup(2, bgs.radG2);
            passHawking.setBindGroup(3, bgs.radG3);
            passHawking.dispatchWorkgroups(workgroups);
            passHawking.end();
        }

        // Pion emission (requires Yukawa + Radiation)
        if (this._yukawaEnabled) {
            const passPion = encoder.beginComputePass({ label: 'pionEmission' });
            passPion.setPipeline(p4.pionEmission.pipeline);
            passPion.setBindGroup(0, bgs.radG0);
            passPion.setBindGroup(1, bgs.radG1);
            passPion.setBindGroup(2, bgs.radG2);
            passPion.setBindGroup(3, bgs.radG3);
            passPion.dispatchWorkgroups(workgroups);
            passPion.end();
        }
    }

    /**
     * Dispatch quadrupole radiation passes (once per frame, after all substeps).
     * Requires Radiation + (Gravity or Coulomb) + at least 2 particles.
     * Three dispatches: CoM reduction → d³ contribution reduction → apply + emit.
     */
    _dispatchQuadrupole(encoder) {
        if (!this._radiationEnabled) return;
        if (this.aliveCount < 2) return;
        // Requires gravity or coulomb (same guard as CPU)
        const t0 = this._toggles0;
        if (!(t0 & GRAVITY_BIT) && !(t0 & COULOMB_BIT)) return; // neither gravity nor coulomb

        const workgroups = Math.ceil(this.aliveCount / 64);
        const bgs = this._phase4BindGroups;
        const p4 = this._phase4;

        function setBindGroups(pass) {
            pass.setBindGroup(0, bgs.quadG0);
            pass.setBindGroup(1, bgs.quadG1);
            pass.setBindGroup(2, bgs.quadG2);
            pass.setBindGroup(3, bgs.quadG3);
        }

        // Pass 1: Reduce CoM + totalKE
        const p1 = encoder.beginComputePass({ label: 'quadrupoleCoM' });
        p1.setPipeline(p4.quadrupoleCoM.pipeline);
        setBindGroups(p1);
        p1.dispatchWorkgroups(workgroups);
        p1.end();

        // Pass 2: Compute d³I/d³Q contributions + reduce
        const p2 = encoder.beginComputePass({ label: 'quadrupoleContrib' });
        p2.setPipeline(p4.quadrupoleContrib.pipeline);
        setBindGroups(p2);
        p2.dispatchWorkgroups(workgroups);
        p2.end();

        // Pass 3: Apply drag + accumulate + emit photons/gravitons
        const p3 = encoder.beginComputePass({ label: 'quadrupoleApply' });
        p3.setPipeline(p4.quadrupoleApply.pipeline);
        setBindGroups(p3);
        p3.dispatchWorkgroups(workgroups);
        p3.end();
    }

    /**
     * Dispatch 1PN velocity-Verlet correction (Phase 4, Pass 14).
     * After drift: rebuild tree if BH on, recompute 1PN via tree walk, apply VV kick.
     * Falls back to O(N²) pairwise when Barnes-Hut is off.
     */
    _dispatch1PNVV(encoder) {
        if (!this._onePNEnabled) return;

        const workgroups = Math.ceil(this.aliveCount / 64);
        const bgs = this._phase4BindGroups;
        const p4 = this._phase4;

        // Step 1: Recompute 1PN forces at post-drift positions
        if (this._barnesHutEnabled) {
            // Rebuild tree at post-drift positions for VV correction
            this._dispatchGhostGen(encoder);
            this._dispatchTreeBuild(encoder);

            this._ensureTreeForceHistoryBG(); // upgrades onePNTreeG3 from dummy
            const passCompute = encoder.beginComputePass({ label: 'compute1PNTree' });
            passCompute.setPipeline(p4.compute1PNTree.pipeline);
            passCompute.setBindGroup(0, bgs.onePNG0);
            passCompute.setBindGroup(1, bgs.onePNTreeG1);
            passCompute.setBindGroup(2, bgs.onePNG2);
            passCompute.setBindGroup(3, bgs.onePNTreeG3);
            passCompute.dispatchWorkgroups(workgroups);
            passCompute.end();
        } else {
            const passCompute = encoder.beginComputePass({ label: 'compute1PN' });
            passCompute.setPipeline(p4.compute1PN.pipeline);
            passCompute.setBindGroup(0, bgs.onePNG0);
            passCompute.setBindGroup(1, bgs.onePNG1);
            passCompute.setBindGroup(2, bgs.onePNG2);
            passCompute.setBindGroup(3, bgs.onePNG3);
            passCompute.dispatchWorkgroups(workgroups);
            passCompute.end();
        }

        // Step 2: Apply VV correction kick
        const passKick = encoder.beginComputePass({ label: 'vvKick1PN' });
        passKick.setPipeline(p4.vvKick1PN.pipeline);
        passKick.setBindGroup(0, bgs.onePNG0);
        passKick.setBindGroup(1, bgs.onePNG1);
        passKick.setBindGroup(2, bgs.onePNG2);
        passKick.setBindGroup(3, bgs.onePNG3);
        passKick.dispatchWorkgroups(workgroups);
        passKick.end();
    }

    /**
     * Dispatch boson update passes (Phase 4, Pass 21).
     * Photon/pion drift, absorption, pion decay.
     * Skipped entirely when radiation and Yukawa are both off (no bosons can exist).
     */
    _dispatchBosonUpdate(encoder) {
        if (!this._radiationEnabled && !this._yukawaEnabled) return;
        const p4 = this._phase4;
        const bgs = this._phase4BindGroups;
        const useBHTree = this._barnesHutEnabled;

        // Select tree or pairwise pipelines for lensing + absorption
        const phUpdatePipeline = useBHTree ? p4.updatePhotonsTree.pipeline : p4.updatePhotons.pipeline;
        const piUpdatePipeline = useBHTree ? p4.updatePionsTree.pipeline : p4.updatePions.pipeline;
        const phAbsorbPipeline = useBHTree ? p4.absorbPhotonsTree.pipeline : p4.absorbPhotons.pipeline;
        const piAbsorbPipeline = useBHTree ? p4.absorbPionsTree.pipeline : p4.absorbPions.pipeline;
        const g1 = useBHTree ? bgs.bosTreeG1 : bgs.bosG1;

        // updatePhotons: drift + lensing
        const phWG = Math.ceil(GPU_MAX_PHOTONS / 64);
        const passPhotons = encoder.beginComputePass({ label: 'updatePhotons' });
        passPhotons.setPipeline(phUpdatePipeline);
        passPhotons.setBindGroup(0, bgs.bosG0);
        passPhotons.setBindGroup(1, g1);
        passPhotons.setBindGroup(2, bgs.bosG2);
        passPhotons.setBindGroup(3, bgs.bosG3);
        passPhotons.dispatchWorkgroups(phWG);
        passPhotons.end();

        // updatePions: drift with proper velocity
        const piWG = Math.ceil(GPU_MAX_PIONS / 64);
        const passPions = encoder.beginComputePass({ label: 'updatePions' });
        passPions.setPipeline(piUpdatePipeline);
        passPions.setBindGroup(0, bgs.bosG0);
        passPions.setBindGroup(1, g1);
        passPions.setBindGroup(2, bgs.bosG2);
        passPions.setBindGroup(3, bgs.bosG3);
        passPions.dispatchWorkgroups(piWG);
        passPions.end();

        // absorbPhotons
        const passAbsorbPh = encoder.beginComputePass({ label: 'absorbPhotons' });
        passAbsorbPh.setPipeline(phAbsorbPipeline);
        passAbsorbPh.setBindGroup(0, bgs.bosG0);
        passAbsorbPh.setBindGroup(1, g1);
        passAbsorbPh.setBindGroup(2, bgs.bosG2);
        passAbsorbPh.setBindGroup(3, bgs.bosG3);
        passAbsorbPh.dispatchWorkgroups(phWG);
        passAbsorbPh.end();

        // absorbPions
        const passAbsorbPi = encoder.beginComputePass({ label: 'absorbPions' });
        passAbsorbPi.setPipeline(piAbsorbPipeline);
        passAbsorbPi.setBindGroup(0, bgs.bosG0);
        passAbsorbPi.setBindGroup(1, g1);
        passAbsorbPi.setBindGroup(2, bgs.bosG2);
        passAbsorbPi.setBindGroup(3, bgs.bosG3);
        passAbsorbPi.dispatchWorkgroups(piWG);
        passAbsorbPi.end();

    }

    /**
     * Dispatch pion decay (once per frame, NOT per substep).
     * Decay probability is calibrated per PHYSICS_DT, matching CPU path.
     */
    _dispatchPionDecay(encoder) {
        if (!this._radiationEnabled && !this._yukawaEnabled) return;
        const p4 = this._phase4;
        const bgs = this._phase4BindGroups;
        const piWG = Math.ceil(GPU_MAX_PIONS / 64);
        const passDecay = encoder.beginComputePass({ label: 'decayPions' });
        passDecay.setPipeline(p4.decayPions.pipeline);
        passDecay.setBindGroup(0, bgs.bosG0);
        passDecay.setBindGroup(1, bgs.bosG1);
        passDecay.setBindGroup(2, bgs.bosG2);
        passDecay.setBindGroup(3, bgs.bosG3);
        passDecay.dispatchWorkgroups(piWG);
        passDecay.end();
    }

    /**
     * Dispatch boson interaction passes (Phase 4).
     * Runs once per frame after all substeps: build boson tree, gravity, Coulomb, annihilation.
     */
    _dispatchBosonInteraction(encoder) {
        if (!this._bosonInterEnabled) return;

        const p4 = this._phase4;
        const bgs = this._phase4BindGroups;
        const b = this.buffers;

        // Reset boson tree node counter to 1 (root = node 0)
        this.device.queue.writeBuffer(b.bosonTreeCounter, 0, _qtNodeCounterData);

        // Clear visitor flags before insertion
        encoder.clearBuffer(b.bosonVisitorFlags);

        // Initialize root node (reuse pre-allocated arrays to avoid GC)
        // Child pointers (words 14-17) = 0xFFFFFFFF (NONE), particleIdx (18) = 0xFFFFFFFF, parent (19) = 0xFFFFFFFF
        _bosonRootData.fill(0);
        _bosonRootF32[0] = 0;              // minX
        _bosonRootF32[1] = 0;              // minY
        _bosonRootF32[2] = this.domainW;   // maxX
        _bosonRootF32[3] = this.domainH;   // maxY
        _bosonRootData[14] = 0xFFFFFFFF;   // NW = NONE
        _bosonRootData[15] = 0xFFFFFFFF;   // NE = NONE
        _bosonRootData[16] = 0xFFFFFFFF;   // SW = NONE
        _bosonRootData[17] = 0xFFFFFFFF;   // SE = NONE
        _bosonRootData[18] = 0xFFFFFFFF;   // particleIdx = NONE
        _bosonRootData[19] = 0xFFFFFFFF;   // parent = NONE (-1)
        this.device.queue.writeBuffer(b.bosonTreeNodes, 0, _bosonRootData);

        const totalBosons = GPU_MAX_PHOTONS + GPU_MAX_PIONS;
        const bosonWG = Math.ceil(totalBosons / 64);

        // insertBosonsIntoTree
        const passInsert = encoder.beginComputePass({ label: 'insertBosonsIntoTree' });
        passInsert.setPipeline(p4.insertBosonsIntoTree.pipeline);
        passInsert.setBindGroup(0, bgs.btG0);
        passInsert.setBindGroup(1, bgs.btG1);
        passInsert.setBindGroup(2, bgs.btG2);
        passInsert.setBindGroup(3, bgs.btG3);
        passInsert.dispatchWorkgroups(bosonWG);
        passInsert.end();

        // computeBosonAggregates (one thread per boson, bottom-up leaf->root walk)
        const passAgg = encoder.beginComputePass({ label: 'computeBosonAggregates' });
        passAgg.setPipeline(p4.computeBosonAggregates.pipeline);
        passAgg.setBindGroup(0, bgs.btG0);
        passAgg.setBindGroup(1, bgs.btG1);
        passAgg.setBindGroup(2, bgs.btG2);
        passAgg.setBindGroup(3, bgs.btG3);
        passAgg.dispatchWorkgroups(bosonWG);
        passAgg.end();

        // computeBosonGravity: particle <- boson gravity (only when gravity enabled)
        if (this._gravityEnabled) {
            const particleWG = Math.ceil(this.aliveCount / 64);
            if (particleWG > 0) {
                const passGrav = encoder.beginComputePass({ label: 'computeBosonGravity' });
                passGrav.setPipeline(p4.computeBosonGravity.pipeline);
                passGrav.setBindGroup(0, bgs.btG0);
                passGrav.setBindGroup(1, bgs.btG1);
                passGrav.setBindGroup(2, bgs.btG2);
                passGrav.setBindGroup(3, bgs.btG3);
                passGrav.dispatchWorkgroups(particleWG);
                passGrav.end();
            }

            // applyBosonBosonGravity: boson <-> boson mutual gravity
            const passBosonBoson = encoder.beginComputePass({ label: 'applyBosonBosonGravity' });
            passBosonBoson.setPipeline(p4.applyBosonBosonGravity.pipeline);
            passBosonBoson.setBindGroup(0, bgs.btG0);
            passBosonBoson.setBindGroup(1, bgs.btG1);
            passBosonBoson.setBindGroup(2, bgs.btG2);
            passBosonBoson.setBindGroup(3, bgs.btG3);
            passBosonBoson.dispatchWorkgroups(bosonWG);
            passBosonBoson.end();
        }

        // applyPionPionCoulomb: pion <-> pion Coulomb via boson tree (only when Coulomb enabled)
        if (this._coulombEnabled) {
            const pionWG = Math.ceil(GPU_MAX_PIONS / 64);
            const passPiCoulomb = encoder.beginComputePass({ label: 'applyPionPionCoulomb' });
            passPiCoulomb.setPipeline(p4.applyPionPionCoulomb.pipeline);
            passPiCoulomb.setBindGroup(0, bgs.btG0);
            passPiCoulomb.setBindGroup(1, bgs.btG1);
            passPiCoulomb.setBindGroup(2, bgs.btG2);
            passPiCoulomb.setBindGroup(3, bgs.btG3);
            passPiCoulomb.dispatchWorkgroups(pionWG);
            passPiCoulomb.end();
        }

        // annihilatePions: π⁺π⁻ → 2 photons
        {
            const pionWG = Math.ceil(GPU_MAX_PIONS / 64);
            const passAnnihilate = encoder.beginComputePass({ label: 'annihilatePions' });
            passAnnihilate.setPipeline(p4.annihilatePions.pipeline);
            passAnnihilate.setBindGroup(0, bgs.btG0);
            passAnnihilate.setBindGroup(1, bgs.btG1);
            passAnnihilate.setBindGroup(2, bgs.btG2);
            passAnnihilate.setBindGroup(3, bgs.btG3);
            passAnnihilate.dispatchWorkgroups(pionWG);
            passAnnihilate.end();
        }
    }

    /**
     * Dispatch signal delay history recording (Phase 4, Pass 23).
     * Runs once every HISTORY_STRIDE frames when relativity is enabled.
     */
    _dispatchRecordHistory(encoder) {
        if (!this._relativityEnabled) return;
        if (!this._phase4BindGroups.historyG0) return;

        const workgroups = Math.ceil(this.aliveCount / 64);
        const p4 = this._phase4;
        const bgs = this._phase4BindGroups;

        const pass = encoder.beginComputePass({ label: 'recordHistory' });
        pass.setPipeline(p4.recordHistory.pipeline);
        pass.setBindGroup(0, bgs.historyG0);
        pass.setBindGroup(1, bgs.historyG1);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
    }

    /**
     * Dispatch dead particle garbage collection (Phase 3).
     * Runs once per frame (not per substep), after all substeps complete.
     */
    _dispatchDeadGC(encoder) {
        if (this.aliveCount === 0) return;

        const pass = encoder.beginComputePass({ label: 'deadGC' });
        pass.setPipeline(this._deadGCPipeline);
        pass.setBindGroup(0, this._deadGCBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.buffers.maxParticles / 64));
        pass.end();
    }

    /**
     * Dispatch collision detection and resolution (Phase 3).
     * Always uses tree-accelerated broadphase (tree built every substep).
     *
     * COL_PASS (0):   skip entirely.
     * COL_MERGE (1):  tree detect → resolveCollisions (merge/annihilation).
     * COL_BOUNCE (2): tree detect → resolveBouncePairwise (Hertz impulse).
     */
    _dispatchCollisions(encoder) {
        if (this._collisionMode === 0) return; // COL_PASS — nothing to do
        if (this.aliveCount === 0) return;

        const b = this.buffers;
        const isMerge  = this._collisionMode === COL_MERGE;
        const isBounce = this._collisionMode === COL_BOUNCE;

        // Reset pair counter (and merge counter for merge mode)
        encoder.clearBuffer(b.collisionPairCounter, 0, 4);
        if (isMerge) encoder.clearBuffer(b.mergeResultCounter, 0, 4);

        // ── Detection phase (always tree-accelerated) ──
        const detectWG = Math.ceil(this.aliveCount / 64);
        const passDetect = encoder.beginComputePass({ label: 'detectCollisions' });
        passDetect.setPipeline(this._collisionPipelines.detectCollisions);
        passDetect.setBindGroup(0, this._collisionBG0);
        passDetect.setBindGroup(1, this._collisionBG1);
        passDetect.setBindGroup(2, this._collisionBG2);
        passDetect.dispatchWorkgroups(detectWG);
        passDetect.end();

        // ── Resolution phase ──
        // Conservatively dispatch enough workgroups to cover worst-case pair count.
        // Each shader reads pairCounter atomically and exits early for out-of-range threads.
        const resolveWG = Math.ceil(b.maxParticles / 64);
        if (isMerge) {
            const passResolve = encoder.beginComputePass({ label: 'resolveCollisions' });
            passResolve.setPipeline(this._collisionPipelines.resolveCollisions);
            passResolve.setBindGroup(0, this._collisionBG0);
            passResolve.setBindGroup(1, this._collisionBG1);
            passResolve.setBindGroup(2, this._collisionBG2);
            passResolve.dispatchWorkgroups(resolveWG);
            passResolve.end();
        } else if (isBounce) {
            const passResolve = encoder.beginComputePass({ label: 'resolveBouncePairwise' });
            passResolve.setPipeline(this._collisionPipelines.resolveBouncePairwise);
            passResolve.setBindGroup(0, this._collisionBG0);
            passResolve.setBindGroup(1, this._collisionBG1);
            passResolve.setBindGroup(2, this._collisionBG2);
            passResolve.dispatchWorkgroups(resolveWG);
            passResolve.end();
        }
    }

    /**
     * Non-blocking readback of merge results for post-processing by JS.
     * Returns pending merge events (annihilation photon bursts, field excitations).
     * Uses 1-frame latency like other readbacks.
     */
    async _readbackMergeResults() {
        if (this._mergeResultsPending) return;
        if (this._collisionMode !== COL_MERGE) return;
        this._mergeResultsPending = true;

        const b = this.buffers;
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(b.mergeResultCounter, 0, b.mergeCountStaging, 0, 4);
        this.device.queue.submit([encoder.finish()]);

        await b.mergeCountStaging.mapAsync(GPUMapMode.READ);
        const countData = new Uint32Array(b.mergeCountStaging.getMappedRange().slice(0));
        b.mergeCountStaging.unmap();

        const mergeCount = countData[0];
        if (mergeCount > 0) {
            const readBytes = Math.min(mergeCount, this.buffers.maxParticles) * 16;
            const encoder2 = this.device.createCommandEncoder();
            encoder2.copyBufferToBuffer(b.mergeResultBuffer, 0, b.mergeResultStaging, 0, readBytes);
            this.device.queue.submit([encoder2.finish()]);

            await b.mergeResultStaging.mapAsync(GPUMapMode.READ);
            const resultData = new Float32Array(b.mergeResultStaging.getMappedRange(0, readBytes).slice(0));
            b.mergeResultStaging.unmap();

            // Parse merge events: each is vec4(x, y, energy, type)
            const events = [];
            for (let i = 0; i < mergeCount; i++) {
                events.push({
                    x: resultData[i * 4],
                    y: resultData[i * 4 + 1],
                    energy: resultData[i * 4 + 2],
                    type: resultData[i * 4 + 3] < 0.5 ? 'annihilation' : 'merge',
                });
            }
            this._pendingMergeEvents = events;
        } else {
            this._pendingMergeEvents = [];
        }

        this._mergeResultsPending = false;
    }

    /**
     * Consume pending merge events (call from main loop for photon bursts, field excitations).
     * @returns {Array} Array of { x, y, energy, type } events
     */
    consumeMergeEvents() {
        const events = this._pendingMergeEvents;
        this._pendingMergeEvents = [];
        return events;
    }

    /**
     * Dispatch tree build sequence (always runs — used by collisions, hit test,
     * and optionally by BH tree-walk force computation).
     * Runs after ghost generation, before force computation.
     *
     * Sequence:
     *   1. Reset nodeCounter to 1 (root = node 0, pre-allocated)
     *   2. Reset bounds to (INT_MAX, INT_MAX, INT_MIN, INT_MIN)
     *   3. Clear visitor flags to 0
     *   4. computeBounds: ceil(totalCount / 256) workgroups
     *   5. initRoot: 1 workgroup (single thread)
     *   6. insertParticles: ceil(totalCount / 64) workgroups
     *   7. computeAggregates: ceil(totalCount / 64) workgroups
     */
    _dispatchTreeBuild(encoder) {

        const totalCount = this.aliveCount + this._ghostCount;
        if (totalCount === 0) return;

        const b = this.buffers;

        // 1-2. Reset nodeCounter and bounds via encoder.copyBufferToBuffer.
        // MUST be encoder ops (not queue.writeBuffer) because _dispatchTreeBuild
        // may be called twice per substep (before forces + after drift for 1PN VV).
        // queue.writeBuffer executes at queue time (before encoder starts), so a
        // second call can't reset state between two tree builds in the same encoder.
        encoder.copyBufferToBuffer(this._qtCounterSrc, 0, b.qtNodeCounter, 0, 4);
        encoder.copyBufferToBuffer(this._qtBoundsSrc, 0, b.qtBoundsBuffer, 0, 16);

        // 3. Clear ALL visitor flags (not just totalCount*6 — a second tree build
        // in the same substep allocates nodes beyond that range).
        const clearSize = Math.min(b.QT_MAX_NODES * 4, b.qtVisitorFlags.size);
        encoder.clearBuffer(b.qtVisitorFlags, 0, clearSize);

        // 4. computeBounds dispatch
        const boundsWG = Math.ceil(totalCount / 256);
        const pass1 = encoder.beginComputePass({ label: 'computeBounds' });
        pass1.setPipeline(this._treeBuild.computeBounds);
        pass1.setBindGroup(0, this._treeBuildBG0);
        pass1.setBindGroup(1, this._treeBuildBG1);
        pass1.setBindGroup(2, this._treeBuildBG2);
        pass1.dispatchWorkgroups(boundsWG);
        pass1.end();

        // 5. initRoot dispatch (single thread reads bounds, writes root node)
        const pass2 = encoder.beginComputePass({ label: 'initRoot' });
        pass2.setPipeline(this._treeBuild.initRoot);
        pass2.setBindGroup(0, this._treeBuildBG0);
        pass2.setBindGroup(1, this._treeBuildBG1);
        pass2.setBindGroup(2, this._treeBuildBG2);
        pass2.dispatchWorkgroups(1);
        pass2.end();

        // 6. insertParticles dispatch
        const insertWG = Math.ceil(totalCount / 64);
        const pass3 = encoder.beginComputePass({ label: 'insertParticles' });
        pass3.setPipeline(this._treeBuild.insertParticles);
        pass3.setBindGroup(0, this._treeBuildBG0);
        pass3.setBindGroup(1, this._treeBuildBG1);
        pass3.setBindGroup(2, this._treeBuildBG2);
        pass3.dispatchWorkgroups(insertWG);
        pass3.end();

        // 7. computeAggregates dispatch
        const aggWG = Math.ceil(totalCount / 64);
        const pass4 = encoder.beginComputePass({ label: 'computeAggregates' });
        pass4.setPipeline(this._treeBuild.computeAggregates);
        pass4.setBindGroup(0, this._treeBuildBG0);
        pass4.setBindGroup(1, this._treeBuildBG1);
        pass4.setBindGroup(2, this._treeBuildBG2);
        pass4.dispatchWorkgroups(aggWG);
        pass4.end();
    }

    /**
     * Non-blocking readback of ghost count for tree build sizing.
     * Uses 1-frame latency like maxAccel readback.
     */
    async _readbackGhostCount() {
        if (this._ghostCountPending) return;
        if (this.boundaryMode !== BOUND_LOOP) {
            this._ghostCount = 0;
            return;
        }
        this._ghostCountPending = true;

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.ghostCounter, 0,
            this.buffers.ghostCountStaging, 0, 4);
        this.device.queue.submit([encoder.finish()]);

        await this.buffers.ghostCountStaging.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(this.buffers.ghostCountStaging.getMappedRange().slice(0));
        this.buffers.ghostCountStaging.unmap();

        this._ghostCount = data[0];
        this._ghostCountPending = false;
    }

    /**
     * Change the scalar field grid resolution. Requires reallocating field buffers.
     * @param {number} res - Power of 2: 64, 128, or 256
     */
    setFieldResolution(res) {
        if (res !== 64 && res !== 128 && res !== 256) return;
        if (res === this._fieldResolution) return;

        this._fieldResolution = res;

        // TODO: Reallocate field buffers at new resolution once field compute is fully wired.
        // this._reallocFieldBuffers(res);

        if (typeof showToast === 'function') showToast(`Field grid: ${res}\u00D7${res}`);
    }

    /**
     * Add a particle to the GPU buffers.
     * Reuses free slots from dead GC when available, otherwise appends.
     */
    addParticle({ x, y, vx = 0, vy = 0, mass: m = 1, charge: q = 0, angw = 0, antimatter = false }) {
        let idx;
        if (this._cpuFreeSlots.length > 0) {
            // Reuse a slot freed by dead GC
            idx = this._cpuFreeSlots.pop();
        } else {
            idx = this.aliveCount;
            if (idx >= MAX_PARTICLES) return -1;
            this.aliveCount++;
        }

        // Write packed ParticleState struct (36 bytes = 9 × f32/u32) using pre-allocated buffers
        _addParticleStateF32[0] = x;          // posX
        _addParticleStateF32[1] = y;          // posY
        _addParticleStateF32[2] = vx;         // velWX
        _addParticleStateF32[3] = vy;         // velWY
        _addParticleStateF32[4] = m;          // mass
        _addParticleStateF32[5] = q;          // charge
        _addParticleStateF32[6] = angw;       // angW
        _addParticleStateF32[7] = m;          // baseMass
        let flags = FLAG_ALIVE;
        if (antimatter && !this._blackHoleEnabled) flags |= FLAG_ANTIMATTER;
        _addParticleStateU32[8] = flags;
        this.device.queue.writeBuffer(this.buffers.particleState, idx * PARTICLE_STATE_SIZE, _addParticleStateData);

        // Write packed ParticleAux struct (20 bytes = 5 × f32/u32)
        _addParticleAuxF32[0] = Math.cbrt(m); // radius
        _addParticleAuxU32[1] = idx;          // particleId
        _addParticleAuxF32[2] = 3.4028235e38;     // deathTime (not dead)
        _addParticleAuxF32[3] = 0;            // deathMass
        _addParticleAuxF32[4] = 0;            // deathAngVel
        this.device.queue.writeBuffer(this.buffers.particleAux, idx * PARTICLE_AUX_SIZE, _addParticleAuxData);

        // Pack color: neutral slate = #8A7E72 -> RGBA
        _addParticleColorData[0] = 0xFF727E8A; // ABGR packed
        this.device.queue.writeBuffer(this.buffers.color, idx * 4, _addParticleColorData);

        // cacheDerived shader computes derived state before forces each substep.
        // No need to initialize derived here — but axYukMod must be set to (1, 1).
        _addParticleModData[0] = 1.0; // axMod
        _addParticleModData[1] = 1.0; // yukMod
        this.device.queue.writeBuffer(this.buffers.axYukMod, idx * 8, _addParticleModData);

        // Initialize radiationState to zero (jerk, accumulators, display force)
        _addParticleRadData.fill(0);
        this.device.queue.writeBuffer(this.buffers.radiationState, idx * RADIATION_STATE_SIZE, _addParticleRadData);

        // Reset trail ring buffer for this slot (clear writeIdx + count)
        if (this._trailBuffers) {
            const tb = this._trailBuffers;
            this.device.queue.writeBuffer(tb.trailWriteIdx, idx * 4, _zeroU32);
            this.device.queue.writeBuffer(tb.trailCount, idx * 4, _zeroU32);
        }

        // Initialize history metadata (4 u32: writeIdx, count, creationTimeBits, _pad)
        if (this.buffers.historyAllocated) {
            _addParticleMetaU32[0] = 0;  // writeIdx
            _addParticleMetaU32[1] = 0;  // count
            _addParticleMetaF32[2] = this.simTime; // creationTime as f32 bits
            _addParticleMetaU32[3] = 0;  // _pad
            this.device.queue.writeBuffer(
                this.buffers.histMeta,
                idx * HIST_META_STRIDE * 4, // 4 u32 per particle
                _addParticleMetaBuf
            );
        }
        return idx;
    }

    /**
     * Remove a particle from the GPU by marking it dead (clearing ALIVE, setting RETIRED).
     * Writes death metadata for signal delay fade-out matching collision/boundary shaders.
     * The dead GC shader will reclaim the slot after the signal delay expiry window.
     * @param {number} idx - GPU buffer index of the particle to remove
     * @param {number} deathMass - particle mass at time of death
     * @param {number} deathAngVel - particle angular velocity at time of death
     */
    removeParticle(idx, deathMass = 0, deathAngVel = 0) {
        if (idx < 0 || idx >= this.aliveCount) return;
        // Clear ALIVE bit, set RETIRED bit in flags (byte offset 8 in ParticleState = 32 bytes in)
        // ParticleState: posX(0), posY(4), velWX(8), velWY(12), mass(16), charge(20), angW(24), baseMass(28), flags(32)
        this.device.queue.writeBuffer(this.buffers.particleState, idx * PARTICLE_STATE_SIZE + 32, _retiredFlagU32);
        // Write death metadata to ParticleAux: deathTime(offset 8), deathMass(12), deathAngVel(16)
        _deathMetaF32[0] = this.simTime;
        _deathMetaF32[1] = deathMass;
        _deathMetaF32[2] = deathAngVel;
        this.device.queue.writeBuffer(this.buffers.particleAux, idx * PARTICLE_AUX_SIZE + 8, _deathMetaData);
    }

    /**
     * Pack toggle booleans into u32 bitfields for GPU uniforms.
     * Must be called whenever a toggle changes.
     */
    setToggles(physics) {
        let t0 = 0;
        if (physics.gravityEnabled) t0 |= GRAVITY_BIT;
        if (physics.coulombEnabled) t0 |= COULOMB_BIT;
        if (physics.magneticEnabled) t0 |= MAGNETIC_BIT;
        if (physics.gravitomagEnabled) t0 |= GRAVITOMAG_BIT;
        if (physics.onePNEnabled) t0 |= ONE_PN_BIT;
        if (physics.relativityEnabled) t0 |= RELATIVITY_BIT;
        if (physics.spinOrbitEnabled) t0 |= SPIN_ORBIT_BIT;
        if (physics.radiationEnabled) t0 |= RADIATION_BIT;
        if (physics.blackHoleEnabled) t0 |= BLACK_HOLE_BIT;
        if (physics.disintegrationEnabled) t0 |= DISINTEGRATION_BIT;
        if (physics.expansionEnabled) t0 |= EXPANSION_BIT;
        if (physics.yukawaEnabled) t0 |= YUKAWA_BIT;
        if (physics.higgsEnabled) t0 |= HIGGS_BIT;
        if (physics.axionEnabled) t0 |= AXION_BIT;
        if (physics.barnesHutEnabled) t0 |= BARNES_HUT_BIT;
        if (physics.bosonInterEnabled) t0 |= BOSON_INTER_BIT;
        this._toggles0 = t0;

        let t1 = 0;
        if (physics.gravityEnabled) t1 |= 1; // field gravity follows gravity
        this._toggles1 = t1;

        this._gravityEnabled = physics.gravityEnabled;
        this._coulombEnabled = physics.coulombEnabled;
        this._magneticEnabled = physics.magneticEnabled;
        this._gravitomagEnabled = physics.gravitomagEnabled;
        this._spinOrbitEnabled = physics.spinOrbitEnabled;
        this._blackHoleEnabled = physics.blackHoleEnabled;
        this._barnesHutEnabled = physics.barnesHutEnabled;
        this._relativityEnabled = physics.relativityEnabled;
        this._onePNEnabled = physics.onePNEnabled;
        this._radiationEnabled = physics.radiationEnabled;
        this._yukawaEnabled = physics.yukawaEnabled;
        this._bosonInterEnabled = physics.bosonInterEnabled;
        this._yukawaCoupling = physics.yukawaCoupling ?? 14;

        // Lazily allocate history buffers when relativity is first enabled
        if (this._relativityEnabled && this._phase4) {
            this._ensureHistoryBindGroups();
        }
        this._yukawaMu = physics.yukawaMu;
        this._higgsMass = physics.higgsMass ?? 0.5;
        this._axionMass = physics.axionMass ?? 0.05;
        this._extGravity = physics.extGravity;
        this._extGravityAngle = physics.extGravityAngle;
        this._extElectric = physics.extElectric;
        this._extElectricAngle = physics.extElectricAngle;
        this._extBz = physics.extBz;
        this._bounceFriction = physics.bounceFriction;
        this._collisionMode = physics.collisionMode || 0;
        this._axionCoupling = 0.05;
        this._higgsCoupling = 1.0;

        // Phase 5 toggle state
        this._higgsEnabled = physics.higgsEnabled;
        this._axionEnabled = physics.axionEnabled;
        this._expansionEnabled = physics.expansionEnabled;
        this._disintegrationEnabled = physics.disintegrationEnabled;
        this._hubbleParam = physics.hubbleParam || 0.001;
        this._heatmapEnabled = physics.heatmapEnabled || false;
        this._heatmapMode = physics.heatmapMode || 'all';

        // Lazily allocate scalar field buffers on first toggle-on (matching CPU pattern)
        if (physics.higgsEnabled && !this._higgsBuffers) {
            this._ensureFieldBuffers('higgs');
        }
        if (physics.axionEnabled && !this._axionBuffers) {
            this._ensureFieldBuffers('axion');
        }

        // Lazily initialize Phase 5 pipelines when any Phase 5 feature is enabled
        const needsPhase5 = physics.higgsEnabled || physics.axionEnabled ||
            physics.expansionEnabled || physics.disintegrationEnabled || this._heatmapEnabled;
        if (needsPhase5 && !this._fieldDeposit) {
            this._ensurePhase5Pipelines();
        }
    }

    /**
     * Lazily allocate GPU buffers for a scalar field.
     * Also ensures shared atomic deposit grid buffer is allocated.
     * Initializes field values: Higgs=1.0 (VEV), Axion=0.0.
     * @param {'higgs'|'axion'} which
     */
    _ensureFieldBuffers(which) {
        if (!this._atomicGrid) {
            this._atomicGrid = createAtomicGridBuffer(this.device);
        }
        if (!this._fieldUniformBuffer) {
            this._fieldUniformBuffer = this.device.createBuffer({
                label: 'fieldUniforms',
                size: 256,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            // Per-field uniform buffers: eliminate encoder split when both Higgs + Axion enabled
            this._higgsUniformBuffer = this.device.createBuffer({
                label: 'higgsFieldUniforms',
                size: 256,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this._axionUniformBuffer = this.device.createBuffer({
                label: 'axionFieldUniforms',
                size: 256,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            // Dummy zero buffer for portal coupling when other field not yet allocated
            this._dummyFieldBuffer = this.device.createBuffer({
                label: 'dummyField',
                size: FIELD_GRID_RES * FIELD_GRID_RES * 4,
                usage: GPUBufferUsage.STORAGE,
            });
        }
        if (which === 'higgs' && !this._higgsBuffers) {
            this._higgsBuffers = createFieldBuffers(this.device, 'higgs', MAX_PARTICLES);
            this._initFieldToVacuum('higgs');
            // Invalidate other field's evolve bind groups (portal coupling references this buffer)
            this._fieldEvolveBGs['axion'] = null;
            this._fieldGradBGs['axion'] = null;
        }
        if (which === 'axion' && !this._axionBuffers) {
            this._axionBuffers = createFieldBuffers(this.device, 'axion', MAX_PARTICLES);
            this._initFieldToVacuum('axion');
            this._fieldEvolveBGs['higgs'] = null;
            this._fieldGradBGs['higgs'] = null;
        }
    }

    /**
     * Initialize field buffers to vacuum values.
     * Higgs VEV = 1.0, Axion vacuum = 0.0. All other arrays zeroed.
     */
    _initFieldToVacuum(which) {
        const gridSq = FIELD_GRID_RES * FIELD_GRID_RES;
        const fb = which === 'higgs' ? this._higgsBuffers : this._axionBuffers;
        if (!fb) return;

        const vacValue = which === 'higgs' ? 1.0 : 0.0;
        _vacuumFieldData.fill(vacValue, 0, gridSq);
        this.device.queue.writeBuffer(fb.field, 0, _vacuumFieldData, 0, gridSq);

        // Zero all other field buffers using pre-allocated zero array
        const zeroBytes = gridSq * 4;
        this.device.queue.writeBuffer(fb.fieldDot, 0, _zeroFieldData, 0, gridSq);
        this.device.queue.writeBuffer(fb.gradX, 0, _zeroFieldData, 0, gridSq);
        this.device.queue.writeBuffer(fb.gradY, 0, _zeroFieldData, 0, gridSq);
        this.device.queue.writeBuffer(fb.source, 0, _zeroFieldData, 0, gridSq);
        this.device.queue.writeBuffer(fb.thermal, 0, _zeroFieldData, 0, gridSq);
        this.device.queue.writeBuffer(fb.energyDensity, 0, _zeroFieldData, 0, gridSq);
        this.device.queue.writeBuffer(fb.sgPhiFull, 0, _zeroFieldData, 0, gridSq);
        this.device.queue.writeBuffer(fb.sgGradX, 0, _zeroFieldData, 0, gridSq);
        this.device.queue.writeBuffer(fb.sgGradY, 0, _zeroFieldData, 0, gridSq);
    }

    /**
     * Lazily initialize Phase 5 pipelines on first use.
     * Called when any Phase 5 feature is first enabled.
     */
    async _ensurePhase5Pipelines() {
        if (this._fieldDeposit) return; // already initialized

        const wgslConstants = buildWGSLConstants();

        // Initialize all Phase 5 pipelines in parallel
        const [deposit, evolve, forces, particleGrav, selfGrav, fft, excitation, heatmap, expansion, disint, pairProd] =
            await Promise.all([
                createFieldDepositPipelines(this.device, wgslConstants),
                createFieldEvolvePipelines(this.device, wgslConstants),
                createFieldForcesPipelines(this.device, wgslConstants),
                createFieldParticleGravPipeline(this.device, wgslConstants),
                createFieldSelfGravPipelines(this.device, wgslConstants),
                createFFTPipelines(this.device, wgslConstants),
                createFieldExcitationPipeline(this.device, wgslConstants),
                createHeatmapPipelines(this.device, wgslConstants),
                createExpansionPipeline(this.device, wgslConstants),
                createDisintegrationPipeline(this.device, wgslConstants),
                createPairProductionPipeline(this.device, wgslConstants),
            ]);

        this._fieldDeposit = deposit;
        this._fieldEvolve = evolve;
        this._fieldForces = forces;
        this._fieldParticleGrav = particleGrav;
        this._fieldSelfGrav = selfGrav;
        this._fftPipelines = fft;
        this._fieldExcitation = excitation;
        this._heatmapPipelines = heatmap;
        this._expansionPipeline = expansion;
        this._disintPipeline = disint;
        this._pairProdPipeline = pairProd;
    }

    /**
     * Write FieldUniforms to a target buffer.
     * When fieldType >= 0, writes to per-field buffer (higgs=0, axion=1).
     * When fieldType < 0, writes to shared field uniform buffer (fieldType defaults to 0).
     * @param {number} dt
     * @param {number} [fieldType=-1] - -1=shared buffer, 0=higgs, 1=axion
     */
    _writeFieldUniforms(dt, fieldType = -1) {
        const buf = fieldType < 0
            ? this._fieldUniformBuffer
            : (fieldType === 0 ? this._higgsUniformBuffer : this._axionUniformBuffer);
        if (!buf) return;
        const f = _fieldUniformF32;
        const u = _fieldUniformU32;
        // Must match FieldUniforms struct in field-common.wgsl exactly:
        f[0] = dt;
        f[1] = this.domainW;
        f[2] = this.domainH;
        u[3] = this.boundaryMode;
        u[4] = this.topologyMode;
        f[5] = this._higgsMass;
        f[6] = this._higgsCoupling;
        f[7] = 0.05;                            // higgsMassFloor
        f[8] = 4.0;                             // higgsMassMaxDelta
        f[9] = this._axionMass;
        f[10] = this._axionCoupling;
        u[11] = this._higgsEnabled ? 1 : 0;
        u[12] = this._axionEnabled ? 1 : 0;
        u[13] = this._coulombEnabled ? 1 : 0;
        u[14] = this._yukawaEnabled ? 1 : 0;
        u[15] = this._gravityEnabled ? 1 : 0;
        u[16] = this._relativityEnabled ? 1 : 0;
        u[17] = this._blackHoleEnabled ? 1 : 0;
        u[18] = this.aliveCount;
        f[19] = this._blackHoleEnabled ? 16 : 64;
        u[20] = fieldType < 0 ? 0 : fieldType;
        this.device.queue.writeBuffer(buf, 0, _fieldUniformData);
    }

    /**
     * Ensure deposit bind groups exist for a given field.
     */
    _ensureDepositBindGroups(which) {
        if (this._fieldDepositBGs[which]) return;
        const fb = which === 'higgs' ? this._higgsBuffers : this._axionBuffers;
        if (!fb || !this._fieldDeposit) return;
        const b = this.buffers;
        const dep = this._fieldDeposit;

        // Group 0: packed particle state
        const g0 = this.device.createBindGroup({
            label: `fieldDeposit_g0_${which}`,
            layout: dep.bindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: b.particleState } },
            ],
        });

        // Use per-field uniform buffer
        const uBuf = which === 'higgs' ? this._higgsUniformBuffer : this._axionUniformBuffer;

        // Group 1: atomicGrid + target grid + uniforms (for source deposition)
        const g1Source = this.device.createBindGroup({
            label: `fieldDeposit_g1_source_${which}`,
            layout: dep.bindGroupLayouts[1],
            entries: [
                { binding: 0, resource: { buffer: this._atomicGrid } },
                { binding: 1, resource: { buffer: fb.source } },
                { binding: 2, resource: { buffer: uBuf } },
            ],
        });

        // Group 1 for thermal (Higgs only, but create for both for simplicity)
        const g1Thermal = this.device.createBindGroup({
            label: `fieldDeposit_g1_thermal_${which}`,
            layout: dep.bindGroupLayouts[1],
            entries: [
                { binding: 0, resource: { buffer: this._atomicGrid } },
                { binding: 1, resource: { buffer: fb.thermal } },
                { binding: 2, resource: { buffer: uBuf } },
            ],
        });

        this._fieldDepositBGs[which] = { g0, g1Source, g1Thermal };
    }

    /**
     * Ensure evolve bind groups exist for a given field.
     */
    _ensureEvolveBindGroups(which) {
        if (this._fieldEvolveBGs[which]) return;
        const fb = which === 'higgs' ? this._higgsBuffers : this._axionBuffers;
        if (!fb || !this._fieldEvolve) return;

        // Use per-field uniform buffer to avoid encoder split when both fields active
        const uBuf = which === 'higgs' ? this._higgsUniformBuffer : this._axionUniformBuffer;

        // Other field's buffer for portal coupling (or dummy zeros if not allocated)
        const otherFb = which === 'higgs' ? this._axionBuffers : this._higgsBuffers;
        const otherFieldBuf = otherFb ? otherFb.field : this._dummyFieldBuffer;

        // Evolve bind group (gradX/Y are read-only for self-gravity cross-terms)
        this._fieldEvolveBGs[which] = this.device.createBindGroup({
            label: `fieldEvolve_${which}`,
            layout: this._fieldEvolve.evolveBindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: fb.field } },
                { binding: 1, resource: { buffer: fb.fieldDot } },
                { binding: 2, resource: { buffer: otherFieldBuf } },
                { binding: 3, resource: { buffer: fb.source } },
                { binding: 4, resource: { buffer: fb.thermal } },
                { binding: 5, resource: { buffer: fb.sgPhiFull } },
                { binding: 6, resource: { buffer: fb.sgGradX } },
                { binding: 7, resource: { buffer: fb.sgGradY } },
                { binding: 8, resource: { buffer: fb.gradX } },
                { binding: 9, resource: { buffer: fb.gradY } },
                { binding: 10, resource: { buffer: uBuf } },
            ],
        });

        // Gradient bind group (gradX/Y are read_write for output)
        this._fieldGradBGs[which] = this.device.createBindGroup({
            label: `fieldGrad_${which}`,
            layout: this._fieldEvolve.gradBindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: fb.field } },
                { binding: 1, resource: { buffer: fb.fieldDot } },
                { binding: 2, resource: { buffer: otherFieldBuf } },
                { binding: 3, resource: { buffer: fb.source } },
                { binding: 4, resource: { buffer: fb.thermal } },
                { binding: 5, resource: { buffer: fb.sgPhiFull } },
                { binding: 6, resource: { buffer: fb.sgGradX } },
                { binding: 7, resource: { buffer: fb.sgGradY } },
                { binding: 8, resource: { buffer: fb.gradX } },
                { binding: 9, resource: { buffer: fb.gradY } },
                { binding: 10, resource: { buffer: uBuf } },
            ],
        });
    }

    /**
     * Ensure self-gravity bind groups exist for a given field.
     */
    _ensureSelfGravBindGroups(which) {
        if (this._fieldSelfGravBGs[which]) return;
        const fb = which === 'higgs' ? this._higgsBuffers : this._axionBuffers;
        if (!fb || !this._fieldSelfGrav) return;

        // Use per-field uniform buffer
        const uBuf = which === 'higgs' ? this._higgsUniformBuffer : this._axionUniformBuffer;

        // Group 0: field arrays + uniform
        this._fieldSelfGravBGs[which] = this.device.createBindGroup({
            label: `fieldSelfGrav_${which}_g0`,
            layout: this._fieldSelfGrav.bindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: fb.field } },
                { binding: 1, resource: { buffer: fb.fieldDot } },
                { binding: 2, resource: { buffer: fb.gradX } },
                { binding: 3, resource: { buffer: fb.gradY } },
                { binding: 4, resource: { buffer: fb.energyDensity } },
                { binding: 5, resource: { buffer: uBuf } },
            ],
        });
        // Group 1: SG output arrays
        this._fieldSelfGravBGs[which + '_g1'] = this.device.createBindGroup({
            label: `fieldSelfGrav_${which}_g1`,
            layout: this._fieldSelfGrav.bindGroupLayouts[1],
            entries: [
                { binding: 0, resource: { buffer: fb.sgPhiFull } },
                { binding: 1, resource: { buffer: fb.sgGradX } },
                { binding: 2, resource: { buffer: fb.sgGradY } },
            ],
        });
        // Group 2: FFT complex buffer (fftA) for fused energy density + pack / unpack + gradient
        this._fieldSelfGravBGs[which + '_g2'] = this.device.createBindGroup({
            label: `fieldSelfGrav_${which}_g2`,
            layout: this._fieldSelfGrav.bindGroupLayouts[2],
            entries: [
                { binding: 0, resource: { buffer: fb.fftA } },
            ],
        });
    }

    /** Ensure FFT bind groups exist for a given field. Also uploads Green's function Ĝ. */
    _ensureFFTBindGroups(which) {
        if (this._fftBGs[which]) return;
        const fb = which === 'higgs' ? this._higgsBuffers : this._axionBuffers;
        if (!fb || !this._fftPipelines) return;

        // Create FFT params uniform buffer (shared, created once)
        if (!this._fftParamsBuffer) {
            this._fftParamsBuffer = this.device.createBuffer({
                label: 'fftParams',
                size: 32, // 8 × u32/f32
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        }

        const fft = this._fftPipelines;

        // Group 0 forward (A→B): read A, write B
        const g0_AB = this.device.createBindGroup({
            label: `fft_g0_AB_${which}`,
            layout: fft.bindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: fb.fftA } },
                { binding: 1, resource: { buffer: fb.fftB } },
                { binding: 2, resource: { buffer: this._fftParamsBuffer } },
            ],
        });

        // Group 0 reverse (B→A): read B, write A
        const g0_BA = this.device.createBindGroup({
            label: `fft_g0_BA_${which}`,
            layout: fft.bindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: fb.fftB } },
                { binding: 1, resource: { buffer: fb.fftA } },
                { binding: 2, resource: { buffer: this._fftParamsBuffer } },
            ],
        });

        // Group 1: Green's function
        const g1Green = this.device.createBindGroup({
            label: `fft_g1_green_${which}`,
            layout: fft.bindGroupLayouts[1],
            entries: [
                { binding: 0, resource: { buffer: fb.greenHat } },
            ],
        });

        this._fftBGs[which] = { g0_AB, g0_BA, g1Green };
    }

    /** Upload precomputed Green's function Ĝ to GPU.
     *  Computes on CPU (same as scalar-field.js), FFTs, uploads once per geometry change. */
    _uploadGreenHat(which, domainW, domainH, softeningSq, periodic, topology) {
        const key = `${domainW},${domainH},${softeningSq},${periodic},${topology}`;
        if (this._greenHatUploaded[which] === key) return;
        this._greenHatUploaded[which] = key;

        const fb = which === 'higgs' ? this._higgsBuffers : this._axionBuffers;
        if (!fb) return;

        const N = FIELD_GRID_RES;
        const N2 = N * N;
        const re = new Float64Array(N2);
        const im = new Float64Array(N2);
        const cellW = domainW / N;
        const cellH = domainH / N;

        // Build real-space Green's function G(r) = -1/√(r²+ε²)
        // Wrapped indices: FFT circular convolution naturally handles periodic boundaries
        for (let iy = 0; iy < N; iy++) {
            for (let ix = 0; ix < N; ix++) {
                const dx = (ix <= N / 2 ? ix : ix - N) * cellW;
                const dy = (iy <= N / 2 ? iy : iy - N) * cellH;
                re[iy * N + ix] = -1 / Math.sqrt(dx * dx + dy * dy + softeningSq);
            }
        }

        // Forward FFT to get Ĝ
        fft2d(re, im, N, false);

        // Convert Float64 → Float32 interleaved and upload
        for (let i = 0; i < N2; i++) {
            _greenHatUploadData[i * 2] = re[i];
            _greenHatUploadData[i * 2 + 1] = im[i];
        }
        this.device.queue.writeBuffer(fb.greenHat, 0, _greenHatUploadData, 0, N2 * 2);
    }

    /**
     * Ensure field forces bind groups exist (shared for both Higgs and Axion).
     */
    _ensureFieldForcesBGs() {
        if (this._fieldForcesBGs) return;
        if (!this._fieldForces || !this._higgsBuffers || !this._axionBuffers) return;
        // Need both field buffers — use dummy if one missing
        const hb = this._higgsBuffers;
        const ab = this._axionBuffers;
        // If one field is missing, we can't create force BGs yet
        // They'll be created when both are available, or on first use with dummies
        if (!hb || !ab) return;
        const b = this.buffers;
        const ff = this._fieldForces;

        const g0 = this.device.createBindGroup({
            label: 'fieldForces_g0',
            layout: ff.bindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: b.particleState } },
                { binding: 1, resource: { buffer: b.derived } },
            ],
        });

        const g1 = this.device.createBindGroup({
            label: 'fieldForces_g1',
            layout: ff.bindGroupLayouts[1],
            entries: [
                { binding: 0, resource: { buffer: hb.field } },
                { binding: 1, resource: { buffer: hb.gradX } },
                { binding: 2, resource: { buffer: hb.gradY } },
                { binding: 3, resource: { buffer: ab.field } },
                { binding: 4, resource: { buffer: ab.gradX } },
                { binding: 5, resource: { buffer: ab.gradY } },
            ],
        });

        const g2 = this.device.createBindGroup({
            label: 'fieldForces_g2',
            layout: ff.bindGroupLayouts[2],
            entries: [
                { binding: 0, resource: { buffer: b.allForces } },
                { binding: 1, resource: { buffer: b.axYukMod } },
            ],
        });

        const g3 = this.device.createBindGroup({
            label: 'fieldForces_g3',
            layout: ff.bindGroupLayouts[3],
            entries: [
                { binding: 0, resource: { buffer: this._fieldUniformBuffer } },
            ],
        });

        this._fieldForcesBGs = { g0, g1, g2, g3 };
    }

    /**
     * Dispatch self-gravity FFT convolution for one field.
     * Sequence: energy density → FFT(ρ·dA) → multiply Ĝ → IFFT → SG gradients.
     * Called twice per KDK cycle when field gravity is on: before 1st half-kick
     * and after drift (refreshes Φ for O(dt²) accuracy on GR correction terms).
     * @param {string} tag - label prefix ('pre' or 'mid') for debug labels
     */
    _dispatchSelfGravity(encoder, which, fb, gridWG, tag) {
        this._ensureSelfGravBindGroups(which);
        this._ensureFFTBindGroups(which);

        const sgBG0 = this._fieldSelfGravBGs[which];
        const sgBG1 = this._fieldSelfGravBGs[which + '_g1'];
        const sgBG2 = this._fieldSelfGravBGs[which + '_g2'];
        const sg = this._fieldSelfGrav;
        const fft = this._fftPipelines;
        const fftBG = this._fftBGs[which];
        const softeningSq = this._blackHoleEnabled ? BH_SOFTENING_SQ : SOFTENING_SQ;

        // Upload Green's function if geometry changed
        this._uploadGreenHat(which, this.domainW, this.domainH,
            softeningSq, this.boundaryMode === BOUND_LOOP, this.topologyMode);

        // Fused: compute energy density and pack directly into fftA (complex format)
        const edPipeline = which === 'higgs'
            ? sg.energyDensityHiggsAndPack
            : sg.energyDensityAxionAndPack;
        {
            const p = encoder.beginComputePass({ label: `energyDensityAndPack_${tag}_${which}` });
            p.setPipeline(edPipeline);
            p.setBindGroup(0, sgBG0);
            p.setBindGroup(1, sgBG1);
            p.setBindGroup(2, sgBG2);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }

        // Forward FFT rows then columns (data starts in fftA)
        const N = FIELD_GRID_RES;
        const N2 = N * N;
        const elemWG = Math.ceil(N2 / 256);
        const logN = Math.log2(N);
        let currentInA = true;

        const dispatchButterfly = (stageLen, axis, dir, isLast) => {
            _fftParamsU32[0] = stageLen;
            _fftParamsI32[1] = dir;
            _fftParamsU32[2] = axis;
            _fftParamsU32[3] = N;
            _fftParamsF32[4] = 1 / N;
            _fftParamsU32[5] = isLast ? 1 : 0;
            this.device.queue.writeBuffer(this._fftParamsBuffer, 0, _fftParamsData);

            const bg = currentInA ? fftBG.g0_AB : fftBG.g0_BA;
            const p = encoder.beginComputePass({ label: `fft_${tag}_${dir < 0 ? 'fwd' : 'inv'}_${axis}_${stageLen}_${which}` });
            p.setPipeline(fft.fftButterfly);
            p.setBindGroup(0, bg);
            p.dispatchWorkgroups(elemWG);
            p.end();
            currentInA = !currentInA;
        };

        // Forward FFT (direction = -1)
        for (let s = 0; s < logN; s++) dispatchButterfly(1 << s, 0, -1, false);
        for (let s = 0; s < logN; s++) dispatchButterfly(1 << s, 1, -1, false);

        // Pointwise multiply by Ĝ (2×logN stages is even → data always in fftA)
        {
            const p = encoder.beginComputePass({ label: `complexMul_${tag}_${which}` });
            p.setPipeline(fft.complexMultiply);
            p.setBindGroup(0, fftBG.g0_AB);
            p.setBindGroup(1, fftBG.g1Green);
            p.dispatchWorkgroups(elemWG);
            p.end();
        }

        // Inverse FFT (direction = +1, normalize on last stage)
        currentInA = true;
        for (let s = 0; s < logN; s++) dispatchButterfly(1 << s, 0, 1, s === logN - 1);
        for (let s = 0; s < logN; s++) dispatchButterfly(1 << s, 1, 1, s === logN - 1);

        // Fused: unpack IFFT result from fftA (complex) + compute SG gradients
        // 2×logN stages is even → data always in fftA
        {
            const p = encoder.beginComputePass({ label: `unpackAndSGGrad_${tag}_${which}` });
            p.setPipeline(sg.unpackAndSGGradients);
            p.setBindGroup(0, sgBG0);
            p.setBindGroup(1, sgBG1);
            p.setBindGroup(2, sgBG2);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }
    }

    /**
     * Dispatch scalar field evolution for one field (Higgs or Axion).
     * Full sequence: deposit → [self-gravity] → Laplacian → halfKick → drift →
     * [refresh self-gravity] → Laplacian → halfKick → NaN fixup → compute gradients
     */
    _dispatchFieldEvolve(encoder, which, dt) {
        const fb = which === 'higgs' ? this._higgsBuffers : this._axionBuffers;
        if (!fb || !this._fieldDeposit || !this._fieldEvolve) return;

        this._ensureDepositBindGroups(which);
        this._ensureEvolveBindGroups(which);
        const depBGs = this._fieldDepositBGs[which];
        const evolveBG = this._fieldEvolveBGs[which];
        const gradBG = this._fieldGradBGs[which];
        const dep = this._fieldDeposit;
        const evo = this._fieldEvolve;
        const gridWG = Math.ceil(FIELD_GRID_RES / 8); // 8x8 workgroup
        const particleWG = Math.ceil(this.aliveCount / 256);

        // Step 1: Atomic deposit (source) — single pass, O(N × 16)
        if (this.aliveCount > 0) {
            const depositPipeline = which === 'axion'
                ? dep.depositAxionSource
                : dep.depositHiggsSource;
            const p = encoder.beginComputePass({ label: `deposit_${which}` });
            p.setPipeline(depositPipeline);
            p.setBindGroup(0, depBGs.g0);
            p.setBindGroup(1, depBGs.g1Source);
            p.dispatchWorkgroups(particleWG);
            p.end();
        }

        // Step 2: Finalize source (atomic i32 → f32, clears atomic grid)
        {
            const p = encoder.beginComputePass({ label: `finalizeSource_${which}` });
            p.setPipeline(dep.finalizeDeposit);
            p.setBindGroup(0, depBGs.g0);
            p.setBindGroup(1, depBGs.g1Source);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }

        // Step 3: Higgs thermal deposition (Higgs only)
        if (which === 'higgs') {
            // Atomic deposit thermal
            if (this.aliveCount > 0) {
                const p = encoder.beginComputePass({ label: 'depositThermal' });
                p.setPipeline(dep.depositThermal);
                p.setBindGroup(0, depBGs.g0);
                p.setBindGroup(1, depBGs.g1Thermal);
                p.dispatchWorkgroups(particleWG);
                p.end();
            }
            // Finalize thermal
            {
                const p = encoder.beginComputePass({ label: 'finalizeThermal' });
                p.setPipeline(dep.finalizeDeposit);
                p.setBindGroup(0, depBGs.g0);
                p.setBindGroup(1, depBGs.g1Thermal);
                p.dispatchWorkgroups(gridWG, gridWG);
                p.end();
            }
        }

        // Step 4: Self-gravity via FFT convolution (if field gravity enabled)
        if (this._gravityEnabled) {
            this._dispatchSelfGravity(encoder, which, fb, gridWG, 'pre');
        }

        // Step 6: KDK Störmer-Verlet (Laplacian computed inline in half-kick, NaN fixup in drift)
        const halfKickPipeline = which === 'higgs'
            ? evo.higgsHalfKick
            : evo.axionHalfKick;

        // Half-kick (1st) — computes Laplacian inline
        {
            const p = encoder.beginComputePass({ label: `halfKick1_${which}` });
            p.setPipeline(halfKickPipeline);
            p.setBindGroup(0, evolveBG);

            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }
        // Field drift (with NaN/Inf fixup)
        {
            const p = encoder.beginComputePass({ label: `fieldDrift_${which}` });
            p.setPipeline(evo.fieldDrift);
            p.setBindGroup(0, evolveBG);

            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }
        // Refresh self-gravity at drifted field (restores O(dt²) for GR correction)
        if (this._gravityEnabled) {
            // Recompute field gradients first (needed by energy density + cross-term)
            {
                const p = encoder.beginComputePass({ label: `gridGradientsMid_${which}` });
                p.setPipeline(evo.computeGridGradients);
                p.setBindGroup(0, gradBG);
    
                p.dispatchWorkgroups(gridWG, gridWG);
                p.end();
            }
            this._dispatchSelfGravity(encoder, which, fb, gridWG, 'mid');
        }
        // Half-kick (2nd) — computes Laplacian inline
        {
            const p = encoder.beginComputePass({ label: `halfKick2_${which}` });
            p.setPipeline(halfKickPipeline);
            p.setBindGroup(0, evolveBG);

            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }
        // Compute grid gradients (uses grad layout — bindings 8-9 are rw)
        {
            const p = encoder.beginComputePass({ label: `gridGradients_${which}` });
            p.setPipeline(evo.computeGridGradients);
            p.setBindGroup(0, gradBG);

            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }
    }

    /**
     * Dispatch scalar field force application (Pass 16).
     * Applies Higgs mass modulation + gradient force, Axion axMod/yukMod + gradient force.
     */
    _dispatchFieldForces(encoder) {
        if (!this._fieldForces || this.aliveCount === 0) return;
        if (!this._higgsEnabled && !this._axionEnabled) return;

        // Ensure both field buffers exist for the shared bind group
        // If only one is enabled, the other's arrays are at vacuum (zero/VEV) — safe.
        if (this._higgsEnabled && !this._axionBuffers) {
            this._ensureFieldBuffers('axion');
        }
        if (this._axionEnabled && !this._higgsBuffers) {
            this._ensureFieldBuffers('higgs');
        }
        this._ensureFieldForcesBGs();
        if (!this._fieldForcesBGs) return;

        const workgroups = Math.ceil(this.aliveCount / 256);
        const fg = this._fieldForcesBGs;
        const ff = this._fieldForces;

        if (this._higgsEnabled) {
            const p = encoder.beginComputePass({ label: 'applyHiggsForces' });
            p.setPipeline(ff.applyHiggsForces);
            p.setBindGroup(0, fg.g0);
            p.setBindGroup(1, fg.g1);
            p.setBindGroup(2, fg.g2);
            p.setBindGroup(3, fg.g3);
            p.dispatchWorkgroups(workgroups);
            p.end();
        }

        if (this._axionEnabled) {
            const p = encoder.beginComputePass({ label: 'applyAxionForces' });
            p.setPipeline(ff.applyAxionForces);
            p.setBindGroup(0, fg.g0);
            p.setBindGroup(1, fg.g1);
            p.setBindGroup(2, fg.g2);
            p.setBindGroup(3, fg.g3);
            p.dispatchWorkgroups(workgroups);
            p.end();
        }
    }

    /**
     * Dispatch particle-field gravity (O(N × GRID²)).
     * Force from scalar field energy density onto particles.
     * Dispatched once per active field when gravity is enabled.
     */
    _dispatchFieldParticleGrav(encoder) {
        if (!this._fieldParticleGrav || !this._gravityEnabled || this.aliveCount === 0) return;
        if (!this._higgsEnabled && !this._axionEnabled) return;

        // Create uniform buffer once
        if (!this._fieldParticleGravUniform) {
            this._fieldParticleGravUniform = this.device.createBuffer({
                label: 'fieldParticleGrav_uniform',
                size: 16, // FGUniforms: domainW, domainH, aliveCount, _pad
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        }

        // Write uniforms (FGUniforms struct)
        _fgUniformF32[0] = this.domainW;
        _fgUniformF32[1] = this.domainH;
        _fgUniformU32[2] = this.aliveCount;
        _fgUniformU32[3] = 0;
        this.device.queue.writeBuffer(this._fieldParticleGravUniform, 0, _fgUniformData, 0, 16);

        const pg = this._fieldParticleGrav;
        const workgroups = Math.ceil(this.aliveCount / 256);
        const b = this.buffers;

        const dispatchForField = (which) => {
            const fb = which === 'higgs' ? this._higgsBuffers : this._axionBuffers;
            if (!fb) return;

            const bgKey = which + '_pg';
            if (!this._fieldParticleGravBGs[bgKey]) {
                this._fieldParticleGravBGs[bgKey] = {
                    g0: this.device.createBindGroup({
                        label: `fieldParticleGrav_g0_${which}`,
                        layout: pg.bindGroupLayouts[0],
                        entries: [
                            { binding: 0, resource: { buffer: this._fieldParticleGravUniform } },
                            { binding: 1, resource: { buffer: b.particleState } },
                            { binding: 2, resource: { buffer: b.allForces } },
                        ],
                    }),
                    g1: this.device.createBindGroup({
                        label: `fieldParticleGrav_g1_${which}`,
                        layout: pg.bindGroupLayouts[1],
                        entries: [
                            { binding: 0, resource: { buffer: fb.sgGradX } },
                            { binding: 1, resource: { buffer: fb.sgGradY } },
                        ],
                    }),
                };
            }

            const bgs = this._fieldParticleGravBGs[bgKey];
            const p = encoder.beginComputePass({ label: `fieldParticleGrav_${which}` });
            p.setPipeline(pg.pipeline);
            p.setBindGroup(0, bgs.g0);
            p.setBindGroup(1, bgs.g1);
            p.dispatchWorkgroups(workgroups);
            p.end();
        };

        if (this._higgsEnabled) dispatchForField('higgs');
        if (this._axionEnabled) dispatchForField('axion');
    }

    /**
     * Dispatch field excitation deposits (Pass 19).
     * Deposits Gaussian wave packets from merge events into active field(s).
     */
    _dispatchFieldExcitations(encoder) {
        if (!this._fieldExcitation) return;
        if (!this._higgsEnabled && !this._axionEnabled) return;

        // Excitation events come from merge results (already on GPU in mergeResultBuffer)
        const mergeCount = this._pendingMergeEvents.length;
        if (mergeCount === 0) return;

        if (!this._excitationBuffers) {
            this._excitationBuffers = createExcitationBuffers(this.device);
        }

        // Upload excitation events from pending merge events
        const maxEvents = 64;
        const eventCount = Math.min(mergeCount, maxEvents);
        for (let i = 0; i < eventCount; i++) {
            const me = this._pendingMergeEvents[i];
            if (me.type !== 'merge') continue;
            _excitationEventData[i * 4] = me.x;
            _excitationEventData[i * 4 + 1] = me.y;
            _excitationEventData[i * 4 + 2] = me.energy;
            _excitationEventData[i * 4 + 3] = 0; // padding
        }
        this.device.queue.writeBuffer(this._excitationBuffers.events, 0, _excitationEventData, 0, eventCount * 4);
        _excitationCountData[0] = eventCount;
        this.device.queue.writeBuffer(this._excitationBuffers.counter, 0, _excitationCountData);

        const gridWG = Math.ceil(FIELD_GRID_RES / 8);
        const exc = this._fieldExcitation;

        const dispatchForField = (which) => {
            const fb = which === 'higgs' ? this._higgsBuffers : this._axionBuffers;
            if (!fb) return;

            if (!this._fieldExcitationBGs[which]) {
                const uBuf = which === 'higgs' ? this._higgsUniformBuffer : this._axionUniformBuffer;
                this._fieldExcitationBGs[which] = this.device.createBindGroup({
                    label: `fieldExcitation_${which}`,
                    layout: exc.bindGroupLayouts[0],
                    entries: [
                        { binding: 0, resource: { buffer: fb.fieldDot } },
                        { binding: 1, resource: { buffer: this._excitationBuffers.events } },
                        { binding: 2, resource: { buffer: uBuf } },
                        { binding: 3, resource: { buffer: this._excitationBuffers.counter } },
                    ],
                });
            }

            const p = encoder.beginComputePass({ label: `depositExcitations_${which}` });
            p.setPipeline(exc.pipeline);
            p.setBindGroup(0, this._fieldExcitationBGs[which]);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        };

        if (this._higgsEnabled) dispatchForField('higgs');
        if (this._axionEnabled) dispatchForField('axion');
    }

    /**
     * Dispatch cosmological expansion (Pass 13).
     */
    _dispatchExpansion(encoder, dt) {
        if (!this._expansionEnabled || !this._expansionPipeline) return;
        if (this.aliveCount === 0) return;

        if (!this._expansionUniformBuffer) {
            this._expansionUniformBuffer = this.device.createBuffer({
                label: 'expansionUniforms',
                size: 32, // 8 floats
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        }

        // Write ExpansionUniforms (reuse pre-allocated field uniform buffer)
        _fieldUniformF32[0] = this._hubbleParam;
        _fieldUniformF32[1] = dt;
        _fieldUniformF32[2] = this.domainW * 0.5;
        _fieldUniformF32[3] = this.domainH * 0.5;
        _fieldUniformU32[4] = this.aliveCount;
        this.device.queue.writeBuffer(this._expansionUniformBuffer, 0, _fieldUniformData, 0, 32);

        if (!this._expansionBG) {
            const b = this.buffers;
            this._expansionBG = this.device.createBindGroup({
                label: 'expansion_g0',
                layout: this._expansionPipeline.bindGroupLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: b.particleState } },
                    { binding: 1, resource: { buffer: this._expansionUniformBuffer } },
                ],
            });
        }

        const workgroups = Math.ceil(this.aliveCount / 256);
        const p = encoder.beginComputePass({ label: 'expansion' });
        p.setPipeline(this._expansionPipeline.pipeline);
        p.setBindGroup(0, this._expansionBG);
        p.dispatchWorkgroups(workgroups);
        p.end();
    }

    /**
     * Dispatch disintegration check (Pass 20).
     */
    _dispatchDisintegration(encoder) {
        if (!this._disintegrationEnabled || !this._disintPipeline) return;
        if (this.aliveCount === 0) return;

        if (!this._disintBuffers) {
            this._disintBuffers = createDisintegrationBuffers(this.device);
        }

        if (!this._disintUniformBuffer) {
            this._disintUniformBuffer = this.device.createBuffer({
                label: 'disintUniforms',
                size: 48, // 12 floats/u32
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        }

        // Write DisintUniforms (pre-allocated buffers, no per-frame GC)
        _disintUniformF32[0] = this._blackHoleEnabled ? 16 : 64; // softeningSq
        _disintUniformF32[1] = this.domainW;
        _disintUniformF32[2] = this.domainH;
        _disintUniformF32[3] = 0.3;   // tidalStrength
        _disintUniformF32[4] = 0.9;   // rocheThreshold
        _disintUniformF32[5] = 0.01;  // rocheTransferRate
        _disintUniformF32[6] = 0.01;  // minMass
        _disintUniformU32[7] = 4;     // spawnCount
        _disintUniformU32[8] = this.aliveCount;
        _disintUniformU32[9] = this.boundaryMode === BOUND_LOOP ? 1 : 0;
        _disintUniformU32[10] = this.topologyMode;
        this.device.queue.writeBuffer(this._disintUniformBuffer, 0, _disintUniformData);

        // Reset event counter
        encoder.clearBuffer(this._disintBuffers.counter, 0, 4);

        if (!this._disintBGs) {
            const b = this.buffers;
            const g0 = this.device.createBindGroup({
                label: 'disint_g0',
                layout: this._disintPipeline.bindGroupLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: b.particleState } },
                    { binding: 1, resource: { buffer: b.particleAux } },
                    { binding: 2, resource: { buffer: b.derived } },
                ],
            });
            const g1 = this.device.createBindGroup({
                label: 'disint_g1',
                layout: this._disintPipeline.bindGroupLayouts[1],
                entries: [
                    { binding: 0, resource: { buffer: this._disintBuffers.events } },
                    { binding: 1, resource: { buffer: this._disintBuffers.counter } },
                    { binding: 2, resource: { buffer: this._disintUniformBuffer } },
                ],
            });
            this._disintBGs = [g0, g1];
        }

        const workgroups = Math.ceil(this.aliveCount / 256);
        const p = encoder.beginComputePass({ label: 'disintegration' });
        p.setPipeline(this._disintPipeline.pipeline);
        p.setBindGroup(0, this._disintBGs[0]);
        p.setBindGroup(1, this._disintBGs[1]);
        p.dispatchWorkgroups(workgroups);
        p.end();
    }

    /**
     * Dispatch pair production check (Pass 22).
     */
    _dispatchPairProduction(encoder) {
        if (this._blackHoleEnabled) return;
        if (!this._pairProdPipeline) return;

        if (!this._pairProdBuffers) {
            this._pairProdBuffers = createPairProductionBuffers(this.device);
        }

        if (!this._pairProdUniformBuffer) {
            this._pairProdUniformBuffer = this.device.createBuffer({
                label: 'pairProdUniforms',
                size: 48, // 12 floats/u32
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        }

        // Write PairProdUniforms (pre-allocated buffers, no per-frame GC)
        _pairProdUniformF32[0] = 0.5;     // minEnergy
        _pairProdUniformF32[1] = 8.0;     // proximity
        _pairProdUniformF32[2] = 0.005;   // probability
        _pairProdUniformF32[3] = 64.0;    // minAge (time units, matches CPU PAIR_PROD_MIN_AGE)
        _pairProdUniformU32[4] = 32;      // maxParticles (PAIR_PROD_MAX_PARTICLES)
        _pairProdUniformU32[5] = this.aliveCount;
        _pairProdUniformU32[6] = GPU_MAX_PHOTONS;
        _pairProdUniformU32[7] = this._blackHoleEnabled ? 1 : 0;
        _pairProdUniformF32[8] = this.simTime;
        this.device.queue.writeBuffer(this._pairProdUniformBuffer, 0, _pairProdUniformData);

        // Reset pair counter
        encoder.clearBuffer(this._pairProdBuffers.counter, 0, 4);

        if (!this._pairProdBGs) {
            const b = this.buffers;
            const g0 = this.device.createBindGroup({
                label: 'pairProd_g0',
                layout: this._pairProdPipeline.bindGroupLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: b.photonPool } },
                    { binding: 1, resource: { buffer: b.phCount } },
                ],
            });
            const g1 = this.device.createBindGroup({
                label: 'pairProd_g1',
                layout: this._pairProdPipeline.bindGroupLayouts[1],
                entries: [
                    { binding: 0, resource: { buffer: b.particleState } },
                ],
            });
            const g2 = this.device.createBindGroup({
                label: 'pairProd_g2',
                layout: this._pairProdPipeline.bindGroupLayouts[2],
                entries: [
                    { binding: 0, resource: { buffer: this._pairProdBuffers.events } },
                    { binding: 1, resource: { buffer: this._pairProdBuffers.counter } },
                    { binding: 2, resource: { buffer: this._pairProdUniformBuffer } },
                ],
            });
            this._pairProdBGs = [g0, g1, g2];
        }

        const workgroups = Math.ceil(GPU_MAX_PHOTONS / 256);
        const p = encoder.beginComputePass({ label: 'pairProduction' });
        p.setPipeline(this._pairProdPipeline.pipeline);
        p.setBindGroup(0, this._pairProdBGs[0]);
        p.setBindGroup(1, this._pairProdBGs[1]);
        p.setBindGroup(2, this._pairProdBGs[2]);
        p.dispatchWorkgroups(workgroups);
        p.end();
    }

    /**
     * Dispatch heatmap compute pass (runs once every HEATMAP_INTERVAL frames).
     * @param {Object} camera - camera state for viewport bounds
     */
    dispatchHeatmap(encoder, camera) {
        if (!this._heatmapEnabled || !this._heatmapPipelines) return;
        if (this.aliveCount === 0) return;

        if (!this._heatmapBuffers) {
            this._heatmapBuffers = createHeatmapBuffers(this.device, GPU_HEATMAP_GRID);
        }

        if (!this._heatmapUniformBuffer) {
            this._heatmapUniformBuffer = this.device.createBuffer({
                label: 'heatmapUniforms',
                size: 96, // HeatmapUniforms struct size (22 fields, padded)
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        }

        // Compute viewport bounds from camera — use logical pixels to match
        // render shader (canvasW/canvasH = canvas.width = window.innerWidth, no DPR)
        const canvasW = camera?.viewportW || 800;
        const canvasH = camera?.viewportH || 600;
        const halfW = canvasW / (2 * (camera?.zoom || 16));
        const halfH = canvasH / (2 * (camera?.zoom || 16));
        const viewLeft = (camera?.x || 0) - halfW;
        const viewTop = (camera?.y || 0) - halfH;
        const cellW = (2 * halfW) / GPU_HEATMAP_GRID;
        const cellH = (2 * halfH) / GPU_HEATMAP_GRID;

        // Write HeatmapUniforms
        // Write HeatmapUniforms (pre-allocated buffers, no per-frame GC)
        _heatmapUniformF32[0] = viewLeft;
        _heatmapUniformF32[1] = viewTop;
        _heatmapUniformF32[2] = cellW;
        _heatmapUniformF32[3] = cellH;
        _heatmapUniformF32[4] = this._blackHoleEnabled ? 16 : 64; // softeningSq
        _heatmapUniformF32[5] = this._yukawaCoupling;
        _heatmapUniformF32[6] = this._yukawaMu;
        _heatmapUniformF32[7] = this.simTime;
        _heatmapUniformF32[8] = this.domainW;
        _heatmapUniformF32[9] = this.domainH;
        const hm = this._heatmapMode;
        _heatmapUniformU32[10] = (this._gravityEnabled && (hm === 'all' || hm === 'gravity')) ? 1 : 0;
        _heatmapUniformU32[11] = (this._coulombEnabled && (hm === 'all' || hm === 'electric')) ? 1 : 0;
        _heatmapUniformU32[12] = (this._yukawaEnabled && (hm === 'all' || hm === 'yukawa')) ? 1 : 0;
        _heatmapUniformU32[13] = (this._relativityEnabled && this.buffers.historyAllocated) ? 1 : 0; // useDelay
        _heatmapUniformU32[14] = this.boundaryMode === BOUND_LOOP ? 1 : 0;
        _heatmapUniformU32[15] = this.topologyMode;
        _heatmapUniformU32[16] = this.aliveCount;
        _heatmapUniformU32[17] = 0; // _padDead (unused — dead particles found via FLAG_RETIRED scan)
        this.device.queue.writeBuffer(this._heatmapUniformBuffer, 0, _heatmapUniformData);

        if (!this._heatmapBGs) {
            const b = this.buffers;
            const hm = this._heatmapPipelines;
            const hmBuf = this._heatmapBuffers;
            this._heatmapBGs = {
                g0: this.device.createBindGroup({
                    label: 'heatmap_g0',
                    layout: hm.heatmapLayouts[0],
                    entries: [
                        { binding: 0, resource: { buffer: b.particleState } },
                        { binding: 1, resource: { buffer: b.particleAux } },
                    ],
                }),
                g1: this.device.createBindGroup({
                    label: 'heatmap_g1',
                    layout: hm.heatmapLayouts[1],
                    entries: [
                        { binding: 0, resource: { buffer: hmBuf.gravPotential } },
                        { binding: 1, resource: { buffer: hmBuf.elecPotential } },
                        { binding: 2, resource: { buffer: hmBuf.yukawaPotential } },
                        { binding: 3, resource: { buffer: this._heatmapUniformBuffer } },
                    ],
                }),
                g2: null, // history bind group (created lazily when history buffers allocated)
            };
        }

        // Ensure history bind group exists (dummy when history not yet allocated)
        if (!this._heatmapBGs.g2) {
            const hm = this._heatmapPipelines;
            if (this.buffers.historyAllocated) {
                const b = this.buffers;
                this._heatmapBGs.g2 = this.device.createBindGroup({
                    label: 'heatmap_g2_history',
                    layout: hm.heatmapLayouts[2],
                    entries: [
                        { binding: 0, resource: { buffer: b.histData } },
                        { binding: 1, resource: { buffer: b.histMeta } },
                    ],
                });
            } else {
                this._heatmapBGs.g2 = this.device.createBindGroup({
                    label: 'heatmap_g2_dummy',
                    layout: hm.heatmapLayouts[2],
                    entries: [
                        { binding: 0, resource: { buffer: this._dummyHistBuf } },
                        { binding: 1, resource: { buffer: this._dummyHistBuf } },
                    ],
                });
                this._heatmapBGs._g2IsDummy = true;
            }
        }

        // Upgrade from dummy to real history bind group when buffers become available
        if (this._heatmapBGs._g2IsDummy && this.buffers.historyAllocated) {
            const b = this.buffers;
            const hm = this._heatmapPipelines;
            this._heatmapBGs.g2 = this.device.createBindGroup({
                label: 'heatmap_g2_history',
                layout: hm.heatmapLayouts[2],
                entries: [
                    { binding: 0, resource: { buffer: b.histData } },
                    { binding: 1, resource: { buffer: b.histMeta } },
                ],
            });
            this._heatmapBGs._g2IsDummy = false;
        }

        const gridWG = Math.ceil(GPU_HEATMAP_GRID / 8);

        // Compute heatmap
        {
            const p = encoder.beginComputePass({ label: 'computeHeatmap' });
            p.setPipeline(this._heatmapPipelines.computeHeatmap);
            p.setBindGroup(0, this._heatmapBGs.g0);
            p.setBindGroup(1, this._heatmapBGs.g1);
            p.setBindGroup(2, this._heatmapBGs.g2);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }

        // Blur each active channel
        const channels = ['gravPotential', 'elecPotential', 'yukawaPotential'];
        const channelToggles = [this._gravityEnabled, this._coulombEnabled, this._yukawaEnabled];
        const hmBuf = this._heatmapBuffers;

        for (let c = 0; c < 3; c++) {
            if (!channelToggles[c]) continue;
            const chName = channels[c];
            if (!this._heatmapBlurBGs[chName]) {
                this._heatmapBlurBGs[chName] = this.device.createBindGroup({
                    label: `heatmapBlur_${chName}`,
                    layout: this._heatmapPipelines.blurLayouts[0],
                    entries: [
                        { binding: 0, resource: { buffer: hmBuf[chName] } },
                        { binding: 1, resource: { buffer: hmBuf.blurTemp } },
                    ],
                });
            }
            // Horizontal blur
            {
                const p = encoder.beginComputePass({ label: `blurH_${chName}` });
                p.setPipeline(this._heatmapPipelines.blurHorizontal);
                p.setBindGroup(0, this._heatmapBlurBGs[chName]);
                p.dispatchWorkgroups(gridWG, gridWG);
                p.end();
            }
            // Vertical blur
            {
                const p = encoder.beginComputePass({ label: `blurV_${chName}` });
                p.setPipeline(this._heatmapPipelines.blurVertical);
                p.setBindGroup(0, this._heatmapBlurBGs[chName]);
                p.dispatchWorkgroups(gridWG, gridWG);
                p.end();
            }
        }
    }

    /** Expose field buffers for renderer overlay drawing */
    getFieldBuffers(which) {
        return which === 'higgs' ? this._higgsBuffers : this._axionBuffers;
    }

    /** Expose heatmap buffers for renderer overlay drawing */
    getHeatmapBuffers() { return this._heatmapBuffers; }

    /** Expose trail buffers for renderer trail drawing */
    getTrailBuffers() { return this._trailBuffers; }

    /** Set trails enabled state and lazily allocate trail buffers */
    setTrailsEnabled(enabled) {
        this._trailsEnabled = enabled;
        if (enabled && !this._trailBuffers) {
            this._trailBuffers = createTrailBuffers(this.device, this.buffers.maxParticles);
            // Create bind group for trail recording
            if (this._trailRecordPipeline) {
                const tb = this._trailBuffers;
                this._trailRecordBindGroup = this.device.createBindGroup({
                    label: 'trailRecord',
                    layout: this._trailRecordLayout,
                    entries: [
                        { binding: 0, resource: { buffer: this.buffers.particleState } },
                        { binding: 1, resource: { buffer: tb.trailX } },
                        { binding: 2, resource: { buffer: tb.trailY } },
                        { binding: 3, resource: { buffer: tb.trailWriteIdx } },
                        { binding: 4, resource: { buffer: tb.trailCount } },
                    ],
                });
            }
        }
    }

    /**
     * Run one frame: adaptive substepping with full Phase 2 force computation + Boris integrator.
     * When paused, the caller does not call update() — no compute dispatches are issued.
     * @param {number} dt - Total simulation time to advance this frame
     */
    update(dt) {
        if (!this._ready || this.aliveCount === 0 || dt <= 0) return;

        // Adaptive substepping: use maxAccel from previous frame
        const softening = this._blackHoleEnabled ? 4 : 8;
        let dtSafe = this._maxAccel > 1e-9
            ? Math.sqrt(softening / this._maxAccel)
            : dt;
        // Cyclotron limit would need Bz readback — for Phase 2, use conservative estimate
        const maxSubsteps = 32;
        const numSubsteps = Math.min(Math.ceil(dt / dtSafe), maxSubsteps);
        const dtSub = dt / numSubsteps;

        for (let step = 0; step < numSubsteps; step++) {
            this.simTime += dtSub;
            this._dispatchSubstep(dtSub);
        }

        this._frameCount++;

        // Post-substep passes (once per frame)
        {
            const encoder = this.device.createCommandEncoder({ label: 'post-substep' });

            // Heatmap compute (every HEATMAP_INTERVAL=4 frames)
            this._heatmapFrame++;
            if (this._heatmapEnabled && this._heatmapFrame >= 4) {
                this._heatmapFrame = 0;
                this.dispatchHeatmap(encoder, this._lastCamera);
            }

            // Pion decay: dispatched once per frame. Probability scaled by dt/PHYSICS_DT
            // in the shader to match CPU (which checks once per PHYSICS_DT tick).
            // Write total dt (not substep dt) so the shader can compute tick count.
            _postSubDt[0] = dt;
            this.device.queue.writeBuffer(this.uniformBuffer, 0, _postSubDt);
            this._dispatchPionDecay(encoder);

            // Boson interaction (if enabled): build boson tree + particle<-boson + boson<->boson + pion Coulomb + annihilation
            this._dispatchBosonInteraction(encoder);

            // Dead particle garbage collection
            this._dispatchDeadGC(encoder);

            // Record signal delay history (every HISTORY_STRIDE physics steps)
            // CPU increments _histStride once per Physics.update() call (= one PHYSICS_DT).
            // GPU update() processes multiple PHYSICS_DT steps per frame, so increment
            // by the number of physics steps to match CPU recording frequency.
            // Clears FLAG_REBORN after resetting stale history for recycled slots
            {
                const physicsSteps = Math.round(dt / PHYSICS_DT);
                this._histStride += physicsSteps;
            }
            if (this._relativityEnabled && this._histStride >= HISTORY_STRIDE) {
                this._histStride -= HISTORY_STRIDE;
                this._dispatchRecordHistory(encoder);
            }

            // Quadrupole radiation (once per frame, after history recording — matches CPU order)
            // Uses PHYSICS_DT constant in shader (not u.dt which holds dtSub from last substep)
            this._dispatchQuadrupole(encoder);

            // Update particle colors from charge/mass/antimatter state
            if (this._updateColorsPipeline && this.aliveCount > 0) {
                _colorUniformData[0] = this._blackHoleEnabled ? 1 : 0;
                _colorUniformData[1] = document.documentElement.dataset.theme === 'dark' ? 1 : 0;
                this.device.queue.writeBuffer(this._colorUniformBuffer, 0, _colorUniformData);
                const p = encoder.beginComputePass({ label: 'updateColors' });
                p.setPipeline(this._updateColorsPipeline);
                p.setBindGroup(0, this._updateColorsBindGroup);
                p.dispatchWorkgroups(Math.ceil(this.aliveCount / 64));
                p.end();
            }

            // Record trail positions (every frame when trails enabled)
            if (this._trailRecordPipeline && this._trailBuffers && this._trailsEnabled && this.aliveCount > 0) {
                const p = encoder.beginComputePass({ label: 'trailRecord' });
                p.setPipeline(this._trailRecordPipeline);
                p.setBindGroup(0, this._trailRecordBindGroup);
                p.dispatchWorkgroups(Math.ceil(this.aliveCount / 64));
                p.end();
            }

            this.device.queue.submit([encoder.finish()]);
        }

        // Readback maxAccel for next frame (non-blocking)
        this._readbackMaxAccel();

        // Readback ghost count for next frame (non-blocking)
        this._readbackGhostCount();

        // Readback merge results for photon bursts / field excitations (non-blocking)
        this._readbackMergeResults();

        // Readback free stack for slot reuse (non-blocking)
        this._readbackFreeStack();
    }

    /**
     * Dispatch a single substep: all compute passes in sequence.
     * @param {number} dtSub - Substep timestep
     */
    _dispatchSubstep(dtSub) {
        // Upload uniforms (now includes Phase 2 parameters)
        writeUniforms(this.device, this.uniformBuffer, {
            dt: dtSub,
            simTime: this.simTime,
            domainW: this.domainW,
            domainH: this.domainH,
            speedScale: 1,
            softening: this._blackHoleEnabled ? 4 : 8,
            softeningSq: this._blackHoleEnabled ? 16 : 64,
            toggles0: this._toggles0,
            toggles1: this._toggles1,
            yukawaCoupling: this._yukawaCoupling,
            yukawaMu: this._yukawaMu,
            higgsMass: this._higgsMass,
            axionMass: this._axionMass,
            boundaryMode: this.boundaryMode,
            topologyMode: this.topologyMode,
            collisionMode: this._collisionMode,
            maxParticles: this.buffers.maxParticles,
            aliveCount: this.aliveCount,
            extGravity: this._extGravity,
            extGravityAngle: this._extGravityAngle,
            extElectric: this._extElectric,
            extElectricAngle: this._extElectricAngle,
            extBz: this._extBz,
            bounceFriction: this._bounceFriction,
            axionCoupling: this._axionCoupling,
            higgsCoupling: this._higgsCoupling,
            particleCount: this.aliveCount + this._ghostCount,
            bhTheta: 0.5,
            frameCount: this._frameCount,
        });

        const workgroups = Math.ceil(this.aliveCount / 64);
        if (workgroups === 0) return; // no particles, skip dispatch
        const p2 = this._phase2;

        let encoder = this.device.createCommandEncoder({ label: 'physics-substep' });

        // Pass 0: ghost generation (Phase 3 — before tree build)
        this._dispatchGhostGen(encoder);

        // Pass 0b: tree build (Phase 3 — after ghost gen, before force computation)
        this._dispatchTreeBuild(encoder);

        // Clear maxAccel for adaptive substepping (reset before force computation)
        encoder.clearBuffer(this.buffers.maxAccelBuffer, 0, 4);

        // Pass 1: resetForces (DMA clear is faster than compute dispatch for zeroing)
        encoder.clearBuffer(this.buffers.allForces, 0, this.aliveCount * 160);

        // Pass 2: cacheDerived
        const pass2 = encoder.beginComputePass({ label: 'cacheDerived' });
        pass2.setPipeline(p2.cacheDerived.pipeline);
        pass2.setBindGroup(0, this._bg_cacheDerived);
        pass2.dispatchWorkgroups(workgroups);
        pass2.end();

        // Pass 5: force computation — BH tree walk or O(N^2) pairwise
        if (this._barnesHutEnabled) {
            // Tree walk force computation (O(N log N))
            this._ensureTreeForceHistoryBG();
            const passTree = encoder.beginComputePass({ label: 'treeForce' });
            passTree.setPipeline(this._treeForcePipeline);
            passTree.setBindGroup(0, this._treeForceGroup0);
            passTree.setBindGroup(1, this._treeForceGroup1);
            passTree.setBindGroup(2, this._treeForceGroup2);
            passTree.setBindGroup(3, this._treeForceGroup3);
            passTree.dispatchWorkgroups(workgroups);
            passTree.end();
        } else {
            // Pairwise force computation (O(N^2) tiled)
            this._ensurePairForceHistoryBG();
            const pass5 = encoder.beginComputePass({ label: 'pairForce' });
            pass5.setPipeline(p2.pairForce.pipeline);
            pass5.setBindGroup(0, this._bg_pairForce0);
            pass5.setBindGroup(1, this._bg_pairForce1);
            pass5.setBindGroup(2, this._bg_pairForce2);
            pass5.setBindGroup(3, this._bg_pairForce3);
            pass5.dispatchWorkgroups(workgroups);
            pass5.end();
        }

        // Pass 5b: externalFields (skip when all external fields are zero)
        if (this._extGravity !== 0 || this._extElectric !== 0 || this._extBz !== 0) {
            const pass5b = encoder.beginComputePass({ label: 'externalFields' });
            pass5b.setPipeline(p2.externalFields.pipeline);
            pass5b.setBindGroup(0, this._bg_extFields);
            pass5b.dispatchWorkgroups(workgroups);
            pass5b.end();
        }

        // Pass 5c: scalar field forces (Higgs gradient + mass mod, Axion gradient + axMod/yukMod)
        // Must run BEFORE Boris so gradient forces are included in totalForce for half-kicks.
        // Uses previous substep's field gradients (computed at end of field evolve).
        if (this._fieldDeposit && (this._higgsEnabled || this._axionEnabled)) {
            this._writeFieldUniforms(dtSub);
        }
        this._dispatchFieldForces(encoder);

        // Pass 5c2: particle-field gravity (O(N×GRID²), if gravity enabled)
        // Uses energyDensity computed by previous substep's field self-gravity pass.
        this._dispatchFieldParticleGrav(encoder);

        // Pass 5d: save 1PN forces for velocity-Verlet correction (before Boris)
        if (this._onePNEnabled) {
            const passSave = encoder.beginComputePass({ label: 'saveF1pn' });
            passSave.setPipeline(p2.saveF1pn.pipeline);
            passSave.setBindGroup(0, this._bg_saveF1pn);
            passSave.dispatchWorkgroups(workgroups);
            passSave.end();
        }

        // Pass 6-8 fused: halfKick1 + borisRotate + halfKick2 in one dispatch.
        // Eliminates 2 barrier syncs + 4 redundant global memory loads/stores per particle.
        const passFused = encoder.beginComputePass({ label: 'borisFused' });
        passFused.setPipeline(p2.borisFused.pipeline);
        passFused.setBindGroup(0, this._bg_borisFused);
        passFused.dispatchWorkgroups(workgroups);
        passFused.end();

        // Pass 9: spinOrbit
        const pass9 = encoder.beginComputePass({ label: 'spinOrbit' });
        pass9.setPipeline(p2.spinOrbit.pipeline);
        pass9.setBindGroup(0, this._bg_spinOrbit);
        pass9.dispatchWorkgroups(workgroups);
        pass9.end();

        // Pass 10: applyTorques
        const pass10 = encoder.beginComputePass({ label: 'applyTorques' });
        pass10.setPipeline(p2.applyTorques.pipeline);
        pass10.setBindGroup(0, this._bg_torques);
        pass10.dispatchWorkgroups(workgroups);
        pass10.end();

        // ── Radiation + drift + advanced passes (same encoder) ──

        // Pass 11: radiation reaction (Larmor, Hawking, pion emission) — BEFORE drift
        this._dispatchRadiation(encoder);

        // Pass 12: borisDrift
        const pass12 = encoder.beginComputePass({ label: 'borisDrift' });
        pass12.setPipeline(p2.borisDrift.pipeline);
        pass12.setBindGroup(0, this._bg_drift);
        pass12.dispatchWorkgroups(workgroups);
        pass12.end();

        // Pass 13: cosmological expansion
        this._dispatchExpansion(encoder, dtSub);

        // Pass 14: 1PN velocity-Verlet correction
        this._dispatch1PNVV(encoder);

        // Pass 15: scalar field evolution (Higgs, Axion)
        // Use per-field uniform buffers to avoid encoder split when both fields active.
        // Each field gets its own uniform buffer with the correct currentFieldType baked in.
        if (this._fieldDeposit) {
            if (!(this._higgsEnabled || this._axionEnabled)) this._writeFieldUniforms(dtSub);
            if (this._higgsEnabled) {
                this._writeFieldUniforms(dtSub, 0); // currentFieldType=0 (Higgs)
                this._dispatchFieldEvolve(encoder, 'higgs', dtSub);
            }
            if (this._axionEnabled) {
                this._writeFieldUniforms(dtSub, 1); // currentFieldType=1 (Axion)
                this._dispatchFieldEvolve(encoder, 'axion', dtSub);
            }
        }

        // (Field forces already dispatched at Pass 5c, before Boris)

        // Pass 17-18: collision detection + resolution
        this._dispatchCollisions(encoder);

        // Pass 19: field excitations from merge events
        this._dispatchFieldExcitations(encoder);

        // Pass 20: disintegration check
        this._dispatchDisintegration(encoder);

        // Pass 21: boson update (photon/pion drift, absorption, decay)
        this._dispatchBosonUpdate(encoder);

        // Pass 22: pair production
        this._dispatchPairProduction(encoder);

        // Pass 24: boundary
        const passBoundary = encoder.beginComputePass({ label: 'boundary' });
        passBoundary.setPipeline(this._boundaryPipeline);
        passBoundary.setBindGroup(0, this._boundaryBindGroup);
        passBoundary.dispatchWorkgroups(workgroups);
        passBoundary.end();

        this.device.queue.submit([encoder.finish()]);
    }

    /**
     * Non-blocking readback of maxAccel from GPU for adaptive substepping.
     * Uses 1-frame latency: reads previous frame's max acceleration.
     */
    async _readbackMaxAccel() {
        if (this._maxAccelPending) return;
        this._maxAccelPending = true;

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.maxAccelBuffer, 0,
            this.buffers.maxAccelStaging, 0, 4);
        this.device.queue.submit([encoder.finish()]);

        await this.buffers.maxAccelStaging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(this.buffers.maxAccelStaging.getMappedRange().slice(0));
        this.buffers.maxAccelStaging.unmap();

        // NaN/Inf guard: corrupted GPU readback would break adaptive substepping
        const val = data[0];
        this._maxAccel = (val === val && val < 1e20) ? val : 0;
        this._maxAccelPending = false;
    }

    /**
     * Non-blocking readback of free stack from GPU for slot reuse.
     * Reads freeTop count first, then copies free indices to CPU pool.
     * Uses 1-frame latency.
     */
    async _readbackFreeStack() {
        if (this._freeTopPending) return;
        this._freeTopPending = true;
        const b = this.buffers;

        try {
            // Copy freeTop counter to staging
            const enc = this.device.createCommandEncoder();
            enc.copyBufferToBuffer(b.freeTop, 0, b.freeTopStaging, 0, 4);
            this.device.queue.submit([enc.finish()]);

            await b.freeTopStaging.mapAsync(GPUMapMode.READ);
            const topData = new Uint32Array(b.freeTopStaging.getMappedRange().slice(0));
            b.freeTopStaging.unmap();

            const count = topData[0];
            if (count > 0 && count <= MAX_PARTICLES) {
                // Copy free stack indices to staging
                const readBytes = count * 4;
                const enc2 = this.device.createCommandEncoder();
                enc2.copyBufferToBuffer(b.freeStack, 0, b.freeStackStaging, 0, readBytes);
                this.device.queue.submit([enc2.finish()]);

                await b.freeStackStaging.mapAsync(GPUMapMode.READ);
                const stackData = new Uint32Array(b.freeStackStaging.getMappedRange(0, readBytes).slice(0));
                b.freeStackStaging.unmap();

                // Transfer to CPU free pool (deduplicate against existing)
                for (let i = 0; i < stackData.length; i++) {
                    const slot = stackData[i];
                    if (slot < MAX_PARTICLES) this._cpuFreeSlots.push(slot);
                }

                // Reset GPU free stack (we've consumed all entries)
                this.device.queue.writeBuffer(b.freeTop, 0, _zeroU32);
            }
        } catch (e) {
            // Device lost or other error — don't block future readbacks
        }
        this._freeTopPending = false;
    }

    /** Store camera state for heatmap viewport calculation */
    setCamera(camera) {
        this._lastCamera = camera;
    }

    /**
     * Read all particle state from GPU and return a save-compatible JSON object.
     * This triggers an async readback. Only called on user action (save/download).
     * @param {Object} sim - The Simulation instance for camera/mode state
     * @returns {Promise<Object>} Save-compatible state object
     */
    async serialize(sim) {
        const count = this.aliveCount;
        const stateByteLen = count * PARTICLE_STATE_SIZE;

        // Create staging buffer for packed ParticleState readback
        const stagingState = this.device.createBuffer({
            size: stateByteLen,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const encoder = this.device.createCommandEncoder({ label: 'serialize-readback' });
        encoder.copyBufferToBuffer(this.buffers.particleState, 0, stagingState, 0, stateByteLen);
        this.device.queue.submit([encoder.finish()]);

        await stagingState.mapAsync(GPUMapMode.READ);
        const stateRaw = new ArrayBuffer(stateByteLen);
        new Uint8Array(stateRaw).set(new Uint8Array(stagingState.getMappedRange()));
        stagingState.unmap();
        stagingState.destroy();

        // Parse packed ParticleState structs (9 × 4 bytes each)
        const stateF32 = new Float32Array(stateRaw);
        const stateU32 = new Uint32Array(stateRaw);
        const STRIDE = PARTICLE_STATE_SIZE / 4; // 9 u32/f32 per particle

        const state = {
            version: 1,
            particles: [],
            toggles: {},
            settings: {
                collision: COL_NAMES[sim.collisionMode],
                boundary: BOUND_NAMES[sim.boundaryMode],
                topology: TOPO_NAMES[sim.topology],
                speed: sim.speedScale,
                friction: this._bounceFriction,
            },
            camera: {
                x: sim.camera.x,
                y: sim.camera.y,
                zoom: sim.camera.zoom,
            },
        };

        for (let i = 0; i < count; i++) {
            const base = i * STRIDE;
            const flags = stateU32[base + 8];
            if (!(flags & FLAG_ALIVE)) continue;
            state.particles.push({
                x: stateF32[base + 0],      // posX
                y: stateF32[base + 1],      // posY
                wx: stateF32[base + 2],     // velWX
                wy: stateF32[base + 3],     // velWY
                mass: stateF32[base + 4],   // mass
                charge: stateF32[base + 5], // charge
                angw: stateF32[base + 6],   // angW
                baseMass: stateF32[base + 7], // baseMass
                antimatter: !!(flags & FLAG_ANTIMATTER),
            });
        }

        // Map internal toggle bits back to boolean flags
        const t0 = this._toggles0;
        const t1 = this._toggles1;
        state.toggles.gravityEnabled = !!(t0 & GRAVITY_BIT);
        state.toggles.coulombEnabled = !!(t0 & COULOMB_BIT);
        state.toggles.magneticEnabled = !!(t0 & MAGNETIC_BIT);
        state.toggles.gravitomagEnabled = !!(t0 & GRAVITOMAG_BIT);
        state.toggles.onePNEnabled = !!(t0 & ONE_PN_BIT);
        state.toggles.relativityEnabled = !!(t0 & RELATIVITY_BIT);
        state.toggles.spinOrbitEnabled = !!(t0 & SPIN_ORBIT_BIT);
        state.toggles.radiationEnabled = !!(t0 & RADIATION_BIT);
        state.toggles.blackHoleEnabled = !!(t0 & BLACK_HOLE_BIT);
        state.toggles.disintegrationEnabled = !!(t0 & DISINTEGRATION_BIT);
        state.toggles.expansionEnabled = !!(t0 & EXPANSION_BIT);
        state.toggles.yukawaEnabled = !!(t0 & YUKAWA_BIT);
        state.toggles.higgsEnabled = !!(t0 & HIGGS_BIT);
        state.toggles.axionEnabled = !!(t0 & AXION_BIT);
        state.toggles.barnesHutEnabled = !!(t0 & BARNES_HUT_BIT);
        state.toggles.bosonInterEnabled = !!(t0 & BOSON_INTER_BIT);
        // field gravity follows gravity (no separate toggle)

        state.yukawaMu = this._yukawaMu;
        state.higgsMass = this._higgsMass;
        state.axionMass = this._axionMass;
        state.hubbleParam = this._hubbleParam;

        return state;
    }

    /**
     * Load a save state into GPU buffers.
     * @param {Object} state - JSON state from serialize() or CPU saveState()
     * @param {Object} sim - The Simulation instance
     * @returns {boolean} Success
     */
    deserialize(state, sim) {
        if (!state || state.version !== 1) return false;

        this.reset();

        // Upload particles to GPU using packed structs (reuse pre-allocated buffers)
        for (const pd of state.particles) {
            const idx = this.aliveCount;
            if (idx >= MAX_PARTICLES) break;

            const m = pd.mass || 1;
            const bm = pd.baseMass ?? m;
            const q = pd.charge || 0;
            const angw = pd.angw || 0;
            const flagBits = FLAG_ALIVE | (pd.antimatter ? FLAG_ANTIMATTER : 0);

            // Reuse pre-allocated addParticle buffers (zero-alloc per particle)
            _addParticleStateF32[0] = pd.x;    _addParticleStateF32[1] = pd.y;
            _addParticleStateF32[2] = pd.wx || 0;  _addParticleStateF32[3] = pd.wy || 0;
            _addParticleStateF32[4] = m;        _addParticleStateF32[5] = q;
            _addParticleStateF32[6] = angw;     _addParticleStateF32[7] = bm;
            _addParticleStateU32[8] = flagBits;
            this.device.queue.writeBuffer(this.buffers.particleState, idx * PARTICLE_STATE_SIZE, _addParticleStateData);

            _addParticleAuxF32[0] = Math.cbrt(m);
            _addParticleAuxU32[1] = idx;
            _addParticleAuxF32[2] = 3.4028235e38;
            _addParticleAuxF32[3] = 0;
            _addParticleAuxF32[4] = 0;
            this.device.queue.writeBuffer(this.buffers.particleAux, idx * PARTICLE_AUX_SIZE, _addParticleAuxData);

            _addParticleColorData[0] = 0xFF727E8A;
            this.device.queue.writeBuffer(this.buffers.color, idx * 4, _addParticleColorData);

            _addParticleModData[0] = 1.0; _addParticleModData[1] = 1.0;
            this.device.queue.writeBuffer(this.buffers.axYukMod, idx * 8, _addParticleModData);

            _addParticleRadData.fill(0);
            this.device.queue.writeBuffer(this.buffers.radiationState, idx * RADIATION_STATE_SIZE, _addParticleRadData);

            // Initialize history metadata for deserialized particles (creationTime = -Infinity)
            if (this.buffers.historyAllocated) {
                _addParticleMetaU32[0] = 0;  // writeIdx
                _addParticleMetaU32[1] = 0;  // count
                _addParticleMetaF32[2] = -Infinity; // creationTime — always existing
                _addParticleMetaU32[3] = 0;  // _pad
                this.device.queue.writeBuffer(
                    this.buffers.histMeta,
                    idx * HIST_META_STRIDE * 4,
                    _addParticleMetaBuf
                );
            }

            this.aliveCount++;
        }

        // Restore slider parameters from save state
        if (state.higgsMass !== undefined) this._higgsMass = state.higgsMass;
        if (state.axionMass !== undefined) this._axionMass = state.axionMass;
        if (state.yukawaMu !== undefined) this._yukawaMu = state.yukawaMu;
        if (state.hubbleParam !== undefined) this._hubbleParam = state.hubbleParam;

        this.syncUniforms();
        return true;
    }

    reset() {
        this.aliveCount = 0;
        this.simTime = 0;
        this._histStride = 0;
        this._frameCount = 0;
        this._heatmapFrame = 0;
        this._pendingMergeEvents = [];
        this._cpuFreeSlots = [];
        // Reset GPU free stack (reuse pre-allocated zero buffer)
        if (this.buffers.freeTop) {
            this.device.queue.writeBuffer(this.buffers.freeTop, 0, _zeroU32);
        }
        this.resetFields();
        // Clear boson pools (reuse pre-allocated zero buffer)
        if (this.buffers.phCount) {
            this.device.queue.writeBuffer(this.buffers.phCount, 0, _zeroU32);
        }
        if (this.buffers.piCount) {
            this.device.queue.writeBuffer(this.buffers.piCount, 0, _zeroU32);
        }
        // Clear trail ring buffers (zero-fill via encoder to avoid allocation)
        if (this._trailBuffers) {
            const tb = this._trailBuffers;
            const enc = this.device.createCommandEncoder({ label: 'clearTrails' });
            enc.clearBuffer(tb.trailWriteIdx);
            enc.clearBuffer(tb.trailCount);
            this.device.queue.submit([enc.finish()]);
        }
    }

    /**
     * Sync all toggle/slider state to the GPU uniforms buffer.
     * Called once per frame before compute dispatch.
     */
    syncUniforms() {
        writeUniforms(this.device, this.uniformBuffer, {
            dt: 0,
            simTime: this.simTime,
            domainW: this.domainW,
            domainH: this.domainH,
            aliveCount: this.aliveCount,
            toggles0: this._toggles0,
            toggles1: this._toggles1,
            boundaryMode: this.boundaryMode,
            topologyMode: this.topologyMode,
            collisionMode: this._collisionMode,
            bounceFriction: this._bounceFriction,
            yukawaCoupling: this._yukawaCoupling,
            yukawaMu: this._yukawaMu,
            higgsMass: this._higgsMass,
            axionMass: this._axionMass,
            higgsCoupling: this._higgsCoupling,
            axionCoupling: this._axionCoupling,
            extGravity: this._extGravity,
            extGravityAngle: this._extGravityAngle,
            extElectric: this._extElectric,
            extElectricAngle: this._extElectricAngle,
            extBz: this._extBz,
            maxParticles: this.buffers.maxParticles,
            particleCount: this.aliveCount + this._ghostCount,
            bhTheta: 0.5,
            frameCount: this._frameCount,
        });
    }

    /** Reset field buffers to vacuum on preset load */
    resetFields() {
        if (this._higgsBuffers) this._initFieldToVacuum('higgs');
        if (this._axionBuffers) this._initFieldToVacuum('axion');
    }

    /**
     * Queue a GPU hit test. Dispatches the hit test compute shader and starts
     * async readback. Result available via readHitResult() once ready.
     * Uses BH tree when available, falls back to O(N) linear scan.
     */
    hitTest(worldX, worldY) {
        if (!this._hitTestPipeline || !this._hitUniformBuffer) return;
        if (this._hitPending) return; // don't queue while readback is in flight

        // Write click coordinates + aliveCount to uniform
        _hitUniformF32[0] = worldX;
        _hitUniformF32[1] = worldY;
        _hitUniformU32[2] = this.aliveCount;
        _hitUniformU32[3] = 0;
        this.device.queue.writeBuffer(this._hitUniformBuffer, 0, _hitUniformData);

        // Dispatch single-thread compute
        const encoder = this.device.createCommandEncoder({ label: 'hitTest' });
        const pass = encoder.beginComputePass({ label: 'hitTest' });
        pass.setPipeline(this._hitTestPipeline);
        pass.setBindGroup(0, this._hitTestBindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();

        // Copy result to staging for readback
        encoder.copyBufferToBuffer(this._hitResultBuffer, 0, this._hitResultStaging, 0, 48);
        this.device.queue.submit([encoder.finish()]);

        // Async readback
        this._hitPending = true;
        this._hitResultStaging.mapAsync(GPUMapMode.READ).then(() => {
            const buf = this._hitResultStaging.getMappedRange().slice(0);
            this._hitResultStaging.unmap();
            this._hitStagingI32 = new Int32Array(buf);
            this._hitStagingF32 = new Float32Array(buf);
            this._hitResultReady = true;
            this._hitPending = false;
        }).catch(() => {
            this._hitPending = false;
        });
    }

    /**
     * Read the result of a previously queued GPU hit test.
     * Returns null if not ready yet, or an object:
     *   { index, mass, charge, radius, velX, velY, angVel, posX, posY }
     * index is -1 if no particle was hit.
     */
    readHitResult() {
        if (!this._hitResultReady || !this._hitStagingI32) return null;
        this._hitResultReady = false;
        const idx = this._hitStagingI32[0];
        if (idx < 0) return { index: -1 };
        const f = this._hitStagingF32;
        return {
            index: idx,
            mass: f[1],
            charge: f[2],
            radius: f[3],
            velX: f[4],
            velY: f[5],
            angVel: f[6],
            posX: f[7],
            posY: f[8],
        };
    }

    /** Rebuild group 1 bind group for stats shader (field buffers). Call when fields toggle. */
    _rebuildStatsFieldBindGroup() {
        if (!this._statsGroup1Layout) return;
        const dummy = this._statsDummyBuffer;
        const hb = this._higgsBuffers;
        const ab = this._axionBuffers;
        this._statsBindGroup1 = this.device.createBindGroup({
            label: 'computeStats_g1',
            layout: this._statsGroup1Layout,
            entries: [
                { binding: 0, resource: { buffer: hb ? hb.field : dummy } },
                { binding: 1, resource: { buffer: hb ? hb.fieldDot : dummy } },
                { binding: 2, resource: { buffer: ab ? ab.field : dummy } },
                { binding: 3, resource: { buffer: ab ? ab.fieldDot : dummy } },
            ],
        });
    }

    /**
     * Dispatch the stats compute shader and start async readback.
     * @param {number} selectedGpuIdx  GPU buffer index of selected particle, or -1
     */
    requestStats(selectedGpuIdx = -1) {
        if (!this._statsPipeline || !this._statsUniformBuffer) return;
        if (this._statsPending) return; // readback still in flight

        // Write uniforms (StatsUniforms: 48 bytes = 12 u32/f32)
        _statsUniformU32[0] = this.aliveCount;
        _statsUniformI32[1] = selectedGpuIdx;
        _statsUniformU32[2] = this._toggles0;
        _statsUniformF32[3] = this.domainW;
        _statsUniformF32[4] = this.domainH;
        _statsUniformF32[5] = this._yukawaMu;
        _statsUniformF32[6] = this._higgsMass;
        _statsUniformF32[7] = this._axionMass || 0.05;
        _statsUniformU32[8] = FIELD_GRID_RES; // fieldGridRes (GPU_SCALAR_GRID)
        _statsUniformU32[9] = 0;
        _statsUniformU32[10] = 0;
        _statsUniformU32[11] = 0;
        this.device.queue.writeBuffer(this._statsUniformBuffer, 0, _statsUniformData);

        // Rebuild field bind group if needed (field buffers may have been lazily allocated)
        if (!this._statsBindGroup1 ||
            (this._higgsBuffers && !this._statsFieldHasHiggs) ||
            (this._axionBuffers && !this._statsFieldHasAxion)) {
            this._rebuildStatsFieldBindGroup();
            this._statsFieldHasHiggs = !!this._higgsBuffers;
            this._statsFieldHasAxion = !!this._axionBuffers;
        }

        // Dispatch
        const encoder = this.device.createCommandEncoder({ label: 'computeStats' });
        const pass = encoder.beginComputePass({ label: 'computeStats' });
        pass.setPipeline(this._statsPipeline);
        pass.setBindGroup(0, this._statsBindGroup0);
        pass.setBindGroup(1, this._statsBindGroup1);
        pass.dispatchWorkgroups(1);
        pass.end();

        // Copy to staging (double-buffered)
        const staging = this._statsStagingFlip
            ? this.buffers.statsStagingB
            : this.buffers.statsStagingA;
        this._statsStagingFlip = !this._statsStagingFlip;
        encoder.copyBufferToBuffer(this.buffers.statsBuffer, 0, staging, 0, 512);
        this.device.queue.submit([encoder.finish()]);

        // Async readback
        this._statsPending = true;
        staging.mapAsync(GPUMapMode.READ).then(() => {
            this._statsData = new Float32Array(staging.getMappedRange().slice(0));
            staging.unmap();
            this._statsResultReady = true;
            this._statsPending = false;
        }).catch(() => {
            this._statsPending = false;
        });
    }

    /**
     * Read the result of a previously requested stats computation.
     * Returns null if not ready, or an object with aggregate stats + selected particle data.
     */
    readStats() {
        if (!this._statsResultReady || !this._statsData) return null;
        this._statsResultReady = false;
        const d = this._statsData;
        const result = {
            linearKE: d[0], spinKE: d[1],
            px: d[2], py: d[3],
            orbitalAngMom: d[4], spinAngMom: d[5],
            comX: d[6], comY: d[7],
            totalMass: d[8], aliveCount: d[9],
            pe: d[10],
            fieldEnergy: d[11], // Darwin
            fieldPx: d[12], fieldPy: d[13], // Darwin momentum
            higgsFieldEnergy: d[14], axionFieldEnergy: d[15],
            pfiEnergy: d[16] + d[17], // higgs + axion particle-field interaction
            scalarFieldMomX: d[18], scalarFieldMomY: d[19],
            selected: null,
        };

        // Selected particle data at offset 32 (mass = -1 signals no selection)
        if (d[36] > 0) {
            const flags = new Uint32Array(d.buffer, 40 * 4, 1)[0];
            result.selected = {
                posX: d[32], posY: d[33],
                velWX: d[34], velWY: d[35],
                mass: d[36], charge: d[37],
                angW: d[38], baseMass: d[39],
                flags,
                radius: d[41],
                velX: d[42], velY: d[43],
                angVel: d[44],
                magMoment: d[45], angMomentum: d[46],
                antimatter: d[47] > 0.5,
                forceGravity:    { x: d[48], y: d[49] },
                forceCoulomb:    { x: d[50], y: d[51] },
                forceMagnetic:   { x: d[52], y: d[53] },
                forceGravitomag: { x: d[54], y: d[55] },
                force1PN:        { x: d[56], y: d[57] },
                forceSpinCurv:   { x: d[58], y: d[59] },
                forceRadiation:  { x: d[60], y: d[61] },
                forceYukawa:     { x: d[62], y: d[63] },
                forceExternal:   { x: d[64], y: d[65] },
                forceHiggs:      { x: d[66], y: d[67] },
                forceAxion:      { x: d[68], y: d[69] },
            };
        }
        return result;
    }

    /**
     * Get the count of alive particles.
     * @returns {number}
     */
    getParticleCount() {
        return this.aliveCount;
    }

    /**
     * Get the cached state of a particle by index.
     * Returns null if no cached readback data is available.
     * @param {number} idx
     * @returns {Object|null} { x, y, radius, mass, charge, antimatter }
     */
    getParticleState(idx) {
        if (!this._cachedParticleState || idx >= this.aliveCount) return null;
        const s = this._cachedParticleState;
        return {
            x: s.posX[idx],
            y: s.posY[idx],
            radius: s.radius[idx],
            mass: s.mass[idx],
            charge: s.charge[idx],
            antimatter: !!(s.flags[idx] & FLAG_ANTIMATTER),
        };
    }

    destroy() {
        this.buffers.destroy();
        this.uniformBuffer.destroy();
        if (this._dummyHistBuf) this._dummyHistBuf.destroy();
        if (this._qtCounterSrc) this._qtCounterSrc.destroy();
        if (this._qtBoundsSrc) this._qtBoundsSrc.destroy();
    }
}

// Flag constants (must match common.wgsl)
const FLAG_ALIVE = 1;
const FLAG_RETIRED = 2;
const FLAG_ANTIMATTER = 4;

// fetchShader imported from gpu-pipelines.js (single source of truth)
