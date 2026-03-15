/**
 * @fileoverview GPUPhysics — WebGPU compute pipeline orchestrator.
 *
 * Phase 2+3+4+5: Full force computation, Boris integrator, tree build, collisions,
 * dead GC, radiation, 1PN VV, boson lifecycle, boson gravity, signal delay history,
 * scalar fields, heatmap, expansion, disintegration, pair production.
 *
 * Dispatch sequence per substep:
 *   1. resetForces
 *   2. cacheDerived
 *   3. generateGhosts        (Phase 3 — if periodic boundary)
 *   4a-d. treeBuild           (Phase 3 — if BH enabled)
 *   5. computeForces          (Phase 2 pairwise OR Phase 3 tree walk)
 *   5b. externalFields
 *   5c. scalarFieldForces      (Phase 5 — gradient forces + mass/axMod modulation, before Boris)
 *   6. borisHalfKick (first)
 *   7. borisRotate
 *   8. borisHalfKick (second)
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
 *  - bosonGravity             (Phase 4 — build boson tree + particle/boson gravity)
 *  - deadParticleGC           (Phase 3)
 *  - recordHistory            (Phase 4 — every HISTORY_STRIDE frames)
 */
import { createParticleBuffers, createUniformBuffer, writeUniforms, createFieldBuffers, createPQSScratchBuffer, createPQSIndexBuffer, createHeatmapBuffers, createExcitationBuffers, createDisintegrationBuffers, createPairProductionBuffers, createTrailBuffers, FIELD_GRID_RES, COARSE_RES, COARSE_SQ, PARTICLE_STATE_SIZE, PARTICLE_AUX_SIZE, RADIATION_STATE_SIZE, PHOTON_SIZE, PION_SIZE, DERIVED_SIZE } from './gpu-buffers.js';
import { createPhase2Pipelines, createGhostGenPipeline, createTreeBuildPipelines, createTreeForcePipeline, createCollisionPipelines, createDeadGCPipeline, createPhase4Pipelines, createFieldDepositPipelines, createFieldEvolvePipelines, createFieldForcesPipelines, createFieldParticleGravPipeline, createFieldSelfGravPipelines, createFieldExcitationPipeline, createHeatmapPipelines, createExpansionPipeline, createDisintegrationPipeline, createPairProductionPipeline, createUpdateColorsPipeline, createTrailRecordPipeline, createHitTestPipeline } from './gpu-pipelines.js';
import { buildWGSLConstants } from './gpu-constants.js';
import {
    HISTORY_STRIDE, MAX_PHOTONS, MAX_PIONS,
    GPU_MAX_PARTICLES,
    COL_MERGE, COL_BOUNCE, BOUND_LOOP,
    COL_NAMES, BOUND_NAMES, TOPO_NAMES,
} from '../config.js';

const MAX_PARTICLES = GPU_MAX_PARTICLES;

// Pre-allocated typed arrays for per-frame writeBuffer calls (avoid GC pressure)
const _qtNodeCounterData = new Uint32Array([1]);
const _qtBoundsResetData = new Int32Array([2147483647, 2147483647, -2147483647, -2147483647]);
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
const _fgUniformData = new ArrayBuffer(32);  // FGUniforms: 8 × f32/u32
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
const _addParticleRadData = new Float32Array(24);    // RADIATION_STATE_SIZE / 4 (was 16, now 24 with Larmor backward-diff fields)
const _zeroU32 = new Uint32Array([0]);                // reusable zero for counter resets
const _retiredFlagU32 = new Uint32Array([2]);         // FLAG_RETIRED for removeParticle

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
        this._bosonGravEnabled = false;

        // Toggle state
        this._toggles0 = 0;
        this._toggles1 = 0;
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
        this._pqsScratch = null;
        this._pqsIndices = null;
        this._heatmapBuffers = null;
        this._excitationBuffers = null;
        this._disintBuffers = null;
        this._pairProdBuffers = null;
        this._higgsEnabled = false;
        this._axionEnabled = false;
        this._fieldResolution = FIELD_GRID_RES; // default: 64, configurable to 128/256
        this._fieldGravEnabled = false;
        this._expansionEnabled = false;
        this._disintegrationEnabled = false;
        this._hubbleParam = 0.001;
        this._heatmapEnabled = false;
        this._heatmapFrame = 0;

        // Phase 5: Pipelines (lazy-initialized)
        this._fieldDeposit = null;
        this._fieldEvolve = null;
        this._fieldForces = null;
        this._fieldParticleGrav = null;
        this._fieldSelfGrav = null;
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
        this._hitStagingData = null;

        // Free slot management: CPU-side mirror of GPU free stack
        this._cpuFreeSlots = [];
        this._freeTopPending = false;

        // Periodic CPU←GPU particle sync
        this._particleSyncPending = false;
        this._particleSyncData = null;  // latest readback: { f32, u32 } views over raw ArrayBuffer
        this._particleSyncFrame = 0;

        // Adaptive substepping state
        this._maxAccel = 0;
        this._maxAccelPending = false;

        this._ready = false;
    }

    /** Load WGSL shaders and create compute pipelines. Must be called before update(). */
    async init() {
        const wgslConstants = buildWGSLConstants();
        const commonWGSL = wgslConstants + '\n' + await fetchShader('common.wgsl');
        const boundaryWGSL = await fetchShader('boundary.wgsl');

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
            this._hitResultBuffer = this.device.createBuffer({
                label: 'hitResult', size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            this._hitResultStaging = this.device.createBuffer({
                label: 'hitResultStaging', size: 4,
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
                ],
            });
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

        // resetForces: uniforms + 1 allForces
        this._bg_resetForces = bg('resetForces', p2.resetForces.bindGroupLayouts[0],
            [this.uniformBuffer, b.allForces]);

        // cacheDerived: uniforms + particleState (packed) + derived + particleAux + axYukMod
        this._bg_cacheDerived = bg('cacheDerived', p2.cacheDerived.bindGroupLayouts[0],
            [this.uniformBuffer, b.particleState, b.derived, b.particleAux, b.axYukMod]);

        // pairForce: 4 bind groups (packed structs)
        this._bg_pairForce0 = bg('pairForce_g0', p2.pairForce.bindGroupLayouts[0],
            [this.uniformBuffer]);
        this._bg_pairForce1 = bg('pairForce_g1', p2.pairForce.bindGroupLayouts[1],
            [b.particleState, b.derived, b.axYukMod]);
        this._bg_pairForce2 = bg('pairForce_g2', p2.pairForce.bindGroupLayouts[2],
            [b.allForces]);
        this._bg_pairForce3 = bg('pairForce_g3', p2.pairForce.bindGroupLayouts[3],
            [b.radiationState, b.maxAccelBuffer]);

        // externalFields (packed particleState + allForces)
        this._bg_extFields = bg('extFields', p2.externalFields.bindGroupLayouts[0],
            [this.uniformBuffer, b.particleState, b.allForces]);

        // borisHalfKick (reads from particleState + allForces)
        this._bg_halfKick = bg('halfKick', p2.borisHalfKick.bindGroupLayouts[0],
            [this.uniformBuffer, b.particleState, b.allForces]);

        // borisRotate (reads from particleState + allForces)
        this._bg_rotate = bg('rotate', p2.borisRotate.bindGroupLayouts[0],
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

        // Group 1: particle state (packed struct) + derived
        this._treeBuildBG1 = this.device.createBindGroup({
            label: 'treeBuild_g1',
            layout: layouts[1],
            entries: [
                { binding: 0, resource: { buffer: b.particleState } },
                { binding: 1, resource: { buffer: b.derived } },
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

        // Group 2: allForces + radiationState + maxAccel
        this._treeForceGroup2 = this.device.createBindGroup({
            label: 'treeForce_g2',
            layout: layouts[2],
            entries: [
                { binding: 0, resource: { buffer: b.allForces } },
                { binding: 1, resource: { buffer: b.radiationState } },
                { binding: 2, resource: { buffer: b.maxAccelBuffer } },
            ],
        });
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

        // ── Boson Tree (insertBosonsIntoTree, computeBosonAggregates, computeBosonGravity, applyBosonBosonGravity) ──
        // Group 0: uniforms + tree nodes (merged to fit within 4 bind groups)
        this._phase4BindGroups.btG0 = bg('bosonTree_g0', p4.insertBosonsIntoTree.bindGroupLayouts[0],
            [this.uniformBuffer, b.bosonTreeNodes, b.bosonTreeCounter]);
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
            [b.histPosX, b.histPosY, b.histVelWX, b.histVelWY, b.histAngW, b.histTime, b.histMeta]);
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
        if (!(t0 & 1) && !(t0 & 2)) return; // neither gravity nor coulomb

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
     * After drift: rebuild tree if BH on, recompute 1PN, apply VV kick.
     */
    _dispatch1PNVV(encoder) {
        if (!this._onePNEnabled) return;

        const workgroups = Math.ceil(this.aliveCount / 64);
        const bgs = this._phase4BindGroups;
        const p4 = this._phase4;

        // Step 1: Recompute 1PN forces at post-drift positions
        const passCompute = encoder.beginComputePass({ label: 'compute1PN' });
        passCompute.setPipeline(p4.compute1PN.pipeline);
        passCompute.setBindGroup(0, bgs.onePNG0);
        passCompute.setBindGroup(1, bgs.onePNG1);
        passCompute.setBindGroup(2, bgs.onePNG2);
        passCompute.dispatchWorkgroups(workgroups);
        passCompute.end();

        // Step 2: Apply VV correction kick
        const passKick = encoder.beginComputePass({ label: 'vvKick1PN' });
        passKick.setPipeline(p4.vvKick1PN.pipeline);
        passKick.setBindGroup(0, bgs.onePNG0);
        passKick.setBindGroup(1, bgs.onePNG1);
        passKick.setBindGroup(2, bgs.onePNG2);
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

        // updatePhotons: drift + lensing
        const phWG = Math.ceil(MAX_PHOTONS / 64);
        const passPhotons = encoder.beginComputePass({ label: 'updatePhotons' });
        passPhotons.setPipeline(p4.updatePhotons.pipeline);
        passPhotons.setBindGroup(0, bgs.bosG0);
        passPhotons.setBindGroup(1, bgs.bosG1);
        passPhotons.setBindGroup(2, bgs.bosG2);
        passPhotons.setBindGroup(3, bgs.bosG3);
        passPhotons.dispatchWorkgroups(phWG);
        passPhotons.end();

        // updatePions: drift with proper velocity
        const piWG = Math.ceil(MAX_PIONS / 64);
        const passPions = encoder.beginComputePass({ label: 'updatePions' });
        passPions.setPipeline(p4.updatePions.pipeline);
        passPions.setBindGroup(0, bgs.bosG0);
        passPions.setBindGroup(1, bgs.bosG1);
        passPions.setBindGroup(2, bgs.bosG2);
        passPions.setBindGroup(3, bgs.bosG3);
        passPions.dispatchWorkgroups(piWG);
        passPions.end();

        // absorbPhotons
        const passAbsorbPh = encoder.beginComputePass({ label: 'absorbPhotons' });
        passAbsorbPh.setPipeline(p4.absorbPhotons.pipeline);
        passAbsorbPh.setBindGroup(0, bgs.bosG0);
        passAbsorbPh.setBindGroup(1, bgs.bosG1);
        passAbsorbPh.setBindGroup(2, bgs.bosG2);
        passAbsorbPh.setBindGroup(3, bgs.bosG3);
        passAbsorbPh.dispatchWorkgroups(phWG);
        passAbsorbPh.end();

        // absorbPions
        const passAbsorbPi = encoder.beginComputePass({ label: 'absorbPions' });
        passAbsorbPi.setPipeline(p4.absorbPions.pipeline);
        passAbsorbPi.setBindGroup(0, bgs.bosG0);
        passAbsorbPi.setBindGroup(1, bgs.bosG1);
        passAbsorbPi.setBindGroup(2, bgs.bosG2);
        passAbsorbPi.setBindGroup(3, bgs.bosG3);
        passAbsorbPi.dispatchWorkgroups(piWG);
        passAbsorbPi.end();

        // decayPions
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
     * Dispatch boson gravity passes (Phase 4).
     * Runs once per frame after all substeps: build boson tree, compute gravity.
     */
    _dispatchBosonGravity(encoder) {
        if (!this._bosonGravEnabled) return;

        const p4 = this._phase4;
        const bgs = this._phase4BindGroups;
        const b = this.buffers;

        // Reset boson tree node counter to 1 (root = node 0)
        this.device.queue.writeBuffer(b.bosonTreeCounter, 0, _qtNodeCounterData);

        // Initialize root node bounds (reuse pre-allocated arrays to avoid GC)
        _bosonRootData.fill(0);
        _bosonRootF32[0] = 0;              // minX
        _bosonRootF32[1] = 0;              // minY
        _bosonRootF32[2] = this.domainW;   // maxX
        _bosonRootF32[3] = this.domainH;   // maxY
        this.device.queue.writeBuffer(b.bosonTreeNodes, 0, _bosonRootData);

        const totalBosons = MAX_PHOTONS + MAX_PIONS;
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

        // computeBosonAggregates
        const maxBosonNodes = b.MAX_BOSON_NODES;
        const aggWG = Math.ceil(maxBosonNodes / 64);
        const passAgg = encoder.beginComputePass({ label: 'computeBosonAggregates' });
        passAgg.setPipeline(p4.computeBosonAggregates.pipeline);
        passAgg.setBindGroup(0, bgs.btG0);
        passAgg.setBindGroup(1, bgs.btG1);
        passAgg.setBindGroup(2, bgs.btG2);
        passAgg.setBindGroup(3, bgs.btG3);
        passAgg.dispatchWorkgroups(aggWG);
        passAgg.end();

        // computeBosonGravity: particle <- boson gravity
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
     *
     * COL_PASS (0):        skip entirely.
     * COL_MERGE (1) + BH:  tree-based detect → resolveCollisions (merge/annihilation).
     * COL_MERGE (1) no BH: pairwise detect   → resolveCollisions (merge/annihilation).
     * COL_BOUNCE (2):      pairwise detect   → resolveBouncePairwise (Hertz impulse).
     */
    _dispatchCollisions(encoder) {
        if (this._collisionMode === 0) return; // COL_PASS — nothing to do
        if (this.aliveCount === 0) return;

        const b = this.buffers;
        const isMerge  = this._collisionMode === COL_MERGE;
        const isBounce = this._collisionMode === COL_BOUNCE;
        const useTree  = isMerge && this._barnesHutEnabled;

        // Reset pair counter (and merge counter for merge mode)
        encoder.clearBuffer(b.collisionPairCounter, 0, 4);
        if (isMerge) encoder.clearBuffer(b.mergeResultCounter, 0, 4);

        // ── Detection phase ──
        const detectWG = Math.ceil(this.aliveCount / 64);
        if (useTree) {
            // Tree-accelerated broadphase (existing path)
            const passDetect = encoder.beginComputePass({ label: 'detectCollisions' });
            passDetect.setPipeline(this._collisionPipelines.detectCollisions);
            passDetect.setBindGroup(0, this._collisionBG0);
            passDetect.setBindGroup(1, this._collisionBG1);
            passDetect.setBindGroup(2, this._collisionBG2);
            passDetect.dispatchWorkgroups(detectWG);
            passDetect.end();
        } else {
            // O(N²) tiled pairwise broadphase (no tree needed)
            const passDetect = encoder.beginComputePass({ label: 'detectCollisionsPairwise' });
            passDetect.setPipeline(this._collisionPipelines.detectCollisionsPairwise);
            passDetect.setBindGroup(0, this._collisionBG0);
            passDetect.setBindGroup(1, this._collisionBG1);
            passDetect.setBindGroup(2, this._collisionBG2);
            passDetect.dispatchWorkgroups(detectWG);
            passDetect.end();
        }

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
     * Dispatch tree build sequence when Barnes-Hut is enabled.
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
        if (!this._barnesHutEnabled) return;

        const totalCount = this.aliveCount + this._ghostCount;
        if (totalCount === 0) return;

        const b = this.buffers;

        // 1. Reset nodeCounter to 1 (root is node 0, already allocated)
        this.device.queue.writeBuffer(b.qtNodeCounter, 0, _qtNodeCounterData);

        // 2. Reset bounds: minX=INT_MAX, minY=INT_MAX, maxX=INT_MIN, maxY=INT_MIN
        this.device.queue.writeBuffer(b.qtBoundsBuffer, 0, _qtBoundsResetData);

        // 3. Clear visitor flags to 0 (write zeros for all nodes)
        // Use clearBuffer instead of allocating a zero array each frame
        const clearSize = Math.min(b.QT_MAX_NODES, totalCount * 6) * 4;
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
        if (antimatter) flags |= FLAG_ANTIMATTER;
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
        return idx;
    }

    /**
     * Remove a particle from the GPU by marking it dead (clearing ALIVE, setting RETIRED).
     * The dead GC shader will reclaim the slot for reuse by addParticle().
     * @param {number} idx - GPU buffer index of the particle to remove
     */
    removeParticle(idx) {
        if (idx < 0 || idx >= this.aliveCount) return;
        // Clear ALIVE bit, set RETIRED bit in flags (byte offset 8 in ParticleState = 32 bytes in)
        // ParticleState: posX(0), posY(4), velWX(8), velWY(12), mass(16), charge(20), angW(24), baseMass(28), flags(32)
        this.device.queue.writeBuffer(this.buffers.particleState, idx * PARTICLE_STATE_SIZE + 32, _retiredFlagU32);
    }

    /**
     * Pack toggle booleans into u32 bitfields for GPU uniforms.
     * Must be called whenever a toggle changes.
     */
    setToggles(physics) {
        let t0 = 0;
        if (physics.gravityEnabled) t0 |= 1;
        if (physics.coulombEnabled) t0 |= 2;
        if (physics.magneticEnabled) t0 |= 4;
        if (physics.gravitomagEnabled) t0 |= 8;
        if (physics.onePNEnabled) t0 |= 16;
        if (physics.relativityEnabled) t0 |= 32;
        if (physics.spinOrbitEnabled) t0 |= 64;
        if (physics.radiationEnabled) t0 |= 128;
        if (physics.blackHoleEnabled) t0 |= 256;
        if (physics.disintegrationEnabled) t0 |= 512;
        if (physics.expansionEnabled) t0 |= 1024;
        if (physics.yukawaEnabled) t0 |= 2048;
        if (physics.higgsEnabled) t0 |= 4096;
        if (physics.axionEnabled) t0 |= 8192;
        if (physics.barnesHutEnabled) t0 |= 16384;
        if (physics.bosonGravEnabled) t0 |= 32768;
        this._toggles0 = t0;

        let t1 = 0;
        if (physics.fieldGravEnabled) t1 |= 1;
        this._toggles1 = t1;

        this._blackHoleEnabled = physics.blackHoleEnabled;
        this._barnesHutEnabled = physics.barnesHutEnabled;
        this._relativityEnabled = physics.relativityEnabled;
        this._onePNEnabled = physics.onePNEnabled;
        this._radiationEnabled = physics.radiationEnabled;
        this._yukawaEnabled = physics.yukawaEnabled;
        this._bosonGravEnabled = physics.bosonGravEnabled;
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
        this._fieldGravEnabled = physics.fieldGravEnabled;
        this._expansionEnabled = physics.expansionEnabled;
        this._disintegrationEnabled = physics.disintegrationEnabled;
        this._hubbleParam = physics.hubbleParam || 0.001;
        this._heatmapEnabled = physics.heatmapEnabled || false;

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
     * Also ensures shared PQS scratch/index buffers are allocated.
     * Initializes field values: Higgs=1.0 (VEV), Axion=0.0.
     * @param {'higgs'|'axion'} which
     */
    _ensureFieldBuffers(which) {
        if (!this._pqsScratch) {
            this._pqsScratch = createPQSScratchBuffer(this.device, MAX_PARTICLES);
            this._pqsIndices = createPQSIndexBuffer(this.device, MAX_PARTICLES);
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
        }
        if (which === 'higgs' && !this._higgsBuffers) {
            this._higgsBuffers = createFieldBuffers(this.device, 'higgs', MAX_PARTICLES);
            this._initFieldToVacuum('higgs');
        }
        if (which === 'axion' && !this._axionBuffers) {
            this._axionBuffers = createFieldBuffers(this.device, 'axion', MAX_PARTICLES);
            this._initFieldToVacuum('axion');
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
        const fieldData = new Float32Array(gridSq).fill(vacValue);
        this.device.queue.writeBuffer(fb.field, 0, fieldData);

        const zeros = new Float32Array(gridSq);
        this.device.queue.writeBuffer(fb.fieldDot, 0, zeros);
        this.device.queue.writeBuffer(fb.gradX, 0, zeros);
        this.device.queue.writeBuffer(fb.gradY, 0, zeros);
        this.device.queue.writeBuffer(fb.source, 0, zeros);
        this.device.queue.writeBuffer(fb.laplacian, 0, zeros);
        this.device.queue.writeBuffer(fb.thermal, 0, zeros);
        this.device.queue.writeBuffer(fb.energyDensity, 0, zeros);
        this.device.queue.writeBuffer(fb.sgPhiFull, 0, zeros);
        this.device.queue.writeBuffer(fb.sgGradX, 0, zeros);
        this.device.queue.writeBuffer(fb.sgGradY, 0, zeros);

    }

    /**
     * Lazily initialize Phase 5 pipelines on first use.
     * Called when any Phase 5 feature is first enabled.
     */
    async _ensurePhase5Pipelines() {
        if (this._fieldDeposit) return; // already initialized

        const wgslConstants = buildWGSLConstants();

        // Initialize all Phase 5 pipelines in parallel
        const [deposit, evolve, forces, particleGrav, selfGrav, excitation, heatmap, expansion, disint, pairProd] =
            await Promise.all([
                createFieldDepositPipelines(this.device, wgslConstants),
                createFieldEvolvePipelines(this.device, wgslConstants),
                createFieldForcesPipelines(this.device, wgslConstants),
                createFieldParticleGravPipeline(this.device, wgslConstants),
                createFieldSelfGravPipelines(this.device, wgslConstants),
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
        this._fieldExcitation = excitation;
        this._heatmapPipelines = heatmap;
        this._expansionPipeline = expansion;
        this._disintPipeline = disint;
        this._pairProdPipeline = pairProd;
    }

    /**
     * Write FieldUniforms to the shared field uniform buffer.
     */
    _writeFieldUniforms(dt) {
        if (!this._fieldUniformBuffer) return;
        const f = _fieldUniformF32;
        const u = _fieldUniformU32;
        // Must match FieldUniforms struct in field-common.wgsl exactly:
        f[0] = dt;                              // dt
        f[1] = this.domainW;                    // domainW
        f[2] = this.domainH;                    // domainH
        u[3] = this.boundaryMode;               // boundaryMode
        u[4] = this.topologyMode;               // topologyMode
        // Higgs params
        f[5] = this._higgsMass;                 // higgsMass
        f[6] = this._higgsCoupling;             // higgsCoupling
        f[7] = 0.05;                            // higgsMassFloor (matches CPU HIGGS_MASS_FLOOR)
        f[8] = 4.0;                             // higgsMassMaxDelta (matches CPU HIGGS_MASS_MAX_DELTA)
        // Axion params
        f[9] = this._axionMass;                 // axionMass
        f[10] = this._axionCoupling;            // axionCoupling
        // Toggle bits
        u[11] = this._higgsEnabled ? 1 : 0;     // higgsEnabled
        u[12] = this._axionEnabled ? 1 : 0;     // axionEnabled
        u[13] = (this._toggles0 & 2) ? 1 : 0;   // coulombEnabled
        u[14] = (this._toggles0 & 2048) ? 1 : 0; // yukawaEnabled
        u[15] = this._fieldGravEnabled ? 1 : 0;  // gravityEnabled (field self-gravity)
        u[16] = this._relativityEnabled ? 1 : 0;  // relativityEnabled
        u[17] = this._blackHoleEnabled ? 1 : 0;   // blackHoleEnabled
        u[18] = this.aliveCount;                  // particleCount
        f[19] = this._blackHoleEnabled ? 16 : 64;  // softeningSq
        u[20] = 0;                                // currentFieldType (0=higgs default, set per-dispatch)
        this.device.queue.writeBuffer(this._fieldUniformBuffer, 0, _fieldUniformData);
    }

    /**
     * Write per-field uniforms to the dedicated field uniform buffer.
     * Eliminates the need for encoder split when both Higgs and Axion are active.
     * @param {number} dt
     * @param {number} fieldType - 0=higgs, 1=axion
     */
    _writePerFieldUniforms(dt, fieldType) {
        const buf = fieldType === 0 ? this._higgsUniformBuffer : this._axionUniformBuffer;
        if (!buf) return;
        const f = _fieldUniformF32;
        const u = _fieldUniformU32;
        f[0] = dt;
        f[1] = this.domainW;
        f[2] = this.domainH;
        u[3] = this.boundaryMode;
        u[4] = this.topologyMode;
        f[5] = this._higgsMass;
        f[6] = this._higgsCoupling;
        f[7] = 0.05;
        f[8] = 4.0;
        f[9] = this._axionMass;
        f[10] = this._axionCoupling;
        u[11] = this._higgsEnabled ? 1 : 0;
        u[12] = this._axionEnabled ? 1 : 0;
        u[13] = (this._toggles0 & 2) ? 1 : 0;
        u[14] = (this._toggles0 & 2048) ? 1 : 0;
        u[15] = this._fieldGravEnabled ? 1 : 0;
        u[16] = this._relativityEnabled ? 1 : 0;
        u[17] = this._blackHoleEnabled ? 1 : 0;
        u[18] = this.aliveCount;
        f[19] = this._blackHoleEnabled ? 16 : 64;
        u[20] = fieldType;
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

        // Group 1: scratch + target grid + uniforms (for source deposition)
        const g1Source = this.device.createBindGroup({
            label: `fieldDeposit_g1_source_${which}`,
            layout: dep.bindGroupLayouts[1],
            entries: [
                { binding: 0, resource: { buffer: this._pqsScratch } },
                { binding: 1, resource: { buffer: this._pqsIndices } },
                { binding: 2, resource: { buffer: fb.source } },
                { binding: 3, resource: { buffer: uBuf } },
            ],
        });

        // Group 1 for thermal (Higgs only, but create for both for simplicity)
        const g1Thermal = this.device.createBindGroup({
            label: `fieldDeposit_g1_thermal_${which}`,
            layout: dep.bindGroupLayouts[1],
            entries: [
                { binding: 0, resource: { buffer: this._pqsScratch } },
                { binding: 1, resource: { buffer: this._pqsIndices } },
                { binding: 2, resource: { buffer: fb.thermal } },
                { binding: 3, resource: { buffer: uBuf } },
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

        // Evolve bind group (gradX/Y are read-only for self-gravity cross-terms)
        this._fieldEvolveBGs[which] = this.device.createBindGroup({
            label: `fieldEvolve_${which}`,
            layout: this._fieldEvolve.evolveBindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: fb.field } },
                { binding: 1, resource: { buffer: fb.fieldDot } },
                { binding: 2, resource: { buffer: fb.laplacian } },
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
                { binding: 2, resource: { buffer: fb.laplacian } },
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

        // Group 0: core field arrays + uniform (6 storage + 1 uniform)
        this._fieldSelfGravBGs[which] = this.device.createBindGroup({
            label: `fieldSelfGrav_${which}_g0`,
            layout: this._fieldSelfGrav.bindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: fb.field } },
                { binding: 1, resource: { buffer: fb.fieldDot } },
                { binding: 2, resource: { buffer: fb.gradX } },
                { binding: 3, resource: { buffer: fb.gradY } },
                { binding: 4, resource: { buffer: fb.energyDensity } },
                { binding: 5, resource: { buffer: fb.coarseRho } },
                { binding: 6, resource: { buffer: uBuf } },
            ],
        });
        // Group 1: self-gravity arrays (4 storage — sgInvR computed inline)
        this._fieldSelfGravBGs[which + '_g1'] = this.device.createBindGroup({
            label: `fieldSelfGrav_${which}_g1`,
            layout: this._fieldSelfGrav.bindGroupLayouts[1],
            entries: [
                { binding: 0, resource: { buffer: fb.coarsePhi } },
                { binding: 1, resource: { buffer: fb.sgPhiFull } },
                { binding: 2, resource: { buffer: fb.sgGradX } },
                { binding: 3, resource: { buffer: fb.sgGradY } },
            ],
        });
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
     * Dispatch scalar field evolution for one field (Higgs or Axion).
     * Full sequence: deposit → [self-gravity] → Laplacian → halfKick → drift →
     * Laplacian → halfKick → NaN fixup → compute gradients
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

        // Step 1: Clear source grid
        {
            const p = encoder.beginComputePass({ label: `clearSource_${which}` });
            p.setPipeline(dep.clearGrid);
            p.setBindGroup(0, depBGs.g0);
            p.setBindGroup(1, depBGs.g1Source);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }

        // Step 2: Scatter deposit (source)
        if (this.aliveCount > 0) {
            const scatterPipeline = which === 'axion'
                ? dep.scatterDepositAxion
                : dep.scatterDeposit;
            const p = encoder.beginComputePass({ label: `scatterDeposit_${which}` });
            p.setPipeline(scatterPipeline);
            p.setBindGroup(0, depBGs.g0);
            p.setBindGroup(1, depBGs.g1Source);
            p.dispatchWorkgroups(particleWG);
            p.end();
        }

        // Step 3: Gather deposit (source)
        {
            const p = encoder.beginComputePass({ label: `gatherDeposit_${which}` });
            p.setPipeline(dep.gatherDeposit);
            p.setBindGroup(0, depBGs.g0);
            p.setBindGroup(1, depBGs.g1Source);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }

        // Step 4: Higgs thermal deposition (Higgs only)
        if (which === 'higgs') {
            // Clear thermal
            {
                const p = encoder.beginComputePass({ label: 'clearThermal' });
                p.setPipeline(dep.clearGrid);
                p.setBindGroup(0, depBGs.g0);
                p.setBindGroup(1, depBGs.g1Thermal);
                p.dispatchWorkgroups(gridWG, gridWG);
                p.end();
            }
            // Scatter thermal
            if (this.aliveCount > 0) {
                const p = encoder.beginComputePass({ label: 'scatterThermal' });
                p.setPipeline(dep.scatterDepositThermal);
                p.setBindGroup(0, depBGs.g0);
                p.setBindGroup(1, depBGs.g1Thermal);
                p.dispatchWorkgroups(particleWG);
                p.end();
            }
            // Gather thermal
            {
                const p = encoder.beginComputePass({ label: 'gatherThermal' });
                p.setPipeline(dep.gatherDeposit);
                p.setBindGroup(0, depBGs.g0);
                p.setBindGroup(1, depBGs.g1Thermal);
                p.dispatchWorkgroups(gridWG, gridWG);
                p.end();
            }
        }

        // Step 5: Self-gravity (if field gravity enabled)
        if (this._fieldGravEnabled) {
            this._ensureSelfGravBindGroups(which);
            const sgBG0 = this._fieldSelfGravBGs[which];
            const sgBG1 = this._fieldSelfGravBGs[which + '_g1'];
            const sg = this._fieldSelfGrav;
            const coarseWG = 1; // 8x8 = one workgroup for 8x8 coarse grid

            // Energy density
            const edPipeline = which === 'higgs'
                ? sg.computeEnergyDensityHiggs
                : sg.computeEnergyDensityAxion;
            {
                const p = encoder.beginComputePass({ label: `energyDensity_${which}` });
                p.setPipeline(edPipeline);
                p.setBindGroup(0, sgBG0);
                p.setBindGroup(1, sgBG1);
                p.dispatchWorkgroups(gridWG, gridWG);
                p.end();
            }
            // Downsample
            {
                const p = encoder.beginComputePass({ label: `downsampleRho_${which}` });
                p.setPipeline(sg.downsampleRho);
                p.setBindGroup(0, sgBG0);
                p.setBindGroup(1, sgBG1);
                p.dispatchWorkgroups(coarseWG, coarseWG);
                p.end();
            }
            // Coarse potential
            {
                const p = encoder.beginComputePass({ label: `coarsePotential_${which}` });
                p.setPipeline(sg.computeCoarsePotential);
                p.setBindGroup(0, sgBG0);
                p.setBindGroup(1, sgBG1);
                p.dispatchWorkgroups(coarseWG, coarseWG);
                p.end();
            }
            // Upsample
            {
                const p = encoder.beginComputePass({ label: `upsamplePhi_${which}` });
                p.setPipeline(sg.upsamplePhi);
                p.setBindGroup(0, sgBG0);
                p.setBindGroup(1, sgBG1);
                p.dispatchWorkgroups(gridWG, gridWG);
                p.end();
            }
            // SG gradients
            {
                const p = encoder.beginComputePass({ label: `sgGradients_${which}` });
                p.setPipeline(sg.computeSelfGravGradients);
                p.setBindGroup(0, sgBG0);
                p.setBindGroup(1, sgBG1);
                p.dispatchWorkgroups(gridWG, gridWG);
                p.end();
            }
        }

        // Step 6: KDK Störmer-Verlet
        const halfKickPipeline = which === 'higgs'
            ? evo.higgsHalfKick
            : evo.axionHalfKick;

        // Laplacian (1st)
        {
            const p = encoder.beginComputePass({ label: `laplacian1_${which}` });
            p.setPipeline(evo.computeLaplacian);
            p.setBindGroup(0, evolveBG);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }
        // Half-kick (1st)
        {
            const p = encoder.beginComputePass({ label: `halfKick1_${which}` });
            p.setPipeline(halfKickPipeline);
            p.setBindGroup(0, evolveBG);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }
        // Field drift
        {
            const p = encoder.beginComputePass({ label: `fieldDrift_${which}` });
            p.setPipeline(evo.fieldDrift);
            p.setBindGroup(0, evolveBG);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }
        // Laplacian (2nd)
        {
            const p = encoder.beginComputePass({ label: `laplacian2_${which}` });
            p.setPipeline(evo.computeLaplacian);
            p.setBindGroup(0, evolveBG);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }
        // Half-kick (2nd)
        {
            const p = encoder.beginComputePass({ label: `halfKick2_${which}` });
            p.setPipeline(halfKickPipeline);
            p.setBindGroup(0, evolveBG);
            p.dispatchWorkgroups(gridWG, gridWG);
            p.end();
        }
        // NaN fixup
        {
            const nanPipeline = which === 'higgs'
                ? evo.nanFixupHiggs
                : evo.nanFixupAxion;
            const p = encoder.beginComputePass({ label: `nanFixup_${which}` });
            p.setPipeline(nanPipeline);
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
     * Dispatched once per active field when fieldGravEnabled.
     */
    _dispatchFieldParticleGrav(encoder) {
        if (!this._fieldParticleGrav || !this._fieldGravEnabled || this.aliveCount === 0) return;
        if (!this._higgsEnabled && !this._axionEnabled) return;

        // Create uniform buffer once
        if (!this._fieldParticleGravUniform) {
            this._fieldParticleGravUniform = this.device.createBuffer({
                label: 'fieldParticleGrav_uniform',
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        }

        // Write uniforms (FGUniforms struct)
        _fgUniformF32[0] = this.domainW;
        _fgUniformF32[1] = this.domainH;
        _fgUniformF32[2] = this._blackHoleEnabled ? 16 : 64; // softeningSq
        _fgUniformU32[3] = this.aliveCount;
        _fgUniformU32[4] = this.boundaryMode;
        _fgUniformU32[5] = this.topologyMode;
        _fgUniformU32[6] = 0;
        _fgUniformU32[7] = 0;
        this.device.queue.writeBuffer(this._fieldParticleGravUniform, 0, _fgUniformData);

        const pg = this._fieldParticleGrav;
        const workgroups = Math.ceil(this.aliveCount / 64);
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
                            { binding: 0, resource: { buffer: fb.energyDensity } },
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
        const eventData = new Float32Array(eventCount * 4); // ExcitationEvent = 4 floats
        for (let i = 0; i < eventCount; i++) {
            const me = this._pendingMergeEvents[i];
            if (me.type !== 'merge') continue;
            eventData[i * 4] = me.x;
            eventData[i * 4 + 1] = me.y;
            eventData[i * 4 + 2] = me.energy;
            eventData[i * 4 + 3] = 0; // padding
        }
        this.device.queue.writeBuffer(this._excitationBuffers.events, 0, eventData);
        this.device.queue.writeBuffer(this._excitationBuffers.counter, 0, new Uint32Array([eventCount]));

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
        _pairProdUniformU32[6] = MAX_PHOTONS;
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

        const workgroups = Math.ceil(MAX_PHOTONS / 256);
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
            this._heatmapBuffers = createHeatmapBuffers(this.device);
        }

        if (!this._heatmapUniformBuffer) {
            this._heatmapUniformBuffer = this.device.createBuffer({
                label: 'heatmapUniforms',
                size: 96, // HeatmapUniforms struct size (22 fields, padded)
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        }

        // Compute viewport bounds from camera
        const halfW = (camera?.canvasW || 800) / (2 * (camera?.zoom || 16));
        const halfH = (camera?.canvasH || 600) / (2 * (camera?.zoom || 16));
        const viewLeft = (camera?.x || 0) - halfW;
        const viewTop = (camera?.y || 0) - halfH;
        const cellW = (2 * halfW) / 64;
        const cellH = (2 * halfH) / 64;

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
        _heatmapUniformU32[10] = (this._toggles0 & 1) ? 1 : 0; // doGravity
        _heatmapUniformU32[11] = (this._toggles0 & 2) ? 1 : 0; // doCoulomb
        _heatmapUniformU32[12] = (this._toggles0 & 2048) ? 1 : 0; // doYukawa
        _heatmapUniformU32[13] = (this._relativityEnabled && this.buffers.historyAllocated) ? 1 : 0; // useDelay
        _heatmapUniformU32[14] = this.boundaryMode === BOUND_LOOP ? 1 : 0;
        _heatmapUniformU32[15] = this.topologyMode;
        _heatmapUniformU32[16] = this.aliveCount;
        _heatmapUniformU32[17] = 0; // deadCount (not tracked on GPU yet)
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

        // Create/recreate history bind group when history buffers become available
        if (this.buffers.historyAllocated && !this._heatmapBGs.g2) {
            const b = this.buffers;
            const hm = this._heatmapPipelines;
            this._heatmapBGs.g2 = this.device.createBindGroup({
                label: 'heatmap_g2_history',
                layout: hm.heatmapLayouts[2],
                entries: [
                    { binding: 0, resource: { buffer: b.histPosX } },
                    { binding: 1, resource: { buffer: b.histPosY } },
                    { binding: 2, resource: { buffer: b.histTime } },
                    { binding: 3, resource: { buffer: b.histMeta } },
                ],
            });
        }

        // Ensure dummy history bind group exists when history not yet allocated
        // (pipeline layout requires 3 bind groups even when useDelay=0)
        if (!this._heatmapBGs.g2) {
            if (!this._heatmapDummyHistBuf) {
                this._heatmapDummyHistBuf = this.device.createBuffer({
                    label: 'heatmap-dummy-hist',
                    size: 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
            }
            const hm = this._heatmapPipelines;
            this._heatmapBGs.g2 = this.device.createBindGroup({
                label: 'heatmap_g2_dummy',
                layout: hm.heatmapLayouts[2],
                entries: [
                    { binding: 0, resource: { buffer: this._heatmapDummyHistBuf } },
                    { binding: 1, resource: { buffer: this._heatmapDummyHistBuf } },
                    { binding: 2, resource: { buffer: this._heatmapDummyHistBuf } },
                    { binding: 3, resource: { buffer: this._heatmapDummyHistBuf } },
                ],
            });
            this._heatmapBGs._g2IsDummy = true;
        }

        // Upgrade from dummy to real history bind group when buffers become available
        if (this._heatmapBGs._g2IsDummy && this.buffers.historyAllocated) {
            const b = this.buffers;
            const hm = this._heatmapPipelines;
            this._heatmapBGs.g2 = this.device.createBindGroup({
                label: 'heatmap_g2_history',
                layout: hm.heatmapLayouts[2],
                entries: [
                    { binding: 0, resource: { buffer: b.histPosX } },
                    { binding: 1, resource: { buffer: b.histPosY } },
                    { binding: 2, resource: { buffer: b.histTime } },
                    { binding: 3, resource: { buffer: b.histMeta } },
                ],
            });
            this._heatmapBGs._g2IsDummy = false;
        }

        const gridWG = Math.ceil(64 / 8);

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
        const channelToggles = [this._toggles0 & 1, this._toggles0 & 2, this._toggles0 & 2048];
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

            // Boson gravity (if enabled): build boson tree + particle<-boson + boson<->boson
            this._dispatchBosonGravity(encoder);

            // Dead particle garbage collection
            this._dispatchDeadGC(encoder);

            // Record signal delay history (once every HISTORY_STRIDE frames)
            // Clears FLAG_REBORN after resetting stale history for recycled slots
            this._histStride++;
            if (this._relativityEnabled && this._histStride >= HISTORY_STRIDE) {
                this._histStride = 0;
                this._dispatchRecordHistory(encoder);
            }

            // Quadrupole radiation (once per frame, after history recording — matches CPU order)
            // Uses PHYSICS_DT constant in shader (not u.dt which holds dtSub from last substep)
            this._dispatchQuadrupole(encoder);

            // Update particle colors from charge/mass/antimatter state
            if (this._updateColorsPipeline && this.aliveCount > 0) {
                _colorUniformData[0] = this._blackHoleEnabled ? 1 : 0;
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

        // Periodic particle sync readback for CPU←GPU consistency (every 8 frames)
        this._particleSyncFrame++;
        if (this._particleSyncFrame >= 8) {
            this._particleSyncFrame = 0;
            this._readbackParticleSync();
        }
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
            const passTree = encoder.beginComputePass({ label: 'treeForce' });
            passTree.setPipeline(this._treeForcePipeline);
            passTree.setBindGroup(0, this._treeForceGroup0);
            passTree.setBindGroup(1, this._treeForceGroup1);
            passTree.setBindGroup(2, this._treeForceGroup2);
            passTree.dispatchWorkgroups(workgroups);
            passTree.end();
        } else {
            // Pairwise force computation (O(N^2) tiled)
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

        // Pass 5c2: particle-field gravity (O(N×GRID²), if fieldGravEnabled)
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
                this._writePerFieldUniforms(dtSub, 0); // currentFieldType=0 (Higgs)
                this._dispatchFieldEvolve(encoder, 'higgs', dtSub);
            }
            if (this._axionEnabled) {
                this._writePerFieldUniforms(dtSub, 1); // currentFieldType=1 (Axion)
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

    /**
     * Periodic readback of all particle states for CPU←GPU sync.
     * Non-blocking, 1-frame latency. Called every SYNC_INTERVAL frames.
     * Updates _particleSyncData which main.js uses to rebuild sim.particles.
     */
    async _readbackParticleSync() {
        if (this._particleSyncPending) return;
        if (this.aliveCount === 0) return;
        this._particleSyncPending = true;

        const b = this.buffers;
        const readBytes = this.aliveCount * PARTICLE_STATE_SIZE;

        try {
            const enc = this.device.createCommandEncoder({ label: 'particleSync' });
            enc.copyBufferToBuffer(b.particleState, 0, b.particleSyncStaging, 0, readBytes);
            this.device.queue.submit([enc.finish()]);

            await b.particleSyncStaging.mapAsync(GPUMapMode.READ);
            const raw = new ArrayBuffer(readBytes);
            new Uint8Array(raw).set(new Uint8Array(b.particleSyncStaging.getMappedRange(0, readBytes)));
            b.particleSyncStaging.unmap();

            this._particleSyncData = {
                f32: new Float32Array(raw),
                u32: new Uint32Array(raw),
                count: this.aliveCount,
            };
        } catch (e) {
            // Device lost — don't block future readbacks
        }
        this._particleSyncPending = false;
    }

    /**
     * Consume the latest particle sync readback data.
     * Returns null if no new data since last consumption.
     * @returns {{ f32: Float32Array, u32: Uint32Array, count: number } | null}
     */
    consumeParticleSync() {
        const data = this._particleSyncData;
        this._particleSyncData = null;
        return data;
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

        // Pack toggle state
        const toggleKeys = [
            'gravityEnabled', 'bosonGravEnabled', 'fieldGravEnabled',
            'coulombEnabled', 'magneticEnabled',
            'gravitomagEnabled', 'relativityEnabled', 'barnesHutEnabled',
            'radiationEnabled', 'blackHoleEnabled', 'disintegrationEnabled',
            'spinOrbitEnabled',
            'onePNEnabled', 'yukawaEnabled', 'axionEnabled',
            'expansionEnabled', 'higgsEnabled',
        ];
        // Map internal toggle bits back to boolean flags
        const t0 = this._toggles0;
        const t1 = this._toggles1;
        state.toggles.gravityEnabled = !!(t0 & 1);
        state.toggles.coulombEnabled = !!(t0 & 2);
        state.toggles.magneticEnabled = !!(t0 & 4);
        state.toggles.gravitomagEnabled = !!(t0 & 8);
        state.toggles.onePNEnabled = !!(t0 & 16);
        state.toggles.relativityEnabled = !!(t0 & 32);
        state.toggles.spinOrbitEnabled = !!(t0 & 64);
        state.toggles.radiationEnabled = !!(t0 & 128);
        state.toggles.blackHoleEnabled = !!(t0 & 256);
        state.toggles.disintegrationEnabled = !!(t0 & 512);
        state.toggles.expansionEnabled = !!(t0 & 1024);
        state.toggles.yukawaEnabled = !!(t0 & 2048);
        state.toggles.higgsEnabled = !!(t0 & 4096);
        state.toggles.axionEnabled = !!(t0 & 8192);
        state.toggles.barnesHutEnabled = !!(t0 & 16384);
        state.toggles.bosonGravEnabled = !!(t0 & 32768);
        state.toggles.fieldGravEnabled = !!(t1 & 1);

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
        this._particleSyncData = null;
        this._particleSyncFrame = 0;
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
        // Clear trail ring buffers
        if (this._trailBuffers) {
            const tb = this._trailBuffers;
            this.device.queue.writeBuffer(tb.trailWriteIdx, 0, new Uint32Array(this.buffers.maxParticles));
            this.device.queue.writeBuffer(tb.trailCount, 0, new Uint32Array(this.buffers.maxParticles));
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
        const data = new Uint8Array(16);
        const f32 = new Float32Array(data.buffer);
        const u32 = new Uint32Array(data.buffer);
        f32[0] = worldX;
        f32[1] = worldY;
        u32[2] = this.aliveCount;
        u32[3] = 0;
        this.device.queue.writeBuffer(this._hitUniformBuffer, 0, data);

        // Dispatch single-thread compute
        const encoder = this.device.createCommandEncoder({ label: 'hitTest' });
        const pass = encoder.beginComputePass({ label: 'hitTest' });
        pass.setPipeline(this._hitTestPipeline);
        pass.setBindGroup(0, this._hitTestBindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();

        // Copy result to staging for readback
        encoder.copyBufferToBuffer(this._hitResultBuffer, 0, this._hitResultStaging, 0, 4);
        this.device.queue.submit([encoder.finish()]);

        // Async readback
        this._hitPending = true;
        this._hitResultStaging.mapAsync(GPUMapMode.READ).then(() => {
            const result = new Int32Array(this._hitResultStaging.getMappedRange().slice(0));
            this._hitResultStaging.unmap();
            this._hitStagingData = result;
            this._hitResultReady = true;
            this._hitPending = false;
        }).catch(() => {
            this._hitPending = false;
        });
    }

    /**
     * Read the result of a previously queued GPU hit test.
     * Returns GPU buffer index of the hit particle, or -1 if none/not ready.
     */
    readHitResult() {
        if (!this._hitResultReady || !this._hitStagingData) return -1;
        this._hitResultReady = false;
        return this._hitStagingData[0];
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
        if (this._heatmapDummyHistBuf) this._heatmapDummyHistBuf.destroy();
    }
}

// Flag constants (must match common.wgsl)
const FLAG_ALIVE = 1;
const FLAG_RETIRED = 2;
const FLAG_ANTIMATTER = 4;

/** Fetch a WGSL shader file relative to src/gpu/shaders/ */
async function fetchShader(filename) {
    const resp = await fetch(`src/gpu/shaders/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    return resp.text();
}
