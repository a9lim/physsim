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
 *  16. scalarFieldForces      (Phase 5 — Higgs mass mod + Axion axMod/yukMod)
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
import { createParticleBuffers, createUniformBuffer, writeUniforms, createFieldBuffers, createPQSScratchBuffer, createPQSIndexBuffer, createHeatmapBuffers, createExcitationBuffers, createDisintegrationBuffers, createPairProductionBuffers, FIELD_GRID_RES, COARSE_RES, COARSE_SQ } from './gpu-buffers.js';
import { createPhase2Pipelines, createGhostGenPipeline, createTreeBuildPipelines, createTreeForcePipeline, createCollisionPipelines, createDeadGCPipeline, createPhase4Pipelines, createFieldDepositPipelines, createFieldEvolvePipelines, createFieldForcesPipelines, createFieldSelfGravPipelines, createFieldExcitationPipeline, createHeatmapPipelines, createExpansionPipeline, createDisintegrationPipeline, createPairProductionPipeline } from './gpu-pipelines.js';

const MAX_PARTICLES = 4096;
const HISTORY_STRIDE = 64;
const MAX_PHOTONS = 512;
const MAX_PIONS = 256;

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
        this._driftPipeline = null;
        this._boundaryPipeline = null;
        this._driftBindGroup = null;
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

        // Phase 5: sgInvR table upload tracking
        this._sgInvRUploaded = { higgs: false, axion: false };

        // Adaptive substepping state
        this._maxAccel = 0;
        this._maxAccelPending = false;

        this._ready = false;
    }

    /** Load WGSL shaders and create compute pipelines. Must be called before update(). */
    async init() {
        const commonWGSL = await fetchShader('common.wgsl');
        const driftWGSL = await fetchShader('drift.wgsl');
        const boundaryWGSL = await fetchShader('boundary.wgsl');

        // --- Drift pipeline (Phase 1, kept for reference/fallback) ---
        const driftModule = this.device.createShaderModule({
            label: 'drift',
            code: commonWGSL + '\n' + driftWGSL,
        });

        const driftBindGroupLayout = this.device.createBindGroupLayout({
            label: 'drift',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ],
        });

        this._driftPipeline = this.device.createComputePipeline({
            label: 'drift',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [driftBindGroupLayout] }),
            compute: { module: driftModule, entryPoint: 'main' },
        });

        this._driftBindGroup = this.device.createBindGroup({
            label: 'drift',
            layout: driftBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.buffers.posX } },
                { binding: 2, resource: { buffer: this.buffers.posY } },
                { binding: 3, resource: { buffer: this.buffers.velWX } },
                { binding: 4, resource: { buffer: this.buffers.velWY } },
                { binding: 5, resource: { buffer: this.buffers.flags } },
            ],
        });

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
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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
                { binding: 1, resource: { buffer: this.buffers.posX } },
                { binding: 2, resource: { buffer: this.buffers.posY } },
                { binding: 3, resource: { buffer: this.buffers.velWX } },
                { binding: 4, resource: { buffer: this.buffers.velWY } },
                { binding: 5, resource: { buffer: this.buffers.flags } },
            ],
        });

        // --- Phase 2 pipelines ---
        this._phase2 = await createPhase2Pipelines(this.device);
        this._createPhase2BindGroups();

        // --- Phase 3: Ghost generation pipeline ---
        const ghostGen = await createGhostGenPipeline(this.device);
        this._ghostGenPipeline = ghostGen.pipeline;
        this._createGhostGenBindGroups(ghostGen.bindGroupLayouts);

        // --- Phase 3: Tree build pipelines ---
        this._treeBuild = await createTreeBuildPipelines(this.device);
        this._createTreeBuildBindGroups(this._treeBuild.bindGroupLayouts);

        // --- Phase 3: Tree force pipeline ---
        const treeForce = await createTreeForcePipeline(this.device);
        this._treeForcePipeline = treeForce.pipeline;
        this._createTreeForceBindGroups(treeForce.bindGroupLayouts);

        // --- Phase 3: Collision detection/resolution pipelines ---
        this._collisionPipelines = await createCollisionPipelines(this.device);
        this._createCollisionBindGroups(this._collisionPipelines.bindGroupLayouts);

        // --- Phase 3: Dead particle GC pipeline ---
        const deadGC = await createDeadGCPipeline(this.device);
        this._deadGCPipeline = deadGC.pipeline;
        this._createDeadGCBindGroup(deadGC.bindGroupLayouts);

        // --- Phase 4: Advanced physics pipelines ---
        this._phase4 = await createPhase4Pipelines(this.device);
        this._createPhase4BindGroups();

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

        // cacheDerived: uniforms + inputs + packed outputs (derived replaces magAngMom+invMassRadSq+vel+angVel, gamma removed)
        this._bg_cacheDerived = bg('cacheDerived', p2.cacheDerived.bindGroupLayouts[0],
            [this.uniformBuffer, b.mass, b.velWX, b.velWY, b.angW, b.charge,
             b.radius, b.derived, b.flags]);

        // pairForce: 4 bind groups (packed: derived, axYukMod, allForces, jerk)
        this._bg_pairForce0 = bg('pairForce_g0', p2.pairForce.bindGroupLayouts[0],
            [this.uniformBuffer]);
        this._bg_pairForce1 = bg('pairForce_g1', p2.pairForce.bindGroupLayouts[1],
            [b.posX, b.posY, b.mass, b.charge, b.derived, b.axYukMod, b.flags]);
        this._bg_pairForce2 = bg('pairForce_g2', p2.pairForce.bindGroupLayouts[2],
            [b.allForces]);
        this._bg_pairForce3 = bg('pairForce_g3', p2.pairForce.bindGroupLayouts[3],
            [b.jerk]);

        // externalFields (packed allForces)
        this._bg_extFields = bg('extFields', p2.externalFields.bindGroupLayouts[0],
            [this.uniformBuffer, b.mass, b.charge, b.flags, b.allForces]);

        // borisHalfKick (reads totalForce from allForces)
        this._bg_halfKick = bg('halfKick', p2.borisHalfKick.bindGroupLayouts[0],
            [this.uniformBuffer, b.velWX, b.velWY, b.mass, b.allForces,
             b.flags]);

        // borisRotate (reads bFields from allForces)
        this._bg_rotate = bg('rotate', p2.borisRotate.bindGroupLayouts[0],
            [this.uniformBuffer, b.velWX, b.velWY, b.charge, b.mass, b.allForces, b.flags]);

        // borisDrift (writes vel to derived)
        this._bg_drift = bg('drift', p2.borisDrift.bindGroupLayouts[0],
            [this.uniformBuffer, b.posX, b.posY, b.velWX, b.velWY, b.flags,
             b.derived]);

        // spinOrbit (packed derived, allForces)
        this._bg_spinOrbit = bg('spinOrbit', p2.spinOrbit.bindGroupLayouts[0],
            [this.uniformBuffer, b.velWX, b.velWY, b.angW, b.mass, b.charge,
             b.radius, b.derived, b.flags, b.allForces]);

        // applyTorques (reads torques from allForces, writes angVel to derived)
        this._bg_torques = bg('torques', p2.applyTorques.bindGroupLayouts[0],
            [this.uniformBuffer, b.angW, b.mass, b.radius, b.allForces, b.flags, b.derived]);
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

        // Group 0: read-only particle SoA inputs
        const group0 = bg('ghostGen_g0', layouts[0],
            [b.posX, b.posY, b.velWX, b.velWY, b.angW, b.mass, b.charge, b.flags]);

        // Group 1: ghost output SoA (dedicated ghost buffers to avoid aliasing)
        //   bindings 0-7: ghost output (read-write)
        //   bindings 8-9: particle derived inputs (read-only, packed magAngMom)
        //   bindings 10-11: ghost derived output (read-write, packed ghostMagAngMom)
        //   binding 12: particle ID input (read-only)
        //   binding 13: ghost particle ID output (read-write)
        const group1 = bg('ghostGen_g1', layouts[1],
            [b.ghostPosX, b.ghostPosY, b.ghostVelWX, b.ghostVelWY,
             b.ghostAngW, b.ghostMass, b.ghostCharge, b.ghostFlags,
             b.radius, b.derived,
             b.ghostRadius, b.ghostDerived,
             b.particleId, b.ghostParticleId]);

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
        const zero = new Uint32Array([0]);
        this.device.queue.writeBuffer(this.buffers.ghostCounter, 0, zero);

        const pass = encoder.beginComputePass({ label: 'ghostGen' });
        pass.setPipeline(this._ghostGenPipeline);
        pass.setBindGroup(0, this._ghostGenBindGroups[0]);
        pass.setBindGroup(1, this._ghostGenBindGroups[1]);
        pass.setBindGroup(2, this._ghostGenBindGroups[2]);
        pass.dispatchWorkgroups(Math.ceil(this.aliveCount / 64));
        pass.end();

        // Copy ghost data from dedicated buffers into main SoA arrays at offset aliveCount.
        // Uses previous frame's ghost count for copy size (1-frame latency, safe for tree build).
        const ghostCount = this._ghostCount;
        if (ghostCount > 0) {
            const ghostBytes = ghostCount * 4;
            const offset = this.aliveCount * 4;
            const b = this.buffers;
            encoder.copyBufferToBuffer(b.ghostPosX, 0, b.posX, offset, ghostBytes);
            encoder.copyBufferToBuffer(b.ghostPosY, 0, b.posY, offset, ghostBytes);
            encoder.copyBufferToBuffer(b.ghostVelWX, 0, b.velWX, offset, ghostBytes);
            encoder.copyBufferToBuffer(b.ghostVelWY, 0, b.velWY, offset, ghostBytes);
            encoder.copyBufferToBuffer(b.ghostAngW, 0, b.angW, offset, ghostBytes);
            encoder.copyBufferToBuffer(b.ghostMass, 0, b.mass, offset, ghostBytes);
            encoder.copyBufferToBuffer(b.ghostCharge, 0, b.charge, offset, ghostBytes);
            encoder.copyBufferToBuffer(b.ghostFlags, 0, b.flags, offset, ghostBytes);
            encoder.copyBufferToBuffer(b.ghostRadius, 0, b.radius, offset, ghostBytes);
            // ghostDerived is ParticleDerived struct (32 bytes per element)
            const derivedOffset = this.aliveCount * 32;
            const derivedBytes = ghostCount * 32;
            encoder.copyBufferToBuffer(b.ghostDerived, 0, b.derived, derivedOffset, derivedBytes);
            encoder.copyBufferToBuffer(b.ghostParticleId, 0, b.particleId, offset, ghostBytes);
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

        // Group 1: particle SoA inputs (read-only)
        this._treeBuildBG1 = this.device.createBindGroup({
            label: 'treeBuild_g1',
            layout: layouts[1],
            entries: [
                { binding: 0, resource: { buffer: b.posX } },
                { binding: 1, resource: { buffer: b.posY } },
                { binding: 2, resource: { buffer: b.velWX } },
                { binding: 3, resource: { buffer: b.velWY } },
                { binding: 4, resource: { buffer: b.mass } },
                { binding: 5, resource: { buffer: b.charge } },
                { binding: 6, resource: { buffer: b.derived } },
                { binding: 7, resource: { buffer: b.flags } },
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

        // Group 1: particle SoA (matches shader @group(1) bindings 0-13, packed derived + axYukMod)
        this._treeForceGroup1 = this.device.createBindGroup({
            label: 'treeForce_g1',
            layout: layouts[1],
            entries: [
                { binding: 0, resource: { buffer: b.posX } },
                { binding: 1, resource: { buffer: b.posY } },
                { binding: 2, resource: { buffer: b.velWX } },
                { binding: 3, resource: { buffer: b.velWY } },
                { binding: 4, resource: { buffer: b.mass } },
                { binding: 5, resource: { buffer: b.charge } },
                { binding: 6, resource: { buffer: b.angW } },
                { binding: 7, resource: { buffer: b.flags } },
                { binding: 8, resource: { buffer: b.radius } },
                { binding: 9, resource: { buffer: b.derived } },
                { binding: 10, resource: { buffer: b.axYukMod } },
                { binding: 11, resource: { buffer: b.particleId } },
                { binding: 12, resource: { buffer: b.ghostOriginalIdx } },
                { binding: 13, resource: { buffer: b.deathMass } },
            ],
        });

        // Group 2: allForces (packed)
        this._treeForceGroup2 = this.device.createBindGroup({
            label: 'treeForce_g2',
            layout: layouts[2],
            entries: [
                { binding: 0, resource: { buffer: b.allForces } },
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

        // Group 1: particle SoA (12 bindings)
        this._collisionBG1 = this.device.createBindGroup({
            label: 'collision_g1',
            layout: layouts[1],
            entries: [
                { binding: 0, resource: { buffer: b.posX } },
                { binding: 1, resource: { buffer: b.posY } },
                { binding: 2, resource: { buffer: b.velWX } },
                { binding: 3, resource: { buffer: b.velWY } },
                { binding: 4, resource: { buffer: b.angW } },
                { binding: 5, resource: { buffer: b.mass } },
                { binding: 6, resource: { buffer: b.baseMass } },
                { binding: 7, resource: { buffer: b.charge } },
                { binding: 8, resource: { buffer: b.flags } },
                { binding: 9, resource: { buffer: b.radius } },
                { binding: 10, resource: { buffer: b.particleId } },
                { binding: 11, resource: { buffer: b.ghostOriginalIdx } },
            ],
        });

        // Group 2: collision pairs + counters + merge results + death metadata
        this._collisionBG2 = this.device.createBindGroup({
            label: 'collision_g2',
            layout: layouts[2],
            entries: [
                { binding: 0, resource: { buffer: b.collisionPairBuffer } },
                { binding: 1, resource: { buffer: b.collisionPairCounter } },
                { binding: 2, resource: { buffer: b.mergeResultBuffer } },
                { binding: 3, resource: { buffer: b.mergeResultCounter } },
                { binding: 4, resource: { buffer: b.deathTime } },
                { binding: 5, resource: { buffer: b.deathMass } },
                { binding: 6, resource: { buffer: b.deathAngVel } },
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
                { binding: 0, resource: { buffer: b.flags } },
                { binding: 1, resource: { buffer: b.deathTime } },
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
            this.uniformBuffer, b.posX, b.posY, b.velWX, b.velWY, b.angW, b.flags,
        ];
        this._phase4BindGroups.historyG0 = null; // lazy
        this._phase4BindGroups.historyG1 = null; // lazy

        // ── 1PN (compute1PN + vvKick1PN share bind groups) ──
        this._phase4BindGroups.onePNG0 = bg('onePN_g0', p4.compute1PN.bindGroupLayouts[0],
            [this.uniformBuffer]);
        this._phase4BindGroups.onePNG1 = bg('onePN_g1', p4.compute1PN.bindGroupLayouts[1],
            [b.posX, b.posY, b.velWX, b.velWY, b.mass, b.charge, b.flags, b.axYukMod, b.derived]);
        this._phase4BindGroups.onePNG2 = bg('onePN_g2', p4.compute1PN.bindGroupLayouts[2],
            [b.allForces, b.f1pnOld, b.velWX, b.velWY]);

        // ── Radiation (lamrorRadiation, hawkingRadiation, pionEmission share bind groups) ──
        this._phase4BindGroups.radG0 = bg('radiation_g0', p4.lamrorRadiation.bindGroupLayouts[0],
            [this.uniformBuffer]);
        // Group 1: particle state (17 bindings)
        // binding 14: jerkInterleaved [x0,y0,x1,y1...]
        // bindings 15-16: separate yukForceX/Y buffers
        this._phase4BindGroups.radG1 = bg('radiation_g1', p4.lamrorRadiation.bindGroupLayouts[1],
            [b.posX, b.posY, b.velWX, b.velWY, b.mass, b.charge, b.flags,
             b.derived, b.baseMass, b.radius, b.angW, b.particleId,
             b.allForces, b.jerk,
             b.yukForceX, b.yukForceY]);
        this._phase4BindGroups.radG2 = bg('radiation_g2', p4.lamrorRadiation.bindGroupLayouts[2],
            [b.radAccum, b.hawkAccum, b.yukawaRadAccum, b.radDisplayX, b.radDisplayY]);
        // Group 3: photon pool (9) + pion pool (11) + charge_rw (1) = 21 bindings
        this._phase4BindGroups.radG3 = bg('radiation_g3', p4.lamrorRadiation.bindGroupLayouts[3],
            [b.phPosX, b.phPosY, b.phVelX, b.phVelY, b.phEnergy, b.phEmitterId, b.phAge, b.phFlags, b.phCount,
             b.piPosX, b.piPosY, b.piWX, b.piWY, b.piMass, b.piCharge, b.piEnergy, b.piEmitterId, b.piAge, b.piFlags, b.piCount,
             b.charge]);

        // ── Bosons (updatePhotons, updatePions, absorbPhotons, absorbPions, decayPions) ──
        this._phase4BindGroups.bosG0 = bg('bosons_g0', p4.updatePhotons.bindGroupLayouts[0],
            [this.uniformBuffer, b.poolMgmt]);
        this._phase4BindGroups.bosG1 = bg('bosons_g1', p4.updatePhotons.bindGroupLayouts[1],
            [b.posX, b.posY, b.mass, b.radius, b.flags, b.particleId,
             b.velWX, b.velWY, b.charge, b.baseMass, b.angW]);
        this._phase4BindGroups.bosG2 = bg('bosons_g2', p4.updatePhotons.bindGroupLayouts[2],
            [b.phPosX, b.phPosY, b.phVelX, b.phVelY, b.phEnergy, b.phEmitterId, b.phAge, b.phFlags, b.phCount]);
        this._phase4BindGroups.bosG3 = bg('bosons_g3', p4.updatePhotons.bindGroupLayouts[3],
            [b.piPosX, b.piPosY, b.piWX, b.piWY, b.piMass, b.piCharge, b.piEnergy, b.piEmitterId, b.piAge, b.piFlags, b.piCount]);

        // ── Boson Tree (insertBosonsIntoTree, computeBosonAggregates, computeBosonGravity, applyBosonBosonGravity) ──
        this._phase4BindGroups.btG0 = bg('bosonTree_g0', p4.insertBosonsIntoTree.bindGroupLayouts[0],
            [this.uniformBuffer]);
        this._phase4BindGroups.btG1 = bg('bosonTree_g1', p4.insertBosonsIntoTree.bindGroupLayouts[1],
            [b.bosonTreeNodes, b.bosonTreeCounter]);
        this._phase4BindGroups.btG2 = bg('bosonTree_g2', p4.insertBosonsIntoTree.bindGroupLayouts[2],
            [b.phPosX, b.phPosY, b.phVelX, b.phVelY, b.phEnergy, b.phFlags, b.phCount]);
        this._phase4BindGroups.btG3 = bg('bosonTree_g3', p4.insertBosonsIntoTree.bindGroupLayouts[3],
            [b.piPosX, b.piPosY, b.piWX, b.piWY, b.piMass, b.piFlags, b.piCount]);
        this._phase4BindGroups.btG4 = bg('bosonTree_g4', p4.insertBosonsIntoTree.bindGroupLayouts[4],
            [b.posX, b.posY, b.mass, b.flags, b.allForces]);
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
            [this.uniformBuffer, b.posX, b.posY, b.velWX, b.velWY, b.angW, b.flags]);
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
        const passLarmor = encoder.beginComputePass({ label: 'lamrorRadiation' });
        passLarmor.setPipeline(p4.lamrorRadiation.pipeline);
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
     */
    _dispatchBosonUpdate(encoder) {
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
        this.device.queue.writeBuffer(b.bosonTreeCounter, 0, new Uint32Array([1]));

        // Initialize root node bounds (will be set by insertBosonsIntoTree)
        // For boson tree we use domain bounds directly
        const rootInit = new Uint32Array(20);
        const rootF32 = new Float32Array(rootInit.buffer);
        rootF32[0] = 0;              // minX
        rootF32[1] = 0;              // minY
        rootF32[2] = this.domainW;   // maxX
        rootF32[3] = this.domainH;   // maxY
        this.device.queue.writeBuffer(b.bosonTreeNodes, 0, rootInit);

        const totalBosons = MAX_PHOTONS + MAX_PIONS;
        const bosonWG = Math.ceil(totalBosons / 64);

        // insertBosonsIntoTree
        const passInsert = encoder.beginComputePass({ label: 'insertBosonsIntoTree' });
        passInsert.setPipeline(p4.insertBosonsIntoTree.pipeline);
        passInsert.setBindGroup(0, bgs.btG0);
        passInsert.setBindGroup(1, bgs.btG1);
        passInsert.setBindGroup(2, bgs.btG2);
        passInsert.setBindGroup(3, bgs.btG3);
        passInsert.setBindGroup(4, bgs.btG4);
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
        passAgg.setBindGroup(4, bgs.btG4);
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
            passGrav.setBindGroup(4, bgs.btG4);
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
        passBosonBoson.setBindGroup(4, bgs.btG4);
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
     * Only runs when collisionMode === COL_MERGE (1).
     * After drift + boundary, detects overlapping pairs via tree query,
     * then resolves merges/annihilations.
     */
    _dispatchCollisions(encoder) {
        if (this._collisionMode !== COL_MERGE) return;
        if (!this._barnesHutEnabled) return; // collision detection requires tree
        if (this.aliveCount === 0) return;

        const b = this.buffers;

        // Reset pair counter and merge counter to 0
        const zero = new Uint32Array([0]);
        this.device.queue.writeBuffer(b.collisionPairCounter, 0, zero);
        this.device.queue.writeBuffer(b.mergeResultCounter, 0, zero);

        // Dispatch detectCollisions: one thread per alive particle
        const detectWG = Math.ceil(this.aliveCount / 64);
        const passDetect = encoder.beginComputePass({ label: 'detectCollisions' });
        passDetect.setPipeline(this._collisionPipelines.detectCollisions);
        passDetect.setBindGroup(0, this._collisionBG0);
        passDetect.setBindGroup(1, this._collisionBG1);
        passDetect.setBindGroup(2, this._collisionBG2);
        passDetect.dispatchWorkgroups(detectWG);
        passDetect.end();

        // Dispatch resolveCollisions: conservatively dispatch for maxParticles
        // (pairCounter checked inside shader to skip excess threads)
        const resolveWG = Math.ceil(this.buffers.maxParticles / 64);
        const passResolve = encoder.beginComputePass({ label: 'resolveCollisions' });
        passResolve.setPipeline(this._collisionPipelines.resolveCollisions);
        passResolve.setBindGroup(0, this._collisionBG0);
        passResolve.setBindGroup(1, this._collisionBG1);
        passResolve.setBindGroup(2, this._collisionBG2);
        passResolve.dispatchWorkgroups(resolveWG);
        passResolve.end();
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
        this.device.queue.writeBuffer(b.qtNodeCounter, 0, new Uint32Array([1]));

        // 2. Reset bounds: minX=INT_MAX, minY=INT_MAX, maxX=INT_MIN, maxY=INT_MIN
        this.device.queue.writeBuffer(b.qtBoundsBuffer, 0, new Int32Array([
            2147483647,   // minX = i32 max
            2147483647,   // minY = i32 max
            -2147483647,  // maxX = i32 min
            -2147483647,  // maxY = i32 min
        ]));

        // 3. Clear visitor flags to 0 (write zeros for all nodes)
        // Only clear up to a reasonable bound based on expected tree size
        const clearSize = Math.min(b.QT_MAX_NODES, totalCount * 6) * 4;
        const zeroData = new Uint8Array(clearSize);
        this.device.queue.writeBuffer(b.qtVisitorFlags, 0, zeroData);

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
     * Phase 1: writes directly via queue.writeBuffer (no free stack yet).
     */
    addParticle({ x, y, vx = 0, vy = 0, mass: m = 1, charge: q = 0 }) {
        const idx = this.aliveCount;
        if (idx >= MAX_PARTICLES) return -1;

        const f32 = new Float32Array([0]);
        const u32 = new Uint32Array([0]);

        f32[0] = x; this.device.queue.writeBuffer(this.buffers.posX, idx * 4, f32);
        f32[0] = y; this.device.queue.writeBuffer(this.buffers.posY, idx * 4, f32);
        f32[0] = vx; this.device.queue.writeBuffer(this.buffers.velWX, idx * 4, f32);
        f32[0] = vy; this.device.queue.writeBuffer(this.buffers.velWY, idx * 4, f32);
        f32[0] = 0; this.device.queue.writeBuffer(this.buffers.angW, idx * 4, f32);
        f32[0] = m; this.device.queue.writeBuffer(this.buffers.mass, idx * 4, f32);
        f32[0] = m; this.device.queue.writeBuffer(this.buffers.baseMass, idx * 4, f32);
        f32[0] = q; this.device.queue.writeBuffer(this.buffers.charge, idx * 4, f32);
        // radius is in its own buffer (read by renderer)
        if (this.buffers.radius) {
            f32[0] = Math.cbrt(m); this.device.queue.writeBuffer(this.buffers.radius, idx * 4, f32);
        }
        // gamma was packed into particleDerived — computed by cacheDerived shader
        // No need to init here; cacheDerived runs before forces each substep.
        u32[0] = FLAG_ALIVE; this.device.queue.writeBuffer(this.buffers.flags, idx * 4, u32);

        // Pack color: neutral slate = #8A7E72 -> RGBA
        u32[0] = 0xFF727E8A; // ABGR packed
        this.device.queue.writeBuffer(this.buffers.color, idx * 4, u32);

        this.aliveCount++;
        return idx;
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
        this._yukawaCoupling = physics.yukawaEnabled ? 14 : 14;  // YUKAWA_COUPLING constant

        // Lazily allocate history buffers when relativity is first enabled
        if (this._relativityEnabled && this._phase4) {
            this._ensureHistoryBindGroups();
        }
        this._yukawaMu = physics.yukawaMu;
        this._higgsMass = physics.higgsEnabled ? 0.5 : 0.5;
        this._axionMass = physics.axionMass;
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

        this._sgInvRUploaded[which] = false;
    }

    /**
     * Lazily initialize Phase 5 pipelines on first use.
     * Called when any Phase 5 feature is first enabled.
     */
    async _ensurePhase5Pipelines() {
        if (this._fieldDeposit) return; // already initialized

        // Initialize all Phase 5 pipelines in parallel
        const [deposit, evolve, forces, selfGrav, excitation, heatmap, expansion, disint, pairProd] =
            await Promise.all([
                createFieldDepositPipelines(this.device),
                createFieldEvolvePipelines(this.device),
                createFieldForcesPipelines(this.device),
                createFieldSelfGravPipelines(this.device),
                createFieldExcitationPipeline(this.device),
                createHeatmapPipelines(this.device),
                createExpansionPipeline(this.device),
                createDisintegrationPipeline(this.device),
                createPairProductionPipeline(this.device),
            ]);

        this._fieldDeposit = deposit;
        this._fieldEvolve = evolve;
        this._fieldForces = forces;
        this._fieldSelfGrav = selfGrav;
        this._fieldExcitation = excitation;
        this._heatmapPipelines = heatmap;
        this._expansionPipeline = expansion;
        this._disintPipeline = disint;
        this._pairProdPipeline = pairProd;
    }

    /**
     * Build sgInvR table for field self-gravity (CPU-side O(SG^4=4096) computation).
     * Uploaded once per field, rebuilt on domain size change.
     */
    _buildSgInvRTable(which) {
        const fb = which === 'higgs' ? this._higgsBuffers : this._axionBuffers;
        if (!fb) return;

        const SG = COARSE_RES;
        const SG_SQ = COARSE_SQ;
        const cellW = this.domainW / SG;
        const cellH = this.domainH / SG;
        const table = new Float32Array(SG_SQ * SG_SQ);

        for (let i = 0; i < SG_SQ; i++) {
            const ix = i % SG;
            const iy = (i / SG) | 0;
            const cx = (ix + 0.5) * cellW;
            const cy = (iy + 0.5) * cellH;
            const rowBase = i * SG_SQ;
            for (let j = 0; j < SG_SQ; j++) {
                if (i === j) {
                    table[rowBase + j] = 0;
                    continue;
                }
                const jx = j % SG;
                const jy = (j / SG) | 0;
                const dx = cx - (jx + 0.5) * cellW;
                const dy = cy - (jy + 0.5) * cellH;
                table[rowBase + j] = 1.0 / Math.sqrt(dx * dx + dy * dy);
            }
        }

        this.device.queue.writeBuffer(fb.sgInvR, 0, table);
        this._sgInvRUploaded[which] = true;
    }

    /**
     * Write FieldUniforms to the shared field uniform buffer.
     */
    _writeFieldUniforms(dt) {
        if (!this._fieldUniformBuffer) return;
        const data = new ArrayBuffer(256);
        const f = new Float32Array(data);
        const u = new Uint32Array(data);
        f[0] = dt;
        f[1] = this.domainW;
        f[2] = this.domainH;
        u[3] = this.boundaryMode;
        u[4] = this.topologyMode;
        f[5] = this._higgsMass;
        f[6] = this._axionMass;
        f[7] = this._higgsCoupling;
        f[8] = this._axionCoupling;
        u[9] = this.aliveCount;
        u[10] = this._fieldGravEnabled ? 1 : 0;
        this.device.queue.writeBuffer(this._fieldUniformBuffer, 0, data);
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

        // Group 0: particle SoA
        const g0 = this.device.createBindGroup({
            label: `fieldDeposit_g0_${which}`,
            layout: dep.bindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: b.posX } },
                { binding: 1, resource: { buffer: b.posY } },
                { binding: 2, resource: { buffer: b.mass } },
                { binding: 3, resource: { buffer: b.baseMass } },
                { binding: 4, resource: { buffer: b.charge } },
                { binding: 5, resource: { buffer: b.flags } },
                { binding: 6, resource: { buffer: b.velWX } },
                { binding: 7, resource: { buffer: b.velWY } },
            ],
        });

        // Group 1: scratch + target grid + uniforms (for source deposition)
        const g1Source = this.device.createBindGroup({
            label: `fieldDeposit_g1_source_${which}`,
            layout: dep.bindGroupLayouts[1],
            entries: [
                { binding: 0, resource: { buffer: this._pqsScratch } },
                { binding: 1, resource: { buffer: this._pqsIndices } },
                { binding: 2, resource: { buffer: fb.source } },
                { binding: 3, resource: { buffer: this._fieldUniformBuffer } },
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
                { binding: 3, resource: { buffer: this._fieldUniformBuffer } },
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
                { binding: 10, resource: { buffer: this._fieldUniformBuffer } },
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
                { binding: 10, resource: { buffer: this._fieldUniformBuffer } },
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

        this._fieldSelfGravBGs[which] = this.device.createBindGroup({
            label: `fieldSelfGrav_${which}`,
            layout: this._fieldSelfGrav.bindGroupLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: fb.field } },
                { binding: 1, resource: { buffer: fb.fieldDot } },
                { binding: 2, resource: { buffer: fb.gradX } },
                { binding: 3, resource: { buffer: fb.gradY } },
                { binding: 4, resource: { buffer: fb.energyDensity } },
                { binding: 5, resource: { buffer: fb.coarseRho } },
                { binding: 6, resource: { buffer: fb.coarsePhi } },
                { binding: 7, resource: { buffer: fb.sgPhiFull } },
                { binding: 8, resource: { buffer: fb.sgGradX } },
                { binding: 9, resource: { buffer: fb.sgGradY } },
                { binding: 10, resource: { buffer: fb.sgInvR } },
                { binding: 11, resource: { buffer: this._fieldUniformBuffer } },
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
                { binding: 0, resource: { buffer: b.posX } },
                { binding: 1, resource: { buffer: b.posY } },
                { binding: 2, resource: { buffer: b.mass } },
                { binding: 3, resource: { buffer: b.baseMass } },
                { binding: 4, resource: { buffer: b.charge } },
                { binding: 5, resource: { buffer: b.flags } },
                { binding: 6, resource: { buffer: b.velWX } },
                { binding: 7, resource: { buffer: b.velWY } },
                { binding: 8, resource: { buffer: b.angW } },
                { binding: 9, resource: { buffer: b.radius } },
                { binding: 10, resource: { buffer: b.derived } },
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
            if (!this._sgInvRUploaded[which]) {
                this._buildSgInvRTable(which);
            }
            const sgBG = this._fieldSelfGravBGs[which];
            const sg = this._fieldSelfGrav;
            const coarseWG = 1; // 8x8 = one workgroup for 8x8 coarse grid

            // Energy density
            const edPipeline = which === 'higgs'
                ? sg.computeEnergyDensityHiggs
                : sg.computeEnergyDensityAxion;
            {
                const p = encoder.beginComputePass({ label: `energyDensity_${which}` });
                p.setPipeline(edPipeline);
                p.setBindGroup(0, sgBG);
                p.dispatchWorkgroups(gridWG, gridWG);
                p.end();
            }
            // Downsample
            {
                const p = encoder.beginComputePass({ label: `downsampleRho_${which}` });
                p.setPipeline(sg.downsampleRho);
                p.setBindGroup(0, sgBG);
                p.dispatchWorkgroups(coarseWG, coarseWG);
                p.end();
            }
            // Coarse potential
            {
                const p = encoder.beginComputePass({ label: `coarsePotential_${which}` });
                p.setPipeline(sg.computeCoarsePotential);
                p.setBindGroup(0, sgBG);
                p.dispatchWorkgroups(coarseWG, coarseWG);
                p.end();
            }
            // Upsample
            {
                const p = encoder.beginComputePass({ label: `upsamplePhi_${which}` });
                p.setPipeline(sg.upsamplePhi);
                p.setBindGroup(0, sgBG);
                p.dispatchWorkgroups(gridWG, gridWG);
                p.end();
            }
            // SG gradients
            {
                const p = encoder.beginComputePass({ label: `sgGradients_${which}` });
                p.setPipeline(sg.computeSelfGravGradients);
                p.setBindGroup(0, sgBG);
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
                this._fieldExcitationBGs[which] = this.device.createBindGroup({
                    label: `fieldExcitation_${which}`,
                    layout: exc.bindGroupLayouts[0],
                    entries: [
                        { binding: 0, resource: { buffer: fb.fieldDot } },
                        { binding: 1, resource: { buffer: this._excitationBuffers.events } },
                        { binding: 2, resource: { buffer: this._fieldUniformBuffer } },
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

        // Write ExpansionUniforms
        const data = new ArrayBuffer(32);
        const f = new Float32Array(data);
        const u = new Uint32Array(data);
        f[0] = this._hubbleParam;
        f[1] = dt;
        f[2] = this.domainW * 0.5;
        f[3] = this.domainH * 0.5;
        u[4] = this.aliveCount;
        this.device.queue.writeBuffer(this._expansionUniformBuffer, 0, data);

        if (!this._expansionBG) {
            const b = this.buffers;
            this._expansionBG = this.device.createBindGroup({
                label: 'expansion_g0',
                layout: this._expansionPipeline.bindGroupLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: b.posX } },
                    { binding: 1, resource: { buffer: b.posY } },
                    { binding: 2, resource: { buffer: b.velWX } },
                    { binding: 3, resource: { buffer: b.velWY } },
                    { binding: 4, resource: { buffer: b.flags } },
                    { binding: 5, resource: { buffer: this._expansionUniformBuffer } },
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

        // Write DisintUniforms
        const data = new ArrayBuffer(48);
        const f = new Float32Array(data);
        const u = new Uint32Array(data);
        f[0] = this._blackHoleEnabled ? 16 : 64; // softeningSq
        f[1] = this.domainW;
        f[2] = this.domainH;
        f[3] = 0.3;   // tidalStrength
        f[4] = 0.9;   // rocheThreshold
        f[5] = 0.01;  // rocheTransferRate
        f[6] = 0.01;  // minMass
        u[7] = 4;     // spawnCount
        u[8] = this.aliveCount;
        u[9] = this.boundaryMode === BOUND_LOOP ? 1 : 0;
        u[10] = this.topologyMode;
        this.device.queue.writeBuffer(this._disintUniformBuffer, 0, data);

        // Reset event counter
        this.device.queue.writeBuffer(this._disintBuffers.counter, 0, new Uint32Array([0]));

        if (!this._disintBGs) {
            const b = this.buffers;
            const g0 = this.device.createBindGroup({
                label: 'disint_g0',
                layout: this._disintPipeline.bindGroupLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: b.posX } },
                    { binding: 1, resource: { buffer: b.posY } },
                    { binding: 2, resource: { buffer: b.mass } },
                    { binding: 3, resource: { buffer: b.charge } },
                    { binding: 4, resource: { buffer: b.radius } },
                    { binding: 5, resource: { buffer: b.derived } },
                    { binding: 6, resource: { buffer: b.flags } },
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

        // Write PairProdUniforms
        const data = new ArrayBuffer(48);
        const f = new Float32Array(data);
        const u = new Uint32Array(data);
        f[0] = 0.5;     // minEnergy
        f[1] = 8.0;     // proximity
        f[2] = 0.005;   // probability
        u[3] = 64;      // minAge
        u[4] = 32;      // maxParticles (PAIR_PROD_MAX_PARTICLES)
        u[5] = this.aliveCount;
        u[6] = MAX_PHOTONS;
        u[7] = this._blackHoleEnabled ? 1 : 0;
        f[8] = this.simTime;
        this.device.queue.writeBuffer(this._pairProdUniformBuffer, 0, data);

        // Reset pair counter
        this.device.queue.writeBuffer(this._pairProdBuffers.counter, 0, new Uint32Array([0]));

        if (!this._pairProdBGs) {
            const b = this.buffers;
            const g0 = this.device.createBindGroup({
                label: 'pairProd_g0',
                layout: this._pairProdPipeline.bindGroupLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: b.phPosX } },
                    { binding: 1, resource: { buffer: b.phPosY } },
                    { binding: 2, resource: { buffer: b.phEnergy } },
                    { binding: 3, resource: { buffer: b.phVelX } },
                    { binding: 4, resource: { buffer: b.phVelY } },
                    { binding: 5, resource: { buffer: b.phAge } },
                    { binding: 6, resource: { buffer: b.phFlags } },
                ],
            });
            const g1 = this.device.createBindGroup({
                label: 'pairProd_g1',
                layout: this._pairProdPipeline.bindGroupLayouts[1],
                entries: [
                    { binding: 0, resource: { buffer: b.posX } },
                    { binding: 1, resource: { buffer: b.posY } },
                    { binding: 2, resource: { buffer: b.mass } },
                    { binding: 3, resource: { buffer: b.flags } },
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
        const data = new ArrayBuffer(96);
        const f = new Float32Array(data);
        const u = new Uint32Array(data);
        f[0] = viewLeft;
        f[1] = viewTop;
        f[2] = cellW;
        f[3] = cellH;
        f[4] = this._blackHoleEnabled ? 16 : 64; // softeningSq
        f[5] = this._yukawaCoupling;
        f[6] = this._yukawaMu;
        f[7] = this.simTime;
        f[8] = this.domainW;
        f[9] = this.domainH;
        u[10] = (this._toggles0 & 1) ? 1 : 0; // doGravity
        u[11] = (this._toggles0 & 2) ? 1 : 0; // doCoulomb
        u[12] = (this._toggles0 & 2048) ? 1 : 0; // doYukawa
        u[13] = 0; // useDelay (not on GPU)
        u[14] = this.boundaryMode === BOUND_LOOP ? 1 : 0;
        u[15] = this.topologyMode;
        u[16] = this.aliveCount;
        u[17] = 0; // deadCount (not tracked on GPU yet)
        this.device.queue.writeBuffer(this._heatmapUniformBuffer, 0, data);

        if (!this._heatmapBGs) {
            const b = this.buffers;
            const hm = this._heatmapPipelines;
            const hmBuf = this._heatmapBuffers;
            this._heatmapBGs = {
                g0: this.device.createBindGroup({
                    label: 'heatmap_g0',
                    layout: hm.heatmapLayouts[0],
                    entries: [
                        { binding: 0, resource: { buffer: b.posX } },
                        { binding: 1, resource: { buffer: b.posY } },
                        { binding: 2, resource: { buffer: b.mass } },
                        { binding: 3, resource: { buffer: b.charge } },
                        { binding: 4, resource: { buffer: b.flags } },
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
            };
        }

        const gridWG = Math.ceil(64 / 8);

        // Compute heatmap
        {
            const p = encoder.beginComputePass({ label: 'computeHeatmap' });
            p.setPipeline(this._heatmapPipelines.computeHeatmap);
            p.setBindGroup(0, this._heatmapBGs.g0);
            p.setBindGroup(1, this._heatmapBGs.g1);
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
            this._histStride++;
            if (this._relativityEnabled && this._histStride >= HISTORY_STRIDE) {
                this._histStride = 0;
                this._dispatchRecordHistory(encoder);
            }

            this.device.queue.submit([encoder.finish()]);
        }

        // Readback maxAccel for next frame (non-blocking)
        this._readbackMaxAccel();

        // Readback ghost count for next frame (non-blocking)
        this._readbackGhostCount();

        // Readback merge results for photon bursts / field excitations (non-blocking)
        this._readbackMergeResults();
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
        const p2 = this._phase2;

        const encoder = this.device.createCommandEncoder({ label: 'physics-phase2' });

        // Pass 0: ghost generation (Phase 3 — before tree build)
        this._dispatchGhostGen(encoder);

        // Pass 0b: tree build (Phase 3 — after ghost gen, before force computation)
        this._dispatchTreeBuild(encoder);

        // Pass 1: resetForces
        const pass1 = encoder.beginComputePass({ label: 'resetForces' });
        pass1.setPipeline(p2.resetForces.pipeline);
        pass1.setBindGroup(0, this._bg_resetForces);
        pass1.dispatchWorkgroups(workgroups);
        pass1.end();

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

        // Pass 5b: externalFields
        const pass5b = encoder.beginComputePass({ label: 'externalFields' });
        pass5b.setPipeline(p2.externalFields.pipeline);
        pass5b.setBindGroup(0, this._bg_extFields);
        pass5b.dispatchWorkgroups(workgroups);
        pass5b.end();

        // Pass 6: borisHalfKick (first)
        const pass6 = encoder.beginComputePass({ label: 'halfKick1' });
        pass6.setPipeline(p2.borisHalfKick.pipeline);
        pass6.setBindGroup(0, this._bg_halfKick);
        pass6.dispatchWorkgroups(workgroups);
        pass6.end();

        // Pass 7: borisRotation
        const pass7 = encoder.beginComputePass({ label: 'borisRotate' });
        pass7.setPipeline(p2.borisRotate.pipeline);
        pass7.setBindGroup(0, this._bg_rotate);
        pass7.dispatchWorkgroups(workgroups);
        pass7.end();

        // Pass 8: borisHalfKick (second — reuse same pipeline + bind group)
        const pass8 = encoder.beginComputePass({ label: 'halfKick2' });
        pass8.setPipeline(p2.borisHalfKick.pipeline);
        pass8.setBindGroup(0, this._bg_halfKick);
        pass8.dispatchWorkgroups(workgroups);
        pass8.end();

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

        // Pass 12: borisDrift
        const pass12 = encoder.beginComputePass({ label: 'borisDrift' });
        pass12.setPipeline(p2.borisDrift.pipeline);
        pass12.setBindGroup(0, this._bg_drift);
        pass12.dispatchWorkgroups(workgroups);
        pass12.end();

        // Pass 24: boundary
        const passBoundary = encoder.beginComputePass({ label: 'boundary' });
        passBoundary.setPipeline(this._boundaryPipeline);
        passBoundary.setBindGroup(0, this._boundaryBindGroup);
        passBoundary.dispatchWorkgroups(workgroups);
        passBoundary.end();

        // Submit core physics (forces + Boris + drift + boundary) — MUST succeed
        this.device.queue.submit([encoder.finish()]);

        // ── Advanced passes (Phase 3-5) — isolated so failures don't affect core ──
        // Each group in its own try/catch + command encoder
        try {
            const enc2 = this.device.createCommandEncoder({ label: 'advanced-physics' });

            // Pass 11: radiation reaction (Larmor, Hawking, pion emission) — Phase 4
            this._dispatchRadiation(enc2);

            // Pass 13: cosmological expansion — Phase 5
            this._dispatchExpansion(enc2, dtSub);

            // Pass 14: 1PN velocity-Verlet correction — Phase 4
            this._dispatch1PNVV(enc2);

            // Pass 15: scalar field evolution (Higgs, Axion) — Phase 5
            if (this._fieldDeposit) {
                this._writeFieldUniforms(dtSub);
                if (this._higgsEnabled) this._dispatchFieldEvolve(enc2, 'higgs', dtSub);
                if (this._axionEnabled) this._dispatchFieldEvolve(enc2, 'axion', dtSub);
            }

            // Pass 16: scalar field forces — Phase 5
            this._dispatchFieldForces(enc2);

            // Pass 17-18: collision detection + resolution (Phase 3)
            this._dispatchCollisions(enc2);

            // Pass 19: field excitations from merge events — Phase 5
            this._dispatchFieldExcitations(enc2);

            // Pass 20: disintegration check — Phase 5
            this._dispatchDisintegration(enc2);

            // Pass 21: boson update (photon/pion drift, absorption, decay) — Phase 4
            this._dispatchBosonUpdate(enc2);

            // Pass 22: pair production — Phase 5
            this._dispatchPairProduction(enc2);

            this.device.queue.submit([enc2.finish()]);
        } catch (e) {
            // Advanced passes failed — core physics still ran
        }
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

        this._maxAccel = data[0];
        this._maxAccelPending = false;
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
        const byteLen = count * 4;

        // Create staging buffers for readback
        const stagingPosX = this.device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const stagingPosY = this.device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const stagingWX = this.device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const stagingWY = this.device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const stagingMass = this.device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const stagingBaseMass = this.device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const stagingCharge = this.device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const stagingAngW = this.device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const stagingFlags = this.device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

        const encoder = this.device.createCommandEncoder({ label: 'serialize-readback' });
        encoder.copyBufferToBuffer(this.buffers.posX, 0, stagingPosX, 0, byteLen);
        encoder.copyBufferToBuffer(this.buffers.posY, 0, stagingPosY, 0, byteLen);
        encoder.copyBufferToBuffer(this.buffers.velWX, 0, stagingWX, 0, byteLen);
        encoder.copyBufferToBuffer(this.buffers.velWY, 0, stagingWY, 0, byteLen);
        encoder.copyBufferToBuffer(this.buffers.mass, 0, stagingMass, 0, byteLen);
        encoder.copyBufferToBuffer(this.buffers.baseMass, 0, stagingBaseMass, 0, byteLen);
        encoder.copyBufferToBuffer(this.buffers.charge, 0, stagingCharge, 0, byteLen);
        encoder.copyBufferToBuffer(this.buffers.angW, 0, stagingAngW, 0, byteLen);
        encoder.copyBufferToBuffer(this.buffers.flags, 0, stagingFlags, 0, byteLen);
        this.device.queue.submit([encoder.finish()]);

        // Map all staging buffers
        await Promise.all([
            stagingPosX.mapAsync(GPUMapMode.READ),
            stagingPosY.mapAsync(GPUMapMode.READ),
            stagingWX.mapAsync(GPUMapMode.READ),
            stagingWY.mapAsync(GPUMapMode.READ),
            stagingMass.mapAsync(GPUMapMode.READ),
            stagingBaseMass.mapAsync(GPUMapMode.READ),
            stagingCharge.mapAsync(GPUMapMode.READ),
            stagingAngW.mapAsync(GPUMapMode.READ),
            stagingFlags.mapAsync(GPUMapMode.READ),
        ]);

        const posX = new Float32Array(stagingPosX.getMappedRange());
        const posY = new Float32Array(stagingPosY.getMappedRange());
        const wx = new Float32Array(stagingWX.getMappedRange());
        const wy = new Float32Array(stagingWY.getMappedRange());
        const mass = new Float32Array(stagingMass.getMappedRange());
        const baseMass = new Float32Array(stagingBaseMass.getMappedRange());
        const charge = new Float32Array(stagingCharge.getMappedRange());
        const angw = new Float32Array(stagingAngW.getMappedRange());
        const flags = new Uint32Array(stagingFlags.getMappedRange());

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
            if (!(flags[i] & FLAG_ALIVE)) continue;
            state.particles.push({
                x: posX[i], y: posY[i],
                wx: wx[i], wy: wy[i],
                mass: mass[i], baseMass: baseMass[i],
                charge: charge[i], angw: angw[i],
                antimatter: !!(flags[i] & FLAG_ANTIMATTER),
            });
        }

        // Unmap and destroy staging buffers
        stagingPosX.unmap(); stagingPosX.destroy();
        stagingPosY.unmap(); stagingPosY.destroy();
        stagingWX.unmap(); stagingWX.destroy();
        stagingWY.unmap(); stagingWY.destroy();
        stagingMass.unmap(); stagingMass.destroy();
        stagingBaseMass.unmap(); stagingBaseMass.destroy();
        stagingCharge.unmap(); stagingCharge.destroy();
        stagingAngW.unmap(); stagingAngW.destroy();
        stagingFlags.unmap(); stagingFlags.destroy();

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

        // Upload particles to GPU
        for (const pd of state.particles) {
            this.addParticle({
                x: pd.x, y: pd.y,
                vx: pd.wx, vy: pd.wy,
                mass: pd.mass,
                charge: pd.charge,
            });
            // Set baseMass, angw, antimatter flags for the last added particle
            const idx = this.aliveCount - 1;
            const f32 = new Float32Array([0]);
            const u32 = new Uint32Array([0]);
            f32[0] = pd.baseMass ?? pd.mass;
            this.device.queue.writeBuffer(this.buffers.baseMass, idx * 4, f32);
            f32[0] = pd.angw || 0;
            this.device.queue.writeBuffer(this.buffers.angW, idx * 4, f32);
            if (pd.antimatter) {
                u32[0] = FLAG_ALIVE | FLAG_ANTIMATTER;
                this.device.queue.writeBuffer(this.buffers.flags, idx * 4, u32);
            }
        }

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
        this.resetFields();
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
            ghostCount: this._ghostCount,
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
            extGravX: this._extGravity * Math.cos(this._extGravityAngle),
            extGravY: this._extGravity * Math.sin(this._extGravityAngle),
            extElecX: this._extElectric * Math.cos(this._extElectricAngle),
            extElecY: this._extElectric * Math.sin(this._extElectricAngle),
            extBz: this._extBz,
            hubbleParam: this._hubbleParam,
        });
    }

    /** Reset field buffers to vacuum on preset load */
    resetFields() {
        if (this._higgsBuffers) this._initFieldToVacuum('higgs');
        if (this._axionBuffers) this._initFieldToVacuum('axion');
    }

    /** Queue a GPU hit test. Result available next frame via readHitResult(). */
    hitTest(worldX, worldY) {
        // Write click position to hit uniform buffer
        const data = new Float32Array([worldX, worldY, 0, 0]);
        this.device.queue.writeBuffer(this._hitUniformBuffer, 0, data);
        this._hitPending = true;
    }

    /** Read the result of a previously queued hit test. Returns particle index or -1. */
    readHitResult() {
        if (!this._hitResultReady) return -1;
        this._hitResultReady = false;
        const view = new Int32Array(this._hitStagingData);
        return view[0];
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
    }
}

// Constants (must match common.wgsl / config.js)
const FLAG_ALIVE = 1;
const FLAG_ANTIMATTER = 4;
const BOUND_LOOP = 2;
const COL_MERGE = 1;

// Save/load name tables (must match config.js)
const COL_NAMES = ['pass', 'merge', 'bounce'];
const BOUND_NAMES = ['despawn', 'bounce', 'loop'];
const TOPO_NAMES = ['torus', 'klein', 'rp2'];

/** Fetch a WGSL shader file relative to src/gpu/shaders/ */
async function fetchShader(filename) {
    const resp = await fetch(`src/gpu/shaders/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    return resp.text();
}
