/**
 * @fileoverview GPUPhysics — WebGPU compute pipeline orchestrator.
 *
 * Phase 2+3+4: Full force computation, Boris integrator, tree build, collisions,
 * dead GC, radiation, 1PN VV, boson lifecycle, boson gravity, signal delay history.
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
 *  14. compute1PN_VV          (Phase 4 — 1PN recompute + VV correction kick)
 *  17-18. collisions          (Phase 3 — detect + resolve)
 *  21. bosonUpdate            (Phase 4 — photon/pion drift, absorption, decay)
 *  24. boundary
 *
 * Post-substep (once per frame):
 *  - bosonGravity             (Phase 4 — build boson tree + particle/boson gravity)
 *  - deadParticleGC           (Phase 3)
 *  - recordHistory            (Phase 4 — every HISTORY_STRIDE frames)
 */
import { createParticleBuffers, createUniformBuffer, writeUniforms, createFieldBuffers, createPQSScratchBuffer, createPQSIndexBuffer, createHeatmapBuffers } from './gpu-buffers.js';
import { createPhase2Pipelines, createGhostGenPipeline, createTreeBuildPipelines, createTreeForcePipeline, createCollisionPipelines, createDeadGCPipeline, createPhase4Pipelines } from './gpu-pipelines.js';

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
        this._higgsEnabled = false;
        this._axionEnabled = false;
        this._fieldGravEnabled = false;

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

        // resetForces: uniforms + 11 force/torque buffers
        this._bg_resetForces = bg('resetForces', p2.resetForces.bindGroupLayouts[0],
            [this.uniformBuffer, b.forces0, b.forces1, b.forces2, b.forces3,
             b.forces4, b.forces5, b.torques, b.bFields, b.bFieldGrads,
             b.totalForceX, b.totalForceY]);

        // cacheDerived: uniforms + inputs + outputs
        this._bg_cacheDerived = bg('cacheDerived', p2.cacheDerived.bindGroupLayouts[0],
            [this.uniformBuffer, b.mass, b.velWX, b.velWY, b.angW, b.charge,
             b.radius, b.gamma, b.magMoment, b.angMomentum, b.velX, b.velY,
             b.angVel, b.invMass, b.radiusSq, b.flags]);

        // pairForce: 3 bind groups
        this._bg_pairForce0 = bg('pairForce_g0', p2.pairForce.bindGroupLayouts[0],
            [this.uniformBuffer]);
        this._bg_pairForce1 = bg('pairForce_g1', p2.pairForce.bindGroupLayouts[1],
            [b.posX, b.posY, b.velX, b.velY, b.mass, b.charge, b.angVel,
             b.magMoment, b.angMomentum, b.axMod, b.yukMod, b.flags, b.radiusSq,
             b.velWX, b.velWY]);
        this._bg_pairForce2 = bg('pairForce_g2', p2.pairForce.bindGroupLayouts[2],
            [b.forces0, b.forces1, b.forces2, b.forces3, b.torques, b.bFields,
             b.bFieldGrads, b.totalForceX, b.totalForceY]);

        // externalFields
        this._bg_extFields = bg('extFields', p2.externalFields.bindGroupLayouts[0],
            [this.uniformBuffer, b.mass, b.charge, b.flags, b.forces4,
             b.totalForceX, b.totalForceY, b.bFields]);

        // borisHalfKick
        this._bg_halfKick = bg('halfKick', p2.borisHalfKick.bindGroupLayouts[0],
            [this.uniformBuffer, b.velWX, b.velWY, b.mass, b.totalForceX,
             b.totalForceY, b.flags]);

        // borisRotate
        this._bg_rotate = bg('rotate', p2.borisRotate.bindGroupLayouts[0],
            [this.uniformBuffer, b.velWX, b.velWY, b.charge, b.mass, b.bFields, b.flags]);

        // borisDrift
        this._bg_drift = bg('drift', p2.borisDrift.bindGroupLayouts[0],
            [this.uniformBuffer, b.posX, b.posY, b.velWX, b.velWY, b.flags,
             b.velX, b.velY]);

        // spinOrbit
        this._bg_spinOrbit = bg('spinOrbit', p2.spinOrbit.bindGroupLayouts[0],
            [this.uniformBuffer, b.velWX, b.velWY, b.angW, b.mass, b.charge,
             b.velX, b.velY, b.magMoment, b.angMomentum, b.radius, b.bFieldGrads,
             b.flags, b.angVel, b.forces2]);

        // applyTorques
        this._bg_torques = bg('torques', p2.applyTorques.bindGroupLayouts[0],
            [this.uniformBuffer, b.angW, b.mass, b.radius, b.torques, b.flags, b.angVel]);
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
        //   bindings 8-10: particle derived inputs (read-only)
        //   bindings 11-13: ghost derived output (read-write)
        //   binding 14: particle ID input (read-only)
        //   binding 15: ghost particle ID output (read-write)
        const group1 = bg('ghostGen_g1', layouts[1],
            [b.ghostPosX, b.ghostPosY, b.ghostVelWX, b.ghostVelWY,
             b.ghostAngW, b.ghostMass, b.ghostCharge, b.ghostFlags,
             b.radius, b.magMoment, b.angMomentum,
             b.ghostRadius, b.ghostMagMoment, b.ghostAngMomentum,
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
            encoder.copyBufferToBuffer(b.ghostMagMoment, 0, b.magMoment, offset, ghostBytes);
            encoder.copyBufferToBuffer(b.ghostAngMomentum, 0, b.angMomentum, offset, ghostBytes);
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
                { binding: 6, resource: { buffer: b.magMoment } },
                { binding: 7, resource: { buffer: b.angMomentum } },
                { binding: 8, resource: { buffer: b.flags } },
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

        // Group 1: particle SoA (matches shader @group(1) bindings 0-15)
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
                { binding: 9, resource: { buffer: b.magMoment } },
                { binding: 10, resource: { buffer: b.angMomentum } },
                { binding: 11, resource: { buffer: b.axMod } },
                { binding: 12, resource: { buffer: b.yukMod } },
                { binding: 13, resource: { buffer: b.particleId } },
                { binding: 14, resource: { buffer: b.ghostOriginalIdx } },
                { binding: 15, resource: { buffer: b.deathMass } },
            ],
        });

        // Group 2: force accumulators (forces0, forces1, forces3, bFields, torques)
        this._treeForceGroup2 = this.device.createBindGroup({
            label: 'treeForce_g2',
            layout: layouts[2],
            entries: [
                { binding: 0, resource: { buffer: b.forces0 } },
                { binding: 1, resource: { buffer: b.forces1 } },
                { binding: 2, resource: { buffer: b.forces3 } },
                { binding: 3, resource: { buffer: b.bFields } },
                { binding: 4, resource: { buffer: b.torques } },
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
            [b.posX, b.posY, b.velWX, b.velWY, b.mass, b.charge, b.flags, b.yukMod, b.invMass]);
        this._phase4BindGroups.onePNG2 = bg('onePN_g2', p4.compute1PN.bindGroupLayouts[2],
            [b.forces2, b.f1pnOld, b.velWX, b.velWY]);

        // ── Radiation (lamrorRadiation, hawkingRadiation, pionEmission share bind groups) ──
        this._phase4BindGroups.radG0 = bg('radiation_g0', p4.lamrorRadiation.bindGroupLayouts[0],
            [this.uniformBuffer]);
        // Group 1: particle state (17 bindings)
        // binding 14: jerkInterleaved [x0,y0,x1,y1...]
        // bindings 15-16: separate yukForceX/Y buffers
        this._phase4BindGroups.radG1 = bg('radiation_g1', p4.lamrorRadiation.bindGroupLayouts[1],
            [b.posX, b.posY, b.velWX, b.velWY, b.mass, b.charge, b.flags,
             b.invMass, b.baseMass, b.radius, b.angW, b.particleId,
             b.totalForceX, b.totalForceY, b.jerkInterleaved,
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
            [b.posX, b.posY, b.mass, b.flags, b.forces0]);
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
        f32[0] = Math.cbrt(m); this.device.queue.writeBuffer(this.buffers.radius, idx * 4, f32);
        const wSq = vx * vx + vy * vy;
        f32[0] = Math.sqrt(1 + wSq); this.device.queue.writeBuffer(this.buffers.gamma, idx * 4, f32);
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

        // Lazily allocate scalar field buffers on first toggle-on (matching CPU pattern)
        this._higgsEnabled = physics.higgsEnabled;
        this._axionEnabled = physics.axionEnabled;
        this._fieldGravEnabled = physics.fieldGravEnabled;
        if (physics.higgsEnabled && !this._higgsBuffers) {
            this._ensureFieldBuffers('higgs');
        }
        if (physics.axionEnabled && !this._axionBuffers) {
            this._ensureFieldBuffers('axion');
        }
    }

    /**
     * Lazily allocate GPU buffers for a scalar field.
     * Also ensures shared PQS scratch/index buffers are allocated.
     * @param {'higgs'|'axion'} which
     */
    _ensureFieldBuffers(which) {
        if (!this._pqsScratch) {
            this._pqsScratch = createPQSScratchBuffer(this.device, MAX_PARTICLES);
            this._pqsIndices = createPQSIndexBuffer(this.device, MAX_PARTICLES);
        }
        if (which === 'higgs' && !this._higgsBuffers) {
            this._higgsBuffers = createFieldBuffers(this.device, 'higgs', MAX_PARTICLES);
        }
        if (which === 'axion' && !this._axionBuffers) {
            this._axionBuffers = createFieldBuffers(this.device, 'axion', MAX_PARTICLES);
        }
    }

    /**
     * Run one frame: adaptive substepping with full Phase 2 force computation + Boris integrator.
     */
    update(dt) {
        if (!this._ready || this.aliveCount === 0) return;

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

        // Pass 11: radiation reaction (Larmor, Hawking, pion emission) — Phase 4
        this._dispatchRadiation(encoder);

        // Pass 12: borisDrift
        const pass12 = encoder.beginComputePass({ label: 'borisDrift' });
        pass12.setPipeline(p2.borisDrift.pipeline);
        pass12.setBindGroup(0, this._bg_drift);
        pass12.dispatchWorkgroups(workgroups);
        pass12.end();

        // Pass 14: 1PN velocity-Verlet correction — Phase 4
        this._dispatch1PNVV(encoder);

        // Pass 17-18: collision detection + resolution (Phase 3)
        // Runs after drift, uses tree built earlier in the substep
        this._dispatchCollisions(encoder);

        // Pass 21: boson update (photon/pion drift, absorption, decay) — Phase 4
        this._dispatchBosonUpdate(encoder);

        // Pass 24: boundary (existing from Phase 1)
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

        this._maxAccel = data[0];
        this._maxAccelPending = false;
    }

    reset() {
        this.aliveCount = 0;
        this.simTime = 0;
        this._histStride = 0;
        this._frameCount = 0;
    }

    destroy() {
        this.buffers.destroy();
        this.uniformBuffer.destroy();
    }
}

// Constants (must match common.wgsl / config.js)
const FLAG_ALIVE = 1;
const BOUND_LOOP = 2;
const COL_MERGE = 1;

/** Fetch a WGSL shader file relative to src/gpu/shaders/ */
async function fetchShader(filename) {
    const resp = await fetch(`src/gpu/shaders/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    return resp.text();
}
