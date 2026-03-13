/**
 * @fileoverview GPUPhysics — WebGPU compute pipeline orchestrator.
 *
 * Phase 1: drift + boundary only. Forces, Boris, fields added in later phases.
 */
import { createParticleBuffers, createUniformBuffer, writeUniforms } from './gpu-buffers.js';

const MAX_PARTICLES = 4096;

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

        // Pipelines will be created in init()
        this._driftPipeline = null;
        this._boundaryPipeline = null;
        this._driftBindGroup = null;
        this._boundaryBindGroup = null;

        this._ready = false;
    }

    /** Load WGSL shaders and create compute pipelines. Must be called before update(). */
    async init() {
        const commonWGSL = await fetchShader('common.wgsl');
        const driftWGSL = await fetchShader('drift.wgsl');
        const boundaryWGSL = await fetchShader('boundary.wgsl');

        // --- Drift pipeline ---
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

        // --- Boundary pipeline (same bind group layout + flags is read_write) ---
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

        this._ready = true;
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
     * Run one substep: drift + boundary.
     * Phase 1 only — no forces, no Boris rotation.
     */
    update(dt) {
        if (!this._ready || this.aliveCount === 0) return;

        this.simTime += dt;

        // Upload uniforms
        writeUniforms(this.device, this.uniformBuffer, {
            dt,
            simTime: this.simTime,
            domainW: this.domainW,
            domainH: this.domainH,
            speedScale: 1,
            softening: 8,
            softeningSq: 64,
            boundaryMode: this.boundaryMode,
            topologyMode: this.topologyMode,
            maxParticles: MAX_PARTICLES,
            aliveCount: this.aliveCount,
        });

        const workgroups = Math.ceil(this.aliveCount / 64);

        const encoder = this.device.createCommandEncoder({ label: 'physics' });

        // Drift pass
        const driftPass = encoder.beginComputePass({ label: 'drift' });
        driftPass.setPipeline(this._driftPipeline);
        driftPass.setBindGroup(0, this._driftBindGroup);
        driftPass.dispatchWorkgroups(workgroups);
        driftPass.end();

        // Boundary pass
        const boundaryPass = encoder.beginComputePass({ label: 'boundary' });
        boundaryPass.setPipeline(this._boundaryPipeline);
        boundaryPass.setBindGroup(0, this._boundaryBindGroup);
        boundaryPass.dispatchWorkgroups(workgroups);
        boundaryPass.end();

        this.device.queue.submit([encoder.finish()]);
    }

    reset() {
        this.aliveCount = 0;
        this.simTime = 0;
    }

    destroy() {
        this.buffers.destroy();
        this.uniformBuffer.destroy();
    }
}

// Flag constants (must match common.wgsl)
const FLAG_ALIVE = 1;

/** Fetch a WGSL shader file relative to src/gpu/shaders/ */
async function fetchShader(filename) {
    const resp = await fetch(`src/gpu/shaders/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    return resp.text();
}
