/**
 * @fileoverview GPURenderer — WebGPU instanced rendering for particles + bosons.
 *
 * Phase 1: particles. Phase 4: photon/pion boson rendering.
 */
import { createBosonRenderPipelines, createFieldRenderPipeline, createHeatmapRenderPipeline } from './gpu-pipelines.js';

export default class GPURenderer {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {GPUDevice} device
     * @param {Object} particleBuffers - from gpu-buffers.js
     */
    constructor(canvas, device, particleBuffers) {
        this.canvas = canvas;
        this.device = device;
        this.buffers = particleBuffers;

        this.context = canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device,
            format: this.format,
            alphaMode: 'premultiplied',
        });

        // Camera state (updated from shared-camera.js)
        this.cameraX = 0;
        this.cameraY = 0;
        this.zoom = 16; // WORLD_SCALE default
        this.canvasWidth = canvas.width;
        this.canvasHeight = canvas.height;
        this.isLight = true;

        // Uniform buffer for camera
        this.cameraBuffer = device.createBuffer({
            label: 'cameraUniforms',
            size: 256, // 2 * mat4x4 + 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._pipeline = null;
        this._bindGroup = null;
        this._ready = false;

        // Boson rendering (Phase 4)
        this._photonPipeline = null;
        this._pionPipeline = null;
        this._bosonBindGroups = null;
        this._bosonReady = false;

        // Field overlay rendering (Phase 5)
        this._fieldRenderPipeline = null;
        this._fieldRenderBindGroups = {};
        this._fieldRenderUniformBuffer = null;
        this._fieldRenderReady = false;

        // Heatmap overlay rendering (Phase 5)
        this._heatmapRenderPipeline = null;
        this._heatmapRenderBindGroup = null;
        this._heatmapRenderUniformBuffer = null;
        this._heatmapRenderReady = false;
    }

    /** Create render pipeline. Must be called after GPUPhysics.init(). */
    async init() {
        const shaderCode = await fetchShader('particle.wgsl');

        const module = this.device.createShaderModule({
            label: 'particle render',
            code: shaderCode,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            label: 'particle render',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // particleState
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // particleAux
                { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // color
            ],
        });

        this._pipeline = this.device.createRenderPipeline({
            label: 'particle render',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module,
                entryPoint: 'vs_main',
            },
            fragment: {
                module,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.format,
                    blend: this.isLight
                        ? {
                            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        }
                        : {
                            // Additive blending for dark mode
                            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                        },
                }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        this._bindGroup = this.device.createBindGroup({
            label: 'particle render',
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.cameraBuffer } },
                { binding: 1, resource: { buffer: this.buffers.particleState } },
                { binding: 2, resource: { buffer: this.buffers.particleAux } },
                { binding: 3, resource: { buffer: this.buffers.color } },
            ],
        });

        this._ready = true;

        // --- Boson render pipelines (Phase 4) ---
        await this._initBosonRendering();
    }

    /** Create boson (photon + pion) render pipelines. */
    async _initBosonRendering() {
        let result;
        try {
            result = await createBosonRenderPipelines(this.device, this.format, this.isLight);
        } catch (e) {
            console.warn('[physsim] Boson render pipeline creation failed, skipping:', e.message);
            return;
        }
        const { photonPipeline, pionPipeline, bindGroupLayouts } = result;
        if (!photonPipeline || !pionPipeline) {
            console.warn('[physsim] Boson render pipelines invalid, skipping');
            return;
        }

        this._photonPipeline = photonPipeline;
        this._pionPipeline = pionPipeline;

        const b = this.buffers;

        // Check that boson buffers exist (allocated lazily)
        if (!b.photonPool || !b.pionPool) {
            console.warn('[physsim] Boson buffers not yet allocated, skipping boson render init');
            return;
        }

        try {
            // Group 0: camera uniforms
            const g0 = this.device.createBindGroup({
                label: 'bosonRender_g0',
                layout: bindGroupLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: this.cameraBuffer } },
                ],
            });

            // Group 1: photon pool (packed) + phCount
            const g1 = this.device.createBindGroup({
                label: 'bosonRender_g1',
                layout: bindGroupLayouts[1],
                entries: [
                    { binding: 0, resource: { buffer: b.photonPool } },
                    { binding: 1, resource: { buffer: b.phCount } },
                ],
            });

            // Group 2: pion pool (packed) + piCount
            const g2 = this.device.createBindGroup({
                label: 'bosonRender_g2',
                layout: bindGroupLayouts[2],
                entries: [
                    { binding: 0, resource: { buffer: b.pionPool } },
                    { binding: 1, resource: { buffer: b.piCount } },
                ],
            });

            this._bosonBindGroups = [g0, g1, g2];
            this._bosonReady = true;
        } catch (e) {
            console.warn('[physsim] Boson bind group creation failed:', e.message);
        }
    }

    /** Update camera uniform buffer. Call before render(). */
    updateCamera(camera) {
        this.cameraX = camera.x;
        this.cameraY = camera.y;
        this.zoom = camera.zoom;
        this.canvasWidth = this.canvas.width;
        this.canvasHeight = this.canvas.height;

        // Build 2D view matrix (world -> clip)
        // clip.x = (worldX - cameraX) * zoom * 2 / canvasWidth
        // clip.y = -(worldY - cameraY) * zoom * 2 / canvasHeight  (y-flip for clip space)
        const sx = this.zoom * 2 / this.canvasWidth;
        const sy = -this.zoom * 2 / this.canvasHeight;
        const tx = -this.cameraX * sx;
        const ty = -this.cameraY * sy;

        // mat4x4 column-major
        const view = new Float32Array([
            sx, 0,  0, 0,
            0,  sy, 0, 0,
            0,  0,  1, 0,
            tx, ty, 0, 1,
        ]);

        // Inverse: world = (clip - translate) / scale
        const isx = 1 / sx;
        const isy = 1 / sy;
        const inv = new Float32Array([
            isx, 0,   0, 0,
            0,   isy, 0, 0,
            0,   0,   1, 0,
            -tx * isx, -ty * isy, 0, 1,
        ]);

        const data = new ArrayBuffer(256);
        const f = new Float32Array(data);
        f.set(view, 0);        // viewMatrix at offset 0 (64 bytes)
        f.set(inv, 16);        // invViewMatrix at offset 64 (64 bytes)
        f[32] = this.zoom;     // offset 128
        f[33] = this.canvasWidth;
        f[34] = this.canvasHeight;
        f[35] = 0; // pad

        this.device.queue.writeBuffer(this.cameraBuffer, 0, data);
    }

    /**
     * Render one frame.
     * The caller (main.js loop) only calls render() when _dirty is true,
     * so no additional dirty-flag gating is needed here.
     */
    render(aliveCount, opts = {}) {
        if (!this._ready || aliveCount === 0) return;

        const textureView = this.context.getCurrentTexture().createView();

        const encoder = this.device.createCommandEncoder({ label: 'render' });

        // Pass 1: Clear + particle rendering
        const pass = encoder.beginRenderPass({
            label: 'particle render',
            colorAttachments: [{
                view: textureView,
                clearValue: this.isLight
                    ? { r: 0.941, g: 0.922, b: 0.894, a: 1 }  // --bg-canvas light: #F0EBE4
                    : { r: 0.047, g: 0.043, b: 0.035, a: 1 },  // --bg-canvas dark: #0C0B09
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bindGroup);
        // 6 vertices per quad (2 triangles), aliveCount instances
        pass.draw(6, aliveCount);
        pass.end();

        // Submit particle render immediately (isolate from optional passes that may fail)
        this.device.queue.submit([encoder.finish()]);

        // Pass 2: Field + heatmap overlays (load to preserve particles)
        const hasOverlays = (this._fieldRenderReady && (opts.higgsField || opts.axionField)) ||
            (this._heatmapRenderReady && opts.heatmapBuffers);
        if (hasOverlays) {
            const overlayEncoder = this.device.createCommandEncoder({ label: 'overlay' });
            const overlayPass = overlayEncoder.beginRenderPass({
                label: 'overlay render',
                colorAttachments: [{
                    view: textureView,
                    loadOp: 'load',
                    storeOp: 'store',
                }],
            });

            if (this._fieldRenderReady) {
                if (opts.higgsField) {
                    this.drawFieldOverlay(overlayPass, 'higgs', opts.higgsField);
                }
                if (opts.axionField) {
                    this.drawFieldOverlay(overlayPass, 'axion', opts.axionField);
                }
            }

            if (this._heatmapRenderReady && opts.heatmapBuffers) {
                this.drawHeatmapOverlay(overlayPass, opts.heatmapBuffers, opts.heatmapOpts || {});
            }

            overlayPass.end();
            this.device.queue.submit([overlayEncoder.finish()]);
        }

        // Pass 3: Boson rendering — isolated encoder so failures don't affect particles
        if (this._bosonReady) {
            try {
                const bosonEncoder = this.device.createCommandEncoder({ label: 'boson-render' });
                const bosonPass = bosonEncoder.beginRenderPass({
                    label: 'boson render',
                    colorAttachments: [{
                        view: textureView,
                        loadOp: 'load',
                        storeOp: 'store',
                    }],
                });

                const bgs = this._bosonBindGroups;

                bosonPass.setPipeline(this._photonPipeline);
                bosonPass.setBindGroup(0, bgs[0]);
                bosonPass.setBindGroup(1, bgs[1]);
                bosonPass.setBindGroup(2, bgs[2]);
                bosonPass.draw(4, 512);

                bosonPass.setPipeline(this._pionPipeline);
                bosonPass.setBindGroup(0, bgs[0]);
                bosonPass.setBindGroup(1, bgs[1]);
                bosonPass.setBindGroup(2, bgs[2]);
                bosonPass.draw(4, 256);

                bosonPass.end();
                this.device.queue.submit([bosonEncoder.finish()]);
            } catch (e) {
                console.warn('[physsim] Boson render failed, disabling:', e.message);
                this._bosonReady = false;
            }
        }
    }

    /** Initialize field overlay render pipeline. Call after GPUPhysics has field buffers. */
    async initFieldOverlay() {
        if (this._fieldRenderReady) return;

        const { pipeline, bindGroupLayouts } =
            await createFieldRenderPipeline(this.device, this.format, this.isLight);
        this._fieldRenderPipeline = pipeline;
        this._fieldRenderLayouts = bindGroupLayouts;

        this._fieldRenderUniformBuffer = this.device.createBuffer({
            label: 'fieldRenderUniforms',
            size: 128, // FieldRenderUniforms struct (padded to 128)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._fieldRenderReady = true;
    }

    /** Initialize heatmap overlay render pipeline. */
    async initHeatmapOverlay() {
        if (this._heatmapRenderReady) return;

        const { pipeline, bindGroupLayouts } =
            await createHeatmapRenderPipeline(this.device, this.format, this.isLight);
        this._heatmapRenderPipeline = pipeline;
        this._heatmapRenderLayouts = bindGroupLayouts;

        this._heatmapRenderUniformBuffer = this.device.createBuffer({
            label: 'heatmapRenderUniforms',
            size: 64, // HeatmapRenderUniforms struct
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._heatmapRenderReady = true;
    }

    /**
     * Ensure field render bind group exists for a given field.
     * @param {'higgs'|'axion'} which
     * @param {GPUBuffer} fieldBuffer - the field array buffer from GPUPhysics
     */
    _ensureFieldRenderBG(which, fieldBuffer) {
        if (this._fieldRenderBindGroups[which]) return;
        this._fieldRenderBindGroups[which] = this.device.createBindGroup({
            label: `fieldRender_${which}`,
            layout: this._fieldRenderLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: fieldBuffer } },
                { binding: 1, resource: { buffer: this._fieldRenderUniformBuffer } },
            ],
        });
    }

    /**
     * Draw field overlay for one field type.
     * @param {GPURenderPassEncoder} pass
     * @param {'higgs'|'axion'} which
     * @param {GPUBuffer} fieldBuffer
     */
    drawFieldOverlay(pass, which, fieldBuffer) {
        if (!this._fieldRenderReady || !fieldBuffer) return;
        this._ensureFieldRenderBG(which, fieldBuffer);

        // Write FieldRenderUniforms
        const data = new ArrayBuffer(128);
        const f = new Float32Array(data);
        const u = new Uint32Array(data);
        f[0] = this.cameraX;
        f[1] = this.cameraY;
        f[2] = this.zoom;
        f[3] = this.canvasWidth;
        f[4] = this.canvasHeight;
        f[5] = this._domainW || 1;
        f[6] = this._domainH || 1;
        u[7] = this.isLight ? 1 : 0;
        u[8] = which === 'higgs' ? 0 : 1;
        // Colors: Higgs depleted=purple, enhanced=lime; Axion positive=indigo, negative=yellow
        if (which === 'higgs') {
            // color0 (depleted/purple): #9C7EB0
            f[12] = 0.612; f[13] = 0.494; f[14] = 0.690; f[15] = 1.0;
            // color1 (enhanced/lime): #82A857
            f[16] = 0.510; f[17] = 0.659; f[18] = 0.341; f[19] = 1.0;
        } else {
            // color0 (positive/indigo): #6C79AC
            f[12] = 0.424; f[13] = 0.475; f[14] = 0.675; f[15] = 1.0;
            // color1 (negative/yellow): #CCA84C
            f[16] = 0.800; f[17] = 0.659; f[18] = 0.298; f[19] = 1.0;
        }
        this.device.queue.writeBuffer(this._fieldRenderUniformBuffer, 0, data);

        pass.setPipeline(this._fieldRenderPipeline);
        pass.setBindGroup(0, this._fieldRenderBindGroups[which]);
        pass.draw(3); // fullscreen triangle
    }

    /**
     * Draw heatmap overlay.
     * @param {GPURenderPassEncoder} pass
     * @param {Object} heatmapBuffers - from GPUPhysics.getHeatmapBuffers()
     * @param {Object} opts - { viewLeft, viewTop, cellW, cellH, doGravity, doCoulomb, doYukawa }
     */
    drawHeatmapOverlay(pass, heatmapBuffers, opts) {
        if (!this._heatmapRenderReady || !heatmapBuffers) return;

        if (!this._heatmapRenderBindGroup) {
            this._heatmapRenderBindGroup = this.device.createBindGroup({
                label: 'heatmapRender_g0',
                layout: this._heatmapRenderLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: heatmapBuffers.gravPotential } },
                    { binding: 1, resource: { buffer: heatmapBuffers.elecPotential } },
                    { binding: 2, resource: { buffer: heatmapBuffers.yukawaPotential } },
                    { binding: 3, resource: { buffer: this._heatmapRenderUniformBuffer } },
                ],
            });
        }

        // Write HeatmapRenderUniforms
        const data = new ArrayBuffer(64);
        const f = new Float32Array(data);
        const u = new Uint32Array(data);
        f[0] = this.cameraX;
        f[1] = this.cameraY;
        f[2] = this.zoom;
        f[3] = this.canvasWidth;
        f[4] = this.canvasHeight;
        f[5] = opts.viewLeft || 0;
        f[6] = opts.viewTop || 0;
        f[7] = opts.cellW || 1;
        f[8] = opts.cellH || 1;
        f[9] = 2.0;    // HEATMAP_SENSITIVITY
        f[10] = 100.0 / 255.0; // HEATMAP_MAX_ALPHA
        u[11] = this.isLight ? 1 : 0;
        u[12] = opts.doGravity ? 1 : 0;
        u[13] = opts.doCoulomb ? 1 : 0;
        u[14] = opts.doYukawa ? 1 : 0;
        this.device.queue.writeBuffer(this._heatmapRenderUniformBuffer, 0, data);

        pass.setPipeline(this._heatmapRenderPipeline);
        pass.setBindGroup(0, this._heatmapRenderBindGroup);
        pass.draw(3); // fullscreen triangle
    }

    /** Set domain dimensions for field rendering */
    setDomain(domainW, domainH) {
        this._domainW = domainW;
        this._domainH = domainH;
    }

    setTheme(isLight) {
        this.isLight = isLight;
        // KNOWN LIMITATION (Phase 1): Blend mode baked at init time — only clear color changes.
        // Spec requires two pre-built pipelines (additive dark / alpha light) swapped on theme change.
        // Will be implemented in Phase 2 when the render pipeline is expanded.
    }

    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.canvasWidth = width;
        this.canvasHeight = height;
    }

    destroy() {
        this.cameraBuffer.destroy();
    }
}

async function fetchShader(filename) {
    const resp = await fetch(`src/gpu/shaders/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    return resp.text();
}
