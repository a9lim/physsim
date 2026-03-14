/**
 * @fileoverview GPURenderer — WebGPU instanced rendering for particles + bosons.
 *
 * Phase 1: particles. Phase 4: photon/pion boson rendering.
 * Also: spin rings, trails, field overlays, heatmap, force arrows.
 *
 * Dual-pipeline per renderer type: light mode uses standard alpha blending,
 * dark mode uses additive blending ('lighter' equivalent, matching CPU Canvas 2D).
 * On theme change setTheme() swaps _pipeline / _photonPipeline / _pionPipeline.
 */
import { createBosonRenderPipelines, createFieldRenderPipeline, createHeatmapRenderPipeline, createArrowRenderPipeline, createSpinRenderPipeline, createTrailRenderPipeline } from './gpu-pipelines.js';
import { TRAIL_LEN } from './gpu-buffers.js';

/** Standard premultiplied alpha-over blend (light mode). */
const BLEND_ALPHA = {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
};

/** Additive blend (dark mode — matches Canvas 2D globalCompositeOperation:'lighter'). */
const BLEND_ADDITIVE = {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
};

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
            size: 256, // 2 * mat4x4 + 4 floats (zoom, canvasWidth, canvasHeight, isDarkMode)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Particle render: two pipelines (light=alpha, dark=additive); _pipeline is the active one.
        this._pipelineLight = null;
        this._pipelineDark = null;
        this._pipeline = null;
        this._bindGroup = null;
        this._ready = false;

        // Boson rendering (Phase 4): two pipeline pairs, active pointers swapped on theme change.
        this._photonPipelineLight = null;
        this._photonPipelineDark = null;
        this._pionPipelineLight = null;
        this._pionPipelineDark = null;
        this._photonPipeline = null;
        this._pionPipeline = null;
        this._bosonBindGroups = null;
        this._bosonReady = false;

        // Field overlay rendering (Phase 5)
        this._fieldRenderPipeline = null;
        this._fieldRenderBindGroups = {};
        this._fieldRenderUniformBuffers = null; // per-field: { higgs, axion }
        this._fieldRenderReady = false;

        // Heatmap overlay rendering (Phase 5)
        this._heatmapRenderPipeline = null;
        this._heatmapRenderBindGroup = null;
        this._heatmapRenderUniformBuffer = null;
        this._heatmapRenderReady = false;

        // Arrow rendering (force/velocity vectors)
        this._arrowPipeline = null;
        this._arrowBindGroup = null;
        this._arrowUniformBuffer = null;
        this._arrowReady = false;

        // Spin ring rendering
        this._spinPipeline = null;
        this._spinBindGroup = null;
        this._spinReady = false;

        // Trail rendering
        this._trailPipeline = null;
        this._trailBindGroup = null;
        this._trailUniformBuffer = null;
        this._trailReady = false;

        // Visual toggle state (synced from CPU renderer via main.js)
        this.showForce = false;
        this.showForceComponents = false;
        this.showVelocity = false;
        this.showTrails = true; // default on (matches CPU)
    }

    /** Create render pipeline. Must be called after GPUPhysics.init(). */
    async init() {
        const shaderCode = await fetchShader('particle.wgsl');

        const module = this.device.createShaderModule({
            label: 'particle render',
            code: shaderCode,
        });

        // Single bind group layout shared by both pipeline variants.
        const bindGroupLayout = this.device.createBindGroupLayout({
            label: 'particle render',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // particleState
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // particleAux
                { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // color
            ],
        });

        const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        const pipelineBase = {
            layout: pipelineLayout,
            vertex: { module, entryPoint: 'vs_main' },
            primitive: { topology: 'triangle-list' },
        };

        // Light mode: standard premultiplied alpha-over blend.
        this._pipelineLight = this.device.createRenderPipeline({
            ...pipelineBase,
            label: 'particle render light',
            fragment: {
                module,
                entryPoint: 'fs_main',
                targets: [{ format: this.format, blend: BLEND_ALPHA }],
            },
        });

        // Dark mode: additive blend (matches Canvas 2D 'lighter').
        this._pipelineDark = this.device.createRenderPipeline({
            ...pipelineBase,
            label: 'particle render dark',
            fragment: {
                module,
                entryPoint: 'fs_main',
                targets: [{ format: this.format, blend: BLEND_ADDITIVE }],
            },
        });

        // Select active pipeline based on current theme.
        this._pipeline = this.isLight ? this._pipelineLight : this._pipelineDark;

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

        // --- Arrow render pipeline (force/velocity vectors) ---
        await this._initArrowRendering();

        // --- Spin ring render pipeline ---
        await this._initSpinRendering();
    }

    /** Create boson (photon + pion) render pipelines — both light and dark variants. */
    async _initBosonRendering() {
        let lightResult, darkResult;
        try {
            [lightResult, darkResult] = await Promise.all([
                createBosonRenderPipelines(this.device, this.format, true),
                createBosonRenderPipelines(this.device, this.format, false),
            ]);
        } catch (e) {
            console.warn('[physsim] Boson render pipeline creation failed, skipping:', e.message);
            return;
        }

        if (!lightResult.photonPipeline || !lightResult.pionPipeline ||
            !darkResult.photonPipeline || !darkResult.pionPipeline) {
            console.warn('[physsim] Boson render pipelines invalid, skipping');
            return;
        }

        this._photonPipelineLight = lightResult.photonPipeline;
        this._pionPipelineLight   = lightResult.pionPipeline;
        this._photonPipelineDark  = darkResult.photonPipeline;
        this._pionPipelineDark    = darkResult.pionPipeline;

        // Active pointers for current theme.
        this._photonPipeline = this.isLight ? this._photonPipelineLight : this._photonPipelineDark;
        this._pionPipeline   = this.isLight ? this._pionPipelineLight   : this._pionPipelineDark;

        const b = this.buffers;

        // Check that boson buffers exist (allocated lazily)
        if (!b.photonPool || !b.pionPool) {
            console.warn('[physsim] Boson buffers not yet allocated, skipping boson render init');
            return;
        }

        // Bind groups are theme-independent (same layouts between light and dark).
        try {
            const bindGroupLayouts = lightResult.bindGroupLayouts;

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

    /** Create arrow render pipeline for force/velocity vectors. */
    async _initArrowRendering() {
        try {
            const { pipeline, bindGroupLayout } =
                await createArrowRenderPipeline(this.device, this.format, this.isLight);
            this._arrowPipeline = pipeline;

            // ArrowUniforms: forceType(u32) + colorR/G/B(f32) + arrowScale(f32) + minMag(f32) + pad0 + pad1 = 32 bytes
            this._arrowUniformBuffer = this.device.createBuffer({
                label: 'arrowUniforms',
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            const b = this.buffers;
            this._arrowBindGroup = this.device.createBindGroup({
                label: 'arrowRender',
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.cameraBuffer } },
                    { binding: 1, resource: { buffer: this._arrowUniformBuffer } },
                    { binding: 2, resource: { buffer: b.particleState } },
                    { binding: 3, resource: { buffer: b.particleAux } },
                    { binding: 4, resource: { buffer: b.allForces } },
                ],
            });

            this._arrowReady = true;
        } catch (e) {
            console.warn('[physsim] Arrow render pipeline creation failed, skipping:', e.message);
        }
    }

    /** Create spin ring render pipeline. */
    async _initSpinRendering() {
        try {
            const { pipeline, bindGroupLayout } =
                await createSpinRenderPipeline(this.device, this.format, this.isLight);
            this._spinPipeline = pipeline;

            const b = this.buffers;
            this._spinBindGroup = this.device.createBindGroup({
                label: 'spinRender',
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.cameraBuffer } },
                    { binding: 1, resource: { buffer: b.particleState } },
                    { binding: 2, resource: { buffer: b.particleAux } },
                    { binding: 3, resource: { buffer: b.color } },
                ],
            });
            this._spinReady = true;
        } catch (e) {
            console.warn('[physsim] Spin render pipeline creation failed, skipping:', e.message);
        }
    }

    /**
     * Initialize trail render pipeline. Call after trail buffers are allocated.
     * @param {Object} trailBuffers - from GPUPhysics.getTrailBuffers()
     */
    async initTrailRendering(trailBuffers) {
        if (this._trailReady || !trailBuffers) return;
        try {
            const { pipeline, bindGroupLayout } =
                await createTrailRenderPipeline(this.device, this.format, this.isLight);
            this._trailPipeline = pipeline;

            this._trailUniformBuffer = this.device.createBuffer({
                label: 'trailUniforms', size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            const b = this.buffers;
            this._trailBindGroup = this.device.createBindGroup({
                label: 'trailRender',
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.cameraBuffer } },
                    { binding: 1, resource: { buffer: this._trailUniformBuffer } },
                    { binding: 2, resource: { buffer: trailBuffers.trailX } },
                    { binding: 3, resource: { buffer: trailBuffers.trailY } },
                    { binding: 4, resource: { buffer: trailBuffers.trailWriteIdx } },
                    { binding: 5, resource: { buffer: trailBuffers.trailCount } },
                    { binding: 6, resource: { buffer: b.color } },
                    { binding: 7, resource: { buffer: b.particleState } },
                ],
            });
            this._trailReady = true;
        } catch (e) {
            console.warn('[physsim] Trail render pipeline creation failed, skipping:', e.message);
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
        f[35] = this.isLight ? 0.0 : 1.0; // isDarkMode (0=light, 1=dark)

        this.device.queue.writeBuffer(this.cameraBuffer, 0, data);
    }

    /**
     * Render one frame.
     * The caller (main.js loop) only calls render() when _dirty is true,
     * so no additional dirty-flag gating is needed here.
     */
    render(aliveCount, opts = {}) {
        if (!this._ready) return;

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

        if (aliveCount > 0) {
            pass.setPipeline(this._pipeline);
            pass.setBindGroup(0, this._bindGroup);
            // 6 vertices per quad (2 triangles), aliveCount instances
            pass.draw(6, aliveCount);
        }
        pass.end();

        // Submit particle render immediately (isolate from optional passes that may fail)
        this.device.queue.submit([encoder.finish()]);

        // Pass 1b: Trail rendering (behind particles, but after clear)
        if (this._trailReady && this.showTrails && aliveCount > 0) {
            try {
                // Write trail uniforms
                const trailData = new ArrayBuffer(16);
                const tu32 = new Uint32Array(trailData);
                const tf32 = new Float32Array(trailData);
                tu32[0] = TRAIL_LEN;
                tf32[1] = this._domainW || 1;
                tf32[2] = this._domainH || 1;
                tf32[3] = 0;
                this.device.queue.writeBuffer(this._trailUniformBuffer, 0, trailData);

                const trailEncoder = this.device.createCommandEncoder({ label: 'trail-render' });
                const trailPass = trailEncoder.beginRenderPass({
                    label: 'trail render',
                    colorAttachments: [{
                        view: textureView,
                        loadOp: 'load',
                        storeOp: 'store',
                    }],
                });
                trailPass.setPipeline(this._trailPipeline);
                trailPass.setBindGroup(0, this._trailBindGroup);
                trailPass.draw(TRAIL_LEN, aliveCount);
                trailPass.end();
                this.device.queue.submit([trailEncoder.finish()]);
            } catch (e) {
                console.warn('[physsim] Trail render failed, disabling:', e.message);
                this._trailReady = false;
            }
        }

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

        // Pass 3b: Spin ring rendering
        if (this._spinReady && aliveCount > 0) {
            try {
                const spinEncoder = this.device.createCommandEncoder({ label: 'spin-render' });
                const spinPass = spinEncoder.beginRenderPass({
                    label: 'spin render',
                    colorAttachments: [{
                        view: textureView,
                        loadOp: 'load',
                        storeOp: 'store',
                    }],
                });
                spinPass.setPipeline(this._spinPipeline);
                spinPass.setBindGroup(0, this._spinBindGroup);
                // 32 vertices per arc (line-strip), aliveCount instances
                spinPass.draw(32, aliveCount);
                spinPass.end();
                this.device.queue.submit([spinEncoder.finish()]);
            } catch (e) {
                console.warn('[physsim] Spin render failed, disabling:', e.message);
                this._spinReady = false;
            }
        }

        // Pass 4: Force/velocity arrows — isolated encoder
        if (this._arrowReady && (this.showForce || this.showForceComponents) && aliveCount > 0) {
            try {
                const arrowEncoder = this.device.createCommandEncoder({ label: 'arrow-render' });
                const arrowPass = arrowEncoder.beginRenderPass({
                    label: 'arrow render',
                    colorAttachments: [{
                        view: textureView,
                        loadOp: 'load',
                        storeOp: 'store',
                    }],
                });

                const enabledForces = opts.enabledForces || {};
                const arrowScale = 256.0 / (this.zoom || 16);
                const minMag = 0.001;

                // Collect enabled force type indices
                const forceTypes = [];
                if (enabledForces.gravity) forceTypes.push(0);
                if (enabledForces.coulomb) forceTypes.push(1);
                if (enabledForces.magnetic) forceTypes.push(2);
                if (enabledForces.gravitomag) forceTypes.push(3);
                if (enabledForces.onePN) forceTypes.push(4);
                if (enabledForces.spinOrbit) forceTypes.push(5);
                if (enabledForces.radiation) forceTypes.push(6);
                if (enabledForces.yukawa) forceTypes.push(7);
                if (enabledForces.external) forceTypes.push(8);
                if (enabledForces.higgs) forceTypes.push(9);
                if (enabledForces.axion) forceTypes.push(10);

                if (forceTypes.length > 0) {
                    this._drawArrows(arrowPass, aliveCount, forceTypes, arrowScale, minMag);
                }

                arrowPass.end();
                this.device.queue.submit([arrowEncoder.finish()]);
            } catch (e) {
                console.warn('[physsim] Arrow render failed, disabling:', e.message);
                this._arrowReady = false;
            }
        }
    }

    /**
     * Force type colors matching CPU renderer (from renderer.js).
     * Index matches getForceVector() in arrow-render.wgsl.
     */
    static FORCE_COLORS = [
        [0.753, 0.314, 0.282], // 0: gravity — red #C05048
        [0.361, 0.573, 0.659], // 1: coulomb — blue #5C92A8
        [0.290, 0.675, 0.627], // 2: magnetic — cyan #4AACA0
        [0.769, 0.384, 0.447], // 3: gravitomag — rose #C46272
        [0.800, 0.557, 0.306], // 4: 1pn — orange #CC8E4E
        [0.612, 0.494, 0.690], // 5: spinCurv — purple #9C7EB0
        [0.800, 0.659, 0.298], // 6: radiation — yellow #CCA84C
        [0.314, 0.596, 0.471], // 7: yukawa — green #509878
        [0.612, 0.408, 0.251], // 8: external — brown #9C6840
        [0.510, 0.659, 0.341], // 9: higgs — lime #82A857
        [0.424, 0.475, 0.675], // 10: axion — indigo #6C79AC
    ];

    /**
     * Render force arrows for one or more force types.
     * @param {GPURenderPassEncoder} pass - active render pass
     * @param {number} aliveCount - number of alive particles
     * @param {number[]} forceTypes - array of force type indices (0-10) to draw
     * @param {number} arrowScale - scale factor for arrow length
     * @param {number} minMag - minimum force magnitude to draw
     */
    _drawArrows(pass, aliveCount, forceTypes, arrowScale = 256.0, minMag = 0.001) {
        if (!this._arrowReady || aliveCount <= 0) return;

        pass.setPipeline(this._arrowPipeline);
        pass.setBindGroup(0, this._arrowBindGroup);

        const data = new ArrayBuffer(32);
        const u32 = new Uint32Array(data);
        const f32 = new Float32Array(data);

        for (const ft of forceTypes) {
            const color = GPURenderer.FORCE_COLORS[ft] || [1, 1, 1];

            u32[0] = ft;           // forceType
            f32[1] = color[0];     // colorR
            f32[2] = color[1];     // colorG
            f32[3] = color[2];     // colorB
            f32[4] = arrowScale;   // arrowScale
            f32[5] = minMag;       // minMag
            f32[6] = 0;            // pad0
            f32[7] = 0;            // pad1

            this.device.queue.writeBuffer(this._arrowUniformBuffer, 0, data);
            // 9 vertices per arrow (3 triangles), aliveCount instances
            pass.draw(9, aliveCount);
        }
    }

    /** Initialize field overlay render pipeline. Call after GPUPhysics has field buffers. */
    async initFieldOverlay() {
        if (this._fieldRenderReady) return;

        const { pipeline, bindGroupLayouts } =
            await createFieldRenderPipeline(this.device, this.format, this.isLight);
        this._fieldRenderPipeline = pipeline;
        this._fieldRenderLayouts = bindGroupLayouts;

        // Separate uniform buffers per field to avoid writeBuffer race condition
        // when both Higgs and Axion draw in the same render pass
        this._fieldRenderUniformBuffers = {
            higgs: this.device.createBuffer({
                label: 'fieldRenderUniforms_higgs',
                size: 128,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }),
            axion: this.device.createBuffer({
                label: 'fieldRenderUniforms_axion',
                size: 128,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }),
        };

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
                { binding: 1, resource: { buffer: this._fieldRenderUniformBuffers[which] } },
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
        this.device.queue.writeBuffer(this._fieldRenderUniformBuffers[which], 0, data);

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

    /**
     * Update theme. Swaps active pipeline pointers to additive (dark) or alpha (light).
     * The camera uniform buffer is NOT re-written here — the isDarkMode field is written
     * fresh each frame by updateCamera(), so the next frame will have the correct value.
     */
    setTheme(isLight) {
        this.isLight = isLight;

        // Swap particle pipeline.
        if (this._pipelineLight && this._pipelineDark) {
            this._pipeline = isLight ? this._pipelineLight : this._pipelineDark;
        }

        // Swap boson pipelines.
        if (this._photonPipelineLight && this._photonPipelineDark) {
            this._photonPipeline = isLight ? this._photonPipelineLight : this._photonPipelineDark;
            this._pionPipeline   = isLight ? this._pionPipelineLight   : this._pionPipelineDark;
        }
    }

    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.canvasWidth = width;
        this.canvasHeight = height;
    }

    destroy() {
        this.cameraBuffer.destroy();
        if (this._arrowUniformBuffer) this._arrowUniformBuffer.destroy();
        if (this._trailUniformBuffer) this._trailUniformBuffer.destroy();
        if (this._fieldRenderUniformBuffers) {
            this._fieldRenderUniformBuffers.higgs.destroy();
            this._fieldRenderUniformBuffers.axion.destroy();
        }
        if (this._heatmapRenderUniformBuffer) this._heatmapRenderUniformBuffer.destroy();
    }
}

async function fetchShader(filename) {
    const resp = await fetch(`src/gpu/shaders/${filename}?v=9`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    return resp.text();
}
