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
import { buildWGSLConstants, paletteRGB } from './gpu-constants.js';
import { HEATMAP_SENSITIVITY, HEATMAP_MAX_ALPHA, MAX_PHOTONS, MAX_PIONS } from '../config.js';

// Pre-allocated typed arrays for per-frame writeBuffer calls (eliminate GC pressure)
const _cameraData = new ArrayBuffer(256);
const _cameraF32 = new Float32Array(_cameraData);
const _trailData = new ArrayBuffer(16);
const _trailU32 = new Uint32Array(_trailData);
const _trailF32 = new Float32Array(_trailData);
const _arrowData = new ArrayBuffer(32);
const _arrowU32 = new Uint32Array(_arrowData);
const _arrowF32 = new Float32Array(_arrowData);
const _fieldRenderData = new ArrayBuffer(128);
const _fieldRenderF32 = new Float32Array(_fieldRenderData);
const _fieldRenderU32 = new Uint32Array(_fieldRenderData);
const _heatmapRenderData = new ArrayBuffer(64);
const _heatmapRenderF32 = new Float32Array(_heatmapRenderData);
const _heatmapRenderU32 = new Uint32Array(_heatmapRenderData);

// Palette-derived colors (computed once at module load from _PALETTE)
const _PAL = window._PALETTE;
const _bgLight = (() => { const [r,g,b] = paletteRGB(_PAL.light.canvas); return {r,g,b,a:1}; })();
const _bgDark = (() => { const [r,g,b] = paletteRGB(_PAL.dark.canvas); return {r,g,b,a:1}; })();
const _accentLight = paletteRGB(_PAL.accent);
const _accentDark = paletteRGB(_PAL.accentLight);
const _textLight = paletteRGB(_PAL.light.text);
const _textDark = paletteRGB(_PAL.dark.text);
const _fieldColors = {
    higgs: [paletteRGB(_PAL.extended.purple), paletteRGB(_PAL.extended.lime)],
    axion: [paletteRGB(_PAL.extended.indigo), paletteRGB(_PAL.extended.yellow)],
};

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

        // Spin ring rendering (arc + arrowhead pipelines)
        this._spinArcPipeline = null;
        this._spinArrowPipeline = null;
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
        const wgslConstants = buildWGSLConstants();
        this._wgslConstants = wgslConstants; // cache for theme-change rebuilds
        const shaderCode = await fetchShader('particle.wgsl', wgslConstants);

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
                createBosonRenderPipelines(this.device, this.format, true, this._wgslConstants || ''),
                createBosonRenderPipelines(this.device, this.format, false, this._wgslConstants || ''),
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
                await createArrowRenderPipeline(this.device, this.format, this.isLight, this._wgslConstants || '');
            this._arrowPipeline = pipeline;

            // ArrowUniforms: forceType(u32) + colorR/G/B(f32) + arrowScale(f32) + minMag(f32) + pad0 + pad1 = 32 bytes
            if (!this._arrowUniformBuffer) {
                this._arrowUniformBuffer = this.device.createBuffer({
                    label: 'arrowUniforms',
                    size: 32,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
            }

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
                    { binding: 5, resource: { buffer: b.derived } },
                ],
            });

            this._arrowReady = true;
        } catch (e) {
            console.warn('[physsim] Arrow render pipeline creation failed, skipping:', e.message);
        }
    }

    /** Create spin ring render pipelines (arc line-strip + arrowhead triangles). */
    async _initSpinRendering() {
        try {
            const { arcPipeline, arrowPipeline, bindGroupLayout } =
                await createSpinRenderPipeline(this.device, this.format, this.isLight, this._wgslConstants || '');
            this._spinArcPipeline = arcPipeline;
            this._spinArrowPipeline = arrowPipeline;

            const b = this.buffers;
            this._spinBindGroup = this.device.createBindGroup({
                label: 'spinRender',
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.cameraBuffer } },
                    { binding: 1, resource: { buffer: b.particleState } },
                    { binding: 2, resource: { buffer: b.particleAux } },
                    { binding: 3, resource: { buffer: b.derived } },
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
                await createTrailRenderPipeline(this.device, this.format, this.isLight, this._wgslConstants || '');
            this._trailPipeline = pipeline;

            // Store trail buffer reference for theme-change rebuild
            this._trailBuffersRef = trailBuffers;

            if (!this._trailUniformBuffer) {
                this._trailUniformBuffer = this.device.createBuffer({
                    label: 'trailUniforms', size: 16,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
            }

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
                    { binding: 8, resource: { buffer: b.particleAux } },
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

        // mat4x4 column-major — write directly into pre-allocated buffer
        const f = _cameraF32;
        // View matrix
        f[0] = sx; f[1] = 0;  f[2] = 0; f[3] = 0;
        f[4] = 0;  f[5] = sy; f[6] = 0; f[7] = 0;
        f[8] = 0;  f[9] = 0;  f[10] = 1; f[11] = 0;
        f[12] = tx; f[13] = ty; f[14] = 0; f[15] = 1;
        // Inverse view matrix
        const isx = 1 / sx;
        const isy = 1 / sy;
        f[16] = isx; f[17] = 0;   f[18] = 0; f[19] = 0;
        f[20] = 0;   f[21] = isy; f[22] = 0; f[23] = 0;
        f[24] = 0;   f[25] = 0;   f[26] = 1; f[27] = 0;
        f[28] = -tx * isx; f[29] = -ty * isy; f[30] = 0; f[31] = 1;
        // Extra uniforms
        f[32] = this.zoom;
        f[33] = this.canvasWidth;
        f[34] = this.canvasHeight;
        f[35] = this.isLight ? 0.0 : 1.0;

        this.device.queue.writeBuffer(this.cameraBuffer, 0, _cameraData);
    }

    /**
     * Render one frame.
     * The caller (main.js loop) only calls render() when _dirty is true,
     * so no additional dirty-flag gating is needed here.
     */
    render(aliveCount, opts = {}) {
        if (!this._ready) return;

        const textureView = this.context.getCurrentTexture().createView();

        // Pass 0: Trail rendering (behind particles — drawn first, like CPU renderer)
        // Trails clear the canvas; particles load on top.
        const trailsDrawn = this._trailReady && this.showTrails && aliveCount > 0;
        if (trailsDrawn) {
            try {
                // Write trail uniforms (pre-allocated buffer)
                _trailU32[0] = TRAIL_LEN;
                _trailF32[1] = this._domainW || 1;
                _trailF32[2] = this._domainH || 1;
                _trailF32[3] = 0;
                this.device.queue.writeBuffer(this._trailUniformBuffer, 0, _trailData);

                const trailEncoder = this.device.createCommandEncoder({ label: 'trail-render' });
                const trailPass = trailEncoder.beginRenderPass({
                    label: 'trail render',
                    colorAttachments: [{
                        view: textureView,
                        clearValue: this.isLight ? _bgLight : _bgDark,
                        loadOp: 'clear',
                        storeOp: 'store',
                    }],
                });
                trailPass.setPipeline(this._trailPipeline);
                trailPass.setBindGroup(0, this._trailBindGroup);
                trailPass.draw(TRAIL_LEN * 2, aliveCount);
                trailPass.end();
                this.device.queue.submit([trailEncoder.finish()]);
            } catch (e) {
                console.warn('[physsim] Trail render failed, disabling:', e.message);
                this._trailReady = false;
            }
        }

        // Pass 1: Particle rendering (load if trails already cleared, otherwise clear here)
        const encoder = this.device.createCommandEncoder({ label: 'render' });
        const pass = encoder.beginRenderPass({
            label: 'particle render',
            colorAttachments: [{
                view: textureView,
                ...(!trailsDrawn ? {
                    clearValue: this.isLight ? _bgLight : _bgDark,
                } : {}),
                loadOp: trailsDrawn ? 'load' : 'clear',
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
                bosonPass.draw(4, MAX_PHOTONS);

                bosonPass.setPipeline(this._pionPipeline);
                bosonPass.setBindGroup(0, bgs[0]);
                bosonPass.setBindGroup(1, bgs[1]);
                bosonPass.setBindGroup(2, bgs[2]);
                bosonPass.draw(4, MAX_PIONS);

                bosonPass.end();
                this.device.queue.submit([bosonEncoder.finish()]);
            } catch (e) {
                console.warn('[physsim] Boson render failed, disabling:', e.message);
                this._bosonReady = false;
            }
        }

        // Pass 3b: Spin ring rendering (arcs + arrowheads)
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
                // Arc triangle-strip ribbon: 32 segments × 2 vertices = 64
                spinPass.setPipeline(this._spinArcPipeline);
                spinPass.setBindGroup(0, this._spinBindGroup);
                spinPass.draw(64, aliveCount);
                // Arrowhead triangles: 3 vertices per instance
                spinPass.setPipeline(this._spinArrowPipeline);
                spinPass.setBindGroup(0, this._spinBindGroup);
                spinPass.draw(3, aliveCount);
                spinPass.end();
                this.device.queue.submit([spinEncoder.finish()]);
            } catch (e) {
                console.warn('[physsim] Spin render failed, disabling:', e.message);
                this._spinReady = false;
            }
        }

        // Pass 4: Force/velocity arrows — one submit per arrow type.
        // Each arrow type needs its own writeBuffer + submit to ensure the uniform
        // data is correct for that draw call (writeBuffer inside a single render pass
        // would cause all draws to use the last-written uniform).
        const wantsArrows = this._arrowReady && aliveCount > 0 &&
            (this.showForce || this.showForceComponents || this.showVelocity);
        if (wantsArrows) {
            try {
                const enabledForces = opts.enabledForces || {};
                const zoom = this.zoom || 16;
                const invZoom = 1 / zoom;
                const arrowScale = 256.0;
                const minMag = 0.1 * invZoom;

                if (this.showForceComponents) {
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
                    for (const ft of forceTypes) {
                        const color = GPURenderer.FORCE_COLORS[ft] || [1, 1, 1];
                        this._submitArrowDraw(textureView, aliveCount, ft, color, arrowScale, minMag, invZoom);
                    }
                } else if (this.showForce) {
                    const accentColor = this.isLight ? _accentLight : _accentDark;
                    this._submitArrowDraw(textureView, aliveCount, 11, accentColor, arrowScale, minMag, invZoom);
                }

                if (this.showVelocity) {
                    const textColor = this.isLight ? _textLight : _textDark;
                    this._submitArrowDraw(textureView, aliveCount, 12, textColor, 1.0, minMag, invZoom);
                }
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
    static FORCE_COLORS = (() => {
        const ext = window._PALETTE.extended;
        const rgb = paletteRGB;
        return [
            rgb(ext.red),       // 0: gravity
            rgb(ext.blue),      // 1: coulomb
            rgb(ext.cyan),      // 2: magnetic
            rgb(ext.rose),      // 3: gravitomag
            rgb(ext.orange),    // 4: 1pn
            rgb(ext.purple),    // 5: spinCurv
            rgb(ext.yellow),    // 6: radiation
            rgb(ext.green),     // 7: yukawa
            rgb(ext.brown),     // 8: external
            rgb(ext.lime),      // 9: higgs
            rgb(ext.indigo),    // 10: axion
        ];
    })();

    /**
     * Submit a single arrow draw with its own encoder + render pass.
     * Each arrow type needs a separate submit so that the uniform writeBuffer
     * takes effect before the draw executes on the GPU.
     */
    _submitArrowDraw(textureView, aliveCount, forceType, color, arrowScale, minMag, invZoom = 0.0625) {
        if (!this._arrowReady || aliveCount <= 0) return;

        _arrowU32[0] = forceType;
        _arrowF32[1] = color[0];
        _arrowF32[2] = color[1];
        _arrowF32[3] = color[2];
        _arrowF32[4] = arrowScale;
        _arrowF32[5] = minMag;
        _arrowF32[6] = invZoom;
        _arrowF32[7] = 0;
        this.device.queue.writeBuffer(this._arrowUniformBuffer, 0, _arrowData);

        const enc = this.device.createCommandEncoder({ label: `arrow-${forceType}` });
        const pass = enc.beginRenderPass({
            label: `arrow ${forceType}`,
            colorAttachments: [{
                view: textureView,
                loadOp: 'load',
                storeOp: 'store',
            }],
        });
        pass.setPipeline(this._arrowPipeline);
        pass.setBindGroup(0, this._arrowBindGroup);
        pass.draw(9, aliveCount);
        pass.end();
        this.device.queue.submit([enc.finish()]);
    }

    /** Initialize field overlay render pipeline. Call after GPUPhysics has field buffers. */
    async initFieldOverlay() {
        if (this._fieldRenderReady) return;

        const { pipeline, bindGroupLayouts } =
            await createFieldRenderPipeline(this.device, this.format, this.isLight, this._wgslConstants || '');
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
            await createHeatmapRenderPipeline(this.device, this.format, this.isLight, this._wgslConstants || '');
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

        // Write FieldRenderUniforms (pre-allocated buffer)
        const f = _fieldRenderF32;
        const u = _fieldRenderU32;
        f[0] = this.cameraX;
        f[1] = this.cameraY;
        f[2] = this.zoom;
        f[3] = this.canvasWidth;
        f[4] = this.canvasHeight;
        f[5] = this._domainW || 1;
        f[6] = this._domainH || 1;
        u[7] = this.isLight ? 1 : 0;
        u[8] = which === 'higgs' ? 0 : 1;
        f[9] = 0; f[10] = 0; f[11] = 0; // clear padding
        const [c0, c1] = _fieldColors[which];
        f[12] = c0[0]; f[13] = c0[1]; f[14] = c0[2]; f[15] = 1.0;
        f[16] = c1[0]; f[17] = c1[1]; f[18] = c1[2]; f[19] = 1.0;
        this.device.queue.writeBuffer(this._fieldRenderUniformBuffers[which], 0, _fieldRenderData);

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

        // Write HeatmapRenderUniforms (pre-allocated buffer)
        _heatmapRenderF32[0] = this.cameraX;
        _heatmapRenderF32[1] = this.cameraY;
        _heatmapRenderF32[2] = this.zoom;
        _heatmapRenderF32[3] = this.canvasWidth;
        _heatmapRenderF32[4] = this.canvasHeight;
        _heatmapRenderF32[5] = opts.viewLeft || 0;
        _heatmapRenderF32[6] = opts.viewTop || 0;
        _heatmapRenderF32[7] = opts.cellW || 1;
        _heatmapRenderF32[8] = opts.cellH || 1;
        _heatmapRenderF32[9] = HEATMAP_SENSITIVITY;
        _heatmapRenderF32[10] = HEATMAP_MAX_ALPHA / 255;
        _heatmapRenderU32[11] = this.isLight ? 1 : 0;
        _heatmapRenderU32[12] = opts.doGravity ? 1 : 0;
        _heatmapRenderU32[13] = opts.doCoulomb ? 1 : 0;
        _heatmapRenderU32[14] = opts.doYukawa ? 1 : 0;
        this.device.queue.writeBuffer(this._heatmapRenderUniformBuffer, 0, _heatmapRenderData);

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

        // Rebuild spin pipeline (blend mode depends on theme).
        if (this._spinReady) {
            this._spinReady = false;
            this._initSpinRendering();
        }

        // Rebuild trail pipeline (blend mode depends on theme).
        // Save trail bind group resources so initTrailRendering can rebind them.
        if (this._trailReady) {
            this._trailReady = false;
            this._rebuildTrailPipeline();
        }

        // Rebuild arrow pipeline (blend mode depends on theme).
        if (this._arrowReady) {
            this._arrowReady = false;
            this._initArrowRendering();
        }
    }

    /** Rebuild trail pipeline with current theme, preserving existing bind group bindings. */
    async _rebuildTrailPipeline() {
        try {
            const { pipeline, bindGroupLayout } =
                await createTrailRenderPipeline(this.device, this.format, this.isLight, this._wgslConstants || '');
            this._trailPipeline = pipeline;

            // Recreate bind group with same buffers using the new layout
            const oldBG = this._trailBindGroup;
            if (oldBG) {
                // Extract buffer references from existing bind group entries
                // We stored the buffers during initTrailRendering, reuse them
                const b = this.buffers;
                const trailBuffers = this._trailBuffersRef;
                if (trailBuffers) {
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
                            { binding: 8, resource: { buffer: b.particleAux } },
                        ],
                    });
                }
            }
            this._trailReady = true;
        } catch (e) {
            console.warn('[physsim] Trail pipeline rebuild failed:', e.message);
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

async function fetchShader(filename, prepend = '') {
    const resp = await fetch(`src/gpu/shaders/${filename}?v=17`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    const source = await resp.text();
    return prepend ? prepend + '\n' + source : source;
}
