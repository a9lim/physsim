import Physics from './src/integrator.js';
import Renderer from './src/renderer.js';
import InputHandler from './src/input.js';
import Particle from './src/particle.js';
import Heatmap from './src/heatmap.js';
import HiggsField from './src/higgs-field.js';
import AxionField from './src/axion-field.js';
import PhasePlot from './src/phase-plot.js';
import EffectivePotentialPlot from './src/effective-potential.js';
import StatsDisplay from './src/stats-display.js';
import { setupUI } from './src/ui.js';
import { TWO_PI, WORLD_SCALE, ZOOM_MIN, ZOOM_MAX, WHEEL_ZOOM_IN, DEFAULT_SPEED_SCALE, PHOTON_LIFETIME, PION_DECAY_PROB, CHARGED_PION_DECAY_PROB, SPAWN_MIN_ENERGY, PHYSICS_DT, MAX_SUBSTEPS, MIN_MASS, MAX_PHOTONS, SOFTENING_SQ, BH_SOFTENING_SQ, MAX_SPEED_RATIO, MAX_FRAME_DT, ACCUMULATOR_CAP, SPAWN_COUNT, spawnOffset, SPAWN_OFFSET_FLOOR, PAIR_PROD_MIN_ENERGY, PAIR_PROD_RADIUS, PAIR_PROD_PROB, PAIR_PROD_MAX_PARTICLES, PAIR_PROD_MIN_AGE, COL_PASS, BOUND_DESPAWN, TORUS, HEATMAP_INTERVAL, STATS_THROTTLE_MASK, SIDEBAR_THROTTLE_MASK } from './src/config.js';
import MasslessBoson from './src/massless-boson.js';
import Pion from './src/pion.js';

import { setVelocity, angwToAngVel } from './src/relativity.js';
import { saveState, loadState, quickSave, quickLoad, downloadState, uploadState } from './src/save-load.js';

import { BACKEND_CPU, BACKEND_GPU } from './src/backend-interface.js';
import CPUPhysics from './src/cpu-physics.js';
import CanvasRenderer from './src/canvas-renderer.js';
import GPUPhysics from './src/gpu/gpu-physics.js';
import GPURenderer from './src/gpu/gpu-renderer.js';

/**
 * Detect WebGPU support and return the best available backend.
 * @returns {Promise<{backend: string, device?: GPUDevice}>}
 */
async function selectBackend() {
    // Allow ?cpu=1 URL param to force CPU backend
    const params = new URLSearchParams(window.location.search);
    if (params.get('cpu') === '1') {
        console.log('[physsim] CPU backend forced via ?cpu=1');
        return { backend: BACKEND_CPU };
    }

    if (typeof navigator === 'undefined' || !navigator.gpu) {
        return { backend: BACKEND_CPU };
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return { backend: BACKEND_CPU };

        const device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: 256 * 1024 * 1024,
                maxBufferSize: 256 * 1024 * 1024,
                maxComputeWorkgroupsPerDimension: 65535,
                maxStorageBuffersPerShaderStage: Math.min(
                    adapter.limits.maxStorageBuffersPerShaderStage, 10),
                maxBindingsPerBindGroup: adapter.limits.maxBindingsPerBindGroup,
            },
        });
        if (!device) return { backend: BACKEND_CPU };

        return { backend: BACKEND_GPU, device };
    } catch (e) {
        console.warn('WebGPU detection failed:', e);
        return { backend: BACKEND_CPU };
    }
}

// Auto-save for GPU error recovery (lightweight, in-memory only)
let _gpuAutoSave = null;
const AUTO_SAVE_INTERVAL = 300;  // frames
let _autoSaveCounter = 0;

/**
 * Attempt to re-acquire GPU device after a loss.
 * Does not auto-switch — just logs availability.
 */
async function _attemptGPURecovery() {
    await new Promise(r => setTimeout(r, 5000));
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return;
        const newDevice = await adapter.requestDevice();
        if (!newDevice) return;
        console.log('[physsim] WebGPU device recovered — GPU backend available again');
    } catch (e) {
        console.warn('[physsim] GPU recovery failed:', e);
    }
}

function updateBackendBadge(backend) {
    const badge = document.getElementById('backend-badge');
    if (!badge) return;
    badge.textContent = backend.toUpperCase();
    badge.classList.toggle('gpu', backend === 'gpu');
    badge.title = backend === 'gpu' ? 'WebGPU compute + render' : 'CPU physics + Canvas 2D';
}

class Simulation {
    constructor() {
        this.canvas = document.getElementById('simCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = window.innerWidth;
        this.height = window.innerHeight;

        this.particles = [];
        this.deadParticles = [];
        this.physics = new Physics();
        this.renderer = new Renderer(this.ctx, this.width, this.height);
        this.domainW = this.width / WORLD_SCALE;
        this.domainH = this.height / WORLD_SCALE;
        this._domainDiagonal = 2 * Math.sqrt(this.domainW * this.domainW + this.domainH * this.domainH);
        this.renderer.domainW = this.domainW;
        this.renderer.domainH = this.domainH;
        this.renderer.setTheme(true);

        this.heatmap = new Heatmap();
        this.renderer.heatmap = this.heatmap;

        // A11: Lazy field initialization — defer grid allocations until first toggle-on
        this.higgsField = null;
        this.axionField = null;

        this.phasePlot = new PhasePlot();
        this.effPotPlot = new EffectivePotentialPlot();

        const sim = this;
        this.camera = createCamera({
            width: this.width, height: this.height,
            x: this.domainW / 2, y: this.domainH / 2,
            zoom: WORLD_SCALE,
            minZoom: ZOOM_MIN, maxZoom: ZOOM_MAX,
            wheelFactor: WHEEL_ZOOM_IN,
            onUpdate() { sim._dirty = true; },
            clamp(cam) {
                const halfW = cam.viewportW / (2 * cam.zoom);
                const halfH = cam.viewportH / (2 * cam.zoom);
                const dw = sim.domainW, dh = sim.domainH;
                if (halfW * 2 >= dw) cam.x = dw / 2;
                else cam.x = clamp(cam.x, halfW, dw - halfW);
                if (halfH * 2 >= dh) cam.y = dh / 2;
                else cam.y = clamp(cam.y, halfH, dh - halfH);
            },
        });

        this.input = new InputHandler(this.canvas, this);
        this.renderer.input = this.input;

        this.lastTime = 0;
        this.running = true;
        this.accumulator = 0;
        this._hidden = false;
        this._loopScheduled = false; // prevent duplicate rAF chains
        this._hmFrame = 0; // heatmap throttle counter
        this._sbFrame = 0;
        this._ttFrame = 0; // tooltip refresh throttle counter
        this._dirty = true; // render dirty flag — skip frames when nothing changed

        this.dom = {
            speedInput: document.getElementById('speedInput'),
            linearKE: document.getElementById('linearKE'),
            spinKE: document.getElementById('spinKE'),
            massE: document.getElementById('massE'),
            potentialE: document.getElementById('potentialE'),
            totalE: document.getElementById('totalE'),
            energyDrift: document.getElementById('energyDrift'),
            fieldE: document.getElementById('fieldE'),
            radiatedE: document.getElementById('radiatedE'),
            momentum: document.getElementById('momentum'),
            particleMom: document.getElementById('particleMom'),
            fieldMom: document.getElementById('fieldMom'),
            radiatedMom: document.getElementById('radiatedMom'),
            momentumDrift: document.getElementById('momentumDrift'),
            angularMomentum: document.getElementById('angularMomentum'),
            orbitalAngMom: document.getElementById('orbitalAngMom'),
            spinAngMom: document.getElementById('spinAngMom'),
            angMomDrift: document.getElementById('angMomDrift'),
        };

        this.collisionMode = COL_PASS;
        this.boundaryMode = BOUND_DESPAWN;
        this.topology = TORUS;
        this.speedScale = DEFAULT_SPEED_SCALE;
        this.selectedParticle = null;
        this.photons = [];
        this.pions = [];
        this._MasslessBosonClass = MasslessBoson;  // expose for Pion.decay()
        this.totalRadiated = 0;
        this.totalRadiatedPx = 0;
        this.totalRadiatedPy = 0;
        this.physics.sim = this;

        // Backend detection (async, completes after first frame)
        this.backend = BACKEND_CPU;
        this._cpuPhysics = new CPUPhysics(this.physics);
        this._canvasRenderer = new CanvasRenderer(this.ctx, this.width, this.height);
        // GPU backend will be initialized in Phase 1

        // Selected particle DOM refs
        this.selDom = {
            details: document.getElementById('particle-details'),
            hint: document.getElementById('particle-hint'),
            phaseSection: document.getElementById('phase-plot-section'),
            effPotSection: document.getElementById('eff-pot-section'),
            mass: document.getElementById('sel-mass'),
            charge: document.getElementById('sel-charge'),
            spin: document.getElementById('sel-spin'),
            speed: document.getElementById('sel-speed'),
            gamma: document.getElementById('sel-gamma'),
            force: document.getElementById('sel-force'),
            fbGravity: document.getElementById('fb-gravity'),
            fbGravityVal: document.getElementById('fb-gravity-val'),
            fbCoulomb: document.getElementById('fb-coulomb'),
            fbCoulombVal: document.getElementById('fb-coulomb-val'),
            fbMagnetic: document.getElementById('fb-magnetic'),
            fbMagneticVal: document.getElementById('fb-magnetic-val'),
            fbGravitomag: document.getElementById('fb-gravitomag'),
            fbGravitomagVal: document.getElementById('fb-gravitomag-val'),
            fb1pn: document.getElementById('fb-1pn'),
            fb1pnVal: document.getElementById('fb-1pn-val'),
            fbSpincurv: document.getElementById('fb-spincurv'),
            fbSpincurvVal: document.getElementById('fb-spincurv-val'),
            fbRadiation: document.getElementById('fb-radiation'),
            fbRadiationVal: document.getElementById('fb-radiation-val'),
            fbYukawa: document.getElementById('fb-yukawa'),
            fbYukawaVal: document.getElementById('fb-yukawa-val'),
            fbExternal: document.getElementById('fb-external'),
            fbExternalVal: document.getElementById('fb-external-val'),
            fbHiggs: document.getElementById('fb-higgs'),
            fbHiggsVal: document.getElementById('fb-higgs-val'),
            fbAxion: document.getElementById('fb-axion'),
            fbAxionVal: document.getElementById('fb-axion-val'),
        };

        // Mount sidebar canvases
        document.getElementById('phase-plot-container').appendChild(this.phasePlot.canvas);
        document.getElementById('eff-pot-container').appendChild(this.effPotPlot.canvas);

        this.stats = new StatsDisplay(this.dom, this.selDom);

        selectBackend().then(async ({ backend, device }) => {
            this.backend = backend;
            this._gpuDevice = device || null;
            updateBackendBadge(backend);
            console.log(`[physsim] Backend: ${backend}${device ? ' (WebGPU available)' : ''}`);

            if (backend === BACKEND_GPU && device) {
                try {
                    // Create a separate canvas for GPU rendering (overlaid on CPU canvas).
                    // Cannot reuse simCanvas — it already has a '2d' context.
                    const gpuCanvas = document.createElement('canvas');
                    gpuCanvas.id = 'gpuCanvas';
                    gpuCanvas.width = this.width;
                    gpuCanvas.height = this.height;
                    gpuCanvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;z-index:-1;';
                    this.canvas.parentElement.appendChild(gpuCanvas);

                    this._gpuPhysics = new GPUPhysics(device, this.domainW, this.domainH);
                    this._gpuRenderer = new GPURenderer(gpuCanvas, device, this._gpuPhysics.buffers);
                    await this._gpuPhysics.init();
                    await this._gpuRenderer.init();
                    this._gpuReady = true;
                    // Sync CPU toggle state to GPU uniforms
                    const gpuToggles = Object.create(this.physics);
                    gpuToggles.heatmapEnabled = this.heatmap && this.heatmap.enabled;
                    this._gpuPhysics.setToggles(gpuToggles);
                    // Sync boundary/collision/topology (live on sim, not sim.physics)
                    this._gpuPhysics.boundaryMode = this.boundaryMode;
                    this._gpuPhysics.topologyMode = this.topology;
                    this._gpuPhysics._collisionMode = this.collisionMode;

                    // Sync any CPU particles that were added before GPU was ready
                    // (e.g., preset loaded while shaders were still compiling)
                    if (this.particles.length > 0 && this._gpuPhysics.aliveCount === 0) {
                        for (const p of this.particles) {
                            p._gpuIdx = this._gpuPhysics.addParticle({
                                x: p.pos.x, y: p.pos.y,
                                vx: p.w.x, vy: p.w.y,
                                mass: p.mass, charge: p.charge,
                                angw: p.angw,
                                antimatter: p.antimatter,
                            });
                        }
                    }

                    console.log('[physsim] GPU backend initialized');
                    if (this._onGPUReady) this._onGPUReady();

                    // Register device.lost handler for error recovery
                    device.lost.then((info) => {
                        console.error('[physsim] GPU device lost:', info.message);
                        this._gpuReady = false;
                        this.backend = BACKEND_CPU;
                        updateBackendBadge(BACKEND_CPU);
                        gpuCanvas.remove();

                        // Restore from auto-save if available
                        if (_gpuAutoSave) {
                            loadState(_gpuAutoSave, this);
                            showToast('GPU lost \u2014 restored from auto-save (CPU mode)');
                        } else {
                            showToast('GPU lost \u2014 switched to CPU mode');
                        }

                        this._dirty = true;
                        if (this._onGPULost) this._onGPULost();
                        _attemptGPURecovery();
                    });

                } catch (e) {
                    console.error('[physsim] GPU init failed, falling back to CPU:', e);
                    this._gpuReady = false;
                }
            }
        });

        this.init();
    }

    // A11: Lazy field initialization — create on first use
    ensureHiggsField() {
        if (!this.higgsField) {
            this.higgsField = new HiggsField();
            this.renderer.higgsField = this.higgsField;
        }
        return this.higgsField;
    }
    ensureAxionField() {
        if (!this.axionField) {
            this.axionField = new AxionField();
            this.renderer.axionField = this.axionField;
        }
        return this.axionField;
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        setupUI(this);

        // Save/Load buttons
        document.getElementById('saveBtn').addEventListener('click', () => quickSave(this));
        document.getElementById('loadBtn').addEventListener('click', () => { quickLoad(this); this._dirty = true; });

        // Ctrl+S / Ctrl+L keyboard shortcuts for save/load
        window.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 's') {
                e.preventDefault();
                quickSave(this);
            } else if (e.ctrlKey && e.key === 'l') {
                e.preventDefault();
                quickLoad(this);
                this._dirty = true;
            } else if (e.ctrlKey && e.shiftKey && e.key === 'S') {
                e.preventDefault();
                downloadState(this);
            } else if (e.ctrlKey && e.shiftKey && e.key === 'L') {
                e.preventDefault();
                uploadState(this);
                this._dirty = true;
            }
        });

        this._scheduleLoop();
    }

    resize() {
        const oldW = this.width, oldH = this.height;
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.domainW = this.width / WORLD_SCALE;
        this.domainH = this.height / WORLD_SCALE;
        this._domainDiagonal = 2 * Math.sqrt(this.domainW * this.domainW + this.domainH * this.domainH);
        this.renderer.resize(this.width, this.height);
        this.renderer.domainW = this.domainW;
        this.renderer.domainH = this.domainH;
        this.input.updateRect();
        // Sync GPU canvas size
        if (this._gpuRenderer) {
            const gpuCanvas = document.getElementById('gpuCanvas');
            if (gpuCanvas) {
                gpuCanvas.width = this.width;
                gpuCanvas.height = this.height;
            }
            this._gpuRenderer.resize(this.width, this.height);
        }
        // R11: Refresh cached layout dimensions for sidebar plots
        this.phasePlot.cacheSize();
        this.effPotPlot.cacheSize();
        // Preserve top-left world position across resize
        this.camera.x += (this.width - oldW) / (2 * this.camera.zoom);
        this.camera.y += (this.height - oldH) / (2 * this.camera.zoom);
        this.camera.viewportW = this.width;
        this.camera.viewportH = this.height;
        this._dirty = true;
    }

    addParticle(x, y, vx, vy, options = {}) {
        // Always maintain CPU-side particle array (needed for presets, sidebar, etc.)
        const p = new Particle(x, y);
        p.mass = options.mass ?? 10;
        p.baseMass = options.baseMass ?? p.mass;
        p.charge = options.charge ?? 0;
        p.antimatter = this.physics.blackHoleEnabled ? false : (options.antimatter ?? false);

        p.creationTime = this.physics.simTime;
        p.updateColor();

        // Spin is surface velocity as fraction of c; convert to angular celerity
        let sv = options.spin ?? 0;
        sv = Math.max(-MAX_SPEED_RATIO, Math.min(MAX_SPEED_RATIO, sv));
        const absSV = Math.abs(sv);
        p.angw = absSV > 0 ? Math.sign(sv) * absSV / (p.radius * Math.sqrt(1 - absSV * absSV)) : 0;
        setVelocity(p, vx, vy);
        p.angVel = this.physics.relativityEnabled ? angwToAngVel(p.angw, p.radius) : p.angw;

        if (this._gpuReady && this.backend === BACKEND_GPU) {
            // GPU path: write directly to GPU SoA buffers (after CPU particle is fully built)
            p._gpuIdx = this._gpuPhysics.addParticle({
                x, y, vx, vy,
                mass: p.mass,
                charge: p.charge,
                angw: p.angw,
                antimatter: p.antimatter,
            });
        }

        this.particles.push(p);
        if (!options.skipBaseline) this.stats.resetBaseline();
        this._dirty = true;
    }

    /** Backend-agnostic reset: clears all simulation state. */
    reset() {
        // CPU state
        this.particles = [];
        this.deadParticles = [];
        this.clearBosons();
        this.totalRadiated = 0;
        this.totalRadiatedPx = 0;
        this.totalRadiatedPy = 0;
        this.selectedParticle = null;
        this.physics._forcesInit = false;
        if (this.higgsField) this.higgsField.reset();
        if (this.axionField) this.axionField.reset();
        // GPU state (if active)
        if (this._gpuPhysics) this._gpuPhysics.reset();
    }

    /** Release all active photons/pions to their pools and clear the arrays. */
    clearBosons() {
        for (let i = 0; i < this.photons.length; i++) MasslessBoson.release(this.photons[i]);
        for (let i = 0; i < this.pions.length; i++) Pion.release(this.pions[i]);
        this.photons.length = 0;
        this.pions.length = 0;
    }

    emitPhotonBurst(x, y, energy, radius, emitterId) {
        const n = Math.min(Math.max(1, Math.floor(energy / SPAWN_MIN_ENERGY)), MAX_PHOTONS - this.photons.length);
        if (n <= 0) return;
        const offset = spawnOffset(radius);
        const ePerPh = energy / n;
        for (let j = 0; j < n; j++) {
            const angle = Math.random() * TWO_PI;
            const cosA = Math.cos(angle), sinA = Math.sin(angle);
            this.photons.push(MasslessBoson.acquire(
                x + cosA * offset, y + sinA * offset,
                cosA, sinA, ePerPh, emitterId
            ));
            this.totalRadiatedPx += ePerPh * cosA;
            this.totalRadiatedPy += ePerPh * sinA;
        }
        this.totalRadiated += energy;
    }

    markDirty() { this._dirty = true; }

    loop(timestamp) {
      this._loopScheduled = false;
      try { this._loopBody(timestamp); }
      catch (e) { console.error('[physsim] loop error:', e); }
      if (!this._hidden) this._scheduleLoop();
    }

    _scheduleLoop() {
        if (this._loopScheduled) return; // prevent duplicate rAF chains
        this._loopScheduled = true;
        requestAnimationFrame((t) => this.loop(t));
    }

    _loopBody(timestamp) {
        const rawDt = Math.min((timestamp - this.lastTime) / 1000, MAX_FRAME_DT);
        this.lastTime = timestamp;

        if (this.running) {
            this._dirty = true;
            this.accumulator += rawDt * this.speedScale;
            const maxAccum = PHYSICS_DT * MAX_SUBSTEPS * ACCUMULATOR_CAP;
            if (this.accumulator > maxAccum) this.accumulator = maxAccum;

            if (this._gpuReady && this.backend === BACKEND_GPU) {
                // ─── GPU physics path ───
                this._gpuPhysics.setCamera(this.camera);
                const substeps = Math.floor(this.accumulator / PHYSICS_DT);
                if (substeps > 0) {
                    this._gpuPhysics.update(PHYSICS_DT * substeps);
                    this.accumulator -= substeps * PHYSICS_DT;
                }

                // Periodic auto-save for GPU error recovery (non-blocking)
                if (++_autoSaveCounter >= AUTO_SAVE_INTERVAL) {
                    _autoSaveCounter = 0;
                    this._gpuPhysics.serialize(this).then(state => { _gpuAutoSave = state; });
                }
            } else {
                // ─── CPU physics path ───
                this.physics._collisionCount = 0;
                while (this.accumulator >= PHYSICS_DT) {
                    this.physics.update(this.particles, PHYSICS_DT, this.collisionMode, this.boundaryMode, this.topology, this.domainW, this.domainH, 0, 0);

                    // Update photons, swap-and-pop dead ones, release to pool
                    // Gravitational lensing only when gravity is on
                    const _bosonGrav = this.physics.bosonGravEnabled;
                    const _pool = (_bosonGrav && this.physics.barnesHutEnabled) ? this.physics.pool : null;
                    const _root = this.physics._lastRoot;
                    const _lensParticles = _bosonGrav ? this.particles : null;
                    let pLen = this.photons.length;
                    for (let i = pLen - 1; i >= 0; i--) {
                        const ph = this.photons[i];
                        ph.update(PHYSICS_DT, _lensParticles, _pool, _root);
                        if (!ph.alive || ph.lifetime > PHOTON_LIFETIME) {
                            ph.alive = false;
                            MasslessBoson.release(ph);
                            this.photons[i] = this.photons[--pLen];
                        }
                    }
                    this.photons.length = pLen;

                    // Pair production: energetic photon near massive body -> matter + antimatter
                    // BH mode: no antimatter (no hair)
                    const canPairProduce = this.particles.length < PAIR_PROD_MAX_PARTICLES && !this.physics.blackHoleEnabled;
                    for (let i = canPairProduce ? pLen - 1 : -1; i >= 0; i--) {
                        const ph = this.photons[i];
                        if (ph.energy < PAIR_PROD_MIN_ENERGY || ph.lifetime < PAIR_PROD_MIN_AGE) continue;
                        // Check proximity to any massive body
                        let nearBody = false;
                        for (let j = 0; j < this.particles.length; j++) {
                            const p = this.particles[j];
                            const dx = ph.pos.x - p.pos.x, dy = ph.pos.y - p.pos.y;
                            if (dx * dx + dy * dy < PAIR_PROD_RADIUS * PAIR_PROD_RADIUS * p.mass) {
                                nearBody = true;
                                break;
                            }
                        }
                        if (!nearBody || Math.random() > PAIR_PROD_PROB) continue;
                        // Produce pair: split photon energy into mass + kinetic
                        const pairMass = ph.energy * 0.5;
                        const offset = SPAWN_OFFSET_FLOOR;
                        // Perpendicular to photon direction
                        const px = -ph.vel.y, py = ph.vel.x;
                        this.addParticle(ph.pos.x + px * offset, ph.pos.y + py * offset,
                            ph.vel.x * 0.1, ph.vel.y * 0.1,
                            { mass: pairMass, charge: 0, antimatter: false, skipBaseline: true });
                        this.addParticle(ph.pos.x - px * offset, ph.pos.y - py * offset,
                            ph.vel.x * 0.1, ph.vel.y * 0.1,
                            { mass: pairMass, charge: 0, antimatter: true, skipBaseline: true });
                        // Kill the photon and release to pool
                        ph.alive = false;
                        MasslessBoson.release(ph);
                        this.photons[i] = this.photons[--pLen];
                    }
                    this.photons.length = pLen;

                    // Update pions: move, decay, swap-and-pop dead, release to pool
                    let piLen = this.pions.length;
                    for (let i = piLen - 1; i >= 0; i--) {
                        const pn = this.pions[i];
                        pn.update(PHYSICS_DT, _lensParticles, _pool, _root);
                        if (!pn.alive) {
                            Pion.release(pn);
                            this.pions[i] = this.pions[--piLen];
                        } else if (Math.random() < (pn.charge === 0 ? PION_DECAY_PROB : CHARGED_PION_DECAY_PROB)) {
                            pn.decay(this);
                            Pion.release(pn);
                            this.pions[i] = this.pions[--piLen];
                        }
                    }
                    this.pions.length = piLen;

                    const { fragments: toFragment, transfers: rocheTransfers } = this.physics.checkDisintegration(this.particles, this.physics._lastRoot);
                    // Handle Roche lobe overflow mass transfers
                    for (let ti = 0; ti < rocheTransfers.length; ti++) {
                        const t = rocheTransfers[ti];
                        const origM = t.source.mass;
                        t.source.mass -= t.mass;
                        if (origM > 0) t.source.baseMass *= t.source.mass / origM;
                        t.source.charge -= t.charge;
                        t.source.updateColor();
                        this.addParticle(t.spawnX, t.spawnY, t.vx, t.vy, {
                            mass: t.mass, charge: t.charge, spin: 0, skipBaseline: true,
                        });
                    }
                    if (toFragment.length > 0) {
                        // Build set for O(1) lookup, spawn fragments, then compact
                        if (!this._fragSet) this._fragSet = new Set();
                        const fragSet = this._fragSet;
                        fragSet.clear();
                        for (let fi = 0; fi < toFragment.length; fi++) fragSet.add(toFragment[fi]);
                        for (const p of toFragment) {
                            const nf = SPAWN_COUNT;
                            const fragMass = p.mass / nf;
                            const fragBaseMass = p.baseMass / nf;
                            const fragCharge = p.charge / nf;
                            for (let fi = 0; fi < nf; fi++) {
                                const angle = (TWO_PI * fi) / nf;
                                const offset = spawnOffset(p.radius);
                                const fx = p.pos.x + Math.cos(angle) * offset;
                                const fy = p.pos.y + Math.sin(angle) * offset;
                                const tangVx = -Math.sin(angle) * p.angVel * offset;
                                const tangVy = Math.cos(angle) * p.angVel * offset;
                                this.addParticle(fx, fy, p.vel.x + tangVx, p.vel.y + tangVy, {
                                    mass: fragMass, baseMass: fragBaseMass, charge: fragCharge, spin: p.angw, skipBaseline: true,
                                });
                            }
                        }
                        // Single-pass compaction (swap-and-pop style)
                        let write = 0;
                        for (let ri = 0; ri < this.particles.length; ri++) {
                            if (!fragSet.has(this.particles[ri])) {
                                this.particles[write++] = this.particles[ri];
                            } else {
                                this.physics._retireParticle(this.particles[ri]);
                            }
                        }
                        this.particles.length = write;
                    }

                    // Hawking evaporation: remove particles below MIN_MASS
                    if (this.physics.blackHoleEnabled) {
                        let writeIdx = 0;
                        for (let i = 0; i < this.particles.length; i++) {
                            const p = this.particles[i];
                            if (p.mass > MIN_MASS) {
                                this.particles[writeIdx++] = p;
                                continue;
                            }
                            // Final burst: emit accumulated + remaining mass-energy as photons
                            const finalE = p._hawkAccum + p.mass;
                            if (finalE > 0) this.emitPhotonBurst(p.pos.x, p.pos.y, finalE, p.radius, p.id);
                            this.physics._retireParticle(p);
                            if (this.selectedParticle === p) this.selectedParticle = null;
                        }
                        this.particles.length = writeIdx;
                    }

                    this.accumulator -= PHYSICS_DT;
                }
                if (this.physics._collisionCount > 0) _haptics.trigger('buzz');

                // P16: Dead-particle GC once per frame (not per substep)
                if (this.deadParticles.length > 0) {
                    const maxDist = this._domainDiagonal;
                    let dw = 0;
                    for (let i = 0; i < this.deadParticles.length; i++) {
                        const dp = this.deadParticles[i];
                        if (this.physics.simTime - dp.deathTime < maxDist) {
                            this.deadParticles[dw++] = dp;
                        }
                    }
                    this.deadParticles.length = dw;
                }
            }
        }

        // Poll for async GPU hit test results (selection override)
        if (this._gpuReady && this.backend === BACKEND_GPU) {
            this.input.pollGPUHitResult();
        }

        // Refresh hover tooltip periodically (every 8th frame, matching stats rate)
        if (this.running && !(++this._ttFrame & STATS_THROTTLE_MASK)) {
            this.input.refreshTooltip();
        }

        // Skip render entirely when nothing has changed (paused, no interaction)
        if (this._dirty) {
            this._dirty = false;

            if (this._gpuReady && this.backend === BACKEND_GPU) {
                // ─── GPU render path ───
                this._gpuRenderer.updateCamera(this.camera);
                // Sync visual toggles from CPU renderer
                this._gpuRenderer.showForce = this.renderer.showForce;
                this._gpuRenderer.showForceComponents = this.renderer.showForceComponents;
                this._gpuRenderer.showVelocity = this.renderer.showVelocity;
                this._gpuRenderer.showTrails = this.renderer.trails;
                this._gpuRenderer.setDomain(this.domainW, this.domainH);

                // Sync trails enabled to GPU physics (lazy trail buffer allocation)
                this._gpuPhysics.setTrailsEnabled(this.renderer.trails);

                // Lazily init trail render pipeline when trail buffers become available
                const trailBufs = this._gpuPhysics.getTrailBuffers();
                if (trailBufs && !this._gpuRenderer._trailReady) {
                    this._gpuRenderer.initTrailRendering(trailBufs);
                }

                // Lazily init field overlay render pipeline when fields are active
                const ph = this.physics;
                if ((ph.higgsEnabled || ph.axionEnabled) && !this._gpuRenderer._fieldRenderReady) {
                    this._gpuRenderer.initFieldOverlay();
                }

                // Lazily init heatmap overlay render pipeline when heatmap is active
                if (this.heatmap.enabled && !this._gpuRenderer._heatmapRenderReady) {
                    this._gpuRenderer.initHeatmapOverlay();
                }

                // Build render opts with field/heatmap buffers
                const gpuPh = this._gpuPhysics;
                const renderOpts = {
                    blackHoleEnabled: ph.blackHoleEnabled,
                    enabledForces: {
                        gravity: ph.gravityEnabled,
                        coulomb: ph.coulombEnabled,
                        magnetic: ph.magneticEnabled,
                        gravitomag: ph.gravitomagEnabled,
                        onePN: ph.onePNEnabled,
                        spinOrbit: ph.spinOrbitEnabled,
                        radiation: ph.radiationEnabled,
                        yukawa: ph.yukawaEnabled,
                        external: (ph.extGravity !== 0 || ph.extElectric !== 0 || ph.extBz !== 0),
                        higgs: ph.higgsEnabled,
                        axion: ph.axionEnabled,
                    },
                };

                // Pass field buffers for overlay rendering
                if (ph.higgsEnabled) {
                    const hb = gpuPh.getFieldBuffers('higgs');
                    if (hb) renderOpts.higgsField = hb.field;
                }
                if (ph.axionEnabled) {
                    const ab = gpuPh.getFieldBuffers('axion');
                    if (ab) renderOpts.axionField = ab.field;
                }

                // Pass heatmap buffers for overlay rendering
                if (this.heatmap.enabled) {
                    const hmBufs = gpuPh.getHeatmapBuffers();
                    if (hmBufs) {
                        renderOpts.heatmapBuffers = hmBufs;
                        // Compute heatmap viewport info from camera — use physical pixels
                        // to match render shader (heatmap-render.wgsl reconstructs world
                        // coords from canvasW which is physical pixel size)
                        const cam = this.camera;
                        const dpr = devicePixelRatio || 1;
                        const viewW = this.width * dpr / cam.zoom;
                        const viewH = this.height * dpr / cam.zoom;
                        const viewLeft = cam.x - viewW / 2;
                        const viewTop = cam.y - viewH / 2;
                        renderOpts.heatmapOpts = {
                            viewLeft,
                            viewTop,
                            cellW: viewW / 64,
                            cellH: viewH / 64,
                            doGravity: ph.gravityEnabled,
                            doCoulomb: ph.coulombEnabled,
                            doYukawa: ph.yukawaEnabled,
                        };
                    }
                }

                this._gpuRenderer.render(gpuPh.aliveCount, renderOpts);

                // Draw drag indicator on CPU canvas (sits above GPU canvas)
                this.renderer.drawDragOverlay(this.camera);
            } else {
                // ─── CPU render path ───
                // Throttle heatmap to every HEATMAP_INTERVAL frames (default 4 = ~15fps)
                // Skip when paused — state hasn't changed (camera changes trigger re-render via renderer)
                if (this.running && ++this._hmFrame >= HEATMAP_INTERVAL) {
                    this._hmFrame = 0;
                    this.heatmap.update(this.particles, this.camera, this.width, this.height,
                        this.physics.pool, this.physics._lastRoot, this.physics.barnesHutEnabled,
                        this.physics.relativityEnabled,
                        this.physics.simTime, this.physics.periodic, this.domainW, this.domainH,
                        this.topology, this.physics.blackHoleEnabled ? BH_SOFTENING_SQ : SOFTENING_SQ,
                        this.physics.yukawaEnabled, this.physics.yukawaMu, this.deadParticles,
                        this.physics.gravityEnabled, this.physics.coulombEnabled);
                }
                this.renderer.render(this.particles, PHYSICS_DT, this.camera, this.photons, this.pions);
            }

            // Sidebar plots + stats
            if (this._gpuReady && this.backend === BACKEND_GPU) {
                // GPU mode: request stats at throttle rate, read results each frame
                if (this.running && !(this._sbFrame & STATS_THROTTLE_MASK)) {
                    const selIdx = this.selectedParticle ? (this.selectedParticle._gpuIdx ?? -1) : -1;
                    this._gpuPhysics.requestStats(selIdx);
                }
                this._sbFrame++;
                const gpuStats = this._gpuPhysics.readStats();
                if (gpuStats) {
                    this.stats.updateEnergyGPU(gpuStats, this);
                    if (gpuStats.selected) {
                        this.stats.updateSelectedGPU(gpuStats.selected, this.physics);
                    } else if (this.selectedParticle) {
                        this.stats.updateSelectedGPU(null, this.physics);
                    }
                }
            } else {
                const sidebarFrame = !(++this._sbFrame & SIDEBAR_THROTTLE_MASK);
                if (sidebarFrame) {
                    this.phasePlot.update(this.particles, this.selectedParticle, this.physics);
                    this.effPotPlot.update(this.particles, this.selectedParticle, this.physics);
                }
                if (sidebarFrame) {
                    this.phasePlot.draw(this.renderer.isLight);
                    this.effPotPlot.draw(this.renderer.isLight);
                }
                if (this.running) this.stats.updateEnergy(this.particles, this.physics, this);
                if (sidebarFrame) {
                    const sel = this.stats.updateSelected(this.selectedParticle, this.particles, this.physics);
                    if (!sel && this.selectedParticle) this.selectedParticle = null;
                }
            }
        }

    }
}

const sim = new Simulation();
window.sim = sim;

// A5: Halt rAF loop when tab is hidden, resume with reset lastTime on visible
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        sim._hidden = true;
    } else {
        sim._hidden = false;
        sim.lastTime = 0;
        sim._scheduleLoop(); // safe: prevents duplicate rAF chains
    }
});
