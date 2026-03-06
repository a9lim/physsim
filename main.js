import Physics from './src/integrator.js';
import Renderer from './src/renderer.js';
import InputHandler from './src/input.js';
import Particle from './src/particle.js';
import Heatmap from './src/heatmap.js';
import PhasePlot from './src/phase-plot.js';
import StatsDisplay from './src/stats-display.js';
import { setupUI } from './src/ui.js';
import { TWO_PI, WORLD_SCALE, ZOOM_MIN, ZOOM_MAX, WHEEL_ZOOM_IN, DEFAULT_SPEED_SCALE, PHOTON_LIFETIME, FRAGMENT_COUNT, PHYSICS_DT, MAX_SUBSTEPS, MIN_MASS, MAX_PHOTONS, SOFTENING_SQ } from './src/config.js';
import Photon from './src/photon.js';

import { setVelocity, angwToAngVel } from './src/relativity.js';
import { quickSave, quickLoad, downloadState, uploadState } from './src/save-load.js';

class Simulation {
    constructor() {
        this.canvas = document.getElementById('simCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = window.innerWidth;
        this.height = window.innerHeight;

        this.particles = [];
        this.physics = new Physics();
        this.renderer = new Renderer(this.ctx, this.width, this.height);
        this.domainW = this.width / WORLD_SCALE;
        this.domainH = this.height / WORLD_SCALE;
        this.renderer.domainW = this.domainW;
        this.renderer.domainH = this.domainH;
        this.renderer.setTheme(true);

        this.heatmap = new Heatmap();
        this.renderer.heatmap = this.heatmap;

        this.phasePlot = new PhasePlot();

        const sim = this;
        this.camera = createCamera({
            width: this.width, height: this.height,
            x: this.domainW / 2, y: this.domainH / 2,
            zoom: WORLD_SCALE,
            minZoom: ZOOM_MIN, maxZoom: ZOOM_MAX,
            wheelFactor: WHEEL_ZOOM_IN,
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

        this.dom = {
            speedInput: document.getElementById('speedInput'),
            linearKE: document.getElementById('linearKE'),
            spinKE: document.getElementById('spinKE'),
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

        this.collisionMode = 'pass';
        this.boundaryMode = 'despawn';
        this.topology = 'torus';
        this.speedScale = DEFAULT_SPEED_SCALE;
        this.selectedParticle = null;
        this.photons = [];
        this.totalRadiated = 0;
        this.totalRadiatedPx = 0;
        this.totalRadiatedPy = 0;
        this.physics.sim = this;

        // Selected particle DOM refs
        this.selDom = {
            details: document.getElementById('particle-details'),
            hint: document.getElementById('particle-hint'),
            phaseSection: document.getElementById('phase-plot-section'),
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
        };

        // Mount sidebar canvases
        document.getElementById('phase-plot-container').appendChild(this.phasePlot.canvas);

        this.stats = new StatsDisplay(this.dom, this.selDom);

        this.init();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        setupUI(this);

        // Save/Load buttons
        document.getElementById('saveBtn').addEventListener('click', () => quickSave(this));
        document.getElementById('loadBtn').addEventListener('click', () => quickLoad(this));

        // Ctrl+S / Ctrl+L keyboard shortcuts for save/load
        window.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 's') {
                e.preventDefault();
                quickSave(this);
            } else if (e.ctrlKey && e.key === 'l') {
                e.preventDefault();
                quickLoad(this);
            } else if (e.ctrlKey && e.shiftKey && e.key === 'S') {
                e.preventDefault();
                downloadState(this);
            } else if (e.ctrlKey && e.shiftKey && e.key === 'L') {
                e.preventDefault();
                uploadState(this);
            }
        });

        requestAnimationFrame((t) => this.loop(t));
    }

    resize() {
        const oldW = this.width, oldH = this.height;
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.domainW = this.width / WORLD_SCALE;
        this.domainH = this.height / WORLD_SCALE;
        this.renderer.resize(this.width, this.height);
        this.renderer.domainW = this.domainW;
        this.renderer.domainH = this.domainH;
        this.input.updateRect();
        // Preserve top-left world position across resize
        this.camera.x += (this.width - oldW) / (2 * this.camera.zoom);
        this.camera.y += (this.height - oldH) / (2 * this.camera.zoom);
        this.camera.viewportW = this.width;
        this.camera.viewportH = this.height;
    }

    addParticle(x, y, vx, vy, options = {}) {
        const p = new Particle(x, y);

        p.mass = options.mass ?? 10;
        p.charge = options.charge ?? 0;

        p.updateColor();

        // Spin is surface velocity as fraction of c; convert to angular celerity
        let sv = options.spin ?? 0;
        sv = Math.max(-0.99, Math.min(0.99, sv));
        const absSV = Math.abs(sv);
        p.angw = absSV > 0 ? Math.sign(sv) * absSV / (p.radius * Math.sqrt(1 - absSV * absSV)) : 0;
        setVelocity(p, vx, vy);
        p.angVel = this.physics.relativityEnabled ? angwToAngVel(p.angw, p.radius) : p.angw;
        this.particles.push(p);
        if (!options.skipBaseline) this.stats.resetBaseline();
    }

    loop(timestamp) {
        const rawDt = Math.min((timestamp - this.lastTime) / 1000, 0.1);
        this.lastTime = timestamp;

        if (this.running) {
            this.accumulator += rawDt * this.speedScale;
            const maxAccum = PHYSICS_DT * MAX_SUBSTEPS * 4;
            if (this.accumulator > maxAccum) this.accumulator = maxAccum;

            while (this.accumulator >= PHYSICS_DT) {
                this.physics.update(this.particles, PHYSICS_DT, this.collisionMode, this.boundaryMode, this.topology, this.domainW, this.domainH, 0, 0);

                // Update photons, swap-and-pop dead ones (O(n) vs splice O(n²))
                let pLen = this.photons.length;
                for (let i = pLen - 1; i >= 0; i--) {
                    const ph = this.photons[i];
                    ph.update(PHYSICS_DT, this.particles);
                    if (!ph.alive || ph.lifetime > PHOTON_LIFETIME) {
                        this.photons[i] = this.photons[--pLen];
                    }
                }
                this.photons.length = pLen;

                const { fragments: toFragment, transfers: rocheTransfers } = this.physics.checkDisintegration(this.particles, this.physics._lastRoot);
                // Handle Roche lobe overflow mass transfers
                for (const t of rocheTransfers) {
                    t.source.mass -= t.mass;
                    t.source.charge -= t.charge;
                    t.source.updateColor();
                    this.addParticle(t.spawnX, t.spawnY, t.vx, t.vy, {
                        mass: t.mass, charge: t.charge, spin: 0, skipBaseline: true,
                    });
                }
                if (toFragment.length > 0) {
                    // Build set for O(1) lookup, spawn fragments, then compact
                    const fragSet = new Set(toFragment);
                    for (const p of toFragment) {
                        const nf = FRAGMENT_COUNT;
                        const fragMass = p.mass / nf;
                        const fragCharge = p.charge / nf;
                        for (let fi = 0; fi < nf; fi++) {
                            const angle = (TWO_PI * fi) / nf;
                            const offset = p.radius * 1.5;
                            const fx = p.pos.x + Math.cos(angle) * offset;
                            const fy = p.pos.y + Math.sin(angle) * offset;
                            const tangVx = -Math.sin(angle) * p.angVel * offset;
                            const tangVy = Math.cos(angle) * p.angVel * offset;
                            this.addParticle(fx, fy, p.vel.x + tangVx, p.vel.y + tangVy, {
                                mass: fragMass, charge: fragCharge, spin: p.angw, skipBaseline: true,
                            });
                        }
                    }
                    // Single-pass compaction (swap-and-pop style)
                    let write = 0;
                    for (let ri = 0; ri < this.particles.length; ri++) {
                        if (!fragSet.has(this.particles[ri])) {
                            this.particles[write++] = this.particles[ri];
                        }
                    }
                    this.particles.length = write;
                }

                // Hawking evaporation: remove particles below MIN_MASS
                if (this.physics.blackHoleEnabled) {
                    let writeIdx = 0;
                    for (let i = 0; i < this.particles.length; i++) {
                        const p = this.particles[i];
                        if (p.mass >= MIN_MASS) {
                            this.particles[writeIdx++] = p;
                            continue;
                        }
                        // Final burst: emit remaining mass-energy as photons
                        const burstE = Math.max(0, p.mass);
                        if (burstE > 0) {
                            const nBurst = Math.min(5, MAX_PHOTONS - this.photons.length);
                            for (let j = 0; j < nBurst; j++) {
                                const angle = Math.random() * TWO_PI;
                                const cosA = Math.cos(angle), sinA = Math.sin(angle);
                                this.photons.push(new Photon(
                                    p.pos.x + cosA * 3, p.pos.y + sinA * 3,
                                    cosA, sinA, burstE / nBurst, p.id
                                ));
                                this.totalRadiatedPx += (burstE / nBurst) * cosA;
                                this.totalRadiatedPy += (burstE / nBurst) * sinA;
                            }
                            this.totalRadiated += burstE;
                        }
                        if (this.selectedParticle === p) this.selectedParticle = null;
                    }
                    this.particles.length = writeIdx;
                }

                this.accumulator -= PHYSICS_DT;
            }
        }

        this.heatmap.update(this.particles, this.camera, this.width, this.height,
            this.physics.pool, this.physics._lastRoot, this.physics.barnesHutEnabled,
            this.physics.signalDelayEnabled, this.physics.relativityEnabled,
            this.physics.simTime, this.physics.periodic, this.domainW, this.domainH,
            this.topology, this.physics.blackHoleEnabled ? 1 : SOFTENING_SQ,
            this.physics.yukawaEnabled, this.physics.yukawaMu);
        this.phasePlot.update(this.particles, this.selectedParticle);
        this.renderer.render(this.particles, PHYSICS_DT, this.camera, this.photons);
        this.phasePlot.draw(this.renderer.isLight);
        if (this.running) this.stats.updateEnergy(this.particles, this.physics, this);
        const sel = this.stats.updateSelected(this.selectedParticle, this.particles, this.physics);
        if (!sel && this.selectedParticle) this.selectedParticle = null;

        requestAnimationFrame((t) => this.loop(t));
    }
}

window.sim = new Simulation();
