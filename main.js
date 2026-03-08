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
import { TWO_PI, WORLD_SCALE, ZOOM_MIN, ZOOM_MAX, WHEEL_ZOOM_IN, DEFAULT_SPEED_SCALE, PHOTON_LIFETIME, PION_DECAY_PROB, CHARGED_PION_DECAY_PROB, SPAWN_MIN_ENERGY, PHYSICS_DT, MAX_SUBSTEPS, MIN_MASS, MAX_PHOTONS, SOFTENING_SQ, BH_SOFTENING_SQ, MAX_SPEED_RATIO, MAX_FRAME_DT, ACCUMULATOR_CAP, SPAWN_COUNT, spawnOffset, SPAWN_OFFSET_FLOOR, PAIR_PROD_MIN_ENERGY, PAIR_PROD_RADIUS, PAIR_PROD_PROB, PAIR_PROD_MAX_PARTICLES, PAIR_PROD_MIN_AGE, COL_PASS, BOUND_DESPAWN, TORUS, HEATMAP_INTERVAL, SIDEBAR_THROTTLE_MASK } from './src/config.js';
import MasslessBoson from './src/massless-boson.js';
import Pion from './src/pion.js';

import { setVelocity, angwToAngVel } from './src/relativity.js';
import { quickSave, quickLoad, downloadState, uploadState } from './src/save-load.js';

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
        this._hmFrame = 0; // heatmap throttle counter
        this._sbFrame = 0;
        this._dirty = true; // render dirty flag — skip frames when nothing changed

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
        this._domainDiagonal = 2 * Math.sqrt(this.domainW * this.domainW + this.domainH * this.domainH);
        this.renderer.resize(this.width, this.height);
        this.renderer.domainW = this.domainW;
        this.renderer.domainH = this.domainH;
        this.input.updateRect();
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
        this.particles.push(p);
        if (!options.skipBaseline) this.stats.resetBaseline();
        this._dirty = true;
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
        const rawDt = Math.min((timestamp - this.lastTime) / 1000, MAX_FRAME_DT);
        this.lastTime = timestamp;

        if (this.running) {
            this._dirty = true;
            this.accumulator += rawDt * this.speedScale;
            const maxAccum = PHYSICS_DT * MAX_SUBSTEPS * ACCUMULATOR_CAP;
            if (this.accumulator > maxAccum) this.accumulator = maxAccum;

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
                        if (p.mass >= MIN_MASS) {
                            this.particles[writeIdx++] = p;
                            continue;
                        }
                        // Final burst: emit remaining mass-energy as photons
                        if (p.mass > 0) this.emitPhotonBurst(p.pos.x, p.pos.y, p.mass, p.radius, p.id);
                        this.physics._retireParticle(p);
                        if (this.selectedParticle === p) this.selectedParticle = null;
                    }
                    this.particles.length = writeIdx;
                }

                this.accumulator -= PHYSICS_DT;
            }

            // P16: Dead-particle GC once per frame (not per substep) — particles need
            // maxDist * 128 substeps to expire, so per-frame check is sufficient
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

        // Skip render entirely when nothing has changed (paused, no interaction)
        if (this._dirty) {
            this._dirty = false;

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
            const sidebarFrame = !(++this._sbFrame & SIDEBAR_THROTTLE_MASK);
            if (sidebarFrame) {
                this.phasePlot.update(this.particles, this.selectedParticle, this.physics);
                this.effPotPlot.update(this.particles, this.selectedParticle, this.physics);
            }
            this.renderer.render(this.particles, PHYSICS_DT, this.camera, this.photons, this.pions);
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

        if (!this._hidden) requestAnimationFrame((t) => this.loop(t));
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
        requestAnimationFrame((t) => sim.loop(t));
    }
});
