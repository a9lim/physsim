import Physics from './src/integrator.js';
import Renderer from './src/renderer.js';
import InputHandler from './src/input.js';
import Particle from './src/particle.js';
import Heatmap from './src/heatmap.js';
import PhasePlot from './src/phase-plot.js';
import SankeyOverlay from './src/sankey.js';
import { setupUI } from './src/ui.js';
import { ZOOM_MIN, ZOOM_MAX, WHEEL_ZOOM_IN, DEFAULT_SPEED_SCALE, INERTIA_K, PHOTON_LIFETIME, FRAGMENT_COUNT, PHYSICS_DT, MAX_SUBSTEPS } from './src/config.js';

import { setVelocity, angwToAngVel } from './src/relativity.js';
import { computeEnergies } from './src/energy.js';

class Simulation {
    constructor() {
        this.canvas = document.getElementById('simCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = window.innerWidth;
        this.height = window.innerHeight;

        this.particles = [];
        this.physics = new Physics();
        this.renderer = new Renderer(this.ctx, this.width, this.height);
        this.renderer.setTheme(true);

        this.heatmap = new Heatmap();
        this.renderer.heatmap = this.heatmap;

        this.phasePlot = new PhasePlot();
        this.sankey = new SankeyOverlay();

        this.camera = createCamera({
            width: this.width, height: this.height,
            x: this.width / 2, y: this.height / 2,
            minZoom: ZOOM_MIN, maxZoom: ZOOM_MAX,
            wheelFactor: WHEEL_ZOOM_IN,
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
        this.speedScale = DEFAULT_SPEED_SCALE;
        this.initialEnergy = null;
        this.initialMomentum = null;
        this.initialAngMom = null;
        this.selectedParticle = null;
        this.photons = [];
        this.totalRadiated = 0;
        this.totalRadiatedPx = 0;
        this.totalRadiatedPy = 0;
        this.physics.sim = this; // give physics access to photon array

        // Selected particle DOM refs
        this.selDom = {
            details: document.getElementById('particle-details'),
            hint: document.getElementById('particle-hint'),
            phaseSection: document.getElementById('phase-plot-section'),
            id: document.getElementById('sel-id'),
            mass: document.getElementById('sel-mass'),
            charge: document.getElementById('sel-charge'),
            spin: document.getElementById('sel-spin'),
            speed: document.getElementById('sel-speed'),
            gamma: document.getElementById('sel-gamma'),
            force: document.getElementById('sel-force'),
        };

        // Mount visualization canvases into sidebar containers
        document.getElementById('phase-plot-container').appendChild(this.phasePlot.canvas);
        document.getElementById('energy-bars-container').appendChild(this.sankey.canvas);

        this.init();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        setupUI(this);
        requestAnimationFrame((t) => this.loop(t));
    }

    resize() {
        const oldW = this.width, oldH = this.height;
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.renderer.resize(this.width, this.height);
        this.input.updateRect();
        // Shift camera center so top-left world position is preserved
        this.camera.x += (this.width - oldW) / 2;
        this.camera.y += (this.height - oldH) / 2;
        this.camera.viewportW = this.width;
        this.camera.viewportH = this.height;
    }

    computeEnergy() {
        const e = computeEnergies(this.particles, this.physics, this);

        const angMom = e.orbitalAngMom + e.spinAngMom;

        // Total momentum = particle + field + radiated (vector sum)
        const totalPx = e.px + e.fieldPx + this.totalRadiatedPx;
        const totalPy = e.py + e.fieldPy + this.totalRadiatedPy;
        const pMag = Math.sqrt(totalPx * totalPx + totalPy * totalPy);

        const total = e.linearKE + e.spinKE + e.pe + e.fieldEnergy + this.totalRadiated;

        if (this.initialEnergy === null && this.particles.length > 0) {
            this.initialEnergy = total;
            this.initialMomentum = pMag;
            this.initialAngMom = angMom;
        }

        const eDrift = this.initialEnergy !== null && this.initialEnergy !== 0
            ? ((total - this.initialEnergy) / Math.abs(this.initialEnergy) * 100)
            : 0;
        const pDrift = this.initialMomentum !== null && this.initialMomentum !== 0
            ? ((pMag - this.initialMomentum) / Math.abs(this.initialMomentum) * 100)
            : 0;
        const aDrift = this.initialAngMom !== null && this.initialAngMom !== 0
            ? ((angMom - this.initialAngMom) / Math.abs(this.initialAngMom) * 100)
            : 0;

        const fmt = (v) => Math.abs(v) < 0.01 ? '0' : Math.abs(v) > 999 ? v.toExponential(1) : v.toFixed(1);
        const fmtDrift = (v) => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';

        this.dom.linearKE.textContent = fmt(e.linearKE);
        this.dom.spinKE.textContent = fmt(e.spinKE);
        this.dom.potentialE.textContent = fmt(e.pe);
        this.dom.totalE.textContent = fmt(total);
        this.dom.energyDrift.textContent = fmtDrift(eDrift);
        this.dom.fieldE.textContent = fmt(e.fieldEnergy);
        this.dom.radiatedE.textContent = fmt(this.totalRadiated);
        this.sankey.update(e.linearKE, e.spinKE, e.pe, e.fieldEnergy, this.totalRadiated);
        this.dom.momentum.textContent = fmt(pMag);
        this.dom.fieldMom.textContent = fmt(Math.sqrt(e.fieldPx * e.fieldPx + e.fieldPy * e.fieldPy));
        this.dom.radiatedMom.textContent = fmt(Math.sqrt(this.totalRadiatedPx * this.totalRadiatedPx + this.totalRadiatedPy * this.totalRadiatedPy));
        this.dom.momentumDrift.textContent = fmtDrift(pDrift);
        this.dom.angularMomentum.textContent = fmt(angMom);
        this.dom.orbitalAngMom.textContent = fmt(e.orbitalAngMom);
        this.dom.spinAngMom.textContent = fmt(e.spinAngMom);
        this.dom.angMomDrift.textContent = fmtDrift(aDrift);
    }

    addParticle(x, y, vx, vy, options = {}) {
        const p = new Particle(x, y);

        const baseMass = options.mass ?? 10;
        p.mass = Math.max(1, baseMass + (Math.random() - 0.5) * baseMass * 0.2);

        const baseCharge = options.charge ?? 0;
        p.charge = baseCharge !== 0 ? baseCharge + (Math.random() - 0.5) * baseCharge * 0.2 : 0;

        // Spin option is surface velocity as fraction of c
        const baseSV = options.spin ?? 0;
        let sv = baseSV !== 0 ? baseSV + (Math.random() - 0.5) * baseSV * 0.2 : 0;
        sv = Math.max(-0.99, Math.min(0.99, sv));
        // Convert surface velocity to angular celerity: angw = v_s / (r * √(1 - v_s²))
        const absSV = Math.abs(sv);
        p.angw = absSV > 0 ? Math.sign(sv) * absSV / (p.radius * Math.sqrt(1 - absSV * absSV)) : 0;

        p.updateColor();
        setVelocity(p, vx, vy);
        p.angVel = this.physics.relativityEnabled ? angwToAngVel(p.angw, p.radius) : p.angw;
        this.particles.push(p);
        this.initialEnergy = null;
        this.initialMomentum = null;
        this.initialAngMom = null;
        this.physics._forcesInit = false;
    }

    loop(timestamp) {
        const rawDt = Math.min((timestamp - this.lastTime) / 1000, 0.1);
        this.lastTime = timestamp;

        if (this.running) {
            this.accumulator += rawDt * this.speedScale;
            const maxAccum = PHYSICS_DT * MAX_SUBSTEPS * 4;
            if (this.accumulator > maxAccum) this.accumulator = maxAccum;

            const cam = this.camera;
            const halfW = this.width / (2 * cam.zoom);
            const halfH = this.height / (2 * cam.zoom);

            while (this.accumulator >= PHYSICS_DT) {
                this.physics.update(this.particles, PHYSICS_DT, this.collisionMode, this.boundaryMode, halfW * 2, halfH * 2, cam.x - halfW, cam.y - halfH);

                // Update photons (inside fixed step for time consistency)
                for (let i = this.photons.length - 1; i >= 0; i--) {
                    const ph = this.photons[i];
                    ph.update(PHYSICS_DT);

                    if (this.physics.radiationEnabled) {
                        for (const p of this.particles) {
                            const dx = ph.pos.x - p.pos.x, dy = ph.pos.y - p.pos.y;
                            const distSq = dx * dx + dy * dy;
                            if (distSq < p.radius * p.radius) {
                                const impulse = ph.energy / p.mass;
                                p.w.x += ph.vel.x * impulse;
                                p.w.y += ph.vel.y * impulse;
                                ph.alive = false;
                                this.totalRadiated = Math.max(0, this.totalRadiated - ph.energy);
                                this.totalRadiatedPx -= ph.vel.x * ph.energy;
                                this.totalRadiatedPy -= ph.vel.y * ph.energy;
                                break;
                            }
                        }
                    }

                    if (!ph.alive || ph.lifetime > PHOTON_LIFETIME) {
                        this.photons.splice(i, 1);
                    }
                }

                // Tidal breakup (inside fixed step)
                const toFragment = this.physics.checkTidalBreakup(this.particles);
                for (const p of toFragment) {
                    const idx = this.particles.indexOf(p);
                    if (idx === -1) continue;
                    this.particles.splice(idx, 1);

                    const n = FRAGMENT_COUNT;
                    const fragMass = p.mass / n;
                    const fragCharge = p.charge / n;

                    for (let i = 0; i < n; i++) {
                        const angle = (2 * Math.PI * i) / n;
                        const offset = p.radius * 1.5;
                        const fx = p.pos.x + Math.cos(angle) * offset;
                        const fy = p.pos.y + Math.sin(angle) * offset;
                        const tangVx = -Math.sin(angle) * p.angVel * offset;
                        const tangVy = Math.cos(angle) * p.angVel * offset;
                        this.addParticle(fx, fy, p.vel.x + tangVx, p.vel.y + tangVy, {
                            mass: fragMass, charge: fragCharge, spin: p.angw
                        });
                    }
                }

                this.accumulator -= PHYSICS_DT;
            }
        }

        this.heatmap.update(this.particles, this.camera, this.width, this.height);
        this.phasePlot.update(this.particles, this.selectedParticle);
        this.renderer.render(this.particles, PHYSICS_DT, this.camera, this.photons);
        this.phasePlot.draw(this.renderer.isLight);
        this.sankey.draw(this.renderer.isLight);
        if (this.running) this.computeEnergy();
        this.updateSelectedParticle();

        requestAnimationFrame((t) => this.loop(t));
    }

    updateSelectedParticle() {
        const p = this.selectedParticle;
        const dom = this.selDom;

        // Clear if particle was removed
        if (p && !this.particles.includes(p)) {
            this.selectedParticle = null;
        }

        if (!this.selectedParticle) {
            dom.details.hidden = true;
            dom.hint.hidden = false;
            dom.phaseSection.hidden = true;
            return;
        }

        dom.details.hidden = false;
        dom.hint.hidden = true;
        dom.phaseSection.hidden = false;
        const fmt = (v) => Math.abs(v) < 0.01 ? '0' : Math.abs(v) > 999 ? v.toExponential(1) : v.toFixed(2);
        const speed = Math.sqrt(p.vel.x * p.vel.x + p.vel.y * p.vel.y);
        const gamma = this.physics.relativityEnabled
            ? Math.sqrt(1 + p.w.magSq())
            : 1;
        // Sum component vectors for total force (includes Boris display forces)
        const totalFx = p.forceGravity.x + p.forceCoulomb.x + p.forceMagnetic.x + p.forceGravitomag.x;
        const totalFy = p.forceGravity.y + p.forceCoulomb.y + p.forceMagnetic.y + p.forceGravitomag.y;
        const forceMag = Math.sqrt(totalFx * totalFx + totalFy * totalFy);

        dom.id.textContent = p.id;
        dom.mass.textContent = fmt(p.mass);
        dom.charge.textContent = fmt(p.charge);
        const surfaceV = p.angVel * p.radius;
        dom.spin.textContent = surfaceV.toFixed(4) + 'c';
        dom.speed.textContent = speed.toFixed(4) + 'c';
        dom.gamma.textContent = gamma.toFixed(3);
        dom.force.textContent = fmt(forceMag);
    }

}

window.sim = new Simulation();
