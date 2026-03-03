import Physics from './src/physics.js';
import Renderer from './src/renderer.js';
import InputHandler from './src/input.js';
import Particle from './src/particle.js';
import { setupUI } from './src/ui.js';
import { ZOOM_MIN, ZOOM_MAX, WHEEL_ZOOM_IN, DEFAULT_SPEED_SCALE, INERTIA_K } from './src/config.js';

import { setVelocity, spinToAngVel } from './src/relativity.js';

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
        this.frameCount = 0;
        this.lastFpsTime = 0;

        this.dom = {
            particleCount: document.getElementById('particleCount'),
            fpsCounter: document.getElementById('fpsCounter'),
            simSpeed: document.getElementById('simSpeed'),
            speedInput: document.getElementById('speedInput'),
            linearKE: document.getElementById('linearKE'),
            spinKE: document.getElementById('spinKE'),
            potentialE: document.getElementById('potentialE'),
            totalE: document.getElementById('totalE'),
            energyDrift: document.getElementById('energyDrift'),
            momentum: document.getElementById('momentum'),
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

        // Selected particle DOM refs
        this.selDom = {
            section: document.getElementById('selected-particle-section'),
            id: document.getElementById('sel-id'),
            mass: document.getElementById('sel-mass'),
            charge: document.getElementById('sel-charge'),
            spin: document.getElementById('sel-spin'),
            speed: document.getElementById('sel-speed'),
            gamma: document.getElementById('sel-gamma'),
            force: document.getElementById('sel-force'),
            torque: document.getElementById('sel-torque'),
        };

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
        let linearKE = 0;
        let spinKE = 0;
        let totalMass = 0;
        let comX = 0, comY = 0;
        let px = 0, py = 0;
        const relativity = this.physics.relativityEnabled;

        // First pass: linear KE, spin KE, momentum, COM
        for (const p of this.particles) {
            const rSq = p.radius * p.radius;
            if (relativity) {
                // Relativistic linear KE: (γ - 1)mc², γ = √(1 + w²)
                const gamma = Math.sqrt(1 + p.w.magSq());
                linearKE += (gamma - 1) * p.mass;
                // Relativistic spin KE: E = m·(√(1 + L²/m²) - 1), L = I·S
                const L = INERTIA_K * p.mass * rSq * p.spin;
                spinKE += (Math.sqrt(1 + L * L / (p.mass * p.mass)) - 1) * p.mass;
            } else {
                const speedSq = p.vel.x * p.vel.x + p.vel.y * p.vel.y;
                linearKE += 0.5 * p.mass * speedSq;
                // Classical spin KE: ½Iω² with I = INERTIA_K·m·r²
                spinKE += 0.5 * INERTIA_K * p.mass * rSq * p.angVel * p.angVel;
            }

            // Relativistic momentum: p = mw; classical: p = mv (w = v when relativity off)
            px += p.mass * p.w.x;
            py += p.mass * p.w.y;

            // Accumulate for COM
            totalMass += p.mass;
            comX += p.mass * p.pos.x;
            comY += p.mass * p.pos.y;
        }

        // Momentum magnitude
        const pMag = Math.sqrt(px * px + py * py);

        // Second pass: orbital + spin angular momentum about COM
        let orbitalAngMom = 0;
        let spinAngMom = 0;
        if (totalMass > 0) {
            comX /= totalMass;
            comY /= totalMass;

            for (const p of this.particles) {
                const dx = p.pos.x - comX;
                const dy = p.pos.y - comY;
                // Orbital angular momentum: (r × p)_z about COM
                orbitalAngMom += dx * (p.mass * p.w.y) - dy * (p.mass * p.w.x);
                // Spin angular momentum: I * S = INERTIA_K * m * r² * spin
                spinAngMom += INERTIA_K * p.mass * p.radius * p.radius * p.spin;
            }
        }

        const angMom = orbitalAngMom + spinAngMom;
        const pe = this.physics.potentialEnergy;
        const total = linearKE + spinKE + pe;

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

        // Format numbers
        const fmt = (v) => Math.abs(v) < 0.01 ? '0' : Math.abs(v) > 999 ? v.toExponential(1) : v.toFixed(1);
        const fmtDrift = (v) => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';

        this.dom.linearKE.textContent = fmt(linearKE);
        this.dom.spinKE.textContent = fmt(spinKE);
        this.dom.potentialE.textContent = fmt(pe);
        this.dom.totalE.textContent = fmt(total);
        this.dom.energyDrift.textContent = fmtDrift(eDrift);
        this.dom.momentum.textContent = fmt(pMag);
        this.dom.momentumDrift.textContent = fmtDrift(pDrift);
        this.dom.angularMomentum.textContent = fmt(angMom);
        this.dom.orbitalAngMom.textContent = fmt(orbitalAngMom);
        this.dom.spinAngMom.textContent = fmt(spinAngMom);
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
        // Convert surface velocity to proper angular velocity: spin = v_s / (r * √(1 - v_s²))
        const absSV = Math.abs(sv);
        p.spin = absSV > 0 ? Math.sign(sv) * absSV / (p.radius * Math.sqrt(1 - absSV * absSV)) : 0;

        p.updateColor();
        setVelocity(p, vx, vy);
        p.angVel = this.physics.relativityEnabled ? spinToAngVel(p.spin, p.radius) : p.spin;
        this.particles.push(p);
        this.initialEnergy = null;
        this.initialMomentum = null;
        this.initialAngMom = null;
        this.physics._forcesInit = false;
    }

    loop(timestamp) {
        const rawDt = (timestamp - this.lastTime) / 1000;
        this.lastTime = timestamp;

        const dt = Math.min(rawDt, 0.1) * this.speedScale;

        const cam = this.camera;
        const halfW = this.width / (2 * cam.zoom);
        const halfH = this.height / (2 * cam.zoom);

        if (this.running) {
            this.physics.update(this.particles, dt, this.collisionMode, this.boundaryMode, halfW * 2, halfH * 2, cam.x - halfW, cam.y - halfH);
        }

        this.renderer.render(this.particles, dt, cam);
        if (this.running) this.computeEnergy();
        this.updateStats();
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
            dom.section.hidden = true;
            return;
        }

        dom.section.hidden = false;
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
        dom.torque.textContent = fmt(p.torque);
    }

    updateStats() {
        this.dom.particleCount.textContent = this.particles.length;

        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsTime >= 1000) {
            this.dom.fpsCounter.textContent = this.frameCount;
            this.dom.simSpeed.textContent = this.speedScale + 'x';
            this.frameCount = 0;
            this.lastFpsTime = now;
        }
    }
}

window.sim = new Simulation();
