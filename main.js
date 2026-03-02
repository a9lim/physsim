import Physics from './src/physics.js';
import Renderer from './src/renderer.js';
import InputHandler from './src/input.js';
import Particle from './src/particle.js';
import { setupUI } from './src/ui.js';
import { ZOOM_MIN, ZOOM_MAX, WHEEL_ZOOM_IN, DEFAULT_SPEED_SCALE } from './src/config.js';

import { setMomentum } from './src/relativity.js';

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
            rotationalKE: document.getElementById('rotationalKE'),
            potentialE: document.getElementById('potentialE'),
            totalE: document.getElementById('totalE'),
            energyDrift: document.getElementById('energyDrift'),
        };

        this.collisionMode = 'pass';
        this.boundaryMode = 'despawn';
        this.speedScale = DEFAULT_SPEED_SCALE;
        this.initialEnergy = null;
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
        let rotationalKE = 0;
        const relativity = this.physics.relativityEnabled;

        for (const p of this.particles) {
            const speedSq = p.vel.x * p.vel.x + p.vel.y * p.vel.y;
            if (relativity) {
                // Relativistic KE: (gamma - 1) * m * c^2, with c=1
                const gamma = 1 / Math.sqrt(1 - Math.min(speedSq, 0.9999));
                linearKE += (gamma - 1) * p.mass;
            } else {
                linearKE += 0.5 * p.mass * speedSq;
            }
            rotationalKE += 0.5 * p.mass * p.spin * p.spin;
        }

        const pe = this.physics.potentialEnergy;
        const total = linearKE + rotationalKE + pe;

        if (this.initialEnergy === null && this.particles.length > 0) {
            this.initialEnergy = total;
        }

        const drift = this.initialEnergy !== null && this.initialEnergy !== 0
            ? ((total - this.initialEnergy) / Math.abs(this.initialEnergy) * 100)
            : 0;

        // Format numbers
        const fmt = (v) => Math.abs(v) < 0.01 ? '0' : Math.abs(v) > 999 ? v.toExponential(1) : v.toFixed(1);

        this.dom.linearKE.textContent = fmt(linearKE);
        this.dom.rotationalKE.textContent = fmt(rotationalKE);
        this.dom.potentialE.textContent = fmt(pe);
        this.dom.totalE.textContent = fmt(total);
        this.dom.energyDrift.textContent = (drift >= 0 ? '+' : '') + drift.toFixed(2) + '%';
    }

    addParticle(x, y, vx, vy, options = {}) {
        const p = new Particle(x, y);

        const baseMass = options.mass ?? 10;
        p.mass = Math.max(1, baseMass + (Math.random() - 0.5) * baseMass * 0.2);

        const baseCharge = options.charge ?? 0;
        p.charge = baseCharge !== 0 ? baseCharge + (Math.random() - 0.5) * baseCharge * 0.2 : 0;

        const baseSpin = options.spin ?? 0;
        p.spin = baseSpin !== 0 ? baseSpin + (Math.random() - 0.5) * baseSpin * 0.2 : 0;

        p.updateColor();
        setMomentum(p, vx, vy);
        this.particles.push(p);
        this.initialEnergy = null;
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
        const speedSq = p.vel.x * p.vel.x + p.vel.y * p.vel.y;
        const speed = Math.sqrt(speedSq);
        const gamma = this.physics.relativityEnabled
            ? 1 / Math.sqrt(1 - Math.min(speedSq, 0.9999))
            : 1;
        const forceMag = Math.sqrt(p.force.x * p.force.x + p.force.y * p.force.y);

        dom.id.textContent = p.id;
        dom.mass.textContent = fmt(p.mass);
        dom.charge.textContent = fmt(p.charge);
        dom.spin.textContent = fmt(p.spin);
        dom.speed.textContent = speed.toFixed(4) + 'c';
        dom.gamma.textContent = gamma.toFixed(3);
        dom.force.textContent = fmt(forceMag);
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
