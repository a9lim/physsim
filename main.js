import Physics from './src/physics.js';
import Renderer from './src/renderer.js';
import InputHandler from './src/input.js';
import Particle from './src/particle.js';
import { setupUI } from './src/ui.js';

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

        this.camera = { x: this.width / 2, y: this.height / 2, zoom: 1 };

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
            zoomLevel: document.getElementById('zoom-level'),
        };

        this.collisionMode = 'pass';
        this.boundaryMode = 'despawn';
        this.speedScale = 20;

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
        this.camera.x += (this.width - oldW) / 2;
        this.camera.y += (this.height - oldH) / 2;
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

        const speedSq = vx * vx + vy * vy;
        if (speedSq < 1) {
            const gamma = 1 / Math.sqrt(1 - speedSq);
            p.vel.set(vx, vy);
            p.momentum.set(vx * gamma * p.mass, vy * gamma * p.mass);
        } else {
            const s = 0.99 / Math.sqrt(speedSq);
            const cvx = vx * s, cvy = vy * s;
            const gamma = 1 / Math.sqrt(1 - 0.99 * 0.99);
            p.vel.set(cvx, cvy);
            p.momentum.set(cvx * gamma * p.mass, cvy * gamma * p.mass);
        }

        this.particles.push(p);
    }

    zoomBy(factor) {
        this.camera.zoom = Math.min(Math.max(this.camera.zoom * factor, 1), 3);
        this.updateZoomDisplay();
    }

    updateZoomDisplay() {
        this.dom.zoomLevel.textContent = Math.round(this.camera.zoom * 100) + '%';
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
        this.updateStats();

        requestAnimationFrame((t) => this.loop(t));
    }

    updateStats() {
        this.dom.particleCount.textContent = this.particles.length;

        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsTime >= 1000) {
            this.dom.fpsCounter.textContent = this.frameCount;
            this.dom.simSpeed.textContent = this.speedScale + 'x';
            this.updateZoomDisplay();
            this.frameCount = 0;
            this.lastFpsTime = now;
        }
    }
}

window.sim = new Simulation();
