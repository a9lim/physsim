import Vec2 from './src/vec2.js';
import PhysicsInfo from './src/physics.js'; // Using PhysicsInfo to avoid name conflict if I export default class Physics
import Renderer from './src/renderer.js';
import InputHandler from './src/input.js';
import Particle from './src/particle.js';

// Configuration
const CONFIG = {
    G: 1.0,
    k: 1.0,
    c: 1.0,
    dt: 0.1, // Base time step, will be scaled by speed slider
};

class Simulation {
    constructor() {
        this.canvas = document.getElementById('simCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = window.innerWidth;
        this.height = window.innerHeight;

        this.particles = [];
        this.physics = new PhysicsInfo(CONFIG);
        this.renderer = new Renderer(this.ctx, this.width, this.height);
        this.input = new InputHandler(this.canvas, this);
        this.renderer.input = this.input;

        this.lastTime = 0;
        this.running = true;

        this.init();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());

        // Setup UI hooks
        this.setupUI();

        // Start loop
        requestAnimationFrame((t) => this.loop(t));
    }

    resize() {
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.renderer.resize(this.width, this.height);
    }

    setupUI() {
        document.getElementById('clearBtn').addEventListener('click', () => {
            this.particles = [];
        });

        document.getElementById('pauseBtn').addEventListener('click', (e) => {
            this.running = !this.running;
            e.target.textContent = this.running ? 'Pause' : 'Resume';
        });

        // Toggle groups logic
        document.querySelectorAll('.mode-toggles').forEach(group => {
            group.addEventListener('click', (e) => {
                if (e.target.classList.contains('mode-btn')) {
                    group.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                }
            });
        });

        // Trails toggle
        document.getElementById('trailsToggle').addEventListener('change', (e) => {
            this.renderer.trails = e.target.checked;
        });

        // Step button
        document.getElementById('stepBtn').addEventListener('click', () => {
            if (!this.running) {
                // Manually trigger one update
                const dt = 0.1 * parseFloat(document.getElementById('speedInput').value);
                const collisionMode = document.querySelector('#collision-toggles .mode-btn.active').dataset.collision;
                const boundaryMode = document.querySelector('#boundary-toggles .mode-btn.active').dataset.boundary;
                this.physics.update(this.particles, dt, collisionMode, boundaryMode);
                this.renderer.render(this.particles);
                this.updateStats();
            }
        });

        // Preset selector
        document.getElementById('presetSelect').addEventListener('change', (e) => {
            this.loadPreset(e.target.value);
            // Reset selector to default optionally, or keep it
            e.target.blur(); // Remove focus
        });

        // Theme toggle
        const themeToggleBtn = document.getElementById('themeToggleBtn');
        if (themeToggleBtn) {
            themeToggleBtn.addEventListener('click', () => {
                document.body.classList.toggle('light-theme');
                const isLight = document.body.classList.contains('light-theme');
                themeToggleBtn.textContent = isLight ? 'Dark Mode' : 'Light Mode';
            });
        }
    }

    loadPreset(name) {
        this.particles = [];
        const width = this.width;
        const height = this.height;
        const cx = width / 2;
        const cy = height / 2;

        if (name === 'solar') {
            // Sun
            this.addParticle(cx, cy, 0, 0, { mass: 80, charge: 0 }); // Reduced from 5000 to keep v < c
            // Planets
            for (let i = 0; i < 5; i++) {
                const dist = 100 + i * 60;
                const angle = Math.random() * Math.PI * 2;
                const speed = Math.sqrt(CONFIG.G * 80 / dist);
                const pos = new Vec2(cx + Math.cos(angle) * dist, cy + Math.sin(angle) * dist);
                const vel = new Vec2(-Math.sin(angle) * speed, Math.cos(angle) * speed);
                this.addParticle(pos.x, pos.y, vel.x, vel.y, { mass: 0.5 + Math.random() * 1.5, charge: 0 });
            }
        } else if (name === 'binary') {
            // Two stars
            const dist = 100;
            const starMass = 50;
            const speed = Math.sqrt(CONFIG.G * starMass / (2 * dist));
            this.addParticle(cx - dist, cy, 0, speed, { mass: starMass, charge: 0 });
            this.addParticle(cx + dist, cy, 0, -speed, { mass: starMass, charge: 0 });
        } else if (name === 'galaxy') {
            // Black hole
            this.addParticle(cx, cy, 0, 0, { mass: 150, charge: 0 });
            // Stars
            for (let i = 0; i < 200; i++) {
                const dist = 150 + Math.random() * 300;
                const angle = Math.random() * Math.PI * 2;
                const speed = Math.sqrt(CONFIG.G * 150 / dist);
                const pos = new Vec2(cx + Math.cos(angle) * dist, cy + Math.sin(angle) * dist);
                const vel = new Vec2(-Math.sin(angle) * speed, Math.cos(angle) * speed);
                this.addParticle(pos.x, pos.y, vel.x, vel.y, { mass: 0.1 + Math.random() * 0.4, charge: (Math.random() - 0.5) * 5 });
            }
        } else if (name === 'collision') {
            // Two clusters
            for (let i = 0; i < 50; i++) {
                this.addParticle(cx - 200 + Math.random() * 50, cy + Math.random() * 50, 0.5, 0, { mass: 1, charge: 0 });
                this.addParticle(cx + 200 + Math.random() * 50, cy + Math.random() * 50, -0.5, 0, { mass: 1, charge: 0 });
            }
        }
    }

    addParticle(x, y, vx, vy, options = {}) {
        const p = new Particle(x, y);

        // Random variation if no specific options passed (or even if passed, maybe slight noise?)
        // Instructions: "make sure particle sizes and color vary on placement"
        // Let's add partial randomness to mass and charge if they are coming from UI inputs
        // to make them "vary on placement".

        const baseMass = options.mass || 10;
        const massVar = baseMass * 0.2; // 20% variation
        p.mass = Math.max(1, baseMass + (Math.random() * massVar - massVar / 2));

        const baseCharge = options.charge || 0;
        // Charge variation? If 0, keep 0? Or small noise?
        // If 0, maybe kept 0. If non-zero, vary.
        // User said "color vary continuously based on charge".
        // Let's add small charge noise.
        let chargeVar = 0;
        if (Math.abs(baseCharge) > 0) chargeVar = baseCharge * 0.2;
        p.charge = baseCharge + (Math.random() * chargeVar - chargeVar / 2);

        p.spin = options.spin || 0;

        p.updateColor(); // Recalc radius/color based on new mass/charge

        p.vel = new Vec2(vx, vy);
        // Momentum calc
        const speedSq = p.vel.magSq();
        const cSq = CONFIG.c * CONFIG.c;
        if (speedSq < cSq) {
            const gamma = 1 / Math.sqrt(1 - speedSq / cSq);
            p.momentum = p.vel.clone().scale(gamma * p.mass);
        } else {
            p.vel.normalize().scale(CONFIG.c * 0.99);
            const gamma = 1 / Math.sqrt(1 - 0.99 * 0.99);
            p.momentum = p.vel.clone().scale(gamma * p.mass);
        }

        this.particles.push(p);
    }

    loop(timestamp) {
        const rawDt = (timestamp - this.lastTime) / 1000;
        this.lastTime = timestamp;

        // Get scale from UI
        const speedScale = parseFloat(document.getElementById('speedInput').value);
        const dt = Math.min(rawDt, 0.1) * speedScale; // Clamp rawDt to prevent huge jumps

        // Get collision mode
        const collisionMode = document.querySelector('#collision-toggles .mode-btn.active').dataset.collision;
        const boundaryMode = document.querySelector('#boundary-toggles .mode-btn.active').dataset.boundary;

        if (this.running) {
            this.physics.update(this.particles, dt, collisionMode, boundaryMode);
        }

        this.renderer.render(this.particles);
        this.updateStats();

        requestAnimationFrame((t) => this.loop(t));
    }

    updateStats() {
        document.getElementById('particleCount').textContent = this.particles.length;
        document.getElementById('speedValue').textContent = parseFloat(document.getElementById('speedInput').value).toFixed(1);

        // FPS
        const now = performance.now();
        if (!this.lastFpsTime) this.lastFpsTime = now;
        if (now - this.lastFpsTime >= 1000) {
            document.getElementById('fpsCounter').textContent = this.frameCount || 0;
            this.frameCount = 0;
            this.lastFpsTime = now;
        }
        this.frameCount = (this.frameCount || 0) + 1;
    }
}

// Start
window.sim = new Simulation();
