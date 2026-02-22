import Vec2 from './src/vec2.js';
import PhysicsInfo from './src/physics.js';
import Renderer from './src/renderer.js';
import InputHandler from './src/input.js';
import Particle from './src/particle.js';

// Natural units: all constants = 1

class Simulation {
    constructor() {
        this.canvas = document.getElementById('simCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = window.innerWidth;
        this.height = window.innerHeight;

        this.particles = [];
        this.physics = new PhysicsInfo();
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
        this.setupUI();
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

        const pauseIcon = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`;
        const playIcon = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;

        document.getElementById('pauseBtn').addEventListener('click', (e) => {
            this.running = !this.running;
            e.currentTarget.innerHTML = this.running ? pauseIcon : playIcon;
            e.currentTarget.title = this.running ? 'Pause' : 'Resume';
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

        // Slider value displays
        const sliderConfig = [
            { id: 'massInput', display: 'massValue' },
            { id: 'chargeInput', display: 'chargeValue' },
            { id: 'spinInput', display: 'spinValue' },
            { id: 'speedInput', display: 'speedValue' },
        ];

        sliderConfig.forEach(({ id, display }) => {
            const slider = document.getElementById(id);
            const label = document.getElementById(display);
            if (slider && label) {
                slider.addEventListener('input', () => {
                    label.textContent = slider.value;
                });
            }
        });

        // Step button
        document.getElementById('stepBtn').addEventListener('click', () => {
            if (!this.running) {
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
            e.target.blur();
        });

        // Theme toggle
        const themeToggleBtn = document.getElementById('themeToggleBtn');
        if (themeToggleBtn) {
            const sunIcon = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 2a10 10 0 0 0 0 20z" fill="currentColor"></path></svg>`;
            const moonIcon = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;

            themeToggleBtn.addEventListener('click', () => {
                document.body.classList.toggle('light-theme');
                const isLight = document.body.classList.contains('light-theme');
                themeToggleBtn.innerHTML = isLight ? moonIcon : sunIcon;
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
            this.addParticle(cx, cy, 0, 0, { mass: 80, charge: 0, spin: 0 });
            for (let i = 0; i < 5; i++) {
                const dist = 100 + i * 60;
                const angle = Math.random() * Math.PI * 2;
                const speed = Math.sqrt(CONFIG.G * 80 / dist);
                const pos = new Vec2(cx + Math.cos(angle) * dist, cy + Math.sin(angle) * dist);
                const vel = new Vec2(-Math.sin(angle) * speed, Math.cos(angle) * speed);
                this.addParticle(pos.x, pos.y, vel.x, vel.y, { mass: 0.5 + Math.random() * 1.5, charge: 0, spin: 0 });
            }
        } else if (name === 'binary') {
            const dist = 100;
            const starMass = 50;
            const speed = Math.sqrt(CONFIG.G * starMass / (2 * dist));
            this.addParticle(cx - dist, cy, 0, speed, { mass: starMass, charge: 0, spin: 10 });
            this.addParticle(cx + dist, cy, 0, -speed, { mass: starMass, charge: 0, spin: 10 });
        } else if (name === 'galaxy') {
            this.addParticle(cx, cy, 0, 0, { mass: 150, charge: 0, spin: 30 });
            for (let i = 0; i < 200; i++) {
                const dist = 150 + Math.random() * 300;
                const angle = Math.random() * Math.PI * 2;
                const speed = Math.sqrt(CONFIG.G * 150 / dist);
                const pos = new Vec2(cx + Math.cos(angle) * dist, cy + Math.sin(angle) * dist);
                const vel = new Vec2(-Math.sin(angle) * speed, Math.cos(angle) * speed);
                this.addParticle(pos.x, pos.y, vel.x, vel.y, {
                    mass: 0.1 + Math.random() * 0.4,
                    charge: (Math.random() - 0.5) * 5,
                    spin: (Math.random() - 0.5) * 10
                });
            }
        } else if (name === 'collision') {
            for (let i = 0; i < 50; i++) {
                this.addParticle(cx - 200 + Math.random() * 50, cy + Math.random() * 50, 0.5, 0, { mass: 1, charge: 0, spin: 0 });
                this.addParticle(cx + 200 + Math.random() * 50, cy + Math.random() * 50, -0.5, 0, { mass: 1, charge: 0, spin: 0 });
            }
        } else if (name === 'magnetic') {
            // Magnetic demo: charged spinning particles attracting via aligned dipoles
            const spacing = 80;
            for (let i = -2; i <= 2; i++) {
                for (let j = -2; j <= 2; j++) {
                    const x = cx + i * spacing + (Math.random() - 0.5) * 20;
                    const y = cy + j * spacing + (Math.random() - 0.5) * 20;
                    const spin = 20 + Math.random() * 10; // All co-rotating
                    this.addParticle(x, y, (Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.1, {
                        mass: 3 + Math.random() * 2,
                        charge: 5 + Math.random() * 5,
                        spin: spin
                    });
                }
            }
        }
    }

    addParticle(x, y, vx, vy, options = {}) {
        const p = new Particle(x, y);

        const baseMass = options.mass || 10;
        const massVar = baseMass * 0.2;
        p.mass = Math.max(1, baseMass + (Math.random() * massVar - massVar / 2));

        const baseCharge = options.charge || 0;
        let chargeVar = 0;
        if (Math.abs(baseCharge) > 0) chargeVar = baseCharge * 0.2;
        p.charge = baseCharge + (Math.random() * chargeVar - chargeVar / 2);

        const baseSpin = options.spin || 0;
        let spinVar = 0;
        if (Math.abs(baseSpin) > 0) spinVar = baseSpin * 0.2;
        p.spin = baseSpin + (Math.random() * spinVar - spinVar / 2);

        p.updateColor();

        p.vel = new Vec2(vx, vy);
        const speedSq = p.vel.magSq();
        if (speedSq < 1) {
            const gamma = 1 / Math.sqrt(1 - speedSq);
            p.momentum = p.vel.clone().scale(gamma * p.mass);
        } else {
            p.vel = p.vel.clone().normalize().scale(0.99);
            const gamma = 1 / Math.sqrt(1 - 0.99 * 0.99);
            p.momentum = p.vel.clone().scale(gamma * p.mass);
        }

        this.particles.push(p);
    }

    loop(timestamp) {
        const rawDt = (timestamp - this.lastTime) / 1000;
        this.lastTime = timestamp;

        const speedScale = parseFloat(document.getElementById('speedInput').value);
        const dt = Math.min(rawDt, 0.1) * speedScale;

        const collisionMode = document.querySelector('#collision-toggles .mode-btn.active').dataset.collision;
        const boundaryMode = document.querySelector('#boundary-toggles .mode-btn.active').dataset.boundary;

        if (this.running) {
            this.physics.update(this.particles, dt, collisionMode, boundaryMode);
        }

        this.renderer.render(this.particles, dt);
        this.updateStats();

        requestAnimationFrame((t) => this.loop(t));
    }

    updateStats() {
        document.getElementById('particleCount').textContent = this.particles.length;

        const speed = parseFloat(document.getElementById('speedInput').value);
        document.getElementById('simSpeed').textContent = speed + 'x';

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
