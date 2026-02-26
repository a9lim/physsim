import Vec2 from './src/vec2.js';
import Physics from './src/physics.js';
import Renderer from './src/renderer.js';
import InputHandler from './src/input.js';
import Particle from './src/particle.js';

class Simulation {
    constructor() {
        this.canvas = document.getElementById('simCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = window.innerWidth;
        this.height = window.innerHeight;

        this.particles = [];
        this.physics = new Physics();
        this.renderer = new Renderer(this.ctx, this.width, this.height);
        this.renderer.setTheme(true); // light mode default
        this.input = new InputHandler(this.canvas, this);
        this.renderer.input = this.input;

        this.lastTime = 0;
        this.running = true;
        this.frameCount = 0;
        this.lastFpsTime = 0;

        // Cached DOM elements — avoids per-frame getElementById/querySelector
        this.dom = {
            particleCount: document.getElementById('particleCount'),
            fpsCounter: document.getElementById('fpsCounter'),
            simSpeed: document.getElementById('simSpeed'),
            speedInput: document.getElementById('speedInput'),
        };

        // Track active modes in JS state instead of querying DOM each frame
        this.collisionMode = 'pass';
        this.boundaryMode = 'despawn';
        this.speedScale = 20;

        this.init();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.setupUI();
        this.setupHintFade();
        requestAnimationFrame((t) => this.loop(t));
    }

    resize() {
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.renderer.resize(this.width, this.height);
        this.input.updateRect();
    }

    setupHintFade() {
        const hint = document.getElementById('hint-bar');
        if (hint) {
            setTimeout(() => hint.classList.add('fade-out'), 5000);
        }
    }

    setupUI() {
        // ─── Panel toggle ───
        const panel = document.getElementById('control-panel');
        const panelToggle = document.getElementById('panelToggle');

        const closePanel = () => {
            panel.classList.remove('open');
            panelToggle.classList.remove('active');
        };
        const togglePanel = () => {
            panel.classList.toggle('open');
            panelToggle.classList.toggle('active');
        };

        panelToggle.addEventListener('click', togglePanel);
        document.getElementById('panelClose').addEventListener('click', closePanel);

        // ─── Preset dialog ───
        const presetDialog = document.getElementById('preset-dialog');
        const presetBtn = document.getElementById('presetBtn');
        const presetBackdrop = presetDialog.querySelector('.preset-backdrop');

        const closePresetDialog = () => presetDialog.classList.remove('open');

        presetBtn.addEventListener('click', () => presetDialog.classList.add('open'));
        presetBackdrop.addEventListener('click', closePresetDialog);

        presetDialog.querySelectorAll('.preset-card').forEach(card => {
            card.addEventListener('click', () => {
                this.loadPreset(card.dataset.preset);
                closePresetDialog();
            });
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closePresetDialog();
            }
        });

        // ─── Clear ───
        document.getElementById('clearBtn').addEventListener('click', () => {
            this.particles = [];
        });

        // ─── Pause / Resume ───
        const pauseBtn = document.getElementById('pauseBtn');
        const pauseIcon = document.getElementById('pauseIcon');
        const playIcon = document.getElementById('playIcon');

        pauseBtn.addEventListener('click', () => {
            this.running = !this.running;
            pauseIcon.hidden = !this.running;
            playIcon.hidden = this.running;
            pauseBtn.title = this.running ? 'Pause' : 'Resume';
        });

        // ─── Mode toggles — track state in JS ───
        const bindToggleGroup = (id, attr, setter) => {
            const group = document.getElementById(id);
            group.addEventListener('click', (e) => {
                const btn = e.target.closest('.mode-btn');
                if (!btn) return;
                group.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                setter(btn.dataset[attr]);
            });
        };

        bindToggleGroup('collision-toggles', 'collision', (v) => { this.collisionMode = v; });
        bindToggleGroup('boundary-toggles', 'boundary', (v) => { this.boundaryMode = v; });
        bindToggleGroup('interaction-toggles', 'mode', (v) => { this.input.mode = v; });

        // ─── Trails toggle ───
        document.getElementById('trailsToggle').addEventListener('change', (e) => {
            this.renderer.trails = e.target.checked;
        });

        // ─── Slider value displays ───
        const sliderConfig = [
            { id: 'massInput', display: 'massValue' },
            { id: 'chargeInput', display: 'chargeValue' },
            { id: 'spinInput', display: 'spinValue' },
        ];

        sliderConfig.forEach(({ id, display }) => {
            const slider = document.getElementById(id);
            const label = document.getElementById(display);
            slider.addEventListener('input', () => { label.textContent = slider.value; });
        });

        // Speed slider — also update cached state
        this.dom.speedInput.addEventListener('input', () => {
            const val = parseFloat(this.dom.speedInput.value);
            this.speedScale = val;
            document.getElementById('speedValue').textContent = val;
        });

        // ─── Step button ───
        document.getElementById('stepBtn').addEventListener('click', () => {
            if (!this.running) {
                const dt = 0.1 * this.speedScale;
                this.physics.update(this.particles, dt, this.collisionMode, this.boundaryMode, this.width, this.height);
                this.renderer.render(this.particles);
                this.updateStats();
            }
        });

        // ─── Theme toggle ───
        document.getElementById('themeToggleBtn').addEventListener('click', () => {
            document.body.dataset.theme = document.body.dataset.theme === 'dark' ? '' : 'dark';
            this.renderer.setTheme(document.body.dataset.theme !== 'dark');
        });
    }

    loadPreset(name) {
        this.particles = [];
        const cx = this.width / 2;
        const cy = this.height / 2;

        if (name === 'solar') {
            this.addParticle(cx, cy, 0, 0, { mass: 80, charge: 0, spin: 0 });
            for (let i = 0; i < 5; i++) {
                const dist = 100 + i * 60;
                const angle = Math.random() * Math.PI * 2;
                const speed = Math.sqrt(80 / dist);
                const cos = Math.cos(angle), sin = Math.sin(angle);
                this.addParticle(cx + cos * dist, cy + sin * dist, -sin * speed, cos * speed,
                    { mass: 0.5 + Math.random() * 1.5, charge: 0, spin: 0 });
            }
        } else if (name === 'binary') {
            const dist = 100;
            const starMass = 50;
            const speed = Math.sqrt(starMass / (2 * dist));
            this.addParticle(cx - dist, cy, 0, speed, { mass: starMass, charge: 0, spin: 10 });
            this.addParticle(cx + dist, cy, 0, -speed, { mass: starMass, charge: 0, spin: 10 });
        } else if (name === 'galaxy') {
            this.addParticle(cx, cy, 0, 0, { mass: 150, charge: 0, spin: 30 });
            for (let i = 0; i < 200; i++) {
                const dist = 150 + Math.random() * 300;
                const angle = Math.random() * Math.PI * 2;
                const speed = Math.sqrt(150 / dist);
                const cos = Math.cos(angle), sin = Math.sin(angle);
                this.addParticle(cx + cos * dist, cy + sin * dist, -sin * speed, cos * speed, {
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
            const spacing = 80;
            for (let i = -2; i <= 2; i++) {
                for (let j = -2; j <= 2; j++) {
                    this.addParticle(
                        cx + i * spacing + (Math.random() - 0.5) * 20,
                        cy + j * spacing + (Math.random() - 0.5) * 20,
                        (Math.random() - 0.5) * 0.1,
                        (Math.random() - 0.5) * 0.1,
                        { mass: 3 + Math.random() * 2, charge: 5 + Math.random() * 5, spin: 20 + Math.random() * 10 }
                    );
                }
            }
        }
    }

    addParticle(x, y, vx, vy, options = {}) {
        const p = new Particle(x, y);

        const baseMass = options.mass ?? 10;
        const massVar = baseMass * 0.2;
        p.mass = Math.max(1, baseMass + (Math.random() - 0.5) * massVar);

        const baseCharge = options.charge ?? 0;
        if (baseCharge !== 0) {
            p.charge = baseCharge + (Math.random() - 0.5) * baseCharge * 0.2;
        } else {
            p.charge = 0;
        }

        const baseSpin = options.spin ?? 0;
        if (baseSpin !== 0) {
            p.spin = baseSpin + (Math.random() - 0.5) * baseSpin * 0.2;
        } else {
            p.spin = 0;
        }

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

    loop(timestamp) {
        const rawDt = (timestamp - this.lastTime) / 1000;
        this.lastTime = timestamp;

        const dt = Math.min(rawDt, 0.1) * this.speedScale;

        if (this.running) {
            this.physics.update(this.particles, dt, this.collisionMode, this.boundaryMode, this.width, this.height);
        }

        this.renderer.render(this.particles, dt);
        this.updateStats();

        requestAnimationFrame((t) => this.loop(t));
    }

    updateStats() {
        this.dom.particleCount.textContent = this.particles.length;

        // FPS — update display only once per second
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
