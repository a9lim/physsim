// ─── UI Setup ───
import { loadPreset } from './presets.js';

const HINT_FADE_DELAY = 5000;

export function setupUI(sim) {
    // ─── Intro screen dismiss ───
    const introScreen = document.getElementById('intro-screen');
    const introStart = document.getElementById('intro-start');
    const panel = document.getElementById('control-panel');
    const panelToggle = document.getElementById('panelToggle');

    if (introStart && introScreen) {
        introStart.addEventListener('click', () => {
            introScreen.classList.add('hidden');
            document.body.classList.add('app-ready');
            requestAnimationFrame(() => requestAnimationFrame(() => {
                panel.classList.add('open');
                panelToggle.classList.add('active');
            }));
            setTimeout(() => { introScreen.style.display = 'none'; }, 850);
            // Hint fade
            const hint = document.getElementById('hint-bar');
            if (hint) setTimeout(() => hint.classList.add('fade-out'), HINT_FADE_DELAY);
        });
    }

    // ─── Panel toggle ───
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

    if (typeof initSwipeDismiss === 'function') {
        initSwipeDismiss(panel, { onDismiss: closePanel });
    }

    // ─── Preset dialog ───
    const presetDialog = document.getElementById('preset-dialog');
    const presetBtn = document.getElementById('presetBtn');
    const presetBackdrop = presetDialog.querySelector('.preset-backdrop');

    const closePresetDialog = () => presetDialog.classList.remove('open');

    presetBtn.addEventListener('click', () => presetDialog.classList.add('open'));
    presetBackdrop.addEventListener('click', closePresetDialog);

    presetDialog.querySelectorAll('.preset-card').forEach(card => {
        card.addEventListener('click', () => {
            loadPreset(card.dataset.preset, sim);
            closePresetDialog();
        });
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closePresetDialog();
    });

    // ─── Clear ───
    document.getElementById('clearBtn').addEventListener('click', () => {
        sim.particles = [];
        sim.camera.x = sim.width / 2;
        sim.camera.y = sim.height / 2;
        sim.camera.zoom = 1;
        sim.updateZoomDisplay();
    });

    // ─── Pause / Resume ───
    const pauseBtn = document.getElementById('pauseBtn');
    const pauseIcon = document.getElementById('pauseIcon');
    const playIcon = document.getElementById('playIcon');

    pauseBtn.addEventListener('click', () => {
        sim.running = !sim.running;
        pauseIcon.hidden = !sim.running;
        playIcon.hidden = sim.running;
        pauseBtn.title = sim.running ? 'Pause' : 'Resume';
    });

    // ─── Mode toggles ───
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

    bindToggleGroup('collision-toggles', 'collision', (v) => { sim.collisionMode = v; });
    bindToggleGroup('boundary-toggles', 'boundary', (v) => { sim.boundaryMode = v; });
    bindToggleGroup('interaction-toggles', 'mode', (v) => { sim.input.mode = v; });

    // ─── Trails toggle ───
    document.getElementById('trailsToggle').addEventListener('change', (e) => {
        sim.renderer.trails = e.target.checked;
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

    sim.dom.speedInput.addEventListener('input', () => {
        const val = parseFloat(sim.dom.speedInput.value);
        sim.speedScale = val;
        document.getElementById('speedValue').textContent = val;
    });

    // ─── Step button ───
    document.getElementById('stepBtn').addEventListener('click', () => {
        if (!sim.running) {
            const cam = sim.camera;
            const halfW = sim.width / (2 * cam.zoom);
            const halfH = sim.height / (2 * cam.zoom);
            const dt = 0.1 * sim.speedScale;
            sim.physics.update(sim.particles, dt, sim.collisionMode, sim.boundaryMode, halfW * 2, halfH * 2, cam.x - halfW, cam.y - halfH);
            sim.renderer.render(sim.particles, 0, cam);
            sim.updateStats();
        }
    });

    // ─── Zoom controls ───
    document.getElementById('zoom-in-btn').addEventListener('click', () => sim.zoomBy(1.25));
    document.getElementById('zoom-out-btn').addEventListener('click', () => sim.zoomBy(1 / 1.25));
    document.getElementById('zoom-reset-btn').addEventListener('click', () => {
        sim.camera.x = sim.width / 2;
        sim.camera.y = sim.height / 2;
        sim.camera.zoom = 1;
        sim.updateZoomDisplay();
    });

    // ─── Theme toggle ───
    document.getElementById('themeToggleBtn').addEventListener('click', () => {
        const html = document.documentElement;
        html.dataset.theme = html.dataset.theme === 'dark' ? 'light' : 'dark';
        sim.renderer.setTheme(html.dataset.theme !== 'dark');
    });
}
