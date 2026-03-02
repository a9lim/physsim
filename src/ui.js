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

    // ─── Clear ───
    document.getElementById('clearBtn').addEventListener('click', () => {
        sim.particles = [];
        sim.initialEnergy = null;
        sim.selectedParticle = null;
        sim.physics._forcesInit = false;
        sim.camera.reset(sim.width / 2, sim.height / 2, 1);
        showToast('Simulation cleared');
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

    // ─── Force toggles ───
    const forceToggles = [
        { id: 'gravity-toggle', prop: 'gravityEnabled' },
        { id: 'coulomb-toggle', prop: 'coulombEnabled' },
        { id: 'magnetic-toggle', prop: 'magneticEnabled' },
        { id: 'gravitomag-toggle', prop: 'gravitomagEnabled' },
        { id: 'relativity-toggle', prop: 'relativityEnabled' },
        { id: 'spinorbit-toggle', prop: 'spinOrbitEnabled' },
    ];
    forceToggles.forEach(({ id, prop }) => {
        const el = document.getElementById(id);
        el.addEventListener('change', () => {
            sim.physics[prop] = el.checked;
            el.setAttribute('aria-checked', el.checked);
        });
    });

    // ─── Visual toggles ───
    document.getElementById('trailsToggle').addEventListener('change', (e) => {
        sim.renderer.trails = e.target.checked;
    });
    document.getElementById('velocityToggle').addEventListener('change', (e) => {
        sim.renderer.showVelocity = e.target.checked;
    });
    document.getElementById('forceToggle').addEventListener('change', (e) => {
        sim.renderer.showForce = e.target.checked;
    });
    document.getElementById('forceComponentsToggle').addEventListener('change', (e) => {
        sim.renderer.showForceComponents = e.target.checked;
    });

    // ─── Slider value displays ───
    const massSlider = document.getElementById('massInput');
    const massLabel = document.getElementById('massValue');
    const chargeSlider = document.getElementById('chargeInput');
    const chargeLabel = document.getElementById('chargeValue');
    const spinSlider = document.getElementById('spinInput');
    const spinLabel = document.getElementById('spinValue');

    massSlider.addEventListener('input', () => { massLabel.textContent = massSlider.value; });
    chargeSlider.addEventListener('input', () => { chargeLabel.textContent = chargeSlider.value; });
    spinSlider.addEventListener('input', () => { spinLabel.textContent = parseFloat(spinSlider.value).toFixed(2); });

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
    sim.camera.bindZoomButtons({
        zoomIn: document.getElementById('zoom-in-btn'),
        zoomOut: document.getElementById('zoom-out-btn'),
        reset: document.getElementById('zoom-reset-btn'),
        display: document.getElementById('zoom-level'),
        onReset: () => sim.camera.reset(sim.width / 2, sim.height / 2, 1),
    });

    // ─── Theme toggle ───
    const toggleTheme = () => {
        const html = document.documentElement;
        html.dataset.theme = html.dataset.theme === 'dark' ? 'light' : 'dark';
        sim.renderer.setTheme(html.dataset.theme !== 'dark');
    };
    document.getElementById('themeToggleBtn').addEventListener('click', toggleTheme);

    // ─── Keyboard shortcuts ───
    const presetKeys = ['solar', 'binary', 'galaxy', 'collision', 'magnetic'];
    const togglePause = () => {
        sim.running = !sim.running;
        pauseIcon.hidden = !sim.running;
        playIcon.hidden = sim.running;
        pauseBtn.title = sim.running ? 'Pause' : 'Resume';
    };

    const stepSim = () => {
        if (!sim.running) {
            const cam = sim.camera;
            const halfW = sim.width / (2 * cam.zoom);
            const halfH = sim.height / (2 * cam.zoom);
            const dt = 0.1 * sim.speedScale;
            sim.physics.update(sim.particles, dt, sim.collisionMode, sim.boundaryMode, halfW * 2, halfH * 2, cam.x - halfW, cam.y - halfH);
            sim.renderer.render(sim.particles, 0, cam);
            sim.updateStats();
        }
    };

    const shortcuts = [
        { key: 'Space', label: 'Pause / Play', group: 'Simulation', action: togglePause },
        { key: 'R', label: 'Reset simulation', group: 'Simulation', action: () => document.getElementById('clearBtn').click() },
        { key: '.', label: 'Step forward', group: 'Simulation', action: stepSim },
        { key: 'P', label: 'Open presets', group: 'Simulation', action: () => presetDialog.classList.add('open') },
        { key: '1', label: 'Solar System', group: 'Presets', action: () => { loadPreset('solar', sim); closePresetDialog(); } },
        { key: '2', label: 'Binary Stars', group: 'Presets', action: () => { loadPreset('binary', sim); closePresetDialog(); } },
        { key: '3', label: 'Galaxy', group: 'Presets', action: () => { loadPreset('galaxy', sim); closePresetDialog(); } },
        { key: '4', label: 'Collision', group: 'Presets', action: () => { loadPreset('collision', sim); closePresetDialog(); } },
        { key: '5', label: 'Magnetic', group: 'Presets', action: () => { loadPreset('magnetic', sim); closePresetDialog(); } },
        { key: 'V', label: 'Toggle velocity vectors', group: 'View', action: () => {
            const el = document.getElementById('velocityToggle');
            el.checked = !el.checked;
            sim.renderer.showVelocity = el.checked;
        }},
        { key: 'F', label: 'Toggle force vectors', group: 'View', action: () => {
            const el = document.getElementById('forceToggle');
            el.checked = !el.checked;
            sim.renderer.showForce = el.checked;
        }},
        { key: 'C', label: 'Toggle force components', group: 'View', action: () => {
            const el = document.getElementById('forceComponentsToggle');
            el.checked = !el.checked;
            sim.renderer.showForceComponents = el.checked;
        }},
        { key: 'T', label: 'Toggle theme', group: 'View', action: toggleTheme },
        { key: 'S', label: 'Toggle sidebar', group: 'View', action: togglePanel },
        { key: 'O', label: 'Toggle spin-orbit', group: 'Simulation', action: () => {
            const el = document.getElementById('spinorbit-toggle');
            el.checked = !el.checked;
            sim.physics.spinOrbitEnabled = el.checked;
            el.setAttribute('aria-checked', el.checked);
        }},
        { key: 'Escape', label: 'Close dialogs', group: 'View', action: closePresetDialog },
    ];

    if (typeof initShortcuts === 'function') {
        initShortcuts(shortcuts, { helpTitle: 'Keyboard Shortcuts' });
    }

    // ─── Info tips ───
    const infoData = {
        energy: { title: 'Energy Conservation', body: 'Total energy should remain constant in a closed system. Drift indicates numerical integration error.' },
        spin: { title: 'Spin', body: 'Intrinsic spin of the particle. Affects magnetic and gravitomagnetic forces. Evolves via spin-orbit coupling when enabled. Positive = counter-clockwise, negative = clockwise.' },
        gravity: { title: 'Gravity', body: 'Attractive force between all massive particles. Proportional to m\u2081m\u2082/r\u00B2. In natural units, G=1.' },
        coulomb: { title: 'Coulomb Force', body: 'Electric force between charged particles. Like charges repel, opposites attract. Proportional to q\u2081q\u2082/r\u00B2.' },
        magnetic: { title: 'Magnetic Force', body: 'Dipole-dipole force (q\u2081s\u2081)(q\u2082s\u2082)/r\u2074 plus Lorentz force from velocity-dependent magnetic fields. Moving charges create B fields that deflect other moving charges perpendicular to their velocity.' },
        gravitomag: { title: 'Gravitomagnetic Force', body: 'Dipole force -(m\u2081s\u2081)(m\u2082s\u2082)/r\u2074 plus linear gravitomagnetism from mass currents. Co-moving masses repel via velocity-dependent fields, analogous to frame-dragging.' },
        spinorbit: { title: 'Spin-Orbit Coupling', body: 'Magnetic and gravitomagnetic fields from moving sources exert torques on particle spin. Enables spin evolution: d(spin)/dt = (q/m)\u00B7B from EM fields, minus Bg from gravitomagnetic fields.' },
        relativity: { title: 'Relativity', body: 'When enabled, uses proper velocity (w = \u03B3v) and derives v = w/\u221A(1+w\u00B2), naturally enforcing the speed-of-light limit. When off, v = w (classical Newtonian mechanics).' },
        interaction: { title: 'Interaction Modes', body: '<b>Place</b> \u2014 spawn a particle at rest.<br><b>Shoot</b> \u2014 drag to set velocity (drag distance \u00D7 0.1).<br><b>Orbit</b> \u2014 spawn in circular orbit around the nearest massive body.' },
        collision: { title: 'Collision Modes', body: '<b>Pass</b> \u2014 particles pass through each other.<br><b>Bounce</b> \u2014 elastic collision with spin-friction transfer.<br><b>Merge</b> \u2014 particles combine, conserving mass, charge, and momentum.' },
        boundary: { title: 'Boundary Modes', body: '<b>Despawn</b> \u2014 particles are removed when they leave the viewport.<br><b>Loop</b> \u2014 particles wrap around to the opposite side.<br><b>Bounce</b> \u2014 particles reflect off the viewport edges.' },
    };

    if (typeof createInfoTip === 'function') {
        document.querySelectorAll('.info-trigger[data-info]').forEach(trigger => {
            const key = trigger.dataset.info;
            if (infoData[key]) {
                createInfoTip(trigger, infoData[key]);
            }
        });
    }
}
