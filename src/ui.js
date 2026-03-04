// ─── UI Setup ───
import { loadPreset } from './presets.js';
import { PHYSICS_DT } from './config.js';

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
        sim.stats.resetBaseline();
        sim.selectedParticle = null;
        sim.physics._forcesInit = false;
        sim.photons = [];
        sim.totalRadiated = 0;
        sim.totalRadiatedPx = 0;
        sim.totalRadiatedPy = 0;
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
        { id: 'radiation-toggle', prop: 'radiationEnabled' },
        { id: 'tidal-toggle', prop: 'tidalEnabled' },
        { id: 'signaldelay-toggle', prop: 'signalDelayEnabled' },
        { id: 'spinorbit-toggle', prop: 'spinOrbitEnabled' },
        { id: 'barneshut-toggle', prop: 'barnesHutEnabled' },
    ];
    forceToggles.forEach(({ id, prop }) => {
        const el = document.getElementById(id);
        el.addEventListener('change', () => {
            sim.physics[prop] = el.checked;
            el.setAttribute('aria-checked', el.checked);
            if (prop === 'signalDelayEnabled') {
                sim.renderer.showSignalDelay = el.checked;
            }
        });
    });

    // ─── Relativity dependency: Radiation + Signal Delay require Relativity ───
    const relativityEl = document.getElementById('relativity-toggle');
    const relDepIds = ['radiation-toggle', 'spinorbit-toggle'];
    const updateRelDeps = () => {
        const on = relativityEl.checked;
        relDepIds.forEach(id => {
            const el = document.getElementById(id);
            el.disabled = !on;
            el.closest('.ctrl-row').classList.toggle('ctrl-disabled', !on);
        });
    };
    relativityEl.addEventListener('change', updateRelDeps);
    updateRelDeps();

    // ─── Barnes-Hut dependency: Signal Delay requires pairwise mode ───
    const bhEl = document.getElementById('barneshut-toggle');
    const sdEl = document.getElementById('signaldelay-toggle');
    const updateBhDeps = () => {
        const bhOn = bhEl.checked;
        sdEl.disabled = bhOn || !relativityEl.checked;
        sdEl.closest('.ctrl-row').classList.toggle('ctrl-disabled',
            bhOn || !relativityEl.checked);
    };
    bhEl.addEventListener('change', updateBhDeps);
    relativityEl.addEventListener('change', updateBhDeps);
    updateBhDeps();

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
    document.getElementById('potentialToggle')?.addEventListener('change', (e) => {
        sim.heatmap.enabled = e.target.checked;
    });

    // ─── Slider value displays ───
    const massSlider = document.getElementById('massInput');
    const massLabel = document.getElementById('massValue');
    const chargeSlider = document.getElementById('chargeInput');
    const chargeLabel = document.getElementById('chargeValue');
    const spinSlider = document.getElementById('spinInput');
    const spinLabel = document.getElementById('spinValue');
    const frictionSlider = document.getElementById('frictionInput');
    const frictionLabel = document.getElementById('frictionValue');

    massSlider.addEventListener('input', () => { massLabel.textContent = massSlider.value; });
    chargeSlider.addEventListener('input', () => { chargeLabel.textContent = chargeSlider.value; });
    spinSlider.addEventListener('input', () => { spinLabel.textContent = parseFloat(spinSlider.value).toFixed(2) + 'c'; });
    frictionSlider.addEventListener('input', () => {
        sim.physics.bounceFriction = parseFloat(frictionSlider.value);
        frictionLabel.textContent = parseFloat(frictionSlider.value).toFixed(2);
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
            const dt = PHYSICS_DT;
            sim.physics.update(sim.particles, dt, sim.collisionMode, sim.boundaryMode, halfW * 2, halfH * 2, cam.x - halfW, cam.y - halfH);
            sim.renderer.render(sim.particles, 0, cam, sim.photons);
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
            const dt = PHYSICS_DT;
            sim.physics.update(sim.particles, dt, sim.collisionMode, sim.boundaryMode, halfW * 2, halfH * 2, cam.x - halfW, cam.y - halfH);
            sim.renderer.render(sim.particles, 0, cam, sim.photons);
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
        { key: 'Escape', label: 'Close dialogs', group: 'View', action: closePresetDialog },
    ];

    if (typeof initShortcuts === 'function') {
        initShortcuts(shortcuts, { helpTitle: 'Keyboard Shortcuts' });
    }

    // ─── Info tips ───
    const infoData = {
        energy: { title: 'Energy Conservation', body: 'Total energy = Linear KE + Spin KE + Potential + Field + Radiated. Field energy includes both EM and gravitational Darwin Lagrangian O(v\u00B2/c\u00B2) corrections. Radiated tracks cumulative energy lost to Larmor radiation. Drift indicates numerical integration error. Spin KE uses (I/r\u00B2)\u00B7(\u221A(1+S\u00B2r\u00B2)\u22121) relativistically, \u00BDI\u03C9\u00B2 classically, where I = (2/5)mr\u00B2.' },
        conserved: { title: 'Conserved Quantities', body: 'Momentum = |p_particle + p_field + p_radiated| (vector sum). Particle momentum is \u03A3(m\u1D62w\u1D62). Field momentum includes EM and gravitational Darwin terms from charged and massive particle pairs. Radiated momentum accumulates from Larmor radiation recoil. Angular momentum about the center of mass splits into orbital \u03A3(r\u1D62 \u00D7 m\u1D62w\u1D62) and spin \u03A3(I\u1D62W\u1D62) where I = (2/5)mr\u00B2 and W is angular celerity. Conserved with gravity and Coulomb only. Velocity-dependent forces (Lorentz, linear gravitomagnetism) do not obey Newton\u2019s third law between particles \u2014 in real physics, the missing momentum is carried by the field.' },
        spin: { title: 'Spin (Angular Celerity)', body: 'The angular celerity W (state variable) is the rotational analog of proper velocity w. Angular velocity \u03C9 = W/\u221A(1+W\u00B2r\u00B2) naturally caps surface velocity below c. Determines magnetic moment and angular momentum. Positive = counter-clockwise, negative = clockwise.' },
        gravity: { title: 'Gravity', body: 'Attractive force between all massive particles. Proportional to m\u2081m\u2082/r\u00B2. In natural units, G=1.' },
        coulomb: { title: 'Coulomb Force', body: 'Electric force between charged particles. Like charges repel, opposites attract. Proportional to q\u2081q\u2082/r\u00B2.' },
        magnetic: { title: 'Magnetic Force', body: 'Dipole-dipole force: aligned \u22A5-to-plane dipoles repel (3\u03BC\u2081\u03BC\u2082/r\u2074). Magnetic moment \u03BC = \u2155q\u03C9r\u00B2 (uniform charge density solid sphere). Lorentz force from velocity-dependent B fields is handled by the Boris rotation. Note: velocity-dependent forces don\u2019t conserve momentum/angular momentum without field degrees of freedom.' },
        gravitomag: { title: 'Gravitomagnetic Force', body: 'Dipole force 3L\u2081L\u2082/r\u2074: co-rotating masses attract. Angular momentum L = (2/5)m\u03C9r\u00B2. Linear gravitomagnetism (co-moving masses attract, frame-dragging) is handled by the Boris rotation. Note: velocity-dependent forces don\u2019t conserve momentum/angular momentum without field degrees of freedom.' },
        relativity: { title: 'Relativity', body: 'When enabled, uses proper velocity (w = \u03B3v) and derives v = w/\u221A(1+w\u00B2), naturally enforcing the speed-of-light limit. When off, v = w (classical Newtonian mechanics).' },
        radiation: { title: 'Larmor Radiation', body: 'Accelerating charges radiate energy via the Larmor formula P = 2q\u00B2a\u00B2/3 (natural units). Applied as Landau\u2013Lifshitz force: jerk term \u03C4\u00B7dF/dt minus Schott damping \u03C4\u00B7F\u00B2v/m, where \u03C4 = 2q\u00B2/(3m). Relativistic correction divides by \u03B3\u00B3. Creates orbital decay in charge\u2013charge systems.' },
        tidal: { title: 'Tidal Forces (Roche Limit)', body: 'Tidal forces arise from differential gravity across a body\u2019s diameter. When tidal stress exceeds self-gravity (Roche limit), the body breaks apart into fragments. The tidal acceleration scales as M\u00B7R/r\u00B3 where M is the source mass, R is the body radius, and r is the separation distance.' },
        signaldelay: { title: 'Signal Delay', body: 'Finite-speed force propagation. Forces use each source particle\u2019s past position and velocity from its history buffer, solving the light-cone equation via Newton\u2013Raphson iteration. Creates realistic lag in distant interactions. Requires Relativity and pairwise mode (incompatible with Barnes\u2013Hut).' },
        spinorbit: { title: 'Spin\u2013Orbit Coupling', body: 'Transfers energy between translational and spin KE. EM: dE = \u2212\u03BC\u00B7(v\u00B7\u2207B_z)\u00B7dt where \u03BC = \u2155q\u03C9r\u00B2 and \u2207B_z includes both radial and angular gradient terms. GM: same pattern using angular momentum L = I\u03C9 and \u2207Bg_z. Requires Relativity and the relevant force toggle.' },
        interaction: { title: 'Interaction Modes', body: '<b>Place</b> \u2014 spawn a particle at rest.<br><b>Shoot</b> \u2014 drag to set velocity (drag distance \u00D7 0.1).<br><b>Orbit</b> \u2014 spawn in circular orbit around the nearest massive body.' },
        barneshut: { title: 'Barnes-Hut Approximation', body: 'When on, uses a quadtree to approximate distant particle groups as single bodies (O(N log N)). When off, computes exact pairwise forces (O(N\u00B2)) \u2014 slower but preserves Newton\u2019s third law exactly, improving conservation of momentum and angular momentum.' },
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
