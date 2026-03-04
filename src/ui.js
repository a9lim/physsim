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
    bindToggleGroup('boundary-toggles', 'boundary', (v) => {
        sim.boundaryMode = v;
        document.getElementById('topology-group').style.display = v === 'loop' ? '' : 'none';
    });
    bindToggleGroup('topology-toggles', 'topology', (v) => { sim.topology = v; });
    bindToggleGroup('interaction-toggles', 'mode', (v) => { sim.input.mode = v; });

    // ─── Force toggles ───
    const forceToggles = [
        { id: 'gravity-toggle', prop: 'gravityEnabled' },
        { id: 'coulomb-toggle', prop: 'coulombEnabled' },
        { id: 'magnetic-toggle', prop: 'magneticEnabled' },
        { id: 'gravitomag-toggle', prop: 'gravitomagEnabled' },
        { id: 'onepn-toggle', prop: 'onePNEnabled' },
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

    // ─── Force toggle dependency helper ───
    // When a parent is off, disable the sub-toggle AND turn it off
    const setDepState = (el, prop, disabled) => {
        el.disabled = disabled;
        el.closest('.ctrl-row').classList.toggle('ctrl-disabled', disabled);
        if (disabled && el.checked) {
            el.checked = false;
            el.setAttribute('aria-checked', 'false');
            sim.physics[prop] = false;
            if (prop === 'signalDelayEnabled') sim.renderer.showSignalDelay = false;
        }
    };

    const relativityEl = document.getElementById('relativity-toggle');
    const bhEl = document.getElementById('barneshut-toggle');
    const gravEl = document.getElementById('gravity-toggle');
    const coulEl = document.getElementById('coulomb-toggle');

    // ─── Relativity → Spin-Orbit, Radiation ───
    const updateRelDeps = () => {
        const on = relativityEl.checked;
        setDepState(document.getElementById('spinorbit-toggle'), 'spinOrbitEnabled', !on);
        setDepState(document.getElementById('radiation-toggle'), 'radiationEnabled', !on);
    };
    relativityEl.addEventListener('change', updateRelDeps);
    updateRelDeps();

    // ─── Signal Delay requires Relativity + BH off ───
    const sdEl = document.getElementById('signaldelay-toggle');
    const updateSdDeps = () => {
        setDepState(sdEl, 'signalDelayEnabled', bhEl.checked || !relativityEl.checked);
    };
    bhEl.addEventListener('change', updateSdDeps);
    relativityEl.addEventListener('change', updateSdDeps);
    updateSdDeps();

    // ─── 1PN requires Gravity + Relativity ───
    const pnEl = document.getElementById('onepn-toggle');
    const updatePnDeps = () => {
        setDepState(pnEl, 'onePNEnabled', !(gravEl.checked && relativityEl.checked));
    };
    gravEl.addEventListener('change', updatePnDeps);
    relativityEl.addEventListener('change', updatePnDeps);
    updatePnDeps();

    // ─── Gravity → Gravitomagnetic ───
    const updateGravDeps = () => {
        setDepState(document.getElementById('gravitomag-toggle'), 'gravitomagEnabled', !gravEl.checked);
    };
    gravEl.addEventListener('change', updateGravDeps);
    updateGravDeps();

    // ─── Coulomb → Magnetic ───
    const updateCoulDeps = () => {
        setDepState(document.getElementById('magnetic-toggle'), 'magneticEnabled', !coulEl.checked);
    };
    coulEl.addEventListener('change', updateCoulDeps);
    updateCoulDeps();

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
    document.getElementById('accelScalingToggle')?.addEventListener('change', (e) => {
        sim.renderer.accelScaling = e.target.checked;
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
            sim.physics.update(sim.particles, PHYSICS_DT, sim.collisionMode, sim.boundaryMode, sim.topology, sim.domainW, sim.domainH, 0, 0);
            sim.renderer.render(sim.particles, 0, sim.camera, sim.photons);
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
            sim.physics.update(sim.particles, PHYSICS_DT, sim.collisionMode, sim.boundaryMode, sim.topology, sim.domainW, sim.domainH, 0, 0);
            sim.renderer.render(sim.particles, 0, sim.camera, sim.photons);
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
        energy: { title: 'Energy', body: 'E = KE + Spin\u2009KE + PE + Field + Radiated.<br>KE = \u03A3(\u03B3\u22121)m, Spin\u2009KE = \u03A3 I(\u221A(1+W\u00B2r\u00B2)\u22121)/r\u00B2, I = \u2075\u2044\u2082mr\u00B2.<br>Field = Darwin O(v\u00B2/c\u00B2) EM + gravitational corrections. Drift = numerical error.' },
        conserved: { title: 'Conserved Quantities', body: 'Momentum: |\u03A3m\u1D62w\u1D62 + p_field + p_radiated| (vector sum).<br>Angular\u2009mom: \u03A3(r\u1D62\u00D7m\u1D62w\u1D62) + \u03A3(I\u1D62W\u1D62) about COM.<br>Exactly conserved with gravity + Coulomb only, pairwise mode. Velocity-dependent forces carry momentum in unmodeled fields.' },
        spin: { title: 'Spin', body: '\u03C9 = W/\u221A(1+W\u00B2r\u00B2), caps surface velocity below c.<br>Determines \u03BC = q\u03C9r\u00B2/5 (magnetic moment) and L = 2m\u03C9r\u00B2/5 (angular momentum). Positive = CCW.' },
        gravity: { title: 'Gravity', body: 'F = m\u2081m\u2082/r\u00B2 (G = 1). Attractive between all massive particles.' },
        coulomb: { title: 'Coulomb', body: 'F = q\u2081q\u2082/r\u00B2. Like charges repel, opposites attract.' },
        magnetic: { title: 'Magnetic', body: 'Dipole: F = 3\u03BC\u2081\u03BC\u2082/r\u2074, \u03BC = q\u03C9r\u00B2/5. Aligned perpendicular dipoles repel.<br>Lorentz: F = q(v\u00D7B) via Boris rotation, B from moving charges + spinning dipoles.' },
        gravitomag: { title: 'Gravitomagnetic', body: 'Dipole: F = 3L\u2081L\u2082/r\u2074, L = 2m\u03C9r\u00B2/5. Co-rotating masses attract (GEM sign flip).<br>Linear: F = 4m(v\u00D7B\u1D4D) via Boris rotation. Frame-dragging torque aligns spins.' },
        relativity: { title: 'Relativity', body: 'State: proper velocity w = \u03B3v (unbounded). Derived: v = w/\u221A(1+w\u00B2), enforcing |v| < c.<br>When off: v = w (classical mechanics).' },
        radiation: { title: 'Radiation', body: 'Larmor power: P = 2q\u00B2a\u00B2/3. Reaction force: \u03C4\u00B7dF/dt / \u03B3\u00B3 (Landau\u2013Lifshitz), \u03C4 = 2q\u00B2/(3m).<br>Accelerating charges emit photons that carry energy and momentum. Creates orbital decay.' },
        tidal: { title: 'Disintegration', body: 'Roche limit: body fragments when tidal + centrifugal + Coulomb stress exceeds self-gravity.<br>Tidal acceleration = M\u00B7R/r\u00B3. Splits into 3 pieces.' },
        signaldelay: { title: 'Signal Delay', body: '|x_src(t_ret) \u2212 x_obs| = t_now \u2212 t_ret (c = 1).<br>Forces use source positions from the light cone, solved via Newton\u2013Raphson on history buffers. Pairwise mode only.' },
        spinorbit: { title: 'Spin\u2013Orbit', body: 'dE = \u2212\u03BC\u00B7(v\u00B7\u2207B)\u00B7dt (EM) or \u2212L\u00B7(v\u00B7\u2207B\u1D4D)\u00B7dt (GM).<br>Transfers energy between translation and spin. Also applies Stern\u2013Gerlach (F = \u03BC\u2207B) and Mathisson\u2013Papapetrou (F = \u2212L\u2207B\u1D4D) center-of-mass kicks.' },
        interaction: { title: 'Spawn Modes', body: '<b>Place</b> \u2014 spawn at rest.<br><b>Shoot</b> \u2014 drag to set velocity.<br><b>Orbit</b> \u2014 circular orbit around nearest massive body (v = \u221A(M/r)).' },
        barneshut: { title: 'Barnes\u2013Hut', body: 'On: O(N\u2009log\u2009N) quadtree approximation (\u03B8 = 0.5).<br>Off: O(N\u00B2) exact pairwise \u2014 slower but conserves momentum and angular momentum exactly.' },
        collision: { title: 'Collisions', body: '<b>Pass</b> \u2014 no interaction.<br><b>Bounce</b> \u2014 elastic + spin friction transfer.<br><b>Merge</b> \u2014 conserves mass, charge, momentum, angular momentum.' },
        boundary: { title: 'Boundaries', body: '<b>Despawn</b> \u2014 removed on leaving viewport.<br><b>Loop</b> \u2014 periodic wrapping (opens topology selector).<br><b>Bounce</b> \u2014 reflect off edges.' },
        topology: { title: 'Topology', body: '<b>Torus</b> \u2014 both axes wrap normally.<br><b>Klein bottle</b> \u2014 y-wrap mirrors x and reverses horizontal velocity.<br><b>RP\u00B2</b> \u2014 both axes wrap with perpendicular flip. Non-orientable.' },
        onepn: { title: '1PN Correction', body: 'Einstein\u2013Infeld\u2013Hoffmann O(v\u00B2/c\u00B2) correction to gravity.<br>Produces perihelion precession \u2248 6\u03C0M/a(1\u2212e\u00B2) rad/orbit. Velocity\u2013Verlet for 2nd-order accuracy. Requires Gravity + Relativity.' },
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
