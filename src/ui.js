// ─── UI Setup ───
// Wires all panel controls, toggles, presets, shortcuts, and info tips to the sim.
import { loadPreset, PRESETS, PRESET_ORDER } from './presets.js';
import { PHYSICS_DT, WORLD_SCALE } from './config.js';
import { REFERENCE } from './reference.js';

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

    // ─── Preset dropdown ───
    const presetSelect = document.getElementById('preset-select');
    presetSelect.addEventListener('change', () => {
        const key = presetSelect.value;
        if (key === 'none') {
            document.getElementById('clearBtn').click();
        } else if (key) {
            loadPreset(key, sim);
        }
    });

    // ─── Clear ───
    document.getElementById('clearBtn').addEventListener('click', () => {
        sim.particles = [];
        sim.deadParticles = [];
        sim.stats.resetBaseline();
        sim.selectedParticle = null;
        sim.physics._forcesInit = false;
        sim.photons = [];
        sim.pions = [];
        sim.totalRadiated = 0;
        sim.totalRadiatedPx = 0;
        sim.totalRadiatedPy = 0;
        if (sim.higgsField) sim.higgsField.reset();
        if (sim.axionField) sim.axionField.reset();
        sim.camera.reset(sim.domainW / 2, sim.domainH / 2, WORLD_SCALE);
        showToast('Simulation cleared');
    });

    // ─── Pause / Resume ───
    const pauseBtn = document.getElementById('pauseBtn');
    const pauseIcon = document.getElementById('pauseIcon');
    const playIcon = document.getElementById('playIcon');

    const togglePause = () => {
        sim.running = !sim.running;
        pauseIcon.hidden = !sim.running;
        playIcon.hidden = sim.running;
        pauseBtn.title = sim.running ? 'Pause' : 'Resume';
    };
    pauseBtn.addEventListener('click', togglePause);

    // ─── Antimatter toggle ───
    const antimatterBtn = document.getElementById('antimatterBtn');
    const toggleAntimatter = () => {
        sim.antimatterMode = !sim.antimatterMode;
        antimatterBtn.classList.toggle('active', sim.antimatterMode);
        antimatterBtn.title = sim.antimatterMode ? 'Antimatter Mode ON (A)' : 'Toggle Antimatter Spawn (A)';
    };
    antimatterBtn.addEventListener('click', toggleAntimatter);

    // ─── Mode toggles ───
    const collisionToggles = document.getElementById('collision-toggles');
    const boundaryToggles = document.getElementById('boundary-toggles');
    const frictionGroup = document.getElementById('friction-group');

    const updateFrictionVisibility = () => {
        frictionGroup.style.display = (sim.collisionMode === 'bounce' || sim.boundaryMode === 'bounce') ? '' : 'none';
    };

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

    bindToggleGroup('collision-toggles', 'collision', (v) => {
        sim.collisionMode = v;
        updateFrictionVisibility();
    });
    bindToggleGroup('boundary-toggles', 'boundary', (v) => {
        sim.boundaryMode = v;
        document.getElementById('topology-group').style.display = v === 'loop' ? '' : 'none';
        updateFrictionVisibility();
    });
    bindToggleGroup('topology-toggles', 'topology', (v) => { sim.topology = v; });
    bindToggleGroup('interaction-toggles', 'mode', (v) => { sim.input.mode = v; });

    // ─── Physics toggles ───
    const toggleDefs = [
        { id: 'gravity-toggle', prop: 'gravityEnabled' },
        { id: 'coulomb-toggle', prop: 'coulombEnabled' },
        { id: 'magnetic-toggle', prop: 'magneticEnabled' },
        { id: 'gravitomag-toggle', prop: 'gravitomagEnabled' },
        { id: 'onepn-toggle', prop: 'onePNEnabled' },
        { id: 'relativity-toggle', prop: 'relativityEnabled' },
        { id: 'radiation-toggle', prop: 'radiationEnabled' },
        { id: 'disintegration-toggle', prop: 'disintegrationEnabled' },
        { id: 'spinorbit-toggle', prop: 'spinOrbitEnabled' },
        { id: 'barneshut-toggle', prop: 'barnesHutEnabled' },
        { id: 'yukawa-toggle', prop: 'yukawaEnabled' },
        { id: 'axion-toggle', prop: 'axionEnabled' },
        { id: 'blackhole-toggle', prop: 'blackHoleEnabled' },
        { id: 'expansion-toggle', prop: 'expansionEnabled' },
        { id: 'higgs-toggle', prop: 'higgsEnabled' },
    ];

    // Cache elements and prop lookup
    const tEl = {};
    const propById = {};
    toggleDefs.forEach(({ id, prop }) => {
        tEl[id] = document.getElementById(id);
        propById[id] = prop;
    });

    // ─── Declarative dependency graph ───
    // Evaluated in order: parents before children so cascading disables propagate
    const DEPS = [
        ['gravitomag-toggle', () => !tEl['gravity-toggle'].checked],
        ['magnetic-toggle', () => !tEl['coulomb-toggle'].checked],
        ['radiation-toggle', () => !tEl['gravity-toggle'].checked && !tEl['coulomb-toggle'].checked && !tEl['yukawa-toggle'].checked],
        ['disintegration-toggle', () => !tEl['gravity-toggle'].checked],
        ['axion-toggle', () => !tEl['coulomb-toggle'].checked],
        ['blackhole-toggle', () => !tEl['relativity-toggle'].checked || !tEl['gravity-toggle'].checked],
        // Children of toggles that may have been disabled above
        ['onepn-toggle', () => !tEl['relativity-toggle'].checked || (!tEl['magnetic-toggle'].checked && !tEl['gravitomag-toggle'].checked && !tEl['yukawa-toggle'].checked)],
        ['spinorbit-toggle', () => !tEl['magnetic-toggle'].checked && !tEl['gravitomag-toggle'].checked],
    ];

    const setDepState = (id, disabled) => {
        const el = tEl[id];
        el.disabled = disabled;
        const row = el.closest('.ctrl-row') || el.closest('.checkbox-label');
        if (row) row.classList.toggle('ctrl-disabled', disabled);
        if (disabled && el.checked) {
            el.checked = false;
            el.setAttribute('aria-checked', 'false');
            sim.physics[propById[id]] = false;
        }
    };

    // Slider groups toggled by their parent checkbox
    const yukawaSliders = document.getElementById('yukawa-sliders');
    const axionSliders = document.getElementById('axion-sliders');
    const hubbleGroup = document.getElementById('hubble-group');
    const higgsSliders = document.getElementById('higgs-sliders');

    const updateAllDeps = () => {
        // 1. Cascade dependency graph
        for (const [id, disabledFn] of DEPS) setDepState(id, disabledFn());

        // 2. Black hole or disintegration locks collision to merge
        const bhOn = tEl['blackhole-toggle'].checked;
        const disintOn = tEl['disintegration-toggle'].checked;
        if (bhOn || disintOn) {
            sim.collisionMode = 'merge';
            collisionToggles.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            collisionToggles.querySelector('[data-collision="merge"]').classList.add('active');
            collisionToggles.classList.add('ctrl-disabled');
        } else {
            collisionToggles.classList.remove('ctrl-disabled');
        }
        // 3. Expansion locks boundary to despawn
        if (tEl['expansion-toggle'].checked) {
            sim.boundaryMode = 'despawn';
            boundaryToggles.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            boundaryToggles.querySelector('[data-boundary="despawn"]').classList.add('active');
            boundaryToggles.classList.add('ctrl-disabled');
            document.getElementById('topology-group').style.display = 'none';
        } else {
            boundaryToggles.classList.remove('ctrl-disabled');
        }

        // 4. Slider group visibility
        yukawaSliders.style.display = tEl['yukawa-toggle'].checked ? '' : 'none';
        axionSliders.style.display = tEl['axion-toggle'].checked ? '' : 'none';
        hubbleGroup.style.display = tEl['expansion-toggle'].checked ? '' : 'none';
        if (higgsSliders) higgsSliders.style.display = tEl['higgs-toggle'].checked ? '' : 'none';
        updateFrictionVisibility();
    };

    // Wire every toggle: sync physics prop + re-evaluate all deps
    toggleDefs.forEach(({ id, prop }) => {
        tEl[id].addEventListener('change', () => {
            sim.physics[prop] = tEl[id].checked;
            tEl[id].setAttribute('aria-checked', tEl[id].checked);
            // Higgs: restore masses when toggled off
            if (id === 'higgs-toggle' && !tEl[id].checked) {
                for (const p of sim.particles) { p.mass = p.baseMass; p.updateColor(); }
            }
            // Axion: reset axMod to 1 and clear field when toggled off
            if (id === 'axion-toggle' && !tEl[id].checked) {
                for (const p of sim.particles) p.axMod = 1;
                if (sim.axionField) sim.axionField.reset();
            }
            // BH toggle changes radius calculation (Kerr-Newman vs cbrt)
            if (id === 'blackhole-toggle') {
                for (const p of sim.particles) p.updateColor();
            }
            updateAllDeps();
        });
    });

    updateAllDeps();

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
    const potentialToggle = document.getElementById('potentialToggle');
    const potentialModeBar = document.getElementById('potential-mode-toggles');
    potentialToggle?.addEventListener('change', (e) => {
        sim.heatmap.enabled = e.target.checked;
        potentialModeBar.style.display = e.target.checked ? '' : 'none';
    });
    potentialModeBar?.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-potential]');
        if (!btn) return;
        potentialModeBar.querySelector('.active')?.classList.remove('active');
        btn.classList.add('active');
        sim.heatmap.mode = btn.dataset.potential;
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

    massSlider.addEventListener('input', () => { massLabel.textContent = parseFloat(massSlider.value).toFixed(2); });
    chargeSlider.addEventListener('input', () => { chargeLabel.textContent = parseFloat(chargeSlider.value).toFixed(2); });
    spinSlider.addEventListener('input', () => { spinLabel.textContent = parseFloat(spinSlider.value).toFixed(2) + 'c'; });
    frictionSlider.addEventListener('input', () => {
        sim.physics.bounceFriction = parseFloat(frictionSlider.value);
        frictionLabel.textContent = parseFloat(frictionSlider.value).toFixed(2);
    });

    // ─── Yukawa slider ───
    const yukawaMuSlider = document.getElementById('yukawaMuInput');
    const yukawaMuLabel = document.getElementById('yukawaMuValue');
    yukawaMuSlider.addEventListener('input', () => {
        const mu = parseFloat(yukawaMuSlider.value);
        sim.physics.yukawaMu = mu;
        yukawaMuLabel.textContent = mu.toFixed(2);
    });

    // ─── Axion slider ───
    const axionMassSlider = document.getElementById('axionMassInput');
    const axionMassLabel = document.getElementById('axionMassValue');
    axionMassSlider.addEventListener('input', () => {
        const m = parseFloat(axionMassSlider.value);
        sim.physics.axionMass = m;
        if (sim.axionField) sim.axionField.mass = m;
        axionMassLabel.textContent = m.toFixed(2);
    });

    // ─── Hubble slider ───
    const hubbleSlider = document.getElementById('hubbleInput');
    const hubbleLabel = document.getElementById('hubbleValue');
    hubbleSlider.addEventListener('input', () => {
        sim.physics.hubbleParam = parseFloat(hubbleSlider.value);
        hubbleLabel.textContent = parseFloat(hubbleSlider.value).toFixed(4);
    });

    // ─── Higgs mass slider ───
    const higgsMassSlider = document.getElementById('higgsMassInput');
    const higgsMassLabel = document.getElementById('higgsMassValue');
    if (higgsMassSlider) {
        higgsMassSlider.addEventListener('input', () => {
            const m = parseFloat(higgsMassSlider.value);
            sim.higgsField.mass = m;
            higgsMassLabel.textContent = m.toFixed(2);
        });
    }

    // ─── External field sliders ───
    const extGravitySlider = document.getElementById('extGravityInput');
    const extGravityLabel = document.getElementById('extGravityValue');
    const extGravityAngleGroup = document.getElementById('extGravityAngleGroup');
    const extGravityAngleSlider = document.getElementById('extGravityAngleInput');
    const extGravityAngleLabel = document.getElementById('extGravityAngleValue');
    extGravitySlider.addEventListener('input', () => {
        const v = parseFloat(extGravitySlider.value);
        sim.physics.extGravity = v;
        extGravityLabel.textContent = v.toFixed(2);
        extGravityAngleGroup.style.display = v > 0 ? '' : 'none';
    });
    extGravityAngleSlider.addEventListener('input', () => {
        const deg = parseFloat(extGravityAngleSlider.value);
        sim.physics.extGravityAngle = deg * Math.PI / 180;
        extGravityAngleLabel.textContent = deg + '°';
    });

    const extElectricSlider = document.getElementById('extElectricInput');
    const extElectricLabel = document.getElementById('extElectricValue');
    const extElectricAngleGroup = document.getElementById('extElectricAngleGroup');
    const extElectricAngleSlider = document.getElementById('extElectricAngleInput');
    const extElectricAngleLabel = document.getElementById('extElectricAngleValue');
    extElectricSlider.addEventListener('input', () => {
        const v = parseFloat(extElectricSlider.value);
        sim.physics.extElectric = v;
        extElectricLabel.textContent = v.toFixed(2);
        extElectricAngleGroup.style.display = v > 0 ? '' : 'none';
    });
    extElectricAngleSlider.addEventListener('input', () => {
        const deg = parseFloat(extElectricAngleSlider.value);
        sim.physics.extElectricAngle = deg * Math.PI / 180;
        extElectricAngleLabel.textContent = deg + '°';
    });

    const extBzSlider = document.getElementById('extBzInput');
    const extBzLabel = document.getElementById('extBzValue');
    extBzSlider.addEventListener('input', () => {
        const v = parseFloat(extBzSlider.value);
        sim.physics.extBz = v;
        extBzLabel.textContent = v.toFixed(2);
    });

    sim.dom.speedInput.addEventListener('input', () => {
        const val = parseFloat(sim.dom.speedInput.value);
        sim.speedScale = val;
        document.getElementById('speedValue').textContent = val;
    });

    // ─── Step button ───
    const stepSim = () => {
        if (!sim.running) {
            sim.physics.update(sim.particles, PHYSICS_DT, sim.collisionMode, sim.boundaryMode, sim.topology, sim.domainW, sim.domainH, 0, 0);
            sim.renderer.render(sim.particles, 0, sim.camera, sim.photons);
        }
    };
    document.getElementById('stepBtn').addEventListener('click', stepSim);

    // ─── Zoom controls ───
    sim.camera.bindZoomButtons({
        zoomIn: document.getElementById('zoom-in-btn'),
        zoomOut: document.getElementById('zoom-out-btn'),
        reset: document.getElementById('zoom-reset-btn'),
        display: document.getElementById('zoom-level'),
        onReset: () => sim.camera.reset(sim.domainW / 2, sim.domainH / 2, WORLD_SCALE),
        formatZoom: (z) => Math.round(z / WORLD_SCALE * 100) + '%',
    });

    // ─── Theme toggle ───
    const toggleTheme = () => {
        const html = document.documentElement;
        html.dataset.theme = html.dataset.theme === 'dark' ? 'light' : 'dark';
        sim.renderer.setTheme(html.dataset.theme !== 'dark');
    };
    document.getElementById('themeToggleBtn').addEventListener('click', toggleTheme);

    // ─── Keyboard shortcuts ───
    const shortcuts = [
        { key: 'Space', label: 'Pause / Play', group: 'Simulation', action: togglePause },
        { key: 'R', label: 'Reset simulation', group: 'Simulation', action: () => document.getElementById('clearBtn').click() },
        { key: '.', label: 'Step forward', group: 'Simulation', action: stepSim },
        ...PRESET_ORDER.map((key, i) => ({
            key: String(i + 1),
            label: PRESETS[key].name,
            group: 'Presets',
            action: () => { loadPreset(key, sim); document.getElementById('preset-select').value = key; },
        })),
        { key: 'V', label: 'Toggle velocity vectors', group: 'View', action: () => {
            const el = document.getElementById('velocityToggle');
            el.checked = !el.checked;
            sim.renderer.showVelocity = el.checked;
        }},
        { key: 'F', label: 'Toggle acceleration vectors', group: 'View', action: () => {
            const el = document.getElementById('forceToggle');
            el.checked = !el.checked;
            sim.renderer.showForce = el.checked;
        }},
        { key: 'C', label: 'Toggle acceleration components', group: 'View', action: () => {
            const el = document.getElementById('forceComponentsToggle');
            el.checked = !el.checked;
            sim.renderer.showForceComponents = el.checked;
        }},
        { key: 'A', label: 'Toggle antimatter spawn', group: 'Simulation', action: toggleAntimatter },
        { key: 'T', label: 'Toggle theme', group: 'View', action: toggleTheme },
        { key: 'S', label: 'Toggle sidebar', group: 'View', action: togglePanel },
        { key: 'Escape', label: 'Close panel', group: 'View', action: closePanel },
    ];

    if (typeof initShortcuts === 'function') {
        initShortcuts(shortcuts, { helpTitle: 'Keyboard Shortcuts' });
    }

    // ─── Info tips ───
    const infoData = {
        energy: { title: 'Energy', body: 'Sum of kinetic, potential, field, and radiated energy. "Drift" tracks cumulative numerical error as a percentage of initial energy.' },
        conserved: { title: 'Conserved Quantities', body: 'Total momentum and angular momentum, including particle, field, and radiated contributions. Exactly conserved with gravity + Coulomb only in pairwise mode.' },
        spin: { title: 'Spin', body: 'Angular velocity of each particle as a solid sphere ($I = \\frac{2}{5}mr^2$). Determines magnetic moment and angular momentum. Positive = counterclockwise.' },
        gravity: { title: 'Gravity', body: 'Attractive $1/r^2$ force between all masses ($F = m_1 m_2/r^2$, $G=1$). The foundation for orbits, binaries, and tidal effects.' },
        coulomb: { title: 'Coulomb', body: 'Electrostatic $1/r^2$ force ($F = q_1 q_2/r^2$). Like charges repel, opposites attract. Combine with gravity for atom-like bound states.' },
        magnetic: { title: 'Magnetic', body: 'Lorentz force on moving charges ($q\\mathbf{v}\\times\\mathbf{B}$) plus dipole interactions from spinning charges ($3\\mu_1\\mu_2/r^4$). Requires Coulomb.' },
        gravitomag: { title: 'Gravitomagnetic', body: 'GR analog of magnetism for masses. Co-rotating masses attract (opposite to EM dipoles). Includes frame-dragging torque. Requires Gravity.' },
        relativity: { title: 'Relativity', body: 'Enforces $|v| < c$ via proper velocity $\\mathbf{w} = \\gamma\\mathbf{v}$. Enables signal delay \u2014 forces propagate at lightspeed.' },
        radiation: { title: 'Radiation', body: 'Accelerating charges emit photons (Larmor); orbiting masses emit gravitational waves (quadrupole); Yukawa interactions emit pions (scalar Larmor). Causes orbital decay. Requires Gravity, Coulomb, or Yukawa.' },
        disintegration: { title: 'Disintegration', body: 'Particles fragment when tidal, centrifugal, and Coulomb stresses exceed self-gravity. Includes Roche lobe mass transfer. Requires Gravity.' },
        spinorbit: { title: 'Spin\u2013Orbit', body: 'Couples translation and rotation via field gradients: Stern\u2013Gerlach (EM) and Mathisson\u2013Papapetrou (gravity) kicks on spinning particles. Requires Magnetic or GM.' },
        interaction: { title: 'Spawn Modes', body: '<b>Place</b> \u2014 spawn at rest. <b>Shoot</b> \u2014 drag to set velocity. <b>Orbit</b> \u2014 circular orbit around nearest mass ($v = \\sqrt{M/r}$).' },
        barneshut: { title: 'Barnes\u2013Hut', body: '$O(N\\log N)$ quadtree approximation ($\\theta = 0.5$). When off, exact $O(N^2)$ pairwise gives machine-precision conservation.' },
        collision: { title: 'Collisions', body: '<b>Pass</b> \u2014 no contact. <b>Bounce</b> \u2014 Hertz elastic repulsion with friction. <b>Merge</b> \u2014 inelastic coalescence conserving mass, charge, and momentum.' },
        boundary: { title: 'Boundaries', body: '<b>Despawn</b> \u2014 removed at edges. <b>Loop</b> \u2014 periodic wrapping (opens topology selector). <b>Bounce</b> \u2014 elastic wall repulsion with friction.' },
        topology: { title: 'Topology', body: '<b>Torus</b> \u2014 normal wrapping. <b>Klein bottle</b> \u2014 y-wrap mirrors x (non-orientable). <b>RP\u00B2</b> \u2014 both axes flip (non-orientable).' },
        blackhole: { title: 'Black Hole', body: 'Kerr\u2013Newman horizons ($r_+ = M+\\sqrt{M^2-a^2-Q^2}$), ergospheres, and Hawking radiation. Extremal BHs stop radiating. Requires Relativity + Gravity.' },
        onepn: { title: '1PN Corrections', body: '$O(v^2/c^2)$ post-Newtonian terms: EIH perihelion precession, Darwin EM corrections, Bazanski cross-terms, scalar Breit (Yukawa). Requires Relativity.' },
        yukawa: { title: 'Yukawa', body: 'Screened $e^{-\\mu r}/r$ potential \u2014 gravity-like at short range, vanishes exponentially beyond $1/\\mu$. Models massive-mediator forces.' },
        axion: { title: 'Axion Field', body: 'Axion-like scalar field with $V(a)=\\frac{1}{2}m_a^2 a^2$. Scalar $aF^2$ coupling makes $\\alpha_{\\text{eff}}=\\alpha(1+ga)$ position-dependent. Source $\\propto gq^2$, gradient force $F=-gq^2\\nabla a$. Requires Coulomb.' },
        expansion: { title: 'Expansion', body: 'Hubble flow ($v_H = Hr$) from domain center. Bound systems resist expansion; unbound particles drift apart.' },
        higgs: { title: 'Higgs Field', body: 'Scalar field with Mexican hat potential. Particles acquire mass from local field value ($m = m_0|\\phi|$). High temperature restores symmetry \u2014 particles become massless.' },
        external: { title: 'External Fields', body: '<b>Gravity</b> \u2014 uniform $\\mathbf{F}=m\\mathbf{g}$. <b>Electric</b> \u2014 uniform $\\mathbf{F}=q\\mathbf{E}$. <b>Magnetic $B_z$</b> \u2014 cyclotron motion via Boris rotation.' },
        pion: { title: 'Pions', body: 'Massive Yukawa force carriers ($m_\\pi = \\mu$). Emitted via scalar Larmor radiation ($P = g^2 m^2 a^2/3$). Travel at $v < c$, experience gravitational deflection with factor $(1+v^2)$, and decay into photons.' },
        fieldExcitation: { title: 'Field Excitations', body: 'Merge collisions deposit Gaussian wave packets into active scalar fields. Higgs excitations ripple around VEV (Higgs boson analog); axion excitations ripple around vacuum $a=0$.' },
    };

    if (typeof createInfoTip === 'function') {
        document.querySelectorAll('.info-trigger[data-info]').forEach(trigger => {
            const key = trigger.dataset.info;
            if (infoData[key]) {
                createInfoTip(trigger, infoData[key]);
            }
        });
    }

    // ─── Reference overlay (Shift+click on info buttons) ───
    const refOverlay = document.getElementById('reference-overlay');
    const refTitle = document.getElementById('reference-title');
    const refBody = document.getElementById('reference-body');
    const refClose = document.getElementById('reference-close');

    const openReference = (key) => {
        const ref = REFERENCE[key];
        if (!ref) return;
        refTitle.textContent = ref.title;
        refBody.innerHTML = ref.body;
        refOverlay.hidden = false;
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(refBody, { delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
            ]});
        }
    };

    if (refOverlay) {
        document.querySelectorAll('.info-trigger[data-info]').forEach(trigger => {
            // Shift+click (desktop)
            trigger.addEventListener('click', (e) => {
                if (!e.shiftKey) return;
                e.stopPropagation();
                openReference(trigger.dataset.info);
            });
            // Long-press (mobile): 500ms touch hold opens reference
            let longPressTimer = 0;
            trigger.addEventListener('touchstart', (e) => {
                longPressTimer = setTimeout(() => {
                    e.preventDefault();
                    openReference(trigger.dataset.info);
                    longPressTimer = 0;
                }, 500);
            }, { passive: false });
            trigger.addEventListener('touchend', () => {
                if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = 0; }
            });
            trigger.addEventListener('touchmove', () => {
                if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = 0; }
            });
        });
        refClose.addEventListener('click', () => { refOverlay.hidden = true; });
        refOverlay.addEventListener('click', (e) => {
            if (e.target === refOverlay) refOverlay.hidden = true;
        });
    }
}
