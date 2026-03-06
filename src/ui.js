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
        sim.stats.resetBaseline();
        sim.selectedParticle = null;
        sim.physics._forcesInit = false;
        sim.photons = [];
        sim.totalRadiated = 0;
        sim.totalRadiatedPx = 0;
        sim.totalRadiatedPy = 0;
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
        { id: 'disintegration-toggle', prop: 'disintegrationEnabled' },
        { id: 'tidallocking-toggle', prop: 'tidalLockingEnabled' },
        { id: 'signaldelay-toggle', prop: 'signalDelayEnabled' },
        { id: 'spinorbit-toggle', prop: 'spinOrbitEnabled' },
        { id: 'barneshut-toggle', prop: 'barnesHutEnabled' },
        { id: 'yukawa-toggle', prop: 'yukawaEnabled' },
        { id: 'axion-toggle', prop: 'axionEnabled' },
        { id: 'quadradiation-toggle', prop: 'quadRadiationEnabled' },
    ];
    forceToggles.forEach(({ id, prop }) => {
        const el = document.getElementById(id);
        el.addEventListener('change', () => {
            sim.physics[prop] = el.checked;
            el.setAttribute('aria-checked', el.checked);
        });
    });

    // ─── Force toggle dependency helper ───
    // Disabling a parent also unchecks and disables its children
    const setDepState = (el, prop, disabled) => {
        el.disabled = disabled;
        const row = el.closest('.ctrl-row') || el.closest('.checkbox-label');
        if (row) row.classList.toggle('ctrl-disabled', disabled);
        if (disabled && el.checked) {
            el.checked = false;
            el.setAttribute('aria-checked', 'false');
            sim.physics[prop] = false;
        }
    };

    const relativityEl = document.getElementById('relativity-toggle');
    const bhEl = document.getElementById('barneshut-toggle');
    const gravEl = document.getElementById('gravity-toggle');
    const coulEl = document.getElementById('coulomb-toggle');

    // ─── Signal Delay requires Relativity ───
    const sdEl = document.getElementById('signaldelay-toggle');
    const updateSdDeps = () => {
        setDepState(sdEl, 'signalDelayEnabled', !relativityEl.checked);
    };
    relativityEl.addEventListener('change', updateSdDeps);
    updateSdDeps();

    // ─── 1PN requires Relativity + (Magnetic or Gravitomagnetic) ───
    const gmEl = document.getElementById('gravitomag-toggle');
    const magEl = document.getElementById('magnetic-toggle');
    const pnEl = document.getElementById('onepn-toggle');
    const updatePnDeps = () => {
        setDepState(pnEl, 'onePNEnabled', !relativityEl.checked || (!magEl.checked && !gmEl.checked));
    };
    relativityEl.addEventListener('change', updatePnDeps);
    magEl.addEventListener('change', updatePnDeps);
    gmEl.addEventListener('change', updatePnDeps);

    // ─── Spin-Orbit requires Magnetic or Gravitomagnetic ───
    const soEl = document.getElementById('spinorbit-toggle');
    const updateSoDeps = () => {
        setDepState(soEl, 'spinOrbitEnabled', !magEl.checked && !gmEl.checked);
    };
    magEl.addEventListener('change', updateSoDeps);
    gmEl.addEventListener('change', updateSoDeps);

    // ─── Gravity → Gravitomagnetic (cascades to 1PN, Spin-Orbit) ───
    const updateGravDeps = () => {
        setDepState(gmEl, 'gravitomagEnabled', !gravEl.checked);
        updatePnDeps();
        updateSoDeps();
    };
    gravEl.addEventListener('change', updateGravDeps);

    // ─── Coulomb → Magnetic (cascades to 1PN, Spin-Orbit) ───
    const updateCoulDeps = () => {
        setDepState(magEl, 'magneticEnabled', !coulEl.checked);
        updatePnDeps();
        updateSoDeps();
    };
    coulEl.addEventListener('change', updateCoulDeps);

    // ─── Radiation requires Coulomb ───
    const radEl = document.getElementById('radiation-toggle');
    const updateRadDeps = () => {
        setDepState(radEl, 'radiationEnabled', !coulEl.checked);
    };
    coulEl.addEventListener('change', updateRadDeps);

    // ─── Tidal Locking requires Gravity ───
    const tlEl = document.getElementById('tidallocking-toggle');
    const updateTlDeps = () => {
        setDepState(tlEl, 'tidalLockingEnabled', !gravEl.checked);
    };
    gravEl.addEventListener('change', updateTlDeps);

    // ─── Disintegration requires Gravity ───
    const disintEl = document.getElementById('disintegration-toggle');
    const updateDisintDeps = () => {
        setDepState(disintEl, 'disintegrationEnabled', !gravEl.checked);
    };
    gravEl.addEventListener('change', updateDisintDeps);

    // ─── Axion requires Coulomb ───
    const axionEl = document.getElementById('axion-toggle');
    const updateAxionDeps = () => {
        setDepState(axionEl, 'axionEnabled', !coulEl.checked);
    };
    coulEl.addEventListener('change', updateAxionDeps);

    // ─── Quadrupole Radiation requires Radiation ───
    const quadEl = document.getElementById('quadradiation-toggle');
    const updateQuadDeps = () => {
        setDepState(quadEl, 'quadRadiationEnabled', !radEl.checked);
    };
    radEl.addEventListener('change', updateQuadDeps);

    // ─── Initialize all dependency states ───
    updateGravDeps();
    updateCoulDeps();
    updateRadDeps();
    updateTlDeps();
    updateDisintDeps();
    updateAxionDeps();
    updateQuadDeps();

    // ─── Black Hole requires Relativity + Gravity; locks collision to Merge ───
    const bhTogEl = document.getElementById('blackhole-toggle');
    const collisionToggles = document.getElementById('collision-toggles');
    const syncBhEffects = () => {
        if (bhTogEl.checked) {
            sim.collisionMode = 'merge';
            collisionToggles.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            collisionToggles.querySelector('[data-collision="merge"]').classList.add('active');
            collisionToggles.classList.add('ctrl-disabled');
        } else {
            collisionToggles.classList.remove('ctrl-disabled');
        }
        for (const p of sim.particles) p.updateColor();
    };
    const updateBhDeps = () => {
        setDepState(bhTogEl, 'blackHoleEnabled', !relativityEl.checked || !gravEl.checked);
        syncBhEffects();
    };
    relativityEl.addEventListener('change', updateBhDeps);
    gravEl.addEventListener('change', updateBhDeps);
    updateBhDeps();
    bhTogEl.addEventListener('change', () => {
        sim.physics.blackHoleEnabled = bhTogEl.checked;
        bhTogEl.setAttribute('aria-checked', bhTogEl.checked);
        syncBhEffects();
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

    massSlider.addEventListener('input', () => { massLabel.textContent = parseFloat(massSlider.value).toFixed(2); });
    chargeSlider.addEventListener('input', () => { chargeLabel.textContent = parseFloat(chargeSlider.value).toFixed(2); });
    spinSlider.addEventListener('input', () => { spinLabel.textContent = parseFloat(spinSlider.value).toFixed(2) + 'c'; });
    frictionSlider.addEventListener('input', () => {
        sim.physics.bounceFriction = parseFloat(frictionSlider.value);
        frictionLabel.textContent = parseFloat(frictionSlider.value).toFixed(2);
    });

    // ─── Yukawa sliders ───
    const yukawaToggle = document.getElementById('yukawa-toggle');
    const yukawaSliders = document.getElementById('yukawa-sliders');
    const yukawaG2Slider = document.getElementById('yukawaG2Input');
    const yukawaG2Label = document.getElementById('yukawaG2Value');
    const yukawaMuSlider = document.getElementById('yukawaMuInput');
    const yukawaMuLabel = document.getElementById('yukawaMuValue');

    yukawaToggle.addEventListener('change', () => {
        yukawaSliders.style.display = yukawaToggle.checked ? '' : 'none';
    });
    yukawaG2Slider.addEventListener('input', () => {
        sim.physics.yukawaG2 = parseFloat(yukawaG2Slider.value);
        yukawaG2Label.textContent = parseFloat(yukawaG2Slider.value).toFixed(2);
    });
    yukawaMuSlider.addEventListener('input', () => {
        const range = parseFloat(yukawaMuSlider.value);
        sim.physics.yukawaMu = 1 / range;
        yukawaMuLabel.textContent = range.toFixed(2);
    });

    // ─── Axion sliders ───
    const axionToggle = document.getElementById('axion-toggle');
    const axionSliders = document.getElementById('axion-sliders');
    const axionGSlider = document.getElementById('axionGInput');
    const axionGLabel = document.getElementById('axionGValue');
    const axionMassSlider = document.getElementById('axionMassInput');
    const axionMassLabel = document.getElementById('axionMassValue');

    axionToggle.addEventListener('change', () => {
        axionSliders.style.display = axionToggle.checked ? '' : 'none';
    });
    axionGSlider.addEventListener('input', () => {
        sim.physics.axionG = parseFloat(axionGSlider.value);
        axionGLabel.textContent = parseFloat(axionGSlider.value).toFixed(2);
    });
    axionMassSlider.addEventListener('input', () => {
        sim.physics.axionMass = parseFloat(axionMassSlider.value);
        axionMassLabel.textContent = parseFloat(axionMassSlider.value).toFixed(2);
    });

    // ─── Expansion toggle + Hubble slider ───
    const expansionEl = document.getElementById('expansion-toggle');
    const hubbleGroup = document.getElementById('hubble-group');
    const hubbleSlider = document.getElementById('hubbleInput');
    const hubbleLabel = document.getElementById('hubbleValue');

    expansionEl.addEventListener('change', () => {
        sim.physics.expansionEnabled = expansionEl.checked;
        hubbleGroup.style.display = expansionEl.checked ? '' : 'none';
    });
    hubbleSlider.addEventListener('input', () => {
        sim.physics.hubbleParam = parseFloat(hubbleSlider.value);
        hubbleLabel.textContent = parseFloat(hubbleSlider.value).toFixed(4);
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
        { key: 'T', label: 'Toggle theme', group: 'View', action: toggleTheme },
        { key: 'S', label: 'Toggle sidebar', group: 'View', action: togglePanel },
        { key: 'Escape', label: 'Close panel', group: 'View', action: closePanel },
    ];

    if (typeof initShortcuts === 'function') {
        initShortcuts(shortcuts, { helpTitle: 'Keyboard Shortcuts' });
    }

    // ─── Info tips ───
    const infoData = {
        energy: { title: 'Energy', body: 'Total energy is broken down into five components: linear kinetic energy ($\\text{KE} = \\sum(\\gamma - 1)m$), rotational kinetic energy from spin, gravitational and electric potential energy, Darwin field corrections at $O(v^2/c^2)$, and energy carried away by radiated photons. The "Drift" line tracks cumulative numerical error so you can judge simulation accuracy.' },
        conserved: { title: 'Conserved Quantities', body: 'Total momentum ($\\sum m_i \\mathbf{w}_i$) and angular momentum ($\\sum \\mathbf{r}_i \\times m_i \\mathbf{w}_i + \\sum I_i W_i$ about the center of mass) are tracked here, including contributions stored in fields and radiated photons. These are exactly conserved when only gravity and Coulomb forces are active in pairwise mode. Velocity-dependent forces (magnetic, gravitomagnetic) carry momentum in fields that the sim does not fully model, so small drift is expected.' },
        spin: { title: 'Spin', body: 'Each particle spins as a solid sphere. The angular celerity $W$ maps to angular velocity via $\\omega = W / \\sqrt{1 + W^2 r^2}$, which naturally keeps surface speed below $c$. Spin determines the particle\'s magnetic moment ($\\mu = q\\omega r^2/5$) and angular momentum ($L = 2m\\omega r^2/5$). Positive values mean counterclockwise.' },
        gravity: { title: 'Gravity', body: 'Newtonian gravity: $F = m_1 m_2 / r^2$ (with $G = 1$). Always attractive between all massive particles. This is the foundation for orbits, galaxy structure, and tidal effects.' },
        coulomb: { title: 'Coulomb', body: 'Electrostatic force: $F = q_1 q_2 / r^2$. Like charges repel and opposite charges attract. Combined with gravity, this lets you build atom-like bound states with charged particles.' },
        magnetic: { title: 'Magnetic', body: 'Two components. Spinning charged particles create magnetic dipoles ($\\mu = q\\omega r^2/5$) that interact via $F = 3\\mu_1 \\mu_2 / r^4$. Moving charges also generate magnetic fields, producing the Lorentz force $\\mathbf{F} = q(\\mathbf{v} \\times \\mathbf{B})$, handled exactly by the Boris integrator. Requires Coulomb.' },
        gravitomag: { title: 'Gravitomagnetic', body: 'The gravitational analog of magnetism, from general relativity\'s gravitoelectromagnetic (GEM) framework. Spinning masses interact via $F = 3L_1 L_2 / r^4$ (co-rotating masses attract, unlike EM dipoles which repel). Moving masses feel a Lorentz-like force $\\mathbf{F} = 4m(\\mathbf{v} \\times \\mathbf{B}_g)$. Frame-dragging torque gradually aligns nearby spins. Requires Gravity.' },
        relativity: { title: 'Relativity', body: 'Switches the simulation between relativistic and classical mechanics. When on, the state variable is proper velocity $\\mathbf{w} = \\gamma\\mathbf{v}$ (which can grow without bound), and coordinate velocity is derived as $\\mathbf{v} = \\mathbf{w}/\\sqrt{1+w^2}$, enforcing $|v| < c$. When off, $\\mathbf{v} = \\mathbf{w}$ with no speed limit.' },
        radiation: { title: 'Radiation', body: 'Accelerating charges radiate energy (Larmor power: $P = 2q^2 a^2/3$). The Landau\u2013Lifshitz reaction force decelerates the emitter with three terms: a jerk term ($\\dot{\\mathbf{F}}/\\gamma^3$), a radiative drag ($-\\mathbf{v}F^2/m\\gamma^2$), and a force-aligned correction ($+\\mathbf{F}(\\mathbf{v}\\!\\cdot\\!\\mathbf{F})/m\\gamma^4$). Emitted photons are visible particles that carry away energy and momentum, causing orbital decay of charged particles. Requires Coulomb.' },
        disintegration: { title: 'Disintegration', body: 'Particles break apart when disruptive forces exceed their self-gravity. The sim checks tidal stretching from neighbors ($\\propto M R / r^3$), centrifugal stress from rapid spin, and Coulomb self-repulsion. When the combined outward forces win, the particle splits into 3 fragments.' },
        tidallocking: { title: 'Tidal Locking', body: 'Tidal torque drives spin toward synchronous rotation ($\\omega_{\\text{spin}} \\to \\omega_{\\text{orbit}}$). The torque is $\\tau \\propto -(M + q_1 q_2/m)^2 R^3 / r^6 \\cdot \\Delta\\omega$. The mixed coupling captures all cross-terms between gravitational and electrostatic tidal fields. Requires Gravity.' },
        signaldelay: { title: 'Signal Delay', body: 'Forces propagate at the speed of light instead of acting instantaneously. Each particle sees others at their past positions on the light cone: $|\\mathbf{x}_{\\text{src}}(t_{\\text{ret}}) - \\mathbf{x}_{\\text{obs}}| = t_{\\text{now}} - t_{\\text{ret}}$. The delayed time is solved analytically using recorded position histories. Requires Relativity.' },
        spinorbit: { title: 'Spin\u2013Orbit', body: 'Couples translational and rotational motion through field gradients. Moving through a non-uniform magnetic or gravitomagnetic field transfers energy between a particle\'s orbit and its spin. Also applies translational kicks: Stern\u2013Gerlach force ($\\mathbf{F} = \\mu\\nabla B$) for EM, and Mathisson\u2013Papapetrou force ($\\mathbf{F} = -L\\nabla B_g$) for gravity.' },
        interaction: { title: 'Spawn Modes', body: '<b>Place</b> \u2014 click to spawn a particle at rest.<br><b>Shoot</b> \u2014 click and drag to set the particle\'s initial velocity.<br><b>Orbit</b> \u2014 spawns in a circular orbit around the nearest massive body, with velocity set to $v = \\sqrt{M/r}$.' },
        barneshut: { title: 'Barnes\u2013Hut', body: 'Controls the force calculation algorithm. When on, uses an $O(N \\log N)$ quadtree approximation ($\\theta = 0.5$) that groups distant particles together, allowing hundreds of particles to run smoothly. When off, computes every pair exactly ($O(N^2)$) which is slower but conserves momentum and angular momentum to machine precision.' },
        collision: { title: 'Collisions', body: '<b>Pass</b> \u2014 particles move through each other freely.<br><b>Bounce</b> \u2014 elastic collision with spin-dependent surface friction that transfers angular momentum.<br><b>Merge</b> \u2014 overlapping particles combine into one, conserving total mass, charge, momentum, and angular momentum.' },
        boundary: { title: 'Boundaries', body: '<b>Despawn</b> \u2014 particles are removed when they leave the viewport.<br><b>Loop</b> \u2014 particles wrap around periodically, creating an unbounded space (opens the topology selector).<br><b>Bounce</b> \u2014 particles reflect elastically off the edges.' },
        topology: { title: 'Topology', body: 'Determines how the space is identified when boundaries are set to Loop.<br><b>Torus</b> \u2014 both axes wrap normally (like Pac-Man).<br><b>Klein bottle</b> \u2014 y-wrap mirrors the x-coordinate and reverses horizontal velocity. Non-orientable.<br><b>RP\u00B2</b> \u2014 both axes wrap with a perpendicular flip. Also non-orientable; the only closed 2D surface where every loop is orientation-reversing.' },
        blackhole: { title: 'Black Hole Mode', body: 'All particles become black holes with Kerr\u2013Newman horizons: $r_+ = M + \\sqrt{M^2 - a^2 - Q^2}$ where $a = J/M$ (spin) and $Q$ is charge. Spinning or charged black holes have smaller horizons and a surrounding ergosphere (dashed ring). Hawking radiation power depends on surface gravity: extremal black holes ($M^2 = a^2 + Q^2$) have zero temperature and stop radiating. Requires Relativity.' },
        onepn: { title: '1PN Correction', body: 'First post-Newtonian $O(v^2/c^2)$ corrections. For gravity: the Einstein\u2013Infeld\u2013Hoffmann (EIH) force produces perihelion precession ($\\Delta\\phi \\approx 6\\pi M / a(1-e^2)$). For electromagnetism: the Darwin correction from the Darwin Lagrangian adds velocity-dependent terms beyond the Lorentz force. Each sector activates only when its velocity-dependent force (Gravitomagnetic or Magnetic) is on. Integrated with a velocity-Verlet scheme for second-order accuracy. Requires Relativity.' },
        yukawa: { title: 'Yukawa Potential', body: 'A screened potential $V(r) = -g^2 e^{-\\mu r}/r$ that falls off exponentially beyond range $1/\\mu$. Models short-range nuclear forces (pion exchange) and any interaction mediated by a massive particle. At short range it behaves like gravity; at long range it vanishes. The coupling $g^2$ sets the strength and $\\mu$ (the mediator mass) sets the range.' },
        axion: { title: 'Axion Coupling', body: 'Models dark matter axions oscillating as a background field $a(t) = a_0 \\cos(m_a t)$, which modulates the electromagnetic coupling: $\\alpha_{\\text{eff}} = \\alpha(1 + g\\cos(m_a t))$. This makes Coulomb and magnetic forces oscillate periodically. The effect is the exact phenomenon that axion detection experiments (CASPEr, ABRACADABRA) search for. Energy is not conserved \u2014 the axion field is an external reservoir. Requires Coulomb.' },
        quadradiation: { title: 'Quadrupole Radiation', body: 'Gravitational wave emission from the mass quadrupole moment: $P = \\frac{1}{5}|\\dddot{I}_{ij}|^2$. For circular binaries this gives $P = \\frac{32}{5} \\frac{m_1^2 m_2^2(m_1+m_2)}{r^5}$, causing orbital inspiral and merger \u2014 exactly what LIGO detects. Emitted gravitons (red) carry energy and momentum away from the system. Also includes EM quadrupole radiation when Radiation is enabled. Requires Gravity.' },
        expansion: { title: 'Cosmological Expansion', body: 'Adds Hubble flow $v_H = H \\cdot r$ from the domain center, causing distant particles to separate. Bound systems (where binding energy exceeds Hubble kinetic energy) resist expansion and stay together, while unbound particles drift apart \u2014 the mechanism that creates large-scale cosmic structure. Includes Hubble drag ($v_{\\text{pec}} \\propto 1/a$) to redshift peculiar velocities, matching the physics of real cosmological N-body simulations.' },
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
