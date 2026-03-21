// ─── UI Setup ───
// Wires all panel controls, toggles, presets, shortcuts, and info tips to the sim.
import { loadPreset, PRESETS, PRESET_ORDER } from './presets.js';
import { PHYSICS_DT, WORLD_SCALE, SCALAR_GRID, GPU_SCALAR_GRID, COL_MERGE, COL_BOUNCE, BOUND_DESPAWN, BOUND_BOUNCE, SPEED_OPTIONS, colFromString, boundFromString, topoFromString } from './config.js';
import { REFERENCE } from './reference.js';
import { BACKEND_CPU, BACKEND_GPU } from './backend-interface.js';
import Particle from './particle.js';

const HINT_FADE_DELAY = 5000;

// C3: Persistent GPU toggle proxy — avoids Object.create() allocation on every slider drag.
// Object.assign copies all own enumerable properties from sim.physics, then we add extras.
const _gpuToggleProxy = {};
function _buildGPUToggles(sim) {
    Object.assign(_gpuToggleProxy, sim.physics);
    _gpuToggleProxy.heatmapEnabled = sim.heatmap ? sim.heatmap.enabled : false;
    _gpuToggleProxy.heatmapMode    = sim.heatmap ? sim.heatmap.mode    : 'all';
    return _gpuToggleProxy;
}

export function setupUI(sim) {
    // ─── Intro screen dismiss ───
    const panel = document.getElementById('control-panel');
    const panelToggle = document.getElementById('panelToggle');

    _intro.init(document.getElementById('intro-screen'), document.getElementById('intro-start'), () => {
        const hint = document.getElementById('hint-bar');
        if (hint) setTimeout(() => hint.classList.add('fade-out'), HINT_FADE_DELAY);
    });

    // ─── Panel toggle (auto-opens on desktop after intro dismiss) ───
    _toolbar.initSidebar(panelToggle, panel, document.getElementById('panelClose'));

    // ─── Preset dropdown ───
    const presetSelect = document.getElementById('preset-select');
    presetSelect.addEventListener('change', () => {
        const key = presetSelect.value;
        if (key === 'none') {
            document.getElementById('clearBtn').click();
        } else if (key) {
            loadPreset(key, sim);
            sim._dirty = true;
            _haptics.trigger('medium');
        }
    });

    // ─── Clear ───
    document.getElementById('clearBtn').addEventListener('click', () => {
        sim.reset();
        sim.camera.reset(sim.domainW / 2, sim.domainH / 2, WORLD_SCALE);
        sim.stats.resetBaseline();
        sim._dirty = true;
        showToast('Simulation cleared');
        _haptics.trigger('warning');
    });

    // ─── Pause / Resume ───
    const playBtn = document.getElementById('playBtn');

    const togglePause = () => {
        sim.running = !sim.running;
        _haptics.trigger(sim.running ? 'medium' : 'light');
        _toolbar.updatePlayBtn(playBtn, sim.running);
    };
    playBtn.addEventListener('click', togglePause);
    _toolbar.updatePlayBtn(playBtn, sim.running);

    // ─── Speed Button ───
    const speedBtn = document.getElementById('speedBtn');

    const cycleSpeed = () => {
        sim.speedIndex = (sim.speedIndex + 1) % SPEED_OPTIONS.length;
        sim.speedScale = SPEED_OPTIONS[sim.speedIndex];
        _toolbar.updateSpeedBtn(speedBtn, sim.speedScale / 16);
        _haptics.trigger('selection');
    };
    const decycleSpeed = () => {
        sim.speedIndex = (sim.speedIndex - 1 + SPEED_OPTIONS.length) % SPEED_OPTIONS.length;
        sim.speedScale = SPEED_OPTIONS[sim.speedIndex];
        _toolbar.updateSpeedBtn(speedBtn, sim.speedScale / 16);
        _haptics.trigger('selection');
    };
    speedBtn.addEventListener('click', cycleSpeed);
    speedBtn.addEventListener('contextmenu', (e) => { e.preventDefault(); decycleSpeed(); });

    // ─── Mode toggles ───
    const collisionToggles = document.getElementById('collision-toggles');
    const boundaryToggles = document.getElementById('boundary-toggles');
    const frictionGroup = document.getElementById('friction-group');

    const updateFrictionVisibility = () => {
        frictionGroup.style.display = (sim.collisionMode === COL_BOUNCE || sim.boundaryMode === BOUND_BOUNCE) ? '' : 'none';
    };

    const _syncModesToGPU = () => {
        if (sim._gpuPhysics) {
            sim._gpuPhysics.boundaryMode = sim.boundaryMode;
            sim._gpuPhysics.topologyMode = sim.topology;
            sim._gpuPhysics._collisionMode = sim.collisionMode;
        }
    };
    _forms.bindModeGroup(collisionToggles, 'collision', (v) => {
        sim.collisionMode = colFromString(v);
        updateFrictionVisibility();
        _syncModesToGPU();
    });
    _forms.bindModeGroup(boundaryToggles, 'boundary', (v) => {
        sim.boundaryMode = boundFromString(v);
        document.getElementById('topology-group').style.display = v === 'loop' ? '' : 'none';
        updateFrictionVisibility();
        _syncModesToGPU();
    });
    _forms.bindModeGroup(document.getElementById('topology-toggles'), 'topology', (v) => {
        sim.topology = topoFromString(v);
        _syncModesToGPU();
    });


    // ─── Physics toggles ───
    const toggleDefs = [
        { id: 'gravity-toggle', prop: 'gravityEnabled' },
        { id: 'bosoninter-toggle', prop: 'bosonInterEnabled' },
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
        ['bosoninter-toggle', () => !tEl['barneshut-toggle'].checked || (!tEl['gravity-toggle'].checked && !tEl['coulomb-toggle'].checked)],
        ['magnetic-toggle', () => !tEl['coulomb-toggle'].checked],
        ['radiation-toggle', () => !tEl['gravity-toggle'].checked && !tEl['coulomb-toggle'].checked && !tEl['yukawa-toggle'].checked],
        ['disintegration-toggle', () => !tEl['gravity-toggle'].checked],
        ['axion-toggle', () => !tEl['coulomb-toggle'].checked && !tEl['yukawa-toggle'].checked],
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
            sim.collisionMode = COL_MERGE;
            collisionToggles.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            collisionToggles.querySelector('[data-collision="merge"]').classList.add('active');
            collisionToggles.classList.add('ctrl-disabled');
        } else {
            collisionToggles.classList.remove('ctrl-disabled');
        }
        // 3. Expansion locks boundary to despawn
        if (tEl['expansion-toggle'].checked) {
            sim.boundaryMode = BOUND_DESPAWN;
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

        // 5. Sync toggle state to GPU backend
        if (sim._gpuPhysics && sim._gpuPhysics.setToggles) {
            // C3: Use persistent proxy object instead of Object.create per call
            sim._gpuPhysics.setToggles(_buildGPUToggles(sim));
            // Sync boundary/collision/topology (live on sim, not sim.physics)
            _syncModesToGPU();
        }
    };

    // Push slider-only changes to GPU (toggles call updateAllDeps which already syncs)
    const _syncSlidersToGPU = () => {
        if (sim._gpuPhysics && sim._gpuPhysics.setToggles) {
            // C3: Use persistent proxy object instead of Object.create per call
            sim._gpuPhysics.setToggles(_buildGPUToggles(sim));
            _syncModesToGPU();
        }
    };

    // Wire every toggle: sync physics prop + re-evaluate all deps
    toggleDefs.forEach(({ id, prop }) => {
        tEl[id].addEventListener('change', () => {
            sim.physics[prop] = tEl[id].checked;
            tEl[id].setAttribute('aria-checked', tEl[id].checked);
            // A11: Lazy field initialization on first toggle-on
            if (id === 'higgs-toggle' && tEl[id].checked) sim.ensureHiggsField();
            if (id === 'axion-toggle' && tEl[id].checked) sim.ensureAxionField();
            // Higgs: restore masses when toggled off
            if (id === 'higgs-toggle' && !tEl[id].checked) {
                for (const p of sim.particles) { p.mass = p.baseMass; p.updateColor(); }
            }
            // Axion: reset axMod/yukMod to 1 and clear field when toggled off
            if (id === 'axion-toggle' && !tEl[id].checked) {
                for (const p of sim.particles) { p.axMod = 1; p.yukMod = 1; }
                if (sim.axionField) sim.axionField.reset();
            }
            // BH toggle: Kerr-Newman radii; no hair — strip antimatter
            if (id === 'blackhole-toggle') {
                if (tEl[id].checked) {
                    for (const p of sim.particles) { p.antimatter = false; p.updateColor(); }
                } else {
                    for (const p of sim.particles) p.updateColor();
                }
            }
            updateAllDeps();
            sim._dirty = true;
            _haptics.trigger('light');
        });
    });

    updateAllDeps();

    // ─── GPU backend toggle ───
    const gpuToggle = document.getElementById('gpu-toggle');
    // Enable the toggle once GPU is ready (async init in main.js)
    sim._onGPUReady = () => {
        gpuToggle.disabled = false;
        gpuToggle.checked = true;
        gpuToggle.setAttribute('aria-checked', 'true');
        const lbl = gpuToggle.closest('.checkbox-label');
        if (lbl) lbl.classList.remove('ctrl-disabled');
    };
    sim._onGPULost = () => {
        gpuToggle.disabled = true;
        gpuToggle.checked = false;
        gpuToggle.setAttribute('aria-checked', 'false');
        const lbl = gpuToggle.closest('.checkbox-label');
        if (lbl) lbl.classList.add('ctrl-disabled');
    };
    gpuToggle.addEventListener('change', async () => {
        const on = gpuToggle.checked;
        gpuToggle.setAttribute('aria-checked', String(on));
        const gpuCanvas = document.getElementById('gpuCanvas');
        if (on && sim._gpuReady) {
            sim.backend = BACKEND_GPU;
            if (gpuCanvas) gpuCanvas.style.display = '';
            // Clear stale CPU canvas underneath
            sim.ctx.clearRect(0, 0, sim.width, sim.height);
            // Sync current state to GPU (C3: persistent proxy, no Object.create)
            sim._gpuPhysics.setToggles(_buildGPUToggles(sim));
            _syncModesToGPU();
            // Re-sync all particles to GPU
            sim._gpuPhysics.reset();
            for (const p of sim.particles) {
                p._gpuIdx = sim._gpuPhysics.addParticle({
                    x: p.pos.x, y: p.pos.y,
                    vx: p.w.x, vy: p.w.y,
                    mass: p.mass, charge: p.charge,
                    angw: p.angw, antimatter: p.antimatter,
                });
            }
        } else {
            // Read back current GPU particle state before switching to CPU
            if (sim._gpuReady && sim._gpuPhysics) {
                try {
                    const gpuState = await sim._gpuPhysics.serialize(sim);
                    // Rebuild CPU particle array from GPU readback
                    sim.particles.length = 0;
                    for (const pd of gpuState.particles) {
                        const p = new Particle(pd.x, pd.y, pd.mass, pd.charge);
                        p.baseMass = pd.baseMass;
                        p.antimatter = pd.antimatter;
                        p.w.set(pd.wx, pd.wy);
                        p.angw = pd.angw;
                        // Derive coordinate velocity from proper velocity
                        const wSq = pd.wx * pd.wx + pd.wy * pd.wy;
                        const gamma = Math.sqrt(1 + wSq);
                        p.vel.set(pd.wx / gamma, pd.wy / gamma);
                        p.angVel = sim.physics.relativityEnabled
                            ? pd.angw / Math.sqrt(1 + pd.angw * pd.angw * p.radius * p.radius)
                            : pd.angw;
                        p.creationTime = sim.physics.simTime;
                        p.updateColor();
                        sim.particles.push(p);
                    }
                    // Read back scalar field data (GPU 128×128 → CPU 64×64)
                    const fieldData = await sim._gpuPhysics.readbackFieldData();
                    const ratio = GPU_SCALAR_GRID / SCALAR_GRID; // 2
                    const cpuGrid = SCALAR_GRID;
                    const _downsample = (gpuArr) => {
                        const cpu = new Float64Array(cpuGrid * cpuGrid);
                        const invBlock = 1 / (ratio * ratio);
                        for (let cy = 0; cy < cpuGrid; cy++) {
                            for (let cx = 0; cx < cpuGrid; cx++) {
                                let sum = 0;
                                for (let dy = 0; dy < ratio; dy++) {
                                    for (let dx = 0; dx < ratio; dx++) {
                                        sum += gpuArr[(cy * ratio + dy) * GPU_SCALAR_GRID + (cx * ratio + dx)];
                                    }
                                }
                                cpu[cy * cpuGrid + cx] = sum * invBlock;
                            }
                        }
                        return cpu;
                    };
                    if (fieldData.higgsField && sim.physics.higgsEnabled) {
                        const hf = sim.ensureHiggsField();
                        hf.field.set(_downsample(fieldData.higgsField));
                        hf.fieldDot.set(_downsample(fieldData.higgsFieldDot));
                    }
                    if (fieldData.axionField && sim.physics.axionEnabled) {
                        const af = sim.ensureAxionField();
                        af.field.set(_downsample(fieldData.axionField));
                        af.fieldDot.set(_downsample(fieldData.axionFieldDot));
                    }

                    sim.clearBosons();
                    sim.stats.resetBaseline();
                } catch (e) {
                    console.warn('[physsim] GPU readback failed, CPU state may be stale:', e);
                }
            }
            sim.backend = BACKEND_CPU;
            if (gpuCanvas) gpuCanvas.style.display = 'none';
        }
        const gpuInd = document.getElementById('gpu-indicator');
        if (gpuInd) gpuInd.hidden = sim.backend !== BACKEND_GPU;
        sim._dirty = true;
        _haptics.trigger('light');
    });

    // ─── Visual toggles ───
    document.getElementById('trailsToggle').addEventListener('change', (e) => {
        sim.renderer.trails = e.target.checked;
        sim._dirty = true;
        _haptics.trigger('light');
    });
    document.getElementById('velocityToggle').addEventListener('change', (e) => {
        sim.renderer.showVelocity = e.target.checked;
        sim._dirty = true;
        _haptics.trigger('light');
    });
    document.getElementById('forceToggle').addEventListener('change', (e) => {
        sim.renderer.showForce = e.target.checked;
        sim._dirty = true;
        _haptics.trigger('light');
    });
    document.getElementById('forceComponentsToggle').addEventListener('change', (e) => {
        sim.renderer.showForceComponents = e.target.checked;
        sim._dirty = true;
        _haptics.trigger('light');
    });
    const potentialToggle = document.getElementById('potentialToggle');
    const potentialModeBar = document.getElementById('potential-mode-toggles');
    potentialToggle?.addEventListener('change', (e) => {
        sim.heatmap.enabled = e.target.checked;
        potentialModeBar.style.display = e.target.checked ? '' : 'none';
        // Sync heatmap state to GPU (C3: persistent proxy, no Object.create)
        if (sim._gpuPhysics) {
            // heatmap.enabled already updated above; _buildGPUToggles reads it
            sim._gpuPhysics.setToggles(_buildGPUToggles(sim));
        }
        sim._dirty = true;
        _haptics.trigger('light');
    });
    potentialModeBar?.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-potential]');
        if (!btn) return;
        potentialModeBar.querySelector('.active')?.classList.remove('active');
        btn.classList.add('active');
        sim.heatmap.mode = btn.dataset.potential;
        // Sync heatmap mode to GPU
        if (sim._gpuPhysics) {
            sim._gpuPhysics._heatmapMode = btn.dataset.potential;
        }
        sim._dirty = true;
        _haptics.trigger('light');
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

    const _fmt2 = v => v.toFixed(2);
    _forms.bindSlider(massSlider, massLabel, () => _haptics.trigger('selection'), _fmt2);
    _forms.bindSlider(chargeSlider, chargeLabel, () => _haptics.trigger('selection'), _fmt2);
    _forms.bindSlider(spinSlider, spinLabel, () => _haptics.trigger('selection'), v => v.toFixed(2) + 'c');
    _forms.bindSlider(frictionSlider, frictionLabel, v => {
        sim.physics.bounceFriction = v;
        _syncSlidersToGPU();
        _haptics.trigger('selection');
    }, _fmt2);

    // ─── Physics parameter sliders ───
    const yukawaMuSlider = document.getElementById('yukawaMuInput');
    const yukawaMuLabel = document.getElementById('yukawaMuValue');
    _forms.bindSlider(yukawaMuSlider, yukawaMuLabel, v => {
        sim.physics.yukawaMu = v;
        _syncSlidersToGPU();
        _haptics.trigger('selection');
    }, _fmt2);

    const axionMassSlider = document.getElementById('axionMassInput');
    const axionMassLabel = document.getElementById('axionMassValue');
    _forms.bindSlider(axionMassSlider, axionMassLabel, v => {
        sim.physics.axionMass = v;
        if (sim.axionField) sim.axionField.mass = v;
        _syncSlidersToGPU();
        _haptics.trigger('selection');
    }, _fmt2);

    const hubbleSlider = document.getElementById('hubbleInput');
    const hubbleLabel = document.getElementById('hubbleValue');
    _forms.bindSlider(hubbleSlider, hubbleLabel, v => {
        sim.physics.hubbleParam = v;
        _syncSlidersToGPU();
        _haptics.trigger('selection');
    }, v => v.toFixed(4));

    const higgsMassSlider = document.getElementById('higgsMassInput');
    const higgsMassLabel = document.getElementById('higgsMassValue');
    if (higgsMassSlider) {
        _forms.bindSlider(higgsMassSlider, higgsMassLabel, v => {
            sim.physics.higgsMass = v;
            if (sim.higgsField) sim.higgsField.mass = v;
            _syncSlidersToGPU();
            _haptics.trigger('selection');
        }, _fmt2);
    }

    // ─── External field sliders ───
    const _fmtDeg = deg => deg + '°';
    const _extSlider = (prop, slider, label, angleGroup, angleSlider, angleLabel, angleProp) => {
        _forms.bindSlider(slider, label, v => {
            sim.physics[prop] = v;
            if (angleGroup) angleGroup.style.display = v !== 0 ? '' : 'none';
            _syncSlidersToGPU();
            _haptics.trigger('selection');
        }, _fmt2);
        if (angleSlider) {
            _forms.bindSlider(angleSlider, angleLabel, deg => {
                sim.physics[angleProp] = deg * Math.PI / 180;
                _syncSlidersToGPU();
                _haptics.trigger('selection');
            }, _fmtDeg);
        }
    };
    _extSlider('extGravity',
        document.getElementById('extGravityInput'), document.getElementById('extGravityValue'),
        document.getElementById('extGravityAngleGroup'),
        document.getElementById('extGravityAngleInput'), document.getElementById('extGravityAngleValue'),
        'extGravityAngle');
    _extSlider('extElectric',
        document.getElementById('extElectricInput'), document.getElementById('extElectricValue'),
        document.getElementById('extElectricAngleGroup'),
        document.getElementById('extElectricAngleInput'), document.getElementById('extElectricAngleValue'),
        'extElectricAngle');
    _forms.bindSlider(document.getElementById('extBzInput'), document.getElementById('extBzValue'), v => {
        sim.physics.extBz = v;
        _syncSlidersToGPU();
        _haptics.trigger('selection');
    }, _fmt2);

    // Speed is now controlled by the toolbar speed button (cycleSpeed/decycleSpeed above)

    // ─── Step button ───
    const stepSim = () => {
        if (!sim.running) {
            if (sim.backend === BACKEND_GPU && sim._gpuPhysics) {
                sim._gpuPhysics.update(PHYSICS_DT);
            } else {
                sim.physics.update(sim.particles, PHYSICS_DT, sim.collisionMode, sim.boundaryMode, sim.topology, sim.domainW, sim.domainH, 0, 0);
            }
            sim._dirty = true;
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
    const syncRendererTheme = () => {
        const isLight = document.documentElement.dataset.theme !== 'dark';
        sim.renderer.setTheme(isLight);
        if (sim._gpuRenderer) sim._gpuRenderer.setTheme(isLight);
        sim._dirty = true;
    };
    _toolbar.initTheme('geon-theme', syncRendererTheme);
    syncRendererTheme();
    const toggleTheme = () => {
        _toolbar.toggleTheme('geon-theme');
        syncRendererTheme();
    };
    document.getElementById('themeToggleBtn').addEventListener('click', toggleTheme);

    // ─── Keyboard shortcuts ───
    const shortcuts = [
        { key: 'Space', label: 'Pause / Play', group: 'Simulation', action: togglePause },
        { key: 'R', label: 'Reset simulation', group: 'Simulation', action: () => document.getElementById('clearBtn').click() },
        { key: '.', label: 'Step forward', group: 'Simulation', action: stepSim },
        ...PRESET_ORDER.slice(0, 9).map((key, i) => ({
            key: String(i + 1),
            label: PRESETS[key].name,
            group: 'Presets',
            action: () => { loadPreset(key, sim); document.getElementById('preset-select').value = key; sim._dirty = true; },
        })),
        { key: 'V', label: 'Toggle velocity vectors', group: 'View', action: () => {
            const el = document.getElementById('velocityToggle');
            el.checked = !el.checked;
            sim.renderer.showVelocity = el.checked;
            sim._dirty = true;
        }},
        { key: 'F', label: 'Toggle acceleration vectors', group: 'View', action: () => {
            const el = document.getElementById('forceToggle');
            el.checked = !el.checked;
            sim.renderer.showForce = el.checked;
            sim._dirty = true;
        }},
        { key: 'C', label: 'Toggle acceleration components', group: 'View', action: () => {
            const el = document.getElementById('forceComponentsToggle');
            el.checked = !el.checked;
            sim.renderer.showForceComponents = el.checked;
            sim._dirty = true;
        }},
        { key: 'T', label: 'Toggle theme', group: 'View', action: toggleTheme },
        { key: 'S', label: 'Toggle sidebar', group: 'View', action: () => _toolbar.toggleSidebar() },
        { key: 'Escape', label: 'Close panel', group: 'View', action: () => _toolbar.closeSidebar() },
    ];

    if (typeof initShortcuts === 'function') {
        initShortcuts(shortcuts, { helpTitle: 'Keyboard Shortcuts' });
    }

    // ─── Info tips ───
    const infoData = {
        energy: { title: 'Energy', body: 'Sum of kinetic, potential, field, and radiated energy. "Drift" tracks cumulative numerical error as a percentage of initial energy.' },
        conserved: { title: 'Conserved Quantities', body: 'Total momentum and angular momentum, including particle, field, and radiated contributions. Exactly conserved with gravity + Coulomb only in pairwise mode.' },
        spin: { title: 'Spin', body: 'Angular velocity of each particle as a solid sphere ($I = \\frac{2}{5}mr^2$). Determines magnetic moment and angular momentum. Positive = clockwise.' },
        gravity: { title: 'Gravity', body: 'Attractive $1/r^2$ force between all masses ($F = m_1 m_2/r^2$, $G=1$). The foundation for orbits, binaries, and tidal effects.' },
        coulomb: { title: 'Coulomb', body: 'Electrostatic $1/r^2$ force ($F = q_1 q_2/r^2$). Like charges repel, opposites attract. Combine with gravity for atom-like bound states.' },
        magnetic: { title: 'Magnetic', body: 'Lorentz force on moving charges ($q\\mathbf{v}\\times\\mathbf{B}$) plus dipole interactions from spinning charges ($3\\mu_1\\mu_2/r^4$). Requires Coulomb.' },
        gravitomag: { title: 'Gravitomagnetic', body: 'GR analog of magnetism for masses. Co-rotating masses attract (opposite to EM dipoles). Includes frame-dragging torque. Requires Gravity.' },
        relativity: { title: 'Relativity', body: 'Enforces $|v| < c$ via proper velocity $\\mathbf{w} = \\gamma\\mathbf{v}$. Enables signal delay \u2014 forces propagate at lightspeed.' },
        radiation: { title: 'Radiation', body: 'Accelerating charges emit photons (Larmor); orbiting masses emit gravitational waves (quadrupole); Yukawa interactions emit pions (scalar Larmor). Causes orbital decay. Requires Gravity, Coulomb, or Yukawa.' },
        disintegration: { title: 'Disintegration', body: 'Particles fragment when tidal, centrifugal, and Coulomb stresses exceed self-gravity. Includes Roche lobe mass transfer. Requires Gravity.' },
        spinorbit: { title: 'Spin\u2013Orbit', body: 'Couples translation and rotation via field gradients: Stern\u2013Gerlach (EM) and Mathisson\u2013Papapetrou (gravity) kicks on spinning particles. Requires Magnetic or GM.' },
        barneshut: { title: 'Barnes\u2013Hut', body: '$O(N\\log N)$ quadtree approximation ($\\theta = 0.5$). When off, exact $O(N^2)$ pairwise gives machine-precision conservation.' },
        collision: { title: 'Collisions', body: '<b>Pass</b> \u2014 no contact. <b>Bounce</b> \u2014 Hertz elastic repulsion with friction. <b>Merge</b> \u2014 inelastic coalescence conserving mass, charge, and momentum.' },
        boundary: { title: 'Boundaries', body: '<b>Despawn</b> \u2014 removed at edges. <b>Loop</b> \u2014 periodic wrapping (opens topology selector). <b>Bounce</b> \u2014 elastic wall repulsion with friction.' },
        topology: { title: 'Topology', body: '<b>Torus</b> \u2014 normal wrapping. <b>Klein bottle</b> \u2014 y-wrap mirrors x (non-orientable). <b>RP\u00B2</b> \u2014 both axes flip (non-orientable).' },
        blackhole: { title: 'Black Hole', body: 'Kerr\u2013Newman horizons ($r_+ = M+\\sqrt{M^2-a^2-Q^2}$), ergospheres, and Hawking radiation. Extremal BHs stop radiating. No hair: antimatter distinction is erased. Requires Relativity + Gravity.' },
        onepn: { title: '1PN Corrections', body: '$O(v^2/c^2)$ post-Newtonian terms: EIH perihelion precession, Darwin EM corrections, Bazanski cross-terms, scalar Breit (Yukawa). Requires Relativity.' },
        yukawa: { title: 'Yukawa', body: 'Screened $e^{-\\mu r}/r$ potential \u2014 gravity-like at short range, vanishes exponentially beyond $1/\\mu$. Models massive-mediator forces.' },
        axion: { title: 'Axion Field', body: 'Quadratic potential ($V=\\frac{1}{2}m_a^2 a^2$) with scalar $aF^2$ EM coupling and pseudoscalar Peccei\u2013Quinn Yukawa coupling. Requires Coulomb or Yukawa.' },
        expansion: { title: 'Expansion', body: 'Hubble flow ($v_H = Hr$) from domain center. Bound systems resist expansion; unbound particles drift apart.' },
        higgs: { title: 'Higgs Field', body: 'Scalar field with Mexican hat potential. Particles acquire mass from local field value ($m = m_0|\\phi|$). High temperature restores symmetry \u2014 particles become massless.' },
        external: { title: 'External Fields', body: '<b>Gravity</b> \u2014 uniform $\\mathbf{F}=m\\mathbf{g}$. <b>Electric</b> \u2014 uniform $\\mathbf{F}=q\\mathbf{E}$. <b>Magnetic $B_z$</b> \u2014 cyclotron motion via Boris rotation.' },
        pion: { title: 'Pions', body: 'Massive Yukawa force carriers ($m_\\pi = \\mu$). Emitted via scalar Larmor radiation ($P = g^2 m^2 a^2/3$). Travel at $v < c$, experience gravitational deflection with factor $(1+v^2)$, and decay into photons.' },
        fieldExcitation: { title: 'Scalar Field Dynamics', body: 'Merge collisions deposit Gaussian wave packets into active scalar fields. When gravity is on, field energy density gravitates particles and curves its own wave equation (weak-field GR).' },
        bosoninter: { title: 'Boson Interaction', body: 'Boson\u2194boson gravity and pion\u2194pion Coulomb via Barnes\u2013Hut tree walks. Requires Barnes\u2013Hut + (Gravity or Coulomb). Includes \u03C0\u207A\u03C0\u207B annihilation into photon pairs.' },
    };

    registerInfoTips(infoData);

    // ─── Reference overlay (Shift+click / long-press on info buttons) ───
    const openReference = initReferenceOverlay(
        document.getElementById('reference-overlay'),
        document.getElementById('reference-title'),
        document.getElementById('reference-body'),
        document.getElementById('reference-close'),
        REFERENCE
    );
    bindReferenceTriggers(openReference);
}
