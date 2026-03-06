import Particle from './particle.js';
import { angwToAngVel } from './relativity.js';

// Maps physics flag names to UI toggle element IDs (same order as presets.js TOGGLE_ORDER)
const TOGGLE_SYNC = [
    ['gravityEnabled', 'gravity-toggle'],
    ['coulombEnabled', 'coulomb-toggle'],
    ['relativityEnabled', 'relativity-toggle'],
    ['gravitomagEnabled', 'gravitomag-toggle'],
    ['magneticEnabled', 'magnetic-toggle'],
    ['signalDelayEnabled', 'signaldelay-toggle'],
    ['onePNEnabled', 'onepn-toggle'],
    ['blackHoleEnabled', 'blackhole-toggle'],
    ['tidalLockingEnabled', 'tidallocking-toggle'],
    ['spinOrbitEnabled', 'spinorbit-toggle'],
    ['radiationEnabled', 'radiation-toggle'],
    ['disintegrationEnabled', 'disintegration-toggle'],
    ['barnesHutEnabled', 'barneshut-toggle'],
    ['yukawaEnabled', 'yukawa-toggle'],
    ['axionEnabled', 'axion-toggle'],
    ['quadRadiationEnabled', 'quadradiation-toggle'],
    ['expansionEnabled', 'expansion-toggle'],
];

const MODE_SYNC = [
    ['collisionMode', 'collision-toggles', 'collision'],
    ['boundaryMode', 'boundary-toggles', 'boundary'],
    ['topology', 'topology-toggles', 'topology'],
];

function syncUI(sim) {
    const ph = sim.physics;
    // Sync toggle checkboxes (parent-first order triggers dependency cascades)
    for (const [prop, elId] of TOGGLE_SYNC) {
        const el = document.getElementById(elId);
        if (!el) continue;
        const want = ph[prop];
        if (el.checked !== want) {
            el.checked = want;
            el.setAttribute('aria-checked', String(want));
            el.dispatchEvent(new Event('change'));
        }
    }
    // Sync mode button groups
    for (const [prop, groupId, attr] of MODE_SYNC) {
        const group = document.getElementById(groupId);
        if (!group) continue;
        const target = group.querySelector(`[data-${attr}="${sim[prop]}"]`);
        if (target) target.click();
    }
    // Sync sliders
    const speedEl = document.getElementById('speedInput');
    if (speedEl) { speedEl.value = sim.speedScale; speedEl.dispatchEvent(new Event('input')); }
    const frictionEl = document.getElementById('frictionInput');
    if (frictionEl) { frictionEl.value = ph.bounceFriction; frictionEl.dispatchEvent(new Event('input')); }
    const hubbleEl = document.getElementById('hubbleInput');
    if (hubbleEl) { hubbleEl.value = ph.hubbleParam; hubbleEl.dispatchEvent(new Event('input')); }
    const yukawaMuEl = document.getElementById('yukawaMuInput');
    if (yukawaMuEl) { yukawaMuEl.value = 1 / ph.yukawaMu; yukawaMuEl.dispatchEvent(new Event('input')); }
    const axionGEl = document.getElementById('axionGInput');
    if (axionGEl) { axionGEl.value = ph.axionG; axionGEl.dispatchEvent(new Event('input')); }
    const axionMassEl = document.getElementById('axionMassInput');
    if (axionMassEl) { axionMassEl.value = ph.axionMass; axionMassEl.dispatchEvent(new Event('input')); }
}

export function saveState(sim) {
    const state = {
        version: 1,
        particles: sim.particles.map(p => ({
            x: p.pos.x, y: p.pos.y,
            wx: p.w.x, wy: p.w.y,
            mass: p.mass, charge: p.charge, angw: p.angw,
        })),
        toggles: {},
        settings: {
            collision: sim.collisionMode,
            boundary: sim.boundaryMode,
            topology: sim.topology,
            speed: sim.speedScale,
            friction: sim.physics.bounceFriction,
        },
        camera: { x: sim.camera.x, y: sim.camera.y, zoom: sim.camera.zoom },
    };
    const ph = sim.physics;
    for (const key of ['gravityEnabled', 'coulombEnabled', 'magneticEnabled',
        'gravitomagEnabled', 'relativityEnabled', 'barnesHutEnabled',
        'radiationEnabled', 'blackHoleEnabled', 'disintegrationEnabled',
        'tidalLockingEnabled', 'signalDelayEnabled', 'spinOrbitEnabled',
        'onePNEnabled', 'yukawaEnabled', 'axionEnabled', 'quadRadiationEnabled',
        'expansionEnabled']) {
        state.toggles[key] = ph[key];
    }
    state.yukawaMu = ph.yukawaMu;
    state.axionG = ph.axionG;
    state.axionMass = ph.axionMass;
    state.hubbleParam = ph.hubbleParam;
    return state;
}

export function loadState(state, sim) {
    if (!state || state.version !== 1) return false;

    sim.particles = [];
    sim.photons = [];
    sim.totalRadiated = 0;
    sim.totalRadiatedPx = 0;
    sim.totalRadiatedPy = 0;
    sim.selectedParticle = null;
    sim.physics._forcesInit = false;

    const ph = sim.physics;
    for (const [key, val] of Object.entries(state.toggles)) {
        if (key in ph) ph[key] = val;
    }
    if (state.yukawaMu != null) ph.yukawaMu = state.yukawaMu;
    if (state.axionG != null) ph.axionG = state.axionG;
    if (state.axionMass != null) ph.axionMass = state.axionMass;
    if (state.hubbleParam != null) ph.hubbleParam = state.hubbleParam;

    if (state.settings) {
        sim.collisionMode = state.settings.collision || 'pass';
        sim.boundaryMode = state.settings.boundary || 'despawn';
        sim.topology = state.settings.topology || 'torus';
        sim.speedScale = state.settings.speed || 100;
        if (state.settings.friction != null) ph.bounceFriction = state.settings.friction;
    }

    if (state.camera) {
        sim.camera.x = state.camera.x;
        sim.camera.y = state.camera.y;
        sim.camera.zoom = state.camera.zoom;
    }

    for (const pd of state.particles) {
        const p = new Particle(pd.x, pd.y, pd.mass, pd.charge);
        p.mass = pd.mass;
        p.charge = pd.charge;
        p.angw = pd.angw;
        p.w.x = pd.wx;
        p.w.y = pd.wy;
        p.updateColor();
        const invG = ph.relativityEnabled ? 1 / Math.sqrt(1 + p.w.magSq()) : 1;
        p.vel.x = p.w.x * invG;
        p.vel.y = p.w.y * invG;
        p.angVel = ph.relativityEnabled ? angwToAngVel(p.angw, p.radius) : p.angw;
        sim.particles.push(p);
    }
    sim.stats.resetBaseline();
    syncUI(sim);
    return true;
}

export function downloadState(sim) {
    const state = saveState(sim);
    const json = JSON.stringify(state);
    const blob = new Blob([json], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'nohair-state.json';
    a.click();
    URL.revokeObjectURL(a.href);
}

export function uploadState(sim, onComplete) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.addEventListener('change', () => {
        const file = input.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
            try {
                const state = JSON.parse(reader.result);
                if (loadState(state, sim)) {
                    showToast('State loaded');
                    if (onComplete) onComplete();
                } else {
                    showToast('Invalid state file');
                }
            } catch { showToast('Failed to parse state file'); }
        };
        reader.readAsText(file);
    });
    input.click();
}

export function quickSave(sim) {
    const state = saveState(sim);
    localStorage.setItem('nohair-quicksave', JSON.stringify(state));
    showToast('Quick saved');
}

export function quickLoad(sim, onComplete) {
    const json = localStorage.getItem('nohair-quicksave');
    if (!json) { showToast('No quick save found'); return; }
    try {
        const state = JSON.parse(json);
        if (loadState(state, sim)) {
            showToast('Quick loaded');
            if (onComplete) onComplete();
        }
    } catch { showToast('Failed to load quick save'); }
}
