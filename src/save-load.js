import Particle from './particle.js';
import { angwToAngVel } from './relativity.js';
import { DEFAULT_SPEED_SCALE, SPEED_OPTIONS, DEFAULT_SPEED_INDEX, MAX_PARTICLES, COL_NAMES, BOUND_NAMES, TOPO_NAMES, colFromString, boundFromString, topoFromString } from './config.js';
import { BACKEND_GPU } from './backend-interface.js';

// Maps physics flag names to UI toggle element IDs (same order as presets.js TOGGLE_ORDER)
const TOGGLE_SYNC = [
    ['gravityEnabled', 'gravity-toggle'],
    ['bosonInterEnabled', 'bosoninter-toggle'],
    ['coulombEnabled', 'coulomb-toggle'],
    ['relativityEnabled', 'relativity-toggle'],
    ['gravitomagEnabled', 'gravitomag-toggle'],
    ['magneticEnabled', 'magnetic-toggle'],
    ['onePNEnabled', 'onepn-toggle'],
    ['blackHoleEnabled', 'blackhole-toggle'],
    ['spinOrbitEnabled', 'spinorbit-toggle'],
    ['radiationEnabled', 'radiation-toggle'],
    ['disintegrationEnabled', 'disintegration-toggle'],
    ['barnesHutEnabled', 'barneshut-toggle'],
    ['yukawaEnabled', 'yukawa-toggle'],
    ['axionEnabled', 'axion-toggle'],
    ['expansionEnabled', 'expansion-toggle'],
    ['higgsEnabled', 'higgs-toggle'],
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
    // Sync speed button
    const speedBtn = document.getElementById('speedBtn');
    if (speedBtn && typeof _playback !== 'undefined') _playback.updateSpeedBtn(speedBtn, sim.speedScale);
    const frictionEl = document.getElementById('frictionInput');
    if (frictionEl) { frictionEl.value = ph.bounceFriction; frictionEl.dispatchEvent(new Event('input')); }
    const hubbleEl = document.getElementById('hubbleInput');
    if (hubbleEl) { hubbleEl.value = ph.hubbleParam; hubbleEl.dispatchEvent(new Event('input')); }
    const yukawaMuEl = document.getElementById('yukawaMuInput');
    if (yukawaMuEl) { yukawaMuEl.value = ph.yukawaMu; yukawaMuEl.dispatchEvent(new Event('input')); }
    const axionMassEl = document.getElementById('axionMassInput');
    if (axionMassEl) { axionMassEl.value = ph.axionMass; axionMassEl.dispatchEvent(new Event('input')); }
    const higgsMassEl = document.getElementById('higgsMassInput');
    if (higgsMassEl) { higgsMassEl.value = ph.higgsMass; higgsMassEl.dispatchEvent(new Event('input')); }
    const extGravEl = document.getElementById('extGravityInput');
    if (extGravEl) { extGravEl.value = ph.extGravity; extGravEl.dispatchEvent(new Event('input')); }
    const extGravAngleEl = document.getElementById('extGravityAngleInput');
    if (extGravAngleEl) { extGravAngleEl.value = (ph.extGravityAngle || 0) * 180 / Math.PI; extGravAngleEl.dispatchEvent(new Event('input')); }
    const extElecEl = document.getElementById('extElectricInput');
    if (extElecEl) { extElecEl.value = ph.extElectric; extElecEl.dispatchEvent(new Event('input')); }
    const extElecAngleEl = document.getElementById('extElectricAngleInput');
    if (extElecAngleEl) { extElecAngleEl.value = (ph.extElectricAngle || 0) * 180 / Math.PI; extElecAngleEl.dispatchEvent(new Event('input')); }
    const extBzEl = document.getElementById('extBzInput');
    if (extBzEl) { extBzEl.value = ph.extBz; extBzEl.dispatchEvent(new Event('input')); }
}

/**
 * Save simulation state. Async for GPU backend (requires buffer readback).
 * @param {Object} sim
 * @returns {Promise<Object>|Object} State object (async for GPU, sync for CPU)
 */
export async function saveState(sim) {
    if (sim.backend === BACKEND_GPU && sim._gpuPhysics) {
        return await sim._gpuPhysics.serialize(sim);
    }
    return _cpuSaveState(sim);
}

function _cpuSaveState(sim) {
    const state = {
        version: 1,
        particles: sim.particles.map(p => ({
            x: p.pos.x, y: p.pos.y,
            wx: p.w.x, wy: p.w.y,
            mass: p.mass, baseMass: p.baseMass, charge: p.charge, angw: p.angw, antimatter: p.antimatter,
        })),
        toggles: {},
        settings: {
            collision: COL_NAMES[sim.collisionMode],
            boundary: BOUND_NAMES[sim.boundaryMode],
            topology: TOPO_NAMES[sim.topology],
            speed: sim.speedScale,
            friction: sim.physics.bounceFriction,
        },
        camera: { x: sim.camera.x, y: sim.camera.y, zoom: sim.camera.zoom },
    };
    const ph = sim.physics;
    for (const key of ['gravityEnabled', 'bosonInterEnabled',
        'coulombEnabled', 'magneticEnabled',
        'gravitomagEnabled', 'relativityEnabled', 'barnesHutEnabled',
        'radiationEnabled', 'blackHoleEnabled', 'disintegrationEnabled',
        'spinOrbitEnabled',
        'onePNEnabled', 'yukawaEnabled', 'axionEnabled',
        'expansionEnabled', 'higgsEnabled']) {
        state.toggles[key] = ph[key];
    }
    state.yukawaMu = ph.yukawaMu;
    state.axionMass = ph.axionMass;
    state.hubbleParam = ph.hubbleParam;
    state.higgsMass = ph.higgsMass;
    state.extGravity = ph.extGravity;
    state.extGravityAngle = ph.extGravityAngle;
    state.extElectric = ph.extElectric;
    state.extElectricAngle = ph.extElectricAngle;
    state.extBz = ph.extBz;
    return state;
}

export function loadState(state, sim) {
    if (!state || state.version !== 1) return false;

    if (sim.backend === BACKEND_GPU && sim._gpuPhysics) {
        // deserialize() calls reset() internally, then uploads particles to GPU
        const ok = sim._gpuPhysics.deserialize(state, sim);
        if (ok) {
            // Clear CPU-side state (GPU is authoritative)
            sim.particles.length = 0;
            sim.deadParticles.length = 0;
            sim.clearBosons();
            sim.selectedParticle = null;
            sim.totalRadiated = 0;
            sim.totalRadiatedPx = 0;
            sim.totalRadiatedPy = 0;

            // Restore toggles to CPU physics (for UI sync and slider state)
            const ph = sim.physics;
            if (state.toggles) {
                for (const [key, val] of Object.entries(state.toggles)) {
                    if (key in ph) ph[key] = val;
                }
            }
            if (state.yukawaMu != null) ph.yukawaMu = state.yukawaMu;
            if (state.axionMass != null) ph.axionMass = state.axionMass;
            if (state.hubbleParam != null) ph.hubbleParam = state.hubbleParam;
            if (state.higgsMass !== undefined) ph.higgsMass = state.higgsMass;
            if (state.extGravity !== undefined) ph.extGravity = state.extGravity;
            if (state.extGravityAngle !== undefined) ph.extGravityAngle = state.extGravityAngle;
            if (state.extElectric !== undefined) ph.extElectric = state.extElectric;
            if (state.extElectricAngle !== undefined) ph.extElectricAngle = state.extElectricAngle;
            if (state.extBz !== undefined) ph.extBz = state.extBz;
            if (state.settings && state.settings.friction != null) ph.bounceFriction = state.settings.friction;

            _restoreSettings(state, sim);

            // Sync GPU toggles/modes from restored CPU physics state
            const gpuToggles = Object.create(ph);
            gpuToggles.heatmapEnabled = sim.heatmap && sim.heatmap.enabled;
            sim._gpuPhysics.setToggles(gpuToggles);
            sim._gpuPhysics.boundaryMode = sim.boundaryMode;
            sim._gpuPhysics.topologyMode = sim.topology;
            sim._gpuPhysics._collisionMode = sim.collisionMode;

            sim.stats.resetBaseline();
            syncUI(sim);
        }
        return ok;
    }
    return _cpuLoadState(state, sim);
}

function _restoreSettings(state, sim) {
    if (state.settings) {
        // Legacy compat: 'repel' was renamed to 'bounce' in an earlier version
        const col = state.settings.collision === 'repel' ? 'bounce' : (state.settings.collision || 'pass');
        sim.collisionMode = colFromString(col);
        sim.boundaryMode = boundFromString(state.settings.boundary);
        sim.topology = topoFromString(state.settings.topology);
        const loadedSpeed = state.settings.speed ?? DEFAULT_SPEED_SCALE;
        let bestIdx = DEFAULT_SPEED_INDEX;
        let bestDist = Infinity;
        for (let i = 0; i < SPEED_OPTIONS.length; i++) {
            const d = Math.abs(SPEED_OPTIONS[i] - loadedSpeed);
            if (d < bestDist) { bestDist = d; bestIdx = i; }
        }
        sim.speedIndex = bestIdx;
        sim.speedScale = SPEED_OPTIONS[bestIdx];
    }
    if (state.camera) {
        sim.camera.x = state.camera.x;
        sim.camera.y = state.camera.y;
        sim.camera.zoom = state.camera.zoom;
    }
}

function _cpuLoadState(state, sim) {
    sim.reset();

    const ph = sim.physics;
    for (const [key, val] of Object.entries(state.toggles)) {
        if (key in ph) ph[key] = val;
    }
    if (state.yukawaMu != null) ph.yukawaMu = state.yukawaMu;
    if (state.axionMass != null) ph.axionMass = state.axionMass;
    if (state.hubbleParam != null) ph.hubbleParam = state.hubbleParam;
    if (state.higgsMass !== undefined) ph.higgsMass = state.higgsMass;
    if (state.extGravity !== undefined) ph.extGravity = state.extGravity;
    if (state.extGravityAngle !== undefined) ph.extGravityAngle = state.extGravityAngle;
    if (state.extElectric !== undefined) ph.extElectric = state.extElectric;
    if (state.extElectricAngle !== undefined) ph.extElectricAngle = state.extElectricAngle;
    if (state.extBz !== undefined) ph.extBz = state.extBz;

    _restoreSettings(state, sim);
    if (state.settings && state.settings.friction != null) ph.bounceFriction = state.settings.friction;

    for (const pd of state.particles) {
        if (sim.particles.length >= MAX_PARTICLES) break;
        const p = new Particle(pd.x, pd.y, pd.mass, pd.charge);
        p.baseMass = pd.baseMass ?? pd.mass;
        p.angw = pd.angw;
        p.antimatter = pd.antimatter || false;
        p.creationTime = -Infinity; // loaded particles treated as always existing
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

export async function downloadState(sim) {
    const state = await saveState(sim);
    const json = JSON.stringify(state);
    const blob = new Blob([json], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'nohair-state.json';
    a.click();
    // A12: Defer revoke to next frame — immediate revoke may fire before download starts
    requestAnimationFrame(() => URL.revokeObjectURL(a.href));
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

export async function quickSave(sim) {
    const state = await saveState(sim);
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
