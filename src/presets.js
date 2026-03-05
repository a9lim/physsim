// ─── Preset Definitions ───
// Each preset configures toggles, engine settings, visuals, and spawns particles.
// Designed to teach one physics concept by enabling only the relevant forces.
import { WORLD_SCALE, SOFTENING_SQ } from './config.js';

// Circular orbit velocity accounting for Plummer softening: v = sqrt(coupling * r / (m * (r² + ε²)))
const _vCirc = (coupling, r, m) => Math.sqrt(Math.abs(coupling) * r / (m * (r * r + SOFTENING_SQ)));

// Gravitational circular orbit velocity (softened)
const _vGrav = (M, r) => Math.sqrt(M * r / (r * r + SOFTENING_SQ));

export const PRESETS = {
    kepler: {
        name: 'Kepler Orbits',
        desc: 'Classical gravity — Keplerian motion and conservation laws',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: false,
            relativity: false, onepn: false, signaldelay: false, blackhole: false,
            radiation: false, tidallocking: false, spinorbit: false, tidal: false, barneshut: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 100 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            sim.addParticle(cx, cy, 0, 0, { mass: 5, charge: 0, spin: 0 });
            const radii = [8, 13, 18, 23, 28];
            for (let i = 0; i < radii.length; i++) {
                const r = radii[i];
                const angle = Math.random() * Math.PI * 2;
                const v = _vGrav(5, r);
                const cos = Math.cos(angle), sin = Math.sin(angle);
                sim.addParticle(cx + cos * r, cy + sin * r, -sin * v, cos * v,
                    { mass: 0.3 + Math.random() * 0.7, charge: 0, spin: 0 });
            }
        },
    },

    precession: {
        name: 'Precession',
        desc: 'Relativistic perihelion advance — watch the rosette orbit',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: true,
            relativity: true, onepn: true, signaldelay: false, blackhole: false,
            radiation: false, tidallocking: false, spinorbit: false, tidal: false, barneshut: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 100 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: true, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            const starM = 5;
            sim.addParticle(cx, cy, 0, 0, { mass: starM, charge: 0, spin: 0 });
            // Eccentric orbit: start at perihelion with ~1.3x circular velocity
            const rPeri = 10;
            const vCirc = _vGrav(starM, rPeri);
            const vPeri = vCirc * 1.3;
            sim.addParticle(cx + rPeri, cy, 0, vPeri, { mass: 0.5, charge: 0, spin: 0 });
        },
    },

    atom: {
        name: 'Atom',
        desc: 'Coulomb binding — electromagnetic atomic structure',
        toggles: {
            gravity: false, coulomb: true, magnetic: true, gravitomag: false,
            relativity: true, onepn: false, signaldelay: false, blackhole: false,
            radiation: false, tidallocking: false, spinorbit: true, tidal: false, barneshut: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 50, interaction: 'orbit' },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            const nucQ = 2;
            sim.addParticle(cx, cy, 0, 0, { mass: 5, charge: nucQ, spin: 0 });
            // 3 electrons in circular Coulomb orbits
            const eM = 0.5, eQ = -1;
            const radii = [10, 14, 19];
            for (let i = 0; i < radii.length; i++) {
                const r = radii[i];
                const angle = (2 * Math.PI * i) / 3;
                const v = _vCirc(nucQ * eQ, r, eM);
                const cos = Math.cos(angle), sin = Math.sin(angle);
                sim.addParticle(cx + cos * r, cy + sin * r, -sin * v, cos * v,
                    { mass: eM, charge: eQ, spin: 0.3 });
            }
        },
    },

    bremsstrahlung: {
        name: 'Bremsstrahlung',
        desc: 'Radiation from accelerating charges — watch the photons',
        toggles: {
            gravity: false, coulomb: true, magnetic: true, gravitomag: false,
            relativity: true, onepn: false, signaldelay: false, blackhole: false,
            radiation: true, tidallocking: false, spinorbit: false, tidal: false, barneshut: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 50, interaction: 'shoot' },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            // Heavy target at center
            sim.addParticle(cx, cy, 0, 0, { mass: 5, charge: 3, spin: 0 });
            // Projectile on near-miss trajectory
            sim.addParticle(cx - 25, cy + 5, 0.5, 0, { mass: 0.2, charge: -2, spin: 0 });
        },
    },

    tidallock: {
        name: 'Tidal Lock',
        desc: 'Tidal friction synchronizes spin to orbit',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: false,
            relativity: false, onepn: false, signaldelay: false, blackhole: false,
            radiation: false, tidallocking: true, spinorbit: false, tidal: false, barneshut: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 100 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            const planetM = 20;
            sim.addParticle(cx, cy, 0, 0, { mass: planetM, charge: 0, spin: 0 });
            // Moon with fast spin — watch it lock
            const r = 18;
            const v = _vGrav(planetM, r);
            sim.addParticle(cx + r, cy, 0, v, { mass: 2, charge: 0, spin: 0.8 });
        },
    },

    hawking: {
        name: 'Hawking Evaporation',
        desc: 'Small black holes radiate and evaporate',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: true,
            relativity: true, onepn: false, signaldelay: false, blackhole: true,
            radiation: false, tidallocking: false, spinorbit: false, tidal: false, barneshut: false,
        },
        settings: { collision: 'merge', boundary: 'despawn', speed: 150 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            // Small BHs that will visibly evaporate (P ∝ 1/M²)
            const masses = [0.3, 0.4, 0.5, 0.65, 0.8];
            for (let i = 0; i < masses.length; i++) {
                const angle = (2 * Math.PI * i) / masses.length;
                const r = 12 + Math.random() * 8;
                sim.addParticle(
                    cx + Math.cos(angle) * r, cy + Math.sin(angle) * r,
                    (Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.1,
                    { mass: masses[i], charge: 0, spin: 0 }
                );
            }
        },
    },

    pulsars: {
        name: 'Binary Pulsars',
        desc: 'Frame-dragging, gravitomagnetism, and signal delay',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: true,
            relativity: true, onepn: true, signaldelay: true, blackhole: false,
            radiation: false, tidallocking: false, spinorbit: false, tidal: false, barneshut: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 50 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            const starM = 20;
            const dist = 15;
            const v = _vGrav(starM, 2 * dist) * 0.9;
            sim.addParticle(cx - dist, cy, 0, v, { mass: starM, charge: 0, spin: 0.9 });
            sim.addParticle(cx + dist, cy, 0, -v, { mass: starM, charge: 0, spin: 0.9 });
        },
    },

    magnetic: {
        name: 'Magnetic Dipoles',
        desc: 'Dipole interactions, Lorentz force, and spin-orbit coupling',
        toggles: {
            gravity: false, coulomb: true, magnetic: true, gravitomag: false,
            relativity: true, onepn: false, signaldelay: false, blackhole: false,
            radiation: false, tidallocking: false, spinorbit: true, tidal: false, barneshut: false,
        },
        settings: { collision: 'bounce', boundary: 'loop', topology: 'torus', speed: 100, friction: 0.4 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            const spacing = 10;
            for (let i = -1; i <= 2; i++) {
                for (let j = -1; j <= 2; j++) {
                    sim.addParticle(
                        cx + i * spacing + (Math.random() - 0.5) * 3,
                        cy + j * spacing + (Math.random() - 0.5) * 3,
                        (Math.random() - 0.5) * 0.08,
                        (Math.random() - 0.5) * 0.08,
                        { mass: 2, charge: 3, spin: 0.8 }
                    );
                }
            }
        },
    },

    galaxy: {
        name: 'Galaxy',
        desc: 'Large-scale gravitational dynamics and accretion',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: true,
            relativity: false, onepn: false, signaldelay: false, blackhole: false,
            radiation: false, tidallocking: false, spinorbit: false, tidal: false, barneshut: true,
        },
        settings: { collision: 'merge', boundary: 'despawn', speed: 150 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            const coreM = 50;
            sim.addParticle(cx, cy, 0, 0, { mass: coreM, charge: 0, spin: 0.8 });
            for (let i = 0; i < 150; i++) {
                const r = 5 + Math.random() * 25;
                const angle = Math.random() * Math.PI * 2;
                const v = _vGrav(coreM, r);
                const cos = Math.cos(angle), sin = Math.sin(angle);
                const m = 0.05 + Math.random() * 0.15;
                sim.addParticle(cx + cos * r, cy + sin * r, -sin * v, cos * v, {
                    mass: m, charge: 0, spin: (Math.random() - 0.5) * 0.5,
                });
            }
        },
    },
};

export const PRESET_ORDER = [
    'kepler', 'precession', 'atom', 'bremsstrahlung', 'tidallock',
    'hawking', 'pulsars', 'magnetic', 'galaxy',
];

// ─── Toggle/Setting Application ───

const TOGGLE_MAP = {
    gravity: 'gravity-toggle',
    coulomb: 'coulomb-toggle',
    magnetic: 'magnetic-toggle',
    gravitomag: 'gravitomag-toggle',
    relativity: 'relativity-toggle',
    onepn: 'onepn-toggle',
    signaldelay: 'signaldelay-toggle',
    blackhole: 'blackhole-toggle',
    radiation: 'radiation-toggle',
    tidallocking: 'tidallocking-toggle',
    spinorbit: 'spinorbit-toggle',
    tidal: 'tidal-toggle',
    barneshut: 'barneshut-toggle',
};

// Parent toggles first so dependency cascades run before children are set
const TOGGLE_ORDER = [
    'gravity', 'coulomb', 'relativity',
    'gravitomag', 'magnetic',
    'signaldelay', 'onepn', 'blackhole',
    'tidallocking', 'spinorbit', 'radiation', 'tidal', 'barneshut',
];

const VISUAL_MAP = {
    trails: 'trailsToggle',
    velocity: 'velocityToggle',
    force: 'forceToggle',
    forceComponents: 'forceComponentsToggle',
    potential: 'potentialToggle',
};

const MODE_GROUPS = {
    collision: { id: 'collision-toggles', attr: 'collision' },
    boundary: { id: 'boundary-toggles', attr: 'boundary' },
    topology: { id: 'topology-toggles', attr: 'topology' },
    interaction: { id: 'interaction-toggles', attr: 'mode' },
};

const SLIDER_MAP = {
    speed: { input: 'speedInput', display: 'speedValue', suffix: '' },
    friction: { input: 'frictionInput', display: 'frictionValue', suffix: '' },
};

export function loadPreset(name, sim) {
    const preset = PRESETS[name];
    if (!preset) return;

    // 1. Clear state
    sim.particles = [];
    sim.photons = [];
    sim.totalRadiated = 0;
    sim.totalRadiatedPx = 0;
    sim.totalRadiatedPy = 0;
    sim.selectedParticle = null;
    sim.physics._forcesInit = false;
    sim.camera.reset(sim.domainW / 2, sim.domainH / 2, WORLD_SCALE);

    // 2. Apply physics toggles in dependency order
    for (const key of TOGGLE_ORDER) {
        if (!(key in preset.toggles)) continue;
        const el = document.getElementById(TOGGLE_MAP[key]);
        if (!el) continue;
        const want = preset.toggles[key];
        if (el.checked !== want) {
            el.checked = want;
            el.setAttribute('aria-checked', String(want));
            el.dispatchEvent(new Event('change'));
        }
    }

    // 3. Apply mode toggle groups
    if (preset.settings) {
        for (const [key, { id, attr }] of Object.entries(MODE_GROUPS)) {
            const val = preset.settings[key];
            if (val == null) continue;
            const group = document.getElementById(id);
            if (!group) continue;
            const target = group.querySelector(`[data-${attr}="${val}"]`);
            if (target) target.click();
        }

        // Sliders
        for (const [key, { input, display, suffix }] of Object.entries(SLIDER_MAP)) {
            const val = preset.settings[key];
            if (val == null) continue;
            const el = document.getElementById(input);
            if (!el) continue;
            el.value = val;
            el.dispatchEvent(new Event('input'));
        }
    }

    // 4. Apply visual toggles
    if (preset.visuals) {
        for (const [key, elId] of Object.entries(VISUAL_MAP)) {
            const val = preset.visuals[key];
            if (val == null) continue;
            const el = document.getElementById(elId);
            if (!el) continue;
            if (el.checked !== val) {
                el.checked = val;
                el.dispatchEvent(new Event('change'));
            }
        }
    }

    // 5. Spawn particles
    preset.spawn(sim);

    // 6. Reset baseline and notify
    sim.stats.resetBaseline();
    showToast(preset.name);
}
