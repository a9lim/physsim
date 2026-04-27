// ─── Preset Definitions ───
// Each preset configures toggles, engine settings, visuals, and spawns particles.
import { PI, TWO_PI, WORLD_SCALE, SOFTENING_SQ, DEFAULT_PION_MASS, DEFAULT_HIGGS_MASS, DEFAULT_AXION_MASS, SPEED_OPTIONS, DEFAULT_SPEED_INDEX } from './config.js';

// Plummer-softened circular orbit velocity: F = coupling*r/(r²+ε²)^{3/2}, set F/m = v²/r
const _vCirc = (coupling, r, m) => {
    const rSq = r * r + SOFTENING_SQ;
    return Math.sqrt(Math.abs(coupling) * r * r / (m * rSq * Math.sqrt(rSq)));
};

// Gravitational circular orbit: _vCirc(M, r, 1)
const _vGrav = (M, r) => {
    const rSq = r * r + SOFTENING_SQ;
    return Math.sqrt(M * r * r / (rSq * Math.sqrt(rSq)));
};

export const PRESETS = {
    // ─── Gravity ───

    kepler: {
        name: 'Kepler Orbits',
        desc: 'Classical gravity — Keplerian motion and conservation laws',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: false,
            relativity: false, onepn: false, blackhole: false,
            radiation: false, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: false, expansion: false, higgs: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 32 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            sim.addParticle(cx, cy, 0, 0, { mass: 5, charge: 0, spin: 0 });
            const radii = [8, 13, 18, 23, 28];
            for (let i = 0; i < radii.length; i++) {
                const r = radii[i];
                const angle = Math.random() * TWO_PI;
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
            relativity: true, onepn: true, blackhole: false,
            radiation: false, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: false, expansion: false, higgs: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 32 },
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

    inspiral: {
        name: 'Binary Inspiral',
        desc: 'Gravitational wave emission — quadrupole radiation drains orbital energy',
        // particles carry zero charge so EM forces contribute nothing.
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: true,
            relativity: true, onepn: true, blackhole: false,
            radiation: true, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: false, expansion: false, higgs: false,
        },
        settings: { collision: 'merge', boundary: 'despawn', speed: 32 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            // Wide, weakly-relativistic binary so 1PN doesn't overshoot at perihelion
            // and radiation reaction can drive a slow inward spiral instead of a fling.
            const M = 3;
            const sep = 22;
            // Equal-mass binary: each circles CoM at radius `sep`; v_circ = sqrt(GM/(4·sep))
            const v = Math.sqrt(M / (4 * sep));
            // Aligned spins — GM dipole 3L₁L₂/r⁴ is attractive (repulsive when anti-aligned),
            // so frame-dragging tightens the orbit instead of throwing them apart.
            sim.addParticle(cx - sep, cy, 0, v, { mass: M, charge: 0, spin: 0.5 });
            sim.addParticle(cx + sep, cy, 0, -v, { mass: M, charge: 0, spin: 0.5 });
        },
    },

    hawking: {
        name: 'Hawking Evaporation',
        desc: 'Small black holes radiate and evaporate',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: true,
            relativity: true, onepn: false, blackhole: true,
            radiation: true, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: false, expansion: false, higgs: false,
        },
        settings: { collision: 'merge', boundary: 'despawn', speed: 32 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            // Small BHs that will visibly evaporate (P ∝ 1/M²)
            const masses = [0.3, 0.4, 0.5, 0.65, 0.8];
            for (let i = 0; i < masses.length; i++) {
                const angle = (TWO_PI * i) / masses.length;
                const r = 12 + Math.random() * 8;
                sim.addParticle(
                    cx + Math.cos(angle) * r, cy + Math.sin(angle) * r,
                    (Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.1,
                    { mass: masses[i], charge: 0, spin: 0 }
                );
            }
        },
    },

    // ─── Electromagnetism ───

    atom: {
        name: 'Atom',
        desc: 'Coulomb binding — electromagnetic atomic structure',
        toggles: {
            gravity: false, coulomb: true, magnetic: true, gravitomag: false,
            relativity: true, onepn: false, blackhole: false,
            radiation: false, spinorbit: true, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: false, expansion: false, higgs: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 32 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            const nucQ = 3;
            sim.addParticle(cx, cy, 0, 0, { mass: 5, charge: nucQ, spin: 0 });
            // 2 electrons with screened Coulomb coupling (inner screens outer)
            const eM = 1.0, eQ = -1;
            const shells = [{ r: 12, effectiveQ: 3 }, { r: 18, effectiveQ: 2 }];
            for (let i = 0; i < shells.length; i++) {
                const { r, effectiveQ } = shells[i];
                const angle = PI * i;
                const v = _vCirc(effectiveQ * Math.abs(eQ), r, eM);
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
            relativity: true, onepn: false, blackhole: false,
            radiation: true, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: false, expansion: false, higgs: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 32 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            // Heavy target at center
            sim.addParticle(cx, cy, 0, 0, { mass: 5, charge: 3, spin: 0 });
            // Projectile on near-miss trajectory
            sim.addParticle(cx - 25, cy + 5, 0.5, 0, { mass: 0.2, charge: -2, spin: 0 });
        },
    },

    magnetic: {
        name: 'Magnetic Dipoles',
        desc: 'Dipole interactions, Lorentz force, and spin-orbit coupling',
        toggles: {
            gravity: false, coulomb: true, magnetic: true, gravitomag: false,
            relativity: true, onepn: false, blackhole: false,
            radiation: false, spinorbit: true, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: false, expansion: false, higgs: false,
        },
        settings: { collision: 'bounce', boundary: 'loop', topology: 'torus', speed: 32, friction: 0.4 },
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

    // ─── Exotic ───

    nucleus: {
        name: 'Atomic Nucleus',
        desc: 'Yukawa potential — short-range force binds nucleons like a liquid drop',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: false,
            relativity: false, onepn: false, blackhole: false,
            radiation: false, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: true, axion: false, expansion: false, higgs: false,
        },
        settings: { collision: 'bounce', boundary: 'bounce', speed: 32, yukawaMu: 1 / 15 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            // Nucleons in a ring — gravity + Yukawa pull them into a bound cluster
            const N = 7;
            for (let i = 0; i < N; i++) {
                const angle = (TWO_PI * i) / N;
                const r = 10 + Math.random() * 4;
                const cos = Math.cos(angle), sin = Math.sin(angle);
                // Small tangential velocity → rotating collapse
                const v = 0.04;
                sim.addParticle(
                    cx + cos * r, cy + sin * r,
                    -sin * v + (Math.random() - 0.5) * 0.02,
                    cos * v + (Math.random() - 0.5) * 0.02,
                    { mass: 2, charge: 0, spin: (Math.random() - 0.5) * 0.3 }
                );
            }
        },
    },

    axion: {
        name: 'Axion Field',
        desc: 'Scalar aF² coupling makes EM strength position-dependent',
        toggles: {
            gravity: false, coulomb: true, magnetic: true, gravitomag: false,
            relativity: true, onepn: false, blackhole: false,
            radiation: false, spinorbit: true, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: true, expansion: false, higgs: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 32, axionMass: 0.15 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            const sep = 28;
            // Two atoms side by side — axion makes both breathe in unison
            for (let s = -1; s <= 1; s += 2) {
                const nx = cx + s * sep / 2;
                const nucQ = 3;
                sim.addParticle(nx, cy, 0, 0, { mass: 4, charge: nucQ, spin: 0 });
                const r = 10;
                const v = _vCirc(nucQ, r, 0.8);
                const angle = s > 0 ? 0 : PI;
                const cos = Math.cos(angle), sin = Math.sin(angle);
                sim.addParticle(
                    nx + cos * r, cy + sin * r,
                    -sin * v, cos * v,
                    { mass: 0.8, charge: -1, spin: 0.2 }
                );
            }
        },
    },

    higgs: {
        name: 'Higgs Mechanism',
        desc: 'Spontaneous symmetry breaking — particles gain mass from the Higgs field',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: false,
            relativity: false, onepn: false, blackhole: false,
            radiation: false, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: false, expansion: false, higgs: true,
        },
        settings: { collision: 'bounce', boundary: 'bounce', speed: 32 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            // Particles scattered around — watch them gain mass and attract
            for (let i = 0; i < 8; i++) {
                const angle = (TWO_PI * i) / 8;
                const r = 8 + Math.random() * 10;
                sim.addParticle(
                    cx + Math.cos(angle) * r, cy + Math.sin(angle) * r,
                    (Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.1,
                    { mass: 1 + Math.random(), charge: 0, spin: 0 }
                );
            }
        },
    },

    pionexchange: {
        name: 'Pion Exchange',
        desc: 'Yukawa scattering emits massive pions — watch them carry force between nucleons',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: false,
            relativity: false, onepn: false, blackhole: false,
            radiation: false, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: true, axion: false, expansion: false, higgs: false,
        },
        settings: { collision: 'bounce', boundary: 'bounce', speed: 32, yukawaMu: 0.1 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            const N = 5;
            for (let i = 0; i < N; i++) {
                const angle = (TWO_PI * i) / N;
                const r = 8;
                const v = 0.04;
                const cos = Math.cos(angle), sin = Math.sin(angle);
                sim.addParticle(cx + cos * r, cy + sin * r,
                    -sin * v + (Math.random() - 0.5) * 0.02,
                    cos * v + (Math.random() - 0.5) * 0.02,
                    { mass: 3, charge: 0, spin: (Math.random() - 0.5) * 0.2 });
            }
        },
    },

    higgsboson: {
        name: 'Higgs Boson',
        desc: 'Particle collisions excite the Higgs field — watch wave packets ripple outward',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: false,
            relativity: false, onepn: false, blackhole: false,
            radiation: false, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: false, expansion: false, higgs: true,
        },
        settings: { collision: 'merge', boundary: 'bounce', speed: 32 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            // Head-on collision to produce Higgs excitation
            sim.addParticle(cx - 15, cy, 0.4, 0, { mass: 3, charge: 0, spin: 0 });
            sim.addParticle(cx + 15, cy, -0.4, 0, { mass: 3, charge: 0, spin: 0 });
            // Stationary witnesses to feel the wave
            for (let i = 0; i < 6; i++) {
                const angle = (TWO_PI * i) / 6;
                const r = 20;
                sim.addParticle(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r, 0, 0,
                    { mass: 1, charge: 0, spin: 0 });
            }
        },
    },

    axionburst: {
        name: 'Axion Burst',
        desc: 'Charged collisions excite the axion field — EM coupling ripples outward',
        toggles: {
            gravity: false, coulomb: true, magnetic: true, gravitomag: false,
            relativity: true, onepn: false, blackhole: false,
            radiation: false, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: true, expansion: false, higgs: false,
        },
        settings: { collision: 'merge', boundary: 'despawn', speed: 32, axionMass: 0.1 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            // Charged head-on collision
            sim.addParticle(cx - 12, cy, 0.3, 0, { mass: 2, charge: 3, spin: 0 });
            sim.addParticle(cx + 12, cy, -0.3, 0, { mass: 2, charge: -3, spin: 0 });
            // Orbiting electrons to feel axion EM modulation
            const r = 22;
            for (let i = 0; i < 4; i++) {
                const angle = (TWO_PI * i) / 4;
                sim.addParticle(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r, 0, 0,
                    { mass: 0.5, charge: -1, spin: 0.2 });
            }
        },
    },

    pecceiQuinn: {
        name: 'Peccei\u2013Quinn',
        desc: 'CP violation \u2014 axion field makes Yukawa bind matter and antimatter differently',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: false,
            relativity: false, onepn: false, blackhole: false,
            radiation: false, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: true, axion: true, expansion: false, higgs: false,
        },
        settings: { collision: 'bounce', boundary: 'bounce', speed: 32, axionMass: 0.05 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            const sep = 24, n = 4, r = 10;
            // Matter cluster (left)
            for (let i = 0; i < n; i++) {
                const angle = (TWO_PI * i) / n;
                sim.addParticle(cx - sep + Math.cos(angle) * r, cy + Math.sin(angle) * r, 0, 0,
                    { mass: 4, charge: 0, spin: 0.5 });
            }
            // Antimatter cluster (right) — mirror image
            for (let i = 0; i < n; i++) {
                const angle = (TWO_PI * i) / n;
                sim.addParticle(cx + sep + Math.cos(angle) * r, cy + Math.sin(angle) * r, 0, 0,
                    { mass: 4, charge: 0, spin: 0.5, antimatter: true });
            }
        },
    },

    // ─── Cosmological ───

    expansion: {
        name: 'Expanding Universe',
        desc: 'Cosmological expansion — Hubble flow vs gravitational binding',
        toggles: {
            gravity: true, coulomb: false, magnetic: false, gravitomag: false,
            relativity: false, onepn: false, blackhole: false,
            radiation: false, spinorbit: false, disintegration: false,
            barneshut: false, bosonInter: false, yukawa: false, axion: false, expansion: true, higgs: false,
        },
        settings: { collision: 'pass', boundary: 'despawn', speed: 32, hubble: 0.008 },
        visuals: { trails: true, velocity: false, force: false, forceComponents: false, potential: false },
        spawn(sim) {
            const cx = sim.domainW / 2, cy = sim.domainH / 2;
            // Uniform cloud at rest — inner particles stay bound, outer ones recede
            for (let i = 0; i < 25; i++) {
                const angle = Math.random() * TWO_PI;
                const r = 2 + Math.random() * 25;
                sim.addParticle(
                    cx + Math.cos(angle) * r, cy + Math.sin(angle) * r,
                    0, 0,
                    { mass: 0.3, charge: 0, spin: 0 }
                );
            }
        },
    },
};

export const PRESET_ORDER = [
    'kepler', 'precession', 'inspiral', 'hawking',
    'atom', 'bremsstrahlung', 'magnetic',
    'nucleus', 'axion', 'pionexchange', 'higgs', 'higgsboson', 'axionburst', 'pecceiQuinn',
    'expansion',
];

// ─── Toggle/Setting Application ───

const TOGGLE_MAP = {
    gravity: 'gravity-toggle',
    bosonInter: 'bosoninter-toggle',
    coulomb: 'coulomb-toggle',
    magnetic: 'magnetic-toggle',
    gravitomag: 'gravitomag-toggle',
    relativity: 'relativity-toggle',
    onepn: 'onepn-toggle',
    blackhole: 'blackhole-toggle',
    radiation: 'radiation-toggle',
    spinorbit: 'spinorbit-toggle',
    disintegration: 'disintegration-toggle',
    barneshut: 'barneshut-toggle',
    yukawa: 'yukawa-toggle',
    axion: 'axion-toggle',
    expansion: 'expansion-toggle',
    higgs: 'higgs-toggle',
};

// Parent toggles first so dependency cascades run before children are set
const TOGGLE_ORDER = [
    'gravity', 'coulomb', 'relativity',
    'bosonInter', 'gravitomag', 'magnetic',
    'onepn', 'blackhole',
    'spinorbit', 'radiation', 'disintegration', 'barneshut',
    'yukawa', 'axion', 'expansion', 'higgs',
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
    potentialMode: { id: 'potential-mode-toggles', attr: 'potential' },
};

const SLIDER_MAP = {
    friction: 'frictionInput',
    yukawaMu: 'yukawaMuInput',
    axionMass: 'axionMassInput',
    hubble: 'hubbleInput',
    higgsMass: 'higgsMassInput',
    extGravity: 'extGravityInput',
    extGravityAngle: 'extGravityAngleInput',
    extElectric: 'extElectricInput',
    extElectricAngle: 'extElectricAngleInput',
    extBz: 'extBzInput',
};

export function loadPreset(name, sim) {
    const preset = PRESETS[name];
    if (!preset) return;

    // 1. Clear state (backend-agnostic)
    sim.reset();
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
        for (const [key, elId] of Object.entries(SLIDER_MAP)) {
            const val = preset.settings[key];
            if (val == null) continue;
            const el = document.getElementById(elId);
            if (!el) continue;
            el.value = val;
            el.dispatchEvent(new Event('input'));
        }

        // Speed: find nearest SPEED_OPTIONS index
        const presetSpeed = preset.settings.speed;
        if (presetSpeed != null) {
            let bestIdx = DEFAULT_SPEED_INDEX;
            let bestDist = Infinity;
            for (let i = 0; i < SPEED_OPTIONS.length; i++) {
                const d = Math.abs(SPEED_OPTIONS[i] - presetSpeed);
                if (d < bestDist) { bestDist = d; bestIdx = i; }
            }
            sim.speedIndex = bestIdx;
        } else {
            sim.speedIndex = DEFAULT_SPEED_INDEX;
        }
        sim.speedScale = SPEED_OPTIONS[sim.speedIndex];
        const speedBtn = document.getElementById('speedBtn');
        if (speedBtn && typeof _toolbar !== 'undefined') _toolbar.updateSpeedBtn(speedBtn, sim.speedScale / 16);
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

    // 5. Reset external fields (presets can override via settings)
    const extDefaults = { extGravity: 0, extGravityAngle: 90, extElectric: 0, extElectricAngle: 0, extBz: 0, yukawaMu: DEFAULT_PION_MASS, higgsMass: DEFAULT_HIGGS_MASS, axionMass: DEFAULT_AXION_MASS };
    for (const [key, elId] of Object.entries(SLIDER_MAP)) {
        if (key in extDefaults && !(preset.settings && key in preset.settings)) {
            const el = document.getElementById(elId);
            if (el && parseFloat(el.value) !== extDefaults[key]) {
                el.value = extDefaults[key];
                el.dispatchEvent(new Event('input'));
            }
        }
    }

    // 6. Spawn particles
    preset.spawn(sim);

    // 7. Reset baseline and notify
    sim.stats.resetBaseline();
    showToast(preset.name);
}
