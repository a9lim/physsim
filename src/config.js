// ─── Named Constants ───
// All physics uses natural units: c = 1, G = 1, ħ = 1.

// ── Math ──
export const PI = Math.PI;
export const TWO_PI = 2 * Math.PI;
export const HALF_PI = Math.PI / 2;

// ── Mode Enums ──
export const COL_PASS = 0;
export const COL_MERGE = 1;
export const COL_BOUNCE = 2;
export const COL_NAMES = ['pass', 'merge', 'bounce'];
export function colFromString(s) { return s === 'merge' ? COL_MERGE : s === 'bounce' ? COL_BOUNCE : COL_PASS; }

export const BOUND_DESPAWN = 0;
export const BOUND_BOUNCE = 1;
export const BOUND_LOOP = 2;
export const BOUND_NAMES = ['despawn', 'bounce', 'loop'];
export function boundFromString(s) { return s === 'loop' ? BOUND_LOOP : s === 'bounce' ? BOUND_BOUNCE : BOUND_DESPAWN; }

export const TORUS = 0;
export const KLEIN = 1;
export const RP2   = 2;
export const TOPO_NAMES = ['torus', 'klein', 'rp2'];
export function topoFromString(s) { return s === 'klein' ? KLEIN : s === 'rp2' ? RP2 : TORUS; }

// ── Numerical Thresholds ──
export const EPSILON = 1e-9;
export const EPSILON_SQ = EPSILON * EPSILON;
export const SOLVE_TOLERANCE = EPSILON / 1000; // quadratic solver tolerance (signal delay)
export const MIN_MASS = 0.05;               // emission threshold, evaporation floor
export const MAX_SPEED_RATIO = 0.99;

// ── Integrator ──
export const PHYSICS_DT = 1 / 128;
export const MAX_SUBSTEPS = 32;
export const SPEED_OPTIONS = [4, 8, 16, 32, 64];
export const DEFAULT_SPEED_INDEX = 2; // 16x (displays as 1x)
export const DEFAULT_SPEED_SCALE = SPEED_OPTIONS[DEFAULT_SPEED_INDEX];
export const MAX_FRAME_DT = 1 / 16;             // frame delta cap
export const ACCUMULATOR_CAP = 2;            // max accumulator as multiple of PHYSICS_DT * MAX_SUBSTEPS (A6: reduced from 4)

// ── Softening ──
export const SOFTENING = 8;
export const SOFTENING_SQ = SOFTENING * SOFTENING;
export const BH_SOFTENING = 4;               // reduced softening for black hole mode
export const BH_SOFTENING_SQ = BH_SOFTENING * BH_SOFTENING;
export const BOSON_SOFTENING_SQ = 4;         // photon/pion gravitational lensing
export const BOSON_MIN_AGE = 4;              // minimum age (substeps) before any absorption

// ── Particle Properties ──
export const INERTIA_K = 0.4;                // I = (2/5)mr²
export const MAG_MOMENT_K = 0.2;             // μ = (1/5)q·ω·r²
export const TIDAL_STRENGTH = 64;

// ── Barnes-Hut ──
export const BH_THETA = 0.5;
export const BH_THETA_SQ = BH_THETA * BH_THETA;
export const QUADTREE_CAPACITY = 4;

// ── Radiation & Bosons ──
export const PHOTON_LIFETIME = 256;
export const MAX_PHOTONS = 1024;
export const MAX_PIONS = 256;
export const GPU_MAX_PHOTONS = 4096;
export const GPU_MAX_PIONS = 1024;
export const MAX_LEPTONS = 256;
export const LEPTON_LIFETIME = 512;
export const PION_HALF_LIFE = 32;                // pi0 half-life (fast EM decay)
export const CHARGED_PION_HALF_LIFE = 64;        // pi+/- half-life (slower weak decay)
export const PION_DECAY_PROB = 1 - Math.exp(-Math.LN2 / PION_HALF_LIFE * PHYSICS_DT);
export const CHARGED_PION_DECAY_PROB = 1 - Math.exp(-Math.LN2 / CHARGED_PION_HALF_LIFE * PHYSICS_DT);
export const ELECTRON_MASS = 0.01;                // decay product mass for pi+/- -> e+/- + photon
export const BOSON_CHARGE = 0.1;                  // magnitude of pion/lepton charge (tunable)
export const MAX_REJECTION_SAMPLES = 32;     // quadrupole rejection sampling cap
export const ABERRATION_THRESHOLD = 1.01;    // min gamma for relativistic aberration

// ── Yukawa ──
export const YUKAWA_COUPLING = 14;           // g² coupling strength (fixed)
export const DEFAULT_PION_MASS = 0.15;       // mediator mass (inverse range), ~m_π/m_N

// ── Scalar Fields (Higgs & Axion) ──
export const SCALAR_GRID = 64;
export const SCALAR_FIELD_MAX = 2;           // field value clamp
export const FIELD_EXCITATION_SIGMA = 2;     // Gaussian width in grid cells (merge wave packets)
export const MERGE_EXCITATION_SCALE = 0.5;   // amplitude = scale * sqrt(keLost)

export const DEFAULT_HIGGS_MASS = 0.50;      // m_H (oscillation frequency), ~m_H/v_EW
export const HIGGS_COUPLING = 1;             // g (source = g·baseMass, force = g·baseMass·∇φ)
export const HIGGS_MASS_FLOOR = 0.05;        // min |φ| for mass: m ≥ 0.05·baseMass (caps accel at 20×)
export const HIGGS_MASS_MAX_DELTA = 4;       // max mass change per unit time (prevents resonant throb)
export const SELFGRAV_PHI_MAX = 0.2;        // clamp |Φ| to keep weak-field approx valid (1+4Φ>0 requires Φ>-0.25)
export const EXCITATION_MAX_AMPLITUDE = 1.0; // cap merge wave-packet amplitude (prevents field shatter)

export const DEFAULT_AXION_MASS = 0.05;      // m_a (oscillation frequency)
export const AXION_COUPLING = 0.05;          // g in L = -(1+g·a)F²/4
export const HIGGS_AXION_COUPLING = 0.01;    // λ in V_portal = ½λφ²a² (Higgs portal)

// ── Black Hole ──

// ── Schwinger Discharge ──
// Vacuum pair production at BH horizon when E > E_cr.
// E_field = |Q| / r+², rate ∝ (E/E_cr)² exp(-π E_cr/E) per unit area.

// ── Superradiance ──
// BH spin → axion field amplification. Rate: Γ = COEFF · (Mμ)² · (Ω_H - μ_a)
export const SUPERRADIANCE_COEFF = 500;

// ── Disintegration & Roche ──
export const SPAWN_COUNT = 4;
export const SPAWN_OFFSET_MULTIPLIER = 1.5;
export const SPAWN_OFFSET_FLOOR = 1;
export const SPAWN_MIN_ENERGY = 0.05;         // min energy per emitted photon (Hawking / annihilation)
export const ROCHE_THRESHOLD = 0.9;
export const ROCHE_TRANSFER_RATE = 0.01;

// ── Pair Production ──
export const PAIR_PROD_MIN_ENERGY = 0.5;       // minimum photon energy for pair production
export const PAIR_PROD_RADIUS = 8;           // proximity to massive body
export const PAIR_PROD_PROB = 0.005;         // probability per substep
export const PAIR_PROD_MAX_PARTICLES = 32;
export const PAIR_PROD_MIN_AGE = 64;         // photon age before eligible

// ── Signal Delay ──
export const HISTORY_SIZE = 256;
export const HISTORY_MASK = HISTORY_SIZE - 1; // bitmask for modulo (power-of-2)
export const HISTORY_STRIDE = 64;            // ~120 snapshots/second

// ── GPU-Specific ──
export const GPU_SCALAR_GRID = 128;            // GPU scalar field grid resolution
export const GPU_SOLVE_TOLERANCE = 1e-5;      // GPU quadratic solver tolerance (f32 precision limit)
export const GPU_HEATMAP_GRID = 128;           // GPU heatmap overlay resolution (tunable to 128)
export const MAX_PARTICLES = 128;             // CPU particle pool cap
export const GPU_MAX_PARTICLES = 512;         // GPU buffer pre-allocation limit
export const GPU_MAX_SPEED_RATIO = 0.9999;    // GPU speed cap (f32 needs tighter bound than CPU 0.99)

// ── Cosmological Expansion ──
export const DEFAULT_HUBBLE = 0.001;

// ── Viewport & Camera ──
export const WORLD_SCALE = 16;
export const ZOOM_MIN = WORLD_SCALE;
export const ZOOM_MAX = 48;
export const WHEEL_ZOOM_IN = 1.1;
export const DESPAWN_MARGIN = 64;

// ── Input ──
export const PINCH_DEBOUNCE = 300;
export const DRAG_THRESHOLD = 4;
export const SHOOT_VELOCITY_SCALE = 0.02;

// ── Display ──
export const MAX_TRAIL_LENGTH = 256;
export const DISPLAY_SCALE = 100;
export const STATS_THROTTLE_MASK = 7;           // stats update every 8th frame
export const FIELD_THROTTLE_MASK = 1;            // field overlay render every 2nd frame
export const SIDEBAR_THROTTLE_MASK = 3;           // phase/effpot/selected update every 4th frame
export const PHASE_BUFFER_LEN = 512;
export const HEATMAP_GRID = 64;
export const HEATMAP_INTERVAL = 4;           // heatmap update every Nth render frame
export const HEATMAP_SENSITIVITY = 2;
export const HEATMAP_MAX_ALPHA = 100;
export const VELOCITY_VECTOR_SCALE = 32;
export const FORCE_VECTOR_SCALE = 256;

// ── Helpers ──

/** Compute photon/fragment spawn offset from particle radius. */
export function spawnOffset(radius) {
    return Math.max(radius * SPAWN_OFFSET_MULTIPLIER, SPAWN_OFFSET_FLOOR);
}

/** Kerr-Newman event horizon radius: r+ = M + sqrt(M² - a² - Q²). */
export function kerrNewmanRadius(M, radiusSq, angVel, charge) {
    const a = INERTIA_K * radiusSq * Math.abs(angVel);
    const disc = M * M - a * a - charge * charge;
    return disc >= 0 ? M + Math.sqrt(Math.max(0, disc)) : M;
}
