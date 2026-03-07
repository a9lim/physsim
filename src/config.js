// ─── Named Constants ───
// All physics uses natural units: c = 1, G = 1, ħ = 1.

// ── Math ──
export const PI = Math.PI;
export const TWO_PI = 2 * Math.PI;
export const HALF_PI = Math.PI / 2;

// ── Numerical Thresholds ──
export const EPSILON = 1e-9;
export const EPSILON_SQ = EPSILON * EPSILON;
export const NR_TOLERANCE = EPSILON / 1000; // Newton-Raphson convergence (signal delay)
export const NR_MAX_ITER = 8;
export const MIN_MASS = 0.01;               // emission threshold, evaporation floor
export const MAX_SPEED_RATIO = 0.99;

// ── Integrator ──
export const PHYSICS_DT = 1 / 128;
export const MAX_SUBSTEPS = 32;
export const DEFAULT_SPEED_SCALE = 64;
export const MAX_FRAME_DT = 0.1;             // frame delta cap (100ms = 10fps floor)
export const ACCUMULATOR_CAP = 4;            // max accumulator as multiple of PHYSICS_DT * MAX_SUBSTEPS

// ── Softening ──
export const SOFTENING = 8;
export const SOFTENING_SQ = SOFTENING * SOFTENING;
export const BH_SOFTENING = 4;               // reduced softening for black hole mode
export const BH_SOFTENING_SQ = BH_SOFTENING * BH_SOFTENING;
export const BOSON_SOFTENING_SQ = 4;         // photon/pion gravitational lensing
export const BOSON_ABSORB_FRACTION = 1;      // absorption cross-section = fraction of target radius
export const BOSON_MIN_AGE = 4;              // minimum age (substeps) before any absorption

// ── Particle Properties ──
export const INERTIA_K = 0.4;                // I = (2/5)mr²
export const MAG_MOMENT_K = 0.2;             // μ = (1/5)q·ω·r²
export const TIDAL_STRENGTH = 2.0;

// ── Barnes-Hut ──
export const BH_THETA = 0.5;
export const QUADTREE_CAPACITY = 4;

// ── Radiation & Bosons ──
export const LL_FORCE_CLAMP = 0.5;           // max |F_rad| as fraction of |F_ext|
export const PHOTON_LIFETIME = 256;
export const MAX_PHOTONS = 1024;
export const PION_LIFETIME = 32;
export const MAX_PIONS = 256;
export const MAX_REJECTION_SAMPLES = 32;     // quadrupole rejection sampling cap
export const QUADRUPOLE_POWER_CLAMP = 0.01;  // max quadrupole dE as fraction of system KE
export const ABERRATION_THRESHOLD = 1.01;    // min gamma for relativistic aberration

// ── Yukawa ──
export const YUKAWA_G2 = 32;                // coupling strength (fixed)
export const DEFAULT_YUKAWA_MU = 0.05;       // mediator mass (inverse range)

// ── Scalar Fields (Higgs & Axion) ──
export const SCALAR_GRID = 64;
export const SCALAR_FIELD_MAX = 2;           // field value clamp
export const FIELD_EXCITATION_SIGMA = 2;     // Gaussian width in grid cells (merge wave packets)
export const MERGE_EXCITATION_SCALE = 0.5;   // amplitude = scale * sqrt(keLost)

export const DEFAULT_HIGGS_MASS = 0.05;      // m_H (oscillation frequency)
export const HIGGS_COUPLING = 1;             // g (source = g·baseMass, force = g·baseMass·∇φ)

export const DEFAULT_AXION_MASS = 0.05;      // m_a (oscillation frequency)
export const AXION_COUPLING = 0.05;          // g in L = -(1+g·a)F²/4

// ── Black Hole ──
export const BH_NAKED_FLOOR = 0.5;           // naked singularity horizon floor (M × this)

// ── Disintegration & Roche ──
export const SPAWN_COUNT = 4;
export const SPAWN_OFFSET_MULTIPLIER = 1.5;
export const SPAWN_OFFSET_FLOOR = 1;
export const SPAWN_MIN_ENERGY = 0.01;         // min energy per emitted photon (Hawking / annihilation)
export const ROCHE_THRESHOLD = 0.9;
export const ROCHE_TRANSFER_RATE = 0.01;

// ── Pair Production ──
export const PAIR_PROD_MIN_ENERGY = 2;       // 2mc²
export const PAIR_PROD_RADIUS = 8;           // proximity to massive body
export const PAIR_PROD_PROB = 0.005;         // probability per substep
export const PAIR_PROD_MAX_PARTICLES = 32;
export const PAIR_PROD_MIN_AGE = 64;         // photon age before eligible

// ── Signal Delay ──
export const HISTORY_SIZE = 256;
export const HISTORY_STRIDE = 64;            // ~120 snapshots/second

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
export const ORBIT_SEARCH_RADIUS = 10;

// ── Display ──
export const MAX_TRAIL_LENGTH = 256;
export const DISPLAY_SCALE = 100;
export const STATS_THROTTLE_MASK = 3;
export const PHASE_BUFFER_LEN = 512;
export const HEATMAP_GRID = 64;
export const HEATMAP_INTERVAL = 8;
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
    return disc > 0 ? M + Math.sqrt(disc) : M * BH_NAKED_FLOOR;
}
