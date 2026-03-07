// ─── Named Constants ───
// All physics uses natural units: c = 1, G = 1, ħ = 1.
export const PI = Math.PI;
export const TWO_PI = 2 * Math.PI;
export const HALF_PI = Math.PI / 2;

export const BH_THETA = 0.5;
export const QUADTREE_CAPACITY = 4;

// Plummer softening: same epsilon in both forces and PE for consistency
export const SOFTENING = 8;
export const SOFTENING_SQ = SOFTENING * SOFTENING;
export const BH_SOFTENING = 4;      // reduced softening for black hole mode
export const BH_SOFTENING_SQ = BH_SOFTENING * BH_SOFTENING;

// Solid sphere: I = (2/5)mr^2, mu = (1/5)q*omega*r^2
export const INERTIA_K = 0.4;
export const MAG_MOMENT_K = 0.2;

export const DESPAWN_MARGIN = 64;
export const MAX_TRAIL_LENGTH = 256;
export const WORLD_SCALE = 16;
export const ZOOM_MIN = WORLD_SCALE;
export const ZOOM_MAX = 48;
export const WHEEL_ZOOM_IN = 1.1;

export const MAX_SUBSTEPS = 32;
export const PHYSICS_DT = 1 / 128;
export const DEFAULT_SPEED_SCALE = 64;
export const MAX_SPEED_RATIO = 0.99;

// Radiation: tau = 2q^2/(3m), P = 2q^2*a^2/3
export const PHOTON_LIFETIME = 256;
export const MIN_MASS = 0.01;
export const MAX_PHOTONS = 1024;
export const LL_FORCE_CLAMP = 0.5; // max |F_rad| as fraction of |F_ext| (LL validity)

// Signal delay history buffer
export const HISTORY_SIZE = 256;
export const HISTORY_STRIDE = 64; // ~120 snapshots/second

export const TIDAL_STRENGTH = 2.0;

// Yukawa potential: V(r) = -g²·exp(-μr)/r
export const YUKAWA_G2 = 1.0;            // coupling strength (fixed)
export const DEFAULT_YUKAWA_MU = 0.05;  // mediator mass (inverse range)

// Scalar fields (shared grid size and field value clamp for Higgs and Axion)
export const SCALAR_GRID = 64;
export const SCALAR_FIELD_MAX = 2;       // field value clamp (prevent runaway)

// Axion field: V(a) = 1/2 m_a² a² (quadratic potential, no SSB)
export const DEFAULT_AXION_MASS = 0.05; // mediator mass (oscillation frequency)
export const AXION_COUPLING = 0.05;      // g in L = -(1+g·a)F²/4; also sets Q = 1/g, ζ = g/2

// Photon gravitational lensing
export const PHOTON_SOFTENING_SQ = 4;  // smaller than particle softening for tighter lensing

// Roche lobe overflow
export const ROCHE_THRESHOLD = 0.9;       // overflow starts at this fraction of Roche radius
export const ROCHE_TRANSFER_RATE = 0.01;  // mass transfer rate coefficient

// Cosmological expansion
export const DEFAULT_HUBBLE = 0.001;  // Hubble parameter

// Higgs field: V(φ) = -(m_H²/4)φ² + (m_H²/8)φ⁴ (VEV=1, λ=m_H²/2)
export const DEFAULT_HIGGS_MASS = 0.05;       // Higgs boson mass (mediator range ~ 1/m_H)
export const HIGGS_COUPLING = 0.05;            // Yukawa coupling g (source = g·baseMass, force = g·baseMass·∇φ)

// Numerical thresholds
export const EPSILON = 1e-9;          // general "effectively zero" guard
export const EPSILON_SQ = EPSILON * EPSILON; // squared epsilon (for magnitude² checks)
export const NR_TOLERANCE = EPSILON / 1000; // Newton-Raphson convergence (signal delay)
export const NR_MAX_ITER = 8;             // Newton-Raphson iteration cap (signal delay)

// Simulation control
export const MAX_FRAME_DT = 0.1;          // frame delta cap (100ms = 10fps floor)
export const ACCUMULATOR_CAP = 4;          // max accumulator as multiple of PHYSICS_DT * MAX_SUBSTEPS
export const MAX_REJECTION_SAMPLES = 32;   // rejection sampling iteration cap
export const QUADRUPOLE_POWER_CLAMP = 0.01;// max quadrupole dE as fraction of system KE
export const ABERRATION_THRESHOLD = 1.01;  // min gamma for relativistic aberration
export const BH_NAKED_FLOOR = 0.5;        // naked singularity horizon floor (M * this)

// Disintegration / Hawking
export const SPAWN_OFFSET_MULTIPLIER = 1.5; // spawn offset = max(radius * this, FLOOR)
export const SPAWN_OFFSET_FLOOR = 4;       // minimum spawn offset (absolute distance)
export const SPAWN_COUNT = 4;          // fragments per disintegration
export const SPAWN_MIN_ENERGY = 0.5;  // minimum energy per emitted photon (Hawking / annihilation)

// Pair production: photon -> matter + antimatter near massive body
export const PAIR_PROD_MIN_ENERGY = 2;    // minimum photon energy for pair production (2mc²)
export const PAIR_PROD_RADIUS = 8;        // proximity to massive body required
export const PAIR_PROD_PROB = 0.005;      // probability per substep per eligible photon
export const PAIR_PROD_MAX_PARTICLES = 32; // suppress pair production above this particle count
export const PAIR_PROD_MIN_AGE = 64;      // photon must survive this many ticks before it can pair-produce

// Input
export const PINCH_DEBOUNCE = 300;         // ms guard after pinch-to-zoom
export const DRAG_THRESHOLD = 4;           // world-space distance: click vs drag
export const SHOOT_VELOCITY_SCALE = 0.02;  // shoot mode: drag distance → velocity
export const ORBIT_SEARCH_RADIUS = 10;     // orbit mode: min distance to consider a body

// Display
export const DISPLAY_SCALE = 100;          // energy/momentum × this for readout
export const STATS_THROTTLE_MASK = 3;      // update stats every (mask+1)th frame
export const PHASE_BUFFER_LEN = 512;       // phase plot ring buffer samples
export const HEATMAP_GRID = 64;            // heatmap resolution (independent of Higgs grid)
export const HEATMAP_INTERVAL = 8;         // frames between heatmap recomputes
export const HEATMAP_SENSITIVITY = 2;      // tanh scaling for potential → alpha
export const HEATMAP_MAX_ALPHA = 100;      // max alpha before cap

// Rendering scales (force/velocity/torque vector lengths)
export const VELOCITY_VECTOR_SCALE = 32;
export const FORCE_VECTOR_SCALE = 256;

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
