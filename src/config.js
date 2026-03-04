// ─── Named Constants ───

// Barnes-Hut
export const BH_THETA = 0.5;
export const QUADTREE_CAPACITY = 4;

// Plummer softening: rSq_eff = rSq + SOFTENING_SQ, keeps F and PE consistent
export const SOFTENING = 10;
export const SOFTENING_SQ = SOFTENING * SOFTENING;

// Moment of inertia: I = INERTIA_K * m * r² (uniform-density solid sphere = 2/5)
export const INERTIA_K = 0.4;

// Magnetic moment: μ = MAG_MOMENT_K * q * ω * r² (uniform charge density solid sphere = 1/5)
export const MAG_MOMENT_K = 0.2;

// Boundary
export const DESPAWN_MARGIN = 100;

// Rendering
export const MAX_TRAIL_LENGTH = 200;

// Camera / Zoom
export const ZOOM_MIN = 1;
export const ZOOM_MAX = 3;
export const WHEEL_ZOOM_IN = 1.1;

// Adaptive substepping
export const MAX_SUBSTEPS = 16;

// Simulation
export const DEFAULT_SPEED_SCALE = 100;
export const MAX_SPEED_RATIO = 0.99;

// Larmor radiation
export const LARMOR_K = 1 / 3; // τ = 2·LARMOR_K·q²/m = 2q²/(3m), P = 2q²a²/3 (c=G=1, ε₀=1/(4π))
export const PHOTON_LIFETIME = 300;          // frames before despawn
export const RADIATION_THRESHOLD = 0.01;     // min energy per frame to emit visible photon
export const MAX_PHOTONS = 500;              // photon pool cap
export const LL_FORCE_CLAMP = 0.5;           // max |F_rad·dt/m| as fraction of |w|

// Signal delay
export const HISTORY_SIZE = 512;    // circular buffer capacity per particle

// Tidal breakup
export const FRAME_DRAG_K = 0.1;             // frame-dragging spin alignment strength

export const TIDAL_STRENGTH = 2.0;
export const MIN_FRAGMENT_MASS = 2; // don't fragment below this mass
export const FRAGMENT_COUNT = 3;    // split into 3 pieces
