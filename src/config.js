// ─── Named Constants ───
// All physics uses natural units: c = 1, G = 1.
export const BH_THETA = 0.5;
export const QUADTREE_CAPACITY = 4;

// Plummer softening: same epsilon in both forces and PE for consistency
export const SOFTENING = 10;
export const SOFTENING_SQ = SOFTENING * SOFTENING;

// Solid sphere: I = (2/5)mr^2, mu = (1/5)q*omega*r^2
export const INERTIA_K = 0.4;
export const MAG_MOMENT_K = 0.2;

export const DESPAWN_MARGIN = 100;
export const MAX_TRAIL_LENGTH = 200;
export const ZOOM_MIN = 1;
export const ZOOM_MAX = 6;
export const WHEEL_ZOOM_IN = 1.1;

export const MAX_SUBSTEPS = 16;
export const PHYSICS_DT = 1 / 120;
export const DEFAULT_SPEED_SCALE = 100;
export const MAX_SPEED_RATIO = 0.99;

// Radiation: tau = 2*LARMOR_K*q^2/m, P = 2q^2*a^2/3
export const LARMOR_K = 1 / 3;
export const PHOTON_LIFETIME = 240;
export const RADIATION_THRESHOLD = 0.01;
export const MAX_PHOTONS = 500;
export const LL_FORCE_CLAMP = 0.5; // max |F_rad*dt/m| as fraction of |w|

// Signal delay history buffer
export const HISTORY_SIZE = 1024;
export const HISTORY_STRIDE = 200; // record every N updates; ~60 snapshots/sec at 100× speed

export const FRAME_DRAG_K = 0.1;
export const TIDAL_STRENGTH = 2.0;
export const MIN_FRAGMENT_MASS = 2;
export const FRAGMENT_COUNT = 3;
