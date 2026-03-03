// ─── Named Constants ───

// Barnes-Hut
export const BH_THETA = 0.5;
export const QUADTREE_CAPACITY = 4;

// Plummer softening: rSq_eff = rSq + SOFTENING_SQ, keeps F and PE consistent
export const SOFTENING_SQ = 25;

// Moment of inertia: I = INERTIA_K * m * r² (uniform-density solid sphere = 2/5)
export const INERTIA_K = 0.4;

// Boundary
export const DESPAWN_MARGIN = 100;

// Rendering
export const MAX_TRAIL_LENGTH = 200;

// Camera / Zoom
export const ZOOM_MIN = 1;
export const ZOOM_MAX = 3;
export const WHEEL_ZOOM_IN = 1.1;

// Simulation
export const DEFAULT_SPEED_SCALE = 20;
export const MAX_SPEED_RATIO = 0.99;
