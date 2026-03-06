// ─── Named Constants ───
// All physics uses natural units: c = 1, G = 1.
export const PI = Math.PI;
export const TWO_PI = 2 * Math.PI;
export const HALF_PI = Math.PI / 2;

export const BH_THETA = 0.5;
export const QUADTREE_CAPACITY = 4;

// Plummer softening: same epsilon in both forces and PE for consistency
export const SOFTENING = 8;
export const SOFTENING_SQ = SOFTENING * SOFTENING;

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
export const MIN_MASS = 0.0078125;
export const MAX_PHOTONS = 1024;
export const LL_FORCE_CLAMP = 0.5; // max |F_rad| as fraction of |F_ext| (LL validity)

// Signal delay history buffer
export const HISTORY_SIZE = 256;
export const HISTORY_STRIDE = 64; // ~120 snapshots/second 

export const TIDAL_STRENGTH = 2.0;
export const FRAGMENT_COUNT = 4;

// Yukawa potential: V(r) = -g²·exp(-μr)/r
export const YUKAWA_G2 = 1.0;            // coupling strength (fixed)
export const DEFAULT_YUKAWA_MU = 0.2;   // mediator mass (inverse range)

// Axion dark matter: oscillating EM coupling α_eff = α·(1 + g·cos(m_a·t))
export const AXION_G = 0.1;              // coupling amplitude (fixed)
export const DEFAULT_AXION_MASS = 0.25; // oscillation frequency (m_a)

// Photon gravitational lensing
export const PHOTON_SOFTENING_SQ = 4;  // smaller than particle softening for tighter lensing

// Gravitational wave radiation (quadrupole formula)
// P = (1/5)|d³I_ij/dt³|² — coefficient used inline in integrator.js as 0.2

// Roche lobe overflow
export const ROCHE_THRESHOLD = 0.9;       // overflow starts at this fraction of Roche radius
export const ROCHE_TRANSFER_RATE = 0.01;  // mass transfer rate coefficient
export const ROCHE_MIN_PACKET = 0.02;     // minimum mass for a stream particle

// Cosmological expansion
export const DEFAULT_HUBBLE = 0.001;  // Hubble parameter
