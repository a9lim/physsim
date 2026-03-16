/**
 * @fileoverview GPU constants generator — single source of truth for WGSL constants.
 *
 * buildWGSLConstants() generates a WGSL const declaration block from config.js
 * exports + _PALETTE hex→RGB conversions. Prepended to all shaders at compile time.
 *
 * paletteRGB() converts hex colors to normalized [r,g,b] for JS-side GPU use.
 */

import {
    SOFTENING, SOFTENING_SQ, BH_SOFTENING, BH_SOFTENING_SQ,
    INERTIA_K, MAG_MOMENT_K, TIDAL_STRENGTH,
    YUKAWA_COUPLING, AXION_COUPLING, EPSILON,
    PI, TWO_PI,
    BOSON_SOFTENING_SQ, BOSON_MIN_AGE, PHYSICS_DT,
    MIN_MASS, BH_NAKED_FLOOR, ELECTRON_MASS,
    PION_DECAY_PROB, CHARGED_PION_DECAY_PROB,
    BH_THETA, BH_THETA_SQ,
    VELOCITY_VECTOR_SCALE,
    MAX_PHOTONS, MAX_PIONS, PHOTON_LIFETIME,
    HISTORY_SIZE, HISTORY_MASK, NR_MAX_ITER, MAX_TRAIL_LENGTH,
    SCALAR_FIELD_MAX, FIELD_EXCITATION_SIGMA, MERGE_EXCITATION_SCALE, SELFGRAV_PHI_MAX, EXCITATION_MAX_AMPLITUDE,
    COL_PASS, COL_MERGE, COL_BOUNCE,
    BOUND_DESPAWN, BOUND_BOUNCE, BOUND_LOOP,
    TORUS, KLEIN, RP2,
    GPU_SCALAR_GRID, GPU_NR_TOLERANCE,
    GPU_HEATMAP_GRID, GPU_MAX_PARTICLES, GPU_MAX_SPEED_RATIO,
    DESPAWN_MARGIN,
} from '../config.js';

const _PAL = window._PALETTE;

/**
 * Convert a hex color string (#RRGGBB) to normalized [r, g, b] floats (0–1).
 * @param {string} hex
 * @returns {[number, number, number]}
 */
export function paletteRGB(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16 & 0xFF) / 255, (n >> 8 & 0xFF) / 255, (n & 0xFF) / 255];
}

/**
 * Convert HSL (h: 0–360, s: 0–1, l: 0–1) to normalized [r, g, b].
 */
function hslToRGB(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;
    let r, g, b;
    if (h < 60) { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    return [r + m, g + m, b + m];
}

/** Format a float for WGSL (always includes decimal point). */
function wf(v) {
    const s = String(v);
    return s.includes('.') || s.includes('e') ? s : s + '.0';
}

/** Format a vec3f from [r,g,b] array. */
function wv3(rgb) {
    return `vec3f(${wf(rgb[0])}, ${wf(rgb[1])}, ${wf(rgb[2])})`;
}

let _cached = null;

/**
 * Build WGSL constants block from config.js + _PALETTE.
 * Called once at GPU init, result is cached.
 * @returns {string} WGSL const declarations
 */
export function buildWGSLConstants() {
    if (_cached) return _cached;

    // Grid derived constants
    const GRID = GPU_SCALAR_GRID;
    const GRID_SQ = GRID * GRID;
    const GRID_LAST = GRID - 1;
    const HGRID = GPU_HEATMAP_GRID;
    const HGRID_SQ = HGRID * HGRID;

    // Palette colors
    const ext = _PAL.extended;
    const colors = {
        SLATE: paletteRGB(ext.slate),
        RED: paletteRGB(ext.red),
        BLUE: paletteRGB(ext.blue),
        GREEN: paletteRGB(ext.green),
        CYAN: paletteRGB(ext.cyan),
        ORANGE: paletteRGB(ext.orange),
        YELLOW: paletteRGB(ext.yellow),
        ROSE: paletteRGB(ext.rose),
        PURPLE: paletteRGB(ext.purple),
        BROWN: paletteRGB(ext.brown),
        LIME: paletteRGB(ext.lime),
        INDIGO: paletteRGB(ext.indigo),
        MAGENTA: paletteRGB(ext.magenta),
    };

    // Theme colors
    const textLight = paletteRGB(_PAL.light.text);
    const textDark = paletteRGB(_PAL.dark.text);
    const accent = paletteRGB(_PAL.accent);
    const accentLight = paletteRGB(_PAL.accentLight);

    // Spin ring colors: HSL-derived from palette hues (80% sat, 60% lightness)
    const spinCW = hslToRGB(_PAL.spinPos, 0.8, 0.6);
    const spinCCW = hslToRGB(_PAL.spinNeg, 0.8, 0.6);

    _cached = `// ── Auto-generated from config.js + _PALETTE ──

// Physics constants
const SOFTENING: f32 = ${wf(SOFTENING)};
const SOFTENING_SQ: f32 = ${wf(SOFTENING_SQ)};
const BH_SOFTENING: f32 = ${wf(BH_SOFTENING)};
const BH_SOFTENING_SQ: f32 = ${wf(BH_SOFTENING_SQ)};
const INERTIA_K: f32 = ${wf(INERTIA_K)};
const MAG_MOMENT_K: f32 = ${wf(MAG_MOMENT_K)};
const TIDAL_STRENGTH: f32 = ${wf(TIDAL_STRENGTH)};
const YUKAWA_COUPLING: f32 = ${wf(YUKAWA_COUPLING)};
const AXION_COUPLING: f32 = ${wf(AXION_COUPLING)};
const EPSILON: f32 = ${wf(EPSILON)};
const EPSILON_SQ: f32 = ${wf(EPSILON * EPSILON)};
const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717959;
const HALF_PI: f32 = 1.5707963268;
const BOSON_SOFTENING_SQ: f32 = ${wf(BOSON_SOFTENING_SQ)};
const BOSON_MIN_AGE: u32 = ${BOSON_MIN_AGE}u;
const BOSON_MIN_AGE_TIME: f32 = ${wf(BOSON_MIN_AGE * PHYSICS_DT)};
const MAX_QUAD_WG: u32 = ${Math.ceil(GPU_MAX_PARTICLES / 64)}u;
const PHYSICS_DT: f32 = ${wf(PHYSICS_DT)};
const MIN_MASS: f32 = ${wf(MIN_MASS)};
const BH_NAKED_FLOOR: f32 = ${wf(BH_NAKED_FLOOR)};
const ELECTRON_MASS: f32 = ${wf(ELECTRON_MASS)};
const MAX_SPEED_RATIO: f32 = ${wf(GPU_MAX_SPEED_RATIO)};
const PION_DECAY_PROB: f32 = ${wf(PION_DECAY_PROB)};
const CHARGED_PION_DECAY_PROB: f32 = ${wf(CHARGED_PION_DECAY_PROB)};
const BH_THETA: f32 = ${wf(BH_THETA)};
const BH_THETA_SQ: f32 = ${wf(BH_THETA_SQ)};
const VELOCITY_VECTOR_SCALE: f32 = ${wf(VELOCITY_VECTOR_SCALE)};

// Capacity constants
const MAX_PARTICLES: u32 = ${GPU_MAX_PARTICLES}u;
const MAX_PHOTONS: u32 = ${MAX_PHOTONS}u;
const MAX_PIONS: u32 = ${MAX_PIONS}u;
const MAX_GHOSTS: u32 = ${GPU_MAX_PARTICLES}u;
const PHOTON_LIFETIME: f32 = ${wf(PHOTON_LIFETIME)};

// Grid constants
const GRID: u32 = ${GRID}u;
const GRID_SQ: u32 = ${GRID_SQ}u;
const GRID_LAST: u32 = ${GRID_LAST}u;
const SCALAR_FIELD_MAX: f32 = ${wf(SCALAR_FIELD_MAX)};
const FIELD_EXCITATION_SIGMA: f32 = ${wf(FIELD_EXCITATION_SIGMA)};
const MERGE_EXCITATION_SCALE: f32 = ${wf(MERGE_EXCITATION_SCALE)};
const SELFGRAV_PHI_MAX: f32 = ${wf(SELFGRAV_PHI_MAX)};
const EXCITATION_MAX_AMPLITUDE: f32 = ${wf(EXCITATION_MAX_AMPLITUDE)};
const HGRID: u32 = ${HGRID}u;
const HGRID_SQ: u32 = ${HGRID_SQ}u;

// Signal delay / history / trails
const HISTORY_LEN: u32 = ${HISTORY_SIZE}u;
const HISTORY_MASK: u32 = ${HISTORY_MASK}u;
const NR_MAX_ITER: u32 = ${NR_MAX_ITER}u;
const NR_TOLERANCE: f32 = ${wf(GPU_NR_TOLERANCE)};
const HIST_STRIDE: u32 = 6u;       // interleaved: posX, posY, velX, velY, angW, time
const HIST_META_STRIDE: u32 = 4u;  // writeIdx, count, creationTimeBits, _pad
const TRAIL_LEN: u32 = ${MAX_TRAIL_LENGTH}u;

// Boundary mode enums
const BOUND_DESPAWN: u32 = ${BOUND_DESPAWN}u;
const BOUND_BOUNCE: u32 = ${BOUND_BOUNCE}u;
const BOUND_LOOP: u32 = ${BOUND_LOOP}u;

// Topology enums (both naming conventions)
const TOPO_TORUS: u32 = ${TORUS}u;
const TOPO_KLEIN: u32 = ${KLEIN}u;
const TOPO_RP2: u32 = ${RP2}u;
const TORUS: u32 = ${TORUS}u;
const KLEIN: u32 = ${KLEIN}u;
const RP2: u32 = ${RP2}u;

// Collision mode enums
const COL_PASS: u32 = ${COL_PASS}u;
const COL_MERGE: u32 = ${COL_MERGE}u;
const COL_BOUNCE: u32 = ${COL_BOUNCE}u;

// Particle flag bits (standardized FLAG_* prefix)
const FLAG_ALIVE: u32 = 1u;
const FLAG_RETIRED: u32 = 2u;
const FLAG_ANTIMATTER: u32 = 4u;
const FLAG_BH: u32 = 8u;
const FLAG_GHOST: u32 = 16u;
const FLAG_REBORN: u32 = 32u;

// Toggle bit constants (toggles0)
const GRAVITY_BIT: u32 = 1u;
const COULOMB_BIT: u32 = 2u;
const MAGNETIC_BIT: u32 = 4u;
const GRAVITOMAG_BIT: u32 = 8u;
const ONE_PN_BIT: u32 = 16u;
const RELATIVITY_BIT: u32 = 32u;
const SPIN_ORBIT_BIT: u32 = 64u;
const RADIATION_BIT: u32 = 128u;
const BLACK_HOLE_BIT: u32 = 256u;
const DISINTEGRATION_BIT: u32 = 512u;
const EXPANSION_BIT: u32 = 1024u;
const YUKAWA_BIT: u32 = 2048u;
const HIGGS_BIT: u32 = 4096u;
const AXION_BIT: u32 = 8192u;
const BARNES_HUT_BIT: u32 = 16384u;
const BOSON_GRAV_BIT: u32 = 32768u;

// Toggle bit constants (toggles1)
const FIELD_GRAV_BIT: u32 = 1u;
const HERTZ_BOUNCE_BIT: u32 = 2u;

// Barnes-Hut tree constants
// NOTE: QT_CAPACITY intentionally NOT included — GPU uses 1 (lock-free), CPU uses 4.
const MAX_DEPTH: u32 = 48u;

// Boundary
const DESPAWN_MARGIN: f32 = ${wf(DESPAWN_MARGIN)};

// Disintegration / pair production
const MAX_DISINT_EVENTS: u32 = 64u;
const MAX_PAIR_EVENTS: u32 = 32u;

// Palette colors
const COLOR_SLATE: vec3f = ${wv3(colors.SLATE)};
const COLOR_RED: vec3f = ${wv3(colors.RED)};
const COLOR_BLUE: vec3f = ${wv3(colors.BLUE)};
const COLOR_GREEN: vec3f = ${wv3(colors.GREEN)};
const COLOR_CYAN: vec3f = ${wv3(colors.CYAN)};
const COLOR_ORANGE: vec3f = ${wv3(colors.ORANGE)};
const COLOR_YELLOW: vec3f = ${wv3(colors.YELLOW)};
const COLOR_ROSE: vec3f = ${wv3(colors.ROSE)};
const COLOR_PURPLE: vec3f = ${wv3(colors.PURPLE)};
const COLOR_BROWN: vec3f = ${wv3(colors.BROWN)};
const COLOR_LIME: vec3f = ${wv3(colors.LIME)};
const COLOR_INDIGO: vec3f = ${wv3(colors.INDIGO)};
const COLOR_MAGENTA: vec3f = ${wv3(colors.MAGENTA)};

// Theme colors
const COLOR_TEXT_LIGHT: vec3f = ${wv3(textLight)};
const COLOR_TEXT_DARK: vec3f = ${wv3(textDark)};
const COLOR_ACCENT: vec3f = ${wv3(accent)};
const COLOR_ACCENT_LIGHT: vec3f = ${wv3(accentLight)};

// Spin ring colors (HSL-derived from palette hues, 80% sat, 60% lightness)
const COLOR_SPIN_CW: vec3f = ${wv3(spinCW)};
const COLOR_SPIN_CCW: vec3f = ${wv3(spinCCW)};

`;
    return _cached;
}

// ── JS-side toggle bit constants (must match WGSL block above) ──

// toggles0
export const GRAVITY_BIT     = 1;
export const COULOMB_BIT     = 2;
export const MAGNETIC_BIT    = 4;
export const GRAVITOMAG_BIT  = 8;
export const ONE_PN_BIT      = 16;
export const RELATIVITY_BIT  = 32;
export const SPIN_ORBIT_BIT  = 64;
export const RADIATION_BIT   = 128;
export const BLACK_HOLE_BIT  = 256;
export const DISINTEGRATION_BIT = 512;
export const EXPANSION_BIT   = 1024;
export const YUKAWA_BIT      = 2048;
export const HIGGS_BIT       = 4096;
export const AXION_BIT       = 8192;
export const BARNES_HUT_BIT  = 16384;
export const BOSON_GRAV_BIT  = 32768;

// toggles1
export const FIELD_GRAV_BIT_T1 = 1;
export const HERTZ_BOUNCE_BIT_T1 = 2;

// History buffer layout
export const HIST_STRIDE = 6;
export const HIST_META_STRIDE = 4;
