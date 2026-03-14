# GPU Constants & Colors Centralization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~50+ hardcoded physics constants, toggle/flag bits, and palette colors across 25 WGSL shader files and 4 GPU JS files by generating a shared WGSL constants block from `config.js` + `_PALETTE` at init time.

**Architecture:** A new module `src/gpu/gpu-constants.js` builds a WGSL `const` declaration string from config.js exports and `_PALETTE` hex→RGB conversions. This string is prepended to all shaders before compilation via `fetchShader()`. GPU JS files replace local constants with config.js imports and palette lookups.

**Tech Stack:** Vanilla JS (ES6 modules), WebGPU/WGSL, no build step or bundler.

**Spec:** `docs/superpowers/specs/2026-03-14-gpu-constants-centralization-design.md`

---

## Chunk 1: Foundation — config.js + gpu-constants.js + fetchShader()

### Task 1: Add new constants to config.js

**Files:**
- Modify: `src/config.js`

- [ ] **Step 1: Add GPU-specific constants after existing constants**

Add after the `// ── Signal Delay ──` section (after line 114):

```js
// ── GPU-Specific ──
export const SELFGRAV_GRID = 8;               // CPU self-gravity coarse grid
export const GPU_SCALAR_GRID = 64;            // GPU scalar field grid resolution (tunable to 128+)
export const GPU_SELFGRAV_GRID = 8;           // GPU self-gravity coarse grid (tunable to 16+)
export const GPU_NR_TOLERANCE = 1e-5;         // GPU Newton-Raphson tolerance (f32 precision limit)
export const GPU_HEATMAP_GRID = 64;           // GPU heatmap overlay resolution (tunable to 128)
export const GPU_MAX_PARTICLES = 4096;        // GPU buffer pre-allocation limit
export const GPU_MAX_SPEED_RATIO = 0.9999;    // GPU speed cap (f32 needs tighter bound than CPU 0.99)
```

- [ ] **Step 2: Commit**

```bash
git add src/config.js
git commit -m "feat: add GPU-specific and SELFGRAV_GRID constants to config.js"
```

---

### Task 2: Replace hardcoded 8 in scalar-field.js with SELFGRAV_GRID

**Files:**
- Modify: `src/scalar-field.js:1` (import line) and `src/scalar-field.js:31` (hardcoded value)

- [ ] **Step 1: Add SELFGRAV_GRID to imports**

In the import statement at the top of `src/scalar-field.js`, add `SELFGRAV_GRID` to the existing config.js import.

- [ ] **Step 2: Replace `gridSize >> 3` with SELFGRAV_GRID**

At line 31, change:
```js
const sgGrid = gridSize >> 3; // 8 for SCALAR_GRID=64 (O(SG⁴)=4096 vs 65536)
```
to:
```js
const sgGrid = SELFGRAV_GRID;
```

- [ ] **Step 3: Verify simulation loads and scalar fields work**

Serve the project and test: enable Higgs field, check field overlay renders, check self-gravity works (toggle Field Gravity on with Gravity).

- [ ] **Step 4: Commit**

```bash
git add src/scalar-field.js
git commit -m "refactor: use SELFGRAV_GRID constant instead of hardcoded gridSize >> 3"
```

---

### Task 3: Create gpu-constants.js

**Files:**
- Create: `src/gpu/gpu-constants.js`

- [ ] **Step 1: Write gpu-constants.js**

```js
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
    YUKAWA_COUPLING, EPSILON,
    PI, TWO_PI,
    BOSON_SOFTENING_SQ, BOSON_MIN_AGE, PHYSICS_DT,
    LL_FORCE_CLAMP, MIN_MASS, BH_NAKED_FLOOR, ELECTRON_MASS,
    PION_DECAY_PROB, CHARGED_PION_DECAY_PROB,
    BH_THETA, BH_THETA_SQ,
    VELOCITY_VECTOR_SCALE,
    MAX_PHOTONS, MAX_PIONS, PHOTON_LIFETIME,
    HISTORY_SIZE, HISTORY_MASK, NR_MAX_ITER, MAX_TRAIL_LENGTH,
    SCALAR_FIELD_MAX, FIELD_EXCITATION_SIGMA, MERGE_EXCITATION_SCALE,
    COL_PASS, COL_MERGE, COL_BOUNCE,
    BOUND_DESPAWN, BOUND_BOUNCE, BOUND_LOOP,
    TORUS, KLEIN, RP2,
    GPU_SCALAR_GRID, GPU_SELFGRAV_GRID, GPU_NR_TOLERANCE,
    GPU_HEATMAP_GRID, GPU_MAX_PARTICLES, GPU_MAX_SPEED_RATIO,
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
    const COARSE = GPU_SELFGRAV_GRID;
    const COARSE_SQ = COARSE * COARSE;
    const SG_RATIO = GRID / COARSE;
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
const YUKAWA_COUPLING_DEFAULT: f32 = ${wf(YUKAWA_COUPLING)};
const EPSILON: f32 = ${wf(EPSILON)};
const EPSILON_SQ: f32 = ${wf(EPSILON * EPSILON)};
const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717959;
const HALF_PI: f32 = 1.5707963268;
const BOSON_SOFTENING_SQ: f32 = ${wf(BOSON_SOFTENING_SQ)};
const BOSON_MIN_AGE: u32 = ${BOSON_MIN_AGE}u;
const BOSON_MIN_AGE_TIME: f32 = ${wf(BOSON_MIN_AGE * PHYSICS_DT)};
const LL_FORCE_CLAMP: f32 = ${wf(LL_FORCE_CLAMP)};
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
const COARSE: u32 = ${COARSE}u;
const COARSE_SQ: u32 = ${COARSE_SQ}u;
const SG_RATIO: u32 = ${SG_RATIO}u;
const SCALAR_FIELD_MAX: f32 = ${wf(SCALAR_FIELD_MAX)};
const FIELD_EXCITATION_SIGMA: f32 = ${wf(FIELD_EXCITATION_SIGMA)};
const MERGE_EXCITATION_SCALE: f32 = ${wf(MERGE_EXCITATION_SCALE)};
const HGRID: u32 = ${HGRID}u;
const HGRID_SQ: u32 = ${HGRID_SQ}u;

// Signal delay / history / trails
const HISTORY_LEN: u32 = ${HISTORY_SIZE}u;
const HISTORY_MASK: u32 = ${HISTORY_MASK}u;
const NR_MAX_ITER: u32 = ${NR_MAX_ITER}u;
const NR_TOLERANCE: f32 = ${wf(GPU_NR_TOLERANCE)};
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

// Particle flag bits (canonical names + aliases used by standalone shaders)
const FLAG_ALIVE: u32 = 1u;
const FLAG_RETIRED: u32 = 2u;
const FLAG_ANTIMATTER: u32 = 4u;
const FLAG_BH: u32 = 8u;
const FLAG_GHOST: u32 = 16u;
const ALIVE_BIT: u32 = 1u;          // alias for standalone shaders
const ANTIMATTER_BIT: u32 = 4u;     // alias for standalone shaders

// Toggle bit constants (toggles0)
const GRAVITY_BIT: u32 = 1u;
const COULOMB_BIT: u32 = 2u;
const MAGNETIC_BIT: u32 = 4u;
const GRAVITOMAG_BIT: u32 = 8u;
const ONE_PN_BIT: u32 = 16u;
const ONEPN_BIT: u32 = 16u;         // alias used by forces-tree.wgsl
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
// QT_CAPACITY stays local in tree-build.wgsl.
const MAX_DEPTH: u32 = 48u;

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
const COLOR_ACCENT: vec3f = ${wv3(accent)};
const COLOR_ACCENT_LIGHT: vec3f = ${wv3(accentLight)};

// Spin ring colors (HSL-derived from palette hues, 80% sat, 60% lightness)
const COLOR_SPIN_CW: vec3f = ${wv3(spinCW)};
const COLOR_SPIN_CCW: vec3f = ${wv3(spinCCW)};

`;
    return _cached;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/gpu/gpu-constants.js
git commit -m "feat: add gpu-constants.js — WGSL constants generator from config.js + _PALETTE"
```

---

### Task 4: Update fetchShader() and pipeline creators in gpu-pipelines.js

**Files:**
- Modify: `src/gpu/gpu-pipelines.js`

- [ ] **Step 1: Add prepend parameter to fetchShader()**

Change the `fetchShader` function (lines 15–18) from:
```js
async function fetchShader(filename) {
    const resp = await fetch(`src/gpu/shaders/${filename}?v=${SHADER_VERSION}`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    return resp.text();
}
```
to:
```js
async function fetchShader(filename, prepend = '') {
    const resp = await fetch(`src/gpu/shaders/${filename}?v=${SHADER_VERSION}`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    const source = await resp.text();
    return prepend ? prepend + '\n' + source : source;
}
```

- [ ] **Step 2: Add `wgslConstants` parameter to all pipeline creation functions**

Every `export async function create*Pipeline(device, ...)` needs a `wgslConstants` parameter added. The constants block is then:

- For **Phase 2** (`createPhase2Pipelines`): prepended before `common.wgsl` in the `commonWGSL` variable.
  Change line 26 from `const commonWGSL = await fetchShader('common.wgsl');` to `const commonWGSL = wgslConstants + '\n' + await fetchShader('common.wgsl');`
  Function signature: `createPhase2Pipelines(device, wgslConstants)`

- For **standalone compute shaders** (Phase 3, Phase 4, etc.): passed as `prepend` to `fetchShader()`.
  Example in `createPhase4Pipelines`: `await fetchShader('bosons.wgsl', wgslConstants)` instead of `await fetchShader('bosons.wgsl')`.

- For **field shaders** (functions that load `field-common.wgsl`): prepended before `field-common.wgsl`.
  Example: `const fieldCommon = wgslConstants + '\n' + await fetchShader('field-common.wgsl');`

- For **render shaders**: passed as `prepend` to `fetchShader()`.
  Example: `await fetchShader('update-colors.wgsl', wgslConstants)`

The full list of functions to update (add `wgslConstants` param):
1. `createPhase2Pipelines(device)` → `createPhase2Pipelines(device, wgslConstants)`
2. `createTreeBuildPipelines(device)` → `createTreeBuildPipelines(device, wgslConstants)`
3. `createTreeForcePipeline(device)` → `createTreeForcePipeline(device, wgslConstants)`
4. `createCollisionPipelines(device)` → `createCollisionPipelines(device, wgslConstants)`
5. `createDeadGCPipeline(device)` → `createDeadGCPipeline(device, wgslConstants)`
6. `createPhase4Pipelines(device)` → `createPhase4Pipelines(device, wgslConstants)`
7. `createBosonRenderPipelines(device, format, isLight)` → `createBosonRenderPipelines(device, format, isLight, wgslConstants)`
8. `createGhostGenPipeline(device)` → `createGhostGenPipeline(device, wgslConstants)`
9. `createFieldDepositPipelines(device)` → `createFieldDepositPipelines(device, wgslConstants)`
10. `createFieldEvolvePipelines(device)` → `createFieldEvolvePipelines(device, wgslConstants)`
11. `createFieldForcesPipelines(device)` → `createFieldForcesPipelines(device, wgslConstants)`
12. `createFieldSelfGravPipelines(device)` → `createFieldSelfGravPipelines(device, wgslConstants)`
13. `createFieldExcitationPipeline(device)` → `createFieldExcitationPipeline(device, wgslConstants)`
14. `createHeatmapPipelines(device)` → `createHeatmapPipelines(device, wgslConstants)`
15. `createExpansionPipeline(device)` → `createExpansionPipeline(device, wgslConstants)`
16. `createDisintegrationPipeline(device)` → `createDisintegrationPipeline(device, wgslConstants)`
17. `createPairProductionPipeline(device)` → `createPairProductionPipeline(device, wgslConstants)`
18. `createUpdateColorsPipeline(device)` → `createUpdateColorsPipeline(device, wgslConstants)`
19. `createSpinRenderPipeline(device, format, isLight)` → `createSpinRenderPipeline(device, format, isLight, wgslConstants)`
20. `createTrailRecordPipeline(device)` → `createTrailRecordPipeline(device, wgslConstants)`
21. `createTrailRenderPipeline(device, format, isLight)` → `createTrailRenderPipeline(device, format, isLight, wgslConstants)`
22. `createArrowRenderPipeline(device, format, isLight)` → `createArrowRenderPipeline(device, format, isLight, wgslConstants)`
23. `createFieldRenderPipeline(device, format, isLight)` → `createFieldRenderPipeline(device, format, isLight, wgslConstants)`
24. `createHeatmapRenderPipeline(device, format, isLight)` → `createHeatmapRenderPipeline(device, format, isLight, wgslConstants)`
25. `createOnePN Pipeline(device)` → `createOnePNPipeline(device, wgslConstants)` (if exists)
26. `createHistoryPipeline(device)` → `createHistoryPipeline(device, wgslConstants)` (if exists)

Inside each function, use `wgslConstants` when loading shaders:
- Phase 2 prepend: `const commonWGSL = wgslConstants + '\n' + await fetchShader('common.wgsl');`
- Field prepend: `const fieldCommonWGSL = wgslConstants + '\n' + await fetchShader('field-common.wgsl');`
- Standalone: `await fetchShader('bosons.wgsl', wgslConstants)`
- Render: `await fetchShader('update-colors.wgsl', wgslConstants)`

**DO NOT commit yet** — this will cause WGSL duplicate const errors until shader cleanup (Tasks 6–9) is done. Commit together with Tasks 5–9 in Task 8 Step 16.

---

### Task 5: Update gpu-physics.js and gpu-renderer.js to pass wgslConstants

**Files:**
- Modify: `src/gpu/gpu-physics.js` (callers of pipeline creation functions)
- Modify: `src/gpu/gpu-renderer.js` (callers of render pipeline creation functions)

- [ ] **Step 1: Import buildWGSLConstants in gpu-physics.js**

Add at top of gpu-physics.js:
```js
import { buildWGSLConstants } from './gpu-constants.js';
```

At the start of the GPU init method (where pipelines are created), call once:
```js
const wgslConstants = buildWGSLConstants();
```

Then pass `wgslConstants` to every `create*Pipeline()` call in gpu-physics.js.

- [ ] **Step 2: Import buildWGSLConstants in gpu-renderer.js**

Add at top of gpu-renderer.js:
```js
import { buildWGSLConstants } from './gpu-constants.js';
```

At the start of renderer init (or lazy init methods like `initFieldOverlay`), get the cached constants:
```js
const wgslConstants = buildWGSLConstants();
```

Pass `wgslConstants` to every render pipeline creation call.

**DO NOT commit or test yet** — Tasks 4–5 (pipeline wiring) and Tasks 6–9 (shader constant removal) must land together. The combined commit is in Task 8 Step 16.

---

## Chunk 2: Remove duplicate constants from shared headers + standalone compute shaders

### Task 6: Clean common.wgsl — remove constants now provided by generated block

**Files:**
- Modify: `src/gpu/shaders/common.wgsl`

- [ ] **Step 1: Remove physics constants block (lines 82–93)**

Remove these lines from common.wgsl:
```wgsl
// Physics constants (from config.js)
const SOFTENING: f32 = 8.0;
const SOFTENING_SQ: f32 = 64.0;
const BH_SOFTENING: f32 = 4.0;
const BH_SOFTENING_SQ: f32 = 16.0;
const INERTIA_K: f32 = 0.4;
const MAG_MOMENT_K: f32 = 0.2;
const TIDAL_STRENGTH: f32 = 0.3;
const YUKAWA_COUPLING_DEFAULT: f32 = 14.0;
const EPSILON: f32 = 1e-9;
const EPSILON_SQ: f32 = 1e-18;
const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717959;
```

- [ ] **Step 2: Remove toggle bit constants (lines 43–62)**

Remove:
```wgsl
// Toggle bit constants (toggles0)
const GRAVITY_BIT: u32       = 1u;
const COULOMB_BIT: u32       = 2u;
const MAGNETIC_BIT: u32      = 4u;
const GRAVITOMAG_BIT: u32    = 8u;
const ONE_PN_BIT: u32        = 16u;
const RELATIVITY_BIT: u32    = 32u;
const SPIN_ORBIT_BIT: u32    = 64u;
const RADIATION_BIT: u32     = 128u;
const BLACK_HOLE_BIT: u32    = 256u;
const DISINTEGRATION_BIT: u32 = 512u;
const EXPANSION_BIT: u32     = 1024u;
const YUKAWA_BIT: u32        = 2048u;
const HIGGS_BIT: u32         = 4096u;
const AXION_BIT: u32         = 8192u;
const BARNES_HUT_BIT: u32    = 16384u;
const BOSON_GRAV_BIT: u32    = 32768u;

// Toggle bit constants (toggles1)
const FIELD_GRAV_BIT: u32    = 1u;
const HERTZ_BOUNCE_BIT: u32  = 2u;
```

- [ ] **Step 3: Remove particle flag bits (lines 64–69)**

Remove:
```wgsl
// Particle flag bits
const FLAG_ALIVE: u32    = 1u;
const FLAG_RETIRED: u32  = 2u;
const FLAG_ANTIMATTER: u32 = 4u;
const FLAG_BH: u32       = 8u;
const FLAG_GHOST: u32    = 16u;
```

- [ ] **Step 4: Remove boundary and topology enums (lines 71–79)**

Remove:
```wgsl
// Boundary modes
const BOUND_DESPAWN: u32 = 0u;
const BOUND_BOUNCE: u32  = 1u;
const BOUND_LOOP: u32    = 2u;

// Topology modes
const TOPO_TORUS: u32 = 0u;
const TOPO_KLEIN: u32 = 1u;
const TOPO_RP2: u32   = 2u;
```

Keep: `SimUniforms` struct, `hasToggle0`/`hasToggle1` helpers, all packed buffer structs (`AllForces`, `ParticleDerived`, `ParticleState`, `ParticleAux`, `RadiationState`, `Photon`, `Pion`), `torusMinImage`, `fullMinImage`.

---

### Task 7: Clean field-common.wgsl — remove constants now provided by generated block

**Files:**
- Modify: `src/gpu/shaders/field-common.wgsl`

- [ ] **Step 1: Remove grid/field constants and enums (lines 5–24)**

Remove:
```wgsl
const GRID: u32 = 64u;
const GRID_SQ: u32 = 4096u;
const GRID_LAST: u32 = 63u;
const COARSE: u32 = 8u;
const COARSE_SQ: u32 = 64u;
const SG_RATIO: u32 = 8u;
const SCALAR_FIELD_MAX: f32 = 2.0;
const FIELD_EXCITATION_SIGMA: f32 = 2.0;
const MERGE_EXCITATION_SCALE: f32 = 0.5;
const EPSILON: f32 = 1e-9;

// Boundary mode enums (must match config.js)
const BOUND_DESPAWN: u32 = 0u;
const BOUND_BOUNCE: u32 = 1u;
const BOUND_LOOP: u32 = 2u;

// Topology enums
const TORUS: u32 = 0u;
const KLEIN: u32 = 1u;
const RP2: u32 = 2u;
```

Keep: `FieldUniforms` struct, `PQSResult` struct, `pqsWeights()`, `nbIndex()`, `isInterior()`.

---

### Task 8: Clean standalone compute shaders — remove duplicate constants

**Files:** All 15 standalone compute shaders listed below.

For each file, remove the local `const` lines identified in the audit. The generated block now provides all of these. Keep shader-specific constants (PCG hash functions, node accessor functions, topology helper functions, etc.).

- [ ] **Step 1: Clean bosons.wgsl**

Remove lines 6–12 (MAX_PHOTONS, MAX_PIONS, BOSON_SOFTENING_SQ, BOSON_MIN_AGE_TIME, BOSON_MIN_AGE, PHOTON_LIFETIME, EPSILON), lines 17–21 (PION_DECAY_PROB, CHARGED_PION_DECAY_PROB, ELECTRON_MASS, MAX_SPEED_RATIO, MAX_PARTICLES), lines 24–25 (ALIVE_BIT, ANTIMATTER_BIT). Keep PCG hash/RNG functions.

- [ ] **Step 2: Clean boson-tree.wgsl**

Remove lines 8–13 (MAX_PHOTONS, MAX_PIONS, BOSON_SOFTENING_SQ, BH_THETA_SQ, EPSILON, MAX_DEPTH), line 112 (FLAG_ALIVE). Keep node accessor functions.

- [ ] **Step 3: Clean radiation.wgsl**

Remove lines 12–20 (ALIVE_BIT, LL_FORCE_CLAMP, MIN_MASS, EPSILON, MAX_PHOTONS, MAX_PIONS, MAX_SPEED_RATIO, INERTIA_K, BH_NAKED_FLOOR), lines 33–37 (COULOMB_BIT, RELATIVITY_BIT, RADIATION_BIT, BLACK_HOLE_BIT, YUKAWA_BIT). Keep PCG hash/RNG functions.

- [ ] **Step 4: Clean collision.wgsl**

Remove lines 7–16 (NONE, MAX_STACK, EPSILON, INERTIA_K, FLAG_ALIVE/RETIRED/ANTIMATTER/GHOST, COL_MERGE, COL_BOUNCE), lines 19–20 (MERGE_ANNIHILATION, MERGE_INELASTIC), lines 23–26 (TOPO_TORUS/KLEIN/RP2, BOUND_LOOP), line 142 (NODE_STRIDE), line 425 (TILE_SIZE_COL). Keep fullMinImageCol topology function.

Note: NONE (-1), MAX_STACK (48), MERGE_ANNIHILATION (0), MERGE_INELASTIC (1), NODE_STRIDE (20), TILE_SIZE_COL (64) are shader-specific — either add them to the generated block if reused elsewhere, or keep them local. Since they're only used in collision.wgsl, keep them local.

- [ ] **Step 5: Clean forces-tree.wgsl**

Remove lines 6–11 (NONE, MAX_STACK, EPSILON, FLAG_ALIVE/RETIRED/GHOST), lines 13–20 (toggle bits), lines 23–25 (MAG_MOMENT_K, INERTIA_K, TIDAL_STRENGTH), lines 28–31 (topology constants), line 95 (NODE_STRIDE), lines 226–227 (ABERRATION_CLAMP_MIN/MAX). Keep topology helper functions, node accessors.

Note: NONE, MAX_STACK, NODE_STRIDE, ABERRATION_CLAMP_MIN/MAX are shader-specific — keep local.

- [ ] **Step 6: Clean heatmap.wgsl**

Remove lines 59–60 (HGRID, HGRID_SQ), lines 63–67 (HISTORY_LEN, HISTORY_MASK, NR_MAX_ITER, NR_TOLERANCE, EPSILON), lines 70–72 (TOPO_TORUS/KLEIN/RP2). Keep hmMinImage, getRetardedPosition, yukawaCutoffSq functions.

- [ ] **Step 7: Clean history.wgsl**

Remove lines 7–11 (HISTORY_LEN, HISTORY_MASK, NR_MAX_ITER, NR_TOLERANCE, EPSILON), line 56 (ALIVE_BIT), lines 101–103 (TOPO_TORUS/KLEIN/RP2). Keep minImageDisp, getDelayedStateGPU functions.

- [ ] **Step 8: Clean trails.wgsl**

Remove line 6 (TRAIL_LEN), line 23 (ALIVE_BIT).

- [ ] **Step 9: Clean ghost-gen.wgsl**

Remove lines 6–8 (TOPO_TORUS/KLEIN/RP2), lines 12–13 (FLAG_ALIVE, FLAG_GHOST), line 96 (MAX_GHOSTS). Keep appendGhost, makeGhostState functions.

- [ ] **Step 10: Clean dead-gc.wgsl**

Remove lines 5–6 (FLAG_ALIVE, FLAG_RETIRED).

- [ ] **Step 11: Clean tree-build.wgsl**

Remove lines 10 (MAX_DEPTH), lines 18–19 (FLAG_ALIVE, FLAG_GHOST). Keep QT_CAPACITY (GPU uses 1, different from CPU's 4), NONE, LOCK_BIT, FP_SCALE/FP_INV_SCALE (shader-specific fixed-point), NODE_STRIDE, node accessor functions.

- [ ] **Step 12: Clean onePN.wgsl**

Remove line 14 (ALIVE_BIT), lines 75–78 (GRAVITOMAG_BIT, MAGNETIC_BIT, YUKAWA_BIT, RELATIVITY_BIT), lines 80–83 (EPSILON, BOUND_LOOP, TOPO_TORUS, TOPO_KLEIN).

- [ ] **Step 13: Clean disintegration.wgsl**

Remove line 6 (EPSILON), line 76 (MAX_DISINT_EVENTS). Keep tidalStrength if it differs from config or is shader-specific; otherwise remove. Keep DISINT_FRAGMENT/DISINT_TRANSFER (shader-specific enums).

- [ ] **Step 14: Clean pair-production.wgsl**

Remove line 70 (MAX_PAIR_EVENTS). Keep pcgHash, randomFloat functions.

- [ ] **Step 15: expansion.wgsl**

No constants to remove — skip.

- [ ] **Step 16: Commit all shader changes + pipeline wiring from Task 5**

```bash
git add src/gpu/shaders/*.wgsl src/gpu/gpu-pipelines.js src/gpu/gpu-physics.js src/gpu/gpu-renderer.js
git commit -m "refactor: centralize GPU constants — remove duplicates from 17 shaders, wire generated block"
```

- [ ] **Step 17: Verify simulation loads and all GPU features work**

Serve the project and test:
1. Basic simulation runs (particles, forces, collisions)
2. Barnes-Hut toggle works
3. Scalar fields (Higgs + Axion) — field overlay renders correctly
4. Radiation — photons and pions emit and render
5. Signal delay — enable Relativity, check history works
6. Heatmap overlay renders
7. Trails render
8. Force arrows render
9. Spin rings render
10. All boundary modes (despawn/bounce/loop) work
11. All topology modes (torus/klein/rp2) work

---

## Chunk 3: Replace hardcoded colors in render shaders

### Task 9: Replace hardcoded RGB in render shaders with COLOR_* constants

**Files:** 5 render shader files

- [ ] **Step 1: Clean update-colors.wgsl**

Remove lines 30–31 (ALIVE_BIT, ANTIMATTER_BIT).

Replace lines 33–46 (SLATE_R/G/B, POS_R/G/B, NEG_R/G/B individual constants) with usage of `COLOR_SLATE`, `COLOR_RED`, `COLOR_BLUE` from the generated block. Update the main function to use `COLOR_SLATE.r` instead of `SLATE_R`, etc.

- [ ] **Step 2: Clean boson-render.wgsl**

Replace hardcoded photon colors at lines 91–96:
- Graviton: `vec4f(0.753, 0.314, 0.282, alpha)` → `vec4f(COLOR_RED, alpha)`
- EM photon: `vec4f(0.800, 0.659, 0.298, alpha)` → `vec4f(COLOR_YELLOW, alpha)`

Replace hardcoded pion color at line 133:
- `vec4f(0.31, 0.6, 0.47, pionAlpha)` → `vec4f(COLOR_GREEN, pionAlpha)`

Replace hardcoded `/ 256.0` at line ~76 with `/ PHOTON_LIFETIME`.

- [ ] **Step 3: Clean spin-render.wgsl**

Remove line 53 (ALIVE_BIT), lines 54–58 (ARC_SEGMENTS, PI, TWO_PI, HALF_PI, MIN_ANGVEL — PI/TWO_PI/HALF_PI now from generated block; keep ARC_SEGMENTS and MIN_ANGVEL as shader-specific).

Replace lines 62–63:
- `const COLOR_CW_RGB: vec3f = vec3f(0.278, 0.878, 0.780);` → use `COLOR_SPIN_CW` from generated block
- `const COLOR_CCW_RGB: vec3f = vec3f(0.922, 0.678, 0.278);` → use `COLOR_SPIN_CCW` from generated block

Update references from `COLOR_CW_RGB`→`COLOR_SPIN_CW` and `COLOR_CCW_RGB`→`COLOR_SPIN_CCW` throughout the shader.

- [ ] **Step 4: Clean heatmap-render.wgsl**

Remove line 40 (HGRID — now from generated block).

Replace hardcoded color values in the fragment shader:
- Lines 91–93 (gravity/slate): `intensity * 0.541` → `intensity * COLOR_SLATE.r`, etc.
- Lines 105–107 (electric red/blue): `mapped * 0.75` → `mapped * COLOR_RED.r`, `abs(mapped) * 0.65` → `abs(mapped) * COLOR_BLUE.b`
- Lines 120–122 (Yukawa/green): `intensity * 0.596` → `intensity * COLOR_GREEN.g`, etc.

- [ ] **Step 5: Clean arrow-render.wgsl**

Remove line 84 (ALIVE_BIT), line 90 (VELOCITY_VECTOR_SCALE — now from generated block). Keep SHAFT_HALF_W, HEAD_HALF_W, HEAD_LEN (shader-specific arrow geometry).

- [ ] **Step 6: Clean trail-render.wgsl**

Remove line 44 (ALIVE_BIT).

- [ ] **Step 7: Clean hit-test.wgsl**

Remove line 49 (ALIVE_BIT).

- [ ] **Step 8: Clean particle.wgsl**

Keep DARK_QUAD_SCALE (shader-specific). No other changes needed — particle.wgsl gets FLAG_* from the generated block but doesn't define them locally (it relies on common.wgsl prepend already). Verify no local flag definitions exist.

- [ ] **Step 9: Clean field-render.wgsl**

Verify GRID is used but not locally defined (comes via field-common.wgsl chain). No changes needed.

- [ ] **Step 10: Commit**

```bash
git add src/gpu/shaders/*.wgsl
git commit -m "refactor: replace hardcoded RGB colors in render shaders with palette constants"
```

- [ ] **Step 11: Visual verification**

Serve and visually confirm:
1. Particle colors: neutral=slate, positive=red gradient, negative=blue gradient
2. Photon colors: EM=yellow, graviton=red, correct alpha fade
3. Pion color: green
4. Spin rings: CW=cyan-ish, CCW=orange-ish
5. Heatmap: gravity=slate, electric=red(+)/blue(-), Yukawa=green
6. Force arrows: 11 distinct colors matching CPU renderer
7. Both light and dark themes

---

## Chunk 4: Replace hardcoded values in GPU JS files

### Task 10: Replace constants in gpu-buffers.js with config.js imports

**Files:**
- Modify: `src/gpu/gpu-buffers.js`

- [ ] **Step 1: Replace local constants with config.js imports**

Add import at top:
```js
import {
    HISTORY_SIZE, MAX_PHOTONS, MAX_PIONS, MAX_TRAIL_LENGTH,
    GPU_SCALAR_GRID, GPU_SELFGRAV_GRID,
} from '../config.js';
```

Replace local constants:
- Line 16: `const HISTORY_LEN = 256;` → `const HISTORY_LEN = HISTORY_SIZE;`
- Lines 19–20: `const MAX_PHOTONS = 1024; const MAX_PIONS = 256;` → remove (use imports directly)
- Line 369: `const FIELD_GRID_RES = 64;` → `const FIELD_GRID_RES = GPU_SCALAR_GRID;`
- Line 371: `const COARSE_RES = 8;` → `const COARSE_RES = GPU_SELFGRAV_GRID;`
- Line 499: `const TRAIL_LEN = 256;` → `const TRAIL_LEN = MAX_TRAIL_LENGTH;`

Update any references to the old local names if they were used in exports or other places.

- [ ] **Step 2: Commit**

```bash
git add src/gpu/gpu-buffers.js
git commit -m "refactor: replace hardcoded constants in gpu-buffers.js with config.js imports"
```

---

### Task 11: Replace constants in gpu-physics.js with config.js imports

**Files:**
- Modify: `src/gpu/gpu-physics.js`

- [ ] **Step 1: Replace local constants with config.js imports**

Add to existing config.js import (or create one):
```js
import {
    HISTORY_STRIDE, MAX_PHOTONS, MAX_PIONS,
    GPU_MAX_PARTICLES,
    COL_MERGE, COL_BOUNCE, BOUND_LOOP,
    COL_NAMES, BOUND_NAMES, TOPO_NAMES,
} from '../config.js';
```

Replace local constants:
- Lines 42–45: `const MAX_PARTICLES = 4096;` → `const MAX_PARTICLES = GPU_MAX_PARTICLES;` (keep local alias for readability). Remove `HISTORY_STRIDE`, `MAX_PHOTONS`, `MAX_PIONS` local declarations — use imports.
- Lines 2988–2998: Remove `FLAG_ALIVE`, `FLAG_RETIRED`, `FLAG_ANTIMATTER`, `BOUND_LOOP`, `COL_MERGE`, `COL_BOUNCE`, `COL_NAMES`, `BOUND_NAMES`, `TOPO_NAMES` — use imports.

Note: `COL_NAMES`, `BOUND_NAMES`, `TOPO_NAMES` are already exported from config.js.

- [ ] **Step 2: Commit**

```bash
git add src/gpu/gpu-physics.js
git commit -m "refactor: replace hardcoded constants in gpu-physics.js with config.js imports"
```

---

### Task 12: Replace hardcoded colors in gpu-renderer.js with palette lookups

**Files:**
- Modify: `src/gpu/gpu-renderer.js`

- [ ] **Step 1: Import paletteRGB and config constants**

Add at top:
```js
import { paletteRGB } from './gpu-constants.js';
import { HEATMAP_SENSITIVITY, HEATMAP_MAX_ALPHA } from '../config.js';
```

- [ ] **Step 2: Replace FORCE_COLORS with palette-derived values**

Replace the static `FORCE_COLORS` array (lines 639–651):
```js
static FORCE_COLORS = (() => {
    const _PAL = window._PALETTE;
    const rgb = paletteRGB;
    return [
        rgb(_PAL.extended.red),       // 0: gravity
        rgb(_PAL.extended.blue),      // 1: coulomb
        rgb(_PAL.extended.cyan),      // 2: magnetic
        rgb(_PAL.extended.rose),      // 3: gravitomag
        rgb(_PAL.extended.orange),    // 4: 1pn
        rgb(_PAL.extended.purple),    // 5: spinCurv
        rgb(_PAL.extended.yellow),    // 6: radiation
        rgb(_PAL.extended.green),     // 7: yukawa
        rgb(_PAL.extended.brown),     // 8: external
        rgb(_PAL.extended.lime),      // 9: higgs
        rgb(_PAL.extended.indigo),    // 10: axion
    ];
})();
```

- [ ] **Step 3: Replace clearValue hardcoded colors**

Replace the two clearValue blocks (lines ~443–445 and ~468–470). Create a helper at the top of the class or module:

```js
const _bgLight = (() => { const [r,g,b] = paletteRGB(window._PALETTE.light.canvas); return {r,g,b,a:1}; })();
const _bgDark = (() => { const [r,g,b] = paletteRGB(window._PALETTE.dark.canvas); return {r,g,b,a:1}; })();
```

Then use `this.isLight ? _bgLight : _bgDark` in both places.

- [ ] **Step 4: Replace accent and text colors**

Line ~616 (accentColor):
```js
const _accentLight = paletteRGB(window._PALETTE.accent);
const _accentDark = paletteRGB(window._PALETTE.accentLight);
```
Use `this.isLight ? _accentLight : _accentDark`.

Lines ~621–622 (textColor):
```js
const _textLight = paletteRGB(window._PALETTE.light.text);
const _textDark = paletteRGB(window._PALETTE.dark.text);
```
Use `this.isLight ? _textLight : _textDark`.

- [ ] **Step 5: Replace drawFieldOverlay hardcoded colors**

Lines ~796–806: Replace with palette lookups:
```js
const _fieldColors = {
    higgs: [paletteRGB(window._PALETTE.extended.purple), paletteRGB(window._PALETTE.extended.lime)],
    axion: [paletteRGB(window._PALETTE.extended.indigo), paletteRGB(window._PALETTE.extended.yellow)],
};
```
Then in `drawFieldOverlay`:
```js
const [c0, c1] = _fieldColors[which];
f[12] = c0[0]; f[13] = c0[1]; f[14] = c0[2]; f[15] = 1.0;
f[16] = c1[0]; f[17] = c1[1]; f[18] = c1[2]; f[19] = 1.0;
```

- [ ] **Step 6: Replace heatmap constants**

Lines ~849–850:
```js
f[9] = 2.0;           // HEATMAP_SENSITIVITY
f[10] = 100.0 / 255.0; // HEATMAP_MAX_ALPHA
```
→
```js
f[9] = HEATMAP_SENSITIVITY;
f[10] = HEATMAP_MAX_ALPHA / 255.0;
```

- [ ] **Step 7: Commit**

```bash
git add src/gpu/gpu-renderer.js
git commit -m "refactor: replace hardcoded colors in gpu-renderer.js with _PALETTE lookups"
```

---

### Task 13: Final integration test

- [ ] **Step 1: Full visual + functional test**

Serve the project from `a9lim.github.io/` and test:

1. **CPU mode**: Temporarily disable WebGPU (or test in Firefox) — verify CPU renderer still works unchanged
2. **GPU mode**: Test in Chrome with WebGPU:
   - All 19 presets load correctly
   - Light and dark themes render correctly
   - All force toggles work (gravity, coulomb, magnetic, GM, 1PN, yukawa, higgs, axion)
   - All physics toggles work (relativity, spin-orbit, radiation, BH mode, disintegration, expansion)
   - Boundary modes: despawn, bounce, loop (torus/klein/RP²)
   - Collision modes: pass, merge, bounce
   - Visual elements: particles, photons, pions, trails, spin rings, force arrows, heatmap, field overlays
   - Barnes-Hut toggle
   - Ghost particles visible in periodic boundary
   - Signal delay (enable relativity)
   - Pair production
3. **Console**: No WGSL compilation errors, no JS errors

- [ ] **Step 2: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address integration issues from GPU constants centralization"
```
