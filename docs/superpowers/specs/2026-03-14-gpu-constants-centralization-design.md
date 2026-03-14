# GPU Constants & Colors Centralization

## Problem

Physics constants and palette colors are duplicated across ~25 WGSL shader files, 4 GPU JS files, and `config.js`. Changes require updating multiple files, values can drift (NR_TOLERANCE already has: `1e-12` in config.js vs `1e-5` in shaders), and colors are hardcoded as RGB float triples instead of reading from `_PALETTE`.

The CPU renderer already does this correctly — it reads `_PALETTE.extended.*` at module load. The GPU side hardcodes the same hex values as float triples.

## Approach

**Generated WGSL constants block** (Approach C): At GPU init, build a string of WGSL `const` declarations from `config.js` exports + `_PALETTE` hex→RGB conversions. Prepend this block to **all** shaders before compilation — both prepended (Category A/field) and standalone (Category B/C). Remove local duplicate `const` lines from individual shaders.

- Zero runtime overhead (compile-time constants)
- Shaders remain valid WGSL (just `const` declarations)
- Piggybacks on existing `common.wgsl` / `field-common.wgsl` prepend infrastructure
- Runtime-variable values (yukawaMu, higgsMass, etc.) stay in uniform buffers

## New File: `src/gpu/gpu-constants.js`

Exports:

- `buildWGSLConstants()` — returns a WGSL string of `const` declarations, generated from `config.js` + `_PALETTE`. Called once at GPU init, cached.
- `paletteRGB(hexString)` — converts `#RRGGBB` to `[number, number, number]` normalized floats (0–1). Used by `gpu-renderer.js` for JS-side color values passed to `writeBuffer`.

### Generated WGSL Output

The block includes physics constants, capacity/grid constants, enum/flag constants, derived constants, and palette colors. All values are dynamically computed from `config.js` and `_PALETTE` at init time — below is an example of the output, not a hardcoded template.

```wgsl
// ── Auto-generated from config.js + _PALETTE ──

// Physics constants
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
const BOSON_SOFTENING_SQ: f32 = 4.0;
const BOSON_MIN_AGE: u32 = 4u;
const BOSON_MIN_AGE_TIME: f32 = 0.03125;  // BOSON_MIN_AGE * PHYSICS_DT
const LL_FORCE_CLAMP: f32 = 0.5;
const MIN_MASS: f32 = 0.05;
const BH_NAKED_FLOOR: f32 = 0.5;
const ELECTRON_MASS: f32 = 0.05;
const GPU_MAX_SPEED_RATIO: f32 = 0.9999;  // GPU-specific: f32 needs tighter cap than CPU 0.99
const PION_DECAY_PROB: f32 = <computed>;
const CHARGED_PION_DECAY_PROB: f32 = <computed>;
const BH_THETA: f32 = 0.5;
const BH_THETA_SQ: f32 = 0.25;
const VELOCITY_VECTOR_SCALE: f32 = 32.0;

// Capacity constants (GPU-specific)
const MAX_PARTICLES: u32 = 4096u;
const MAX_PHOTONS: u32 = 1024u;
const MAX_PIONS: u32 = 256u;
const MAX_GHOSTS: u32 = 4096u;  // = MAX_PARTICLES
const PHOTON_LIFETIME: f32 = 256.0;

// Grid constants (GPU-specific, independently tunable)
const GRID: u32 = 64u;
const GRID_SQ: u32 = 4096u;   // GRID * GRID
const GRID_LAST: u32 = 63u;   // GRID - 1
const COARSE: u32 = 8u;
const COARSE_SQ: u32 = 64u;   // COARSE * COARSE
const SG_RATIO: u32 = 8u;     // GRID / COARSE
const SCALAR_FIELD_MAX: f32 = 2.0;
const FIELD_EXCITATION_SIGMA: f32 = 2.0;
const MERGE_EXCITATION_SCALE: f32 = 0.5;
const HGRID: u32 = 64u;       // matches GPU_HEATMAP_GRID from config.js
const HGRID_SQ: u32 = 4096u;  // HGRID * HGRID

// Signal delay / history / trails
const HISTORY_LEN: u32 = 256u;
const HISTORY_MASK: u32 = 255u;  // HISTORY_LEN - 1
const NR_MAX_ITER: u32 = 8u;
const NR_TOLERANCE: f32 = 1e-5;  // GPU-specific: f32 precision limit (CPU uses 1e-12)
const TRAIL_LEN: u32 = 256u;

// Boundary mode enums
const BOUND_DESPAWN: u32 = 0u;
const BOUND_BOUNCE: u32 = 1u;
const BOUND_LOOP: u32 = 2u;

// Topology enums (both naming conventions for cross-shader compat)
const TOPO_TORUS: u32 = 0u;
const TOPO_KLEIN: u32 = 1u;
const TOPO_RP2: u32 = 2u;
const TORUS: u32 = 0u;
const KLEIN: u32 = 1u;
const RP2: u32 = 2u;

// Collision mode enums
const COL_PASS: u32 = 0u;
const COL_MERGE: u32 = 1u;
const COL_BOUNCE: u32 = 2u;

// Particle flag bits
const FLAG_ALIVE: u32 = 1u;
const FLAG_RETIRED: u32 = 2u;
const FLAG_ANTIMATTER: u32 = 4u;
const FLAG_BH: u32 = 8u;
const FLAG_GHOST: u32 = 16u;

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
const QT_CAPACITY: u32 = 4u;
const MAX_DEPTH: u32 = 48u;

// Disintegration / pair production
const MAX_DISINT_EVENTS: u32 = 64u;
const MAX_PAIR_EVENTS: u32 = 32u;

// Palette colors (from _PALETTE.extended.* — hex source in comment)
const COLOR_SLATE: vec3f = vec3f(0.541, 0.494, 0.447);    // #8A7E72
const COLOR_RED: vec3f = vec3f(0.753, 0.314, 0.282);      // #C05048
const COLOR_BLUE: vec3f = vec3f(0.361, 0.573, 0.659);     // #5C92A8
const COLOR_GREEN: vec3f = vec3f(0.314, 0.596, 0.471);    // #509878
const COLOR_CYAN: vec3f = vec3f(0.290, 0.675, 0.627);     // #4AACA0
const COLOR_ORANGE: vec3f = vec3f(0.800, 0.557, 0.306);   // #CC8E4E
const COLOR_YELLOW: vec3f = vec3f(0.800, 0.659, 0.298);   // #CCA84C
const COLOR_ROSE: vec3f = vec3f(0.769, 0.384, 0.447);     // #C46272
const COLOR_PURPLE: vec3f = vec3f(0.612, 0.494, 0.690);   // #9C7EB0
const COLOR_BROWN: vec3f = vec3f(0.612, 0.408, 0.251);    // #9C6840
const COLOR_LIME: vec3f = vec3f(0.510, 0.659, 0.341);     // #82A857
const COLOR_INDIGO: vec3f = vec3f(0.424, 0.475, 0.675);   // #6C79AC
const COLOR_MAGENTA: vec3f = vec3f(0.706, 0.412, 0.612);  // #B4689C

// Theme colors
const COLOR_ACCENT: vec3f = vec3f(0.996, 0.231, 0.004);       // #FE3B01
const COLOR_ACCENT_LIGHT: vec3f = vec3f(1.0, 0.463, 0.259);   // #FF7642

// Spin ring colors (HSL-derived from palette hues, brighter saturation for visibility)
// CPU renderer uses _PALETTE.spinPos/spinNeg hues → HSLA(170,80%,60%) / HSLA(30,80%,60%)
// Generated from _PALETTE.spinPos/spinNeg hue values via HSL→RGB conversion
const COLOR_SPIN_CW: vec3f = vec3f(0.278, 0.878, 0.780);     // HSLA(170, 80%, 60%)
const COLOR_SPIN_CCW: vec3f = vec3f(0.922, 0.678, 0.278);    // HSLA(30, 80%, 60%)
```

## GPU-Specific Constants in `config.js`

New constants for values where GPU can use different (potentially higher) values due to parallelism, or where f32 precision requires different thresholds:

| Constant | Default | Purpose |
|---|---|---|
| `SELFGRAV_GRID` | 8 | CPU self-gravity coarse grid (new — currently hardcoded as `8` in scalar-field.js) |
| `GPU_SCALAR_GRID` | 64 | GPU scalar field grid resolution (tunable to 128+) |
| `GPU_SELFGRAV_GRID` | 8 | GPU self-gravity coarse grid (tunable to 16+) |
| `GPU_NR_TOLERANCE` | 1e-5 | GPU Newton-Raphson tolerance (f32 precision limit) |
| `GPU_HEATMAP_GRID` | 64 | GPU heatmap overlay resolution (tunable to 128) |
| `GPU_MAX_PARTICLES` | 4096 | GPU buffer pre-allocation limit |
| `GPU_MAX_SPEED_RATIO` | 0.9999 | GPU speed cap (f32 needs tighter bound than CPU's 0.99) |

GPU defaults match CPU values where appropriate, diverge only where f32 precision demands it.

## Bug Fixes

### NR_TOLERANCE Mismatch

CPU uses `NR_TOLERANCE = 1e-12` (Float64). GPU shaders had `1e-5` hardcoded — correct for f32 precision but undocumented divergence. Now explicit as `GPU_NR_TOLERANCE = 1e-5` in `config.js`.

### MAX_SPEED_RATIO Mismatch

CPU uses `MAX_SPEED_RATIO = 0.99`. `radiation.wgsl` uses `0.9999` — intentional for f32 pion emission kinematics where `0.99` caused clamping artifacts. Now explicit as `GPU_MAX_SPEED_RATIO = 0.9999` in `config.js`.

## Injection Points

### Shader categories and how constants reach them

The constants block is prepended to **all** shaders — both prepended and standalone. This means toggle bits, flag bits, boundary/topology/collision enums, and all physics constants are available everywhere. Local declarations of any of these are removed from all shader files.

**Category A — Prepended shaders (Phase 2 + boundary.wgsl):**
Constants block prepended *before* `common.wgsl`. Physics constants AND toggle/flag/enum constants removed from `common.wgsl`. Structs, toggle query helpers (`hasToggle0`/`hasToggle1`), topology helpers (`torusMinImage`/`fullMinImage`), and packed buffer struct definitions remain in `common.wgsl`.

**Category B — Standalone compute shaders:**
`fetchShader()` gains optional `prepend` parameter. Pipeline creators pass constants block. Local duplicate `const` lines (physics, flags, toggles, enums) removed from: `bosons.wgsl`, `boson-tree.wgsl`, `radiation.wgsl`, `collision.wgsl`, `forces-tree.wgsl`, `heatmap.wgsl`, `history.wgsl`, `trails.wgsl`, `ghost-gen.wgsl`, `dead-gc.wgsl`, `tree-build.wgsl`, `onePN.wgsl`, `disintegration.wgsl`, `pair-production.wgsl`, `expansion.wgsl`.

**Category C — Render shaders:**
Same prepend approach. Local flag/enum constants AND hardcoded RGB values removed/replaced: `update-colors.wgsl`, `boson-render.wgsl`, `spin-render.wgsl`, `heatmap-render.wgsl`, `arrow-render.wgsl`, `trail-render.wgsl`, `hit-test.wgsl`, `particle.wgsl`, `field-render.wgsl`.

**Field shaders (field-common.wgsl prepended):**
Constants block prepended *before* `field-common.wgsl`. Grid constants (GRID, SCALAR_FIELD_MAX, etc.) AND boundary/topology enums removed from `field-common.wgsl`. PQS weight computation, `FieldUniforms` struct, `nbIndex()`, and `isInterior()` remain.

### `gpu-pipelines.js` changes

`fetchShader(filename, { prepend } = {})` — optional prepend string inserted before shader source. All pipeline creation functions receive the constants block as a parameter.

Flow: `buildWGSLConstants()` called once → cached string → passed to `createPhase2Pipelines()`, `createPhase3Pipelines()`, etc. → each passes to `fetchShader()` or prepends to `common.wgsl`/`field-common.wgsl`.

### GPU JS file changes

**`gpu-renderer.js`:**
- `FORCE_COLORS` static array: built from `_PALETTE.extended.*` via `paletteRGB()`
- `clearValue` backgrounds: `paletteRGB(_PALETTE.light.canvas)` / `paletteRGB(_PALETTE.dark.canvas)`
- `drawFieldOverlay` color uniforms: `paletteRGB(_PALETTE.extended.purple)` etc.
- Accent/text colors for custom arrows: from `_PALETTE.accent`/`accentLight`/`light.text`/`dark.text`

**`gpu-buffers.js`:**
Replace local constants with `config.js` imports: `HISTORY_LEN`→`HISTORY_SIZE`, `MAX_PHOTONS`, `MAX_PIONS`, `FIELD_GRID_RES`→`GPU_SCALAR_GRID`, `COARSE_RES`→`GPU_SELFGRAV_GRID`, `TRAIL_LEN`→`MAX_TRAIL_LENGTH`.

**`gpu-physics.js`:**
Replace local constants with `config.js` imports: `MAX_PARTICLES`→`GPU_MAX_PARTICLES`, `HISTORY_STRIDE`, `MAX_PHOTONS`, `MAX_PIONS`. Also replace local `FLAG_ALIVE`, `FLAG_RETIRED`, `FLAG_ANTIMATTER`, `BOUND_LOOP`, `COL_MERGE`, `COL_BOUNCE` with config.js imports.

## CPU-Side Change: `scalar-field.js`

Import `SELFGRAV_GRID` from `config.js`, replace hardcoded `8` in self-gravity grid code.

## Spin Ring Color Handling

Spin ring colors are intentionally different from raw palette values — they are HSL-derived with 80% saturation and 60% lightness for on-screen visibility, computed from `_PALETTE.spinPos` / `_PALETTE.spinNeg` hue values (which come from `_PALETTE.extended.cyan` / `_PALETTE.extended.orange` hues). The generated block includes dedicated `COLOR_SPIN_CW` and `COLOR_SPIN_CCW` constants computed via HSL→RGB conversion from these hue values, matching the CPU renderer's behavior.

## Heatmap Render Color Notes

`heatmap-render.wgsl` uses simplified single-channel intensities for electric potential (red=0.75 for positive, blue=0.65 for negative) that are NOT direct palette colors — they are intentional approximations for the heatmap overlay. These will use `COLOR_RED` and `COLOR_BLUE` channel values from the palette instead, which are close enough and maintain consistency. The gravity channel uses `COLOR_SLATE` and the Yukawa channel uses `COLOR_GREEN`.

## What Does NOT Change

- Struct definitions stay in `common.wgsl` / `field-common.wgsl` / standalone shaders
- Toggle query helpers (`hasToggle0`/`hasToggle1`) stay in `common.wgsl`
- Topology helpers (`torusMinImage`/`fullMinImage`) stay in `common.wgsl`
- PQS weights, `nbIndex()`, `isInterior()` stay in `field-common.wgsl`
- Struct byte sizes in `gpu-buffers.js` (`PARTICLE_STATE_SIZE` etc.) — buffer layout, not physics
- Runtime-variable values already in uniform buffers (yukawaMu, higgsMass, external fields, etc.)
- `colors.js` — already reads `_PALETTE`, no changes needed
- Shader-local constants that have no `config.js` equivalent and are truly shader-specific (e.g., `DARK_QUAD_SCALE` in `particle.wgsl`)

## File Change Summary

| File | Action |
|---|---|
| **New:** `src/gpu/gpu-constants.js` | `buildWGSLConstants()` + `paletteRGB()` |
| `src/config.js` | Add `SELFGRAV_GRID`, `GPU_*` constants |
| `src/scalar-field.js` | Import `SELFGRAV_GRID`, replace hardcoded 8 |
| `src/gpu/gpu-pipelines.js` | `fetchShader()` gains prepend, all pipeline creators receive constants |
| `src/gpu/gpu-renderer.js` | Colors from `_PALETTE` via `paletteRGB()` |
| `src/gpu/gpu-buffers.js` | Import constants from `config.js` |
| `src/gpu/gpu-physics.js` | Import constants from `config.js` |
| **Shared headers** | |
| `shaders/common.wgsl` | Remove physics constants, toggle bits, flag bits, enums (keep structs + helpers) |
| `shaders/field-common.wgsl` | Remove grid/field constants, boundary/topology enums (keep PQS + FieldUniforms) |
| **Standalone compute shaders** | |
| `shaders/bosons.wgsl` | Remove ~14 local constants + flag/toggle bits |
| `shaders/boson-tree.wgsl` | Remove ~4 local constants + flag bits |
| `shaders/radiation.wgsl` | Remove ~7 local constants + flag/toggle bits |
| `shaders/collision.wgsl` | Remove INERTIA_K + flag/collision mode constants |
| `shaders/forces-tree.wgsl` | Remove 3 physics constants + flag/toggle bits |
| `shaders/heatmap.wgsl` | Remove 4+ local constants (standalone compute, NOT field-common prepended) |
| `shaders/history.wgsl` | Remove 3 local constants + flag bits |
| `shaders/trails.wgsl` | Remove TRAIL_LEN + flag bits |
| `shaders/ghost-gen.wgsl` | Remove topology enums, flag bits, MAX_GHOSTS |
| `shaders/dead-gc.wgsl` | Remove flag bits |
| `shaders/tree-build.wgsl` | Remove flag bits, MAX_DEPTH, QT_CAPACITY |
| `shaders/onePN.wgsl` | Remove flag/toggle bits, EPSILON, boundary/topology enums |
| `shaders/disintegration.wgsl` | Remove EPSILON, MAX_DISINT_EVENTS |
| `shaders/pair-production.wgsl` | Remove MAX_PAIR_EVENTS |
| `shaders/expansion.wgsl` | Remove any local constants |
| **Field compute shaders** (inherit constants via field-common.wgsl chain) | |
| `shaders/field-deposit.wgsl` | No local constant changes needed |
| `shaders/field-evolve.wgsl` | No local constant changes needed |
| `shaders/field-forces.wgsl` | No local constant changes needed |
| `shaders/field-selfgrav.wgsl` | No local constant changes needed |
| `shaders/field-excitation.wgsl` | No local constant changes needed |
| **Render shaders** | |
| `shaders/update-colors.wgsl` | Replace hardcoded RGB → COLOR_SLATE/RED/BLUE |
| `shaders/boson-render.wgsl` | Replace hardcoded RGB → COLOR_YELLOW/RED/GREEN, use PHOTON_LIFETIME |
| `shaders/spin-render.wgsl` | Replace hardcoded RGB → COLOR_SPIN_CW/CCW |
| `shaders/heatmap-render.wgsl` | Remove HGRID, replace colors → COLOR_SLATE/RED/BLUE/GREEN |
| `shaders/arrow-render.wgsl` | Remove flag bits, VELOCITY_VECTOR_SCALE |
| `shaders/trail-render.wgsl` | Remove flag bits |
| `shaders/hit-test.wgsl` | Remove flag bits |
| `shaders/particle.wgsl` | Remove flag bits (keep DARK_QUAD_SCALE — shader-specific) |
| `shaders/field-render.wgsl` | Remove any local constants provided by block |
