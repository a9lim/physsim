# Performance Audit: physsim

**Date**: 2026-03-07
**Scope**: Full codebase — physics hot paths, rendering, memory/data structures, main loop, UI

---

## Executive Summary

The codebase has already implemented several key optimizations: SoA quadtree, object pooling for bosons/pions, fused PQS stencil walks, batched force arrows, dirty-flag render skip, PE cache in force loop, tree reuse, and display throttles. This audit identifies **54 additional findings** across four domains, prioritized by impact and effort.

**Top 5 highest-impact changes:**

1. **Quadtree insert child selection** — direct quadrant computation instead of trying all 4 children (HIGH)
2. **Yukawa cutoff distance** — skip `Math.exp` for distant pairs where force is negligible (HIGH)
3. **Flatten 11 display force Vec2s** — eliminate pointer indirection in O(N^2) inner loop (HIGH)
4. **Batch photon/pion canvas draws** — reduce up to 1280 canvas state changes to ~17 (HIGH)
5. **Inline torus `minImage`** — eliminate cross-module function call in O(N^2) force loop (HIGH)

---

## Table of Contents

- [1. Physics Hot Paths](#1-physics-hot-paths)
- [2. Rendering Pipeline](#2-rendering-pipeline)
- [3. Memory and Data Structures](#3-memory-and-data-structures)
- [4. Main Loop, UI, and Architecture](#4-main-loop-ui-and-architecture)
- [5. Prioritized Action Plan](#5-prioritized-action-plan)

---

## 1. Physics Hot Paths

### P1. Quadtree `insert()` tries all 4 children instead of computing correct child directly [HIGH]

**File**: `src/quadtree.js:143-146`

**Current code:**
```js
return this.insert(this.nw[idx], particle, depth + 1) ||
    this.insert(this.ne[idx], particle, depth + 1) ||
    this.insert(this.sw[idx], particle, depth + 1) ||
    this.insert(this.se[idx], particle, depth + 1);
```

Each `insert` call immediately calls `_contains()` (4 comparisons) to check if the particle is inside. But since the quadtree partitions space into four non-overlapping quadrants, a particle can be in exactly one child. The child can be determined directly by comparing `pos.x` vs `bx` and `pos.y` vs `by`.

**Fix:**
```js
const goRight = particle.pos.x > this.bx[idx];
const goDown  = particle.pos.y > this.by[idx];
const childIdx = goRight
    ? (goDown ? this.se[idx] : this.ne[idx])
    : (goDown ? this.sw[idx] : this.nw[idx]);
return this.insert(childIdx, particle, depth + 1);
```

Reduces expected `_contains` checks from ~2.5 per level to 0. For N particles with tree depth D, saves N x D x 4 comparisons. Also enables trivial iterative conversion since there's only ever one recursive branch.

---

### P2. `Math.exp(-mu*r)` in innermost pair loop without cutoff [HIGH]

**File**: `src/forces.js:339-351`

`Math.exp` is one of the most expensive transcendental functions, called once per pair per substep when Yukawa is enabled. For 50 particles: 2500 calls per force recomputation x 2-8 substeps per frame.

**Fix:** Add a cached cutoff distance. For `exp(-mu*r) < 0.002`, the force is negligible:
```js
// In config.js or computed when yukawaMu changes:
const yukawaCutoffSq = (6 / yukawaMu) ** 2;

// In pairForce, before Math.exp:
if (rawRSq > yukawaCutoffSq) { /* skip Yukawa block */ }
```

For default `yukawaMu = 0.15`, cutoff is at `r = 40` world units — a substantial fraction of the domain.

---

### P3. Gravity/Coulomb jerk computed even when radiation is off [MEDIUM]

**File**: `src/forces.js:193-216`

The jerk accumulator (`p.jerk.x/y`, `vrx/vry`, `rDotVr`) is used exclusively by Landau-Lifshitz radiation and quadrupole radiation. It is computed for every particle pair regardless of `radiationEnabled`. Each jerk computation costs 7 multiplies + 4 additions per force type per pair. For N=50: 7500+ unnecessary operations per substep.

**Fix:** Guard with `if (toggles.radiationEnabled)` around the jerk accumulation blocks and the `vrx/vry/rDotVr` calculations.

---

### P4. `aberr` multiplied into every force when it equals 1 [MEDIUM]

**File**: `src/forces.js:169-173`

When `signalDelayed === false` (the common case for BH tree aggregate nodes), `aberr = 1` is still multiplied into `fDir` for every force type — six redundant multiplications per pair.

**Fix:** Branch on `signalDelayed` to skip the `* aberr` multiplications entirely in the non-delayed path.

---

### P5. `_accum1PN` recomputes all distance geometry independently [MEDIUM]

**File**: `src/forces.js:390-450`

The 1PN velocity-Verlet pass recomputes `invRSq`, `invR`, `r`, `nx`, `ny` and calls `minImage` independently. When `!periodic`, the `minImage` branch adds an unnecessary check per pair.

**Fix:** Specialize a non-periodic version of `_accum1PN` that inlines the direct subtraction.

---

### P6. Adaptive substep O(N) scan repeated every substep iteration [MEDIUM]

**File**: `src/integrator.js:520-546`

The max-acceleration scan runs O(N) at the top of every substep. For MAX_SUBSTEPS=32 with 50 particles: 1600 max-scan operations before even starting the Boris kick, using stale force values from the previous substep.

**Fix:** Maintain running `_maxAccelSq` and `_maxCyclotron` as instance fields, updated incrementally during `computeAllForces()`. Converts per-substep O(N) scan to O(1).

---

### P7. `insert()` is recursive with up to depth 48 [MEDIUM]

**File**: `src/quadtree.js:119-146`

Each recursive call pushes a JS stack frame. For N=100 particles with depth 20: 2000 recursive calls per build. After implementing P1, the insert loop has at most one recursive branch per level, making iterative conversion trivial.

**Fix:** Convert to iterative using a simple integer stack array.

---

### P8. `_subdivide` checks pool capacity 4 times instead of once [LOW-MEDIUM]

**File**: `src/quadtree.js:109-117`

`alloc()` checks `this.count >= this.maxNodes` on every call. Four sequential allocations in `_subdivide` redundantly check capacity four times.

**Fix:** Check once before the four calls: `if (this.count + 4 > this.maxNodes) this._grow();`

---

### P9. `Math.sqrt(yukMod * sYukMod)` per pair — pre-cache sqrt on particle [MEDIUM]

**File**: `src/forces.js:204, 340, 445`

`Math.sqrt` is computed for `axMod` and `yukMod` geometric means per pair. With Yukawa+Axion+1PN: two sqrt calls per pair across O(N^2) pairs.

**Fix:** Pre-cache `p._yukModSqrt = Math.sqrt(p.yukMod)` after `_syncAxionField`. Geometric mean becomes `p._yukModSqrt * o._yukModSqrt` — one multiply instead of one sqrt.

---

### P10. `needAxMod` boolean evaluated per pair call [LOW]

**File**: `src/forces.js:203`

Three field reads + boolean ops computed per pair when it's constant per frame.

**Fix:** Precompute `toggles.axModEnabled` in `_syncToggles()`.

---

### P11. Photon renormalization O(N_photons) every frame [LOW]

**File**: `src/forces.js:752-758`

All photon velocities renormalized with `Math.sqrt` per photon per frame. Gravity deflections are tiny.

**Fix:** Only renormalize when `|v^2 - 1| > epsilon`.

---

### P12. `invR * invR * invR` in boson gravity [LOW-MEDIUM]

**File**: `src/forces.js:653-654, 698-699`

3 multiplies where 2 would suffice.

**Fix:** `const invRSq = 1/rSq; const invR = Math.sqrt(invRSq); const invR3 = invR * invRSq;`

---

### P13. Quadrupole radiation: third O(N) pass for contribution sums [LOW]

**File**: `src/integrator.js:1276-1277`

Contribution sums accumulated in a separate loop when they could be fused into the jerk+quadrupole loop.

---

### P14. Separate O(N) zero loop for `forceRadiation` [LOW]

**File**: `src/integrator.js:507-510`

Could be fused into `resetForces`.

---

### P15. Pair production inner loop is O(photons x particles) per substep [MEDIUM]

**File**: `main.js:294-323`

With 1024 photons and 100 particles: 102,400 comparisons per substep x 32 substeps = 3.27M comparisons per frame. Pre-filtering photons by energy/age would skip most iteration.

---

### P16. Dead-particle GC scan runs every substep [LOW-MEDIUM]

**File**: `main.js:410-420`

Particles need `maxDist * 128` substeps to become eligible for GC. Checking every 32 substeps instead of every 1 would cost nothing in accuracy.

---

### P17. `for...of rocheTransfers` allocates iterator [LOW]

**File**: `main.js:347`

Should use indexed loop for consistency with existing pattern.

---

---

## 2. Rendering Pipeline

### R1. Photon per-particle `globalAlpha` — up to 1024 state changes [HIGH]

**File**: `src/renderer.js:627`

Each photon gets its own `ctx.globalAlpha = alpha * alphaScale` inside the loop.

**Fix:** Bucket photons into 16 alpha levels by `(alpha * 16) | 0`. For each non-empty bin, set `globalAlpha` once, build one `beginPath` with all arcs, call `fill()`. Reduces fill calls from 1024 to at most 16.

---

### R2. Per-particle `shadowBlur` for charged particles [HIGH]

**File**: `src/renderer.js:301-312`

Every charged particle gets `ctx.shadowBlur = absQ * 3 + 10`, which invalidates the compositing layer for the shadow.

**Fix:** Sort charged particles into 3-4 blur-level buckets. Within each bucket, set `shadowBlur` once. Most simulations collapse to 2-3 distinct charge values.

---

### R3. Pion per-particle `fill()` — fixable to 1 fill [MEDIUM]

**File**: `src/renderer.js:652-661`

Each pion calls `beginPath()`, `arc()`, and `fill()` separately. Pion alpha is constant, so all pions share one path.

**Fix:** Hoist `ctx.beginPath()` before the loop, `ctx.fill()` after. Three-line change, eliminates up to 255 redundant fill calls.

---

### R4. Trail renders 4 strokes per particle [MEDIUM]

**File**: `src/renderer.js:238-261`

Each particle's trail is split into 4 alpha groups with separate `ctx.stroke()` calls. For N=20: 80 strokes per frame.

**Fix:** Reduce to 2 alpha groups (old/new half), or use `createLinearGradient()` along the trail path for smooth alpha in a single stroke.

---

### R5. Spin rings: no batching, 9 canvas calls each [MEDIUM]

**File**: `src/renderer.js:565-600`

`drawSpinRing()` issues ~9 canvas API calls per particle.

**Fix:** Batch spin rings by sign into two passes, one arc path and one arrowhead path per batch.

---

### R6. `setLineDash` 2x per particle (ergosphere + antimatter) [MEDIUM]

**File**: `src/renderer.js:333-349`

Each ergosphere/antimatter draw calls `setLineDash(_ERGO_DASH)` then `setLineDash(_NO_DASH)`.

**Fix:** Collect all ergosphere particles, draw in one `setLineDash` pass, then one `setLineDash([])`.

---

### R7. `ctx.save()/restore()` for 2-3 property writes [LOW-MEDIUM]

**Files**: `src/renderer.js`, `src/scalar-field.js:784-792`

`save()/restore()` snapshots ~30+ canvas properties when only 2-3 change.

**Fix:** Replace with explicit property write-then-restore.

---

### R8. Force component `Math.sqrt(fx*fx+fy*fy)` computed twice [LOW]

**File**: `src/renderer.js:485-486`

`magSq` is computed for the threshold check, then recomputed for the sqrt.

**Fix:** Cache `magSq` and reuse.

---

### R9. Heatmap: 3 separate blur passes on Float32 arrays [MEDIUM]

**File**: `src/heatmap.js:230-232`

Three potential arrays (grav, elec, yukawa) each get a separate 3x3 box blur — 6 x 4096 writes.

**Fix:** Fuse into one combined blur pass on the composited RGBA `_imgData` directly.

---

### R10. Heatmap: `Math.exp` for Yukawa per cell-particle pair [MEDIUM]

**File**: `src/heatmap.js:190-192`

64x64 grid x N particles = 81,920 `Math.exp` evaluations per heatmap update.

**Fix:** Same Yukawa cutoff as P2: `if (yukawaMu * r > 6) continue;` or a 256-entry LUT with linear interpolation.

---

### R11. `clientWidth` reflow + canvas resize every `draw()` [MEDIUM]

**Files**: `src/effective-potential.js:124-131`, `src/phase-plot.js:67-72`

`clientWidth` is a layout-triggering property read. Called ~30 times/second.

**Fix:** Cache `clientWidth` and `devicePixelRatio` at construction time and on window `resize` events.

---

### R12. Full V_eff recompute every sidebar frame [MEDIUM]

**File**: `src/effective-potential.js`

`update()` computes 200 samples from scratch even if the selected particle barely moved.

**Fix:** Hash key inputs (particle ID, `r` to 1 decimal, toggle state) and skip recomputation when unchanged. Split `_curveDirty` / `_markerDirty` for partial redraws.

---

### R13. Phase plot: full 512-point path redraw every sidebar frame [MEDIUM]

**File**: `src/phase-plot.js`

The ring buffer adds exactly one new point per update. Full redraw is O(count).

**Fix:** Draw onto a persistent offscreen canvas, incrementally update. Or throttle full redraws to every 32 points.

---

### R14. Stats display: 11-object array allocated per `updateSelected()` [MEDIUM]

**File**: `src/stats-display.js:97-109`

`const forces = [{ row, val, vec }, ...]` creates 11 objects 30 times/second = 330 short-lived objects/second.

**Fix:** Pre-allocate the force descriptor array once in the constructor.

---

### R15. Unconditional `textContent` sets (no change detection) [LOW]

**File**: `src/stats-display.js:48-63`

17 `.textContent` assignments regardless of whether values changed.

**Fix:** Compare before writing: `if (elem.textContent !== newVal) elem.textContent = newVal;`

---

---

## 3. Memory and Data Structures

### M1. Flatten 11 display force Vec2s to 22 scalars [HIGH]

**File**: `src/particle.js:19-41`

Each particle has 11 force Vec2 objects (`forceGravity`, `forceCoulomb`, etc.) as separate heap allocations (~704 bytes per particle for display-only data). In the O(N^2) pairwise loop, `pairForce()` dereferences these through pointers — each `p.forceGravity.x += ...` is a pointer load + property write.

**Fix:** Store as flat numeric properties (`fGravX`, `fGravY`, etc.). Eliminates 11 heap allocations per particle and one level of pointer indirection per force accumulation. `resetForces` collapses from 26 pointer-dereferences to 22 direct property writes.

---

### M2. Flatten `_f1pnOld` Vec2 to 2 scalars [HIGH]

**File**: `src/particle.js:41`

The `_f1pnOld` Vec2 wrapper is always accessed as `p._f1pnOld.x/y` in the integrator. Replacing with `_f1pnOldX`/`_f1pnOldY` eliminates one Vec2 allocation per particle and collapses a pointer dereference in the hot path to a direct property read.

---

### M3. Inline torus `minImage` in O(N^2) force loop [HIGH]

**File**: `src/topology.js:18`, called from `src/forces.js`

`minImage()` is called once per pair in the O(N^2) loop. For N=50 with 32 substeps: 40,000 cross-module function calls per frame. The function is 60+ lines and not inlinable across module boundaries by V8.

**Fix:** Specialize the force loop for the common TORUS topology with an inline fast path:
```js
if (periodic && topology === TORUS) {
    dx = sx - p.pos.x;
    if (dx > halfDomW) dx -= domW; else if (dx < -halfDomW) dx += domW;
    dy = sy - p.pos.y;
    if (dy > halfDomH) dy -= domH; else if (dy < -halfDomH) dy += domH;
} else if (periodic) {
    minImage(..., _miOut); dx = _miOut.x; dy = _miOut.y;
} else {
    dx = sx - p.pos.x; dy = sy - p.pos.y;
}
```

Eliminates 4 branches + 1 indirect call per pair for the dominant topology.

---

### M4. Interior fast path for `interpolateWithGradient()` [MEDIUM]

**File**: `src/scalar-field.js`

`interpolate()`, `gradient()`, and `interpolateWithGradient()` always call `_nb()` 16 times per stencil. `_depositPQS` and `_computeLaplacian` already have interior fast paths. ~90% of particle positions fall in the interior.

**Fix:** Mirror the `_depositPQS` pattern — check if stencil `[ix-1..ix+2]x[iy-1..iy+2]` is fully inside grid, use direct index arithmetic when true. Eliminates 16 `_nb()` calls per interpolation for ~90% of cases. With N=30 and both fields active: 960 `_nb()` calls reduced to ~240 arithmetic operations per substep.

---

### M5. Fuse `modulateMasses` + `applyForces` in Higgs/Axion [MEDIUM]

**Files**: `src/higgs-field.js:183-255`, `src/axion-field.js`

Each substep, Higgs calls `interpolate()` per particle for mass modulation, then `interpolateWithGradient()` per particle for forces — traversing the 4x4 PQS stencil twice. Same for Axion.

**Fix:** Combine into one method with a single `interpolateWithGradient()` call per particle. For N=30: eliminates 960 `_nb()` calls per substep x 32 substeps = 30,720 calls per frame.

---

### M6. Float32 downgrade for non-integration arrays [MEDIUM]

**File**: `src/scalar-field.js`

All 10 arrays per field are `Float64Array` (320KB per field, 640KB total). `_energyDensity`, `_gradX/Y`, `_sgPhiFull/GradX/GradY` don't need 64-bit precision.

**Fix:** Downgrade to Float32Array: saves ~160KB total, and Float32 arrays process 2x as many elements per SIMD instruction, reducing memory bandwidth by half for gradient/energy density loops.

---

### M7. Signal delay history buffers — interleaved layout [MEDIUM]

**File**: `src/particle.js:83-90`, `src/signal-delay.js`

Six separate `Float64Array(256)` per particle (12KB total) scattered across six heap regions. The NR solver accesses `histTime[lo]`, `histX[lo]`, `histY[lo]` — five array accesses across five disjoint memory regions per iteration.

**Fix:** Single interleaved buffer: `hist = Float64Array(HISTORY_SIZE * 6)` where `hist[h*6+0]=x, hist[h*6+1]=y, ..., hist[h*6+5]=time`. Each slot's data fits in a single cache line (48 bytes).

---

### M8. Sparse rho optimization in `_computeCoarsePotential` [LOW-MEDIUM]

**File**: `src/scalar-field.js:505-524`

O(SG^4) = 4096 operations iterating all 64 source cells for each of 64 observers. With sparse excitation (2-3 particles), only 10-20 coarse cells have nonzero rho.

**Fix:** Pre-build a list of non-vacuum source indices. Inner loop becomes N_nonzero^2 instead of 64^2.

---

### M9. Cap pool size to avoid unbounded GC tracing [LOW]

**Files**: `src/massless-boson.js:8-9`, `src/pion.js:10-11`

Pool arrays can grow without bound. Each live pool slot is a reference the GC must trace.

**Fix:** `if (_poolSize < MAX_PHOTONS) _pool[_poolSize++] = b;` — drop excess instances.

---

### M10. `applyGravForces` O(4096) existence check [LOW]

**File**: `src/scalar-field.js:644-647`

Scans all 4096 cells for `rho >= EPSILON` before the main loop. `computeSelfGravity` already has an early exit.

**Fix:** Track `_hasEnergy` boolean set by `computeSelfGravity()`.

---

---

## 4. Main Loop, UI, and Architecture

### A1. Web Worker for physics [HIGH — large effort]

The physics engine is the dominant CPU consumer. Moving it to a Web Worker would allow physics to run at 128Hz independent of the render thread. Requires SoA particle layout with `SharedArrayBuffer` and COOP/COEP headers for GitHub Pages. Could double effective physics throughput.

---

### A2. KaTeX lazy-load [MEDIUM]

**File**: `index.html:11, 18-19`

KaTeX CSS (~90KB blocking) + JS (~100KB deferred) loaded unconditionally. Only needed when reference overlay opens (rare).

**Fix:** Lazy-load on first reference open. `createInfoTip()` already guards `typeof renderMathInElement === 'function'`. Convert CSS to non-render-blocking preload.

---

### A3. KaTeX re-render cache [MEDIUM]

**File**: `src/ui.js:490-502`

Every reference overlay open re-renders all KaTeX from raw HTML + LaTeX. KaTeX rendering costs ~10-50ms per page.

**Fix:** Cache rendered HTML per key in a `Map`. First open pays the cost; subsequent opens are instant.

---

### A4. Batch preset/load DOM events [MEDIUM]

**Files**: `src/presets.js:603-687`, `src/save-load.js:30-61`

`loadPreset()` fires up to 30 synthetic DOM events. Each toggle `change` event calls `updateAllDeps()` (8 DOM queries). Total: 120 redundant querySelector/closest calls per preset load.

**Fix:** Add a `_batchingPreset` flag. Skip `updateAllDeps()` while batching, call once at the end.

---

### A5. `visibilitychange` to halt rAF loop [MEDIUM]

**File**: `main.js:459`

rAF schedules unconditionally. When tab is hidden, the accumulator can build up to ACCUMULATOR_CAP seconds of physics debt, causing a spike when returning.

**Fix:** Pause rAF scheduling on `visibilitychange` hidden, resume with `lastTime = 0` on visible.

---

### A6. Accumulator cap reduction [MEDIUM]

**File**: `src/config.js:41`

`ACCUMULATOR_CAP = 4` allows up to 1 second of physics debt.

**Fix:** Reduce to 1 or 2. Limits frame-rate spikes when returning to tab.

---

### A7. Defer shared scripts in `<head>` [MEDIUM]

**File**: `index.html:14-22`

Seven synchronous `<script>` tags block HTML parsing. All can be `defer`-ed since `main.js` is a module (implicitly deferred) and execution order is preserved.

---

### A8. Throttle `findParticleAt` to rAF [LOW-MEDIUM]

**File**: `src/input.js:193`

O(N) search on every `mousemove` event. High-refresh displays: 120-240Hz x 100 particles.

**Fix:** Set `_pendingHoverUpdate` flag in mousemove, resolve in rAF callback.

---

### A9. `Array.filter` in `_deleteParticle` [LOW]

**File**: `src/input.js:156`

Allocates a new array. Should use swap-and-pop like the rest of the codebase.

---

### A10. Tooltip `style.left/top` to `transform` [LOW]

**File**: `src/input.js:199-200`

`style.left/top` triggers layout. `transform: translate()` only triggers composite.

---

### A11. Eager field initialization [MEDIUM]

**File**: `main.js:21-155`

`HiggsField` and `AxionField` are created unconditionally in the constructor with full grid allocations, even though most users never enable them.

**Fix:** Lazy-initialize on first toggle-on.

---

### A12. `downloadState` revokeObjectURL timing [LOW]

**File**: `src/save-load.js:156-161`

`URL.revokeObjectURL` called immediately after `click()` may revoke before download starts.

**Fix:** Revoke in `requestAnimationFrame` callback after click.

---

---

## 5. Prioritized Action Plan

### Tier 1: High Impact, Low-Medium Effort

| # | Finding | Est. Speedup | Effort | Files |
|---|---------|-------------|--------|-------|
| P1 | Quadtree direct child selection | ~2x insert speed | Low | quadtree.js |
| P2 | Yukawa `Math.exp` cutoff | Skip 50%+ of exp calls | Low | forces.js, config.js |
| R1 | Batch photon alpha drawing | 1024 -> 16 fills | Low | renderer.js |
| R3 | Batch pion drawing | 256 -> 1 fill | Trivial | renderer.js |
| M3 | Inline torus minImage | Eliminate 40K fn calls/frame | Low | forces.js |
| P3 | Guard jerk behind radiation flag | 7500 ops/substep saved | Low | forces.js |

### Tier 2: Medium Impact, Low-Medium Effort

| # | Finding | Est. Speedup | Effort | Files |
|---|---------|-------------|--------|-------|
| M1 | Flatten display force Vec2s | Reduced pointer chasing | Medium | particle.js, forces.js, renderer.js |
| M4 | Interior fast path interpolation | 90% fewer `_nb()` calls | Low | scalar-field.js |
| M5 | Fuse modulateMasses + applyForces | Halve PQS stencil walks | Low | higgs-field.js, axion-field.js |
| P6 | Running max acceleration | O(N) -> O(1) per substep | Low | integrator.js, forces.js |
| R2 | Bucket charged particle glow | O(N) -> O(4) shadowBlur | Low | renderer.js |
| R4 | Reduce trail alpha groups | 4N -> 2N strokes | Low | renderer.js |
| A4 | Batch preset DOM events | 120 -> 8 DOM queries | Medium | presets.js, ui.js |
| A2 | KaTeX lazy-load | -190KB initial load | Low | index.html |
| A3 | KaTeX render cache | Skip re-render on reopen | Low | ui.js |
| A5 | visibilitychange halt | Prevent accumulator spike | Low | main.js |

### Tier 3: Lower Impact or Higher Effort

| # | Finding | Files |
|---|---------|-------|
| P7 | Iterative quadtree insert | quadtree.js |
| P4 | Skip aberr multiply when 1 | forces.js |
| P9 | Pre-cache yukMod sqrt | forces.js |
| M6 | Float32 for non-integration arrays | scalar-field.js |
| M7 | Interleaved history buffers | particle.js, signal-delay.js |
| M8 | Sparse rho in self-gravity | scalar-field.js |
| R9 | Fuse heatmap blur passes | heatmap.js |
| R10 | Heatmap Yukawa cutoff | heatmap.js |
| R11 | Cache clientWidth | effective-potential.js, phase-plot.js |
| R12 | V_eff dirty flag | effective-potential.js |
| R14 | Pre-allocate force descriptor | stats-display.js |
| A7 | Defer shared scripts | index.html |
| A8 | Throttle findParticleAt | input.js |
| A11 | Lazy field initialization | main.js |

### Tier 4: Architectural (High Effort, High Reward)

| # | Finding | Description |
|---|---------|-------------|
| A1 | Web Worker physics | SoA + SharedArrayBuffer, COOP/COEP headers. Could 2x throughput. |
| — | OffscreenCanvas fields | Heatmap/field overlays in dedicated worker. |
| M1+ | Full SoA particle layout | Typed arrays for all particle data. Prerequisite for Web Workers. |

---

## Appendix: Quick Wins (< 10 lines each)

1. **Batch pion draws** (R3): Move `beginPath()` before loop, `fill()` after — 3 lines
2. **Yukawa cutoff** (P2): Add one constant + one `if` — 3 lines
3. **Guard jerk** (P3): Wrap jerk blocks in `if (radiationEnabled)` — 6 lines
4. **Quadtree child** (P1): Replace 4-way `||` with 2 comparisons — 5 lines
5. **`for...of` -> indexed** (P17): Replace iterator with index — 1 line
6. **Precompute `needAxMod`** (P10): Add to toggles — 2 lines
7. **Skip `aberr * 1`** (P4): Branch on `signalDelayed` — 4 lines
8. **`filter` -> swap-and-pop** (A9): Replace in input.js — 4 lines
9. **textContent change detection** (R15): Add comparison — 1 line per element
10. **Tooltip transform** (A10): Use `transform: translate()` — 2 lines
