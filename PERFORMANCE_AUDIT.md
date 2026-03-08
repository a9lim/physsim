# Performance Audit: physsim

**Date**: 2026-03-07
**Scope**: Full codebase — physics hot paths, rendering, memory/data structures, main loop, UI
**Status**: 38 of 54 findings implemented, 1 already handled, 15 deferred

---

## Executive Summary

The codebase has already implemented several key optimizations: SoA quadtree, object pooling for bosons/pions, fused PQS stencil walks, batched force arrows, dirty-flag render skip, PE cache in force loop, tree reuse, and display throttles. This audit identified **54 additional findings** across four domains, prioritized by impact and effort. **39 have been resolved** (38 implemented + 1 already handled). 15 are deferred with rationale.

### Implementation Status

| Status | Count | Findings |
|--------|-------|----------|
| Done | 38 | P1-P4, P7-P8, P10-P13, P16-P17, R1-R3, R5-R8, R11-R12, R14-R15, M3-M4, M8-M10, A2-A3, A5-A12 |
| Already handled | 1 | P15 |
| Deferred | 15 | P5-P6, P9, P14, R4, R9-R10, R13, M1-M2, M5-M7, A1, A4 |

---

## Table of Contents

- [1. Physics Hot Paths](#1-physics-hot-paths)
- [2. Rendering Pipeline](#2-rendering-pipeline)
- [3. Memory and Data Structures](#3-memory-and-data-structures)
- [4. Main Loop, UI, and Architecture](#4-main-loop-ui-and-architecture)
- [5. Prioritized Action Plan](#5-prioritized-action-plan)

---

## 1. Physics Hot Paths

### P1. Quadtree `insert()` tries all 4 children instead of computing correct child directly [HIGH] — DONE

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

### P2. `Math.exp(-mu*r)` in innermost pair loop without cutoff [HIGH] — DONE

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

### P3. Gravity/Coulomb jerk computed even when radiation is off [MEDIUM] — DONE

**File**: `src/forces.js:193-216`

The jerk accumulator (`p.jerk.x/y`, `vrx/vry`, `rDotVr`) is used exclusively by Landau-Lifshitz radiation and quadrupole radiation. It is computed for every particle pair regardless of `radiationEnabled`. Each jerk computation costs 7 multiplies + 4 additions per force type per pair. For N=50: 7500+ unnecessary operations per substep.

**Fix:** Guard with `if (toggles.radiationEnabled)` around the jerk accumulation blocks and the `vrx/vry/rDotVr` calculations.

---

### P4. `aberr` multiplied into every force when it equals 1 [MEDIUM] — DONE (+ TDZ bug fix)

**File**: `src/forces.js:169-173`

When `signalDelayed === false` (the common case for BH tree aggregate nodes), `aberr = 1` is still multiplied into `fDir` for every force type — six redundant multiplications per pair.

**Fix:** Branch on `signalDelayed` to skip the `* aberr` multiplications entirely in the non-delayed path.

---

### P5. `_accum1PN` recomputes all distance geometry independently [MEDIUM] — DEFERRED

**File**: `src/forces.js:390-450`

The 1PN velocity-Verlet pass recomputes `invRSq`, `invR`, `r`, `nx`, `ny` and calls `minImage` independently. When `!periodic`, the `minImage` branch adds an unnecessary check per pair.

**Fix:** Specialize a non-periodic version of `_accum1PN` that inlines the direct subtraction.

**Status:** Partially addressed by M3 (inline torus minImage in `_accum1PN`). Remaining geometry recomputation is inherent — 1PN runs at post-drift positions, so `invRSq`/`invR`/`r` cannot be shared with the pre-drift force loop.

---

### P6. Adaptive substep O(N) scan repeated every substep iteration [MEDIUM] — DEFERRED

**File**: `src/integrator.js:520-546`

The max-acceleration scan runs O(N) at the top of every substep. For MAX_SUBSTEPS=32 with 50 particles: 1600 max-scan operations before even starting the Boris kick, using stale force values from the previous substep.

**Fix:** Maintain running `_maxAccelSq` and `_maxCyclotron` as instance fields, updated incrementally during `computeAllForces()`. Converts per-substep O(N) scan to O(1).

**Status:** Deferred — would couple `forces.js` state to `integrator.js`. The O(N) scan is cheap relative to the O(N²) force computation that dominates each substep.

---

### P7. `insert()` is recursive with up to depth 48 [MEDIUM] — DONE

**File**: `src/quadtree.js:119-146`

Each recursive call pushes a JS stack frame. For N=100 particles with depth 20: 2000 recursive calls per build. After implementing P1, the insert loop has at most one recursive branch per level, making iterative conversion trivial.

**Fix:** Convert to iterative using a simple integer stack array.

---

### P8. `_subdivide` checks pool capacity 4 times instead of once [LOW-MEDIUM] — DONE

**File**: `src/quadtree.js:109-117`

`alloc()` checks `this.count >= this.maxNodes` on every call. Four sequential allocations in `_subdivide` redundantly check capacity four times.

**Fix:** Check once before the four calls: `if (this.count + 4 > this.maxNodes) this._grow();`

---

### P9. `Math.sqrt(yukMod * sYukMod)` per pair — pre-cache sqrt on particle [MEDIUM] — DEFERRED

**File**: `src/forces.js:204, 340, 445`

`Math.sqrt` is computed for `axMod` and `yukMod` geometric means per pair. With Yukawa+Axion+1PN: two sqrt calls per pair across O(N^2) pairs.

**Fix:** Pre-cache `p._yukModSqrt = Math.sqrt(p.yukMod)` after `_syncAxionField`. Geometric mean becomes `p._yukModSqrt * o._yukModSqrt` — one multiply instead of one sqrt.

**Status:** Deferred — only relevant when both Yukawa + Axion are active (rare combination). The `sqrt` cost is amortized across few pairs.

---

### P10. `needAxMod` boolean evaluated per pair call [LOW] — DONE

**File**: `src/forces.js:203`

Three field reads + boolean ops computed per pair when it's constant per frame.

**Fix:** Precompute `toggles.axModEnabled` in `_syncToggles()`.

---

### P11. Photon renormalization O(N_photons) every frame [LOW] — DONE

**File**: `src/forces.js:752-758`

All photon velocities renormalized with `Math.sqrt` per photon per frame. Gravity deflections are tiny.

**Fix:** Only renormalize when `|v^2 - 1| > epsilon`.

---

### P12. `invR * invR * invR` in boson gravity [LOW-MEDIUM] — DONE

**File**: `src/forces.js:653-654, 698-699`

3 multiplies where 2 would suffice.

**Fix:** `const invRSq = 1/rSq; const invR = Math.sqrt(invRSq); const invR3 = invR * invRSq;`

---

### P13. Quadrupole radiation: third O(N) pass for contribution sums [LOW] — DONE

**File**: `src/integrator.js:1276-1277`

Contribution sums accumulated in a separate loop when they could be fused into the jerk+quadrupole loop.

**Implementation:** KE scan merged into the quadrupole loop, eliminating the separate O(N) pass.

---

### P14. Separate O(N) zero loop for `forceRadiation` [LOW] — DEFERRED

**File**: `src/integrator.js:507-510`

Could be fused into `resetForces`.

**Status:** Deferred — LOW impact. Separate zero loop is clearer and costs O(N) vs the O(N²) force computation that dominates.

---

### P15. Pair production inner loop is O(photons x particles) per substep [MEDIUM] — ALREADY HANDLED

**File**: `main.js:294-323`

With 1024 photons and 100 particles: 102,400 comparisons per substep x 32 substeps = 3.27M comparisons per frame. Pre-filtering photons by energy/age would skip most iteration.

**Status:** Already handled — existing energy/age `continue` checks at the top of the photon loop effectively pre-filter most photons before the inner particle distance check runs.

---

### P16. Dead-particle GC scan runs every substep [LOW-MEDIUM] — DONE

**File**: `main.js:410-420`

Particles need `maxDist * 128` substeps to become eligible for GC. Checking every 32 substeps instead of every 1 would cost nothing in accuracy.

---

### P17. `for...of rocheTransfers` allocates iterator [LOW] — DONE

**File**: `main.js:347`

Should use indexed loop for consistency with existing pattern.

---

---

## 2. Rendering Pipeline

### R1. Photon per-particle `globalAlpha` — up to 1024 state changes [HIGH] — DONE

**File**: `src/renderer.js:627`

Each photon gets its own `ctx.globalAlpha = alpha * alphaScale` inside the loop.

**Fix:** Bucket photons into 16 alpha levels by `(alpha * 16) | 0`. For each non-empty bin, set `globalAlpha` once, build one `beginPath` with all arcs, call `fill()`. Reduces fill calls from 1024 to at most 16.

---

### R2. Per-particle `shadowBlur` for charged particles [HIGH] — DONE

**File**: `src/renderer.js:301-312`

Every charged particle gets `ctx.shadowBlur = absQ * 3 + 10`, which invalidates the compositing layer for the shadow.

**Fix:** Sort charged particles into 3-4 blur-level buckets. Within each bucket, set `shadowBlur` once. Most simulations collapse to 2-3 distinct charge values.

---

### R3. Pion per-particle `fill()` — fixable to 1 fill [MEDIUM] — DONE

**File**: `src/renderer.js:652-661`

Each pion calls `beginPath()`, `arc()`, and `fill()` separately. Pion alpha is constant, so all pions share one path.

**Fix:** Hoist `ctx.beginPath()` before the loop, `ctx.fill()` after. Three-line change, eliminates up to 255 redundant fill calls.

---

### R4. Trail renders 4 strokes per particle [MEDIUM] — DEFERRED

**File**: `src/renderer.js:238-261`

Each particle's trail is split into 4 alpha groups with separate `ctx.stroke()` calls. For N=20: 80 strokes per frame.

**Fix:** Reduce to 2 alpha groups (old/new half), or use `createLinearGradient()` along the trail path for smooth alpha in a single stroke.

**Status:** Deferred — reducing alpha groups causes visible banding. The 4-group gradient is a deliberate visual quality choice.

---

### R5. Spin rings: no batching, 9 canvas calls each [MEDIUM] — DONE

**File**: `src/renderer.js:565-600`

`drawSpinRing()` issues ~9 canvas API calls per particle.

**Fix:** Batch spin rings by sign into two passes, one arc path and one arrowhead path per batch.

---

### R6. `setLineDash` 2x per particle (ergosphere + antimatter) [MEDIUM] — DONE

**File**: `src/renderer.js:333-349`

Each ergosphere/antimatter draw calls `setLineDash(_ERGO_DASH)` then `setLineDash(_NO_DASH)`.

**Fix:** Collect all ergosphere particles, draw in one `setLineDash` pass, then one `setLineDash([])`.

---

### R7. `ctx.save()/restore()` for 2-3 property writes [LOW-MEDIUM] — DONE

**Files**: `src/renderer.js`, `src/scalar-field.js:784-792`

`save()/restore()` snapshots ~30+ canvas properties when only 2-3 change.

**Fix:** Replace with explicit property write-then-restore.

---

### R8. Force component `Math.sqrt(fx*fx+fy*fy)` computed twice [LOW] — DONE

**File**: `src/renderer.js:485-486`

`magSq` is computed for the threshold check, then recomputed for the sqrt.

**Fix:** Cache `magSq` and reuse.

---

### R9. Heatmap: 3 separate blur passes on Float32 arrays [MEDIUM] — DEFERRED

**File**: `src/heatmap.js:230-232`

Three potential arrays (grav, elec, yukawa) each get a separate 3x3 box blur — 6 x 4096 writes.

**Fix:** Fuse into one combined blur pass on the composited RGBA `_imgData` directly.

**Status:** Deferred — complex interaction with per-channel coloring (grav=red, elec=blue, yukawa=green). Fusing would require compositing in a single buffer with channel isolation. Heatmap already throttled to every 4th frame.

---

### R10. Heatmap: `Math.exp` for Yukawa per cell-particle pair [MEDIUM] — DEFERRED

**File**: `src/heatmap.js:190-192`

64x64 grid x N particles = 81,920 `Math.exp` evaluations per heatmap update.

**Fix:** Same Yukawa cutoff as P2: `if (yukawaMu * r > 6) continue;` or a 256-entry LUT with linear interpolation.

**Status:** Deferred — heatmap already throttled to HEATMAP_INTERVAL=4 frames, reducing effective cost by 4x. The Yukawa cutoff (P2) was applied to the force loop where it matters most.

---

### R11. `clientWidth` reflow + canvas resize every `draw()` [MEDIUM] — DONE

**Files**: `src/effective-potential.js:124-131`, `src/phase-plot.js:67-72`

`clientWidth` is a layout-triggering property read. Called ~30 times/second.

**Fix:** Cache `clientWidth` and `devicePixelRatio` at construction time and on window `resize` events.

---

### R12. Full V_eff recompute every sidebar frame [MEDIUM] — DONE

**File**: `src/effective-potential.js`

`update()` computes 200 samples from scratch even if the selected particle barely moved.

**Fix:** Hash key inputs (particle ID, `r` to 1 decimal, toggle state) and skip recomputation when unchanged. Split `_curveDirty` / `_markerDirty` for partial redraws.

---

### R13. Phase plot: full 512-point path redraw every sidebar frame [MEDIUM] — DEFERRED

**File**: `src/phase-plot.js`

The ring buffer adds exactly one new point per update. Full redraw is O(count).

**Fix:** Draw onto a persistent offscreen canvas, incrementally update. Or throttle full redraws to every 32 points.

**Status:** Deferred — phase plot already throttled via SIDEBAR_THROTTLE_MASK (every 2nd frame). Incremental canvas update would add complexity for a small sidebar element.

---

### R14. Stats display: 11-object array allocated per `updateSelected()` [MEDIUM] — DONE

**File**: `src/stats-display.js:97-109`

`const forces = [{ row, val, vec }, ...]` creates 11 objects 30 times/second = 330 short-lived objects/second.

**Fix:** Pre-allocate the force descriptor array once in the constructor.

---

### R15. Unconditional `textContent` sets (no change detection) [LOW] — DONE

**File**: `src/stats-display.js:48-63`

17 `.textContent` assignments regardless of whether values changed.

**Fix:** Compare before writing: `if (elem.textContent !== newVal) elem.textContent = newVal;`

---

---

## 3. Memory and Data Structures

### M1. Flatten 11 display force Vec2s to 22 scalars [HIGH] — DEFERRED

**File**: `src/particle.js:19-41`

Each particle has 11 force Vec2 objects (`forceGravity`, `forceCoulomb`, etc.) as separate heap allocations (~704 bytes per particle for display-only data). In the O(N^2) pairwise loop, `pairForce()` dereferences these through pointers — each `p.forceGravity.x += ...` is a pointer load + property write.

**Fix:** Store as flat numeric properties (`fGravX`, `fGravY`, etc.). Eliminates 11 heap allocations per particle and one level of pointer indirection per force accumulation. `resetForces` collapses from 26 pointer-dereferences to 22 direct property writes.

**Status:** Deferred — high effort (touches particle.js, forces.js, renderer.js, integrator.js, stats-display.js, energy.js). V8 hidden class optimization already provides stable property access. Benefit is moderate given all Vec2 properties are declared upfront in the constructor.

---

### M2. Flatten `_f1pnOld` Vec2 to 2 scalars [HIGH] — DEFERRED

**File**: `src/particle.js:41`

The `_f1pnOld` Vec2 wrapper is always accessed as `p._f1pnOld.x/y` in the integrator. Replacing with `_f1pnOldX`/`_f1pnOldY` eliminates one Vec2 allocation per particle and collapses a pointer dereference in the hot path to a direct property read.

**Status:** Deferred — same rationale as M1. Low standalone benefit for a single Vec2.

---

### M3. Inline torus `minImage` in O(N^2) force loop [HIGH] — DONE

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

### M4. Interior fast path for `interpolateWithGradient()` [MEDIUM] — DONE

**File**: `src/scalar-field.js`

`interpolate()`, `gradient()`, and `interpolateWithGradient()` always call `_nb()` 16 times per stencil. `_depositPQS` and `_computeLaplacian` already have interior fast paths. ~90% of particle positions fall in the interior.

**Fix:** Mirror the `_depositPQS` pattern — check if stencil `[ix-1..ix+2]x[iy-1..iy+2]` is fully inside grid, use direct index arithmetic when true. Eliminates 16 `_nb()` calls per interpolation for ~90% of cases. With N=30 and both fields active: 960 `_nb()` calls reduced to ~240 arithmetic operations per substep.

---

### M5. Fuse `modulateMasses` + `applyForces` in Higgs/Axion [MEDIUM] — DEFERRED

**Files**: `src/higgs-field.js:183-255`, `src/axion-field.js`

Each substep, Higgs calls `interpolate()` per particle for mass modulation, then `interpolateWithGradient()` per particle for forces — traversing the 4x4 PQS stencil twice. Same for Axion.

**Fix:** Combine into one method with a single `interpolateWithGradient()` call per particle. For N=30: eliminates 960 `_nb()` calls per substep x 32 substeps = 30,720 calls per frame.

**Status:** Deferred — not feasible. `modulateMasses()` runs before drift/collisions, `applyForces()` runs after. Particle positions change between the two calls, so caching the gradient from mass modulation time would be at the wrong position.

---

### M6. Float32 downgrade for non-integration arrays [MEDIUM] — DEFERRED

**File**: `src/scalar-field.js`

All 10 arrays per field are `Float64Array` (320KB per field, 640KB total). `_energyDensity`, `_gradX/Y`, `_sgPhiFull/GradX/GradY` don't need 64-bit precision.

**Fix:** Downgrade to Float32Array: saves ~160KB total, and Float32 arrays process 2x as many elements per SIMD instruction, reducing memory bandwidth by half for gradient/energy density loops.

**Status:** Deferred — risk of precision artifacts in gradient and energy density computations, especially self-gravity where small Φ corrections compound. Would need validation across all presets.

---

### M7. Signal delay history buffers — interleaved layout [MEDIUM] — DEFERRED

**File**: `src/particle.js:83-90`, `src/signal-delay.js`

Six separate `Float64Array(256)` per particle (12KB total) scattered across six heap regions. The NR solver accesses `histTime[lo]`, `histX[lo]`, `histY[lo]` — five array accesses across five disjoint memory regions per iteration.

**Fix:** Single interleaved buffer: `hist = Float64Array(HISTORY_SIZE * 6)` where `hist[h*6+0]=x, hist[h*6+1]=y, ..., hist[h*6+5]=time`. Each slot's data fits in a single cache line (48 bytes).

**Status:** Deferred — large refactor touching particle.js and signal-delay.js for modest cache locality gain. The NR solver's access pattern (binary search across time) would not benefit as much as sequential access would.

---

### M8. Sparse rho optimization in `_computeCoarsePotential` [LOW-MEDIUM] — DONE

**File**: `src/scalar-field.js:505-524`

O(SG^4) = 4096 operations iterating all 64 source cells for each of 64 observers. With sparse excitation (2-3 particles), only 10-20 coarse cells have nonzero rho.

**Fix:** Pre-build a list of non-vacuum source indices. Inner loop becomes N_nonzero^2 instead of 64^2.

---

### M9. Cap pool size to avoid unbounded GC tracing [LOW] — DONE

**Files**: `src/massless-boson.js:8-9`, `src/pion.js:10-11`

Pool arrays can grow without bound. Each live pool slot is a reference the GC must trace.

**Fix:** `if (_poolSize < MAX_PHOTONS) _pool[_poolSize++] = b;` — drop excess instances.

---

### M10. `applyGravForces` O(4096) existence check [LOW] — DONE

**File**: `src/scalar-field.js:644-647`

Scans all 4096 cells for `rho >= EPSILON` before the main loop. `computeSelfGravity` already has an early exit.

**Fix:** Track `_hasEnergy` boolean set by `computeSelfGravity()`.

---

---

## 4. Main Loop, UI, and Architecture

### A1. Web Worker for physics [HIGH — large effort] — DEFERRED

The physics engine is the dominant CPU consumer. Moving it to a Web Worker would allow physics to run at 128Hz independent of the render thread. Requires SoA particle layout with `SharedArrayBuffer` and COOP/COEP headers for GitHub Pages. Could double effective physics throughput.

**Status:** Deferred — architectural change requiring SharedArrayBuffer + COOP/COEP headers (GitHub Pages compatibility), SoA particle layout conversion, and message-passing protocol design. Prerequisite: M1 (flatten Vec2s to SoA).

---

### A2. KaTeX lazy-load [MEDIUM] — DONE

**File**: `index.html:11, 18-19`

KaTeX CSS (~90KB blocking) + JS (~100KB deferred) loaded unconditionally. Only needed when reference overlay opens (rare).

**Fix:** Lazy-load on first reference open. `createInfoTip()` already guards `typeof renderMathInElement === 'function'`. Convert CSS to non-render-blocking preload.

---

### A3. KaTeX re-render cache [MEDIUM] — DONE

**File**: `src/ui.js:490-502`

Every reference overlay open re-renders all KaTeX from raw HTML + LaTeX. KaTeX rendering costs ~10-50ms per page.

**Fix:** Cache rendered HTML per key in a `Map`. First open pays the cost; subsequent opens are instant.

---

### A4. Batch preset/load DOM events [MEDIUM] — DEFERRED

**Files**: `src/presets.js:603-687`, `src/save-load.js:30-61`

`loadPreset()` fires up to 30 synthetic DOM events. Each toggle `change` event calls `updateAllDeps()` (8 DOM queries). Total: 120 redundant querySelector/closest calls per preset load.

**Fix:** Add a `_batchingPreset` flag. Skip `updateAllDeps()` while batching, call once at the end.

**Status:** Deferred — complex flag threading through presets.js and ui.js. Preset load is a one-time cost (user clicks preset), not a per-frame cost. Low user-facing impact.

---

### A5. `visibilitychange` to halt rAF loop [MEDIUM] — DONE

**File**: `main.js:459`

rAF schedules unconditionally. When tab is hidden, the accumulator can build up to ACCUMULATOR_CAP seconds of physics debt, causing a spike when returning.

**Fix:** Pause rAF scheduling on `visibilitychange` hidden, resume with `lastTime = 0` on visible.

---

### A6. Accumulator cap reduction [MEDIUM] — DONE

**File**: `src/config.js:41`

`ACCUMULATOR_CAP = 4` allows up to 1 second of physics debt.

**Fix:** Reduce to 1 or 2. Limits frame-rate spikes when returning to tab.

---

### A7. Defer shared scripts in `<head>` [MEDIUM] — DONE

**File**: `index.html:14-22`

Seven synchronous `<script>` tags block HTML parsing. All can be `defer`-ed since `main.js` is a module (implicitly deferred) and execution order is preserved.

---

### A8. Throttle `findParticleAt` to rAF [LOW-MEDIUM] — DONE

**File**: `src/input.js:193`

O(N) search on every `mousemove` event. High-refresh displays: 120-240Hz x 100 particles.

**Fix:** Set `_pendingHoverUpdate` flag in mousemove, resolve in rAF callback.

---

### A9. `Array.filter` in `_deleteParticle` [LOW] — DONE

**File**: `src/input.js:156`

Allocates a new array. Should use swap-and-pop like the rest of the codebase.

---

### A10. Tooltip `style.left/top` to `transform` [LOW] — DONE

**File**: `src/input.js:199-200`

`style.left/top` triggers layout. `transform: translate()` only triggers composite.

---

### A11. Eager field initialization [MEDIUM] — DONE

**File**: `main.js:21-155`

`HiggsField` and `AxionField` are created unconditionally in the constructor with full grid allocations, even though most users never enable them.

**Fix:** Lazy-initialize on first toggle-on.

---

### A12. `downloadState` revokeObjectURL timing [LOW] — DONE

**File**: `src/save-load.js:156-161`

`URL.revokeObjectURL` called immediately after `click()` may revoke before download starts.

**Fix:** Revoke in `requestAnimationFrame` callback after click.

---

---

## 5. Prioritized Action Plan

### Tier 1: High Impact, Low-Medium Effort — ALL DONE

| # | Finding | Est. Speedup | Status |
|---|---------|-------------|--------|
| P1 | Quadtree direct child selection | ~2x insert speed | DONE |
| P2 | Yukawa `Math.exp` cutoff | Skip 50%+ of exp calls | DONE |
| R1 | Batch photon alpha drawing | 1024 -> 4 fills | DONE |
| R3 | Batch pion drawing | 256 -> 1 fill | DONE |
| M3 | Inline torus minImage | Eliminate 40K fn calls/frame | DONE |
| P3 | Guard jerk behind radiation flag | 7500 ops/substep saved | DONE |

### Tier 2: Medium Impact, Low-Medium Effort — 6/10 DONE

| # | Finding | Est. Speedup | Status |
|---|---------|-------------|--------|
| M4 | Interior fast path interpolation | 90% fewer `_nb()` calls | DONE |
| R2 | Bucket charged particle glow | O(N) -> O(4) shadowBlur | DONE |
| A2 | KaTeX CSS preload | Non-blocking CSS load | DONE |
| A3 | KaTeX render cache | Skip re-render on reopen | DONE |
| A5 | visibilitychange halt | Prevent accumulator spike | DONE |
| R5 | Batch spin rings | 9N -> 4 canvas calls | DONE |
| M1 | Flatten display force Vec2s | Reduced pointer chasing | DEFERRED |
| M5 | Fuse modulateMasses + applyForces | Halve PQS stencil walks | DEFERRED (infeasible) |
| P6 | Running max acceleration | O(N) -> O(1) per substep | DEFERRED |
| R4 | Reduce trail alpha groups | 4N -> 2N strokes | DEFERRED (visual quality) |
| A4 | Batch preset DOM events | 120 -> 8 DOM queries | DEFERRED |

### Tier 3: Lower Impact or Higher Effort — 10/14 DONE

| # | Finding | Status |
|---|---------|--------|
| P7 | Iterative quadtree insert | DONE |
| P4 | Aberration pre-multiply (+ TDZ bug fix) | DONE |
| M8 | Sparse rho in self-gravity | DONE |
| R11 | Cache clientWidth | DONE |
| R12 | V_eff dirty flag | DONE |
| R14 | Pre-allocate force descriptor | DONE |
| A7 | Defer shared scripts | DONE |
| A8 | Throttle findParticleAt | DONE |
| A11 | Lazy field initialization | DONE |
| A12 | revokeObjectURL timing | DONE |
| P9 | Pre-cache yukMod sqrt | DEFERRED |
| M6 | Float32 for non-integration arrays | DEFERRED |
| M7 | Interleaved history buffers | DEFERRED |
| R9 | Fuse heatmap blur passes | DEFERRED |
| R10 | Heatmap Yukawa cutoff | DEFERRED |
| R13 | Phase plot incremental redraw | DEFERRED |

### Tier 4: Architectural (High Effort, High Reward) — DEFERRED

| # | Finding | Description | Status |
|---|---------|-------------|--------|
| A1 | Web Worker physics | SoA + SharedArrayBuffer, COOP/COEP headers | DEFERRED |
| — | OffscreenCanvas fields | Heatmap/field overlays in dedicated worker | DEFERRED |
| M1+ | Full SoA particle layout | Typed arrays for all particle data | DEFERRED |

### Additional Implemented (not in original tiers)

| # | Finding | Description |
|---|---------|-------------|
| P8 | Subdivide capacity check | Pre-check pool capacity once instead of 4x |
| P10 | Precompute `_needAxMod` | Per-frame flag instead of per-pair evaluation |
| P11 | Conditional photon renorm | Skip renormalization when `\|v²-1\| < ε` |
| P12 | `invR*invRSq` in boson gravity | 2 multiplies instead of 3 |
| P13 | Merge KE scan into quadrupole | Eliminate separate O(N) pass |
| P16 | Dead-particle GC per-frame | Check once per frame instead of per substep |
| P17 | Indexed loop for rocheTransfers | Eliminate iterator allocation |
| R6 | Batch ergospheres + antimatter | One `setLineDash` pass instead of per-particle |
| R7 | Eliminate `ctx.save()/restore()` | Explicit property write-then-restore |
| R8 | Cache `magSq` | Avoid recomputing `fx*fx+fy*fy` |
| R15 | textContent change detection | Compare before writing DOM |
| M9 | Pool caps (64) | Prevent unbounded pool growth |
| M10 | `hasEnergy` flag | Skip O(4096) existence check |
| A6 | Accumulator cap = 2 | Reduce from 4 to limit frame spikes |
| A9 | Swap-and-pop in `_deleteParticle` | Replace `Array.filter` |
| A10 | Tooltip `transform` positioning | Avoid layout trigger from `style.left/top` |
