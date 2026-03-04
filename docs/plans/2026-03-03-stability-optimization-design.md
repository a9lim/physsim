# Physsim Stability & Optimization Design

**Date**: 2026-03-03
**Target**: 50-100 particles, accuracy and stability focus
**Approach**: Approach B — Stability-First Refactor

## Context

Audit of the physsim simulation identified issues across physics accuracy, numerical stability, GC pressure, and code organization. This design addresses all of them without over-engineering for scale beyond the 50-100 particle target.

## 1. Fixed-Timestep Physics Loop

**Problem**: Physics dt is tied to frame rate. Slow frames cause large dt, causing instability.

**Solution**: Accumulator-based fixed timestep in `main.js` loop.

- `PHYSICS_DT = 1/120` (fixed tick rate before speedScale)
- Accumulator collects `rawDt * speedScale` per frame
- While loop drains accumulator in fixed-size steps
- Capped by `MAX_SUBSTEPS * PHYSICS_DT` to prevent spiral of death
- Photon updates and tidal breakup move inside the fixed-step loop
- Energy computation and DOM updates stay per-frame (outside loop)
- `speedScale` multiplies into accumulator, not into PHYSICS_DT — changing speed changes tick count per frame, not tick granularity

**New constant**: `PHYSICS_DT` in config.js.

## 2. Targeted Physics Fixes

### 2a. Lazy History Buffers

Remove 5x Float64Array(512) from Particle constructor. Add `_initHistory()` called on first use in signal-delay recording. Saves ~20KB/particle when signal delay is off.

### 2b. Softening in Tidal Breakup

Add `SOFTENING_SQ` to `distSq` in `checkTidalBreakup`, matching force computation. Prevents infinite tidal acceleration at close range.

### 2c. Scale Bounce Overlap Push-out

Replace hardcoded `+ 0.25` in `resolveBounce` with `+ minDist * 0.01`. Scales with particle size.

### 2d. Time-Based Photon Lifetime

Change `Photon.lifetime` to accumulate `dt` instead of `++`. Change `PHOTON_LIFETIME` from 300 frames to 30 sim-time-units. Consistent regardless of frame rate or speed scale.

### 2e. Write-Pointer Compaction for Despawn

Replace `particles.splice(i, 1)` in boundary despawn with write-pointer compaction pattern (already used in merge cleanup). Mark for removal, compact in one pass.

### 2f. Cyclotron Frequency in Adaptive Substepping

After acceleration-based `dtSafe`, also compute `dtCyclotron = 2pi / (max |q*Bz/m|)`. Take `dtSafe = min(dtSafe_accel, dtCyclotron / 8)`. Ensures Boris rotation resolves at least 8 steps per cyclotron orbit.

## 3. QuadTree Node Pooling

**Problem**: Every substep constructs a fresh QuadTree — ~50-130 nodes with Rect and Vec2 allocations. Up to 125K short-lived objects/second at max substeps.

**Solution**: `QuadTreePool` with pre-allocated flat node array.

- 512 pre-allocated node objects (generous for 100 particles, ~4N typical)
- Boundary and center-of-mass stored as inline fields (no Rect/Vec2 objects)
- Points stored in fixed-size array of capacity QUADTREE_CAPACITY, nulled on reset
- Children stored as pool indices (-1 = none), not object references
- `reset()` sets counter to 0; `alloc()` returns next index
- Falls back to dynamic allocation if pool exhausted
- Physics holds one pool instance, reused across all substeps and frames
- External API unchanged: insert(), query(), calculateMassDistribution()
- Rect class still exported for collision query ranges (not worth pooling)

## 4. Physics Module Split

**Problem**: Physics class is 930 lines handling integration, forces, collisions, PE, tidal, and signal delay.

**New file structure**:

```
src/integrator.js    — Physics class (substep loop, Boris rotation, tidal breakup) ~250 lines
src/forces.js        — computeAllForces, resetForces, _pairForce, calculateForce (BH walk)
src/collisions.js    — handleCollisions, resolveMerge, resolveBounce (pure functions)
src/potential.js     — computePE, _treePE, _pairPE
src/signal-delay.js  — getDelayedState, interpolateHistory (pure functions)
```

- Toggle state lives on Physics instance in integrator.js
- Force/PE/collision functions receive toggles as parameters
- All external API unchanged — main.js imports Physics from integrator.js
- checkTidalBreakup stays in integrator.js (small, part of update loop)

## 5. Stats Display Extraction

**Problem**: Simulation class mixes orchestration with 15+ DOM writes per frame.

**Solution**: `StatsDisplay` class in `src/stats-display.js`.

- Takes cached DOM ref objects (dom, selDom) in constructor
- `updateEnergy(particles, physics, sim)` — formats and writes energy/momentum/drift
- `updateSelected(particle, physics)` — formats and writes selected particle info
- Owns initialEnergy/initialMomentum/initialAngMom baseline tracking
- `resetBaseline()` replaces scattered null assignments
- SankeyOverlay update call moves here
- Simulation class no longer caches stat DOM refs

## Implementation Order

1. Fixed-timestep loop (highest stability impact, foundational)
2. Targeted physics fixes (independent, can be done in parallel)
3. QuadTree pooling (independent of module split)
4. Physics module split (cleanest when done on final code)
5. Stats display extraction (lowest priority, purely organizational)

## Non-Goals

- SoA particle layout (overkill for 50-100 particles)
- WebWorker offloading (unnecessary at this scale)
- Specialized force kernels (branch cost negligible at this scale)
- Reducing SOFTENING constant (deferred — needs careful substep tuning)
