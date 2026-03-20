# Comprehensive Performance Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 52 performance fixes from the 2026-03-19 audit across CPU physics, scalar fields, renderer, GPU backend, GPU shaders, and dead code cleanup.

**Architecture:** 13 file-clustered phases — each file edited once, ordered by dependency. No build step, no test framework; verification is manual (load presets, check stats, toggle forces).

**Tech Stack:** Vanilla JS ES6 modules, WebGPU/WGSL, Canvas 2D. No dependencies.

**Spec:** `docs/superpowers/specs/2026-03-19-comprehensive-perf-audit-design.md`

---

## Verification Protocol (all phases)

After every phase commit, run this checklist:
1. Serve from `a9lim.github.io/`: `python -m http.server`
2. Open `/physsim` — no console errors
3. Load presets 1-9 via keyboard, then remaining via dropdown — no NaN, no crash
4. Toggle CPU ↔ GPU in Engine tab — consistent behavior
5. Toggle all force/physics switches — no regressions
6. Check Stats tab: energy/momentum values are finite and reasonable
7. Test Torus/Klein/RP² boundary modes

---

## Task 1: Phase 1 — Dead Code & Cleanup

**Files:**
- Modify: `src/vec2.js:17-38`
- Modify: `src/massless-boson.js:57-60`
- Modify: `src/pion.js:66-69`
- Modify: `src/renderer.js:1`
- Modify: `src/relativity.js:4`
- Modify: `src/ui.js:4`
- Modify: `src/heatmap.js:112`
- Modify: `src/collisions.js:24,112`
- Modify: `src/potential.js:66,142`
- Modify: `src/cpu-physics.js:24`
- Modify: `src/canvas-renderer.js:19`
- Modify: `styles.css:112`
- Modify: `src/forces.js:710-887`
- Modify: `src/boson-utils.js`

### D1: Delete dead Vec2 methods

- [ ] **Step 1: Remove `scale()`, `mag()`, `normalize()` from `src/vec2.js`**

Delete these 3 methods (keep `set`, `clone`, `magSq`, `dist`):

```js
// DELETE these methods:
scale(s) { this.x *= s; this.y *= s; return this; }

mag() { return Math.sqrt(this.x * this.x + this.y * this.y); }

normalize() {
    const m = this.mag();
    if (m > 0) { this.x /= m; this.y /= m; }
    return this;
}
```

### D2: Delete dead `clearPool()` methods

- [ ] **Step 2: Remove `clearPool()` from `src/massless-boson.js:57-60`**

Delete:
```js
static clearPool() { _poolSize = 0; _pool.length = 0; }
```

- [ ] **Step 3: Remove `clearPool()` from `src/pion.js:66-69`**

Same pattern — delete the static method.

### D3-D5: Remove unused imports

- [ ] **Step 4: In `src/renderer.js:1`, remove `PI` from the import**

Change:
```js
import { PI, TWO_PI, HALF_PI, ... } from './config.js';
```
To:
```js
import { TWO_PI, HALF_PI, ... } from './config.js';
```

- [ ] **Step 5: In `src/relativity.js:4`, remove `EPSILON` from the import**

Remove `EPSILON` from the `import { ... } from './config.js'` statement.

- [ ] **Step 6: In `src/ui.js:4`, remove `DEFAULT_SPEED_INDEX` from the import**

Remove `DEFAULT_SPEED_INDEX` from the config import statement.

### D6-D7: Un-export internal-only symbols

- [ ] **Step 7: In `src/heatmap.js:112`, remove `export` keyword from `HEATMAP_MODES`**

Change `export const HEATMAP_MODES` to `const HEATMAP_MODES`.

- [ ] **Step 8: In `src/collisions.js:112`, remove `export` from `resolveMerge`**

Change `export function resolveMerge` to `function resolveMerge`.

- [ ] **Step 9: In `src/potential.js`, remove `export` from `treePE` (line ~66) and `pairPE` (line ~142)**

Both are only called internally by `computePE()`.

### D8: Delete unused `_engine` getters

- [ ] **Step 10: In `src/cpu-physics.js:24`, delete the `get _engine()` getter**

- [ ] **Step 11: In `src/canvas-renderer.js:19`, delete the `get _engine()` getter**

### D9: Delete dead CSS rule

- [ ] **Step 12: In `styles.css:112`, delete the `.tog-tidal` rule**

```css
/* DELETE: */
.tog-tidal { --tog-color: var(--ext-slate); }
```

### D10: Remove unused params from collision functions

- [ ] **Step 13: In `src/collisions.js`, remove `bounceFriction` param from `handleCollisions`**

Current signature (line ~24):
```js
export function handleCollisions(particles, pool, root, mode, bounceFriction, relativityEnabled, periodic, domW, domH, topology) {
```

New signature:
```js
export function handleCollisions(particles, pool, root, mode, periodic, domW, domH, topology) {
```

Remove `bounceFriction`, `relativityEnabled` params. Search the function body to confirm neither is referenced.

- [ ] **Step 14: Update the caller in `src/integrator.js`**

Find the `handleCollisions()` call site and remove the corresponding arguments.

- [ ] **Step 15: In `src/collisions.js`, remove `relativityEnabled` and `periodic` params from `resolveMerge`**

Current signature (line ~112):
```js
function resolveMerge(p1, p2, relativityEnabled, periodic, miDx, miDy) {
```

New signature:
```js
function resolveMerge(p1, p2, miDx, miDy) {
```

Update all call sites within `handleCollisions`.

### D12: Extract shared boson tree walk skeleton

- [ ] **Step 16: In `src/forces.js`, extract `_walkBosonTreeCore` from `_walkBosonTree` and `_walkBosonTreeCharge`**

Both functions at lines ~710-765 and ~830-887 share identical tree walk structure (stack-based BH walk with theta criterion, `minImage` wrapping, softening). They differ only in:
- Which aggregate field to read (`totalMass` vs `totalCharge`)
- Skip condition (`mass < EPSILON` vs `charge === 0`)
- Accumulation formula

Extract a shared skeleton that takes a callback for the leaf/aggregate force computation:

```js
function _walkBosonTreeCore(pool, rootIdx, px, py, softeningSq, periodic, domW, domH, halfDomW, halfDomH, topology, leafFn, aggregateFn) {
    // Identical stack-based BH walk
    // At leaf: call leafFn(boson, dx, dy, distSq)
    // At aggregate: call aggregateFn(nodeIdx, dx, dy, distSq)
}
```

Then `_walkBosonTree` and `_walkBosonTreeCharge` become thin wrappers passing their specific leaf/aggregate lambdas. Keep `treeDeflectBoson` in `boson-utils.js` unchanged (different tree, different output).

### D11: Document potential.js as fallback

- [ ] **Step 17: Add a comment at the top of `src/potential.js`**

```js
// FALLBACK ONLY: PE is accumulated inline in pairForce() via _peAccum during simulation.
// This module is used only for preset-load recomputation when the force loop hasn't run yet.
```

- [ ] **Step 18: Commit Phase 1**

```bash
git add src/vec2.js src/massless-boson.js src/pion.js src/renderer.js src/relativity.js src/ui.js src/heatmap.js src/collisions.js src/potential.js src/cpu-physics.js src/canvas-renderer.js styles.css src/forces.js src/integrator.js src/boson-utils.js
git commit -m "perf(cleanup): dead code removal, unused imports/exports/params — Phase 1"
```

- [ ] **Step 19: Run verification protocol**

---

## Task 2: Phase 2 — Quadtree Optimizations

**Files:**
- Modify: `src/quadtree.js:117-119, 151-181, 224, 360`

### C1: Unroll `_subdivide()` to eliminate array allocations

- [ ] **Step 1: Replace array-based child initialization in `_subdivide()`**

Current code (lines ~117-135):
```js
const ids = [c, c + 1, c + 2, c + 3];
const xs = [x - hw, x + hw, x - hw, x + hw];
const ys = [y - hh, y - hh, y + hh, y + hh];
for (let k = 0; k < 4; k++) {
    const id = ids[k];
    this.bx[id] = xs[k]; this.by[id] = ys[k];
    // ... zero all aggregate fields ...
}
```

Replace with 4 inline blocks — no array allocation:
```js
_subdivide(idx) {
    const x = this.bx[idx], y = this.by[idx];
    const hw = this.bw[idx] * 0.5, hh = this.bh[idx] * 0.5;
    if (this.count + 4 > this.maxNodes) this._grow();
    const c = this.count;
    this.count = c + 4;

    // NW (c)
    this.bx[c] = x - hw; this.by[c] = y - hh; this.bw[c] = hw; this.bh[c] = hh;
    this.comX[c] = x - hw; this.comY[c] = y - hh;
    this.totalMass[c] = 0; this.totalCharge[c] = 0;
    this.totalMagneticMoment[c] = 0; this.totalAngularMomentum[c] = 0;
    this.totalMomentumX[c] = 0; this.totalMomentumY[c] = 0;
    this.nw[c] = -1; this.ne[c] = -1; this.sw[c] = -1; this.se[c] = -1;
    this.pointCount[c] = 0; this.divided[c] = 0;

    // NE (c+1)
    this.bx[c+1] = x + hw; this.by[c+1] = y - hh; this.bw[c+1] = hw; this.bh[c+1] = hh;
    this.comX[c+1] = x + hw; this.comY[c+1] = y - hh;
    this.totalMass[c+1] = 0; this.totalCharge[c+1] = 0;
    this.totalMagneticMoment[c+1] = 0; this.totalAngularMomentum[c+1] = 0;
    this.totalMomentumX[c+1] = 0; this.totalMomentumY[c+1] = 0;
    this.nw[c+1] = -1; this.ne[c+1] = -1; this.sw[c+1] = -1; this.se[c+1] = -1;
    this.pointCount[c+1] = 0; this.divided[c+1] = 0;

    // SW (c+2)
    this.bx[c+2] = x - hw; this.by[c+2] = y + hh; this.bw[c+2] = hw; this.bh[c+2] = hh;
    this.comX[c+2] = x - hw; this.comY[c+2] = y + hh;
    this.totalMass[c+2] = 0; this.totalCharge[c+2] = 0;
    this.totalMagneticMoment[c+2] = 0; this.totalAngularMomentum[c+2] = 0;
    this.totalMomentumX[c+2] = 0; this.totalMomentumY[c+2] = 0;
    this.nw[c+2] = -1; this.ne[c+2] = -1; this.sw[c+2] = -1; this.se[c+2] = -1;
    this.pointCount[c+2] = 0; this.divided[c+2] = 0;

    // SE (c+3)
    this.bx[c+3] = x + hw; this.by[c+3] = y + hh; this.bw[c+3] = hw; this.bh[c+3] = hh;
    this.comX[c+3] = x + hw; this.comY[c+3] = y + hh;
    this.totalMass[c+3] = 0; this.totalCharge[c+3] = 0;
    this.totalMagneticMoment[c+3] = 0; this.totalAngularMomentum[c+3] = 0;
    this.totalMomentumX[c+3] = 0; this.totalMomentumY[c+3] = 0;
    this.nw[c+3] = -1; this.ne[c+3] = -1; this.sw[c+3] = -1; this.se[c+3] = -1;
    this.pointCount[c+3] = 0; this.divided[c+3] = 0;

    this.nw[idx] = c; this.ne[idx] = c + 1; this.sw[idx] = c + 2; this.se[idx] = c + 3;
    this.divided[idx] = 1;
}
```

### C10: Split polymorphic insert work stack

- [ ] **Step 2: Replace mixed-type work stack with two typed stacks**

Current code (lines ~151-181):
```js
const work = this._insertWork || (this._insertWork = []);
work.length = 0;
work.push(idx, particle);
while (work.length > 0) {
    let pt = work.pop();
    let nodeIdx = work.pop();
    // ...
    work.push(childIdx, pt);
}
```

Replace with separate stacks (module-level pre-allocated):
```js
// At module level:
let _workNodeStack = new Int32Array(64);
let _workPartStack = new Array(64);
let _workTop = 0;

// In insert():
insert(rootIdx, particle) {
    _workNodeStack[0] = rootIdx;
    _workPartStack[0] = particle;
    _workTop = 1;

    while (_workTop > 0) {
        _workTop--;
        const nodeIdx = _workNodeStack[_workTop];
        const pt = _workPartStack[_workTop];

        // ... existing logic, but push becomes:
        if (_workTop >= _workNodeStack.length) {
            const newLen = _workNodeStack.length * 2;
            const nn = new Int32Array(newLen);
            nn.set(_workNodeStack);
            _workNodeStack = nn;
            const np = new Array(newLen);
            for (let k = 0; k < _workPartStack.length; k++) np[k] = _workPartStack[k];
            _workPartStack = np;
        }
        _workNodeStack[_workTop] = childIdx;
        _workPartStack[_workTop] = pt;
        _workTop++;
    }
}
```

### H10: Remove dead `rSq` read

- [ ] **Step 3: Delete `const rSq = p.radiusSq;` at line ~224 in `calculateMassDistribution`**

### M5: Replace `for...of` with indexed loop

- [ ] **Step 4: In `build()` (line ~360), replace:**

```js
for (const p of particles) this.insert(root, p);
```
With:
```js
for (let i = 0, n = particles.length; i < n; i++) this.insert(root, particles[i]);
```

- [ ] **Step 5: Commit Phase 2**

```bash
git add src/quadtree.js
git commit -m "perf(quadtree): eliminate subdivide allocations, typed insert stack — Phase 2"
```

- [ ] **Step 6: Run verification protocol**

---

## Task 3: Phase 3 — Forces Optimizations

**Files:**
- Modify: `src/forces.js:391, 437`
- Modify: `src/potential.js:155-156`

### C4: Replace `Math.cbrt` with `Math.sqrt` in tidal locking

- [ ] **Step 1: In `src/forces.js:437`, replace tidal locking body radius computation**

Current:
```js
const ri5 = p.bodyRadiusSq * p.bodyRadiusSq * Math.cbrt(p.mass);
```

Replace with:
```js
const bodyR = Math.sqrt(p.bodyRadiusSq);
const ri5 = p.bodyRadiusSq * p.bodyRadiusSq * bodyR;
```

Mathematical equivalence: `bodyRadiusSq = cbrt(mass)²`, so `sqrt(bodyRadiusSq) = cbrt(mass)`. `sqrt` is ~2x faster than `cbrt`.

Do the same for `sRi5` on the source side (check a few lines below — the source particle's tidal term may also use `cbrt`).

### C5: Guard yukMod/higgsMod sqrt behind toggle checks

- [ ] **Step 2: In `src/forces.js:391`, guard yukMod sqrt**

Current:
```js
const yukModPair = Math.sqrt(p.yukMod * sYukMod);
```

Replace with:
```js
const yukModPair = _needAxMod ? Math.sqrt(p.yukMod * sYukMod) : 1;
```

Note: `_needAxMod` is already set to `(coulombEnabled || magneticEnabled) && axionEnabled` in `computeAllForces`. When axion is off, both `yukMod` values are 1.0, so the guard avoids a redundant `sqrt(1)`.

Note: The higgsMod guard at `forces.js:388` (`muEff = toggles.higgsEnabled ? ... : yukawaMu`) already exists — no change needed there.

- [ ] **Step 3: In `src/potential.js:155-156`, apply same guards**

```js
const axModPair = this._needAxMod ? Math.sqrt(p.axMod * sAxMod) : 1;
const yukModPair = _needAxMod ? Math.sqrt(p.yukMod * sYukMod) : 1;
```

Note: `potential.js` is fallback-only (preset load), so this is correctness cleanup, not hot-path perf.

- [ ] **Step 5: Commit Phase 3**

```bash
git add src/forces.js src/potential.js
git commit -m "perf(forces): sqrt guards for yukMod/higgsMod, cbrt→sqrt tidal — Phase 3"
```

- [ ] **Step 6: Run verification protocol**

Pay special attention to: Yukawa preset, Higgs Boson preset, Axion preset — these exercise the guarded paths.

---

## Task 4: Phase 4 — Integrator Loop Fusion

**Files:**
- Modify: `src/integrator.js:597-725`

### C6: Fuse spin-orbit and torque loops

- [ ] **Step 1: Merge the spin-orbit loop (lines ~679-707) and torque loop (lines ~710-725) into one**

Current: Two separate `for` loops over all particles with different guards:
```js
// Loop 1: spin-orbit (lines 679-707)
if (this.spinOrbitEnabled && (hasMagnetic || hasGM)) {
    for (let i = 0; i < n; i++) { /* Stern-Gerlach, Mathisson-Papapetrou, spin-orbit energy */ }
}
// Loop 2: torques (lines 710-725)
if ((hasGM && relOn) || hasGrav || bounce) {
    for (let i = 0; i < n; i++) { /* frame-drag, tidal, contact torque */ }
}
```

Merge into single loop:
```js
const needSpinOrbit = this.spinOrbitEnabled && (hasMagnetic || hasGM);
const needTorques = (hasGM && relOn) || hasGrav || bounce;
if (needSpinOrbit || needTorques) {
    for (let i = 0; i < n; i++) {
        const p = particles[i];
        if (needSpinOrbit) {
            // ... existing spin-orbit body (Stern-Gerlach, MP, spin-orbit energy) ...
        }
        if (needTorques) {
            // ... existing torque body (frame-drag, tidal, contact) ...
        }
    }
}
```

This eliminates one full particle iteration and one redundant `angwToAngVel()` computation per substep.

### H2: Defer pion `_syncVel()` calls

- [ ] **Step 2: In `src/integrator.js`, find `applyBosonBosonGravity()` and `applyPionPionCoulomb()` calls**

In the per-substep section, after both boson interaction calls, add a batch `_syncVel()`:

```js
// After both boson interaction calls:
if (bosonInteractionEnabled) {
    applyBosonBosonGravity(photons, pions, dtSub, this._bosonPool, bosonRoot);
    applyPionPionCoulomb(pions, dtSub, this._bosonPool, bosonRoot);
    // Batch sync all pions once (instead of per-walk):
    for (let i = 0; i < pions.length; i++) {
        if (pions[i].alive) pions[i]._syncVel();
    }
}
```

Then in `applyBosonBosonGravity()` and `applyPionPionCoulomb()` in `forces.js`, remove the per-pion `_syncVel()` calls at the end of each function.

### H8: Fuse adaptive dt estimation with half-kick

- [ ] **Step 3: Move maxAccel computation into the half-kick loop (one substep behind)**

Current: A separate loop (lines ~597-617) computes `maxAccelSq` by iterating all particles before the kick.

Change: Track `maxAccelSq` from the *previous* substep's kick. Initialize with current value on first substep, then update during each half-kick:

```js
// Before substep loop — compute initial maxAccel on first substep only:
if (substep === 0) {
    // ... existing maxAccel loop (only runs once, not per substep) ...
}

// Inside half-kick loop, accumulate maxAccel for NEXT substep:
let nextMaxAccelSq = 0;
for (let i = 0; i < n; i++) {
    const p = particles[i];
    // ... existing half-kick code ...
    // At end of particle: track max accel for next substep
    const aSq = p.forceGravity.x * p.forceGravity.x + p.forceGravity.y * p.forceGravity.y; // simplified
    // Actually: use total force magnitude / mass²
    const fx = /* sum of all force x */ , fy = /* sum of all force y */;
    const invMSq = 1 / (p.mass * p.mass);
    const accelSq = (fx * fx + fy * fy) * invMSq;
    if (accelSq > nextMaxAccelSq) nextMaxAccelSq = accelSq;
}
maxAccelSq = nextMaxAccelSq;
```

**Caution**: This changes adaptive behavior subtly. The first substep uses the correctly-computed value; subsequent substeps use the previous substep's value. Forces are continuous, so this is a valid approximation. Verify that energy drift doesn't increase measurably with the Orbit and Binary Star presets.

- [ ] **Step 4: Commit Phase 4**

```bash
git add src/integrator.js src/forces.js
git commit -m "perf(integrator): fuse spin-orbit+torque loops, defer pion sync — Phase 4"
```

- [ ] **Step 5: Run verification protocol**

Extra checks: Load Binary Star preset, let it run 30s — energy drift should be similar to before.

---

## Task 5: Phase 5 — FFT & Scalar Field Base

**Files:**
- Modify: `src/fft.js:69-70, 87-88`
- Modify: `src/scalar-field.js:490-498, 547-576, 625-637`

### C7: Pre-allocate FFT temporary arrays

- [ ] **Step 1: In `src/fft.js`, move temp arrays to module level**

Current (inside `fft2d()`):
```js
const rowRe = new Float64Array(N);
const rowIm = new Float64Array(N);
// ... and later:
const colRe = new Float64Array(N);
const colIm = new Float64Array(N);
```

Replace with module-level lazy allocation:
```js
// Module level:
let _rowRe = null, _rowIm = null, _colRe = null, _colIm = null;
let _fftBufSize = 0;

function _ensureFftBufs(N) {
    if (N > _fftBufSize) {
        _rowRe = new Float64Array(N);
        _rowIm = new Float64Array(N);
        _colRe = new Float64Array(N);
        _colIm = new Float64Array(N);
        _fftBufSize = N;
    }
}

// In fft2d():
export function fft2d(re, im, N, inverse) {
    _ensureFftBufs(N);
    // Use _rowRe, _rowIm, _colRe, _colIm instead of local arrays
    // ...
}
```

### C14: Skip `_sgPhiFull` copy — use `_fftRe` directly

- [ ] **Step 2: In `src/scalar-field.js:625-637`, remove the copy loop**

Current:
```js
// After IFFT:
for (let i = 0; i < total; i++) phi[i] = re[i];
this._computeSelfGravGradients(bcMode, topoConst);
```

Replace:
```js
// After IFFT: _fftRe already contains the potential
// Point _computeSelfGravGradients to read from _fftRe directly
this._computeSelfGravGradients(bcMode, topoConst, this._fftRe);
```

Update `_computeSelfGravGradients` to accept a source array parameter (default `this._sgPhiFull` for backward compat):
```js
_computeSelfGravGradients(bcMode, topoConst, srcArray) {
    const phi = srcArray || this._sgPhiFull;
    // ... existing gradient code using phi[...] ...
}
```

Verify: `_fftRe` is not overwritten between the IFFT call and `_computeSelfGravGradients`. The FFT writes to `_fftRe` in-place, then gradient computation only reads from it.

### C15: Fuse `_computeEnergyDensity` + `_addPotentialEnergy`

- [ ] **Step 3: Add `potentialFn` parameter to `_computeEnergyDensity`**

Current (lines ~490-498):
```js
_computeEnergyDensity(domainW, domainH) {
    // ... computes KE + gradient energy ...
    this._addPotentialEnergy(this._energyDensity);
}
```

Change to:
```js
_computeEnergyDensity(domainW, domainH, potentialFn) {
    const rho = this._energyDensity;
    const field = this.field;
    const fd = this.fieldDot;
    const gx = this._gradX, gy = this._gradY;
    const cellWSq = (domainW / GRID) ** 2;
    const cellHSq = (domainH / GRID) ** 2;
    const total = GRID * GRID;
    for (let i = 0; i < total; i++) {
        const f = field[i];
        rho[i] = 0.5 * fd[i] * fd[i]
               + 0.5 * (gx[i] * gx[i] / cellWSq + gy[i] * gy[i] / cellHSq)
               + potentialFn(f);
    }
}
```

Then in `HiggsField`:
```js
_computeEnergyDensity(domainW, domainH) {
    const muSq = 0.5 * this.mass * this.mass; // same as local computation in update()
    const vacOffset = 0.25 * muSq;
    super._computeEnergyDensity(domainW, domainH, f => {
        const fSq = f * f;
        return muSq * (-0.5 * fSq + 0.25 * fSq * fSq) + vacOffset;
    });
}
```

And in `AxionField`:
```js
_computeEnergyDensity(domainW, domainH) {
    const halfMaSq = 0.5 * this._maSq;
    super._computeEnergyDensity(domainW, domainH, f => halfMaSq * f * f);
}
```

Remove `_addPotentialEnergy` overrides from both subclasses — no longer needed.

- [ ] **Step 4: Commit Phase 5**

```bash
git add src/fft.js src/scalar-field.js src/higgs-field.js src/axion-field.js
git commit -m "perf(fields): pre-alloc FFT bufs, skip sgPhi copy, fuse energy density — Phase 5"
```

- [ ] **Step 5: Run verification protocol**

Extra: Load Higgs Boson and Phase Transition presets — check Stats tab for field energy values.

---

## Task 6: Phase 6 — Higgs & Axion Field Optimizations

**Files:**
- Modify: `src/higgs-field.js:55-68, 95-133`
- Modify: `src/axion-field.js:89-133, 209`

### C8: Fuse source + thermal deposition

- [ ] **Step 1: In `src/higgs-field.js`, merge `_depositSources` and `_depositThermal` into one particle loop**

Current (lines ~60-68):
```js
this._source.fill(0);
this._depositSources(particles, domainW, domainH);
this._thermal.fill(0);
this._depositThermal(particles, domainW, domainH);
```

Replace with a fused deposition method:
```js
this._source.fill(0);
this._thermal.fill(0);
this._depositSourcesAndThermal(particles, domainW, domainH);
```

In the new method, compute PQS coordinates once per particle, then deposit into both arrays:
```js
_depositSourcesAndThermal(particles, domainW, domainH) {
    for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        if (!p.alive) continue;
        const { ix, iy, wx, wy } = this._pqsCoords(p.pos.x, p.pos.y, domainW, domainH);

        // Source deposition (existing _depositSources logic per particle)
        const sourceVal = /* existing source computation */ ;
        this._depositPQSWeighted(ix, iy, wx, wy, sourceVal, this._source);

        // Thermal deposition (existing _depositThermal logic per particle)
        const wSq = p.w.x * p.w.x + p.w.y * p.w.y;
        const ke = wSq / (Math.sqrt(1 + wSq) + 1) * p.mass;
        this._depositPQSWeighted(ix, iy, wx, wy, ke, this._thermal);
    }
}
```

Need to add a `_depositPQSWeighted` that takes pre-computed weights and a target array, or refactor `_depositPQS` to accept pre-computed coords. The key optimization is avoiding duplicate `_pqsCoords()` calls.

### C16: Inline Laplacian + viscosity into kick loops

- [ ] **Step 2: In both `higgs-field.js` and `axion-field.js`, inline the 5-point stencil computations**

This is the most complex change. Currently each half-kick calls `_computeLaplacian()` and `_computeViscosity()` before the kick loop. These each do a full grid traversal. Inlining them into the kick loop eliminates 2 grid passes per kick (4 total per `update()`).

For the **interior cells** (1 to GRID-2 in both dimensions), the Laplacian is:
```js
const lap = (field[i-1] + field[i+1] - 2*field[i]) * invCWSq
          + (field[i-GRID] + field[i+GRID] - 2*field[i]) * invCHSq;
```

And viscosity is:
```js
const visc = nu * ((fieldDot[i-1] + fieldDot[i+1] - 2*fieldDot[i]) * invCWSq
                  + (fieldDot[i-GRID] + fieldDot[i+GRID] - 2*fieldDot[i]) * invCHSq);
```

For **border cells** (first/last row/column), use `_nb()` for topology-aware neighbor lookup.

Pattern for the kick loop:
```js
// Pre-compute constants:
const invCWSq = 1 / (cellW * cellW);
const invCHSq = 1 / (cellH * cellH);
const nu = 1 / (2 * Math.sqrt(invCWSq + invCHSq));

for (let iy = 0; iy < GRID; iy++) {
    for (let ix = 0; ix < GRID; ix++) {
        const i = iy * GRID + ix;
        let lap, visc;

        if (ix > 0 && ix < GRID - 1 && iy > 0 && iy < GRID - 1) {
            // Interior fast path
            lap = (field[i-1] + field[i+1] - 2*field[i]) * invCWSq
                + (field[i-GRID] + field[i+GRID] - 2*field[i]) * invCHSq;
            visc = nu * ((fieldDot[i-1] + fieldDot[i+1] - 2*fieldDot[i]) * invCWSq
                       + (fieldDot[i-GRID] + fieldDot[i+GRID] - 2*fieldDot[i]) * invCHSq);
        } else {
            // Border path with _nb()
            const fL = field[this._nb(ix-1, iy, bcMode, topoConst)];
            const fR = field[this._nb(ix+1, iy, bcMode, topoConst)];
            const fU = field[this._nb(ix, iy-1, bcMode, topoConst)];
            const fD = field[this._nb(ix, iy+1, bcMode, topoConst)];
            lap = (fL + fR - 2*field[i]) * invCWSq + (fU + fD - 2*field[i]) * invCHSq;

            const dL = fieldDot[this._nb(ix-1, iy, bcMode, topoConst)];
            const dR = fieldDot[this._nb(ix+1, iy, bcMode, topoConst)];
            const dU = fieldDot[this._nb(ix, iy-1, bcMode, topoConst)];
            const dD = fieldDot[this._nb(ix, iy+1, bcMode, topoConst)];
            visc = nu * ((dL + dR - 2*fieldDot[i]) * invCWSq + (dU + dD - 2*fieldDot[i]) * invCHSq);
        }

        // ... existing kick computation using lap and visc ...
        fieldDot[i] += (lap + /* potential terms */ + visc + /* SG/portal terms */) * dt;
    }
}
```

Apply this pattern to both half-kicks in both `higgs-field.js` and `axion-field.js`. Remove the `_computeLaplacian()` and `_computeViscosity()` calls from both `update()` methods.

**Important**: The `_laplacian` array on ScalarField can be removed if no other code reads it. Check that `_computeLaplacian` is not called from anywhere else. If it's only used by the subclass `update()` methods, it can be left as dead code for now (or removed from scalar-field.js).

### C18: Optimize polynomial evaluation in Higgs kick

- [ ] **Step 3: In `src/higgs-field.js` kick loops, precompute phi powers**

Current pattern inside kick loop:
```js
muSq * phiVal * phiVal * phiVal
```

Replace with:
```js
const phiSq = phiVal * phiVal;
const phiCu = phiSq * phiVal;
// Then use phiCu instead of phiVal*phiVal*phiVal
// And phiSq*phiSq instead of phiVal*phiVal*phiVal*phiVal
```

### A2: Early return in `interpolateAxMod`

- [ ] **Step 4: In `src/axion-field.js:209`, add early return**

At the top of `interpolateAxMod()`:
```js
interpolateAxMod(particles, coulombEnabled, yukawaEnabled) {
    if (!coulombEnabled && !yukawaEnabled) {
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (!p.alive) continue;
            p.axMod = 1;
            p.yukMod = 1;
        }
        return;
    }
    // ... existing interpolation code ...
}
```

- [ ] **Step 5: Commit Phase 6**

```bash
git add src/higgs-field.js src/axion-field.js src/scalar-field.js
git commit -m "perf(fields): fuse deposition, inline Laplacian+viscosity, phi powers — Phase 6"
```

- [ ] **Step 6: Run verification protocol**

Extra: Load Phase Transition preset, toggle Higgs on/off, watch for field energy stability. Load Axion preset, check that axMod values display correctly in particle details.

---

## Task 7: Phase 7 — Renderer Optimizations

**Files:**
- Modify: `src/renderer.js:258-282, 321-361, 483-530, 503, 598-644`
- Modify: `src/heatmap.js:305-314`

### C11: Fuse spin ring passes

- [ ] **Step 1: Merge arc and arrowhead passes into single loop per sign**

Current pattern (lines ~321-361): Two separate passes per sign (4 total loops):
```js
// Pass 1: arcs
for (sign of [true, false]) {
    ctx.beginPath();
    for each particle: if matches sign, draw arc
    ctx.stroke();
}
// Pass 2: arrowheads
for (sign of [true, false]) {
    ctx.beginPath();
    for each particle: if matches sign, draw arrowhead triangle
    ctx.fill();
}
```

Merge into: One loop per sign that draws arc, then compute arrowhead position (same trig):
```js
for (const isPos of [true, false]) {
    ctx.strokeStyle = isPos ? posColor : negColor;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    const arrowParts = []; // collect arrowhead data during arc pass
    for (let i = 0; i < n; i++) {
        const p = particles[i];
        if (p.angVel === 0 || (p.angVel > 0) !== isPos) continue;
        // Draw arc (existing code)
        // ... compute ringRadius, startAngle, endAngle, dir ...
        ctx.moveTo(...); ctx.arc(...);
        // Save arrowhead data (reuse trig from arc)
        arrowParts.push(endX, endY, angle);
    }
    ctx.stroke();
    // Draw arrowheads from saved data
    ctx.beginPath();
    for (let j = 0; j < arrowParts.length; j += 3) {
        // triangle from saved endX, endY, angle
    }
    ctx.fill();
}
```

### C12: Fuse torque arc passes

- [ ] **Step 2: Same fusion pattern for `_drawTorqueArc` — merge arc + arrowhead**

The `_drawTorqueArc` method (lines ~598-644) currently does two full particle iterations (arc pass, then arrowhead pass) with identical `getValue(p)` computation. Fuse them: compute getValue once, draw arc, save arrowhead data, then draw all arrowheads.

### C19: Pass length instead of `subarray()`

- [ ] **Step 3: In `_batchArrowsDraw` (and callers), pass count instead of using `subarray`**

Current:
```js
const sub = lines.subarray(0, lc);
_batchArrowsDraw(ctx, sub, ...);
```

Change callers to pass `lc` directly:
```js
_batchArrowsDraw(ctx, lines, lc, ...);
```

And in `_batchArrowsDraw`, loop `for (let j = 0; j < count; j += 4)` instead of `lines.length`.

### C22: Remove redundant threshold check

- [ ] **Step 4: In `drawForceVectors` (line ~503), delete the `threshold` check**

```js
// DELETE: if (mag < threshold) continue;
// KEEP: if (mag < minLen) continue;  // minLen (0.5*invZoom) > threshold (0.1*invZoom)
```

Also remove the `threshold` variable declaration.

### C20: Set `imageSmoothingEnabled` once

- [ ] **Step 5: In `src/heatmap.js`, move imageSmoothing setup to constructor/init**

Remove from `draw()` (lines ~305-314):
```js
ctx.imageSmoothingEnabled = true;
ctx.imageSmoothingQuality = 'high';
```

Add once in the constructor or wherever the heatmap canvas context is first created.

### C21: Reduce trail alpha groups from 4 to 2

- [ ] **Step 6: In `drawTrails` (lines ~258-282), change `groupCount = 4` to `groupCount = 2`**

This halves `ctx.stroke()` calls (from 4N to 2N). Visual change: slightly coarser alpha gradient on trails.

- [ ] **Step 7: Commit Phase 7**

```bash
git add src/renderer.js src/heatmap.js
git commit -m "perf(renderer): fuse spin/torque passes, reduce trail groups, cleanup — Phase 7"
```

- [ ] **Step 8: Run verification protocol**

Visual checks: Spin rings display correctly in both directions. Torque arcs visible with correct arrowheads. Trails fade smoothly. Force arrows display at correct scale.

---

## Task 8: Phase 8 — Main Loop & UI Optimizations

**Files:**
- Modify: `main.js:731-746, 823-836`
- Modify: `src/ui.js:260-272`

### C2: Pre-allocate render options objects

- [ ] **Step 1: In `main.js`, create persistent objects in constructor**

In the `Simulation` constructor, add:
```js
this._enabledForces = {
    gravity: false, coulomb: false, magnetic: false, gravitomag: false,
    onePn: false, yukawa: false, higgs: false, axion: false
};
this._renderOpts = {
    blackHoleEnabled: false,
    enabledForces: this._enabledForces,
    higgsField: null, axionField: null
};
this._heatmapOpts = { mode: 'all' };
```

Then in `_render()` (lines ~731-746), mutate the existing objects instead of creating new ones:
```js
const ef = this._enabledForces;
ef.gravity = ph.gravityEnabled;
ef.coulomb = ph.coulombEnabled;
// ... etc
const opts = this._renderOpts;
opts.blackHoleEnabled = ph.blackHoleEnabled;
opts.higgsField = ph.higgsField;
opts.axionField = ph.axionField;
```

### C3: Pre-allocate GPU toggle proxy

- [ ] **Step 2: In `src/ui.js`, replace `Object.create` with a persistent proxy object**

At module level:
```js
let _gpuProxy = null;
```

Replace each `Object.create(sim.physics)` call with:
```js
if (!_gpuProxy) _gpuProxy = {};
Object.assign(_gpuProxy, sim.physics);
_gpuProxy.someExtraField = value; // the 2 GPU-specific overrides
gpuPhysics.setToggles(_gpuProxy);
```

Check which properties `setToggles()` actually reads from the proxy — it may only need the toggle booleans + a few slider values. If so, only copy those specific properties instead of `Object.assign` of the full physics object.

### C13: Gate sidebar plots behind panel/tab visibility

- [ ] **Step 3: In `main.js:823-836`, add visibility check**

Before the sidebar throttle block:
```js
if ((this._frameCount & SIDEBAR_THROTTLE_MASK) === 0) {
    // Add check:
    const particleTabVisible = this._panelOpen && this._activeTab === 'particle';
    if (particleTabVisible) {
        this.phasePlot.update(this.physics);
        this.effPotPlot.update(this.physics);
        this.phasePlot.draw();
        this.effPotPlot.draw();
    }
    this.stats.updateSelected(/* ... */);
    // ...
}
```

Need to identify how the panel open state and active tab are tracked. Look for `_panelOpen` or panel element class checks. The tab state is likely in the shared tab system. May need to cache `document.querySelector('.tab-btn.active')` result or read from a module-level variable set by the tab switch handler.

- [ ] **Step 4: Commit Phase 8**

```bash
git add main.js src/ui.js
git commit -m "perf(main): pre-alloc render opts, toggle proxy, gate sidebar plots — Phase 8"
```

- [ ] **Step 5: Run verification protocol**

Extra: Open Particle tab, select a particle — phase plot and V_eff should update. Close panel — they should stop updating. Reopen — they should resume.

---

## Task 9: Phase 9 — GPU Uniform Split

**Files:**
- Modify: `src/gpu/gpu-buffers.js:550-570`
- Modify: `src/gpu/gpu-constants.js`
- Modify: `src/gpu/shaders/shared-structs.wgsl`

### G6: Split SimUniforms into per-frame and per-substep

- [ ] **Step 1: Define `SubstepUniforms` struct in WGSL**

In `shared-structs.wgsl`, add after SimUniforms:
```wgsl
struct SubstepUniforms {
    dt: f32,
    simTime: f32,
    aliveCount: u32,
    particleCount: u32,
    frameCount: u32,
    _pad: u32, // align to 16 bytes (24B total → pad to 32B)
    _pad2: u32,
    _pad3: u32,
}
```

Remove `dt`, `simTime`, `aliveCount`, `particleCount`, `frameCount` from `SimUniforms` (they become per-substep).

- [ ] **Step 2: In `gpu-buffers.js`, create the substep uniform buffer**

Add to buffer allocation:
```js
this.substepUniformBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'substep-uniforms'
});
```

Add `writeSubstepUniforms(dt, simTime, aliveCount, particleCount, frameCount)`:
```js
writeSubstepUniforms(dt, simTime, aliveCount, particleCount, frameCount) {
    const buf = this._substepBuf32 || (this._substepBuf32 = new Float32Array(8));
    buf[0] = dt;
    buf[1] = simTime;
    new Uint32Array(buf.buffer)[2] = aliveCount;
    new Uint32Array(buf.buffer)[3] = particleCount;
    new Uint32Array(buf.buffer)[4] = frameCount;
    this.device.queue.writeBuffer(this.substepUniformBuffer, 0, buf);
}
```

Rename `writeUniforms` to `writeFrameUniforms` and remove the 5 fields that moved to substep.

### G14: Cache external field trig

- [ ] **Step 3: In `setToggles()`, precompute and cache trig values**

```js
setToggles(physics) {
    // ... existing toggle packing ...
    // Cache trig for writeFrameUniforms:
    this._cachedExtGravDirX = Math.cos(physics.extGravityAngle || 0);
    this._cachedExtGravDirY = Math.sin(physics.extGravityAngle || 0);
    this._cachedExtElecDirX = Math.cos(physics.extElectricAngle || 0);
    this._cachedExtElecDirY = Math.sin(physics.extElectricAngle || 0);
}
```

Then in `writeFrameUniforms`, use the cached values instead of recomputing.

### G17: Add precomputed cell dimensions to FieldUniforms

- [ ] **Step 4: In `gpu-constants.js` or the field uniform write code, add `cellW`, `cellH`, `invCellWSq`, `invCellHSq`**

These are `domainW/GRID`, `domainH/GRID`, `GRID²/domainW²`, `GRID²/domainH²`. Computed once per frame, written with field uniforms.

Update the WGSL `FieldUniforms` struct (in `field-common.wgsl` or wherever it's defined) to include these 4 fields.

- [ ] **Step 5: Update all bind group layouts that reference SimUniforms**

Every compute/render pipeline that reads `@group(0) @binding(0) var<uniform> uniforms: SimUniforms` now also needs `@group(0) @binding(N) var<uniform> substep: SubstepUniforms`. This is the biggest coordination task — identify all shaders that read `uniforms.dt`, `uniforms.simTime`, `uniforms.aliveCount`, `uniforms.particleCount`, or `uniforms.frameCount` and update them to read from `substep.*` instead.

Create a checklist of affected shaders by grepping for these field names across all `.wgsl` files.

- [ ] **Step 6: Update bind group creation in `gpu-physics.js` and `gpu-renderer.js`**

Every bind group that includes the main uniform buffer must also bind the new substep buffer. Update `_createPhase2BindGroup`, `_createBosonBindGroups`, `_createStatsBindGroup`, and all render bind groups.

- [ ] **Step 7: Bump `SHADER_VERSION` in `gpu-pipelines.js`**

Increment to invalidate cached shader modules.

- [ ] **Step 8: Commit Phase 9**

```bash
git add src/gpu/gpu-buffers.js src/gpu/gpu-constants.js src/gpu/gpu-pipelines.js src/gpu/gpu-physics.js src/gpu/gpu-renderer.js src/gpu/shaders/
git commit -m "perf(gpu): split uniforms into per-frame + per-substep, cache trig — Phase 9"
```

- [ ] **Step 9: Run verification protocol**

GPU-specific: Toggle GPU on, load all presets, verify stats match CPU mode. Check console for WebGPU validation errors.

---

## Task 10: Phase 10 — GPU Physics Optimizations

**Files:**
- Modify: `src/gpu/gpu-physics.js`

### G5: Use staging ring for substep uniforms

- [ ] **Step 1: Replace `queue.writeBuffer` for substep uniforms with `copyBufferToBuffer`**

Pre-allocate a ring of MAX_SUBSTEPS (32) staging buffers at init:
```js
this._substepStagingRing = [];
for (let i = 0; i < MAX_SUBSTEPS; i++) {
    const buf = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
        label: `substep-staging-${i}`
    });
    buf.unmap();
    this._substepStagingRing.push(buf);
}
```

Before the substep loop, pre-compute and upload all substep uniform data:
```js
// Pre-fill staging ring before encoding
for (let s = 0; s < numSubsteps; s++) {
    const staging = this._substepStagingRing[s];
    // Write dt, simTime + s*dtSub, aliveCount, particleCount, frameCount
    device.queue.writeBuffer(staging, 0, substepData[s]);
}
```

Then inside the encoder, use `copyBufferToBuffer` per substep:
```js
encoder.copyBufferToBuffer(
    this._substepStagingRing[s], 0,
    this._buffers.substepUniformBuffer, 0, 32
);
```

This moves the queue-time writes before the encoder starts, and the copy commands execute inline with compute passes.

### G12: Size boson dispatches to actual count

- [ ] **Step 2: Track photon/pion counts and dispatch accordingly**

Add CPU-side counters updated when bosons are emitted/absorbed:
```js
this._photonCount = 0;
this._pionCount = 0;
```

Update on emit (in radiation/pion emission dispatches) and absorb (readback from absorption pass). Use 1-frame-latency estimates:

```js
const photonWG = this._photonCount > 0 ? Math.ceil(this._photonCount / 64) : 0;
const pionWG = this._pionCount > 0 ? Math.ceil(this._pionCount / 64) : 0;
if (photonWG > 0) encoder.dispatchWorkgroups(photonWG);
if (pionWG > 0) encoder.dispatchWorkgroups(pionWG);
```

### G13: Pre-compute FFT butterfly params

- [ ] **Step 3: Build all FFT params at init time**

```js
// At field init:
const numStages = Math.log2(GRID); // 7 for GRID=128
const numParams = numStages * 2 * 2; // stages × 2 axes × 2 directions
this._fftParamsArray = new Float32Array(numParams * 8); // 8 floats per param set
// Pre-fill with all (stageLen, direction, axis, N, invN, isLast) combinations

this._fftParamsBuffer = device.createBuffer({
    size: this._fftParamsArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(this._fftParamsBuffer, 0, this._fftParamsArray);
```

Then during FFT dispatch, instead of `writeBuffer` per stage, pass a `stageIndex` uniform or use buffer offset.

### G15: Use `clearBuffer` for field initialization

- [ ] **Step 4: In `_initFieldToVacuum`, replace zero-fill `writeBuffer` with `clearBuffer`**

Current:
```js
device.queue.writeBuffer(this._fieldDotBuffer, 0, zeros);
device.queue.writeBuffer(this._gradXBuffer, 0, zeros);
// ... 9 similar calls with zeros ...
```

Replace:
```js
const enc = device.createCommandEncoder();
enc.clearBuffer(this._fieldDotBuffer);
enc.clearBuffer(this._gradXBuffer);
// ... for all 9 zero-filled buffers ...
device.queue.submit([enc.finish()]);
// Only writeBuffer for the non-zero buffer (Higgs vacuum = 1.0):
device.queue.writeBuffer(this._fieldBuffer, 0, vacuumData);
```

### G16: Batch `deserialize` uploads

- [ ] **Step 5: In `deserialize()`, build full packed arrays then do single writes**

Current: Per-particle loop with 5-7 `writeBuffer` calls each (up to 3584 total).

Change: Pre-allocate packed arrays sized for max particles:
```js
const stateArray = new Float32Array(GPU_MAX_PARTICLES * 9); // 36B / 4
const auxArray = new Float32Array(GPU_MAX_PARTICLES * 5);   // 20B / 4
// ... etc for each buffer type

for (let i = 0; i < particles.length; i++) {
    const p = particles[i];
    const off = i * 9;
    stateArray[off] = p.pos.x;
    stateArray[off+1] = p.pos.y;
    // ... pack all fields ...
}

device.queue.writeBuffer(this._particleStateBuffer, 0, stateArray, 0, particles.length * 9);
device.queue.writeBuffer(this._particleAuxBuffer, 0, auxArray, 0, particles.length * 5);
// ... 7 writes total instead of 3584
```

- [ ] **Step 6: Commit Phase 10**

```bash
git add src/gpu/gpu-physics.js
git commit -m "perf(gpu): staging ring, sized dispatches, FFT params, batch uploads — Phase 10"
```

- [ ] **Step 7: Run verification protocol**

GPU-specific: Save/load state (Ctrl+S / Ctrl+L) — verify particles restore correctly. Toggle radiation — check boson counts in stats.

---

## Task 11: Phase 11 — GPU Renderer Batching

**Files:**
- Modify: `src/gpu/gpu-renderer.js`
- Modify: `src/gpu/gpu-pipelines.js`

### G4: Batch render passes into 2-3 submits

- [ ] **Step 1: Group render passes into batched submits**

Current: 6+ separate encoder/submit cycles for trails, particles, overlays, bosons, rings, arrows.

Batch into 3 groups:
1. **Background** (1 submit): trails + field overlays + heatmap (load: clear, store: store)
2. **Particles** (1 submit): particles + bosons + spin rings + dashed rings (load: load, store: store)
3. **Overlays** (1 submit): all force arrows + torque arcs (load: load, store: store)

For each group, create one encoder with one render pass that draws all geometry in sequence.

### H5: Dynamic uniform buffer for arrows/torques

- [ ] **Step 2: Allocate a uniform buffer with slots for all 15 draw types**

```js
// 15 slots: 11 forces + velocity + 3 torques
const ARROW_SLOT_SIZE = 256; // align to 256B (WebGPU minUniformBufferOffsetAlignment)
this._arrowUniformBuf = device.createBuffer({
    size: 15 * ARROW_SLOT_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
```

Pre-fill all 15 slots' color/type data at init. Per frame, only write the scale/minMag values that change.

Create bind group with dynamic offsets:
```js
const arrowBindGroup = device.createBindGroup({
    layout: arrowPipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: { buffer: this._arrowUniformBuf, size: ARROW_SLOT_SIZE } },
        // ... particle buffers ...
    ]
});
```

Then draw all arrows in one render pass with different dynamic offsets:
```js
for (let slot = 0; slot < enabledArrows.length; slot++) {
    pass.setBindGroup(0, arrowBindGroup, [slot * ARROW_SLOT_SIZE]);
    pass.draw(9, aliveCount);
}
```

Same pattern for torques.

### M5: Cache shared prefix

- [ ] **Step 3: In `gpu-pipelines.js`, cache the concatenated shared prefix**

```js
let _cachedPrefix = null;
let _cachedPrefixKey = null;

export function getSharedPrefix(wgslConstants) {
    if (wgslConstants === _cachedPrefixKey) return _cachedPrefix;
    _cachedPrefixKey = wgslConstants;
    _cachedPrefix = wgslConstants + _structs + _topo + _rng;
    return _cachedPrefix;
}
```

### M6: Share shader module for light/dark pipeline pairs

- [ ] **Step 4: In boson (and other) pipeline creation, fetch shader once**

Current:
```js
Promise.all([
    createBosonRenderPipelines(device, wgslConstants, 'light'),
    createBosonRenderPipelines(device, wgslConstants, 'dark')
])
```

Change: Fetch shader + create module once, pass to both:
```js
const shaderCode = await fetchShader('boson-render.wgsl', wgslConstants, 'boson');
const module = device.createShaderModule({ code: shaderCode });
const [light, dark] = await Promise.all([
    createBosonRenderPipelinesFromModule(device, module, 'light'),
    createBosonRenderPipelinesFromModule(device, module, 'dark')
]);
```

Apply same pattern to all dual-theme pipeline pairs (particles, trails, rings, arrows, etc.).

- [ ] **Step 5: Commit Phase 11**

```bash
git add src/gpu/gpu-renderer.js src/gpu/gpu-pipelines.js
git commit -m "perf(gpu-render): batch submits, dynamic arrow uniforms, cache prefix — Phase 11"
```

- [ ] **Step 6: Run verification protocol**

Visual: All GPU render elements (particles, trails, arrows, torques, rings, field overlays, bosons) display correctly in both light and dark theme.

---

## Task 12: Phase 12 — GPU Shader Micro-optimizations

**Files:**
- Modify: `src/gpu/shaders/forces-tree.wgsl`
- Modify: `src/gpu/shaders/pair-force.wgsl`
- Modify: `src/gpu/shaders/field-fft.wgsl`
- Modify: `src/gpu/shaders/tree-build.wgsl`
- Modify: `src/gpu/shaders/collision.wgsl`
- Modify: `src/gpu/shaders/onePN.wgsl`
- Modify: `src/gpu/shaders/field-evolve.wgsl` (+ Higgs/Axion variants)
- Modify: `src/gpu/gpu-pipelines.js` (bump SHADER_VERSION)

### G8: Pack `accumulateForce` params

- [ ] **Step 1: In `forces-tree.wgsl`, group `accumulateForce` parameters into vec4s**

Current: 25+ scalar parameters.

Change to ~8 packed parameters:
```wgsl
fn accumulateForce(
    pPosVel: vec4<f32>,     // (posX, posY, velX, velY)
    pProps: vec4<f32>,      // (mass, charge, angVel, magMoment)
    pDerived: vec4<f32>,    // (angMomentum, invMass, bodyRSq, radiusSq)
    pMods: vec4<f32>,       // (axMod, yukMod, higgsMod, aberrFlag)
    sPosVel: vec4<f32>,     // source (posX, posY, velX, velY)
    sProps: vec4<f32>,      // source (mass, charge, angVel, magMoment)
    sDerived: vec4<f32>,    // source (angMomentum, 0, bodyRSq, 0)
    sMods: vec4<f32>,       // source (axMod, yukMod, higgsMod, 0)
) -> ... {
    let px = pPosVel.x; let py = pPosVel.y;
    // ... unpack at top, then existing logic ...
}
```

Update all call sites in the tree walk loop.

### G9: Hoist toggle reads

- [ ] **Step 2: In `forces-tree.wgsl`, read toggle bits once before the tree walk loop**

```wgsl
let t0 = uniforms.toggles0;
let t1 = uniforms.toggles1;
let gravOn = (t0 & GRAVITY_BIT) != 0u;
let coulOn = (t0 & COULOMB_BIT) != 0u;
// ... etc for all toggles used in accumulateForce ...
```

Pass these as additional params or read from shared memory.

### G10: FFT twiddle lookup buffer

- [ ] **Step 3: In `field-fft.wgsl`, replace per-thread trig with buffer lookup**

Add a binding for precomputed twiddle factors:
```wgsl
@group(0) @binding(N) var<storage, read> twiddles: array<vec2<f32>>;
```

Index: `twiddles[k % halfLen]` where k is the butterfly index within the stage. The buffer contains `(cos(2πk/N), sin(2πk/N))` for k=0..N/2-1.

Wire the buffer through `gpu-pipelines.js` and `gpu-physics.js`. Pre-fill at init:
```js
const twiddleBuf = new Float32Array(GRID * 2); // cos, sin pairs
for (let k = 0; k < GRID; k++) {
    const angle = 2 * Math.PI * k / GRID;
    twiddleBuf[2*k] = Math.cos(angle);
    twiddleBuf[2*k+1] = Math.sin(angle);
}
```

### G11: Non-atomic tree node initialization

- [ ] **Step 4: In `tree-build.wgsl:223-276`, replace atomicStore with plain store for new nodes**

The `subdivide()` function initializes newly allocated child nodes. Since only the allocating thread touches them before child pointers are published, these writes are safe without atomics:

```wgsl
// Replace atomicStore(&nodes[childIdx].field, value) with:
nodes[childIdx].field = value;
```

Keep the atomic stores only for the final `nw[idx] = c`, `ne[idx] = c+1`, etc. on the parent — these publish the children to other threads.

### G17: Read cell dims from FieldUniforms

- [ ] **Step 5: In `field-evolve.wgsl` (and Higgs/Axion variants), replace computed cell dims**

Replace:
```wgsl
let cellW = fieldUniforms.domainW / f32(GRID);
let cellH = fieldUniforms.domainH / f32(GRID);
let invCWsq = 1.0 / (cellW * cellW);
let invCHsq = 1.0 / (cellH * cellH);
```

With:
```wgsl
let cellW = fieldUniforms.cellW;
let cellH = fieldUniforms.cellH;
let invCWsq = fieldUniforms.invCellWSq;
let invCHsq = fieldUniforms.invCellHSq;
```

### G18: Replace `pow` with derived values in pair-force tidal

- [ ] **Step 6: In `pair-force.wgsl:431`, replace `pow(pMass, 5.0/3.0)`**

```wgsl
// Replace:
let pRi5 = pow(pMass, 5.0 / 3.0);
// With:
let pBodyR = sqrt(pBodyRadiusSq);
let pRi5 = pBodyRadiusSq * pBodyRadiusSq * pBodyR;
```

### G19: Guard child push in collision walk

- [ ] **Step 7: In `collision.wgsl:114-119`, add NONE check**

```wgsl
let nw = getNW(nodeIdx); if (nw != NONE) { stack[stackTop] = nw; stackTop++; }
let ne = getNE(nodeIdx); if (ne != NONE) { stack[stackTop] = ne; stackTop++; }
let sw = getSW(nodeIdx); if (sw != NONE) { stack[stackTop] = sw; stackTop++; }
let se = getSE(nodeIdx); if (se != NONE) { stack[stackTop] = se; stackTop++; }
```

### G20: Read `derived.bodyRSq` in tile load

- [ ] **Step 8: In `pair-force.wgsl:502`, use derived buffer instead of `pow`**

```wgsl
// Replace:
tile[localIdx].bodyRadSq = pow(sp.mass, 2.0 / 3.0);
// With:
tile[localIdx].bodyRadSq = derived[tileSrcIdx].bodyRSq;
```

### G21: Hoist `axYukMod[i]` in pairwise 1PN

- [ ] **Step 9: In `onePN.wgsl:207-213`, hoist reads before inner loop**

```wgsl
let pYukMod = axYukMod[i].y;
let pHiggsMod = axYukMod[i].z;
for (var j = 0u; j < aliveCount; j++) {
    // Use pYukMod, pHiggsMod instead of axYukMod[i].y, axYukMod[i].z
    // ...
}
```

- [ ] **Step 10: Bump SHADER_VERSION in `gpu-pipelines.js`**

Increment `SHADER_VERSION` by 1.

- [ ] **Step 11: Commit Phase 12**

```bash
git add src/gpu/shaders/ src/gpu/gpu-pipelines.js src/gpu/gpu-physics.js
git commit -m "perf(shaders): pack params, hoist toggles, twiddle LUT, micro-opts — Phase 12"
```

- [ ] **Step 12: Run verification protocol**

GPU-specific: Load all presets in GPU mode. Compare energy/momentum values to CPU mode — should match within floating-point tolerance. Test Barnes-Hut on/off, 1PN on/off, all force combinations.

---

## Task 13: Phase 13 — GPU Shader Major Refactors

**Files:**
- Modify: `src/gpu/shaders/compute-stats.wgsl`
- Modify: `src/gpu/shaders/bosons.wgsl`
- Modify: `src/gpu/shaders/bosons-tree-walk.wgsl`
- Modify: `src/gpu/shaders/heatmap.wgsl`
- Modify: `src/gpu/shaders/pair-force.wgsl`
- Modify: `src/gpu/shaders/signal-delay-common.wgsl`
- Modify: `src/gpu/gpu-physics.js` (new dispatch patterns)

### G1: Parallelize compute-stats

- [ ] **Step 1: Redesign `compute-stats.wgsl` with multi-dispatch parallel reduction**

This is the largest shader refactor. Split into separate entry points:

**Entry 1 — `statsKEMomentum`** (`@workgroup_size(64)`):
- One thread per particle. Each computes its own KE, momentum contribution.
- Write to shared memory, then workgroup reduction.
- Final workgroup result written to a partial sums buffer.
- Dispatch: `ceil(aliveCount / 64)` workgroups.

**Entry 2 — `statsReducePartials`** (`@workgroup_size(64)`):
- Reduces partial sums into final KE/momentum/COM.
- Single workgroup.

**Entry 3 — `statsPE`** (`@workgroup_size(TILE_SIZE)`):
- Tiled pairwise PE computation (same pattern as `pair-force.wgsl`).
- Shared memory tile of particle data. Each thread accumulates PE for its assigned particle over all tiles.
- Workgroup-level atomic accumulation into partial PE sums.
- Dispatch: `ceil(aliveCount / TILE_SIZE)` workgroups.

**Entry 4 — `statsField`** (`@workgroup_size(64)`):
- One thread per grid cell. Computes field energy density.
- Workgroup reduction, then partial sums.
- Dispatch: `ceil(GRID*GRID / 64)` workgroups.

**Entry 5-6 — Keep single-threaded** for PFI (O(N) with PQS, small N) and selected particle copy.

Update `gpu-physics.js` to dispatch these as separate passes with intermediate buffers.

- [ ] **Step 2: Create partial sums buffer and update bind groups**

```js
this._statsPartialBuffer = device.createBuffer({
    size: 1024, // enough for 16 workgroups × 64 bytes
    usage: GPUBufferUsage.STORAGE,
});
```

### G2: Parallelize boson absorption

- [ ] **Step 3: Implement two-pass boson absorption**

**Pass 1 — Detection** (`@workgroup_size(64)`):
One thread per boson. Each checks overlap with all nearby particles (or uses BH tree query). If a candidate is found, writes `(bosonIdx, particleIdx, distSq)` to an append buffer via `atomicAdd` on a counter.

```wgsl
@group(0) @binding(N) var<storage, read_write> absorptionCandidates: array<vec4<u32>>;
@group(0) @binding(N+1) var<storage, read_write> candidateCount: atomic<u32>;

// Per thread:
if (distSq < softSq && boson.emitterId != particle.particleId && boson.age >= MIN_AGE) {
    let idx = atomicAdd(&candidateCount, 1u);
    absorptionCandidates[idx] = vec4(bosonIdx, particleIdx, bitcast<u32>(distSq), 0u);
}
```

**Pass 2 — Resolution** (`@workgroup_size(1)`):
Single thread iterates candidates, resolves conflicts (closest boson wins per particle), applies momentum/charge transfer.

Update `gpu-physics.js` with the new dispatch pattern and intermediate buffer.

### G3: Tree-accelerate GPU heatmap

- [ ] **Step 4: In `heatmap.wgsl`, replace all-pairs with BH tree walk**

Add `shared-tree-nodes.wgsl` to the heatmap prepend chain. Replace the particle loop:

```wgsl
// Replace:
for (var i = 0u; i < particleCount; i++) { ... }

// With BH tree walk:
var stack: array<u32, 48>;
stack[0] = 0u; // root
var stackTop = 1u;
while (stackTop > 0u) {
    stackTop--;
    let nodeIdx = stack[stackTop];
    // ... standard BH walk: check theta criterion, accumulate potential ...
}
```

Add dead particle count uniform to skip the dead loop when no dead particles exist:
```wgsl
if (substep.deadCount > 0u) {
    // ... existing dead particle loop (keep pairwise) ...
}
```

### G7: Mitigate signal delay warp divergence

- [ ] **Step 5: In `pair-force.wgsl`, add signal-delay-aware tile strategy**

When `signalDelayed` is true, the tile preload into shared memory is wasted (delayed positions come from the history buffer). Add a conditional path:

```wgsl
if (SIGNAL_DELAYED) {
    // Skip tile preload — read directly from global memory
    // Prefetch history metadata into shared memory per tile
    if (localIdx < TILE_SIZE) {
        // Load historyStart, sampleCount, creationTime for tile's particles
        sharedHistMeta[localIdx] = historyMeta[tileStart + localIdx];
    }
    workgroupBarrier();
    // ... force loop using getDelayedStateGPU with shared metadata ...
} else {
    // ... existing tiled path with shared memory particle data ...
}
```

This reduces global memory pressure in the NR solver by keeping frequently-accessed metadata in fast shared memory.

- [ ] **Step 6: Bump SHADER_VERSION**

- [ ] **Step 7: Commit Phase 13**

```bash
git add src/gpu/shaders/ src/gpu/gpu-physics.js
git commit -m "perf(shaders): parallelize stats/absorption, tree heatmap, SD divergence — Phase 13"
```

- [ ] **Step 8: Final verification**

Full verification protocol plus:
- Compare GPU stats output to CPU mode for all presets — must match within tolerance
- Profile frame time before/after Phase 13 with 128+ particles in GPU mode
- Test with signal delay on/off, BH tree on/off
- Test heatmap in all 4 modes (All/Grav/Elec/Yukawa)
- Test save/load round-trip in GPU mode

---

## Summary

| Phase | Task | Files Modified | Fixes | Risk |
|-------|------|---------------|-------|------|
| 1 | Dead code cleanup | 14 files | D1-D12 | Low |
| 2 | Quadtree | 1 file | C1, C10, H10, M5 | Low |
| 3 | Forces | 2 files | C4, C5 | Low-Med |
| 4 | Integrator | 2 files | C6, H2, H8 | Medium |
| 5 | FFT + ScalarField | 4 files | C7, C14, C15 | Medium |
| 6 | Higgs + Axion | 3 files | C8, C16, C18, A2 | Med-High |
| 7 | Renderer | 2 files | C11, C12, C19-C22 | Low |
| 8 | Main + UI | 2 files | C2, C3, C13 | Low |
| 9 | GPU Buffers | 5+ files | G6, G14, G17 | Medium |
| 10 | GPU Physics | 1 file | G5, G12-G13, G15-G16 | Medium |
| 11 | GPU Renderer | 2 files | G4, H5, M5, M6 | Medium |
| 12 | GPU Shaders Micro | 8+ shaders | G8-G11, G17-G21 | Low-Med |
| 13 | GPU Shaders Major | 5+ shaders | G1-G3, G7 | High |
