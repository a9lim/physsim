# Physics Accuracy & Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix physics bugs (LARMOR_K, Schott sign, spin KE formula, spin-orbit gradient, photon bookkeeping), refactor energy computation into dedicated module, rename `spin` → `angw`, add gravitational Darwin terms, fix BH PE, fix relativistic bounce friction, update info tips and UI dependencies.

**Architecture:** Rename first (pervasive but safe), then fix bugs (surgical edits), then extract energy module (structural refactor), then fix PE/bounce/UI (depends on earlier work). No test framework — verify by running `python -m http.server` from the `a9lim.github.io/` parent directory and checking energy conservation displays.

**Tech Stack:** Vanilla JS ES6 modules, Canvas 2D, no build step or dependencies.

---

### Task 1: Rename `spin` → `angw` in particle.js and relativity.js

**Files:**
- Modify: `src/particle.js`
- Modify: `src/relativity.js`

**Step 1: Update particle.js**

In `src/particle.js`, rename all `this.spin` to `this.angw`:

```js
// Line 23: change
this.spin = 0;      // proper angular velocity (unbounded state variable)
// to
this.angw = 0;      // angular celerity (proper angular velocity, unbounded)
```

**Step 2: Update relativity.js**

Rename `spinToAngVel` → `angwToAngVel`, update parameter name, and add `angVelToAngw`:

```js
// Replace entire file content:
// ─── Relativistic Helpers (c = 1) ───

import { MAX_SPEED_RATIO } from './config.js';

/**
 * Derive angular velocity from angular celerity via rotational Lorentz factor.
 * ω = W / √(1 + W²r²), naturally caps surface velocity |ωr| < c.
 */
export function angwToAngVel(angw, radius) {
    return angw / Math.sqrt(1 + angw * angw * radius * radius);
}

/**
 * Derive angular celerity from angular velocity (inverse of angwToAngVel).
 * W = ω / √(1 - ω²r²), analogous to w = v / √(1 - v²).
 */
export function angVelToAngw(angVel, radius) {
    const sr = angVel * radius;
    const srSq = sr * sr;
    if (srSq >= 1) {
        const clampedSr = MAX_SPEED_RATIO;
        return Math.sign(angVel) * clampedSr / (radius * Math.sqrt(1 - clampedSr * clampedSr));
    }
    return angVel / Math.sqrt(1 - srSq);
}

/**
 * Set particle proper velocity from velocity components.
 * Clamps |v| < MAX_SPEED_RATIO, then sets p.vel and p.w = γv.
 */
export function setVelocity(p, vx, vy) {
    const speedSq = vx * vx + vy * vy;
    if (speedSq >= 1) {
        const s = MAX_SPEED_RATIO / Math.sqrt(speedSq);
        vx *= s;
        vy *= s;
    }
    const gamma = 1 / Math.sqrt(1 - vx * vx - vy * vy);
    p.vel.set(vx, vy);
    p.w.set(vx * gamma, vy * gamma);
}
```

**Step 3: Commit**

```bash
git add src/particle.js src/relativity.js
git commit -m "refactor: rename spin to angw in particle.js and relativity.js

Angular celerity (angw) mirrors the linear celerity (w) naming pattern.
Add angVelToAngw inverse helper for bounce friction fix."
```

---

### Task 2: Rename `spin` → `angw` in physics.js

**Files:**
- Modify: `src/physics.js`

**Step 1: Update imports**

Change the import from relativity.js:
```js
import { setVelocity, angwToAngVel } from './relativity.js';
```

Also add `angVelToAngw` import (needed later for bounce fix):
```js
import { setVelocity, angwToAngVel, angVelToAngw } from './relativity.js';
```

**Step 2: Global rename in physics.js**

Replace all occurrences:
- `spinToAngVel` → `angwToAngVel` (3 occurrences: lines ~62, 279, 481)
- `p.spin` → `p.angw` (all occurrences in spin-orbit, frame-drag, merge, bounce sections)
- `p1.spin` → `p1.angw`, `p2.spin` → `p2.angw` (in merge and bounce)
- `sSpin` parameter in `_pairForce` → `sAngVel` (parameter name and frame-drag usage)

In the `_pairForce` signature (line ~654):
```js
_pairForce(p, sx, sy, svx, svy, sMass, sCharge, sAngVel, sMagMoment, sAngMomentum, out) {
```

In the frame-dragging torque (line ~736):
```js
const torque = FRAME_DRAG_K * sMass * (sAngVel - p.angVel) * invR * invRSq;
```

In `_computeAllForces` pairwise call (~line 413-416):
```js
this._pairForce(p, sx, sy, svx, svy,
    o.mass, o.charge, sAngVel,
    MAG_MOMENT_K * o.charge * sAngVel * oRSq,
    INERTIA_K * o.mass * sAngVel * oRSq, p.force);
```

In `resolveMerge` (~lines 470-480):
```js
const Lspin = INERTIA_K * p1.mass * p1.radius * p1.radius * p1.angw
    + INERTIA_K * p2.mass * p2.radius * p2.radius * p2.angw;
// ...
p1.angw = (Lorb + Lspin) / newI;
p1.angVel = this.relativityEnabled ? angwToAngVel(p1.angw, p1.radius) : p1.angw;
```

In `resolveBounce` — all `p1.spin`, `p2.spin` → `p1.angw`, `p2.angw` (both relativistic and classical paths).

**Step 3: Commit**

```bash
git add src/physics.js
git commit -m "refactor: rename spin to angw in physics.js, sSpin to sAngVel"
```

---

### Task 3: Rename `spin` → `angw` in main.js, input.js, presets.js

**Files:**
- Modify: `main.js`
- Modify: `src/input.js`
- Modify: `src/presets.js`

**Step 1: main.js**

Update import:
```js
import { setVelocity, angwToAngVel } from './src/relativity.js';
```

In `computeEnergy()` (~line 135):
```js
const L = INERTIA_K * p.mass * rSq * p.angw;
```

In spin angular momentum (~line 167):
```js
spinAngMom += INERTIA_K * p.mass * p.radius * p.radius * p.angw;
```

In `addParticle()` (~lines 261-269): rename `sv` logic and `p.spin` to `p.angw`:
```js
p.angw = absSV > 0 ? Math.sign(sv) * absSV / (p.radius * Math.sqrt(1 - absSV * absSV)) : 0;
// ...
p.angVel = this.physics.relativityEnabled ? angwToAngVel(p.angw, p.radius) : p.angw;
```

In `updateSelectedParticle()` — no `spin` references (uses `angVel` for display). No changes needed.

In tidal fragment loop (~line 335): `spin: p.angw` in the addParticle options.

**Step 2: input.js**

The `spinInput` DOM ref name stays (it's an HTML id). No `p.spin` references in input.js — the spin value is passed as an option to `addParticle()` via `{ mass, charge, spin }`. This key name in the options object should stay as `spin` since it represents the user-facing "spin" concept (surface velocity fraction), not the internal `angw` state variable. The conversion happens inside `addParticle()`.

No changes needed in input.js.

**Step 3: presets.js**

The `spin` key in preset options `{ mass, charge, spin }` represents surface velocity fraction passed to `addParticle()`. This is the user-facing API — keep as `spin`. No changes needed.

**Step 4: Verify**

Start a local server from the `a9lim.github.io` parent directory:
```bash
cd /path/to/a9lim.github.io && python -m http.server
```
Navigate to `http://localhost:8000/physsim/`. Load the "Binary Stars" preset. Verify:
- Simulation runs without console errors
- Spin rings render on particles
- Energy stats display in sidebar

**Step 5: Commit**

```bash
git add main.js src/input.js src/presets.js
git commit -m "refactor: rename spin to angw in main.js"
```

---

### Task 4: Fix LARMOR_K constant

**Files:**
- Modify: `src/config.js`

**Step 1: Fix the constant**

Replace line 36:
```js
export const LARMOR_K = 1 / (6 * Math.PI); // q²a²/(6π) in natural units (c=G=1)
```
With:
```js
export const LARMOR_K = 1 / 3; // τ = 2·LARMOR_K·q²/m = 2q²/(3m), P = 2q²a²/3 (c=G=1, ε₀=1/(4π))
```

**Step 2: Commit**

```bash
git add src/config.js
git commit -m "fix: correct LARMOR_K from 1/(6π) to 1/3

Unit conversion error: SI formula P=q²a²/(6πε₀c³) with ε₀=1/(4π), c=1
gives P=2q²a²/3, requiring LARMOR_K=1/3. Radiation was ~6.3x too weak."
```

---

### Task 5: Fix Schott damping sign

**Files:**
- Modify: `src/physics.js`

**Step 1: Fix the sign**

In the Abraham-Lorentz radiation section (~line 207), change:
```js
let fRadX = tau * (jerkX + schottX);
let fRadY = tau * (jerkY + schottY);
```
To:
```js
let fRadX = tau * (jerkX - schottX);
let fRadY = tau * (jerkY - schottY);
```

**Step 2: Commit**

```bash
git add src/physics.js
git commit -m "fix: correct Schott damping sign in Landau-Lifshitz radiation

The Schott term τ·F²·v/m must be subtracted (decelerating) not added
(accelerating). The + sign was pushing particles in the direction of
motion instead of opposing it."
```

---

### Task 6: Fix spin-orbit gradient

**Files:**
- Modify: `src/physics.js`

**Step 1: Fix EM gradient**

In `_pairForce()`, replace the EM gradient block (~lines 708-712):
```js
// B_z gradient for spin-orbit coupling
// B_z ~ q_s * cross/(r³), so dB_z/dr ~ -3 * B_z / r
const Bz_contribution = sCharge * crossSV * invR * invRSq;
const dBzdr = -3 * Bz_contribution * invR;
p.dBzdx += dBzdr * rx * invR;  // rx/r = r̂_x (note: rx = sx - p.pos.x)
p.dBzdy += dBzdr * ry * invR;
```

With:
```js
// ∇Bz w.r.t. observer position (radial + angular terms)
// ∂Bz/∂px = +3·Bz·rx/r² + q_s·vsy/r³
// ∂Bz/∂py = +3·Bz·ry/r² - q_s·vsx/r³
const Bz_contribution = sCharge * crossSV * invR * invRSq;
p.dBzdx += 3 * Bz_contribution * rx * invRSq + sCharge * svy * invR * invRSq;
p.dBzdy += 3 * Bz_contribution * ry * invRSq - sCharge * svx * invR * invRSq;
```

**Step 2: Fix GM gradient**

Replace the GM gradient block (~lines 730-733):
```js
// Bgz gradient for GM spin-orbit coupling (same pattern as EM dBz)
const Bgz_contribution = sMass * crossSV * invR * invRSq;
const dBgzdr = -3 * Bgz_contribution * invR;
p.dBgzdx += dBgzdr * rx * invR;
p.dBgzdy += dBgzdr * ry * invR;
```

With:
```js
// ∇Bgz w.r.t. observer position (radial + angular terms)
const Bgz_contribution = sMass * crossSV * invR * invRSq;
p.dBgzdx += 3 * Bgz_contribution * rx * invRSq + sMass * svy * invR * invRSq;
p.dBgzdy += 3 * Bgz_contribution * ry * invRSq - sMass * svx * invR * invRSq;
```

**Step 3: Commit**

```bash
git add src/physics.js
git commit -m "fix: correct spin-orbit B-field gradient sign and add angular terms

The gradient of B_z = q_s(v_s×r)_z/r³ w.r.t. observer position has:
1. Radial term: +3·Bz·r̂/r (was -3, sign was wrong)
2. Angular term: from differentiating the cross product numerator
Both EM and GM gradients corrected."
```

---

### Task 7: Fix photon absorption bookkeeping

**Files:**
- Modify: `main.js`

**Step 1: Update photon absorption**

In the photon absorption block inside `loop()` (~lines 296-308), after `ph.alive = false;` and before `break;`, add:
```js
// Subtract absorbed energy from radiated totals
this.totalRadiated = Math.max(0, this.totalRadiated - ph.energy);
this.totalRadiatedPx -= ph.vel.x * ph.energy;
this.totalRadiatedPy -= ph.vel.y * ph.energy;
```

**Step 2: Commit**

```bash
git add main.js
git commit -m "fix: subtract absorbed photon energy from totalRadiated

When a photon is absorbed by a particle, its energy transfers to KE.
Previously totalRadiated was not decremented, causing total energy
(KE+PE+Field+Radiated) to drift upward on each absorption."
```

---

### Task 8: Verify all bug fixes

**Step 1: Run simulation**

Start server, load physsim. Load "Magnetic" preset (has charged spinning particles — exercises radiation, magnetic forces, spin-orbit).

**Verify:**
- No console errors
- Radiation photons emit and absorb without energy drift
- Energy drift stays small (< ±1% over 10 seconds)
- Spin rings display correctly
- Force component vectors look reasonable

**Step 2: Test radiation specifically**

Enable Relativity + Radiation. Place two oppositely-charged particles near each other. Verify:
- Photons are emitted (radiation is now 6.3x stronger than before)
- The orbiting particle's orbit decays (Schott damping now decelerates)
- Energy drift tracks properly (radiated energy + KE + PE ≈ constant)

If radiation is now too strong (causes instability), tune LL_FORCE_CLAMP in config.js. Try values 0.3 or 0.2 if 0.5 causes issues.

---

### Task 9: Create `src/energy.js` — energy module

**Files:**
- Create: `src/energy.js`

**Step 1: Write the module**

```js
// ─── Energy & Momentum Computation ───
import { INERTIA_K, SOFTENING_SQ } from './config.js';

/**
 * Compute all energy, momentum, and angular momentum quantities.
 * Returns object with all values needed for display.
 */
export function computeEnergies(particles, physics, sim) {
    let linearKE = 0;
    let spinKE = 0;
    let totalMass = 0;
    let comX = 0, comY = 0;
    let px = 0, py = 0;
    const relOn = physics.relativityEnabled;

    // ─── Pass 1: KE, momentum, COM ───
    for (const p of particles) {
        const rSq = p.radius * p.radius;
        if (relOn) {
            // Relativistic linear KE: (γ - 1)mc², γ = √(1 + w²)
            const gamma = Math.sqrt(1 + p.w.magSq());
            linearKE += (gamma - 1) * p.mass;
            // Relativistic spin KE: m_rot·(γ_rot - 1) where m_rot = I/r² = INERTIA_K·m
            const srSq = p.angw * p.angw * rSq;
            spinKE += INERTIA_K * p.mass * (Math.sqrt(1 + srSq) - 1);
        } else {
            const speedSq = p.vel.x * p.vel.x + p.vel.y * p.vel.y;
            linearKE += 0.5 * p.mass * speedSq;
            // Classical spin KE: ½Iω²
            spinKE += 0.5 * INERTIA_K * p.mass * rSq * p.angVel * p.angVel;
        }

        // Momentum: p = m·w (relativistic or classical, since w = v when rel off)
        px += p.mass * p.w.x;
        py += p.mass * p.w.y;

        totalMass += p.mass;
        comX += p.mass * p.pos.x;
        comY += p.mass * p.pos.y;
    }

    // ─── Pass 2: Angular momentum about COM ───
    let orbitalAngMom = 0;
    let spinAngMom = 0;
    if (totalMass > 0) {
        comX /= totalMass;
        comY /= totalMass;

        for (const p of particles) {
            const dx = p.pos.x - comX;
            const dy = p.pos.y - comY;
            orbitalAngMom += dx * (p.mass * p.w.y) - dy * (p.mass * p.w.x);
            spinAngMom += INERTIA_K * p.mass * p.radius * p.radius * p.angw;
        }
    }

    // ─── Pass 3: Darwin field energy & momentum (O(v²/c²) correction) ───
    // EM: U = -(1/2) Σ_{i<j} (qi·qj/r) [(vi·vj) + (vi·r̂)(vj·r̂)]
    // Grav: U = +(1/2) Σ_{i<j} (mi·mj/r) [(vi·vj) + (vi·r̂)(vj·r̂)]  (opposite sign)
    let fieldEnergy = 0;
    let fieldPx = 0, fieldPy = 0;
    const n = particles.length;
    const hasCoulomb = physics.coulombEnabled;
    const hasGM = physics.gravitomagEnabled;

    if (hasCoulomb || hasGM) {
        for (let i = 0; i < n; i++) {
            const pi = particles[i];
            for (let j = i + 1; j < n; j++) {
                const pj = particles[j];
                const dx = pj.pos.x - pi.pos.x;
                const dy = pj.pos.y - pi.pos.y;
                const rSq = dx * dx + dy * dy + SOFTENING_SQ;
                const invR = 1 / Math.sqrt(rSq);
                const rx = dx * invR, ry = dy * invR;
                const viDotVj = pi.vel.x * pj.vel.x + pi.vel.y * pj.vel.y;
                const viDotR = pi.vel.x * rx + pi.vel.y * ry;
                const vjDotR = pj.vel.x * rx + pj.vel.y * ry;
                const velTerm = viDotVj + viDotR * vjDotR;

                // Sum velocity for field momentum
                const svx = pi.vel.x + pj.vel.x, svy = pi.vel.y + pj.vel.y;
                const svDotR = svx * rx + svy * ry;

                if (hasCoulomb) {
                    const qqInvR = pi.charge * pj.charge * invR;
                    fieldEnergy -= 0.5 * qqInvR * velTerm;
                    const coeff = qqInvR * 0.5;
                    fieldPx += coeff * (svx + rx * svDotR);
                    fieldPy += coeff * (svy + ry * svDotR);
                }

                if (hasGM) {
                    const mmInvR = pi.mass * pj.mass * invR;
                    fieldEnergy += 0.5 * mmInvR * velTerm;
                    const coeff = mmInvR * 0.5;
                    fieldPx += coeff * (svx + rx * svDotR);
                    fieldPy += coeff * (svy + ry * svDotR);
                }
            }
        }
    }

    return {
        linearKE, spinKE,
        pe: physics.potentialEnergy,
        fieldEnergy, fieldPx, fieldPy,
        px, py,
        orbitalAngMom, spinAngMom,
        comX, comY,
    };
}
```

**Step 2: Commit**

```bash
git add src/energy.js
git commit -m "feat: create energy.js module with corrected spin KE and gravitational Darwin terms

Extracts energy/momentum computation from main.js. Includes:
- Corrected relativistic spin KE: INERTIA_K·m·(√(1+S²r²)-1)
- NEW: Gravitational Darwin field energy and momentum
- EM Darwin field energy and momentum (moved from main.js)"
```

---

### Task 10: Wire energy.js into main.js

**Files:**
- Modify: `main.js`

**Step 1: Add import**

At the top of `main.js`, add:
```js
import { computeEnergies } from './src/energy.js';
```

**Step 2: Replace computeEnergy() body**

Replace the body of `Simulation.computeEnergy()` with a thin wrapper that calls `computeEnergies()` and formats results for DOM. Keep all the DOM update logic, remove the physics calculations:

```js
computeEnergy() {
    const e = computeEnergies(this.particles, this.physics, this);

    const angMom = e.orbitalAngMom + e.spinAngMom;

    // Total momentum = particle + field + radiated (vector sum)
    const totalPx = e.px + e.fieldPx + this.totalRadiatedPx;
    const totalPy = e.py + e.fieldPy + this.totalRadiatedPy;
    const pMag = Math.sqrt(totalPx * totalPx + totalPy * totalPy);

    const total = e.linearKE + e.spinKE + e.pe + e.fieldEnergy + this.totalRadiated;

    if (this.initialEnergy === null && this.particles.length > 0) {
        this.initialEnergy = total;
        this.initialMomentum = pMag;
        this.initialAngMom = angMom;
    }

    const eDrift = this.initialEnergy !== null && this.initialEnergy !== 0
        ? ((total - this.initialEnergy) / Math.abs(this.initialEnergy) * 100)
        : 0;
    const pDrift = this.initialMomentum !== null && this.initialMomentum !== 0
        ? ((pMag - this.initialMomentum) / Math.abs(this.initialMomentum) * 100)
        : 0;
    const aDrift = this.initialAngMom !== null && this.initialAngMom !== 0
        ? ((angMom - this.initialAngMom) / Math.abs(this.initialAngMom) * 100)
        : 0;

    const fmt = (v) => Math.abs(v) < 0.01 ? '0' : Math.abs(v) > 999 ? v.toExponential(1) : v.toFixed(1);
    const fmtDrift = (v) => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';

    this.dom.linearKE.textContent = fmt(e.linearKE);
    this.dom.spinKE.textContent = fmt(e.spinKE);
    this.dom.potentialE.textContent = fmt(e.pe);
    this.dom.totalE.textContent = fmt(total);
    this.dom.energyDrift.textContent = fmtDrift(eDrift);
    this.dom.fieldE.textContent = fmt(e.fieldEnergy);
    this.dom.radiatedE.textContent = fmt(this.totalRadiated);
    this.sankey.update(e.linearKE, e.spinKE, e.pe, e.fieldEnergy, this.totalRadiated);
    this.dom.momentum.textContent = fmt(pMag);
    this.dom.fieldMom.textContent = fmt(Math.sqrt(e.fieldPx * e.fieldPx + e.fieldPy * e.fieldPy));
    this.dom.radiatedMom.textContent = fmt(Math.sqrt(this.totalRadiatedPx * this.totalRadiatedPx + this.totalRadiatedPy * this.totalRadiatedPy));
    this.dom.momentumDrift.textContent = fmtDrift(pDrift);
    this.dom.angularMomentum.textContent = fmt(angMom);
    this.dom.orbitalAngMom.textContent = fmt(e.orbitalAngMom);
    this.dom.spinAngMom.textContent = fmt(e.spinAngMom);
    this.dom.angMomDrift.textContent = fmtDrift(aDrift);
}
```

Remove the now-unused `INERTIA_K` and `SOFTENING_SQ` imports from main.js if they're no longer needed there (check other usages first — `INERTIA_K` is used in `addParticle` and tidal fragment, `SOFTENING_SQ` is not used in main.js).

**Step 3: Verify**

Run simulation, load Solar System preset. Check Stats tab — energy values should display correctly. Compare with a fresh page load to make sure values are reasonable.

**Step 4: Commit**

```bash
git add main.js
git commit -m "refactor: wire energy.js into main.js, replace inline computation"
```

---

### Task 11: Extract PE from _pairForce, add computePE method

**Files:**
- Modify: `src/physics.js`

**Step 1: Remove PE accumulation from _pairForce**

In `_pairForce()`, remove all `this.potentialEnergy +=/-=` lines:
- Remove: `this.potentialEnergy -= p.mass * sMass * invR * 0.5;` (gravity PE)
- Remove: `this.potentialEnergy += p.charge * sCharge * invR * 0.5;` (Coulomb PE)
- Remove: `this.potentialEnergy += (pMagMoment * sMagMoment) * invR * invRSq * 0.5;` (magnetic dipole PE)
- Remove: `this.potentialEnergy -= (pAngMomentum * sAngMomentum) * invR * invRSq * 0.5;` (GM dipole PE)

**Step 2: Add computePE method**

Add a new method to the Physics class that computes PE using tree traversal (BH-consistent):

```js
/**
 * Compute potential energy using same tree/pairwise method as forces.
 * When BH is on: traverses tree per-particle with BH_THETA, divides by 2.
 * When BH is off: exact pairwise i<j (no double-counting).
 */
computePE(particles, qt) {
    let pe = 0;

    if (this.barnesHutEnabled && qt) {
        for (const p of particles) {
            pe += this._treePE(p, qt, BH_THETA);
        }
        pe *= 0.5; // Each pair counted from both sides
    } else {
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            for (let j = i + 1; j < particles.length; j++) {
                const o = particles[j];
                pe += this._pairPE(p, o.pos.x, o.pos.y,
                    o.mass, o.charge, o.angVel,
                    MAG_MOMENT_K * o.charge * o.angVel * o.radius * o.radius,
                    INERTIA_K * o.mass * o.angVel * o.radius * o.radius);
            }
        }
    }

    this.potentialEnergy = pe;
}

_treePE(particle, node, theta) {
    if (node.totalMass === 0) return 0;

    const dx = node.centerOfMass.x - particle.pos.x;
    const dy = node.centerOfMass.y - particle.pos.y;
    const dSq = dx * dx + dy * dy;
    const d = Math.sqrt(dSq);
    const size = node.boundary.w * 2;

    if ((!node.divided && node.points.length > 0) || (node.divided && (size / d < theta))) {
        if (!node.divided) {
            let pe = 0;
            for (const other of node.points) {
                if (other === particle) continue;
                const oRSq = other.radius * other.radius;
                pe += this._pairPE(particle, other.pos.x, other.pos.y,
                    other.mass, other.charge, other.angVel,
                    MAG_MOMENT_K * other.charge * other.angVel * oRSq,
                    INERTIA_K * other.mass * other.angVel * oRSq);
            }
            return pe;
        } else {
            return this._pairPE(particle, node.centerOfMass.x, node.centerOfMass.y,
                node.totalMass, node.totalCharge, 0,
                node.totalMagneticMoment, node.totalAngularMomentum);
        }
    } else if (node.divided) {
        return this._treePE(particle, node.northwest, theta)
            + this._treePE(particle, node.northeast, theta)
            + this._treePE(particle, node.southwest, theta)
            + this._treePE(particle, node.southeast, theta);
    }
    return 0;
}

_pairPE(p, sx, sy, sMass, sCharge, sAngVel, sMagMoment, sAngMomentum) {
    const rx = sx - p.pos.x;
    const ry = sy - p.pos.y;
    const rSq = rx * rx + ry * ry + SOFTENING_SQ;
    const r = Math.sqrt(rSq);
    const invR = 1 / r;
    const invRSq = 1 / rSq;
    const pRSq = p.radius * p.radius;
    const pMagMoment = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
    const pAngMomentum = INERTIA_K * p.mass * p.angVel * pRSq;

    let pe = 0;
    if (this.gravityEnabled)  pe -= p.mass * sMass * invR;
    if (this.coulombEnabled)  pe += p.charge * sCharge * invR;
    if (this.magneticEnabled) pe += (pMagMoment * sMagMoment) * invR * invRSq;
    if (this.gravitomagEnabled) pe -= (pAngMomentum * sAngMomentum) * invR * invRSq;
    return pe;
}
```

**Step 3: Wire computePE into update loop**

In `update()`, after force calculation in each substep (Step 7), replace:
```js
this.potentialEnergy = 0;
```
With:
```js
// PE computed separately for BH consistency
```

And after the substep loop ends (after all substeps), add:
```js
// Compute PE using same approximation as forces
const finalQt = new QuadTree(this.boundary, QUADTREE_CAPACITY);
for (const p of particles) finalQt.insert(p);
finalQt.calculateMassDistribution();
this.computePE(particles, finalQt);
```

Actually, we already have a tree rebuilt in the last substep. We can reuse it. The last tree is built inside the substep loop at Step 5, but it's a local variable. Instead, let's compute PE once after the loop using a fresh tree (or reuse the one from last substep by saving it). The simplest approach: compute PE once after the substep loop:

In `update()`, after the substep `for` loop and before the boundary handling:
```js
// Compute PE (once per frame, after final positions)
this.computePE(particles, qt);
```

Wait, `qt` is scoped inside the substep loop. Easiest fix: save the last tree:

At the start of `update()`, declare `let lastQt = null;`. Inside the substep loop, after rebuilding the tree, set `lastQt = qt;`. After the loop, call `this.computePE(particles, lastQt);`.

**Step 4: Remove `this.potentialEnergy = 0` from the substep loop**

The line `this.potentialEnergy = 0;` before `_resetForces` in each substep is no longer needed since PE isn't accumulated in `_pairForce`. Remove it.

**Step 5: Commit**

```bash
git add src/physics.js
git commit -m "refactor: extract PE computation from _pairForce into dedicated BH-consistent method

PE is now computed in a separate tree traversal using the same BH theta
criterion as forces, ensuring self-consistent Hamiltonian tracking.
When BH is off, uses exact pairwise i<j computation."
```

---

### Task 12: Fix relativistic bounce friction

**Files:**
- Modify: `src/physics.js`

**Step 1: Fix relativistic bounce path**

In `resolveBounce()`, in the relativistic block, replace the spin friction section (~lines 564-569):
```js
// Spin friction: Δspin = J·r / I = J / (INERTIA_K·m·r)
// Same sign for both — torque arms on opposite sides
p1.spin -= tangentialImpulse / (INERTIA_K * m1 * p1.radius);
p2.spin -= tangentialImpulse / (INERTIA_K * m2 * p2.radius);
p1.angVel = spinToAngVel(p1.spin, p1.radius);
p2.angVel = spinToAngVel(p2.spin, p2.radius);
```

With (using the already-renamed `angw` and `angwToAngVel`):
```js
// Spin friction: compute new coordinate ω, then convert to angular celerity
const I1 = INERTIA_K * m1 * p1.radius * p1.radius;
const I2 = INERTIA_K * m2 * p2.radius * p2.radius;
const omega1New = p1.angVel - tangentialImpulse / I1;
const omega2New = p2.angVel - tangentialImpulse / I2;
p1.angw = angVelToAngw(omega1New, p1.radius);
p2.angw = angVelToAngw(omega2New, p2.radius);
p1.angVel = angwToAngVel(p1.angw, p1.radius);
p2.angVel = angwToAngVel(p2.angw, p2.radius);
```

**Step 2: Fix classical bounce path**

In the classical bounce block, replace:
```js
p1.spin -= tangentialImpulse / (INERTIA_K * m1 * p1.radius);
p2.spin -= tangentialImpulse / (INERTIA_K * m2 * p2.radius);
p1.angVel = p1.spin;
p2.angVel = p2.spin;
```

With:
```js
const I1 = INERTIA_K * m1 * p1.radius * p1.radius;
const I2 = INERTIA_K * m2 * p2.radius * p2.radius;
p1.angw -= tangentialImpulse / I1;
p2.angw -= tangentialImpulse / I2;
p1.angVel = p1.angw;
p2.angVel = p2.angw;
```

Note: In the classical case, `angw = angVel` (identity mapping), so we can modify `angw` directly. The change from `J/(INERTIA_K*m*r)` to `J/I = J/(INERTIA_K*m*r²)` also fixes the classical formula — the old code divided by `INERTIA_K*m*r` (units of angular impulse / (mass·radius)) instead of `INERTIA_K*m*r²` (= I, moment of inertia). This is an additional bug fix: `Δω = J/I`, not `J/(mr)`.

**Step 3: Verify**

Load simulation, enable bounce collisions. Place several spinning particles. Verify:
- Bounce collisions transfer angular momentum correctly
- Spin rings change direction on collision
- No NaN or extreme values after bounce

**Step 4: Commit**

```bash
git add src/physics.js
git commit -m "fix: correct relativistic bounce friction to use proper angw conversion

Tangential impulse now properly converts through coordinate angular
velocity space: compute new ω = ω_old - J/I, then convert to angular
celerity via angVelToAngw. Also fixes classical path: was dividing
by INERTIA_K*m*r instead of I = INERTIA_K*m*r²."
```

---

### Task 13: Add BH/signal-delay UI dependency

**Files:**
- Modify: `src/ui.js`

**Step 1: Add dependency handler**

After the relativity dependency block (~line 141), add:

```js
// ─── Barnes-Hut dependency: Signal Delay requires pairwise mode ───
const bhEl = document.getElementById('barneshut-toggle');
const sdEl = document.getElementById('signaldelay-toggle');
const updateBhDeps = () => {
    const bhOn = bhEl.checked;
    sdEl.disabled = bhOn || !relativityEl.checked;
    sdEl.closest('.ctrl-row').classList.toggle('ctrl-disabled',
        bhOn || !relativityEl.checked);
};
bhEl.addEventListener('change', updateBhDeps);
// Also update when relativity changes (since signal delay depends on both)
relativityEl.addEventListener('change', updateBhDeps);
updateBhDeps();
```

Update the existing `updateRelDeps` to not manage signal delay (it's now managed by `updateBhDeps`):

Change `relDepIds` from:
```js
const relDepIds = ['radiation-toggle', 'signaldelay-toggle', 'spinorbit-toggle'];
```
To:
```js
const relDepIds = ['radiation-toggle', 'spinorbit-toggle'];
```

**Step 2: Commit**

```bash
git add src/ui.js
git commit -m "feat: gray out signal delay when Barnes-Hut is on

Signal delay requires pairwise force computation (incompatible with
BH tree traversal). The toggle is now disabled when BH is on, using
the same .ctrl-disabled pattern as relativity dependencies."
```

---

### Task 14: Update info tips

**Files:**
- Modify: `src/ui.js`

**Step 1: Update infoData object**

Replace the `infoData` object entries that need updates:

```js
energy: { title: 'Energy Conservation', body: 'Total energy = Linear KE + Spin KE + Potential + Field + Radiated. Field energy includes both EM and gravitational Darwin Lagrangian O(v\u00B2/c\u00B2) corrections. Radiated tracks cumulative energy lost to Larmor radiation. Drift indicates numerical integration error. Spin KE uses (I/r\u00B2)\u00B7(\u221A(1+S\u00B2r\u00B2)\u22121) relativistically, \u00BDI\u03C9\u00B2 classically, where I = (2/5)mr\u00B2.' },
conserved: { title: 'Conserved Quantities', body: 'Momentum = |p_particle + p_field + p_radiated| (vector sum). Particle momentum is \u03A3(m\u1D62w\u1D62). Field momentum includes EM and gravitational Darwin terms from charged and massive particle pairs. Radiated momentum accumulates from Larmor radiation recoil. Angular momentum about the center of mass splits into orbital \u03A3(r\u1D62 \u00D7 m\u1D62w\u1D62) and spin \u03A3(I\u1D62W\u1D62) where I = (2/5)mr\u00B2 and W is angular celerity. Conserved with gravity and Coulomb only. Velocity-dependent forces (Lorentz, linear gravitomagnetism) do not obey Newton\u2019s third law between particles \u2014 in real physics, the missing momentum is carried by the field.' },
spin: { title: 'Spin (Angular Celerity)', body: 'The angular celerity W (state variable) is the rotational analog of proper velocity w. Angular velocity \u03C9 = W/\u221A(1+W\u00B2r\u00B2) naturally caps surface velocity below c. Determines magnetic moment and angular momentum. Positive = counter-clockwise, negative = clockwise.' },
radiation: { title: 'Larmor Radiation', body: 'Accelerating charges radiate energy via the Larmor formula P = 2q\u00B2a\u00B2/3 (natural units). Applied as Landau\u2013Lifshitz force: jerk term \u03C4\u00B7dF/dt minus Schott damping \u03C4\u00B7F\u00B2v/m, where \u03C4 = 2q\u00B2/(3m). Relativistic correction divides by \u03B3\u00B3. Creates orbital decay in charge\u2013charge systems.' },
signaldelay: { title: 'Signal Delay', body: 'Finite-speed force propagation. Forces use each source particle\u2019s past position and velocity from its history buffer, solving the light-cone equation via Newton\u2013Raphson iteration. Creates realistic lag in distant interactions. Requires Relativity and pairwise mode (incompatible with Barnes\u2013Hut).' },
spinorbit: { title: 'Spin\u2013Orbit Coupling', body: 'Transfers energy between translational and spin KE. EM: dE = \u2212\u03BC\u00B7(v\u00B7\u2207B_z)\u00B7dt where \u03BC = \u2155q\u03C9r\u00B2 and \u2207B_z includes both radial and angular gradient terms. GM: same pattern using angular momentum L = I\u03C9 and \u2207Bg_z. Requires Relativity and the relevant force toggle.' },
```

**Step 2: Commit**

```bash
git add src/ui.js
git commit -m "docs: update info tip text for corrected physics

Reflects: corrected Larmor formula, corrected spin KE formula, angular
celerity terminology, gravitational Darwin terms, corrected spin-orbit
gradient, signal delay BH incompatibility note."
```

---

### Task 15: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update all references**

Key updates needed:
- Rename `p.spin` → `p.angw` / `spinToAngVel` → `angwToAngVel` in all documentation sections
- Update spin KE formula in Energy Conservation section
- Add gravitational Darwin field energy/momentum to Energy Conservation and Conserved Quantities sections
- Update Larmor radiation description (P = 2q²a²/3, τ = 2q²/(3m))
- Update Schott damping description (subtract, not add)
- Update spin-orbit gradient description (correct sign + angular terms)
- Add note that signal delay requires pairwise mode (BH off)
- Update bounce friction description (proper angVel↔angw conversion)
- Add `src/energy.js` to the module dependency graph
- Update PE description (separate BH-consistent computation)
- Mention `angVelToAngw` in relativity.js

This is a documentation-only task. Update each section systematically.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for physics accuracy refactor

Reflects all changes: spin→angw rename, corrected formulas, energy.js
module, BH PE, gravitational Darwin terms, bounce friction fix,
signal delay BH dependency."
```

---

### Task 16: Final verification

**Step 1: Full regression test**

Load each preset and verify no console errors:
- Solar System: stable orbits, energy drift < ±1%
- Binary Stars: orbit each other, spin rings visible
- Galaxy: particles orbit core, no blowup
- Collision: two groups collide
- Magnetic: charged spinning particles interact

**Step 2: Test specific fixes**

1. **Radiation**: Enable Relativity + Radiation. Two opposite charges orbiting. Verify orbit decays (Schott damping works). Energy drift tracks radiated energy.

2. **Signal delay**: Turn BH OFF, enable signal delay. Verify ghost circles appear. Turn BH ON — verify signal delay toggle grays out.

3. **Bounce friction**: Enable bounce collisions. Place particles with high spin near each other. Verify angular momentum transfers on collision. Check energy before/after bounce is reasonable.

4. **Spin-orbit**: Enable Relativity + Magnetic + Spin-Orbit. Place charged spinning particles. Verify spin changes affect translational KE (look at spin KE and linear KE in stats).

5. **Gravitational Darwin**: Enable gravitomagnetic, give particles velocity. Check that Field energy shows non-zero value in Stats tab (previously only showed EM Darwin).

6. **BH PE**: Toggle BH on/off with the Solar System preset. Compare PE values — they should be similar (not identical due to approximation, but within ~5%).
