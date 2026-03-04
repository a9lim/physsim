# Physics Accuracy & Refactor Design

Date: 2026-03-03

## Context

Thorough audit of physsim's physical accuracy revealed several bugs in constants, formulas, and energy bookkeeping, plus opportunities for structural improvement. The simulation uses natural units (G=c=1, epsilon_0=1/(4pi)) with a proper velocity (celerity) framework and Boris integrator.

## Scope

Bug fixes + physics core refactor + PE accuracy fix + info tip updates + relativistic bounce friction fix. No new visual features (light cone, presets) in this batch.

---

## 1. Constant & Formula Bug Fixes

### 1a. LARMOR_K (config.js)

**Bug:** `LARMOR_K = 1/(6*Math.PI)` — unit conversion error. The SI Larmor formula `P = q^2 a^2 / (6*pi*eps_0*c^3)` was simplified by dropping `eps_0` and `c` without substituting `eps_0 = 1/(4*pi)`, `c = 1`.

**Correct value:** With eps_0 = 1/(4pi), c = 1: `P = 2*q^2*a^2/3`. The Abraham-Lorentz time constant `tau = 2*q^2/(3*m)` requires `LARMOR_K = 1/3`.

**Fix:** `LARMOR_K = 1/3` with updated comment:
```js
export const LARMOR_K = 1 / 3; // tau = 2*LARMOR_K*q^2/m = 2q^2/(3m), P = 2q^2a^2/3 (c=G=1, eps_0=1/(4pi))
```

Radiation was ~6.3x too weak. This fix strengthens it, so LL_FORCE_CLAMP may need tuning for stability.

### 1b. Schott Damping Sign (physics.js)

**Bug:** `fRad = tau * (jerkX + schottX)` — the Schott term `F^2*v/m` is added, pushing in the direction of motion (accelerating). The LL approximation requires subtraction (decelerating).

**Fix:** `fRad = tau * (jerkX - schottX)` (and Y component).

### 1c. Relativistic Spin KE (energy.js, formerly in main.js)

**Bug:** `E_spin = m(sqrt(1 + L^2/m^2) - 1)` where `L = I*S`. NR limit gives `L^2/(2m) = I^2*S^2/(2m)`, not `L^2/(2I) = (1/2)*I*omega^2`. Off by factor `I/m = (2/5)*r^2`.

**Correct formula:** `E_spin = (I/r^2) * (sqrt(1 + S^2*r^2) - 1) = INERTIA_K * m * (sqrt(1 + S^2*r^2) - 1)`

Derivation:
- `m_rot = I/r^2 = INERTIA_K * m` (the "rotational mass")
- `u = S*r` (surface celerity, analog of `w` for linear)
- `E_spin = m_rot * (sqrt(1 + u^2) - 1)` — exact analog of `E_linear = m*(sqrt(1+w^2)-1)`
- NR: `approx m_rot * u^2/2 = (I/r^2)*S^2*r^2/2 = I*S^2/2 = (1/2)*I*omega^2` (since S ~ omega in NR)
- `dE/dS = I*omega` — consistent with spin-orbit energy transfer
- Diverges as surface velocity -> c

```js
const srSq = p.angw * p.angw * rSq;
spinKE += INERTIA_K * p.mass * (Math.sqrt(1 + srSq) - 1);
```

### 1d. Spin-Orbit Gradient (physics.js)

**Bug:** B_z gradient computed as `-3*Bz/r * r_hat`, but the correct gradient of `B_z = q_s*(v_s x r)_z / r^3` w.r.t. observer position is:

```
dBz/dpx = +3*Bz*rx/r^2 + q_s*vsy/r^3
dBz/dpy = +3*Bz*ry/r^2 - q_s*vsx/r^3
```

Two errors: (1) sign flipped (-3 should be +3), (2) angular derivative terms missing.

**Fix:** Replace gradient accumulation in `_pairForce`:
```js
// EM B_z gradient (correct: radial + angular terms)
const Bz_contrib = sCharge * crossSV * invR * invRSq;
p.dBzdx += 3 * Bz_contrib * rx * invRSq + sCharge * svy * invR * invRSq;
p.dBzdy += 3 * Bz_contrib * ry * invRSq - sCharge * svx * invR * invRSq;
```

Same pattern for GM gradient (using sMass instead of sCharge).

### 1e. Photon Absorption Bookkeeping (main.js)

**Bug:** When a photon is absorbed, its energy transfers to the absorbing particle's KE, but `totalRadiated` is not decremented. Total energy (KE+PE+Field+Radiated) drifts upward on each absorption.

**Fix:** After photon absorption, subtract photon energy from totalRadiated:
```js
this.totalRadiated = Math.max(0, this.totalRadiated - ph.energy);
```

Also subtract from radiated momentum (proportionally in photon's direction):
```js
this.totalRadiatedPx -= ph.vel.x * ph.energy;
this.totalRadiatedPy -= ph.vel.y * ph.energy;
```

### 1f. Frame-Dragging Parameter Name (physics.js)

**Not a bug** but misleading: `_pairForce` parameter `sSpin` is actually passed `o.angVel` (coordinate angular velocity) in the pairwise loop. Rename parameter to `sAngVel` for clarity. The frame-dragging torque `(sAngVel - p.angVel)` already compares coordinate-to-coordinate correctly.

---

## 2. Rename: spin -> angw

Rename the `spin` state variable to `angw` (angular celerity) everywhere for clarity. This mirrors `w` (linear celerity/proper velocity).

| Old | New | Meaning |
|-----|-----|---------|
| `p.spin` | `p.angw` | Angular celerity (proper angular velocity, unbounded) |
| `p.angVel` | `p.angVel` | Coordinate angular velocity (derived, bounded by c/r) |
| `spinToAngVel()` | `angwToAngVel()` | Convert angular celerity to coordinate angular velocity |
| `spinKE` | `spinKE` | (display name unchanged — "spin KE" is standard) |
| input slider "spin" | input slider "spin" | (UI label unchanged — "angular celerity" too technical for UI) |

Files touched: `particle.js`, `physics.js`, `main.js`, `relativity.js`, `input.js`, `presets.js`, `ui.js`, `renderer.js`, `quadtree.js`.

The `angwToAngVel` function in `relativity.js`:
```js
export function angwToAngVel(angw, radius) {
    return angw / Math.sqrt(1 + angw * angw * radius * radius);
}
```

Add inverse `angVelToAngw` for the bounce friction fix:
```js
export function angVelToAngw(angVel, radius) {
    const sr = angVel * radius;
    const srSq = sr * sr;
    if (srSq >= 1) {
        const clampedSr = MAX_SPEED_RATIO;
        return Math.sign(angVel) * clampedSr / (radius * Math.sqrt(1 - clampedSr * clampedSr));
    }
    return angVel / Math.sqrt(1 - srSq);
}
```

---

## 3. Energy Module Extraction

### New file: `src/energy.js`

Extract energy/momentum computation from `main.js` into a dedicated module.

**Exports:**
```js
export function computeEnergies(particles, physics, sim) {
    // Returns: { linearKE, spinKE, pe, fieldEnergy, fieldPx, fieldPy,
    //            px, py, orbitalAngMom, spinAngMom, comX, comY }
}
```

**Contents:**
- Linear KE: relativistic `(gamma-1)*m` or classical `(1/2)*m*v^2`
- Spin KE: `INERTIA_K*m*(sqrt(1+S^2*r^2)-1)` or classical `(1/2)*I*omega^2`
- PE: from `physics.potentialEnergy` (computed during force pass or dedicated PE pass)
- EM Darwin field energy: `-(1/2) * SUM_{i<j} (qi*qj/r) * [(vi.vj) + (vi.r_hat)(vj.r_hat)]`
- **NEW:** Gravitational Darwin field energy: `+(1/2) * SUM_{i<j} (mi*mj/r) * [(vi.vj) + (vi.r_hat)(vj.r_hat)]` (same structure, opposite sign — gravity attracts)
- EM field momentum (existing)
- **NEW:** Gravitational field momentum (same structure as EM, using masses)
- Particle momentum: `SUM(m*w)`
- Angular momentum about COM (orbital + spin)

**main.js changes:** `computeEnergy()` calls `computeEnergies()` and formats results for DOM.

---

## 4. PE Fix for Barnes-Hut (O(N log N))

### Problem

PE accumulated in `_pairForce` with `*0.5` assumes each pair counted twice. BH aggregate nodes break this assumption.

### Solution

Compute PE using the same BH tree traversal and theta criterion as forces. Each particle traverses the tree, accumulating PE contributions. Total divided by 2.

**New method:** `Physics.computePE(particles, qt)` that mirrors `calculateForce` tree traversal but outputs PE only. Uses same `BH_THETA`, same aggregate quantities (totalMass, totalCharge, totalMagneticMoment, totalAngularMomentum).

When BH is off, degenerates to exact pairwise (i<j) — no /2 needed. When BH is on, PE uses the same approximation as forces, giving self-consistent Hamiltonian tracking.

Remove PE accumulation from `_pairForce` (it no longer has that responsibility).

---

## 5. Relativistic Bounce Friction Fix

### Problem

Tangential impulse computed from coordinate surface velocities, but spin update does `angw += J/(I)` — directly modifying proper angular velocity using a coordinate-space impulse, without accounting for the non-linear mapping.

### Fix

1. Compute tangential impulse `J` from coordinate surface velocities (unchanged — correct).
2. Compute new coordinate angular velocity: `omega_new = omega - J/I`
3. Convert back to angular celerity: `angw_new = angVelToAngw(omega_new, r)`
4. Clamp to prevent superluminal surface velocity.

Applies to both relativistic and classical bounce paths (classical path has identity mapping, so result is unchanged).

---

## 6. Signal Delay + BH UI Dependency

Gray out signal delay toggle when Barnes-Hut is on (incompatible — signal delay only works in pairwise mode).

In `ui.js`:
- BH toggle change handler checks signal delay, applies `.ctrl-disabled`
- Same pattern as relativity -> radiation/signaldelay/spinorbit dependencies
- If BH is turned on while signal delay is active, signal delay is visually disabled but the physics flag stays — the code already ignores signal delay in BH path

---

## 7. Info Tip Updates

Update all info tip text in `ui.js` `infoData` to reflect:
- Corrected Larmor formula (P = 2q^2*a^2/3)
- Corrected spin KE formula (INERTIA_K*m*(sqrt(1+S^2*r^2)-1))
- "Angular celerity" terminology for the state variable (keep "spin" for the physical concept)
- Gravitational Darwin field energy/momentum in conserved quantities description
- Spin-orbit coupling corrected gradient description
- Frame-dragging torque clarification

---

## Files Changed Summary

| File | Changes |
|------|---------|
| `src/config.js` | Fix LARMOR_K = 1/3 |
| `src/relativity.js` | Rename `spinToAngVel` -> `angwToAngVel`, add `angVelToAngw` |
| `src/particle.js` | Rename `spin` -> `angw` |
| `src/physics.js` | Schott sign fix, gradient fix, PE extraction, `spin` -> `angw`, `sSpin` -> `sAngVel`, BH PE method |
| `src/energy.js` | **NEW** — energy/momentum computation with gravitational Darwin terms |
| `main.js` | Thin `computeEnergy` wrapper, photon absorption fix, `spin` -> `angw`, bounce friction fix |
| `src/input.js` | `spin` -> `angw` in spawn |
| `src/presets.js` | `spin` -> `angw` in preset definitions |
| `src/ui.js` | Info tip updates, BH/signal-delay dependency, `spin` -> `angw` references |
| `src/renderer.js` | `angVel` references only (no `spin` used) — minimal changes |
| `src/quadtree.js` | `angVel` references only — minimal changes |
| `CLAUDE.md` | Update to reflect all changes |
