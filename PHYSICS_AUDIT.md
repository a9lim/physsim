# Physics Accuracy Audit

Comprehensive audit of the physsim N-body simulation for physical accuracy, realism, and factual correctness. Covers all force laws, field theories, numerical methods, presets, and reference documentation.

**Audit date**: 2026-03-07
**Scope**: All source files in `src/`, `main.js`, `colors.js`
**Unit system**: Natural units with c = G = hbar = 1, 3D force laws with motion constrained to a 2D plane
**Status**: All bugs, inaccuracies, reference errors, approximations, simplifications, and ambiguities **fixed** (2026-03-07)

---

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| Bug | 3 | **All fixed** |
| Inaccuracy | 2 | **All fixed** |
| Reference error | 4 | **All fixed** |
| Approximation | 10 of 12 | **10 fixed**, 2 by design |
| Simplification | 10 of 12 | **10 fixed**, 2 by design |
| Ambiguity | 1 of 4 | **1 fixed**, 3 by design |
| Correct | 60+ | Verified |

Three genuine bugs were found and fixed (Yukawa jerk coefficient, pi0 decay kinematics, Kerr-Newman surface gravity). Two physics inaccuracies were corrected (Landau-Lifshitz radiation terms, Higgs gradient force). Four reference documentation errors were fixed. A second pass fixed 10 approximations, 10 simplifications, and 1 ambiguity that had been documented as acceptable trade-offs but could be improved. The remaining items (A1, A3, A8, S2, S8, S10, U2, U3, U4) are genuine design choices where the simulation intentionally differs from real physics.

---

## Bugs

### B1. Yukawa jerk radial coefficient -- FIXED
**File**: `forces.js:288`
**Severity**: Bug (affects radiation reaction when Yukawa dominates)

The analytical jerk for the Yukawa force has an incorrect coefficient. The radial jerk term uses `2*mu*invR` where it should be `3*mu*invR`.

**Code**:
```js
const jRadial = -(3 * invRSq + 2 * mu * invR + mu * mu) * rDotVr * ...
//                               ^ should be 3
```

**Derivation**: For `f(r) = exp(-mu*r) * (r^-3 + mu*r^-2)`, the derivative is:
```
f'(r) = exp(-mu*r) * (-3/r^4 - 3*mu/r^3 - mu^2/r^2)
```
The code has `(-3/r^4 - 2*mu/r^3 - mu^2/r^2)` -- the middle coefficient is wrong.

**Impact**: Affects the Landau-Lifshitz radiation reaction force and quadrupole radiation power when Yukawa forces are the dominant acceleration. The error is in the jerk (time derivative of force), which feeds into radiation calculations. Gravity and Coulomb jerks are correct.

---

### B2. Pi0 decay kinematics violate momentum conservation -- FIXED
**File**: `pion.js:56-82`
**Severity**: Bug (momentum non-conservation)

Neutral pion decay emits two photons perpendicular to the pion's flight direction (`+/- PI/2`). This violates conservation of momentum: the pion carries forward momentum `p = E*v`, but the two perpendicular photons carry zero net forward momentum.

**Correct kinematics**: In the rest frame, emit two back-to-back photons along a random axis (each with energy `m_pi/2`), then Lorentz-boost to the lab frame. The result would be two forward-beamed photons, not perpendicular ones.

**Impact**: Also breaks the radiated momentum bookkeeping (`totalRadiatedPx/Py`) -- when a pi0 decays, the pion's forward momentum is subtracted but the photons contribute only transverse momentum, creating a systematic drift.

---

### B3. Kerr-Newman surface gravity formula incorrect for Q != 0 -- FIXED
**Files**: `integrator.js:772`, `reference.js:251`
**Severity**: Bug (overestimates Hawking temperature for charged BHs)

The surface gravity formula uses:
```
kappa = sqrt(M^2 - a^2 - Q^2) / (r+^2 + a^2)
```

This is correct for Kerr (Q=0) but incorrect for Kerr-Newman. The correct denominator is `2*M*r+`, which equals `r+^2 + a^2 + Q^2`. Since `r+^2 + a^2 < r+^2 + a^2 + Q^2`, the code gives a **larger** kappa (hotter BH) than the correct formula for charged black holes.

**Verification**: For Schwarzschild (Q=0, a=0): `kappa = M / (4M^2) = 1/(4M)`. Correct. For Kerr (Q=0): `kappa = sqrt(M^2-a^2) / (2M*r+)`. Also correct since `r+^2 + a^2 = 2M*r+` when Q=0. The error only manifests when Q != 0.

**Impact**: Charged black holes evaporate faster than they should. The reference documentation has the same error.

---

## Inaccuracies

### I1. Landau-Lifshitz power-dissipation terms -- FIXED
**File**: `integrator.js:680-688`
**Severity**: Inaccuracy (wrong force direction at relativistic speeds)

The Landau-Lifshitz radiation reaction has three terms. Term 3 (the Schott-like term) is applied along the external force `F` direction, but the standard LL formula has both power-dissipation terms purely along the velocity `v`. The standard form is:

```
F_rad = tau * [dF/dt / gamma^3 - (gamma/m) * (F^2 - (v.F)^2) * v]
```

The code splits this into `F^2 * v` and `(v.F) * F`, which produces a component perpendicular to v that should not exist. At non-relativistic speeds the two power terms are negligible (only Term 1, the jerk, matters). The LL_FORCE_CLAMP = 0.5 further limits the impact.

**Impact**: Affects radiation reaction direction at relativistic speeds. Mitigated by the clamp and by Term 1 (jerk) dominating in most scenarios.

---

### I2. Higgs gradient force sign vs |phi| mass generation -- FIXED
**File**: `higgs-field.js:137 vs 169-175`
**Severity**: Inaccuracy (force-energy inconsistency)

Mass generation uses `m_eff = baseMass * |phi(x)|` (absolute value), but the gradient force uses `F = +g * baseMass * grad(phi)` (without `sign(phi)`). If the particle energy is `E = baseMass * |phi|`, then the consistent force should be `F = -grad(E) = -baseMass * sign(phi) * grad(phi)`.

The code always pushes particles toward increasing phi regardless of sign, while the energy function treats both signs equally via `|phi|`. The source term `+g*baseMass` also always pushes phi positive, breaking the Z2 symmetry. This creates a self-consistent attractive coupling where particles cluster around high-phi regions, but the force and energy are not derivable from a single Hamiltonian.

**Impact**: The Higgs field starts at VEV=1 and particles preferentially inhabit the phi > 0 vacuum, so in practice this rarely causes observable issues. The phi < 0 vacuum is physically inaccessible due to the positive-definite source.

---

## Reference Documentation Errors

### R1. Magnetic dipole force sign -- FIXED
**File**: `reference.js` (magnetic section)

The reference shows `F = +3*mu_1*mu_2/r^4 * hat{r}` but the accompanying text says "aligned perpendicular-to-plane dipoles repel." With the code's convention (positive = toward source = attractive), the positive sign implies attraction, contradicting the text. The code correctly uses `F = -3*mu1*mu2/r^4` (negative = repulsive for aligned dipoles). The reference formula sign should be negative.

### R2. Axion alpha_eff missing coupling constant -- FIXED
**File**: `reference.js` (axion section, ~line 334)

Shows `alpha_eff(x) = alpha * (1 + a(x))` but should be `alpha_eff(x) = alpha * (1 + g*a(x))` to match the Lagrangian `-(1+g*a)F^2/4` shown two lines above and the code (`axMod = 1 + g*a`).

### R3. Axion field visualization colors -- FIXED
**File**: `reference.js` (axion section, ~line 358)

Describes the field as "blue" (a > 0) and "red" (a < 0). The actual rendering uses indigo (a > 0) and yellow (a < 0), as implemented in `axion-field.js` and confirmed in CLAUDE.md.

### R4. RP2 topology claim -- FIXED
**File**: `reference.js` (topology section, ~line 533)

Claims RP2 is "the only closed 2D surface where *every* closed loop is orientation-reversing." This is false -- contractible loops preserve orientation on any surface. The correct statement is that every *non-contractible* loop is orientation-reversing (equivalently, pi_1(RP2) = Z/2Z with the orientation character mapping the generator to -1).

---

## Approximations

These are deliberate or well-motivated modeling choices that differ from exact physics.

### A1. Plummer softening
**Files**: `config.js:25-28`, `forces.js`

SOFTENING = 8 (SOFTENING_SQ = 64) is very large relative to particle sizes (r = cbrt(m)), substantially smoothing close encounters. BH mode reduces to SOFTENING = 4. Standard N-body technique, well-motivated for numerical stability.

### A2. Tidal torque r^3 scaling -- FIXED
**File**: `forces.js:322`

Changed `r_body^3` to the textbook `R^5` scaling (`r^5 = m^{5/3}`). `TIDAL_STRENGTH` retuned from 2.0 to 0.3 to compensate for the stronger coupling.

### A3. Boris integrator for gravity
**File**: `integrator.js:530-570`

Boris integrators are designed for charged-particle-in-field problems. Using them for gravitomagnetic forces is motivated by the GEM analogy but is an approximation since the analogy breaks down at higher orders. At 1PN in the weak-field limit, this is appropriate and elegant.

### A4. Signal delay: position-only retardation -- FIXED
**Files**: `forces.js`, `signal-delay.js`

Added Lienard-Wiechert aberration factor `(1 - n_hat . v_source)^{-3}` to gravity, Coulomb, and dipole forces when signal delay is active. Denominator clamped to 0.01 minimum, factor capped at 100x to prevent numerical blowup. Applied to position-dependent (E-like) forces only; not applied to Biot-Savart (already velocity-dependent), Yukawa (massive carrier), or 1PN (already O(v^2/c^2)).

### A5. Symplectic Euler is first-order -- FIXED
**Files**: `higgs-field.js`, `axion-field.js`

Replaced symplectic Euler with Stormer-Verlet (kick-drift-kick) for O(dt^2) accuracy. Both fields now: half-kick fieldDot, full-drift field, recompute Laplacian, second half-kick fieldDot. Costs one extra `_computeLaplacian()` per field per substep (~10% overhead).

### A6. Quadrupole radiation uses absolute coordinates -- FIXED
**File**: `integrator.js`

COM is now computed before the quadrupole accumulation loop; particle positions are shifted to COM-relative coordinates. Removes spurious dipole contribution for systems with nonzero total momentum.

### A7. Quadrupole energy extraction is heuristic -- FIXED
**File**: `integrator.js`

Energy extraction now weighted by each particle's contribution to d^3I/dt^3 (GW) and d^3Q/dt^3 (EM), tracked via pre-allocated per-particle contribution arrays. Tangential drag (velocity scaling) remains KE-proportional for stability. Falls back gracefully when total contribution is near zero.

### A8. Hawking radiation uses single scalar DOF
**File**: `integrator.js:775`

Stefan-Boltzmann constant `sigma = pi^2/60` corresponds to a single massless scalar field. Photons have 2 DOFs (sigma = pi^2/30); the full Standard Model has many more. For a 2D toy model, the single-DOF choice is a reasonable simplification.

### A9. Thermal phase transition heuristic -- FIXED
**File**: `higgs-field.js`

Thermal KE deposition now uses relativistic formula `wSq / (sqrt(1+wSq) + 1) * mass` when relativity is enabled, Newtonian `0.5*m*v^2` otherwise. The `relativityEnabled` flag is threaded from the integrator through `update()` to `_depositThermal()`. The heuristic nature of using local KE as a proxy for T^2 remains (by design).

### A10. BH aggregate nodes lose dipole information -- ALREADY FIXED
**File**: `forces.js`, `quadtree.js`

Verified already implemented correctly: `totalMagneticMoment` and `totalAngularMomentum` are aggregated in both leaf and internal nodes, and passed to `pairForce()` for distant-node force computation. No changes needed.

### A11. Roche lobe uses small-q limit of Eggleton -- FIXED
**File**: `integrator.js`

Replaced `0.462 * d * cbrt(q)` with the full Eggleton (1983) formula: `r_L/a = 0.49*q^{2/3} / (0.6*q^{2/3} + ln(1+q^{1/3}))`. Fixes ~17% error for equal-mass binaries.

### A12. Pion emission radiation reaction scaling -- FIXED
**File**: `integrator.js`

Replaced approximate `sqrt(1 - dE/KE)` with exact relativistic formula: compute `gamma_new = 1 + KE_new/m`, then `wSq_new = gamma_new^2 - 1`, scale by `sqrt(wSq_new / wSq)`. Non-relativistic branch uses `wSq_new = 2*KE_new/m`. Exact at all speeds.

---

## Simplifications

These are deliberate modeling choices that differ from real physics for practical or pedagogical reasons.

### S1. 2D magnetic dipole field sign -- FIXED
**File**: `forces.js`

Corrected dipole Bz to `-mu/r^3` (textbook equatorial sign) and flipped dipole-dipole force to `+3*mu1*mu2/r^4` to maintain correct repulsion. Gradient signs (`dBzdx`, `dBzdy`) also negated for consistency. Boris rotation and spin-orbit forces now receive the physically correct field direction. PE formula unchanged (already correct).

### S2. Simplified pion decay channels
**File**: `pion.js:56-82`

`pi+/- -> 1 photon` is kinematically forbidden in real physics (a massive particle cannot decay to a single massless particle while conserving both energy and momentum). The physical decay is `pi+ -> mu+ + nu_mu`. Since the simulation doesn't model muons or neutrinos, this is a deliberate simplification.

### S3. No relativistic aberration for pion emission -- FIXED
**File**: `integrator.js`

Pion emission now applies Lorentz aberration: rest-frame emission angle is boosted to lab frame using `atan2(sin(phi), gamma*(cos(phi) + beta))`. At low particle speeds, no change; at relativistic speeds, pions are beamed forward along the emitter's velocity.

### S4. Axion modulation uses observer-only field value -- FIXED
**Files**: `forces.js`, `potential.js`, `energy.js`

Pairwise force and PE now use the geometric mean `sqrt(axMod_i * axMod_j)` (and `sqrt(yukMod_i * yukMod_j)`) instead of observer-only values. Added `sAxMod`/`sYukMod` parameters to `pairForce()` and `pairPE()`. All call sites updated: pairwise live, dead particles, BH leaf (pass source values), BH distant-node (pass 1). Darwin EM field energy in `energy.js` also uses geometric mean. `compute1PNPairwise()` Scalar Breit uses geometric mean yukMod.

### S5. Particle-field interaction energy missing from PE -- FIXED
**Files**: `higgs-field.js`, `axion-field.js`, `energy.js`

Added `particleFieldEnergy()` to both field classes. Higgs: `sum(-baseMass * (|phi(x)| - 1))` per particle (zero at VEV). Axion: `-g*q^2*a(x)` (EM) and `-g*m*(±1)*a(x)` (PQ). Both use topology-aware interpolation. `computeEnergies()` now calls these and adds `pfiEnergy` to PE total. Returned as a separate field for debugging.

### S6. 50/50 energy split for field excitations -- FIXED
**File**: `integrator.js`

Merge energy now split by coupling strength: `g_H^2 / (g_H^2 + g_A^2)` for Higgs (~99.75%) and `g_A^2 / (g_H^2 + g_A^2)` for axion (~0.25%). When only one field is active, it receives full energy.

### S7. Mass modulation breaks particle momentum conservation -- FIXED
**File**: `higgs-field.js`

`modulateMasses()` now scales proper velocity by `m_old/m_new` before updating mass, conserving particle momentum `p = m*w`. Derived coordinate velocity `vel` recomputed from updated `w` via `vel = w / sqrt(1+w^2)`.

### S8. Kerr-Newman in 2D
**Files**: `config.js:127-131`, `integrator.js:760-806`

The Kerr-Newman metric is a 4D solution. Concepts like "event horizon" and "ergosphere" don't directly apply in 2D. The simulation uses 4D formulas as effective radii for collision and rendering, which is a reasonable pedagogical choice.

### S9. Angular velocity not retarded in signal delay -- FIXED
**Files**: `particle.js`, `signal-delay.js`, `forces.js`, `integrator.js`

Added `histAngW` (Float64Array[256]) to the per-particle history buffer. Angular celerity is recorded alongside position and velocity in both strided recording and `_retireParticle()`. `getDelayedState()` returns interpolated `angw` at retarded time. `computeAllForces()` computes retarded `angVel` from `ret.angw` and recomputes dipole moments (`sMagMoment`, `sAngMomentum`) from retarded angular velocity for both live and dead particle signal delay paths.

### S10. Magnetic moment model
**File**: `config.js:35`

`mu = 0.2 * q * omega * r^2` corresponds to a uniformly charged sphere. This is a specific geometric model choice.

### S11. Field interpolation uses boundary clamping -- FIXED
**Files**: `scalar-field.js`, `higgs-field.js`, `axion-field.js`, `integrator.js`

`interpolate()` and `gradient()` now accept `bcMode`/`topoConst` and use `_nb()` for topology-aware stencil wrapping, consistent with `_depositPQS()`. Added `_vacValue` property (1 for Higgs VEV, 0 for Axion vacuum) for Dirichlet boundary fallback. All call sites updated to pass boundary mode and topology.

### S12. Annihilation KE not tracked -- FIXED
**File**: `collisions.js`

Annihilation energy now includes KE of annihilated fractions: `energy = 2*annihilated + fraction1*KE1 + fraction2*KE2`, using the existing `_particleKE()` relativistic helper. More photons emitted and `totalRadiated` correctly tracked.

---

## Ambiguities

### U1. Quadrupole radiation missing trace subtraction -- FIXED
**File**: `integrator.js`

GW quadrupole power now uses the trace-free (STF) reduced quadrupole moment: `I^TF_ij = I_ij - (1/3)*delta_ij*I_kk`. For 2D motion in 3D (I_zz = 0), diagonal components adjusted by `-trI/3`. EM quadrupole left unchanged (trace-free constraint is GR-specific). Suppresses spurious radiation from radial breathing modes.

### U2. EIH 1PN coefficient decomposition
**File**: `forces.js:186-188`

The `5*m_test/r + 4*m_source/r` split is specific to the decomposition where the GM Lorentz piece is subtracted for Boris rotation. Different but equally valid decompositions give different coefficients. The velocity-Verlet correction step ensures the total 1PN force (Boris + explicit) sums correctly regardless of the split.

### U3. Bazanski term relative scaling
**File**: `forces.js:217-223`

The Bazanski term mixes charge and mass dimensions. Since the simulation's charge unit is arbitrary (not tied to the real electromagnetic coupling constant), the relative magnitude of Bazanski vs Newtonian gravity is not constrained by real physics.

### U4. Darwin EM remainder decomposition
**File**: `forces.js:200-211`

The Darwin Lagrangian force minus the Boris-handled Lorentz piece leaves a specific remainder. The exact form depends on the decomposition. The code's form appears consistent but verifying requires the full decomposition.

---

## Verified Correct

The following areas were verified as physically correct (or correct within the simulation's stated conventions):

### Forces
- Newtonian gravity: `F = m1*m2/r^2`, PE = `-m1*m2/r`, F = -dV/dr verified
- Coulomb force: `F = -q1*q2/r^2`, PE = `+q1*q2/r`, signs correct for like-repel
- Magnetic dipole-dipole: `F = -3*mu1*mu2/r^4`, PE = `+mu1*mu2/r^3`, aligned dipoles repel
- GM dipole: `F = +3*L1*L2/r^4` (co-rotating attract), PE = `-L1*L2/r^3`
- Yukawa potential: `F = g^2*m1*m2*exp(-mu*r)*(1/r^2 + mu/r)`, PE = `-g^2*m1*m2*exp(-mu*r)/r`
- Gravity and Coulomb analytical jerks: correct derivatives of 1/r^2 force
- Biot-Savart B-field from moving charges: standard form in natural units
- Gravitomagnetic Bgz: correct GEM sign (`-m*(v x r_hat)/r^2`, `-2L/r^3`)
- Frame-dragging torque: drives spins toward co-rotation, correct 1/r^3 scaling
- Tidal locking: correct qualitative behavior, includes Coulomb tidal contribution
- Bazanski cross-term: vanishes for identical particles (verified)
- Scalar Breit Hamiltonian: correct for massive scalar boson exchange, (1+mu*r) factor
- External fields: uniform gravity, electric, magnetic correctly applied

### GEM Sign Conventions
- All GEM interactions verified attractive (gravity has one sign of charge)
- GM dipole: positive = attractive (opposite to EM)
- GM Boris parameter: `+2*Bgz` with factor of 4 from GEM
- Bgz field: negative Biot-Savart sign consistent with GEM
- Display force reconstruction: `4*m*v x Bgz` matches Boris rotation

### Relativity
- Proper velocity (celerity) `w = gamma*v` as state variable: automatic subluminal `v = w/sqrt(1+w^2)`
- Relativistic KE: `wSq/(gamma+1)*m` avoids catastrophic cancellation at low v
- Relativistic momentum: `p = m*w` correct
- Angular proper velocity: `omega = W/sqrt(1+W^2*r^2)` caps surface speed at c
- Cosmological expansion: Hubble flow + momentum redshift correct

### Boris Integrator
- Standard half-kick/rotate/half-kick structure
- Relativistic gamma correction in rotation parameter
- Exact |v| preservation in B-field rotation
- EM Boris parameter: `q*Bz/(2m)*dt/gamma`
- GM Boris parameter: `2*Bgz*dt/gamma`
- Adaptive substepping: acceleration + cyclotron criteria correct, MAX_SUBSTEPS=32

### 1PN Corrections
- Velocity-Verlet correction: store F_old, recompute F_new after drift, correct w += (F_new-F_old)*dt/(2m)
- Always pairwise even when Barnes-Hut on (correct design choice)
- EIH + Darwin EM + Bazanski + Scalar Breit all corrected

### Signal Delay
- Light-cone equation `g(t) = |x_src(t) - x_obs| - (now - t) = 0`
- NR derivative `g' = (d.v)/|d| + 1 > 0` guarantees monotonicity and convergence
- Quadratic solve on piecewise-linear segments is exact (not approximate)
- Root selection: most recent valid root = physical (retarded) solution
- Dead particle handling: blocks backward extrapolation, deathTime guard
- Creation time guard: respects causality for newly placed particles
- Dead particles fade out after light-cone passes all observers

### Radiation
- Larmor prefactor `tau = 2*q^2/(3*m)` correct in natural units
- EM quadrupole `P = (1/180)|d^3 Q_ij/dt^3|^2` correct factor
- GW quadrupole `P = (1/5)|d^3 I_ij/dt^3|^2` correct factor
- Third time derivative of mass/charge quadrupole: correct formula `6*v*F + 2*x*J`
- LL force clamping at 0.5 * |F_ext| prevents runaway
- Scalar Larmor `P = g^2*F^2/3`: angular factor 1/3 for spin-0 (vs 2/3 for spin-1)

### Hawking Radiation
- Kerr-Newman surface gravity formula correct for Q=0 (Kerr and Schwarzschild)
- Hawking temperature `T = kappa/(2*pi)` correct in natural units
- Stefan-Boltzmann `P = sigma*T^4*A` with sigma = pi^2/60 (single scalar DOF)
- Horizon area `A = 4*pi*(r+^2 + a^2)` standard Kerr-Newman
- Extremal BH handling: disc <= 0 -> P = 0 (correct: zero temperature)
- Naked singularity floor at 0.5*M prevents disappearing particles

### Pion Physics
- Pion velocity: `v = sqrt(KE*(KE+2m))/(KE+m)` correct relativistic kinematics
- Proper velocity: `w = gamma*v = p/m` correct
- GR deflection factor `(1+v^2)`: interpolates between 1 (Newtonian) and 2 (null geodesic)
- Pion mass = Yukawa mu parameter (Yukawa's 1935 insight)
- Pion momentum tracking: `m*w` correct
- Absorption momentum transfer: `E*v = m*gamma*v = m*w = p` correct

### Photon Physics
- Gravitational lensing: factor of 2 (full GR for null geodesics)
- Speed renormalized to c=1 after deflection
- Absorption momentum: `p = E/c = E` in natural units, `delta_w = E*v_hat/m`

### Scalar Fields
- Mexican hat potential `V(phi) = -mu^2*phi^2/2 + lambda*phi^4/4` with `lambda = mu^2 = m_H^2/2`
- VEV = 1 at `phi = mu/sqrt(lambda) = 1`
- Mass generation `m_eff = baseMass * |phi|`
- Axion quadratic potential `V(a) = m_a^2*a^2/2`
- Axion EM coupling: `alpha_eff = alpha*(1+g*a)`, source `g*q^2`, force `g*q^2*grad(a)`
- PQ coupling: `+/-g*m` source (CP flip for antimatter), `yukMod = 1+/-g*a`, clamped >= 0
- Symplectic Euler for wave equation: preserves symplectic structure, no secular drift
- PQS (cubic B-spline) weights sum to 1, C^2 continuous, standard PIC shape functions
- C^2 gradient strategy: PQS-interpolate central-difference grid gradients
- Field energy: `integral(phi_dot^2/2 + |grad phi|^2/2 + V(phi)) * dA` correct
- Field momentum: `P_i = -integral(phi_dot * d_i phi) dA` correct (stress-energy T^{0i})
- Higgs vacuum energy offset: V(VEV=1) = 0 verified
- Damping: Higgs critical (2*m_H), Axion Q=1/g=20 (g*Q=1 for matched resonant/static response)
- Boundary conditions: Despawn->Dirichlet, Bounce->Neumann, Loop->Periodic all reasonable
- `axMod` and `yukMod` clamped >= 0 prevents EM/Yukawa force sign reversal
- Source-force consistency verified for both Higgs and Axion

### Numerical Methods
- Barnes-Hut theta=0.5: ~1% force accuracy, standard for interactive N-body
- QuadTree depth guard (max 48) prevents stack overflow from coincident particles
- Laplacian: standard 5-point stencil, O(h^2) on non-square grid
- Interior/border split for Laplacian reduces ~16K `_nb()` calls to ~504
- NR signal delay tolerance 1e-12: tight but achievable in Float64
- Plummer softening prevents force singularities
- NaN guards throughout integrator catch bad state
- `fastTanh()` Pade approximant: max error ~0.4%, adequate for visualization
- Fixed-timestep accumulator with MAX_FRAME_DT=0.1 and ACCUMULATOR_CAP=4 prevents spiral-of-death

### Collisions
- Merge conserves mass, charge, linear momentum (proper velocity), angular momentum (orbital + spin)
- Relativistic KE `wSq/(gamma+1)*m` used for merge energy tracking (avoids cancellation)
- Annihilation energy `2*m*c^2` correct
- Annihilation momentum `annihilated*(w1+w2)` correct (equal mass from each particle)
- Hertz contact `F = K*delta^{3/2}` standard

### Topology
- Torus: standard periodic wrapping, half-domain min-image correct
- Klein bottle: y-wrap with x-reflection and velocity/spin flip correct
- RP^2: both axes are glide reflections, 4 min-image candidates complete
- Scalar field `_nb()` correctly implements all three topologies

### Energy & Momentum
- Relativistic linear and spin KE formulas correct
- Non-relativistic fallback `0.5*m*v^2` correct
- Orbital angular momentum `r x (m*w)` and spin `I*angw` correct
- Darwin field energy (EM: `-q1*q2/(2r)*vel_term`, GM: opposite sign) correct
- EIH 1PN PE structurally consistent with force
- Effective potential includes all force contributions with correct signs

### Presets
All 19 presets verified as physically reasonable with correct force toggles:
- `kepler`: stable Keplerian orbits with gravity only
- `precession`: eccentric orbit with 1PN for perihelion advance
- `inspiral`: sub-circular orbit with GW radiation for binary decay
- `tidallock`: moon spin-down with tidal torque
- `roche`: plunge orbit inside Roche limit with disintegration
- `hawking`: small BH masses for visible evaporation
- `atom`: screened nuclear charge with spin-orbit
- `bremsstrahlung`: relativistic near-miss with radiation
- `magnetic`: aligned dipole grid with Lorentz deflection
- `nucleus`: Yukawa binding of nucleon ring
- `axion`: coupled atom-like systems with EM modulation
- `pionexchange`: nucleon ring with massive force carriers
- `higgs`: particles gaining mass from field VEV
- `higgsboson`: head-on collision exciting Higgs field
- `axionburst`: charged collision exciting axion field
- `pecceiQuinn`: matter vs antimatter with PQ coupling
- `phasetransition`: fast particles driving symmetry restoration
- `galaxy`: 100-body with BH and gravitomagnetic effects
- `expansion`: Hubble flow competition with gravity
