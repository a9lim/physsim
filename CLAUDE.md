# CLAUDE.md

Part of the **a9l.im** portfolio. See parent `site-meta/CLAUDE.md` for the shared design system. Sibling projects: `biosim`, `gerry`.

## Style Rule

Never use the phrase "retarded potential(s)" in code, comments, or user-facing text. Use "signal delay" or "finite-speed force propagation" instead.

## Running Locally

```bash
cd path/to/a9lim.github.io && python -m http.server
```

Serve from the parent `a9lim.github.io/` directory -- shared files (`/shared-base.css`, `/shared-tokens.js`, etc.) load via absolute paths. ES6 modules require HTTP (no `file://`). No build step, test framework, or linter.

## File Map

```
main.js                     ~290 lines Simulation class, fixed-timestep loop, save/load wiring, window.sim
index.html                  ~470 lines UI structure, 4-tab sidebar, reference overlay, zoom controls
styles.css                  ~480 lines Project-specific CSS overrides, reference overlay styles
colors.js                    18 lines  Project color tokens (particle hues, spin ring colors)
src/
  integrator.js             ~950 lines Physics class: adaptive Boris substep loop, radiation, tidal, GW quadrupole, expansion, Roche overflow
  forces.js                 ~350 lines pairForce(), computeAllForces(), calculateForce() (BH walk), compute1PNPairwise(), Yukawa force
  signal-delay.js           248 lines  getDelayedState() (3-phase light-cone solver)
  ui.js                     ~480 lines setupUI(), toggle dependencies, info tips, reference overlay, keyboard shortcuts
  renderer.js               ~430 lines Canvas 2D: particles, trails, spin rings, ergosphere, vectors, torque arcs, photons/gravitons, delay ghosts
  input.js                  260 lines  InputHandler: mouse/touch, Place/Shoot/Orbit modes, hover tooltip
  collisions.js             213 lines  handleCollisions(), resolveMerge(), resolveBounce() (rel + classical)
  quadtree.js               235 lines  QuadTreePool: SoA flat typed arrays, pool-based, zero GC
  potential.js              ~165 lines computePE(), treePE(), pairPE() (6 PE terms: grav, Coulomb, mag dipole, GM dipole, 1PN, Yukawa)
  topology.js               129 lines  TORUS/KLEIN/RP2 constants, minImage(), wrapPosition()
  energy.js                 139 lines  computeEnergies(): KE, spin KE, momentum, angular momentum, Darwin field
  phase-plot.js             116 lines  PhasePlot: r vs v_r sidebar canvas (500-sample ring buffer)
  stats-display.js          ~115 lines StatsDisplay: energy/momentum/drift DOM updates, selected particle info, force breakdown
  presets.js                ~370 lines PRESETS object (9 scenarios), loadPreset(), TOGGLE_MAP/TOGGLE_ORDER
  heatmap.js                ~160 lines Heatmap: 48x48 grav+electrostatic potential field overlay, signal-delayed positions, 6-frame interval
  particle.js               ~115 lines Particle entity: pos, vel, w, angw, per-type force vectors (incl. Yukawa), history buffers
  vec2.js                    65 lines  Vec2 class: set, clone, add, sub, scale, mag, magSq, normalize, dist
  config.js                 ~60 lines  Named constants (BH_THETA, SOFTENING, INERTIA_K, Yukawa, Axion, Roche, Hubble, GW, etc.)
  relativity.js              41 lines  angwToAngVel(), angVelToAngw(), setVelocity()
  photon.js                 ~40 lines  Photon entity: pos, vel, energy, lifetime, emitterId, type ('em'/'gw'), gravitational lensing
  save-load.js              ~120 lines saveState(), loadState(), downloadState(), uploadState(), quickSave(), quickLoad()
  reference.js              ~220 lines REFERENCE object: extended physics reference content for each concept (KaTeX math)
```

## Module Dependency Graph

```
main.js (Simulation class, window.sim)
  imports: Physics (integrator), Renderer, InputHandler, Particle, Heatmap, PhasePlot,
           StatsDisplay, setupUI, config constants, Photon, relativity helpers,
           quickSave + quickLoad + downloadState + uploadState (save-load)

  src/integrator.js (Physics class)
    imports: QuadTreePool + Rect, config constants (incl. Roche, Hubble), Photon, angwToAngVel,
             resetForces + computeAllForces + compute1PNPairwise (forces),
             handleCollisions (collisions), computePE (potential),
             TORUS + KLEIN + RP2 + minImage + wrapPosition (topology)

  src/forces.js
    imports: config (BH_THETA, SOFTENING_SQ, INERTIA_K, MAG_MOMENT_K, TIDAL_STRENGTH),
             getDelayedState (signal-delay), TORUS + minImage (topology)

  src/energy.js
    imports: config (INERTIA_K, SOFTENING_SQ), TORUS + minImage (topology)

  src/potential.js
    imports: config (BH_THETA, SOFTENING_SQ, INERTIA_K, MAG_MOMENT_K),
             TORUS + minImage (topology)

  src/stats-display.js
    imports: computeEnergies (energy)

  src/ui.js
    imports: loadPreset (presets), PHYSICS_DT + WORLD_SCALE (config), REFERENCE (reference)

  src/presets.js
    imports: WORLD_SCALE + SOFTENING_SQ (config)

  src/renderer.js
    imports: config (MAX_TRAIL_LENGTH, PHOTON_LIFETIME, INERTIA_K, HISTORY_SIZE)

  src/heatmap.js
    imports: config (SOFTENING_SQ, BH_THETA), getDelayedState (signal-delay)

  src/input.js
    imports: Vec2

  src/collisions.js
    imports: INERTIA_K (config), setVelocity + angwToAngVel + angVelToAngw (relativity),
             TORUS + minImage + wrapPosition (topology)

  src/signal-delay.js
    imports: HISTORY_SIZE (config), TORUS + minImage (topology)

  src/save-load.js
    imports: Particle (particle), angwToAngVel (relativity)

  src/reference.js
    (no imports - pure data module)
```

## HTML `<head>` Loading Order

Actual order from `index.html`:

1. Google Fonts `<link>` (Noto Sans, Noto Sans Mono, Noto Serif)
2. KaTeX CSS (`<link>` from CDN)
3. `<link rel="stylesheet" href="/shared-base.css">`
4. `<link rel="stylesheet" href="styles.css">`
5. `<script src="/shared-tokens.js">`
6. `<script src="/shared-touch.js">`
7. `<script src="/shared-utils.js">`
8. `<script src="/shared-camera.js">`
9. KaTeX JS (`<script defer>`)
10. KaTeX auto-render (`<script defer>`)
11. `<script src="/shared-info.js">`
12. `<script src="/shared-shortcuts.js">`
13. `<script src="colors.js">`
14. Inline `<script>` -- tab switching logic
15. `<script type="module" src="main.js">` (deferred by `type="module"`)

## Physics Engine

### Natural Units

c = 1, G = 1 throughout. All velocities are fractions of c. All forces are dimensionless.

### State Variables

Both linear and rotational state use the proper-velocity pattern:

| State variable | Derived | Formula | Cap |
|---|---|---|---|
| `p.w` (proper velocity, gamma*v) | `p.vel` | **v** = **w** / sqrt(1 + w^2) | \|v\| < c |
| `p.angw` (angular celerity) | `p.angVel` | omega = W / sqrt(1 + W^2 * r^2) | surface vel < c |

When relativity is off: `vel = w`, `angVel = angw` (identity).

Derived quantities from spin:
- Moment of inertia: `I = INERTIA_K * m * r^2` (INERTIA_K = 0.4 = 2/5, solid sphere)
- Magnetic moment: `mu = MAG_MOMENT_K * q * omega * r^2` (MAG_MOMENT_K = 0.2 = 1/5, uniform charge sphere)
- Angular momentum: `L = I * omega`
- Particle radius: `r = cbrt(mass)` (density rho = 3/(4*pi)); in Black Hole mode: Kerr-Newman `r+ = M + sqrt(M²-a²-Q²)`

### Per-Particle Force/Torque Display Vectors

Each particle stores per-type force vectors for component visualization: `forceGravity`, `forceCoulomb`, `forceMagnetic`, `forceGravitomag`, `force1PN`, `force1PNEM`, `forceSpinCurv`, `forceRadiation`, `forceYukawa`. All reset each substep via `resetForces()`. `forceSpinCurv` accumulates both Stern-Gerlach (+mu * grad(Bz)) and Mathisson-Papapetrou (-L * grad(Bgz)).

Torque display scalars: `torqueSpinOrbit` (EM + GM spin-orbit power) and `torqueFrameDrag`. Rendered as circular arc arrows around particles -- purple for spin-orbit, rose for frame-drag.

### Boris Integrator

Per substep (inside `Physics.update()` while loop):

1. Store `_f1pnOld` (if 1PN enabled)
2. **Half-kick**: `w += F/m * dt/2` (E-like forces only)
3. **Boris rotation**: rotate w in combined Bz + Bgz plane
   - `t = ((q/(2m)) * Bz + 2 * Bgz) * dt / gamma`
   - `s = 2t / (1 + t^2)`
   - `w' = w + (w + w x t) x s` (preserves |v| exactly)
4. **Half-kick**: `w += F/m * dt/2`
5. Spin-orbit energy coupling (EM + GM)
6. Stern-Gerlach / Mathisson-Papapetrou center-of-mass kicks
7. Frame-dragging torque
8. Radiation reaction (Landau-Lifshitz with 1/c² power terms)
9. **Drift**: derive `vel = w / sqrt(1 + w^2)`, `pos += vel * dt`
10. **Cosmological expansion** (if enabled): Hubble flow + peculiar velocity drag
11. Advance `simTime`
12. **1PN velocity-Verlet correction**: re-derive vel, recompute 1PN at new positions (always pairwise via `compute1PNPairwise()`), kick `w += (F_new - F_old) * dt / (2m)`
13. Rebuild quadtree
14. Handle collisions
14. Photon absorption
15. Save radiation display force, then reset forces + compute new forces for next substep

After all substeps: record signal-delay history (strided, once per HISTORY_STRIDE calls), compute PE, reconstruct velocity-dependent display forces from final-substep fields.

### Adaptive Substepping

Computed at the start of each substep from current forces and B fields:
- `dtSafe_accel = sqrt(SOFTENING / a_max)`
- `dtSafe_cyclotron = (2*pi / omega_c) / 8` where `omega_c = max(|q * Bz / m|, 4 * |Bgz|)`
- `dtSub = dtRemain / min(ceil(dtRemain / dtSafe), budgetLeft)`
- Capped at MAX_SUBSTEPS = 32 per frame

### Fixed-Timestep Loop (main.js)

`PHYSICS_DT = 1/128`. Accumulator collects `rawDt * speedScale` per animation frame. Drained in fixed-step chunks of PHYSICS_DT, capped at `MAX_SUBSTEPS * PHYSICS_DT * 4`. Photon updates and tidal breakup inside the loop; energy/rendering/DOM outside.

## Force Types

### E-like Forces (radial, position-dependent)

All use Plummer softening: `r_eff = sqrt(r^2 + SOFTENING_SQ)`, where SOFTENING = 8, SOFTENING_SQ = 64.

**Gravity**: `F = +m1*m2 / r^2` (attractive, toward source)
- PE: `U = -m1*m2 / r`

**Coulomb**: `F = -q1*q2 / r^2` (like-repels, opposite-attracts)
- PE: `U = +q1*q2 / r`

**Magnetic dipole** (requires Coulomb toggle): `F = -3*mu1*mu2 / r^4` (aligned perpendicular-to-plane dipoles repel)
- `mu = MAG_MOMENT_K * q * omega * r^2 = q*omega*r^2/5`
- PE: `U = +mu1*mu2 / r^3`

**GM dipole** (requires Gravity toggle): `F = +3*L1*L2 / r^4` (co-rotating masses attract; GEM sign flip)
- `L = INERTIA_K * m * omega * r^2 = 2*m*omega*r^2/5`
- PE: `U = -L1*L2 / r^3`

### B-like Forces (velocity-dependent, Boris rotation)

**Lorentz** (requires Coulomb + Magnetic toggles):
- Bz from moving charge: `q_s * (v_s x r_hat)_z / r^2` (Plummer-softened: `invR * invRSq`)
- Bz from spinning dipole: `+mu_source / r^3`
- Effect: `F = q(v x B)`, handled implicitly by Boris rotation
- Display-only: `forceMagnetic += (q*vel.y*Bz, -q*vel.x*Bz)` (computed after substep loop)

**Linear gravitomagnetic** (requires Gravity + GM toggles):
- Bgz from moving mass: `-m_s * (v_s x r_hat)_z / r^2`
- Bgz from spinning mass: `-2 * L_source / r^3`
- Boris parameter: `t_gm = +2 * Bgz * dt / gamma` (positive -> co-moving attract)
- Display-only: `forceGravitomag += (4*m*vel.y*Bgz, -4*m*vel.x*Bgz)`

**Frame-dragging torque**: `tau = 2 * L_s * (omega_s - omega_p) / r^3` (Lense-Thirring coefficient = 2)
- Uses source angular momentum `L_s = INERTIA_K * m * omega * r^2`, not mass * angVel
- Applied as `angw += tau * dt / I`; drives spin alignment

### Tidal Locking

Requires Gravity toggle. Computed in `pairForce()`, applied in integrator.

Drives spin toward synchronous rotation (ω_spin → ω_orbit) via dissipative tidal torque:
```
ω_orbit = (r × v_rel)_z / r²
Δω = ω_spin - ω_orbit
coupling = m_other + q₁q₂/m₁   (gravity + Coulomb when enabled)
τ = -TIDAL_STRENGTH * coupling² * r_body³ / r⁶ * Δω
```

The mixed coupling `(m_other + q₁q₂/m)²` captures all four cross-terms: the tidal field (gravity or Coulomb) raises a bulge, and the same or other field torques it. The `q₁q₂/m` term reflects that charge is tied to mass (uniform q/m) and the restoring force is self-gravity. Applied as `angw += tau * dt / I` from all neighbors.

### Yukawa Potential

Independent toggle. Screened force between massive particles:
```
V(r) = -g² · exp(-μr) / r
F = -g² · m₁m₂ · exp(-μr) / r² · (1 + μr)
```
Parameters: `yukawaG2` (coupling strength, default 1.0), `yukawaMu` (mediator mass = inverse range, default 0.2). At short range behaves like gravity; at long range vanishes exponentially. Includes analytical jerk for radiation reaction. PE term: `-g²·m₁m₂·exp(-μr)/r`.

### Axion Dark Matter Coupling

Requires Coulomb. Modulates the electromagnetic coupling constant:
```
α_eff = α · (1 + g · cos(m_a · t))
```
Parameters: `axionG` (coupling amplitude, default 0.1), `axionMass` (oscillation frequency, default 0.5). Applied as `axionModulation` multiplier on Coulomb force and PE. Energy is not conserved — the axion field is an external reservoir.

### 1PN Corrections (EIH + Darwin EM + Bazanski)

Requires Relativity. The 1PN toggle gates three O(v^2/c^2) correction sectors:

The velocity-dependent sectors (EIH, Darwin EM) follow the same pattern: subtract the Lorentz-like piece (handled by Boris when the corresponding B-force toggle is on, absent when off) from the full O(v^2/c^2) correction, feed only the symmetric remainder as an E-like force. NOT Newton's 3rd law — each particle's force uses its own velocity. Both forces computed independently in the pair loop.

**EIH (gravity 1PN)** — requires Gravity + 1PN. Symmetric remainder from EIH after subtracting the GM Lorentz piece:

```
radial  = -v1^2 - 2*v2^2 + 1.5*(n_hat.v2)^2 + 5*m1/r + 4*m2/r
a_1PN   = (m2/r^2) * [n_hat * radial + v1*(4*(n_hat.v1) - 3*(n_hat.v2)) + v2*3*(n_hat.v2)]
```

Produces perihelion precession approximately 6*pi*M / (a*(1-e^2)) rad/orbit.

**Darwin EM (electromagnetic 1PN)** — requires Coulomb + 1PN. Symmetric remainder from the Darwin Lagrangian after subtracting the Lorentz force:

```
F1_sym = (q1*q2)/(2*r^2) * { v1*(v2.n_hat) - 3*n_hat*(v1.n_hat)*(v2.n_hat) }
```

**Bazanski cross-term (gravity-EM 1PN)** — requires Gravity + Coulomb + GM + Magnetic + 1PN. Position-dependent mixed interaction from the Bazanski Lagrangian (no velocity cross-terms):

```
F_mixed = [q₁q₂(m₁+m₂) − (q₁²m₂ + q₂²m₁)] / r³   (along r̂)
```

Vanishes when q=0 (pure gravity) or m₁=m₂ with q₁=q₂ (identical particles). Accumulates into `force1PN`.

**Velocity-Verlet**: stores `_f1pnOld` and `_f1pnEMOld` before drift, recomputes after drift via `compute1PNPairwise()` (always pairwise, even in BH mode), applies correction kick `(F_new - F_old) * dt / (2m)` for EIH, Darwin, and Bazanski forces.

1PN PE (computed in `pairPE()`):
```
U_1PN_grav  = -(m1*m2/r) * [1.5*(v1^2+v2^2) - 3.5*(v1.v2) - 0.5*(v1.n)(v2.n) + m1/r + m2/r]
U_1PN_em    = -(q1*q2)/(2*r) * [(v1.v2) + (v1.n_hat)(v2.n_hat)]
U_1PN_mixed = [q₁q₂(m₁+m₂) − (q₁²m₂ + q₂²m₁)] / (2r²)
```

### Radiation

Independent toggle (no longer requires Relativity).

**Larmor power**: `P = 2*q^2*a^2/3`

**Landau-Lifshitz force** (full 1/c² terms):
```
F_rad = tau * [dF/dt / gamma^3 - v*F^2/(m*gamma^2) + F*(v.F)/(m*gamma^4)]
tau = 2*q^2/(3*m)
```
Term 1 (jerk) is hybrid: analytical `dF/dt = k·[v_rel/r³ − 3·r·(r·v_rel)/r⁵]` for gravity + Coulomb (accumulated into `p.jerk` in `pairForce()`), plus O(dt²) 3-point backward difference with variable step sizes for residual forces (magnetic dipole, GM dipole, 1PN, spin-curvature). Falls back to 2-point when < 2 samples stored. Terms 2-3 (power dissipation) only active when relativity is on.
Clamped: `|F_rad| <= LL_FORCE_CLAMP * |F_ext|` (LL_FORCE_CLAMP = 0.5) to enforce LL perturbative validity.

**Photon emission**: Energy accumulated in `_radAccum` per particle. Emits when >= MIN_MASS and pool < MAX_PHOTONS (1024). Emission angle sampled from sin^2(theta) dipole pattern with relativistic aberration (beamed toward velocity at high gamma). Photon travels at c = 1.

**Photon absorption**: Quadtree query at photon position (radius SOFTENING). Self-absorption guard: emitter skipped for first 2 substeps (age < 3). On absorb: `target.w += ph.energy * ph.vel / target.mass`. Bookkeeping: `totalRadiated` decremented, radiated momentum decremented.

### Black Hole Mode

Toggle under Relativity (`physics.blackHoleEnabled`). When on:
- **Kerr-Newman horizon**: `r+ = M + sqrt(M²-a²-Q²)` where `a = J/M` (spin parameter) and `Q` is charge. Replaces `cbrt(M)` radius. Naked singularity floor at `M*0.5` when `disc < 0`. Set in `particle.updateColor()`.
- **Ergosphere**: rendered as dashed purple ring at `r_ergo = M + sqrt(M²-a²)` when visible.
- **Reduced softening**: Plummer softening reduced from 64 to 1 in BH mode (both forces and PE).
- **Collision lock**: collision mode forced to Merge, UI disabled.
- **Hawking radiation**: Kerr-Newman surface gravity temperature. `κ = (r+ - r-)/(2(r+² + a²))`, `T = κ/(2π)`, `P = σT⁴A` where `A = 4π(r+² + a²)`. Extremal BHs (M² = a² + Q²) have zero temperature and stop radiating. Mass decremented each substep. Photon emission uses accumulator pattern (`_hawkAccum`, threshold MIN_MASS). Isotropic emission.
- **Evaporation**: particles below `MIN_MASS` are removed with a final photon burst (up to 5 photons carrying remaining mass-energy). Handled in main.js loop after tidal breakup.

### Signal Delay

Requires Relativity. Works in both pairwise and Barnes-Hut modes.

Light-cone equation: `|x_source(t_ret) - x_obs(now)| = now - t_ret` (c = 1).

Three-phase solver on per-particle circular history buffers (`Float64Array[HISTORY_SIZE=256]` each for x, y, vx, vy, time; recorded every `HISTORY_STRIDE=64` `update()` calls):

1. **Newton-Raphson** (up to NR_MAX_ITER=6 iterations) on `g(t) = |x_s(t) - x_obs| - (now - t)` to locate the correct history segment. Uses proportional segment estimate + short walk for initial segment. Guaranteed convergent for subluminal sources (`g' = d_hat . v_eff + 1 > 0`).
2. **Exact quadratic solve** on the converged segment +/- 1 neighbor. Piecewise-linear trajectory makes the light-cone equation a quadratic: `(v^2 - 1)*s^2 + 2*(d.v + T)*s + (r^2 - T^2) = 0` with closed-form roots.
3. **Constant-velocity extrapolation** from the oldest buffer entry when t_ret predates recorded history. Same quadratic with s <= 0.

Early rejection: pairs with current distance > 2 * buffer time span skip straight to extrapolation (O(1)).

**Barnes-Hut hybrid**: In BH mode, signal delay is applied at leaf level (individual particle interactions within nearby nodes). Distant aggregate nodes use current-time center-of-mass data — the retarded correction is negligible at distances where the BH approximation (size/d < theta) kicks in.

Returns a pre-allocated `_delayedOut` object (caller must consume before next call).

### Spin-Orbit Coupling

Requires relevant force toggle (Magnetic for EM, GM for gravitational) + Spin-Orbit toggle. Independent of Relativity.

**Energy transfer**:
- EM: `dE = -mu * (v . grad(Bz)) * dt`
- GM: `dE = -L * (v . grad(Bgz)) * dt`
- Applied as `angw += dE / (I * omega)`, then re-derive angVel

**Center-of-mass kicks** (spin-curvature forces):
- Stern-Gerlach (EM): `F = +mu * grad(Bz)`
- Mathisson-Papapetrou (GM): `F = -L * grad(Bgz)` (GEM sign flip)
- Both accumulate into `p.forceSpinCurv`

**Field gradients** computed in `pairForce()` (both radial + angular terms):
- `dBz/dpx = +3*Bz*rx/r^2 + q_s*vy_s/r^3`, dipole: `+3*mu*rx/r^5`
- `dBz/dpy = +3*Bz*ry/r^2 - q_s*vx_s/r^3`, dipole: `+3*mu*ry/r^5`
- `dBgz/dpx = +3*Bgz*rx/r^2 - m_s*vy_s/r^3`, dipole: `-6*L*rx/r^5`
- `dBgz/dpy = +3*Bgz*ry/r^2 + m_s*vx_s/r^3`, dipole: `-6*L*ry/r^5`

### Tidal Breakup

Independent toggle (`tidalEnabled`). Fragments when any combination exceeds self-gravity:
```
tidal:       TIDAL_STRENGTH * M_other * r_body / r_sep^3     (TIDAL_STRENGTH = 2.0)
centrifugal: omega^2 * r
coulomb:     q^2 / (4*r^2)
self-grav:   m / r^2
```
Splits into FRAGMENT_COUNT (4) pieces at 90-degree intervals, radius * 1.5 from original center. Each gets mass/4, charge/4, tangential velocity from spin. Min mass to fragment: `MIN_MASS * FRAGMENT_COUNT`.

**Roche Lobe Overflow**: When a particle fills its Roche lobe (Eggleton: `r_Roche ≈ 0.462·d·(m/(m+M))^(1/3)`) but is not violently disrupted, it continuously transfers mass toward the companion through the L1 point. Transfer rate: `dM = overflow * ROCHE_TRANSFER_RATE * m`, capped at 10% of mass per check. Spawns a stream particle with tangential velocity from orbital motion. Returns `{ fragments, transfers }` from `checkTidalBreakup()`.

### Gravitational Wave Radiation

Toggle (`gwRadiationEnabled`), requires Gravity. Computes mass quadrupole moment tensor and its 3rd time derivative via finite differences on a 5-sample history buffer:
```
I_ij = Σ m·x_i·x_j
P_GW = (1/5)·|d³I_ij/dt³|²
```
Also computes EM quadrupole when Radiation + Coulomb are enabled: `P_EM = (1/180)·|d³Q_ij/dt³|²`.

Orbital decay: radial kick toward COM proportional to `m·r` (quadrupole weighting). Emits graviton particles (`type: 'gw'`, rendered green) from system COM when accumulated energy exceeds MIN_MASS.

### Photon Gravitational Lensing

Photons experience 2× Newtonian gravitational deflection (null geodesic): `dv = 2·M·r̂/r²·dt`. Velocity renormalized to c=1 after each step. Uses PHOTON_SOFTENING_SQ=4 (smaller than particle softening for tighter lensing).

### Cosmological Expansion

Toggle (`expansionEnabled`). Applies Hubble flow from domain center after each drift step:
```
pos += H·(pos - center)·dt     (Hubble flow)
w *= (1 - H·dt)                 (peculiar velocity redshift)
```
Parameter: `hubbleParam` (default 0.001). Bound systems resist expansion; unbound particles drift apart.

## Sign Conventions (IMPORTANT)

All GEM interactions are **attractive** (gravity has one sign of "charge"):
- GM dipole: `+3*L1*L2/r^4` (positive = attractive)
- GM Boris parameter: `+2*Bgz` (co-moving masses attract)
- Bgz field: `-m_s*(v_s x r_hat)_z/r^2` (negative sign in code: `p.Bgz -= ...`)
- Frame-drag torque: positive coefficient drives spins toward co-rotation

Do NOT flip these signs.

## Potential Energy

Computed separately from forces via `computePE()` in `potential.js`. Same BH theta criterion -- tree traversal via `treePE()` when BH on (divides by 2 to avoid double-counting), exact pairwise `pairPE()` with i < j when off. Seven terms: gravitational (-m1*m2/r), Coulomb (+q1*q2/r with axion modulation), magnetic dipole (+mu1*mu2/r^3), GM dipole (-L1*L2/r^3), 1PN PE (velocity-dependent EIH + Darwin EM), Bazanski cross-term PE (position-dependent mixed gravity-EM), Yukawa (-g²·m1·m2·exp(-μr)/r). All Plummer-softened (reduced to 1 in BH mode).

## Energy & Momentum (`src/energy.js`)

`computeEnergies()` returns: `linearKE`, `spinKE`, `pe`, `fieldEnergy`, `fieldPx/Py`, `px/py`, `orbitalAngMom`, `spinAngMom`, `comX/comY`.

| Quantity | Relativistic | Classical |
|---|---|---|
| Linear KE | sum(wSq / (gamma + 1) * m) | sum(0.5 * m * \|v\|^2) |
| Spin KE | sum(INERTIA_K * m * srSq / (gammaRot + 1)) | sum(0.5 * I * omega^2) |
| Momentum | sum(m * w) | sum(m * v) |
| Angular mom. | sum(r x m*w) + sum(I * W) about COM | same |

**Darwin field corrections** (O(v^2/c^2), computed when Magnetic or GM enabled but 1PN is off):
- EM field energy: `-0.5 * sum_{i<j}(qi*qj/r) * [(vi.vj) + (vi.r_hat)(vj.r_hat)]`
- GM field energy: `+0.5 * sum_{i<j}(mi*mj/r) * [(vi.vj) + (vi.r_hat)(vj.r_hat)]` (opposite sign)
- Bazanski cross-term: `+0.5 * [q₁q₂(m₁+m₂) − (q₁²m₂ + q₂²m₁)] / r²` (position-dependent, no field momentum)
- Field momentum: analogous terms with `(vi + vj)` and `(vi + vj).r_hat` (EM + GM only; Bazanski has no velocity terms)

When 1PN is on, field energy terms are dropped (they are absorbed into the 1PN PE correction in `pairPE()`).

Conservation: exact with gravity + Coulomb only, pairwise mode (BH off). Velocity-dependent forces break Newton's 3rd law -- missing momentum carried by unmodeled fields.

## Collisions (`src/collisions.js`)

Detection uses quadtree query (radius `p1.radius * 2`). Ghost particles resolve against their `original`. ID comparison (`p1.id >= real2.id`) prevents double-processing.

**Pass**: no-op.

**Merge**: conserves mass, charge, linear momentum (m * w), angular momentum. Uses minimum-image offset for periodic boundaries. Orbital L about pair COM + spin L -> merged angw via `I = INERTIA_K * m * r^2`.

**Bounce**: two paths depending on `relativityEnabled`:
- *Relativistic*: decompose proper velocities into normal/tangential. Lorentz-boost normal components to COM frame (invariant mass M = sqrt(E^2 - P_n^2)), reverse normal w, boost back. Tangential friction: `J = bounceFriction * (surfaceV1 - surfaceV2) * m_eff` where `surfaceV = v_tangential + omega*r`. Spin updated: `omega_new = omega_old - J/I`, converted back to angw.
- *Classical*: standard elastic formulas `(v1n*(m1-m2) + 2*m2*v2n) / (m1+m2)` with same friction model.

Configurable friction (default 0.4). Overlap separation applied after resolution.

## Topology (`src/topology.js`)

When boundary = "loop", topology selector chooses identification:

| Topology | Wrapping | min-image candidates |
|---|---|---|
| Torus (T^2) | Both axes normal wrap | 1 |
| Klein (K) | x normal; y-wrap mirrors x, negates w.x/vel.x/angw/angVel | 2 |
| RP^2 | x-wrap mirrors y (negates w.y/vel.y/angw/angVel); y-wrap mirrors x | 4 |

`minImage(ox, oy, sx, sy, topology, W, H, halfW, halfH, out)`: minimum-image separation. Klein/RP^2 need absolute positions (glide reflections depend on source coords). Zero-alloc via `out` parameter. Uses `torusWrap()` helper for axis wrapping.

`wrapPosition(p, topology, W, H)`: wraps position into [0,W]x[0,H] and applies velocity/spin flips for non-orientable crossings.

**Ghost generation** (`_generateGhosts` in integrator.js): topology-aware. `_addGhost()` accepts `flipVx`/`flipVy` flags that negate velocity components and spin for non-orientable topologies. Ghost margin = `max(W, H) * BH_THETA`.

`sim.topology` string ('torus'/'klein'/'rp2') -> `physics._topologyConst` integer (TORUS=0 / KLEIN=1 / RP2=2).

## Barnes-Hut (`src/quadtree.js`)

`QuadTreePool`: SoA flat typed arrays (Float64Array/Int32Array/Uint8Array), pre-allocated 512 nodes (doubles on overflow via `_grow()`). `pool.reset()` + `pool.build()` per substep, zero GC pressure.

Node capacity: QUADTREE_CAPACITY = 4 (max points per leaf before subdivision).

Aggregates per node: `totalMass`, `totalCharge`, `totalMagneticMoment`, `totalAngularMomentum`, `totalMomentumX/Y`, `comX/comY`.

BH_THETA = 0.5. Off by default -- exact pairwise gives better conservation.

Tree traversal in `calculateForce()`: for leaf nodes, iterates actual particles via `pairForce()`; for distant internal nodes, uses aggregate quantities. Ghost particles are skipped if their `original` is the test particle.

## Toggle Dependencies

```
Forces:                        Physics:
  Gravity                        Relativity
    -> Gravitomagnetic             -> Signal Delay     [requires Rel]
    -> GW Radiation                -> 1PN              [requires Rel + Magnetic + GM]
  Coulomb                          -> Black Hole       [requires Rel; locks collision to Merge]
    -> Magnetic                  Tidal                 [independent]
    -> Axion                     Spin-Orbit            [requires Magnetic + GM]
  Yukawa                   [independent]
                                 Radiation             [requires Coulomb]
Disintegration                   [independent]
Barnes-Hut                       [independent]
Expansion                        [independent, in Engine tab]
```

1PN internally: gravity EIH requires `gravityEnabled`, EM Darwin requires `coulombEnabled`. Both sectors activate only when their parent force is on.

Disabled sub-toggles: `.ctrl-disabled` class (opacity 0.4, pointer-events none) applied by `setDepState()` in `ui.js`. When a parent toggle is turned off, its children are automatically unchecked and their physics flags set to false.

Default on load: all force toggles on (gravity, coulomb, magnetic, gravitomag, 1PN, relativity, signal delay, spin-orbit, radiation) except Yukawa, Axion, GW Radiation, Tidal, Expansion, and Barnes-Hut which default to off.

## UI

### 4-Tab Sidebar

1. **Settings**: particle mass (0.05-5) / charge (-5 to 5) / spin (-0.99 to 0.99) sliders, interaction mode (Place/Shoot/Orbit), force toggles (Gravity -> Gravitomagnetic/GW Radiation, Coulomb -> Magnetic/Axion, Yukawa with sliders), physics toggles (Relativity -> Signal Delay/1PN/Black Hole, Tidal, Spin-Orbit, Radiation)
2. **Engine**: Barnes-Hut toggle, collision mode (Pass/Bounce/Merge), bounce friction slider (0-1, default 0.4), boundary mode (Despawn/Loop/Bounce), topology selector (Torus/Klein/RP^2, only visible when boundary=Loop), disintegration toggle, expansion toggle with Hubble slider, visual toggles (trails, velocity/force/component vectors, potential field, acceleration scaling), sim speed slider (0-200, default 128)
3. **Stats**: energy breakdown (total, linear KE, spin KE, PE, field, radiated, drift), conserved quantities (momentum with particle/field/radiated components, angular momentum with orbital/spin, drift percentages)
4. **Particle**: selected particle details (ID, mass, charge, spin, speed, gamma, |F|), per-force-type magnitude breakdown (gravity, Coulomb, magnetic, GM, 1PN, spin-curvature, radiation, Yukawa), phase space plot canvas (r vs v_r relative to most massive body)

Tab switching is handled by an inline `<script>` in `index.html` (not in a module).

### Topbar

Home link (logo) | Brand "No-Hair" | Pause/Step/Reset/Save/Load buttons | Theme toggle | Panel toggle

### Presets

| # | Name | Key | Description |
|---|---|---|---|
| 1 | Kepler Orbits | kepler | Star (m=5) + 5 planets in circular orbits |
| 2 | Precession | precession | Relativistic perihelion advance with eccentric orbit |
| 3 | Atom | atom | Coulomb binding with nucleus + 2 electrons |
| 4 | Bremsstrahlung | bremsstrahlung | Radiation from accelerating charges |
| 5 | Tidal Lock | tidallock | Tidal friction synchronizes spin to orbit |
| 6 | Hawking | hawking | Small black holes radiate and evaporate |
| 7 | Binary Pulsars | pulsars | Frame-dragging, gravitomagnetism, signal delay |
| 8 | Magnetic Dipoles | magnetic | 4x4 grid of charged spinning particles |
| 9 | Galaxy | galaxy | Core + 100 particles, BH mode, circular orbits |

### Keyboard Shortcuts

Registered via `initShortcuts()` from shared module:

| Key | Action | Group |
|---|---|---|
| Space | Pause / Play | Simulation |
| R | Reset simulation | Simulation |
| . | Step forward | Simulation |
| 1-9 | Load preset directly | Presets |
| Ctrl+S | Quick save | State |
| Ctrl+L | Quick load | State |
| Ctrl+Shift+S | Download state JSON | State |
| Ctrl+Shift+L | Upload state JSON | State |
| V | Toggle velocity vectors | View |
| F | Toggle force vectors | View |
| C | Toggle force components | View |
| T | Toggle theme | View |
| S | Toggle sidebar | View |
| Escape | Close dialogs | View |
| ? | Help overlay | (shared) |

### Info Tips

Data object `infoData` in `ui.js` maps keys to `{ title, body }` objects. HTML body strings contain KaTeX math (`$...$`). Triggers are `.info-trigger[data-info]` buttons in the HTML. Created via `createInfoTip()` from shared module. Shift+click on info buttons opens full reference pages (from `REFERENCE` in `reference.js`) in a modal overlay with extended derivations and equations.

### Responsive

900px -> bottom sheet + 48px toolbar. 600px/440px shared breakpoints from shared-base.css (`.hide-sm`, brand shrink, tighter spacing).

## Renderer

Canvas 2D. Dark mode uses additive blending (`globalCompositeOperation: 'lighter'`).

- **Particles**: filled circle at `r = cbrt(mass)` (BH mode: Kerr-Newman `r+`), glow shadow in dark mode (larger glow for charged particles)
- **Ergosphere**: dashed purple ring at `r_ergo = M + sqrt(M²-a²)` in BH mode when visible
- **Spin rings**: arc at radius+0.5, length proportional to |omega*r| (caps at 2*pi), arrow shows CW/CCW (h=1, spread=0.4, lineWidth=0.2), colored by spin sign (cyan=positive, orange=negative from `colors.js`)
- **Trails**: circular Float32Array buffer (MAX_TRAIL_LENGTH=256 points), 4 opacity groups, lineWidth = 0.5*radius, wrap-detection for periodic boundaries (skips segment if position jumps > half domain)
- **Force vectors**: scale=256 (divide by mass for acceleration). Total (accent color) sums all 9 component vectors (incl. Yukawa). Per-type component arrows colored by force type (Yukawa=brown)
- **Torque arcs**: spin-orbit (purple, offset 2), frame-drag (rose, offset 1.5), tidal (offset 1), total (accent, offset 2.5). Arc length proportional to |power|, scale=256/INERTIA_K
- **Photons**: yellow circles for EM, green for gravitons (`ph.type === 'gw'`). Size = `0.2 + energy*20` (cap at 5px), glow in dark mode, alpha fades over PHOTON_LIFETIME=256. Gravitational lensing deflects all photons.
- **Signal delay ghosts**: stroked circle outline at each particle's oldest history buffer position, drawn behind particles when signal delay is enabled. Reduced opacity (0.3 light / 0.4 dark)
- **Velocity vectors**: scale=40, muted text color
- **Heatmap**: 48x48 offscreen canvas, diverging colormap (blue=gravity well, red=repulsive), updates every 6 frames. Uses signal-delayed source positions when signal delay is enabled.

Particle color: neutral = `_PAL.neutral` (extended.slate `#8A7E72`). Charged: smooth RGB lerp from slate toward `extended.red` (positive) or `extended.blue` (negative), intensity = `|charge| / 5`. Uses inline hex parser (`_hex`) since `_parseHex` from shared-tokens.js is script-scoped and inaccessible from ES6 modules.

## Input (`src/input.js`)

- **Left click** (drag < 5 world units): select particle if hit (hitbox = radius), otherwise spawn at rest
- **Left drag**: spawn with velocity (Shoot: dragVector * 0.02) or at rest (Place/Orbit)
- **Right click**: remove particle within radius
- **Orbit mode**: finds particle with max gravitational force `(m/d^2)` on spawn point, spawns perpendicular at `v = sqrt(M/r)`, capped at 0.99c
- **Hover**: tooltip with m, q, spin (surface velocity), speed
- **Touch**: single=spawn, two-finger=pinch-zoom + pan via shared camera (300ms wasPinching guard prevents spawn after pinch)
- **Wheel zoom**: delegated to `camera.bindWheel(canvas)`

## World Scale

`WORLD_SCALE = 16`. The physics domain is `viewport / WORLD_SCALE` in each dimension (e.g. 120x67.5 at 1920x1080). The camera starts at zoom = WORLD_SCALE, centered on domain midpoint. `ZOOM_MIN = WORLD_SCALE` prevents zooming out beyond the domain. Zoom display is normalized: `zoom / WORLD_SCALE * 100` so default view shows "100%".

All world coordinates (particle positions, presets, camera resets) use `sim.domainW / sim.domainH`, not `sim.width / sim.height`. The renderer and input system use the shared camera's `screenToWorld`/`worldToScreen` transforms, which handle the zoom automatically.

## Key Patterns

- `Vec2` for all vector math. `vec.set(x,y)` in hot paths to avoid allocation.
- `pairForce()`: accumulates into `out` Vec2 parameter plus per-type display vectors, no allocations. Toggle flags via reusable `_toggles` object synced once per `update()`.
- QuadTreePool: SoA, pre-allocated, `reset()`+`build()` per substep. Zero GC.
- DOM cached in `Simulation.dom` (energy/momentum elements) and `Simulation.selDom` (selected particle elements). Shared by reference with StatsDisplay.
- `window.sim` for console debugging. `_PALETTE`/`_FONT` frozen by colors.js.
- Dark mode: `globalCompositeOperation: 'lighter'` (additive blending) for particles and trails.
- Icon swaps (pause/play, sun/moon): toggle `hidden` attribute, not innerHTML.
- Theme: `data-theme` on `<html>` (not body). Light default for FOUC prevention.
- Module-level `_miOut` objects in forces.js, energy.js, potential.js, signal-delay.js, collisions.js for zero-alloc minImage output.
- Signal delay returns pre-allocated `_delayedOut` -- caller must read before next call.
- Particle constructor declares all dynamic properties (`_radAccum`, `_hawkAccum`, `_radDisplayX`, `_radDisplayY`, `_frameDragTorque`, `_tidalTorque`, etc.) to prevent V8 hidden class transitions.
- InputHandler pre-allocates `_posOut` Vec2 for `getPos()` to avoid per-call allocation on mouse move.
- Relativistic KE uses `wSq / (gamma + 1)` instead of `gamma - 1` to avoid catastrophic cancellation at low velocities.
- Relativistic bounce guards invariant mass: `Math.sqrt(Math.max(0, E*E - Pn*Pn))` to prevent NaN from floating-point underflow.

## Gotchas

- Serve from `a9lim.github.io/` parent -- `/shared-base.css` and `/shared-tokens.js` use absolute paths
- `#preset-dialog` needs both ID and `class="preset-dialog"` (shared CSS uses class, JS uses ID)
- `photon.js` is imported by `integrator.js` for radiation -- not related to input modes
- 1PN velocity-Verlet correction is always pairwise (via `compute1PNPairwise()`), even when BH is on
- Radiation force: full LL with jerk + power-dissipation terms (−v·F²/mγ² and +F·(v·F)/mγ⁴); power terms only active when relativity is on
- Shoot mode velocity scale is 0.02 (drag pixels * 0.02 = velocity)
- Spin-orbit, Stern-Gerlach, and Mathisson-Papapetrou are all gated by the same `spinOrbitEnabled` toggle
- `compute1PNPairwise()` zeroes `force1PN` and `force1PNEM` before accumulating -- do not mix with `pairForce()` 1PN output in the same step
- Adaptive substepping uses Bz/Bgz values persisting from the previous substep's force computation for cyclotron frequency estimation -- no separate preliminary force pass
- History recording is strided (HISTORY_STRIDE=64) and happens after the substep loop, not inside each substep. The stride counts `update()` calls (one per PHYSICS_DT chunk), not substeps or frames
- Tab switching logic is in an inline `<script>` in index.html, not in ui.js or main.js
- `shared-touch.js` is loaded in the HTML head (between shared-tokens.js and shared-utils.js) but not documented in the parent CLAUDE.md loading order
- `_parseHex` from `shared-tokens.js` is script-scoped (`const`), not on `window` -- ES6 modules cannot access it. `particle.js` uses its own inline hex parser instead.
- After merge collisions, `particles.length` changes -- any loop variable `n` must be updated with `n = particles.length` after `handleCollisions()`
- World coordinates use `sim.domainW/H` (viewport / WORLD_SCALE), not `sim.width/height` (viewport pixels). Camera resets and presets must use domain coordinates.
- `checkTidalBreakup()` returns `{ fragments, transfers }` (not a flat array) — both must be destructured by caller
- Yukawa force uses `forceYukawa` display vector — must be included in total force sum in stats-display.js
- GW quadrupole history buffer (`_quadHistory`) needs 4+ samples for 3rd derivative — first few frames produce no GW output
- Photon.type distinguishes `'em'` vs `'gw'` — renderer colors differ (yellow vs green)
- `save-load.js` uses `showToast()` which is a global from shared-utils.js
- Heatmap signal delay is expensive (GRID²×N delay solves) — mitigated by 6-frame update interval
- Reference overlay uses `renderMathInElement` from KaTeX auto-render (loaded deferred)
