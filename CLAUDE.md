# CLAUDE.md

Part of the **a9l.im** portfolio. See parent `site-meta/CLAUDE.md` for shared design system, head loading order, and CSS conventions. Sibling projects: `biosim`, `gerry`.

## Style Rule

Never use the phrase "retarded potential(s)" in code, comments, or user-facing text. Use "signal delay" or "finite-speed force propagation" instead.

## Running Locally

```bash
cd path/to/a9lim.github.io && python -m http.server
```

Serve from `a9lim.github.io/` -- shared files load via absolute paths. ES6 modules require HTTP. No build step, test framework, or linter.

## File Map

```
main.js                  376 lines  Simulation class, emitPhotonBurst(), fixed-timestep loop, save/load, pair production, window.sim
index.html               508 lines  UI: 4-tab sidebar, reference overlay, zoom controls, field sliders, antimatter button
styles.css               235 lines  Project-specific CSS overrides
colors.js                 18 lines  Project color tokens (particle hues, spin ring colors)
src/
  integrator.js         1222 lines  Physics class: Boris substep loop, radiation, tidal, GW quadrupole, expansion, Roche, external fields, Hertz bounce, scalar fields
  ui.js                  526 lines  setupUI(), declarative dependency graph, info tips, reference overlay, keyboard shortcuts
  renderer.js            503 lines  Canvas 2D: particles, trails, spin rings, ergosphere, antimatter rings, vectors, torque arcs, photons, delay ghosts, field overlays
  forces.js              450 lines  pairForce(), computeAllForces(), calculateForce() (BH walk), compute1PNPairwise(), Yukawa
  presets.js             585 lines  PRESETS (15 scenarios, 4 groups), loadPreset(), SLIDER_MAP, TOGGLE_MAP/TOGGLE_ORDER
  reference.js           638 lines  REFERENCE object: physics reference content (KaTeX math)
  scalar-field.js        239 lines  ScalarField base class: PQS grid, topology-aware deposition, Laplacian, interpolation, gradient
  higgs-field.js         235 lines  HiggsField extends ScalarField: Mexican hat potential, thermal phase transitions, mass modulation
  axion-field.js         214 lines  AxionField extends ScalarField: quadratic potential, scalar aF^2 coupling, EM modulation
  quadtree.js            279 lines  QuadTreePool: SoA flat typed arrays, pool-based, zero GC
  input.js               262 lines  InputHandler: mouse/touch, Place/Shoot/Orbit modes, hover tooltip
  signal-delay.js        249 lines  getDelayedState() (3-phase light-cone solver)
  heatmap.js             224 lines  Heatmap: 64x64 potential field overlay, mode selector, signal-delayed positions
  effective-potential.js 203 lines  EffectivePotentialPlot: V_eff(r) sidebar canvas, auto-scaling
  save-load.js           204 lines  saveState(), loadState(), downloadState(), uploadState(), quickSave/Load(), baseMass persistence
  potential.js           152 lines  computePE(), treePE(), pairPE() (7 PE terms)
  energy.js              153 lines  computeEnergies(): KE, spin KE, momentum, angular momentum, Darwin, field energies
  stats-display.js       131 lines  StatsDisplay: energy/momentum/drift DOM updates (x100 display scale)
  config.js              126 lines  Named constants, spawnOffset(), kerrNewmanRadius() helpers
  particle.js            122 lines  Particle: pos, vel, w, angw, baseMass, antimatter, cached magMoment/angMomentum, 11 force Vec2s, axMod, history
  phase-plot.js          117 lines  PhasePlot: r vs v_r sidebar canvas (512-sample ring buffer)
  collisions.js          116 lines  handleCollisions(), resolveMerge(), antimatter annihilation, baseMass conservation
  topology.js            112 lines  TORUS/KLEIN/RP2 constants, minImage(), wrapPosition()
  vec2.js                 61 lines  Vec2 class: set, clone, add, sub, scale, mag, magSq, normalize, dist, static sub
  photon.js               88 lines  Photon: pos, vel, energy, lifetime, type ('em'/'grav'), gravitational lensing (BH tree walk)
  relativity.js           22 lines  angwToAngVel(), setVelocity()
```

## Key Imports

```
main.js <- Physics (integrator), Renderer, InputHandler, Particle, HiggsField, AxionField,
           Heatmap, PhasePlot, EffectivePotentialPlot, StatsDisplay, setupUI, config, Photon, save-load

integrator.js <- QuadTreePool, config, Photon, angwToAngVel, forces (resetForces/computeAllForces/compute1PNPairwise),
                 handleCollisions, computePE, topology (accesses sim.higgsField/axionField via this.sim backref)

forces.js     <- config, getDelayedState, topology
energy.js     <- config, topology (accesses sim.higgsField/axionField via window.sim)
potential.js  <- config, topology
renderer.js   <- config (higgsField/axionField set by main.js)
heatmap.js    <- config, getDelayedState
scalar-field.js <- config, topology
higgs-field.js  <- config, ScalarField + bcFromString
axion-field.js  <- config, ScalarField + bcFromString
```

## Physics Engine

### Natural Units & State Variables

c = 1, G = 1, h-bar = 1 throughout. All velocities are fractions of c.

Both linear and rotational state use proper-velocity (celerity):

| State | Derived | Formula | Cap |
|---|---|---|---|
| `p.w` (gamma\*v) | `p.vel` | v = w / sqrt(1 + w^2) | \|v\| < c |
| `p.angw` | `p.angVel` | omega = W / sqrt(1 + W^2 r^2) | surface vel < c |

When relativity is off: `vel = w`, `angVel = angw` (identity).

Key derived quantities (INERTIA_K = 0.4, MAG_MOMENT_K = 0.2):
- Moment of inertia: `I = 0.4 * m * r^2`
- Magnetic moment: `mu = 0.2 * q * omega * r^2` -- cached as `p.magMoment`
- Angular momentum: `L = I * omega` -- cached as `p.angMomentum`
- Particle radius: `r = cbrt(mass)`; BH mode: `kerrNewmanRadius()` in config.js

`magMoment`/`angMomentum` cached per particle at start of `computeAllForces()`. Used by `pairForce()`, `pairPE()`, BH leaf walks, spin-orbit, display. Ghost particles carry these cached fields. Quadtree's `calculateMassDistribution()` computes inline from current state.

### Per-Particle Force Vectors

11 Vec2s reset each substep via `resetForces()`: `forceGravity`, `forceCoulomb`, `forceMagnetic`, `forceGravitomag`, `force1PN`, `forceSpinCurv`, `forceRadiation`, `forceYukawa`, `forceExternal`, `forceHiggs`, `forceAxion`.

3 torque scalars: `torqueSpinOrbit`, `torqueFrameDrag`, `torqueTidal`.

### Boris Integrator

Per substep (inside `Physics.update()` while loop):

1. Store `_f1pnOld` (if 1PN enabled)
2. **Half-kick**: `w += F/m * dt/2` (E-like forces)
3. **Boris rotation**: rotate w in combined Bz + Bgz + extBz plane (preserves |v| exactly)
4. **Half-kick**: `w += F/m * dt/2`
5. Spin-orbit energy coupling, Stern-Gerlach/Mathisson-Papapetrou kicks, frame-drag torque
6. Radiation reaction (Landau-Lifshitz)
7. **Drift**: `vel = w / sqrt(1 + w^2)`, `pos += vel * dt`
8. Cosmological expansion (if enabled)
9. **1PN velocity-Verlet correction**: recompute 1PN at new positions (always pairwise via `compute1PNPairwise()`), kick `w += (F_new - F_old) * dt / (2m)`
10. **Scalar fields**: evolve Higgs (symplectic Euler), modulate masses; evolve axion, interpolate axMod
11. Rebuild quadtree, handle collisions (with annihilation), repel contact forces, photon absorption
12. Apply external fields, Higgs/Axion gradient forces, sync axMod, reset forces + compute new forces

After all substeps: record signal-delay history (strided, once per HISTORY_STRIDE=64 `update()` calls), compute PE, reconstruct velocity-dependent display forces.

### Adaptive Substepping

- `dtSafe_accel = sqrt(softening / a_max)`
- `dtSafe_cyclotron = (2*pi / omega_c) / 8` where `omega_c = max(|q*Bz/m|, 4*|Bgz|, |q*extBz/m|)`
- Capped at MAX_SUBSTEPS = 32

### Fixed-Timestep Loop

`PHYSICS_DT = 1/128`. Accumulator collects `rawDt * speedScale`, drained in fixed chunks. Photon updates and tidal breakup inside the loop; energy/rendering/DOM outside.

## Force Types

### E-like Forces (radial, position-dependent)

Plummer softening: SOFTENING = 8 (SOFTENING_SQ = 64); BH mode: BH_SOFTENING = 4 (BH_SOFTENING_SQ = 16).

| Force | Formula | PE | Toggle |
|---|---|---|---|
| Gravity | `+m1*m2/r^2` (attractive) | `-m1*m2/r` | Gravity |
| Coulomb | `-q1*q2/r^2` (like-repels) | `+q1*q2/r` | Coulomb |
| Magnetic dipole | `-3*mu1*mu2/r^4` | `+mu1*mu2/r^3` | Coulomb + Magnetic |
| GM dipole | `+3*L1*L2/r^4` (co-rotating attract) | `-L1*L2/r^3` | Gravity + GM |

### B-like Forces (velocity-dependent, Boris rotation)

**Lorentz** (Coulomb + Magnetic): Bz from moving charge (`q_s*(v_s x r_hat)_z/r^2`) + spinning dipole (`+mu/r^3`). Display: `forceMagnetic += (q*vel.y*Bz, -q*vel.x*Bz)`.

**Gravitomagnetic** (Gravity + GM): Bgz from moving mass (`-m_s*(v_s x r_hat)_z/r^2`) + spinning mass (`-2L/r^3`). Boris parameter: `+2*Bgz*dt/gamma`. Display: `forceGravitomag += (4m*vel.y*Bgz, -4m*vel.x*Bgz)`.

**Frame-dragging torque**: `tau = 2*L_s*(omega_s - omega_p)/r^3`. Drives spin alignment.

### Tidal Locking

Requires Gravity. `coupling = m_other + q1*q2/m1`. `tau = -TIDAL_STRENGTH * coupling^2 * r_body^3 / r^6 * (omega_spin - omega_orbit)`.

### Yukawa Potential

Independent toggle. `F = -g^2 * m1*m2 * exp(-mu*r)/r^2 * (1+mu*r)`. Parameters: `yukawaG2` (default 1.0), `yukawaMu` (default 0.05, slider 0.01-0.25). Includes analytical jerk for radiation.

### External Background Fields

Uniform fields via `_applyExternalFields()`. No toggle -- controlled by slider values (default 0).

| Field | Effect | Integration |
|---|---|---|
| Gravity (`extGravity`, `extGravityAngle`) | `F = m*g` along angle (default pi/2 = down) | E-like, into `forceExternal` |
| Electric (`extElectric`, `extElectricAngle`) | `F = q*E` along angle (default 0 = right) | E-like, into `forceExternal` |
| Magnetic (`extBz`) | Uniform Bz | B-like (Boris rotation) |

Direction angle sliders auto-show when strength > 0. External Bz included in cyclotron frequency for adaptive substepping.

### Bounce (Hertz Contact)

Collision mode `'bounce'` and boundary mode `'bounce'`: `F = K * delta^1.5` (K=1 baked in). Tangential friction transfers torque. Particle-particle via `_applyRepulsion()` (quadtree when BH on, O(n^2) fallback when off). Boundary walls via `_applyBoundaryForces()`.

## Scalar Fields

### Base Class (`ScalarField`)

Shared PQS (cubic B-spline, order 3) grid infrastructure for Higgs and Axion. 4x4 = 16 node stencil per particle. C^2 interpolation, C^1 gradients. Pre-allocated weight arrays for zero-alloc hot path.

Key methods: `_nb()` (boundary-aware neighbor), `_depositPQS()` (topology-aware deposition), `_computeLaplacian()`, `interpolate()`, `gradient()`, `draw()`.

`bcFromString()` converts boundary mode string to integer (BC_DESPAWN=0 / BC_BOUNCE=1 / BC_LOOP=2).

Field arrays are `field`/`fieldDot` (not `phi`/`phiDot` or `a`/`aDot`). Grid size: SCALAR_GRID = 64. Field clamp: SCALAR_FIELD_MAX = 2.

### Higgs Field

Independent toggle. Mexican hat potential `V(phi) = -1/2 mu^2 phi^2 + 1/4 lambda phi^4`. VEV=1; free parameter is m_H (slider 0.01-0.25, default 0.05). With VEV=1: `lambda = mu^2 = m_H^2/2`.

- **Mass generation**: `m_eff = baseMass * |phi(x)|`. At VEV, m_eff = baseMass. Symmetric phase (phi->0): effectively massless (floored at EPSILON).
- **Gradient force**: `F = -g * baseMass * grad(phi)` where g = HIGGS_COUPLING = 0.05. Into `forceHiggs`.
- **Field equation**: `d^2 phi/dt^2 = laplacian(phi) + mu^2_eff * phi - mu^2 * phi^3 + source/cellArea - 2*m_H * d(phi)/dt`. Symplectic Euler. Source: `g * baseMass` via PQS.
- **Phase transitions**: `mu^2_eff = mu^2 - KE_local` (thermalK=1). When local KE > mu^2, field relaxes to phi=0.
- **Boundary**: Despawn -> Dirichlet (phi=1). Bounce -> Neumann. Loop -> periodic (topology-aware).
- **Energy**: `E = integral(1/2 phi_dot^2 + 1/2 |grad(phi)|^2 + V(phi)) dA`, shifted so V(1)=0 (vacOffset = mu^2/4).
- **Damping**: Critical damping `damp = 2*m_H`.
- **Rendering**: Magenta = depleted (phi < 1), cyan = enhanced (phi > 1). Alpha proportional to |deviation|.
- **baseMass sync**: All mass-modifying operations (merge, annihilation, Roche, disintegration, Hawking) proportionally scale baseMass. Toggle-off restores mass to baseMass.

### Axion Field

Requires Coulomb. Quadratic potential `V(a) = 1/2 m_a^2 a^2`. No symmetry breaking (vacuum at a=0). Uses **scalar** `aF^2` coupling (not pseudoscalar `aFF~` which vanishes in 2D).

- **Source**: Charged particles deposit `g * q^2` (g = AXION_COUPLING = 0.05). Neutral particles don't interact.
- **EM modulation**: `alpha_eff(x) = alpha * (1 + g*a(x))`. Per-particle `p.axMod` interpolated from local field. Clamped >= 0 to prevent EM force sign reversal. Used in `pairForce()` and `pairPE()`.
- **Gradient force**: `F = -g * q^2 * grad(a)`. Into `forceAxion`.
- **Field equation**: `d^2 a/dt^2 = laplacian(a) - m_a^2 * a - g*m_a * d(a)/dt + source/cellArea`. Damping: zeta = g/2, Q = 1/g, so g*Q = 1 (resonant buildup matches coupling strength).
- **Boundary**: Same as Higgs via `ScalarField._nb()`, but Dirichlet uses a=0 (not a=1).
- **Energy**: `E = integral(1/2 a_dot^2 + 1/2 |grad(a)|^2 + 1/2 m_a^2 a^2) dA`. No offset needed.
- **Parameters**: One slider: m_a (0.01-0.25, default 0.05).
- **Rendering**: Blue = positive (a > 0), red = negative (a < 0). Alpha proportional to |a|*4.

## Advanced Physics

### 1PN Corrections (EIH + Darwin EM + Bazanski)

Requires Relativity. Three O(v^2/c^2) sectors, all into `force1PN`:

- **EIH** (GM + 1PN): Remainder from EIH after subtracting GM Lorentz. Produces perihelion precession.
- **Darwin EM** (Magnetic + 1PN): Remainder from Darwin Lagrangian after subtracting Lorentz force.
- **Bazanski** (GM + Magnetic + 1PN): Mixed 1/r^3 force. Vanishes for identical particles.

NOT Newton's 3rd law. Velocity-Verlet: stores `_f1pnOld` before drift, recomputes after via `compute1PNPairwise()` (always pairwise, even when BH on).

### Radiation

Requires Gravity or Coulomb. Single toggle controls three mechanisms:

- **Larmor dipole** (requires Coulomb): Landau-Lifshitz force. Jerk is hybrid: analytical for gravity+Coulomb, numerical backward difference for residual. Power-dissipation terms only active with relativity on. Clamped: `|F_rad| <= 0.5 * |F_ext|`.
- **EM quadrupole** (requires Coulomb): `P = (1/180)|d^3 Q_ij/dt^3|^2`. Emits photons (type: 'em').
- **GW quadrupole** (requires Gravity): `P = (1/5)|d^3 I_ij/dt^3|^2`. Emits gravitons (type: 'grav', rendered red).

Both quadrupole types use TT-projected angular emission via rejection sampling. Photon absorption via quadtree query (self-absorption guard: age < 3).

### Black Hole Mode

Toggle under Relativity (requires Gravity). Locks collision to Merge.
- **Kerr-Newman horizon**: `r+ = M + sqrt(M^2 - a^2 - Q^2)` where `a = INERTIA_K*r^2*|omega|`, naked singularity floor at `M*BH_NAKED_FLOOR`
- **Ergosphere**: dashed ring at `r_ergo = M + sqrt(M^2 - a^2)` (visual only)
- **Reduced softening**: BH_SOFTENING_SQ = 16
- **Hawking radiation** (requires Radiation): `kappa = sqrt(disc)/(r+^2+a^2)`, `T = kappa/(2*pi)`, `P = sigma*T^4*A`. Extremal BHs stop radiating.
- **Evaporation**: below MIN_MASS -> removed with photon burst via `emitPhotonBurst()`

### Signal Delay

Auto-activates with Relativity. Three-phase solver on per-particle circular history buffers (Float64Array[256], recorded every HISTORY_STRIDE=64 `update()` calls):
1. Newton-Raphson segment search (up to 8 iterations)
2. Exact quadratic solve on converged segment
3. Constant-velocity extrapolation for t_ret before recorded history

BH mode: signal delay at leaf level only; distant aggregates use current positions.

### Spin-Orbit Coupling

Requires Magnetic + GM + Spin-Orbit toggle. Independent of Relativity. Stern-Gerlach `F = +mu*grad(Bz)`, Mathisson-Papapetrou `F = -L*grad(Bgz)` (GEM flip). Both into `forceSpinCurv`.

### Disintegration & Roche

Requires Gravity. Locks collision to Merge. Fragments when tidal + centrifugal + Coulomb stress exceeds self-gravity. Splits into SPAWN_COUNT (4) pieces. Roche overflow: Eggleton formula, continuous mass transfer through L1. Returns `{ fragments, transfers }`.

### Cosmological Expansion

Toggle. `pos += H*(pos - center)*dt` (Hubble flow), `w *= (1 - H*dt)` (redshift). Default H = 0.001. Locks boundary to "despawn".

### Antimatter & Pair Production

`p.antimatter` boolean. Toolbar button (key `A`). Matter+antimatter merge annihilates lesser mass, emits photons via `emitPhotonBurst()`. Pair production: photons with energy >= 2 near massive body (dist < 8) can produce matter+antimatter pair (prob 0.005/substep).

## Sign Conventions (IMPORTANT)

All GEM interactions are **attractive** (gravity has one sign of "charge"):
- GM dipole: `+3*L1*L2/r^4` (positive = attractive)
- GM Boris parameter: `+2*Bgz` (co-moving masses attract)
- Bgz field: `-m_s*(v_s x r_hat)_z/r^2` (negative sign in code: `p.Bgz -= ...`)
- Frame-drag torque: positive coefficient drives spins toward co-rotation

Do NOT flip these signs.

## Energy, PE & Collisions

**PE** (`potential.js`): Tree traversal via `treePE()` when BH on (divides by 2), exact pairwise `pairPE()` with i < j when off. Seven terms: gravitational, Coulomb (with axMod), magnetic dipole (with axMod), GM dipole, 1PN, Bazanski, Yukawa.

**Energy** (`energy.js`): Returns linearKE, spinKE, pe, fieldEnergy, momentum, angular momentum, COM, higgsFieldEnergy, axionFieldEnergy. Relativistic KE uses `wSq / (gamma + 1)`. Darwin field corrections when Magnetic/GM on but 1PN off. Conservation exact with gravity + Coulomb only, pairwise mode.

**Collisions**: Three modes -- pass (none), bounce (Hertz contact via `_applyRepulsion()`), merge (quadtree overlap detection, conserves mass/charge/momentum/angular momentum). `handleCollisions()` returns `annihilations` array for photon emission.

## Topology

When boundary = "loop": Torus (both axes normal), Klein (y-wrap mirrors x, negates w.x/angw), RP^2 (both axes glide reflections, 4 min-image candidates). `minImage()` zero-alloc via `out` parameter. `sim.topology` string -> `physics._topologyConst` integer (TORUS=0/KLEIN=1/RP2=2).

## Barnes-Hut

`QuadTreePool`: SoA flat typed arrays, pre-allocated 512 nodes (doubles via `_grow()`). Zero GC. BH_THETA = 0.5, QUADTREE_CAPACITY = 4. Off by default. Aggregates: totalMass, totalCharge, totalMagneticMoment, totalAngularMomentum, totalMomentumX/Y, comX/comY.

## Toggle Dependencies

```
Forces:                        Physics:
  Gravity                        Relativity          [signal delay auto-activates]
    -> Gravitomagnetic             -> 1PN             [requires Magnetic or GM]
    -> Tidal Locking               -> Black Hole      [+Gravity, locks collision to Merge]
  Coulomb                        Spin-Orbit           [requires Magnetic or GM]
    -> Magnetic                  Radiation             [requires Gravity or Coulomb]
    -> Axion                       Larmor + EM quad   [when Coulomb on]
  Yukawa               [independent]  GW quad         [when Gravity on]
  Higgs                [independent]
Disintegration                   [requires Gravity, locks collision to Merge]
Barnes-Hut                       [independent]
Expansion                        [independent, in Engine tab]
```

1PN internally: EIH requires `gravitomagEnabled`, Darwin EM requires `magneticEnabled`. Bazanski requires both.

Declarative `DEPS` array in `ui.js`, evaluated in topological order by `updateAllDeps()`. `setDepState()` applies `.ctrl-disabled` and auto-unchecks disabled toggles.

Defaults on: gravity, coulomb, magnetic, gravitomag, 1PN, relativity, spin-orbit, radiation, tidal locking. Defaults off: Yukawa, Axion, Higgs, Disintegration, Expansion, Barnes-Hut, Black Hole.

## UI

4-tab sidebar: Settings (mass/charge/spin, spawn mode, force/physics toggles), Engine (BH, collisions, boundary/topology, external fields, visuals, speed), Stats (energy/momentum/drift), Particle (selected details, force breakdown, phase plot, effective potential).

Topbar: Home | Brand "No-Hair" | Pause/Step/Reset/Save/Load | Antimatter | Theme | Panel toggle.

15 presets in 4 `<optgroup>` categories: Gravity (6), Electromagnetism (3), Exotic (4), Cosmological (2). First 9 via keyboard `1`-`9`. Sim speed range 1-128, default 64.

## Renderer

Canvas 2D. Dark mode: additive blending (`lighter`). WORLD_SCALE = 16 (domain = viewport / 16). Camera starts at zoom = WORLD_SCALE.

- **Particles**: r = cbrt(mass) (BH: Kerr-Newman r+), glow in dark mode. Neutral = slate. Charged: RGB lerp toward red(+)/blue(-), intensity = |q|/5.
- **Trails**: circular Float32Array[256], wrap-detection for periodic boundaries
- **Force vectors**: component colors: gravity=red, coulomb=blue, magnetic=cyan, GM=rose, 1PN=orange, spin-curv=purple, radiation=yellow, yukawa=green, external=white, higgs=magenta, axion=orange
- **Field overlays**: 64x64 offscreen canvas, bilinear-upscaled. Higgs: magenta/cyan. Axion: red/blue.
- **Photons**: yellow (EM) / red (gravitons), alpha fades over PHOTON_LIFETIME=256
- **Effective potential plot**: V_eff(r) sidebar canvas. 200-sample curve. Includes gravity, Coulomb, mag dipole, GM dipole, Yukawa.

## Key Patterns

- `window.sim` for console debugging. `_PALETTE`/`_FONT` frozen by colors.js
- `Vec2.set(x,y)` in hot paths; `pairForce()` accumulates into `out` Vec2, zero alloc
- Module-level `_miOut` objects for zero-alloc `minImage()` output
- Particle constructor declares all dynamic properties to prevent V8 hidden class transitions
- `_parseHex` from shared-tokens.js is script-scoped -- ES6 modules use inline `_hex` parser
- Icon swaps: toggle `hidden` attribute, not innerHTML
- Theme: `data-theme` on `<html>` (not body)
- World coordinates use `sim.domainW/H` (viewport / WORLD_SCALE), not pixel dimensions

## Gotchas

- Serve from `a9lim.github.io/` parent -- absolute paths for shared files
- `compute1PNPairwise()` zeroes `force1PN` before accumulating -- do not mix with `pairForce()` 1PN output in same step
- 1PN velocity-Verlet correction is always pairwise, even when BH is on
- Adaptive substepping uses Bz/Bgz from previous substep's force computation -- no preliminary force pass
- History recording is strided (HISTORY_STRIDE=64) after the substep loop, counting `update()` calls not substeps
- After merge collisions, `particles.length` changes -- update loop variable `n`
- GW quadrupole history buffer needs 4+ samples for 3rd derivative -- first frames produce no output
- Radiation power-dissipation terms only active when relativity on
- `forceRadiation` cleared for all particles before substep loop to prevent stale accumulation
- `.mode-toggles` in shared-base.css sets `display: grid` which overrides `hidden` attribute -- use `style.display` toggling
- All numerical thresholds (EPSILON, NR_TOLERANCE, etc.) are in config.js -- no inline magic numbers
- Bounce collision uses `_applyRepulsion()` which needs O(n^2) fallback when BH off (root < 0) -- do not early-return
- `handleCollisions()` only runs for merge mode; returns `annihilations` array -- integrator must emit photons
- Old save files with `collision: 'repel'` are migrated to `'bounce'` in loadState()
- External Bz enters Boris rotation alongside particle-sourced Bz -- included in `needBoris` condition check
- ScalarField arrays are `field`/`fieldDot` (not `phi`/`phiDot` or `a`/`aDot`)
- PQS stencil extends to `[ix-1..ix+2]`; `_fieldAt()` uses boundary clamping, `_depositPQS()` uses `_nb()` for topology wrapping
- Higgs `modulateMasses()` updates radius/radiusSq/invMass inline (not via `updateColor()`) to avoid per-substep string allocation
- `baseMass` must be saved/loaded and proportionally scaled wherever `mass` is modified
- Higgs field `energy()` shifts potential by +mu^2/4 so V(1)=0
- Higgs/Axion field reset on preset load and clear; Higgs mass -> baseMass on toggle-off; Axion axMod -> 1 on toggle-off
- Axion `p.axMod` is per-particle (interpolated from local field), not global -- used in `pairForce()`/`pairPE()`
- Axion `p.axMod` clamped >= 0 -- without this, EM force sign reversal causes runaway acceleration
- `magMoment`/`angMomentum` cache reflects previous `computeAllForces()` state -- consistent with B-field gradients used in same substep
- Ghost particles must carry `magMoment`/`angMomentum` fields (set in `_addGhost()`)
- Photon `update()` takes optional pool/root for BH tree lensing; falls back to O(N) when null or root < 0
