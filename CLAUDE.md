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
main.js                     ~310 lines Simulation class, fixed-timestep loop, save/load wiring, window.sim
index.html                  ~474 lines UI structure, 4-tab sidebar, reference overlay, zoom controls
styles.css                  ~500 lines Project-specific CSS overrides, reference overlay styles
colors.js                    18 lines  Project color tokens (particle hues, spin ring colors)
src/
  integrator.js             ~966 lines Physics class: adaptive Boris substep loop, radiation, tidal, GW quadrupole, expansion, Roche overflow
  ui.js                     ~440 lines setupUI(), declarative dependency graph, info tips, reference overlay, keyboard shortcuts
  renderer.js               ~490 lines Canvas 2D: particles, trails, spin rings, ergosphere, vectors, torque arcs, photons/gravitons, delay ghosts
  forces.js                 ~460 lines pairForce(), computeAllForces(), calculateForce() (BH walk), compute1PNPairwise(), Yukawa force
  presets.js                ~497 lines PRESETS object (13 scenarios in 4 groups), loadPreset(), declarative SLIDER_MAP, TOGGLE_MAP/TOGGLE_ORDER
  reference.js              ~285 lines REFERENCE object: extended physics reference content for each concept (KaTeX math)
  quadtree.js               ~280 lines QuadTreePool: SoA flat typed arrays, pool-based, zero GC
  input.js                   260 lines InputHandler: mouse/touch, Place/Shoot/Orbit modes, hover tooltip
  signal-delay.js            250 lines getDelayedState() (3-phase light-cone solver)
  collisions.js              210 lines handleCollisions(), resolveMerge(), resolveBounce() (rel + classical)
  save-load.js              ~210 lines saveState(), loadState(), downloadState(), uploadState(), quickSave(), quickLoad()
  heatmap.js                ~190 lines Heatmap: 48x48 grav+electrostatic potential field overlay, signal-delayed positions, 6-frame interval
  potential.js              ~160 lines computePE(), treePE(), pairPE() (7 PE terms: grav, Coulomb, mag dipole, GM dipole, 1PN, Bazanski, Yukawa)
  energy.js                  139 lines computeEnergies(): KE, spin KE, momentum, angular momentum, Darwin field
  stats-display.js          ~115 lines StatsDisplay: energy/momentum/drift DOM updates, selected particle info, force breakdown
  phase-plot.js              116 lines PhasePlot: r vs v_r sidebar canvas (500-sample ring buffer)
  particle.js               ~115 lines Particle entity: pos, vel, w, angw, per-type force vectors, history buffers
  topology.js                112 lines TORUS/KLEIN/RP2 constants, minImage(), wrapPosition()
  vec2.js                     65 lines Vec2 class: set, clone, add, sub, scale, mag, magSq, normalize, dist
  config.js                   60 lines Named constants (BH_THETA, SOFTENING, INERTIA_K, Yukawa, Axion, Roche, Hubble, GW, etc.)
  photon.js                   40 lines Photon entity: pos, vel, energy, lifetime, emitterId, type ('em'/'grav'), gravitational lensing
  relativity.js               33 lines angwToAngVel(), angVelToAngw(), setVelocity()
```

## Module Dependency Graph

```
main.js (Simulation, window.sim)
  <- Physics (integrator), Renderer, InputHandler, Particle, Heatmap, PhasePlot,
     StatsDisplay, setupUI, config, Photon, relativity helpers, save-load

integrator.js (Physics)
  <- QuadTreePool + Rect, config, Photon, angwToAngVel,
     resetForces + computeAllForces + compute1PNPairwise (forces),
     handleCollisions (collisions), computePE (potential),
     TORUS + KLEIN + RP2 + minImage + wrapPosition (topology)

forces.js        <- config, getDelayedState (signal-delay), TORUS + minImage (topology)
energy.js        <- config (INERTIA_K, SOFTENING_SQ), TORUS + minImage (topology)
potential.js     <- config, TORUS + minImage (topology)
stats-display.js <- computeEnergies (energy)
ui.js            <- loadPreset (presets), config (PHYSICS_DT, WORLD_SCALE), REFERENCE (reference)
presets.js       <- config (WORLD_SCALE, SOFTENING_SQ)
renderer.js      <- config (MAX_TRAIL_LENGTH, PHOTON_LIFETIME, INERTIA_K, HISTORY_SIZE)
heatmap.js       <- config (SOFTENING_SQ, BH_THETA), getDelayedState (signal-delay)
input.js         <- Vec2
collisions.js    <- INERTIA_K (config), relativity helpers, topology
signal-delay.js  <- HISTORY_SIZE (config), TORUS + minImage (topology)
save-load.js     <- Particle, angwToAngVel (relativity)
reference.js     (no imports - pure data)
```

## Physics Engine

### Natural Units

c = 1, G = 1 throughout. All velocities are fractions of c. All forces are dimensionless.

### State Variables

Both linear and rotational state use the proper-velocity (celerity) pattern:

| State variable | Derived | Formula | Cap |
|---|---|---|---|
| `p.w` (gamma*v) | `p.vel` | **v** = **w** / sqrt(1 + w²) | \|v\| < c |
| `p.angw` | `p.angVel` | omega = W / sqrt(1 + W²r²) | surface vel < c |

When relativity is off: `vel = w`, `angVel = angw` (identity).

Key derived quantities:
- Moment of inertia: `I = INERTIA_K * m * r²` (INERTIA_K = 0.4)
- Magnetic moment: `mu = MAG_MOMENT_K * q * omega * r²` (MAG_MOMENT_K = 0.2)
- Angular momentum: `L = I * omega`
- Particle radius: `r = cbrt(mass)`; in BH mode: Kerr-Newman `r+ = M + sqrt(M²-a²-Q²)`

### Per-Particle Display Vectors

Force vectors (8 Vec2s, reset each substep via `resetForces()`): `forceGravity`, `forceCoulomb`, `forceMagnetic`, `forceGravitomag`, `force1PN`, `forceSpinCurv`, `forceRadiation`, `forceYukawa`. `forceSpinCurv` accumulates both Stern-Gerlach and Mathisson-Papapetrou.

Torque scalars (3): `torqueSpinOrbit` (EM + GM spin-orbit power), `torqueFrameDrag`, `torqueTidal`. Rendered as circular arc arrows.

### Boris Integrator

Per substep (inside `Physics.update()` while loop):

1. Store `_f1pnOld` (if 1PN enabled)
2. **Half-kick**: `w += F/m * dt/2` (E-like forces)
3. **Boris rotation**: rotate w in combined Bz + Bgz plane (preserves |v| exactly)
4. **Half-kick**: `w += F/m * dt/2`
5. Spin-orbit energy coupling, Stern-Gerlach/Mathisson-Papapetrou kicks, frame-drag torque
6. Radiation reaction (Landau-Lifshitz)
7. **Drift**: `vel = w / sqrt(1 + w²)`, `pos += vel * dt`
8. Cosmological expansion (if enabled)
9. **1PN velocity-Verlet correction**: recompute 1PN at new positions (always pairwise via `compute1PNPairwise()`), kick `w += (F_new - F_old) * dt / (2m)`
10. Rebuild quadtree, handle collisions, photon absorption
11. Reset forces + compute new forces for next substep

After all substeps: record signal-delay history (strided, once per HISTORY_STRIDE=64 `update()` calls), compute PE, reconstruct velocity-dependent display forces.

### Adaptive Substepping

- `dtSafe_accel = sqrt(SOFTENING / a_max)`
- `dtSafe_cyclotron = (2*pi / omega_c) / 8` where `omega_c = max(|q*Bz/m|, 4*|Bgz|)`
- Capped at MAX_SUBSTEPS = 32 per frame

### Fixed-Timestep Loop

`PHYSICS_DT = 1/128`. Accumulator collects `rawDt * speedScale`, drained in fixed chunks. Photon updates and tidal breakup inside the loop; energy/rendering/DOM outside.

## Force Types

### E-like Forces (radial, position-dependent)

All use Plummer softening: `r_eff = sqrt(r² + SOFTENING_SQ)` (SOFTENING = 8, SOFTENING_SQ = 64).

| Force | Formula | PE | Toggle |
|---|---|---|---|
| Gravity | `+m₁m₂/r²` (attractive) | `-m₁m₂/r` | Gravity |
| Coulomb | `-q₁q₂/r²` (like-repels) | `+q₁q₂/r` | Coulomb |
| Magnetic dipole | `-3μ₁μ₂/r⁴` (aligned ⊥-dipoles repel) | `+μ₁μ₂/r³` | Coulomb + Magnetic |
| GM dipole | `+3L₁L₂/r⁴` (co-rotating attract; GEM flip) | `-L₁L₂/r³` | Gravity + GM |

### B-like Forces (velocity-dependent, Boris rotation)

**Lorentz** (Coulomb + Magnetic): Bz from moving charge (`q_s*(v_s×r̂)_z/r²`) + spinning dipole (`+μ/r³`). Display: `forceMagnetic += (q*vel.y*Bz, -q*vel.x*Bz)`.

**Gravitomagnetic** (Gravity + GM): Bgz from moving mass (`-m_s*(v_s×r̂)_z/r²`) + spinning mass (`-2L/r³`). Boris parameter: `+2*Bgz*dt/gamma`. Display: `forceGravitomag += (4m*vel.y*Bgz, -4m*vel.x*Bgz)`.

**Frame-dragging torque**: `tau = 2*L_s*(omega_s - omega_p)/r³`. Drives spin alignment.

### Tidal Locking

Requires Gravity. Dissipative torque driving spin toward synchronous rotation:
```
coupling = m_other + q₁q₂/m₁
τ = -TIDAL_STRENGTH * coupling² * r_body³ / r⁶ * (ω_spin - ω_orbit)
```

### Yukawa Potential

Independent toggle. `F = -g²·m₁m₂·exp(-μr)/r²·(1+μr)`. Parameters: `yukawaG2` (default 1.0), `yukawaMu` (default 0.2). Includes analytical jerk for radiation reaction.

### Axion Dark Matter Coupling

Requires Coulomb. Modulates EM coupling: `α_eff = α·(1 + g·cos(m_a·t))`. Applied as `axionModulation` multiplier on all charge-dependent terms. Energy not conserved (external reservoir).

### 1PN Corrections (EIH + Darwin EM + Bazanski)

Requires Relativity. Three O(v²/c²) sectors, all accumulate into `force1PN`:

- **EIH** (GM + 1PN): Symmetric remainder from EIH after subtracting GM Lorentz piece. Produces perihelion precession ~6πM/a(1-e²) rad/orbit.
- **Darwin EM** (Magnetic + 1PN): Symmetric remainder from Darwin Lagrangian after subtracting Lorentz force.
- **Bazanski** (GM + Magnetic + 1PN): Position-dependent mixed 1/r³ force. `F = [q₁q₂(m₁+m₂) − (q₁²m₂ + q₂²m₁)] / r³`. Vanishes for identical particles.

NOT Newton's 3rd law — each particle uses its own velocity. Velocity-Verlet: stores `_f1pnOld` before drift, recomputes after via `compute1PNPairwise()` (always pairwise, even in BH mode).

### Radiation

Requires Coulomb.

**Landau-Lifshitz force**: `F_rad = tau * [dF/dt / gamma³ - v*F²/(m*gamma²) + F*(v·F)/(m*gamma⁴)]` where `tau = 2q²/(3m)`. Jerk is hybrid: analytical for gravity+Coulomb (accumulated into `p.jerk`), numerical backward difference for residual forces. Power-dissipation terms only active when relativity on. Clamped: `|F_rad| <= 0.5 * |F_ext|`.

**Photon emission**: Energy accumulated in `_radAccum`. Dipole pattern with relativistic aberration. **Absorption**: quadtree query, self-absorption guard (age < 3).

### Black Hole Mode

Toggle under Relativity (`physics.blackHoleEnabled`):
- **Kerr-Newman horizon**: `r+ = M + sqrt(M²-a²-Q²)`, naked singularity floor at `M*0.5`
- **Ergosphere**: dashed ring at `r_ergo = M + sqrt(M²-a²)`
- **Reduced softening**: SOFTENING_SQ = 1 (not 64)
- **Collision lock**: forced to Merge
- **Hawking radiation**: `κ = sqrt(disc)/(r+²+a²)`, `T = κ/(2π)`, `P = σT⁴A` where `σ = π²/60`, `A = 4π(r+²+a²)`. Extremal BHs stop radiating.
- **Evaporation**: below MIN_MASS → removed with photon burst

### Signal Delay

Requires Relativity. Three-phase solver on per-particle circular history buffers (Float64Array[256], recorded every 64 `update()` calls):
1. Newton-Raphson segment search (up to 6 iterations)
2. Exact quadratic solve on converged segment
3. Constant-velocity extrapolation for t_ret before recorded history

In BH mode: signal delay at leaf level only; distant aggregates use current positions. Returns pre-allocated `_delayedOut` (consume before next call).

### Spin-Orbit Coupling

Requires Magnetic + GM + Spin-Orbit toggle. Independent of Relativity.

Energy transfer: `dE = -mu*(v·∇Bz)*dt` (EM), `dE = -L*(v·∇Bgz)*dt` (GM). Center-of-mass kicks: Stern-Gerlach `F = +mu*∇Bz`, Mathisson-Papapetrou `F = -L*∇Bgz` (GEM flip). Both accumulate into `forceSpinCurv`. Field gradients computed in `pairForce()`.

### Disintegration

Toggle (`disintegrationEnabled`), requires Gravity. Fragments when tidal + centrifugal + Coulomb stress exceeds self-gravity. Splits into 4 pieces. Min mass: `MIN_MASS * 4`.

**Roche Lobe Overflow**: Eggleton formula. Continuous mass transfer toward companion through L1. Rate: `dM = overflow * ROCHE_TRANSFER_RATE * m`, capped 10%. Returns `{ fragments, transfers }`.

### GW Radiation

Toggle (`quadRadiationEnabled`), requires Radiation. Mass quadrupole 3rd time derivative via hybrid analytical+numerical jerk: `P_GW = (1/5)|d³I_ij/dt³|²`. Also computes EM quadrupole `P_EM = (1/180)|d³Q_ij/dt³|²` when Coulomb enabled. Energy extracted via tangential velocity scaling `scale = 1 - dE/(2·KE)`. Drag accumulated into `_radDisplayX/Y` for the radiation force arrow. Emits graviton particles (`type: 'grav'`, rendered red).

### Photon Gravitational Lensing

`dv = 2·M·r̂/r²·dt` (2× Newtonian, null geodesic). Velocity renormalized to c=1. Uses PHOTON_SOFTENING_SQ=4.

### Cosmological Expansion

Toggle (`expansionEnabled`). `pos += H*(pos - center)*dt` (Hubble flow), `w *= (1 - H*dt)` (redshift). Default `hubbleParam = 0.001`. Enabling expansion locks boundary mode to "despawn" (particles leave the domain).

## Sign Conventions (IMPORTANT)

All GEM interactions are **attractive** (gravity has one sign of "charge"):
- GM dipole: `+3*L1*L2/r^4` (positive = attractive)
- GM Boris parameter: `+2*Bgz` (co-moving masses attract)
- Bgz field: `-m_s*(v_s × r_hat)_z/r^2` (negative sign in code: `p.Bgz -= ...`)
- Frame-drag torque: positive coefficient drives spins toward co-rotation

Do NOT flip these signs.

## Potential Energy

`computePE()` in `potential.js`. Tree traversal via `treePE()` when BH on (divides by 2), exact pairwise `pairPE()` with i < j when off. Seven terms: gravitational, Coulomb (with axion), magnetic dipole, GM dipole, 1PN (EIH + Darwin EM), Bazanski, Yukawa. All Plummer-softened (reduced to 1 in BH mode).

## Energy & Momentum

`computeEnergies()` returns: `linearKE`, `spinKE`, `pe`, `fieldEnergy`, `fieldPx/Py`, `px/py`, `orbitalAngMom`, `spinAngMom`, `comX/comY`. Relativistic KE uses `wSq / (gamma + 1)` to avoid cancellation.

Darwin field corrections (O(v²/c²)) computed when Magnetic or GM enabled but 1PN is off. When 1PN on, absorbed into PE. Conservation: exact with gravity + Coulomb only, pairwise mode.

## Collisions

Detection via quadtree query. Ghost particles resolve against `original`. ID comparison prevents double-processing.

- **Merge**: conserves mass, charge, momentum (m*w), angular momentum. Minimum-image for periodic boundaries.
- **Bounce**: relativistic (Lorentz-boost to COM frame) or classical. Tangential friction (default 0.4) couples linear and spin. Overlap separation after resolution.

## Topology

When boundary = "loop": Torus (both axes normal), Klein (y-wrap mirrors x, negates w.x/angw), RP² (both axes carry glide reflections, 4 min-image candidates).

`minImage()` zero-alloc via `out` parameter. `wrapPosition()` applies velocity/spin flips. Ghost generation topology-aware with `flipVx`/`flipVy` flags.

`sim.topology` string → `physics._topologyConst` integer (TORUS=0 / KLEIN=1 / RP2=2).

## Barnes-Hut

`QuadTreePool`: SoA flat typed arrays, pre-allocated 512 nodes (doubles via `_grow()`). Zero GC. BH_THETA = 0.5, QUADTREE_CAPACITY = 4. Off by default -- pairwise gives better conservation.

Aggregates: `totalMass`, `totalCharge`, `totalMagneticMoment`, `totalAngularMomentum`, `totalMomentumX/Y`, `comX/comY`. Tree walk in `calculateForce()`: leaves → `pairForce()`, distant nodes → aggregates.

## Toggle Dependencies

```
Forces:                        Physics:
  Gravity                        Relativity
    -> Gravitomagnetic             -> Signal Delay
    -> Tidal Locking               -> 1PN              [requires Magnetic or GM]
  Coulomb                          -> Black Hole       [+Gravity, locks collision to Merge]
    -> Magnetic                  Spin-Orbit            [requires Magnetic or GM]
    -> Axion                     Radiation             [requires Coulomb]
  Yukawa               [independent]  -> GW Radiation
Disintegration                   [requires Gravity]
Barnes-Hut                       [independent]
Expansion                        [independent, in Engine tab]
```

1PN internally: EIH requires `gravitomagEnabled`, Darwin EM requires `magneticEnabled`. Bazanski requires both.

Implemented as a declarative `DEPS` array in `ui.js`, evaluated in topological order by `updateAllDeps()` on every toggle change. Each entry maps a toggle ID to a disabled-condition function. `setDepState()` applies `.ctrl-disabled` class and auto-unchecks disabled toggles. Cascades automatically because DEPS are ordered parents-before-children.

Defaults on: gravity, coulomb, magnetic, gravitomag, 1PN, relativity, signal delay, spin-orbit, radiation, tidal locking, GW radiation. Defaults off: Yukawa, Axion, Disintegration, Expansion, Barnes-Hut.

## UI

4-tab sidebar: Settings (mass/charge/spin sliders, spawn mode, force/physics toggles), Engine (BH, collisions, boundary/topology, visuals, speed), Stats (energy/momentum/drift), Particle (selected details, force breakdown, phase plot).

Topbar: Home | Brand "No-Hair" | Pause/Step/Reset/Save/Load | Theme | Panel toggle.

Preset selector uses `<optgroup>` categories: Gravity, EM, Exotic, Cosmological. 13 presets total (9 via keyboard `1`-`9`, rest dropdown-only).

Sim speed slider: range 1–128, default 64. Bounce friction slider only visible when collision mode or boundary mode is "bounce" (controlled by `updateFrictionVisibility()` inside `updateAllDeps()`). Expansion toggle locks boundary mode to "despawn".

Tab switching in inline `<script>` in index.html (not in a module). Info tips via `createInfoTip()`. Shift+click opens reference overlay from `REFERENCE` in `reference.js`. Responsive: 900px → bottom sheet, 600px/440px shared breakpoints.

## Renderer

Canvas 2D. Dark mode: additive blending (`lighter`).

- **Particles**: `r = cbrt(mass)` (BH: Kerr-Newman r+), glow in dark mode
- **Spin rings**: arc length ∝ |omega*r|, cyan=positive, orange=negative
- **Trails**: circular Float32Array[256], wrap-detection for periodic boundaries
- **Force vectors**: scale=256 (÷ mass for accel). Component colors: gravity=red, coulomb=blue, magnetic=cyan, GM=rose, 1PN=orange, spin-curv=purple, radiation=yellow, yukawa=green
- **Torque arcs**: spin-orbit=purple, frame-drag=rose, tidal=red, total=accent
- **Photons**: yellow (EM, `type: 'em'`) / red (gravitons, `type: 'grav'`), alpha fades over PHOTON_LIFETIME=256
- **Signal delay ghosts**: stroked outline at oldest history position

Particle color: neutral=slate `#8A7E72`. Charged: RGB lerp toward red (positive) or blue (negative), intensity=`|q|/5`. Uses inline hex parser (`_hex`) since `_parseHex` from shared-tokens.js is script-scoped.

## World Scale

`WORLD_SCALE = 16`. Domain = viewport / 16 in each dimension. Camera starts at zoom = WORLD_SCALE. All world coordinates use `sim.domainW/H`, not `sim.width/height`.

## Key Patterns

- `Vec2.set(x,y)` in hot paths; `pairForce()` accumulates into `out` Vec2 + display vectors, zero alloc
- QuadTreePool: SoA, `reset()`+`build()` per substep, zero GC
- `window.sim` for console debugging. `_PALETTE`/`_FONT` frozen by colors.js
- Module-level `_miOut` objects for zero-alloc minImage output
- Particle constructor declares all dynamic properties to prevent V8 hidden class transitions
- InputHandler pre-allocates `_posOut` Vec2 for mouse move
- Icon swaps: toggle `hidden` attribute, not innerHTML
- Theme: `data-theme` on `<html>` (not body)

## Gotchas

- Serve from `a9lim.github.io/` parent -- absolute paths for shared files
- `_parseHex` from `shared-tokens.js` is script-scoped -- ES6 modules use inline `_hex` parser
- `compute1PNPairwise()` zeroes `force1PN` before accumulating -- do not mix with `pairForce()` 1PN output in same step
- 1PN velocity-Verlet correction is always pairwise, even when BH is on
- Adaptive substepping uses Bz/Bgz from previous substep's force computation -- no preliminary force pass
- History recording is strided (HISTORY_STRIDE=64) after the substep loop, counting `update()` calls, not substeps
- After merge collisions, `particles.length` changes -- update loop variable `n`
- `checkDisintegration()` returns `{ fragments, transfers }` (not a flat array)
- GW quadrupole history buffer needs 4+ samples for 3rd derivative -- first frames produce no output
- Radiation power-dissipation terms (−v·F²/mγ² and +F·(v·F)/mγ⁴) only active when relativity on
- Heatmap signal delay is expensive (GRID²×N delay solves) -- mitigated by 6-frame update interval
- Reference overlay uses `renderMathInElement` from KaTeX auto-render (loaded deferred)
- World coordinates use `sim.domainW/H` (viewport / WORLD_SCALE), not pixel dimensions
