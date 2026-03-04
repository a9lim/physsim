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
main.js                     212 lines  Simulation class, fixed-timestep loop, window.sim
index.html                  415 lines  UI structure, 4-tab sidebar, preset dialog, zoom controls
styles.css                  560 lines  Project-specific CSS overrides
colors.js                    27 lines  Project color tokens (particle hues, spin ring colors)
src/
  integrator.js             763 lines  Physics class: adaptive Boris substep loop, radiation, tidal
  forces.js                 335 lines  pairForce(), computeAllForces(), calculateForce() (BH walk), compute1PNPairwise()
  signal-delay.js           315 lines  getDelayedState() (3-phase light-cone solver), interpolateHistory()
  ui.js                     338 lines  setupUI(), toggle dependencies, info tips (infoData), keyboard shortcuts
  renderer.js               406 lines  Canvas 2D: particles, trails, spin rings, vectors, torque arcs, photons
  input.js                  262 lines  InputHandler: mouse/touch, Place/Shoot/Orbit modes, hover tooltip
  collisions.js             259 lines  handleCollisions(), resolveMerge(), resolveBounce() (rel + classical)
  quadtree.js               256 lines  QuadTreePool: SoA flat typed arrays, pool-based, zero GC
  potential.js              158 lines  computePE(), treePE(), pairPE() (4 PE terms + 1PN PE)
  topology.js               129 lines  TORUS/KLEIN/RP2 constants, minImage(), wrapPosition()
  energy.js                 127 lines  computeEnergies(): KE, spin KE, momentum, angular momentum, Darwin field
  phase-plot.js             120 lines  PhasePlot: r vs v_r sidebar canvas (500-sample ring buffer)
  sankey.js                  98 lines  SankeyOverlay: energy bar chart (orphaned -- not imported by any module)
  stats-display.js           92 lines  StatsDisplay: energy/momentum/drift DOM updates, selected particle info
  presets.js                 87 lines  PRESETS object (5 scenarios), loadPreset()
  heatmap.js                 83 lines  Heatmap: 48x48 grav+electrostatic potential field overlay, 6-frame interval
  particle.js                79 lines  Particle entity: pos, vel, w, angw, per-type force vectors, history buffers
  vec2.js                    69 lines  Vec2 class: set, clone, add, sub, scale, mag, magSq, normalize, dot, dist
  config.js                  54 lines  Named constants (BH_THETA, SOFTENING, INERTIA_K, MAX_SUBSTEPS, etc.)
  relativity.js              41 lines  angwToAngVel(), angVelToAngw(), setVelocity()
  photon.js                  19 lines  Photon entity: pos, vel, energy, lifetime, emitterId
```

## Module Dependency Graph

```
main.js (Simulation class, window.sim)
  imports: Physics (integrator), Renderer, InputHandler, Particle, Heatmap, PhasePlot,
           StatsDisplay, setupUI, config constants, relativity helpers

  src/integrator.js (Physics class)
    imports: QuadTreePool + Rect, config constants, Photon, angwToAngVel,
             resetForces + computeAllForces + compute1PNPairwise (forces),
             handleCollisions (collisions), computePE (potential),
             TORUS + KLEIN + RP2 + minImage + wrapPosition (topology)

  src/forces.js
    imports: config (BH_THETA, SOFTENING_SQ, INERTIA_K, MAG_MOMENT_K, FRAME_DRAG_K),
             getDelayedState (signal-delay), TORUS + minImage (topology)

  src/energy.js
    imports: config (INERTIA_K, SOFTENING_SQ), TORUS + minImage (topology)

  src/potential.js
    imports: config (BH_THETA, SOFTENING_SQ, INERTIA_K, MAG_MOMENT_K),
             TORUS + minImage (topology)

  src/stats-display.js
    imports: computeEnergies (energy)

  src/ui.js
    imports: loadPreset (presets), PHYSICS_DT (config)

  src/renderer.js
    imports: config (MAX_TRAIL_LENGTH, PHOTON_LIFETIME, INERTIA_K)

  src/input.js
    imports: Vec2

  src/collisions.js
    imports: INERTIA_K (config), setVelocity + angwToAngVel + angVelToAngw (relativity),
             TORUS + minImage + wrapPosition (topology)

  src/signal-delay.js
    imports: HISTORY_SIZE (config), TORUS + minImage (topology)
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
- Particle radius: `r = cbrt(mass)` (density rho = 3/(4*pi))

### Per-Particle Force/Torque Display Vectors

Each particle stores per-type force vectors for component visualization: `forceGravity`, `forceCoulomb`, `forceMagnetic`, `forceGravitomag`, `force1PN`, `forceSpinCurv`, `forceRadiation`. All reset each substep via `resetForces()`. `forceSpinCurv` accumulates both Stern-Gerlach (+mu * grad(Bz)) and Mathisson-Papapetrou (-L * grad(Bgz)).

Torque display scalars: `torqueSpinOrbit` (EM + GM spin-orbit power) and `torqueFrameDrag`. Rendered as circular arc arrows around particles -- orange for spin-orbit, purple for frame-drag.

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
8. Radiation reaction (Landau-Lifshitz jerk term)
9. **Drift**: derive `vel = w / sqrt(1 + w^2)`, `pos += vel * dt`
10. Advance `simTime`
11. **1PN velocity-Verlet correction**: re-derive vel, recompute 1PN at new positions (always pairwise via `compute1PNPairwise()`), kick `w += (F_new - F_old) * dt / (2m)`
12. Rebuild quadtree
13. Handle collisions
14. Photon absorption
15. Save radiation display force, then reset forces + compute new forces for next substep

After all substeps: record signal-delay history (strided, once per HISTORY_STRIDE=200 calls), compute PE, reconstruct velocity-dependent display forces from final-substep fields.

### Adaptive Substepping

Computed at the start of each substep from current forces and B fields:
- `dtSafe_accel = sqrt(SOFTENING / a_max)`
- `dtSafe_cyclotron = (2*pi / omega_c) / 8` where `omega_c = max(|q * Bz / m|, 4 * |Bgz|)`
- `dtSub = dtRemain / min(ceil(dtRemain / dtSafe), budgetLeft)`
- Capped at MAX_SUBSTEPS = 16 per frame

### Fixed-Timestep Loop (main.js)

`PHYSICS_DT = 1/120`. Accumulator collects `rawDt * speedScale` per animation frame. Drained in fixed-step chunks of PHYSICS_DT, capped at `MAX_SUBSTEPS * PHYSICS_DT * 4`. Photon updates and tidal breakup inside the loop; energy/rendering/DOM outside.

## Force Types

### E-like Forces (radial, position-dependent)

All use Plummer softening: `r_eff = sqrt(r^2 + SOFTENING_SQ)`, where SOFTENING = 10, SOFTENING_SQ = 100.

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

**Frame-dragging torque**: `tau = FRAME_DRAG_K * m_s * (omega_s - omega_p) / r^3` (FRAME_DRAG_K = 0.1)
- Applied as `angw += tau * dt / I`; drives spin alignment

### 1PN Correction (EIH)

Requires Gravity + Relativity. O(v^2/c^2) correction to gravity using coordinate velocities.

```
radial  = -v1^2 - 2*v2^2 + 4*(v1.v2) + 1.5*(n_hat.v2)^2 + 5*m1/r + 4*m2/r
tangent = 4*(n_hat.v1) - 3*(n_hat.v2)
a_1PN   = (m2/r^2) * [n_hat * radial + (v1-v2) * tangent]
```

Computed in `pairForce()` when `onePNEnabled` flag is set. In the code, `base = sMass * invRSq * invR` and the tangential term is multiplied by `r` to convert from unit-vector to direction-vector form.

Velocity-Verlet: stores `_f1pnOld` before drift, recomputes after drift via `compute1PNPairwise()` (always pairwise, even in BH mode), applies correction kick `(F_new - F_old) * dt / (2m)`. Produces perihelion precession approximately 6*pi*M / (a*(1-e^2)) rad/orbit.

1PN PE (computed in `pairPE()`):
```
U_1PN = -(m1*m2/r) * [1.5*(v1^2+v2^2) - 3.5*(v1.v2) - 0.5*(v1.n)(v2.n) + m1/r + m2/r]
```

### Radiation

Requires Relativity.

**Larmor power**: `P = 2*q^2*a^2/3`

**Landau-Lifshitz force** (jerk term only, no Schott damping):
```
F_rad = tau * (F - F_prev) / dt / gamma^3
tau = 2 * LARMOR_K * q^2 / m = 2*q^2/(3*m)    (LARMOR_K = 1/3)
```
Clamped: `|F_rad * dt / m| <= LL_FORCE_CLAMP * |w|` (LL_FORCE_CLAMP = 0.5)

**Photon emission**: Energy accumulated in `_radAccum` per particle. Emits when >= RADIATION_THRESHOLD (0.01) and pool < MAX_PHOTONS (500). Emission angle sampled from sin^2(theta) dipole pattern with relativistic aberration (beamed toward velocity at high gamma). Photon travels at c = 1.

**Photon absorption**: Quadtree query at photon position (radius SOFTENING). Self-absorption guard: emitter skipped for first 2 substeps (age < 3). On absorb: `target.w += ph.energy * ph.vel / target.mass`. Bookkeeping: `totalRadiated` decremented, radiated momentum decremented.

### Signal Delay

Requires Relativity + Barnes-Hut off (pairwise only).

Light-cone equation: `|x_source(t_ret) - x_obs(now)| = now - t_ret` (c = 1).

Three-phase solver on per-particle circular history buffers (`Float64Array[HISTORY_SIZE=1024]` each for x, y, vx, vy, time; recorded every `HISTORY_STRIDE=200` physics updates, ~60 snapshots/sec at 100× speed, covering approximately 1707 time units at PHYSICS_DT=1/120):

1. **Newton-Raphson** (up to NR_MAX_ITER=6 iterations) on `g(t) = |x_s(t) - x_obs| - (now - t)` to locate the correct history segment. Uses proportional segment estimate + short walk for initial segment. Guaranteed convergent for subluminal sources (`g' = d_hat . v_eff + 1 > 0`).
2. **Exact quadratic solve** on the converged segment +/- 1 neighbor. Piecewise-linear trajectory makes the light-cone equation a quadratic: `(v^2 - 1)*s^2 + 2*(d.v + T)*s + (r^2 - T^2) = 0` with closed-form roots.
3. **Constant-velocity extrapolation** from the oldest buffer entry when t_ret predates recorded history. Same quadratic with s <= 0.

Early rejection: pairs with current distance > 2 * buffer time span skip straight to extrapolation (O(1)).

Returns a pre-allocated `_delayedOut` object (caller must consume before next call).

### Spin-Orbit Coupling

Requires Relativity + relevant force toggle (Magnetic for EM, GM for gravitational) + Spin-Orbit toggle.

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
Splits into FRAGMENT_COUNT (3) pieces at 120-degree intervals, radius * 1.5 from original center. Each gets mass/3, charge/3, tangential velocity from spin. Min mass to fragment: `MIN_FRAGMENT_MASS * FRAGMENT_COUNT = 2 * 3 = 6`.

## Sign Conventions (IMPORTANT)

All GEM interactions are **attractive** (gravity has one sign of "charge"):
- GM dipole: `+3*L1*L2/r^4` (positive = attractive)
- GM Boris parameter: `+2*Bgz` (co-moving masses attract)
- Bgz field: `-m_s*(v_s x r_hat)_z/r^2` (negative sign in code: `p.Bgz -= ...`)
- Frame-drag torque: positive coefficient drives spins toward co-rotation

Do NOT flip these signs.

## Potential Energy

Computed separately from forces via `computePE()` in `potential.js`. Same BH theta criterion -- tree traversal via `treePE()` when BH on (divides by 2 to avoid double-counting), exact pairwise `pairPE()` with i < j when off. Five terms: gravitational (-m1*m2/r), Coulomb (+q1*q2/r), magnetic dipole (+mu1*mu2/r^3), GM dipole (-L1*L2/r^3), 1PN PE (velocity-dependent correction). All Plummer-softened.

## Energy & Momentum (`src/energy.js`)

`computeEnergies()` returns: `linearKE`, `spinKE`, `pe`, `fieldEnergy`, `fieldPx/Py`, `px/py`, `orbitalAngMom`, `spinAngMom`, `comX/comY`.

| Quantity | Relativistic | Classical |
|---|---|---|
| Linear KE | sum((gamma - 1) * m) | sum(0.5 * m * \|v\|^2) |
| Spin KE | sum(INERTIA_K * m * (sqrt(1 + W^2*r^2) - 1)) | sum(0.5 * I * omega^2) |
| Momentum | sum(m * w) | sum(m * v) |
| Angular mom. | sum(r x m*w) + sum(I * W) about COM | same |

**Darwin field corrections** (O(v^2/c^2), computed when Coulomb or GM enabled):
- EM field energy: `-0.5 * sum_{i<j}(qi*qj/r) * [(vi.vj) + (vi.r_hat)(vj.r_hat)]`
- GM field energy: `+0.5 * sum_{i<j}(mi*mj/r) * [(vi.vj) + (vi.r_hat)(vj.r_hat)]` (opposite sign)
- Field momentum: analogous terms with `(vi + vj)` and `(vi + vj).r_hat`

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
Gravity
  -> Gravitomagnetic        [requires Gravity]
  -> 1PN                    [requires Gravity + Relativity]

Coulomb
  -> Magnetic                [requires Coulomb]

Relativity
  -> Signal Delay            [requires Relativity + BH off]
  -> Spin-Orbit              [requires Relativity]
  -> Radiation               [requires Relativity]

Tidal (Disintegration)       [independent]
Barnes-Hut                   [independent]
```

Disabled sub-toggles: `.ctrl-disabled` class (opacity 0.4, pointer-events none) applied by `setDepState()` in `ui.js`. When a parent toggle is turned off, its children are automatically unchecked and their physics flags set to false.

Default on load: all force toggles on (gravity, coulomb, magnetic, gravitomag, 1PN, relativity, signal delay, spin-orbit) except Radiation, Tidal, and Barnes-Hut which default to off.

## UI

### 4-Tab Sidebar

1. **Settings**: particle mass/charge/spin sliders, interaction mode (Place/Shoot/Orbit), force toggles (Gravity -> Gravitomagnetic/1PN, Coulomb -> Magnetic), physics toggles (Relativity -> Signal Delay/Spin-Orbit, Radiation)
2. **Engine**: Barnes-Hut toggle, collision mode (Pass/Bounce/Merge), bounce friction slider (0-1, default 0.4), boundary mode (Despawn/Loop/Bounce), topology selector (Torus/Klein/RP^2, only visible when boundary=Loop), disintegration toggle, visual toggles (trails, velocity/force/component vectors, potential field, acceleration scaling), sim speed slider (0-200, default 100)
3. **Stats**: energy breakdown (total, linear KE, spin KE, PE, field, radiated, drift), conserved quantities (momentum with particle/field/radiated components, angular momentum with orbital/spin, drift percentages)
4. **Particle**: selected particle details (ID, mass, charge, spin, speed, gamma, |F|), phase space plot canvas (r vs v_r relative to most massive body)

Tab switching is handled by an inline `<script>` in `index.html` (not in a module).

### Topbar

Home link (logo) | Brand "No-Hair" | Presets button (ghost-btn) | Pause/Step/Reset buttons | Theme toggle | Panel toggle

### Presets

| # | Name | Key | Description |
|---|---|---|---|
| 1 | Solar System | solar | Star (m=80) + 5 planets in circular orbits |
| 2 | Binary Stars | binary | Two m=50 stars, spin=0.8c, counter-orbiting |
| 3 | Galaxy | galaxy | Core (m=150) + 200 particles, circular orbits, random charge/spin |
| 4 | Collision | collision | Two groups of 50 particles heading at each other (v=+-0.5) |
| 5 | Magnetic Spin | magnetic | 5x5 grid of charged spinning particles |

### Keyboard Shortcuts

Registered via `initShortcuts()` from shared module:

| Key | Action | Group |
|---|---|---|
| Space | Pause / Play | Simulation |
| R | Reset simulation | Simulation |
| . | Step forward | Simulation |
| P | Open presets | Simulation |
| 1-5 | Load preset directly | Presets |
| V | Toggle velocity vectors | View |
| F | Toggle force vectors | View |
| C | Toggle force components | View |
| T | Toggle theme | View |
| S | Toggle sidebar | View |
| Escape | Close dialogs | View |
| ? | Help overlay | (shared) |

### Info Tips

Data object `infoData` in `ui.js` maps keys to `{ title, body }` objects. HTML body strings contain KaTeX math (`$...$`). Triggers are `.info-trigger[data-info]` buttons in the HTML. Created via `createInfoTip()` from shared module.

### Responsive

900px -> bottom sheet + 48px toolbar. 600px/440px shared breakpoints from shared-base.css (`.hide-sm`, brand shrink, tighter spacing).

## Renderer

Canvas 2D. Dark mode uses additive blending (`globalCompositeOperation: 'lighter'`).

- **Particles**: filled circle at `r = cbrt(mass)`, glow shadow in dark mode (larger glow for charged particles)
- **Spin rings**: arc at radius+2, length proportional to |omega*r| (caps at 2*pi), arrow shows CW/CCW, colored by spin sign (cyan=positive, orange=negative from `colors.js`)
- **Trails**: circular Float32Array buffer (MAX_TRAIL_LENGTH=200 points), 4 opacity groups, wrap-detection for periodic boundaries (skips segment if position jumps > half domain)
- **Force vectors**: scale=5 (divide by mass if acceleration scaling on). Total (accent color) sums all 7 component vectors. Per-type component arrows colored by force type
- **Torque arcs**: spin-orbit (orange, offset 7), frame-drag (purple, offset 5), total (accent, offset 9). Arc length proportional to |power|, arrow at end
- **Photons**: yellow circles, size = `1.5 + energy*20` (cap at 5px), glow in dark mode, alpha fades over PHOTON_LIFETIME=240
- **Velocity vectors**: scale=40, muted text color
- **Heatmap**: 48x48 offscreen canvas, diverging colormap (blue=gravity well, red=repulsive), updates every 6 frames

Particle color: neutral = `_PAL.neutral` (extended.slate). Charged: hue from `chargePos` (red extended hue) / `chargeNeg` (blue extended hue), intensity from |q|/20.

## Input (`src/input.js`)

- **Left click** (drag < 5 world units): select particle if hit, otherwise spawn at rest
- **Left drag**: spawn with velocity (Shoot: dragVector * 0.02) or at rest (Place/Orbit)
- **Right click**: remove particle within radius+5
- **Orbit mode**: finds particle with max gravitational force `(m/d^2)` on spawn point, spawns perpendicular at `v = sqrt(M/r)`, capped at 0.99c
- **Hover**: tooltip with m, q, spin (surface velocity), speed
- **Touch**: single=spawn, two-finger=pinch-zoom + pan via shared camera (300ms wasPinching guard prevents spawn after pinch)
- **Wheel zoom**: delegated to `camera.bindWheel(canvas)`

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

## Gotchas

- Serve from `a9lim.github.io/` parent -- `/shared-base.css` and `/shared-tokens.js` use absolute paths
- `#preset-dialog` needs both ID and `class="preset-dialog"` (shared CSS uses class, JS uses ID)
- `photon.js` is imported by `integrator.js` for radiation -- not related to input modes
- `sankey.js` exists but is orphaned (not imported by any module) -- was part of an earlier design
- 1PN velocity-Verlet correction is always pairwise (via `compute1PNPairwise()`), even when BH is on
- Radiation force uses jerk term only (no Schott damping term `-tau*F^2*v/m^2`)
- Shoot mode velocity scale is 0.02 (drag pixels * 0.02 = velocity)
- Spin-orbit, Stern-Gerlach, and Mathisson-Papapetrou are all gated by the same `spinOrbitEnabled` toggle
- `compute1PNPairwise()` zeroes `force1PN` before accumulating -- do not mix with `pairForce()` 1PN output in the same step
- Preliminary force pass runs before adaptive substep loop when magnetic or GM forces are active (ensures B fields are current for cyclotron frequency estimation)
- History recording is strided (HISTORY_STRIDE=200) and happens after the substep loop, not inside each substep
- Tab switching logic is in an inline `<script>` in index.html, not in ui.js or main.js
- `shared-touch.js` is loaded in the HTML head (between shared-tokens.js and shared-utils.js) but not documented in the parent CLAUDE.md loading order
