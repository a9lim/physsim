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
main.js                  414 lines  Simulation class, emitPhotonBurst(), fixed-timestep loop, save/load, pair production, pion loop, deadParticles cleanup, window.sim
index.html               511 lines  UI: 4-tab sidebar, reference overlay, zoom controls, field sliders, antimatter button
styles.css               245 lines  Project-specific CSS overrides, toggle/slider theme colors
colors.js                 18 lines  Project color tokens (particle hues, spin ring colors)
src/
  integrator.js         1349 lines  Physics class: Boris substep loop, radiation, pion emission/absorption, field excitations, tidal, GW quadrupole, expansion, Roche, external fields, Hertz bounce, scalar fields, _retireParticle
  ui.js                  529 lines  setupUI(), declarative dependency graph, info tips, reference overlay, keyboard shortcuts
  renderer.js            530 lines  Canvas 2D: particles, trails, spin rings, ergosphere, antimatter rings, vectors, torque arcs, photons, pions, delay ghosts, field overlays
  forces.js              496 lines  pairForce(), computeAllForces(), calculateForce() (BH walk), compute1PNPairwise(), Yukawa, dead particle forces
  presets.js             665 lines  PRESETS (18 scenarios, 4 groups), loadPreset(), SLIDER_MAP, TOGGLE_MAP/TOGGLE_ORDER
  reference.js           690 lines  REFERENCE object: physics reference content (KaTeX math)
  scalar-field.js        345 lines  ScalarField base class: PQS grid, topology-aware deposition, Laplacian, interpolation, gradient, field energy, field excitations
  higgs-field.js         203 lines  HiggsField extends ScalarField: Mexican hat potential, thermal phase transitions, mass modulation
  axion-field.js         181 lines  AxionField extends ScalarField: quadratic potential, scalar aF^2 coupling, EM modulation
  quadtree.js            280 lines  QuadTreePool: SoA flat typed arrays, pool-based, zero GC, depth guard
  input.js               270 lines  InputHandler: mouse/touch, Place/Shoot/Orbit modes, hover tooltip
  signal-delay.js        255 lines  getDelayedState() (3-phase light-cone solver, creationTime/deathTime guards)
  heatmap.js             248 lines  Heatmap: 64x64 potential field overlay, mode selector, signal-delayed positions, dead particle contributions, force-toggle-aware
  effective-potential.js 203 lines  EffectivePotentialPlot: V_eff(r) sidebar canvas, auto-scaling
  save-load.js           205 lines  saveState(), loadState(), downloadState(), uploadState(), quickSave/Load(), baseMass persistence
  potential.js           152 lines  computePE(), treePE(), pairPE() (7 PE terms)
  energy.js              150 lines  computeEnergies(): KE, spin KE, momentum, angular momentum, Darwin, field energies
  stats-display.js       131 lines  StatsDisplay: energy/momentum/drift DOM updates (x100 display scale)
  config.js              136 lines  Named constants, spawnOffset(), kerrNewmanRadius() helpers, pion/field excitation constants
  particle.js            127 lines  Particle: pos, vel, w, angw, baseMass, antimatter, cached magMoment/angMomentum, 11 force Vec2s, axMod, _yukawaRadAccum, history, creationTime/deathTime/_deathMass/_deathAngVel
  phase-plot.js          117 lines  PhasePlot: r vs v_r sidebar canvas (512-sample ring buffer)
  collisions.js          138 lines  handleCollisions(), resolveMerge(), antimatter annihilation, baseMass conservation, relativistic merge KE tracking, returns removed particles
  topology.js            112 lines  TORUS/KLEIN/RP2 constants, minImage(), wrapPosition()
  vec2.js                 61 lines  Vec2 class: set, clone, add, sub, scale, mag, magSq, normalize, dist, static sub
  boson-utils.js          58 lines  treeDeflectBoson(): shared BH tree walk for gravitational lensing of photons and pions
  photon.js               45 lines  Photon: pos, vel, energy, lifetime, type ('em'/'grav'), gravitational lensing via boson-utils
  pion.js                 79 lines  Pion: massive Yukawa force carrier, proper velocity, (1+v^2) GR deflection, decay -> photons
  relativity.js           22 lines  angwToAngVel(), setVelocity()
```

## Key Imports

```
main.js <- Physics (integrator), Renderer, InputHandler, Particle, HiggsField, AxionField,
           Heatmap, PhasePlot, EffectivePotentialPlot, StatsDisplay, setupUI, config, Photon, Pion, save-load

integrator.js <- QuadTreePool, config, Photon, Pion, angwToAngVel, forces (resetForces/computeAllForces/compute1PNPairwise),
                 handleCollisions, computePE, topology (accesses sim.higgsField/axionField via this.sim backref)

boson-utils.js <- config (BH_THETA, BOSON_SOFTENING_SQ)
photon.js     <- Vec2, config (EPSILON), boson-utils (treeDeflectBoson)
pion.js       <- Vec2, config (BOSON_SOFTENING_SQ), boson-utils (treeDeflectBoson)

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
7. Pion emission (scalar Larmor, when Yukawa enabled) + radiation reaction on emitter
8. **Drift**: `vel = w / sqrt(1 + w^2)`, `pos += vel * dt`
9. Cosmological expansion (if enabled)
10. **1PN velocity-Verlet correction**: recompute 1PN at new positions (always pairwise via `compute1PNPairwise()`), kick `w += (F_new - F_old) * dt / (2m)`
11. **Scalar fields**: evolve Higgs (symplectic Euler), modulate masses; evolve axion, interpolate axMod
12. Rebuild quadtree, handle collisions (with annihilation + merge KE tracking), repel contact forces, photon absorption, pion absorption
13. Deposit field excitations from merge KE into active scalar fields
14. Apply external fields, Higgs/Axion gradient forces, sync axMod, reset forces + compute new forces

After all substeps: record signal-delay history (strided, once per HISTORY_STRIDE=64 `update()` calls), compute PE, reconstruct velocity-dependent display forces.

### Adaptive Substepping

- `dtSafe_accel = sqrt(softening / a_max)`
- `dtSafe_cyclotron = (2*pi / omega_c) / 8` where `omega_c = max(|q*Bz/m|, 4*|Bgz|, |q*extBz/m|)`
- Capped at MAX_SUBSTEPS = 32

### Fixed-Timestep Loop

`PHYSICS_DT = 1/128`. Accumulator collects `rawDt * speedScale`, drained in fixed chunks. Photon updates, pion updates/decay, and tidal breakup inside the loop; energy/rendering/DOM outside.

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

Always active when Gravity is on (no separate toggle). `coupling = m_other + q1*q2/m1`. `tau = -TIDAL_STRENGTH * coupling^2 * r_body^3 / r^6 * (omega_spin - omega_orbit)`.

### Yukawa Potential

Independent toggle. `F = -g^2 * m1*m2 * exp(-mu*r)/r^2 * (1+mu*r)`. Parameters: `yukawaG2` (default 32), `yukawaMu` (default 0.05, slider 0.01-0.25). Includes analytical jerk for radiation. Emits pions as massive force carriers (see Pion section). **Scalar Breit correction** (requires 1PN): O(v^2/c^2) velocity-dependent correction from massive scalar boson exchange Hamiltonian `δH = g²m₁m₂e^{-μr}/(2r) * [v₁·v₂ + (r̂·v₁)(r̂·v₂)(1+μr)]`. Force into `force1PN`. Velocity-Verlet corrected via `compute1PNPairwise()`.

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

Key methods: `_nb()` (boundary-aware neighbor), `_depositPQS()` (topology-aware deposition), `_computeLaplacian()` (interior fast path + border path), `interpolate()`, `gradient()`, `_fieldEnergy(domainW, domainH, potentialFn)` (shared KE+gradient+potential grid integration), `draw()`, `depositExcitation()` (Gaussian wave packet into `fieldDot`).

`bcFromString()` converts boundary mode string to integer (BC_DESPAWN=0 / BC_BOUNCE=1 / BC_LOOP=2).

Field arrays are `field`/`fieldDot` (not `phi`/`phiDot` or `a`/`aDot`). Grid size: SCALAR_GRID = 64. Field clamp: SCALAR_FIELD_MAX = 2.

### Higgs Field

Independent toggle. Mexican hat potential `V(phi) = -1/2 mu^2 phi^2 + 1/4 lambda phi^4`. VEV=1; free parameter is m_H (slider 0.01-0.25, default 0.05). With VEV=1: `lambda = mu^2 = m_H^2/2`.

- **Mass generation**: `m_eff = baseMass * |phi(x)|`. At VEV, m_eff = baseMass. Symmetric phase (phi->0): effectively massless (floored at EPSILON).
- **Gradient force**: `F = +g * baseMass * grad(phi)` where g = HIGGS_COUPLING = 1. Into `forceHiggs`.
- **Field equation**: `d^2 phi/dt^2 = laplacian(phi) + mu^2_eff * phi - mu^2 * phi^3 + source/cellArea - 2*m_H * d(phi)/dt`. Symplectic Euler. Source: `g * baseMass` via PQS.
- **Phase transitions**: `mu^2_eff = mu^2 - KE_local` (thermalK=1). When local KE > mu^2, field relaxes to phi=0.
- **Boundary**: Despawn -> Dirichlet (phi=1). Bounce -> Neumann. Loop -> periodic (topology-aware).
- **Energy**: delegates to `_fieldEnergy()` with Mexican hat potential lambda, shifted so V(1)=0 (vacOffset = mu^2/4).
- **Damping**: Critical damping `damp = 2*m_H`.
- **Rendering**: Lime = depleted (phi < 1), cyan = enhanced (phi > 1). Alpha proportional to |deviation|.
- **baseMass sync**: All mass-modifying operations (merge, annihilation, Roche, disintegration, Hawking) proportionally scale baseMass. Toggle-off restores mass to baseMass.

### Axion Field

Requires Coulomb. Quadratic potential `V(a) = 1/2 m_a^2 a^2`. No symmetry breaking (vacuum at a=0). Uses **scalar** `aF^2` coupling (not pseudoscalar `aFF~` which vanishes in 2D).

- **Source**: Charged particles deposit `g * q^2` (g = AXION_COUPLING = 0.05). Neutral particles don't interact.
- **EM modulation**: `alpha_eff(x) = alpha * (1 + g*a(x))`. Per-particle `p.axMod` interpolated from local field. Clamped >= 0 to prevent EM force sign reversal. Used in `pairForce()` and `pairPE()`.
- **Gradient force**: `F = +g * q^2 * grad(a)`. Into `forceAxion`.
- **Field equation**: `d^2 a/dt^2 = laplacian(a) - m_a^2 * a - g*m_a * d(a)/dt + source/cellArea`. Damping: zeta = g/2, Q = 1/g, so g*Q = 1 (resonant buildup matches coupling strength).
- **Boundary**: Same as Higgs via `ScalarField._nb()`, but Dirichlet uses a=0 (not a=1).
- **Energy**: delegates to `_fieldEnergy()` with quadratic potential `V(a) = 1/2 m_a^2 a^2`. No offset needed.
- **Parameters**: One slider: m_a (0.01-0.25, default 0.05).
- **Rendering**: Indigo = positive (a > 0), yellow = negative (a < 0). Alpha proportional to |a|*4.

## Pions (Massive Force Carriers)

`Pion` class in `pion.js`. Massive Yukawa force carriers, analogous to `Photon` but with `v < c`. Yukawa's 1935 insight: pion mass equals `yukawaMu`.

### Emission (Scalar Larmor)

Yukawa interactions emit pions via scalar Larmor radiation: `P = g^2 * m^2 * a^2 / 3 = g^2 * F_yuk^2 / 3`. Scalar charge `Q = g*m` (Yukawa couples proportional to mass); `1/3` angular factor for spin-0 (vs `2/3` for spin-1 EM). Energy accumulated in `p._yukawaRadAccum`; emits when accumulation exceeds `MIN_MASS` (0.01, same threshold as photon emission) and pion KE would be positive (total energy > pion rest mass). Species: pi0 (neutral, 50%), pi+ or pi- (charged, 25% each). Capped at MAX_PIONS = 256.

**Radiation reaction**: After emission, particle's proper velocity `w` is scaled down to subtract the emitted energy from KE, preventing double-counting (force already computed directly).

### Velocity & Deflection

Proper velocity `w` (celerity): `vel = w / sqrt(1 + w^2)`, so `|v| < c` always. GR deflection uses `(1 + v^2)` factor (not `2x`), which correctly reduces to `2x` at `v -> c` (null geodesic) and `1x` at `v -> 0` (Newtonian).

### Decay

`pi0 -> 2 photons` (back-to-back perpendicular to flight), `pi+/- -> 1 photon` (along flight direction). Uses `sim._PhotonClass` reference to avoid circular import.

### Absorption

Quadtree overlap query after photon absorption. Transfers momentum and charge (pi+/-) to absorbing particle. Self-absorption guard: `pion.emitterId != particle index` and `pion.age >= 3`.

### Constants

`PION_HALF_LIFE = 32` (probabilistic decay: `P = 1 - exp(-ln2/t_half * dt)` per substep), `MAX_PIONS = 256`, `BOSON_SOFTENING_SQ = 4` (shared by photon and pion lensing), `BOSON_ABSORB_FRACTION = 1` (absorption cross-section), `BOSON_MIN_AGE = 4` (minimum substeps before absorption).

## Field Excitations

Merge collisions deposit Gaussian wave packets into active scalar fields via `ScalarField.depositExcitation()`. The existing Klein-Gordon wave equation propagates them naturally.

- **Trigger**: KE lost in inelastic merge (`keBefore - keAfter`), tracked by `handleCollisions()` returning `{ annihilations, merges }`.
- **Amplitude**: `MERGE_EXCITATION_SCALE * sqrt(keLost)` (MERGE_EXCITATION_SCALE = 0.5).
- **Shape**: Gaussian bump deposited into `fieldDot` array with `sigma = FIELD_EXCITATION_SIGMA` (2 grid cells), 3-sigma cutoff.
- **Higgs excitations**: Merge energy excites oscillations around VEV=1 ("Higgs boson" analog).
- **Axion excitations**: Merge energy excites oscillations around vacuum a=0 ("axion particle" analog).
- **Constants**: `FIELD_EXCITATION_SIGMA = 2`, `MERGE_EXCITATION_SCALE = 0.5`.

## Advanced Physics

### 1PN Corrections (EIH + Darwin EM + Bazanski)

Requires Relativity. Four O(v^2/c^2) sectors, all into `force1PN`:

- **EIH** (GM + 1PN): Remainder from EIH after subtracting GM Lorentz. Produces perihelion precession.
- **Darwin EM** (Magnetic + 1PN): Remainder from Darwin Lagrangian after subtracting Lorentz force.
- **Bazanski** (GM + Magnetic + 1PN): Mixed 1/r^3 force. Vanishes for identical particles.
- **Scalar Breit** (Yukawa + 1PN): Full Breit correction for massive scalar exchange. No subtracted piece (scalar exchange has no magnetic analog).

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
3. Constant-velocity extrapolation for t_ret before recorded history (skipped for dead particles)

**Light-cone causality**: Both particle creation and deletion respect finite propagation speed. Newly placed particles have `creationTime = simTime`; the solver rejects extrapolation past creation (particle didn't exist yet). Deleted particles are moved to `sim.deadParticles[]` via `Physics._retireParticle()`, which records a final history snapshot, saves `_deathMass`/`deathTime`, and caches dipole moments. Dead particles continue to exert forces/potential via signal delay until their light-cone fades past all observers, then are garbage-collected (`simTime - deathTime > 2 * domain_diagonal`). The solver skips backward extrapolation for dead particles to prevent spurious solutions.

**Dead particle force path**: `computeAllForces()` and `Heatmap.update()` iterate `deadParticles` as additional sources (always pairwise with signal delay, even when Barnes-Hut is on for live particles). `_deathMass` is used instead of `mass` (which may be zeroed by merge). Dead particles are excluded from `compute1PNPairwise()` (their contribution is constant across drift, so the velocity-Verlet correction is zero).

**Retirement points**: boundary despawn (integrator), collision merge/annihilation (collisions.js returns `removed`), disintegration (main.js), Hawking evaporation (main.js), right-click delete (input.js). All reset paths (preset load, clear, save-load) clear `deadParticles`.

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

**Collisions**: Three modes -- pass (none), bounce (Hertz contact via `_applyRepulsion()`), merge (quadtree overlap detection, conserves mass/charge/momentum/angular momentum). `handleCollisions()` returns `{ annihilations, merges, removed }` -- integrator emits photons from annihilations, deposits field excitations from merges, and retires removed particles for signal delay fade-out.

## Topology

When boundary = "loop": Torus (both axes normal), Klein (y-wrap mirrors x, negates w.x/angw), RP^2 (both axes glide reflections, 4 min-image candidates). `minImage()` zero-alloc via `out` parameter. `sim.topology` string -> `physics._topologyConst` integer (TORUS=0/KLEIN=1/RP2=2).

## Barnes-Hut

`QuadTreePool`: SoA flat typed arrays, pre-allocated 512 nodes (doubles via `_grow()`). Zero GC. BH_THETA = 0.5, QUADTREE_CAPACITY = 4. Off by default. Aggregates: totalMass, totalCharge, totalMagneticMoment, totalAngularMomentum, totalMomentumX/Y, comX/comY.

## Toggle Dependencies

```
Forces:                        Physics:
  Gravity                        Relativity          [signal delay auto-activates]
    -> Gravitomagnetic             -> 1PN             [requires Magnetic, GM, or Yukawa]
    (+ tidal locking, always)      -> Black Hole      [+Gravity, locks collision to Merge]
  Coulomb                        Spin-Orbit           [requires Magnetic or GM]
    -> Magnetic                  Radiation             [requires Gravity or Coulomb]
    -> Axion                       Larmor + EM quad   [when Coulomb on]
  Yukawa               [independent]  GW quad         [when Gravity on]
  Higgs                [independent]
Disintegration                   [requires Gravity, locks collision to Merge]
Barnes-Hut                       [independent]
Expansion                        [independent, in Engine tab]
```

1PN internally: EIH requires `gravitomagEnabled`, Darwin EM requires `magneticEnabled`. Bazanski requires both. Scalar Breit requires `yukawaEnabled`.

Declarative `DEPS` array in `ui.js`, evaluated in topological order by `updateAllDeps()`. `setDepState()` applies `.ctrl-disabled` and auto-unchecks disabled toggles.

Defaults on: gravity, coulomb, magnetic, gravitomag, 1PN, relativity, spin-orbit, radiation. Defaults off: Yukawa, Axion, Higgs, Disintegration, Expansion, Barnes-Hut, Black Hole. Tidal locking has no toggle — always active when gravity is on.

## UI

4-tab sidebar: Settings (mass/charge/spin, spawn mode, force/physics toggles), Engine (BH, collisions, boundary/topology, external fields, visuals, speed), Stats (energy/momentum/drift), Particle (selected details, force breakdown, phase plot, effective potential).

Topbar: Home | Brand "No-Hair" | Pause/Step/Reset/Save/Load | Antimatter | Theme | Panel toggle.

18 presets in 4 `<optgroup>` categories: Gravity (6), Electromagnetism (3), Exotic (7), Cosmological (2). First 9 via keyboard `1`-`9`. Sim speed range 1-128, default 64.

## Renderer

Canvas 2D. Dark mode: additive blending (`lighter`). WORLD_SCALE = 16 (domain = viewport / 16). Camera starts at zoom = WORLD_SCALE.

- **Particles**: r = cbrt(mass) (BH: Kerr-Newman r+), glow in dark mode. Neutral = slate. Charged: RGB lerp toward red(+)/blue(-), intensity = |q|/5.
- **Trails**: circular Float32Array[256], wrap-detection for periodic boundaries
- **Force vectors**: component colors: gravity=red, coulomb=blue, magnetic=cyan, GM=rose, 1PN=orange, spin-curv=purple, radiation=yellow, yukawa=green, external=brown, higgs=lime, axion=indigo
- **Field overlays**: 64x64 offscreen canvas, bilinear-upscaled. Higgs: magenta/cyan. Axion: indigo/yellow.
- **Photons**: yellow (EM) / red (gravitons), alpha fades over PHOTON_LIFETIME=256
- **Pions**: green circles (`_PALETTE.extended.green`), glow in dark mode, constant alpha (no fade — decay is probabilistic via half-life)
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
- `handleCollisions()` only runs for merge mode; returns `{ annihilations, merges, removed }` -- integrator emits photons, deposits field excitations, and retires removed particles
- Old save files with `collision: 'repel'` are migrated to `'bounce'` in loadState()
- External Bz enters Boris rotation alongside particle-sourced Bz -- included in `needBoris` condition check
- ScalarField arrays are `field`/`fieldDot` (not `phi`/`phiDot` or `a`/`aDot`)
- PQS stencil extends to `[ix-1..ix+2]`; `_fieldAt()` uses boundary clamping, `_depositPQS()` uses `_nb()` for topology wrapping
- Higgs `modulateMasses()` updates radius/radiusSq/invMass inline (not via `updateColor()`) to avoid per-substep string allocation
- `baseMass` must be saved/loaded and proportionally scaled wherever `mass` is modified
- `_fieldEnergy()` in ScalarField base handles KE+gradient+potential grid integration; subclasses pass potential lambda
- Higgs `energy()` shifts potential by +mu^2/4 so V(1)=0
- Higgs/Axion field reset on preset load and clear; Higgs mass -> baseMass on toggle-off; Axion axMod -> 1 on toggle-off
- Axion `p.axMod` is per-particle (interpolated from local field), not global -- used in `pairForce()`/`pairPE()`
- Axion `p.axMod` clamped >= 0 -- without this, EM force sign reversal causes runaway acceleration
- `magMoment`/`angMomentum` cache reflects previous `computeAllForces()` state -- consistent with B-field gradients used in same substep
- Ghost particles must carry `magMoment`/`angMomentum` fields (set in `_addGhost()`)
- Both Photon and Pion use shared `treeDeflectBoson()` from `boson-utils.js` for BH tree lensing; fall back to O(N) when pool is null or root < 0
- Photon passes grFactor=2 (null geodesic), Pion passes grFactor=1+v² (massive particle)
- Pion decay uses `sim._PhotonClass` reference (set in main.js) to avoid circular import with photon.js
- `_yukawaRadAccum` on Particle accumulates pion emission energy -- reset to 0 after each emission
- Field excitation `depositExcitation()` writes to `fieldDot` (not `field`) -- wave equation propagates naturally
- `sim.pions` array must be cleared on preset load and reset (in main.js, save-load.js, ui.js)
- `sim.deadParticles` must be cleared on preset load, clear, and save-load (same locations as pions)
- Dead particles use `_deathMass` (not `mass`) in force/heatmap code -- merged particles have mass=0 but `_deathMass` preserves the pre-merge value
- `_retireParticle()` must be called BEFORE the particle is removed from the array (needs valid pos/vel for final history snapshot)
- Collision code saves `_deathMass` before `resolveMerge()` zeroes mass -- `_retireParticle` uses `p.mass > 0` check to decide whether to overwrite
- Dead particles skip backward extrapolation in signal-delay solver (deathTime < Infinity guard) -- prevents spurious solutions when true retarded time is past death
- Dead particles carry `_deathAngVel` (cached by `_retireParticle()`) -- used for magnetic/GM dipole forces in signal-delay path
- QuadTree `insert()` has depth guard (max 48) to prevent stack overflow from coincident particles
- `_computeLaplacian()` splits into interior fast path (direct ±1/±GRID indexing) and border path (uses `_nb()`) for performance
- Heatmap `update()` takes `gravityEnabled`/`coulombEnabled` params to skip disabled force contributions
- Collisions use relativistic KE (`wSq / (gamma+1) * mass`) for merge energy tracking, not `0.5*m*v²`
- External field trig (`cos`/`sin` of angle sliders) cached once per frame via `_cacheExternalFields()`
