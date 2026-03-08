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
index.html               477 lines  UI: 4-tab sidebar, reference overlay, zoom controls, field sliders
styles.css               269 lines  Project-specific CSS overrides, toggle/slider theme colors
colors.js                 18 lines  Project color tokens (particle hues, spin ring colors)
src/
  integrator.js         1463 lines  Physics class: Boris substep loop, radiation, pion emission/absorption, field excitations, tidal, GW quadrupole, expansion, Roche, external fields, Hertz bounce, scalar fields, _retireParticle
  ui.js                  517 lines  setupUI(), declarative dependency graph, info tips, reference overlay, keyboard shortcuts
  renderer.js            532 lines  Canvas 2D: particles, trails, spin rings, ergosphere, antimatter rings, vectors, torque arcs, photons, pions, delay ghosts, field overlays
  forces.js              598 lines  pairForce(), computeAllForces(), calculateForce() (BH walk), compute1PN() (BH walk + pairwise), Yukawa, dead particle forces
  presets.js             688 lines  PRESETS (19 scenarios, 4 groups), loadPreset(), SLIDER_MAP, TOGGLE_MAP/TOGGLE_ORDER
  reference.js           690 lines  REFERENCE object: physics reference content (KaTeX math)
  scalar-field.js        381 lines  ScalarField base class: PQS grid, topology-aware deposition, Laplacian, interpolation, gradient, field energy, field excitations
  higgs-field.js         259 lines  HiggsField extends ScalarField: Mexican hat potential, thermal phase transitions, mass modulation
  axion-field.js         261 lines  AxionField extends ScalarField: quadratic potential, scalar aF^2 coupling, PQ pseudoscalar coupling, EM + Yukawa modulation
  quadtree.js            274 lines  QuadTreePool: SoA flat typed arrays, pool-based, zero GC, depth guard
  input.js               238 lines  InputHandler: mouse/touch, left/right-click symmetry (matter/antimatter), hover tooltip
  signal-delay.js        257 lines  getDelayedState() (3-phase light-cone solver, creationTime/deathTime guards)
  heatmap.js             248 lines  Heatmap: 64x64 potential field overlay, mode selector, signal-delayed positions, dead particle contributions, force-toggle-aware
  effective-potential.js 204 lines  EffectivePotentialPlot: V_eff(r) sidebar canvas, auto-scaling
  save-load.js           205 lines  saveState(), loadState(), downloadState(), uploadState(), quickSave/Load(), baseMass persistence
  potential.js           170 lines  computePE(), treePE(), pairPE() (7 PE terms)
  energy.js              177 lines  computeEnergies(): KE, spin KE, momentum, angular momentum, Darwin, field energies
  stats-display.js       131 lines  StatsDisplay: energy/momentum/drift DOM updates (x100 display scale)
  config.js              153 lines  Named constants, mode enums (COL_*/BOUND_*/TORUS/KLEIN/RP2), spawnOffset(), kerrNewmanRadius() helpers
  particle.js            132 lines  Particle: pos, vel, w, angw, baseMass, antimatter, cached magMoment/angMomentum, 11 force Vec2s, axMod, _yukawaRadAccum, history, creationTime/deathTime/_deathMass/_deathAngVel
  phase-plot.js          117 lines  PhasePlot: r vs v_r sidebar canvas (512-sample ring buffer)
  collisions.js          141 lines  handleCollisions(), resolveMerge(), antimatter annihilation, baseMass conservation, relativistic merge KE tracking, returns removed particles
  topology.js            105 lines  minImage(), wrapPosition() (constants moved to config.js)
  vec2.js                 61 lines  Vec2 class: set, clone, add, sub, scale, mag, magSq, normalize, dist, static sub
  boson-utils.js          58 lines  treeDeflectBoson(): shared BH tree walk for gravitational lensing of photons and pions
  massless-boson.js       45 lines  MasslessBoson: pos, vel, energy, lifetime, type ('em'/'grav'), gravitational lensing via boson-utils
  pion.js                187 lines  Pion: massive Yukawa force carrier, proper velocity, (1+v^2) GR deflection, charged decay -> electron/positron + photon, neutral decay -> 2 photons
  relativity.js           22 lines  angwToAngVel(), setVelocity()
```

## Key Imports

```
main.js <- Physics (integrator), Renderer, InputHandler, Particle, HiggsField, AxionField,
           Heatmap, PhasePlot, EffectivePotentialPlot, StatsDisplay, setupUI, config, MasslessBoson, Pion, save-load

integrator.js <- QuadTreePool, config, MasslessBoson, Pion, angwToAngVel, forces (resetForces/computeAllForces/compute1PN),                 handleCollisions, computePE, topology (accesses sim.higgsField/axionField via this.sim backref)

boson-utils.js <- config (BH_THETA, BOSON_SOFTENING_SQ)
massless-boson.js <- Vec2, config (EPSILON), boson-utils (treeDeflectBoson)
pion.js       <- Vec2, config (BOSON_SOFTENING_SQ), boson-utils (treeDeflectBoson)

forces.js     <- config, getDelayedState, topology
energy.js     <- config, topology (accesses sim.higgsField/axionField via window.sim)
potential.js  <- config, topology
renderer.js   <- config (higgsField/axionField set by main.js)
heatmap.js    <- config, getDelayedState, topology
effective-potential.js <- config, topology
phase-plot.js  <- config, topology
scalar-field.js <- config, topology
higgs-field.js  <- config, ScalarField
axion-field.js  <- config, ScalarField
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
10. **1PN velocity-Verlet correction**: rebuild tree (if BH on), recompute 1PN at new positions via `compute1PN()` (BH tree walk or pairwise), kick `w += (F_new - F_old) * dt / (2m)`
11. **Scalar fields**: evolve Higgs (Störmer-Verlet), modulate masses (momentum-conserving); evolve axion (Störmer-Verlet), interpolate axMod
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
| Magnetic dipole | `+3*mu1*mu2/r^4` | `+mu1*mu2/r^3` | Coulomb + Magnetic |
| GM dipole | `+3*L1*L2/r^4` (co-rotating attract) | `-L1*L2/r^3` | Gravity + GM |

### B-like Forces (velocity-dependent, Boris rotation)

**Lorentz** (Coulomb + Magnetic): Bz from moving charge (`q_s*(v_s x r_hat)_z/r^2`) + spinning dipole (`-mu/r^3`). Display: `forceMagnetic += (q*vel.y*Bz, -q*vel.x*Bz)`.

**Gravitomagnetic** (Gravity + GM): Bgz from moving mass (`-m_s*(v_s x r_hat)_z/r^2`) + spinning mass (`-2L/r^3`). Boris parameter: `+2*Bgz*dt/gamma`. Display: `forceGravitomag += (4m*vel.y*Bgz, -4m*vel.x*Bgz)`.

**Frame-dragging torque**: `tau = 2*L_s*(omega_s - omega_p)/r^3`. Drives spin alignment.

### Tidal Locking

Always active when Gravity is on (no separate toggle). `coupling = m_other + q1*q2/m1`. `tau = -TIDAL_STRENGTH * coupling^2 * r_body^5 / r^6 * (omega_spin - omega_orbit)`. TIDAL_STRENGTH = 0.3.

### Yukawa Potential

Independent toggle. `F = -g^2 * m1*m2 * exp(-mu*r)/r^2 * (1+mu*r)`. Parameters: `yukawaCoupling` (default 14), `yukawaMu` (default 0.15, slider 0.05-0.25). Includes analytical jerk for radiation. Emits pions as massive force carriers (see Pion section). **Scalar Breit correction** (requires 1PN): O(v^2/c^2) velocity-dependent correction from massive scalar boson exchange Hamiltonian `δH = g²m₁m₂e^{-μr}/(2r) * [v₁·v₂ + (r̂·v₁)(r̂·v₂)(1+μr)]`. Force into `force1PN`. Velocity-Verlet corrected via `compute1PN()`.

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

Shared PQS (cubic B-spline, order 3) grid infrastructure for Higgs and Axion. 4x4 = 16 node stencil per particle. C^2 interpolation, C^2 gradients (PQS-interpolated central-difference grid gradients). Pre-allocated weight arrays for zero-alloc hot path.

Key methods: `_nb()` (boundary-aware neighbor), `_depositPQS()` (topology-aware deposition), `_computeLaplacian()` (interior fast path + border path), `_computeGridGradients()` (central differences, interior fast path + border path), `interpolate()`, `gradient()` (PQS-interpolates pre-computed grid gradients), `_fieldEnergy(domainW, domainH, potentialFn)` (shared KE+gradient+potential grid integration), `draw()`, `depositExcitation()` (Gaussian wave packet into `fieldDot`).

Boundary mode integers (BOUND_DESPAWN=0 / BOUND_BOUNCE=1 / BOUND_LOOP=2) defined in config.js, passed directly from integrator.

Field arrays are `field`/`fieldDot` (not `phi`/`phiDot` or `a`/`aDot`). Grid size: SCALAR_GRID = 64. Field clamp: SCALAR_FIELD_MAX = 2.

### Higgs Field

Independent toggle. Mexican hat potential `V(phi) = -1/2 mu^2 phi^2 + 1/4 lambda phi^4`. VEV=1; free parameter is m_H (slider 0.25-0.75, default 0.50). With VEV=1: `lambda = mu^2 = m_H^2/2`.

- **Mass generation**: `m_eff = baseMass * max(|phi(x)|, HIGGS_MASS_FLOOR)`. At VEV, m_eff = baseMass. Symmetric phase (phi→0): floored at 5% baseMass (HIGGS_MASS_FLOOR=0.05), capping gradient-force acceleration at 20× and preventing thermal-depletion runaway.
- **Gradient force**: `F = +g * baseMass * sign(phi) * grad(phi)` where g = HIGGS_COUPLING = 1. Into `forceHiggs`. The `sign(phi)` ensures consistency with mass generation `m = baseMass * |phi|`.
- **Field equation**: `d^2 phi/dt^2 = laplacian(phi) + mu^2_eff * phi - mu^2 * phi^3 + source/cellArea - 2*m_H * d(phi)/dt`. Störmer-Verlet (KDK, O(dt²)). Source: `g * baseMass` via PQS.
- **Phase transitions**: `mu^2_eff = mu^2 - KE_local` (thermalK=1). When local KE > mu^2, field relaxes to phi=0.
- **Boundary**: Despawn -> Dirichlet (phi=1). Bounce -> Neumann. Loop -> periodic (topology-aware).
- **Energy**: delegates to `_fieldEnergy()` with Mexican hat potential lambda, shifted so V(1)=0 (vacOffset = mu^2/4).
- **Damping**: Critical damping `damp = 2*m_H`.
- **Rendering**: Purple = depleted (phi < 1), lime = enhanced (phi > 1). Alpha proportional to |deviation|.
- **baseMass sync**: All mass-modifying operations (merge, annihilation, Roche, disintegration, Hawking) proportionally scale baseMass. Toggle-off restores mass to baseMass.

### Axion Field

Independent toggle; requires Coulomb or Yukawa. Quadratic potential `V(a) = 1/2 m_a^2 a^2`. No symmetry breaking (vacuum at a=0). Two independent coupling channels, gated by their respective force toggles:

**Scalar EM coupling (aF², active when Coulomb on)**: Same for matter and antimatter. The QCD pseudoscalar `aFF~` vanishes in 2D.
- **Source**: `g * q^2` (g = AXION_COUPLING = 0.05).
- **EM modulation**: `alpha_eff(x) = alpha * (1 + g*a(x))`. Per-particle `p.axMod` interpolated from local field. Clamped >= 0. Set to 1 when Coulomb off. Pairwise interactions use geometric mean `sqrt(axMod_i * axMod_j)` for Newton's 3rd law symmetry.
- **Gradient force**: `F = +g * q^2 * grad(a)`. Into `forceAxion`.

**Pseudoscalar PQ coupling (aGG~ analog, active when Yukawa on)**: Peccei-Quinn mechanism. Flips sign under CP (matter vs antimatter).
- **Source**: `±g * m` (positive for matter, negative for antimatter).
- **Yukawa modulation**: `g^2_eff = g^2 * yukMod`. Per-particle `p.yukMod`: `1 + g*a` for matter, `1 - g*a` for antimatter. Clamped >= 0. Set to 1 when Yukawa off. Pairwise interactions use geometric mean `sqrt(yukMod_i * yukMod_j)`. Used in `pairForce()`, `pairPE()`, `compute1PN()` (Scalar Breit), `_vEff()`.
- **Gradient force**: `F = ±g * m * grad(a)`. Into `forceAxion`.
- At vacuum (a=0): `yukMod = 1` for both → CP conserved (PQ solution).
- **Field equation**: `d^2 a/dt^2 = laplacian(a) - m_a^2 * a - g*m_a * d(a)/dt + source/cellArea`. Störmer-Verlet (KDK, O(dt²)). Damping: zeta = g/2, Q = 1/g, so g*Q = 1 (resonant buildup matches coupling strength).
- **Boundary**: Same as Higgs via `ScalarField._nb()`, but Dirichlet uses a=0 (not a=1).
- **Energy**: delegates to `_fieldEnergy()` with quadratic potential `V(a) = 1/2 m_a^2 a^2`. No offset needed.
- **Parameters**: One slider: m_a (0.01-0.10, default 0.05).
- **Rendering**: Indigo = positive (a > 0), yellow = negative (a < 0). Alpha proportional to |a|*4.

## Pions (Massive Force Carriers)

`Pion` class in `pion.js`. Massive Yukawa force carriers, analogous to `MasslessBoson` but with `v < c`. Yukawa's 1935 insight: pion mass equals `yukawaMu`.

### Emission (Scalar Larmor)

Yukawa interactions emit pions via scalar Larmor radiation: `P = g^2 * m^2 * a^2 / 3 = g^2 * F_yuk^2 / 3`. Scalar charge `Q = g*m` (Yukawa couples proportional to mass); `1/3` angular factor for spin-0 (vs `2/3` for spin-1 EM). Energy accumulated in `p._yukawaRadAccum`; emits when accumulation exceeds `MIN_MASS` (0.01, same threshold as photon emission) and pion KE would be positive (total energy > pion rest mass). Species: pi0 (neutral, 50%), pi+ or pi- (charged, 25% each). Capped at MAX_PIONS = 256.

**Radiation reaction**: After emission, particle's proper velocity `w` is rescaled using the exact relativistic formula (`gamma_new = 1 + KE_new/m`, `w_new^2 = gamma_new^2 - 1`), preventing double-counting. Pion emission angles are Lorentz-aberrated from the particle's rest frame to the lab frame.

### Velocity & Deflection

Proper velocity `w` (celerity): `vel = w / sqrt(1 + w^2)`, so `|v| < c` always. GR deflection uses `(1 + v^2)` factor (not `2x`), which correctly reduces to `2x` at `v -> c` (null geodesic) and `1x` at `v -> 0` (Newtonian).

### Decay

Neutral and charged pions have different half-lives and decay channels:

- `pi0 -> 2 photons` (half-life 32): back-to-back in rest frame, Lorentz-boosted to lab frame.
- `pi+ -> positron + photon` (half-life 128): two-body kinematics in rest frame (exact energy/momentum split for `ELECTRON_MASS = 0.01`), Lorentz-boosted. Positron is `antimatter=true`, inherits pion charge.
- `pi- -> electron + photon` (half-life 128): same kinematics. Electron is `antimatter=false`, inherits pion charge.

Uses `sim._MasslessBosonClass` reference to avoid circular import. Photon decay products inherit the pion's `emitterId`. Electron/positron spawned via `sim.addParticle()` with `skipBaseline: true`.

### Absorption

Quadtree overlap query after photon absorption. Transfers momentum and charge (pi+/-) to absorbing particle. Self-absorption permanently blocked: `emitterId != particle index`.

### Constants

`PION_HALF_LIFE = 32` (pi0, fast EM decay), `CHARGED_PION_HALF_LIFE = 128` (pi+/-, slower weak decay). Both have pre-computed per-substep `PION_DECAY_PROB` / `CHARGED_PION_DECAY_PROB`. `ELECTRON_MASS = 0.01` (charged pion decay product). `MAX_PIONS = 256`, `BOSON_SOFTENING_SQ = 4` (shared by photon and pion lensing), `BOSON_ABSORB_FRACTION = 1` (absorption cross-section), `BOSON_MIN_AGE = 4` (minimum substeps before absorption), `ABERRATION_THRESHOLD = 1.01` (min gamma for Lorentz aberration of emission angles), `QUADRUPOLE_POWER_CLAMP = 0.01` (max quadrupole dE as fraction of system KE).

## Field Excitations

Merge collisions deposit Gaussian wave packets into active scalar fields via `ScalarField.depositExcitation()`. The existing Klein-Gordon wave equation propagates them naturally.

- **Trigger**: KE lost in inelastic merge (`keBefore - keAfter`), tracked by `handleCollisions()` returning `{ annihilations, merges, removed }`.
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

NOT Newton's 3rd law. Velocity-Verlet: stores `_f1pnOld` before drift, rebuilds tree at post-drift positions (if BH on), recomputes via `compute1PN()` (BH tree walk O(N log N) or pairwise O(N²)).

### Radiation

Requires Gravity, Coulomb, or Yukawa. Single toggle controls four mechanisms:

- **Larmor dipole** (requires Coulomb): Landau-Lifshitz force. Jerk is hybrid: analytical for gravity+Coulomb+Yukawa, numerical backward difference for residual. Power-dissipation terms only active with relativity on. Clamped: `|F_rad| <= 0.5 * |F_ext|`.
- **EM quadrupole** (requires Coulomb): `P = (1/180)|d^3 Q_ij/dt^3|^2`. Emits photons (type: 'em').
- **GW quadrupole** (requires Gravity): `P = (1/5)|d^3 I^TF_ij/dt^3|^2` (trace-free STF tensor, COM-relative coordinates). Per-particle energy extraction weighted by contribution to d^3I/dt^3. Emits gravitons (type: 'grav', rendered red).
- **Pion emission / scalar Larmor** (requires Yukawa): `P = g^2 * F_yuk^2 / 3`. Emits pions (see Pion section).

Photon quadrupole types use TT-projected angular emission via rejection sampling. Photon absorption via quadtree query (self-absorption permanently blocked by emitterId).

### Black Hole Mode

Toggle under Relativity (requires Gravity). Locks collision to Merge.
- **No hair**: Antimatter distinction is erased. Toggling BH on converts all existing antimatter to matter. `addParticle()` blocks antimatter creation. Right-click spawns matter (not antimatter). Pair production disabled. Pion charged decay products forced to matter. Input handler uses simple left-click=select, right-click=delete.
- **Kerr-Newman horizon**: `r+ = M + sqrt(M^2 - a^2 - Q^2)` where `a = INERTIA_K*r^2*|omega|`, naked singularity floor at `M*BH_NAKED_FLOOR`
- **Ergosphere**: dashed ring at `r_ergo = M + sqrt(M^2 - a^2)` (visual only)
- **Reduced softening**: BH_SOFTENING_SQ = 16
- **Hawking radiation** (requires Radiation): `kappa = sqrt(disc)/(2*M*r+)`, `T = kappa/(2*pi)`, `P = sigma*T^4*A`. Extremal BHs stop radiating.
- **Evaporation**: below MIN_MASS -> removed with photon burst via `emitPhotonBurst()`

### Signal Delay

Auto-activates with Relativity. Three-phase solver on per-particle circular history buffers (Float64Array[256] for pos/vel/angw, recorded every HISTORY_STRIDE=64 `update()` calls):
1. Newton-Raphson segment search (up to 8 iterations)
2. Exact quadratic solve on converged segment
3. Constant-velocity extrapolation for t_ret before recorded history (skipped for dead particles)

**Light-cone causality**: Both particle creation and deletion respect finite propagation speed. Newly placed particles have `creationTime = simTime`; the solver rejects extrapolation past creation (particle didn't exist yet). Deleted particles are moved to `sim.deadParticles[]` via `Physics._retireParticle()`, which records a final history snapshot, saves `_deathMass`/`deathTime`, and caches dipole moments. Dead particles continue to exert forces/potential via signal delay until their light-cone fades past all observers, then are garbage-collected (`simTime - deathTime > 2 * domain_diagonal`). The solver skips backward extrapolation for dead particles to prevent spurious solutions.

**Dead particle force path**: `computeAllForces()` and `Heatmap.update()` iterate `deadParticles` as additional sources (always pairwise with signal delay, even when Barnes-Hut is on for live particles). `_deathMass` is used instead of `mass` (which may be zeroed by merge). Dead particles are excluded from `compute1PN()` (their contribution is constant across drift, so the velocity-Verlet correction is zero).

**Retirement points**: boundary despawn (integrator), collision merge/annihilation (collisions.js returns `removed`), disintegration (main.js), Hawking evaporation (main.js), right-click delete (input.js). All reset paths (preset load, clear, save-load) clear `deadParticles`.

**Liénard-Wiechert aberration**: Signal-delayed forces include the `(1 - n̂·v_source)^{-3}` aberration factor from the retarded Green's function, clamped to [0.01, 100]. Applied to gravity, Coulomb, Yukawa, and dipole forces in `pairForce()`. Not applied to 1PN corrections (already O(v²), aberration would give O(v³)). Retarded angular velocity interpolated from `histAngW` for accurate dipole moments.

BH mode: signal delay at leaf level only; distant aggregates use current positions.

### Spin-Orbit Coupling

Requires Magnetic + GM + Spin-Orbit toggle. Independent of Relativity. Stern-Gerlach `F = +mu*grad(Bz)`, Mathisson-Papapetrou `F = -L*grad(Bgz)` (GEM flip). Both into `forceSpinCurv`.

### Disintegration & Roche

Requires Gravity. Locks collision to Merge. Fragments when tidal + centrifugal + Coulomb stress exceeds self-gravity. Splits into SPAWN_COUNT (4) pieces. Roche overflow: full Eggleton (1983) formula `r_L/a = 0.49*q^{2/3} / (0.6*q^{2/3} + ln(1+q^{1/3}))`, continuous mass transfer through L1. Returns `{ fragments, transfers }`.

### Cosmological Expansion

Toggle. `pos += H*(pos - center)*dt` (Hubble flow), `w *= (1 - H*dt)` (redshift). Default H = 0.001. Locks boundary to "despawn".

### Antimatter & Pair Production

`p.antimatter` boolean. Right-click spawns antimatter (with negated charge and spin). Left-click on antimatter deletes it; right-click on matter deletes it. Symmetric: same-type click selects, opposite-type click deletes. Matter+antimatter merge annihilates lesser mass, emits photons via `emitPhotonBurst()`. Pair production: photons with energy >= 0.5 near massive body (dist < 8) can produce matter+antimatter pair (prob 0.005/substep, min age 64 substeps, max 32 particles). **Black Hole mode disables all antimatter** (see Black Hole Mode section).

## Sign Conventions (IMPORTANT)

All GEM interactions are **attractive** (gravity has one sign of "charge"):
- GM dipole: `+3*L1*L2/r^4` (positive = attractive)
- GM Boris parameter: `+2*Bgz` (co-moving masses attract)
- Bgz field: `-m_s*(v_s x r_hat)_z/r^2` (negative sign in code: `p.Bgz -= ...`)
- Frame-drag torque: positive coefficient drives spins toward co-rotation

Do NOT flip these signs.

### Angular Velocity Convention (Y-Down Canvas)

Canvas uses y-down coordinates. The 2D cross product `rx*vy - ry*vx` gives positive for clockwise rotation on screen. All angular quantities follow this convention: **positive angVel/angw = clockwise on screen**. Torque arcs and spin rings in the renderer use negated direction to convert from this convention to the canvas `arc()` anticlockwise parameter.

## Energy, PE & Collisions

**PE** (`potential.js`): Tree traversal via `treePE()` when BH on (divides by 2), exact pairwise `pairPE()` with i < j when off. Seven terms: gravitational, Coulomb (with axMod), magnetic dipole (with axMod), GM dipole, 1PN, Bazanski, Yukawa.

**Energy** (`energy.js`): Returns linearKE, spinKE, pe, fieldEnergy, pfiEnergy, momentum, angular momentum, COM, higgsFieldEnergy, axionFieldEnergy. Relativistic KE uses `wSq / (gamma + 1)`. `pfiEnergy` = particle-field interaction energy from Higgs (`-baseMass*(|phi|-1)`) and Axion (`-g*q²*a`, `∓g*m*a`), added to PE. Darwin field corrections when Magnetic/GM on but 1PN off. Conservation exact with gravity + Coulomb only, pairwise mode.

**Collisions**: Three modes -- pass (none), bounce (Hertz contact via `_applyRepulsion()`), merge (quadtree overlap detection, conserves mass/charge/momentum/angular momentum). `handleCollisions()` returns `{ annihilations, merges, removed }` -- integrator emits photons from annihilations, deposits field excitations from merges, and retires removed particles for signal delay fade-out.

## Topology

When boundary = "loop": Torus (both axes normal), Klein (y-wrap mirrors x, negates w.x/angw), RP^2 (both axes glide reflections, 4 min-image candidates). `minImage()` zero-alloc via `out` parameter. `sim.topology` is an integer constant (TORUS=0/KLEIN=1/RP2=2) from config.js; UI converts from string via `topoFromString()`.

## Barnes-Hut

`QuadTreePool`: SoA flat typed arrays, pre-allocated 512 nodes (doubles via `_grow()`). Zero GC. BH_THETA = 0.5, QUADTREE_CAPACITY = 4. Off by default. Aggregates: totalMass, totalCharge, totalMagneticMoment, totalAngularMomentum, totalMomentumX/Y, comX/comY.

## Toggle Dependencies

```
Forces:                        Physics:
  Gravity                        Relativity          [signal delay auto-activates]
    -> Gravitomagnetic             -> 1PN             [requires Magnetic, GM, or Yukawa]
    (+ tidal locking, always)      -> Black Hole      [+Gravity, locks collision to Merge]
  Coulomb                        Spin-Orbit           [requires Magnetic or GM]
    -> Magnetic                  Radiation             [requires Gravity, Coulomb, or Yukawa]
  Yukawa               [independent]   Larmor + EM quad   [when Coulomb on]
  Axion                [requires Coulomb or Yukawa]    GW quad [when Gravity on]
    aF² channel (when Coulomb on)  Pion emission      [when Yukawa on]
    PQ channel  (when Yukawa on)
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

Topbar: Home | Brand "No-Hair" | Pause/Step/Reset/Save/Load | Theme | Panel toggle.

19 presets in 4 `<optgroup>` categories: Gravity (6), Electromagnetism (3), Exotic (8), Cosmological (2). First 9 via keyboard `1`-`9`. Sim speed range 1-64, default 32.

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
- `compute1PN()` zeroes `force1PN` before accumulating -- do not mix with `pairForce()` 1PN output in same step
- 1PN velocity-Verlet correction rebuilds tree at post-drift positions when BH is on (extra O(N log N) build per substep)
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
- PQS stencil extends to `[ix-1..ix+2]`; `interpolate()` and `gradient()` use `_nb()` for topology-aware boundary handling (not `_fieldAt()` clamping); `_depositPQS()` also uses `_nb()` for topology wrapping
- Higgs `modulateMasses()` updates radius/radiusSq/invMass inline (not via `updateColor()`) to avoid per-substep string allocation
- `baseMass` must be saved/loaded and proportionally scaled wherever `mass` is modified
- `_fieldEnergy()` in ScalarField base handles KE+gradient+potential grid integration; subclasses pass potential lambda
- Higgs `energy()` shifts potential by +mu^2/4 so V(1)=0
- Higgs/Axion field reset on preset load and clear; Higgs mass -> baseMass on toggle-off; Axion axMod/yukMod -> 1 on toggle-off
- Axion `p.axMod` is per-particle (interpolated from local field), not global -- used in `pairForce()`/`pairPE()`
- Axion `p.axMod` clamped >= 0 -- without this, EM force sign reversal causes runaway acceleration
- Axion `p.yukMod` is per-particle PQ modulation for Yukawa -- `1 + g*a` for matter, `1 - g*a` for antimatter, clamped >= 0
- PQ source/force use `±g*m` (sign flip for antimatter); EM source/force use `g*q²` (same for both) -- combined into single `coupling` in `applyForces()`
- `magMoment`/`angMomentum` cache reflects previous `computeAllForces()` state -- consistent with B-field gradients used in same substep
- Ghost particles must carry `magMoment`/`angMomentum` fields (set in `_addGhost()`)
- Both MasslessBoson and Pion use shared `treeDeflectBoson()` from `boson-utils.js` for BH tree lensing; fall back to O(N) when pool is null or root < 0
- MasslessBoson passes grFactor=2 (null geodesic), Pion passes grFactor=1+v² (massive particle)
- Pion decay uses `sim._MasslessBosonClass` reference (set in main.js) to avoid circular import with massless-boson.js
- Pion decay products inherit the pion's `emitterId`, so the original emitter can never reabsorb them
- Self-absorption is permanently blocked for both photons and pions: `emitterId` match always skips absorption
- `_yukawaRadAccum` on Particle accumulates pion emission energy -- reset to 0 after each emission
- Field excitation `depositExcitation()` writes to `fieldDot` (not `field`) -- wave equation propagates naturally
- `sim.pions` array must be cleared on preset load and reset (in main.js, save-load.js, ui.js)
- `sim.deadParticles` must be cleared on preset load, clear, and save-load (same locations as pions)
- Dead particles use `_deathMass` (not `mass`) in force/heatmap code -- merged particles have mass=0 but `_deathMass` preserves the pre-merge value
- `_retireParticle()` must be called BEFORE the particle is removed from the array (needs valid pos/vel for final history snapshot)
- Collision code saves `_deathMass` before `resolveMerge()` zeroes mass -- `_retireParticle` uses `p.mass > 0` check to decide whether to overwrite
- Annihilation photon energy includes KE of the annihilated mass fraction (weighted by `annihilated/p.mass`), not just rest mass `2*annihilated`
- Dead particles skip backward extrapolation in signal-delay solver (deathTime < Infinity guard) -- prevents spurious solutions when true retarded time is past death
- Dead particles carry `_deathAngVel` (cached by `_retireParticle()`) -- used for magnetic/GM dipole forces in signal-delay path
- QuadTree `insert()` has depth guard (max 48) to prevent stack overflow from coincident particles
- `_computeLaplacian()` splits into interior fast path (direct ±1/±GRID indexing) and border path (uses `_nb()`) for performance
- Heatmap `update()` takes `gravityEnabled`/`coulombEnabled` params to skip disabled force contributions
- Collisions use relativistic KE (`wSq / (gamma+1) * mass`) for merge energy tracking, not `0.5*m*v²`
- Field excitation energy split between Higgs and Axion by coupling-weighted ratio `g²_H/(g²_H + g²_A)` when both are active, not 50/50
- Signal delay `histAngW` buffer records proper angular velocity for retarded dipole moment interpolation -- must be saved/restored alongside `histX`/`histY`/`histVX`/`histVY`
- `pairForce()` signal-delayed path recomputes `sMagMoment`/`sAngMomentum` from retarded `angw` (not current cached values)
- External field trig (`cos`/`sin` of angle sliders) cached once per frame via `_cacheExternalFields()`
