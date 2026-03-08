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
main.js                  419 lines  Simulation class, fixed-timestep loop, pair production, pion loop, window.sim
index.html               476 lines  UI: 4-tab sidebar, reference overlay, zoom controls, field sliders
styles.css               269 lines  Project-specific CSS overrides, toggle/slider theme colors
colors.js                 18 lines  Project color tokens (extends shared-tokens.js)
src/
  integrator.js         1500 lines  Physics: Boris substep loop, radiation, pion emission/absorption, field excitations,
                                     tidal, GW quadrupole, expansion, Roche, external fields, Hertz bounce, scalar fields
  forces.js              760 lines  pairForce(), computeAllForces(), calculateForce() (BH walk), compute1PN(), boson gravity,
                                     PE accumulator (resetPEAccum/getPEAccum)
  reference.js           714 lines  REFERENCE object: physics reference content (KaTeX math)
  presets.js             688 lines  PRESETS (19 scenarios, 4 groups), loadPreset(), SLIDER_MAP, TOGGLE_MAP/TOGGLE_ORDER
  scalar-field.js        660 lines  ScalarField base: PQS grid, topology-aware deposition, Laplacian, C² gradients,
                                     field energy, excitations, particle-field gravity, self-gravity (8×8 coarse)
  renderer.js            532 lines  Canvas 2D: particles, trails, spin rings, ergosphere, vectors, photons, pions, fields
  ui.js                  521 lines  setupUI(), declarative dependency graph, info tips, reference overlay, shortcuts
  heatmap.js             309 lines  64x64 potential field overlay, signal-delayed positions, force-toggle-aware
  axion-field.js         297 lines  AxionField: quadratic potential, aF² EM coupling, PQ pseudoscalar coupling
  higgs-field.js         296 lines  HiggsField: Mexican hat potential, mass modulation, thermal phase transitions
  quadtree.js            274 lines  QuadTreePool: SoA flat typed arrays, pool-based, zero GC, depth guard
  signal-delay.js        257 lines  getDelayedState() (3-phase light-cone solver, creationTime/deathTime guards)
  input.js               249 lines  InputHandler: mouse/touch, left/right-click symmetry (matter/antimatter)
  effective-potential.js 214 lines  V_eff(r) sidebar canvas, auto-scaling, axMod/yukMod modulation
  potential.js           211 lines  computePE(), treePE(), pairPE() (7 PE terms)
  save-load.js           205 lines  saveState(), loadState(), downloadState(), uploadState(), quickSave/Load()
  energy.js              191 lines  KE, spin KE, PE, field energy, momentum, angular momentum
  pion.js                187 lines  Massive Yukawa force carrier: proper velocity, (1+v²) GR deflection, decay channels
  config.js              157 lines  Named constants, mode enums (COL_*/BOUND_*/TORUS/KLEIN/RP²), helpers
  collisions.js          142 lines  handleCollisions(), resolveMerge(), annihilation, relativistic merge KE
  particle.js            132 lines  Particle: pos, vel, w, angw, baseMass, 11 force Vec2s, signal delay history
  topology.js            131 lines  minImage(), wrapPosition() for Torus/Klein/RP²
  phase-plot.js          128 lines  Phase space r-v_r plot (512-sample ring buffer)
  stats-display.js       123 lines  Sidebar energy/momentum/drift readout (×100 display scale)
  vec2.js                 61 lines  Vec2 class: set, clone, add, sub, scale, mag, normalize, dist
  boson-utils.js          58 lines  treeDeflectBoson(): shared BH tree walk for photon/pion lensing
  massless-boson.js       45 lines  MasslessBoson: pos, vel, energy, type ('em'/'grav'), BH tree lensing
  relativity.js           22 lines  angwToAngVel(), setVelocity()
```

## Key Imports

```
main.js       <- Physics, Renderer, InputHandler, Particle, HiggsField, AxionField,
                 Heatmap, PhasePlot, EffectivePotentialPlot, StatsDisplay, setupUI, config, MasslessBoson, Pion, save-load

integrator.js <- QuadTreePool, config, MasslessBoson, Pion, angwToAngVel,
                 forces (resetForces/computeAllForces/compute1PN/computeBosonGravity),
                 handleCollisions, computePE, topology

forces.js     <- config, getDelayedState, topology
energy.js     <- config, topology (window.sim for fields)
potential.js  <- config, topology
scalar-field.js <- config, topology (minImage for gravity)
higgs-field.js  <- config, ScalarField
axion-field.js  <- config, ScalarField
boson-utils.js  <- config (BH_THETA, BOSON_SOFTENING_SQ)
massless-boson.js <- Vec2, config, boson-utils
pion.js         <- Vec2, config, boson-utils
```

## Physics Engine

### Units & State Variables

c = G = ħ = 1. All velocities are fractions of c.

Both linear and rotational state use proper-velocity (celerity):

| State | Derived | Formula | Cap |
|---|---|---|---|
| `p.w` (γv) | `p.vel` | v = w / √(1 + w²) | \|v\| < c |
| `p.angw` | `p.angVel` | ω = W / √(1 + W²r²) | surface vel < c |

When relativity is off: `vel = w`, `angVel = angw` (identity).

Key derived quantities (INERTIA_K = 0.4, MAG_MOMENT_K = 0.2):
- Moment of inertia: `I = 0.4mr²`
- Magnetic moment: `μ = 0.2qωr²` — cached as `p.magMoment`
- Angular momentum: `L = Iω` — cached as `p.angMomentum`
- Particle radius: `r = ∛(mass)`; BH mode: `kerrNewmanRadius()` in config.js

`magMoment`/`angMomentum` cached per particle at start of `computeAllForces()`. Used by `pairForce()`, `pairPE()`, BH leaf walks, spin-orbit, display. Ghost particles carry these cached fields.

### Per-Particle Force Vectors

11 Vec2s reset each substep via `resetForces()`: `forceGravity`, `forceCoulomb`, `forceMagnetic`, `forceGravitomag`, `force1PN`, `forceSpinCurv`, `forceRadiation`, `forceYukawa`, `forceExternal`, `forceHiggs`, `forceAxion`.

3 torque scalars: `torqueSpinOrbit`, `torqueFrameDrag`, `torqueTidal`.

### Boris Integrator

Per substep (inside `Physics.update()` while loop):

1. Store `_f1pnOld` (if 1PN enabled)
2. **Half-kick**: `w += F/m · dt/2` (E-like forces)
3. **Boris rotation**: rotate w in combined Bz + Bgz + extBz plane (preserves |v| exactly)
4. **Half-kick**: `w += F/m · dt/2`
5. Spin-orbit energy coupling, Stern-Gerlach/Mathisson-Papapetrou kicks, frame-drag torque
6. Radiation reaction (Landau-Lifshitz)
7. Pion emission (scalar Larmor, when Yukawa enabled) + radiation reaction on emitter
8. **Drift**: `vel = w / √(1 + w²)`, `pos += vel · dt`
9. Cosmological expansion (if enabled)
10. **1PN velocity-Verlet correction**: rebuild tree (if BH on), recompute 1PN at new positions, kick `w += (F_new - F_old) · dt/(2m)`. VV tree reused for step 12 when BH+1PN both on.
11. **Scalar fields**: evolve Higgs/Axion (Störmer-Verlet KDK), modulate masses, interpolate axMod
12. Reuse VV tree or rebuild quadtree, collisions (annihilation + merge KE tracking), repel, photon/pion absorption
13. Deposit field excitations from merge KE into active scalar fields
14. External fields, Higgs/Axion gradient forces, sync axMod, reset + recompute forces

After all substeps: record signal-delay history (strided, HISTORY_STRIDE=64), read cached PE from force-loop accumulator (`getPEAccum()`), reconstruct velocity-dependent display forces.

**Adaptive substepping**: `dtSafe = min(√(softening/a_max), (2π/ω_c)/8)` where `ω_c = max(|qBz/m|, 4|Bgz|, |q·extBz/m|)`. Capped at MAX_SUBSTEPS = 32.

**Fixed-timestep loop**: PHYSICS_DT = 1/128. Accumulator collects `rawDt × speedScale`, drained in fixed chunks.

## Force Types

### E-like Forces (radial)

Plummer softening: SOFTENING = 8 (SQ = 64); BH mode: BH_SOFTENING = 4 (SQ = 16).

| Force | Formula | PE | Toggle |
|---|---|---|---|
| Gravity | `+m₁m₂/r²` (attractive) | `-m₁m₂/r` | Gravity |
| Coulomb | `-q₁q₂/r²` (like-repels) | `+q₁q₂/r` | Coulomb |
| Magnetic dipole | `+3μ₁μ₂/r⁴` | `+μ₁μ₂/r³` | Coulomb + Magnetic |
| GM dipole | `+3L₁L₂/r⁴` (co-rotating attract) | `-L₁L₂/r³` | Gravity + GM |

### B-like Forces (velocity-dependent, Boris rotation)

**Lorentz** (Coulomb + Magnetic): Bz from moving charge + spinning dipole. Display: `forceMagnetic += (q·vy·Bz, -q·vx·Bz)`.

**Gravitomagnetic** (Gravity + GM): Bgz from moving/spinning mass. Boris parameter: `+2Bgz·dt/γ`. Display: `forceGravitomag += (4m·vy·Bgz, -4m·vx·Bgz)`.

**Frame-dragging torque**: `τ = 2L_s(ω_s - ω_p)/r³`. Drives spin alignment.

### Tidal Locking

Always active when Gravity on (no toggle). `τ = -0.3 · coupling² · r_body⁵/r⁶ · (ω_spin - ω_orbit)`.

### Yukawa Potential

Independent toggle. `F = -g²m₁m₂e^{-μr}/r² · (1+μr)`. Parameters: `yukawaCoupling` (14), `yukawaMu` (0.15, slider 0.05–0.25). Includes analytical jerk for radiation. Emits pions as massive force carriers.

**Scalar Breit correction** (requires 1PN): O(v²/c²) from massive scalar boson exchange. `δH = g²m₁m₂e^{-μr}/(2r) · [v₁·v₂ + (r̂·v₁)(r̂·v₂)(1+μr)]`. Into `force1PN`, velocity-Verlet corrected.

### External Background Fields

No toggle — controlled by slider values (default 0). Gravity (`F = mg`), Electric (`F = qE`), Magnetic (uniform Bz via Boris rotation). Direction angle sliders auto-show when strength > 0.

### Bounce (Hertz Contact)

`F = K · δ^{1.5}` (K=1). Tangential friction transfers torque. Quadtree-accelerated when BH on, O(n²) fallback when off — do not early-return when `root < 0`.

## Scalar Fields

### Base Class (`ScalarField`)

PQS (cubic B-spline, order 3) grid on 64×64. 4×4 stencil per particle. C² interpolation and C² gradients (PQS-interpolated central-difference grid gradients). Pre-allocated weight arrays for zero-alloc hot path.

Key methods: `_nb()` (topology-aware neighbor, absolute coords), `_depositPQS()` (interior fast path + border fallback), `_computeLaplacian()` (interior fast path + border path), `_computeGridGradients()`, `interpolate()`, `gradient()`, `_fieldEnergy()`, `depositExcitation()`, `_computeEnergyDensity()`, `applyGravForces()`, `gravPE()`, `computeSelfGravity()`.

**Particle-field gravity** (requires `gravityEnabled`): Field energy density gravitates particles via direct O(N×GRID²) summation. Each cell is a point mass `ρ·dA`. Only excitations gravitate (ρ=0 at vacuum). Subclasses override `_addPotentialEnergy()` for V(φ). Call `applyGravForces()` AFTER field `update()` (needs current `_gradX`/`_gradY`).

**Field self-gravity** (requires `gravityEnabled`): Weak-field GR correction to Klein-Gordon: `φ̈ = (1+4Φ)∇²φ + 2∇Φ·∇φ - (1+2Φ)V'(φ)`. Φ from field energy density via coarse 8×8 grid O(SG⁴≈4K), bilinear-upsampled to 64×64. Φ computed once per `update()`. `∇Φ·∇φ` cross-term uses stale `_gradX/_gradY` (error O(dt²Φ)).

Field arrays: `field`/`fieldDot` (not `phi`/`phiDot`). Field clamp: SCALAR_FIELD_MAX = 2.

### Higgs Field

Independent toggle. Mexican hat `V(φ) = -½μ²φ² + ¼λφ⁴`. VEV=1; `λ = μ² = m_H²/2`. Slider: m_H 0.25–0.75 (default 0.50).

- **Mass generation**: `m_eff = baseMass · max(|φ(x)|, 0.05)`. Floor caps gradient-force acceleration at 20×.
- **Gradient force**: `F = +g · baseMass · sign(φ) · ∇φ` (g = HIGGS_COUPLING = 1). Into `forceHiggs`.
- **Field equation**: `φ̈ = ∇²φ + μ²_eff·φ - μ²φ³ + source/cellArea - 2m_H·φ̇`. Störmer-Verlet KDK.
- **Phase transitions**: `μ²_eff = μ² - KE_local`. High local KE → symmetric phase (φ→0).
- **Boundary**: Despawn→Dirichlet (φ=1), Bounce→Neumann, Loop→periodic.
- **Energy**: `_fieldEnergy()` with Mexican hat potential, shifted so V(1)=0.
- **baseMass sync**: All mass-modifying operations proportionally scale baseMass. Toggle-off restores mass to baseMass.

### Axion Field

Independent toggle; requires Coulomb or Yukawa. Quadratic `V(a) = ½m_a²a²`, vacuum at a=0. Slider: m_a 0.01–0.10 (default 0.05).

**Scalar EM coupling (aF², when Coulomb on)**: Same for matter/antimatter.
- Source: `g·q²` (g = AXION_COUPLING = 0.05). EM modulation: `α_eff = α(1+g·a)`. Per-particle `p.axMod`, clamped ≥ 0. Pairwise: geometric mean `√(axMod_i · axMod_j)`.

**PQ coupling (when Yukawa on)**: Flips sign for antimatter.
- Source: `±g·m`. Yukawa modulation: `g²_eff = g²·yukMod`. `yukMod = 1+g·a` (matter), `1-g·a` (antimatter), clamped ≥ 0. At vacuum (a=0): yukMod=1 → CP conserved.

Both channels: gradient force `F = coupling · ∇a` into `forceAxion`. Damping: ζ=g/2, Q=1/g, g·Q=1.

## Pions (Massive Force Carriers)

Pion class in `pion.js`. Mass = `yukawaMu`. Proper velocity `w`: `vel = w/√(1+w²)`. GR deflection: `(1+v²)` factor.

**Emission**: Scalar Larmor `P = g²F_yuk²/3`. Accumulated in `p._yukawaRadAccum`. Species: π⁰ (50%), π⁺/π⁻ (25% each). MAX_PIONS = 256. Radiation reaction rescales emitter `w` exactly.

**Decay**: π⁰→2γ (half-life 32), π⁺→e⁺+γ (half-life 128), π⁻→e⁻+γ (half-life 128). Two-body kinematics in rest frame, Lorentz-boosted. Decay products inherit `emitterId`. Uses `sim._MasslessBosonClass` to avoid circular import.

**Absorption**: Quadtree overlap query. Transfers momentum + charge. Self-absorption permanently blocked by `emitterId`.

**Boson gravity** (requires Gravity): Boson→particle O(N×N_bosons) into `forceGravity`. Boson↔boson O(N_bosons²) mutual gravity with correct GR deflection factors (2 for photons, 1+v² for pions). Both use `BOSON_SOFTENING_SQ = 4`.

## Field Excitations

Merge KE deposits Gaussian wave packets into active scalar fields via `depositExcitation()` (writes to `fieldDot`, propagated by Klein-Gordon). Amplitude: `0.5·√(keLost)`. σ = 2 grid cells. Split between Higgs/Axion by coupling-weighted ratio when both active.

## Advanced Physics

### 1PN Corrections

Requires Relativity. Four O(v²/c²) sectors into `force1PN`:
- **EIH** (GM + 1PN): perihelion precession. Requires `gravitomagEnabled`.
- **Darwin EM** (Magnetic + 1PN): EM remainder. Requires `magneticEnabled`.
- **Bazanski** (GM + Magnetic + 1PN): mixed 1/r³. Requires both.
- **Scalar Breit** (Yukawa + 1PN): massive scalar exchange. Requires `yukawaEnabled`.

NOT Newton's 3rd law. Velocity-Verlet: stores `_f1pnOld` → drift → rebuild tree → recompute → kick. `compute1PN()` zeroes `force1PN` before accumulating — do not mix with `pairForce()` 1PN output.

### Radiation

Requires Gravity, Coulomb, or Yukawa. Single toggle, four mechanisms:
- **Larmor dipole** (Coulomb): Landau-Lifshitz. Analytical jerk + numerical backward difference. Clamped ≤ 0.5|F_ext|. Power-dissipation terms require relativity.
- **EM quadrupole** (Coulomb): `P = (1/180)|d³Q_ij/dt³|²`. Emits photons (type: 'em').
- **GW quadrupole** (Gravity): `P = (1/5)|d³I^TF_ij/dt³|²`. Emits gravitons (type: 'grav', red).
- **Pion emission** (Yukawa): `P = g²F_yuk²/3`. Emits pions.

Self-absorption permanently blocked by `emitterId` for both photons and pions.

### Black Hole Mode

Requires Gravity + Relativity. Locks collision to Merge.
- **No hair**: Antimatter erased. `addParticle()` blocks antimatter. Pair production disabled. Charged pion decay products forced to matter.
- **Kerr-Newman**: `r₊ = M + √(M² - a² - Q²)`, `a = INERTIA_K·r²·|ω|`, naked singularity floor.
- **Hawking** (requires Radiation): `T = κ/(2π)`, `P = σT⁴A`. Uses `∛(mass)²` (not stale `radiusSq`). Extremal BHs stop. Evaporation → photon burst.
- BH_SOFTENING_SQ = 16. Ergosphere at `r_ergo = M + √(M² - a²)` (visual only).

### Signal Delay

Auto-activates with Relativity. Per-particle circular history buffers (Float64Array[256], recorded every HISTORY_STRIDE=64 `update()` calls):
1. Newton-Raphson segment search (≤8 iterations)
2. Exact quadratic solve on converged segment
3. Constant-velocity extrapolation for early history (skipped for dead particles)

**Causality**: `creationTime` rejects extrapolation past creation. Dead particles (`_retireParticle()` → `sim.deadParticles[]`) continue exerting forces via signal delay, using `_deathMass`/`_deathAngVel`. Garbage-collected when `simTime - deathTime > 2·domain_diagonal`.

**Liénard-Wiechert aberration**: `(1 - n̂·v_source)^{-3}`, clamped [0.01, 100]. Applied to gravity, Coulomb, Yukawa, dipole. Not 1PN (already O(v²)). Retarded angw interpolated from `histAngW`.

Dead particles: always pairwise (even when BH on), excluded from `compute1PN()`. All reset paths clear `deadParticles`. `_retireParticle()` must be called BEFORE array removal.

### Spin-Orbit Coupling

Requires Magnetic + GM + Spin-Orbit toggle. Stern-Gerlach `F = +μ·∇(Bz)`, Mathisson-Papapetrou `F = -L·∇(Bgz)` (GEM flip). Into `forceSpinCurv`.

### Disintegration & Roche

Requires Gravity. Locks collision to Merge. Tidal + centrifugal + Coulomb stress vs self-gravity → SPAWN_COUNT (4) fragments. Roche: Eggleton (1983) formula, continuous L1 mass transfer.

### Cosmological Expansion

`pos += H(pos - center)dt`, `w *= (1-Hdt)`. Default H = 0.001. Locks boundary to despawn.

### Antimatter & Pair Production

Right-click spawns antimatter (negated charge/spin). Same-type click selects, opposite-type deletes. Annihilation emits photons via `emitPhotonBurst()`. Pair production: photons with energy ≥ 0.5 near massive body (dist < 8, prob 0.005/substep, min age 64, max 32 particles). BH mode disables all antimatter.

## Sign Conventions (IMPORTANT)

All GEM interactions are **attractive** (gravity has one sign of "charge"):
- GM dipole: `+3L₁L₂/r⁴` (positive = attractive)
- GM Boris parameter: `+2Bgz` (co-moving masses attract)
- Bgz field: `-m_s(v_s × r̂)_z/r²` (negative sign in code: `p.Bgz -= ...`)
- Frame-drag torque: positive drives co-rotation

Do NOT flip these signs.

**Angular velocity convention (y-down canvas)**: 2D cross product `rx·vy - ry·vx` gives positive for clockwise on screen. All angular quantities follow this. Renderer negates direction for canvas `arc()`.

## Energy, PE & Collisions

**PE** (primary: `forces.js` accumulator; fallback: `potential.js`): PE accumulated inline during `pairForce()` via module-level `_peAccum` (reset by `resetPEAccum()`, read by `getPEAccum()`). 9 terms: gravitational, Coulomb (axMod), magnetic dipole (axMod), GM dipole, 1PN EIH, Darwin EM, Bazanski, Yukawa, Scalar Breit. Dead-particle pairs excluded (`_accumulatePE = false`). `potential.js` (`treePE()`/`pairPE()`) kept as fallback for preset-load recomputation.

**Energy** (`energy.js`): Relativistic KE = `wSq/(γ+1)·mass`. `pfiEnergy` = particle-field interaction from Higgs + Axion, added to PE. Conservation exact with gravity + Coulomb, pairwise only.

**Collisions**: Pass / bounce (Hertz) / merge. `handleCollisions()` returns `{ annihilations, merges, removed }`. Merge uses relativistic KE. Integrator emits photons from annihilations, deposits field excitations, retires removed particles.

## Topology

Boundary "loop": Torus (TORUS=0), Klein bottle (KLEIN=1, y-wrap mirrors x), RP² (RP2=2, both glide reflections). `minImage()` zero-alloc via `out` parameter.

## Barnes-Hut

`QuadTreePool`: SoA typed arrays, 512-node pool (grows via `_grow()`). BH_THETA = 0.5, QUADTREE_CAPACITY = 4. Depth guard max 48. Aggregates: totalMass, totalCharge, totalMagneticMoment, totalAngularMomentum, totalMomentumX/Y, comX/comY.

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

Declarative `DEPS` array in `ui.js`, topological evaluation via `updateAllDeps()`.

Defaults on: gravity, coulomb, magnetic, gravitomag, 1PN, relativity, spin-orbit, radiation. Defaults off: Yukawa, Axion, Higgs, Disintegration, Expansion, Barnes-Hut, Black Hole.

## UI

4-tab sidebar: Settings (mass/charge/spin, spawn mode, force/physics toggles), Engine (BH, collisions, boundary/topology, external fields, visuals, speed), Stats (energy/momentum/drift), Particle (selected details, force breakdown, phase plot, effective potential).

Topbar: Home | "No-Hair" | Pause/Step/Reset/Save/Load | Theme | Panel toggle.

19 presets in 4 groups: Gravity (6), Electromagnetism (3), Exotic (8), Cosmological (2). First 9 via keyboard `1`–`9`. Speed: 1–64, default 32.

## Renderer

Canvas 2D. Dark mode: additive blending (`lighter`). WORLD_SCALE = 16. Camera starts at zoom = WORLD_SCALE. Viewport culling: particles, photons, pions skip draw when outside camera bounds (`_vpLeft/Right/Top/Bottom`). Field overlays throttled to every FIELD_RENDER_INTERVAL (2) frames.

- **Particles**: r = ∛(mass) (BH: r₊), glow in dark. Neutral=slate. Charged: RGB lerp red(+)/blue(-), intensity=|q|/5.
- **Trails**: circular Float32Array[256], wrap-detection for periodic boundaries
- **Force vectors**: gravity=red, coulomb=blue, magnetic=cyan, GM=rose, 1PN=orange, spin-curv=purple, radiation=yellow, yukawa=green, external=brown, higgs=lime, axion=indigo
- **Field overlays**: 64×64 offscreen, bilinear-upscaled. Higgs: purple(depleted)/lime(enhanced). Axion: indigo(+)/yellow(-).
- **Photons**: yellow (EM) / red (grav), alpha fades over PHOTON_LIFETIME=256
- **Pions**: green, glow in dark, constant alpha (decay is probabilistic)
- **V_eff plot**: 200-sample sidebar canvas

## Key Patterns

- `window.sim` for console debugging. `_PALETTE`/`_FONT` frozen by colors.js
- `Vec2.set(x,y)` in hot paths; `pairForce()` accumulates into `out` Vec2, zero alloc
- Module-level `_miOut` for zero-alloc `minImage()` output
- Particle constructor declares all properties upfront (V8 hidden class stability)
- World coordinates: `sim.domainW/H` (viewport / WORLD_SCALE), not pixels
- Theme: `data-theme` on `<html>` (not body)
- `.mode-toggles` sets `display: grid` overriding `hidden` — use `style.display`
- External field trig cached once per frame via `_cacheExternalFields()`
- `forceRadiation` cleared for all particles before substep loop (stale prevention)
- History recording counts `update()` calls, not substeps
- `sim.pions`, `sim.deadParticles` must be cleared on preset load, clear, and save-load
- PE accumulated inline in `pairForce()` via `_peAccum`; `potential.js` is fallback only (preset loads)
- Display throttles: STATS_THROTTLE_MASK=7 (energy, 8th frame), SIDEBAR_THROTTLE_MASK=1 (phase/effpot/selected, 2nd frame), FIELD_RENDER_INTERVAL=2, HEATMAP_INTERVAL=4
