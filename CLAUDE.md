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
main.js                  801 lines  Simulation class, fixed-timestep loop, backend selection (CPU/GPU),
                                     pair production, pion loop, dirty-flag render, window.sim
index.html               494 lines  UI: 4-tab sidebar, reference overlay, zoom controls, field sliders
styles.css               295 lines  Project-specific CSS overrides, toggle/slider theme colors
colors.js                 18 lines  Project color tokens (extends shared-tokens.js)
src/
  integrator.js         1565 lines  CPU physics: Boris substep loop, radiation, pion emission/absorption,
                                     field excitations, tidal, GW quadrupole, expansion, Roche, external fields,
                                     Hertz bounce, scalar fields
  forces.js              794 lines  CPU pairForce(), computeAllForces(), calculateForce() (BH walk), compute1PN(),
                                     boson gravity, PE accumulator (resetPEAccum/getPEAccum), inline torus minImage
  reference.js           714 lines  REFERENCE object: physics reference content (KaTeX math)
  presets.js             680 lines  PRESETS (19 scenarios, 4 groups), loadPreset(), SLIDER_MAP, TOGGLE_MAP/TOGGLE_ORDER
  scalar-field.js        858 lines  ScalarField base: PQS grid, topology-aware deposition, Laplacian, C² gradients,
                                     field energy, excitations, particle-field gravity, self-gravity (8×8 coarse),
                                     fused interpolateWithGradient(), interior fast-path PQS, self-gravity early exit
  renderer.js            729 lines  CPU Canvas 2D renderer (used as fallback when GPU unavailable)
  ui.js                  609 lines  setupUI(), declarative dependency graph, info tips, reference overlay, shortcuts,
                                     dirty flag, KaTeX render cache, lazy field init triggers
  heatmap.js             315 lines  64x64 potential field overlay, signal-delayed positions, force-toggle-aware
  higgs-field.js         309 lines  HiggsField: Mexican hat potential, mass modulation, thermal phase transitions
  axion-field.js         299 lines  AxionField: quadratic potential, aF² EM coupling, PQ pseudoscalar coupling
  quadtree.js            348 lines  QuadTreePool: SoA flat typed arrays, pool-based, zero GC, depth guard,
                                     iterative insert, direct quadrant child selection, boson distribution
  input.js               307 lines  InputHandler: mouse/touch, left/right-click symmetry (matter/antimatter)
  signal-delay.js        260 lines  getDelayedState() (3-phase light-cone solver, creationTime/deathTime guards)
  effective-potential.js 244 lines  V_eff(r) sidebar canvas, auto-scaling, axMod/yukMod modulation, dirty-flag skip
  pion.js                236 lines  Massive Yukawa force carrier: proper velocity, (1+v²) GR deflection, decay, pool
  save-load.js           229 lines  saveState(), loadState(), downloadState(), uploadState(), quickSave/Load()
  potential.js           211 lines  computePE(), treePE(), pairPE() (7 PE terms)
  energy.js              191 lines  KE, spin KE, PE, field energy, momentum, angular momentum
  config.js              166 lines  Named constants, mode enums (COL_*/BOUND_*/TORUS/KLEIN/RP²), helpers
  collisions.js          152 lines  handleCollisions(), resolveMerge(), annihilation, relativistic merge KE
  stats-display.js       138 lines  Sidebar energy/momentum/drift readout, textContent change detection
  phase-plot.js          137 lines  Phase space r-v_r plot (512-sample ring buffer)
  particle.js            135 lines  Particle: pos, vel, w, angw, baseMass, 11 force Vec2s, signal delay history
  topology.js            131 lines  minImage(), wrapPosition() for Torus/Klein/RP²
  massless-boson.js       91 lines  MasslessBoson: pos, vel, energy, type ('em'/'grav'), BH tree lensing, pool
  vec2.js                 61 lines  Vec2 class: set, clone, add, sub, scale, mag, normalize, dist
  boson-utils.js          59 lines  treeDeflectBoson(): shared BH tree walk for photon/pion lensing
  backend-interface.js    57 lines  PhysicsBackend/RenderBackend typedefs, BACKEND_CPU/BACKEND_GPU constants
  cpu-physics.js          25 lines  CPUPhysics: thin adapter wrapping Physics (integrator.js) to PhysicsBackend
  relativity.js           25 lines  angwToAngVel(), setVelocity()
  canvas-renderer.js      20 lines  CanvasRenderer: thin adapter wrapping Renderer to RenderBackend
  gpu/
    gpu-physics.js      3070 lines  GPUPhysics: WebGPU compute pipeline orchestrator, addParticle/serialize,
                                     all dispatch methods, bind group creation, adaptive substepping, readback,
                                     per-field uniform buffers (Higgs/Axion), pre-allocated write buffers
    gpu-pipelines.js    1448 lines  Pipeline + bind group layout creation for all compute/render shaders
    gpu-renderer.js      977 lines  WebGPU instanced rendering: particles, bosons, field overlays, heatmap,
                                     trails, force arrows, spin rings (dual light/dark pipeline variants)
    gpu-buffers.js       592 lines  Buffer allocation: packed structs, quadtree, collision, field, history,
                                     trail buffers, staging
    gpu-constants.js     257 lines  buildWGSLConstants(): generates WGSL const block from config.js +
                                     _PALETTE colors, single source of truth for JS/WGSL constants
    shaders/
      common.wgsl        209 lines  Shared structs (SimUniforms, ParticleState, ParticleAux, ParticleDerived,
                                     AllForces, RadiationState, Photon, Pion), toggle bits, constants,
                                     pcgHash/pcgRand RNG, torusMinImage/fullMinImage topology helpers
      field-common.wgsl  111 lines  Shared field constants (GRID=64), FieldUniforms, nbIndex(), PQS weights
      --- Phase 2: Core physics (prepended with common.wgsl) ---
      reset-forces.wgsl        Zero AllForces accumulator
      cache-derived.wgsl       Compute ParticleDerived (magMoment, angMomentum, vel, radius)
      pair-force.wgsl          O(N²) tiled pairwise force (gravity, Coulomb, dipoles, Yukawa, 1PN, jerk)
      external-fields.wgsl     Uniform background gravity/electric/magnetic
      save-f1pn.wgsl           Save 1PN forces to f1pnOld buffer before Boris (for velocity-Verlet)
      boris-half-kick.wgsl     w += F/m · dt/2
      boris-rotate.wgsl        Boris rotation in B+Bg+extBz plane
      drift.wgsl               Drift: pos += vel·dt
      boris.wgsl               Combined drift + derive coordinate velocity
      spin-orbit.wgsl          Stern-Gerlach + Mathisson-Papapetrou + frame-drag
      apply-torques.wgsl       Angular acceleration from torque accumulator
      --- Phase 3: Spatial partitioning (standalone, own structs) ---
      ghost-gen.wgsl           Periodic boundary ghost particles (Torus/Klein/RP²)
      tree-build.wgsl          Barnes-Hut quadtree: bounds, init, insert, aggregates
      forces-tree.wgsl         BH tree walk force computation (O(N log N))
      collision.wgsl           Broadphase detection + merge/annihilation resolve
      dead-gc.wgsl             Dead particle garbage collection (free stack)
      boundary.wgsl            Despawn/bounce/wrap boundary conditions
      --- Phase 4: Advanced physics (standalone, own structs) ---
      radiation.wgsl           Larmor, Hawking, pion emission (3 entry points)
      bosons.wgsl              Photon/pion drift, lensing, absorption, decay (5 entry points)
      boson-tree.wgsl          Boson BH tree: insert, aggregate, particle↔boson gravity
      history.wgsl             Signal delay history ring buffer recording
      onePN.wgsl               1PN compute + velocity-Verlet correction kick
      --- Phase 5: Scalar fields & extras (field-common.wgsl or standalone) ---
      field-deposit.wgsl       PQS particle→grid deposition (scatter/gather)
      field-evolve.wgsl        Störmer-Verlet KDK: Laplacian, half-kick, drift, gradients
      field-forces.wgsl        Higgs mass modulation + axion axMod/yukMod + gradient forces
      field-selfgrav.wgsl      Coarse-grid self-gravity potential (8×8 → 64×64 upsample)
      field-excitation.wgsl    Gaussian wave packet deposition from merge events
      heatmap.wgsl             Potential field computation (gravity/electric/Yukawa)
      expansion.wgsl           Hubble flow + momentum drag
      disintegration.wgsl      Tidal + Roche breakup detection
      pair-production.wgsl     Photon → particle pair creation
      --- Render shaders (standalone, vertex+fragment) ---
      particle.wgsl            Instanced particle circles
      boson-render.wgsl        Photon/pion point rendering
      spin-render.wgsl         Angular velocity ring indicators
      trail-render.wgsl        Position history trail lines
      arrow-render.wgsl        Force vector arrows
      update-colors.wgsl       Particle charge→color mapping
      trails.wgsl              Trail history recording
      hit-test.wgsl            Mouse→particle selection
      field-render.wgsl        Scalar field 64×64 overlay
      heatmap-render.wgsl      Potential heatmap overlay
```

## Key Imports

```
main.js       <- Physics, Renderer, InputHandler, Particle, HiggsField, AxionField,
                 Heatmap, PhasePlot, EffectivePotentialPlot, StatsDisplay, setupUI, config,
                 MasslessBoson, Pion, save-load, relativity,
                 BACKEND_CPU/BACKEND_GPU (backend-interface), CPUPhysics, CanvasRenderer,
                 GPUPhysics, GPURenderer

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
save-load.js    <- BACKEND_GPU (backend-interface)

gpu-physics.js   <- gpu-buffers, gpu-pipelines, gpu-constants (buildWGSLConstants)
gpu-pipelines.js <- gpu-constants (buildWGSLConstants), fetchShader (loads .wgsl files)
gpu-renderer.js  <- gpu-pipelines (render pipeline creators)
gpu-constants.js <- config, _PALETTE (generates WGSL const block from JS constants + palette)
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

Key methods: `_nb()` (topology-aware neighbor, absolute coords), `_depositPQS()` (interior fast path + border fallback), `_computeLaplacian()` (interior fast path + border path), `_computeGridGradients()`, `interpolate()`, `gradient()`, `interpolateWithGradient()` (fused, single stencil walk), `_fieldEnergy()`, `depositExcitation()`, `_computeEnergyDensity()`, `applyGravForces()`, `gravPE()`, `computeSelfGravity()`.

**Particle-field gravity** (requires Gravity → Field Gravity toggle): Field energy density gravitates particles via direct O(N×GRID²) summation. Each cell is a point mass `ρ·dA`. Only excitations gravitate (ρ=0 at vacuum). Subclasses override `_addPotentialEnergy()` for V(φ). Call `applyGravForces()` AFTER field `update()` (needs current `_gradX`/`_gradY`). Default off.

**Field self-gravity** (requires Gravity → Field Gravity toggle): Weak-field GR correction to Klein-Gordon: `φ̈ = (1+4Φ)∇²φ + 2∇Φ·∇φ - (1+2Φ)V'(φ)`. Φ from field energy density via coarse 8×8 grid O(SG⁴≈4K), bilinear-upsampled to 64×64. Φ computed once per `update()`. `∇Φ·∇φ` cross-term uses stale `_gradX/_gradY` (error O(dt²Φ)).

Field arrays: `field`/`fieldDot` (not `phi`/`phiDot`). Field clamp: SCALAR_FIELD_MAX = 2.

### Higgs Field

Independent toggle. Mexican hat `V(φ) = -½μ²φ² + ¼λφ⁴`. VEV=1; `λ = μ² = m_H²/2`. Slider: m_H 0.25–0.75 (default 0.50).

- **Mass generation**: `m_eff = baseMass · max(|φ(x)|, 0.05)`. Floor caps gradient-force acceleration at 20×.
- **Mass rate clamp**: `HIGGS_MASS_MAX_DELTA = 4` — mass change per substep clamped to `±4·dt`. Prevents resonant oscillation where field source → φ overshoot → mass spike → velocity jitter.
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

Pion class in `pion.js`. Mass = `yukawaMu`. Proper velocity `w`: `vel = w/√(1+w²)`. GR deflection: `(1+v²)` factor. Cached `vSq`/`gravMass` via `_syncVel()` (refreshed after every `w` mutation).

**Emission**: Scalar Larmor `P = g²F_yuk²/3`. Accumulated in `p._yukawaRadAccum`. Species: π⁰ (50%), π⁺/π⁻ (25% each). MAX_PIONS = 256. Radiation reaction rescales emitter `w` exactly.

**Decay**: π⁰→2γ (half-life 32), π⁺→e⁺+γ (half-life 128), π⁻→e⁻+γ (half-life 128). Two-body kinematics in rest frame, Lorentz-boosted. Decay products inherit `emitterId`. Uses `sim._MasslessBosonClass` to avoid circular import.

**Absorption**: Quadtree overlap query. Transfers momentum + charge. Self-absorption permanently blocked by `emitterId`.

**Boson gravity** (requires Gravity + Barnes-Hut → Boson Gravity toggle): Particle→boson BH tree lensing, boson→particle O(N×log(N_bosons)) via BH tree walk, boson↔boson O(N_bosons×log(N_bosons)) mutual gravity with correct GR deflection factors (2 for photons, 1+v² for pions). All use `BOSON_SOFTENING_SQ = 4`. Default off.

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
    -> Field Gravity               -> Black Hole      [+Gravity, locks collision to Merge]
  Coulomb                        Spin-Orbit           [requires Magnetic or GM]
    -> Magnetic                  Radiation             [requires Gravity, Coulomb, or Yukawa]
  Gravity + Barnes-Hut             Larmor + EM quad   [when Coulomb on]
    -> Boson Gravity               GW quad            [when Gravity on]
    (+ tidal locking, always)      Pion emission      [when Yukawa on]
  Yukawa               [independent]
  Axion                [requires Coulomb or Yukawa]
    aF² channel (when Coulomb on)
    PQ channel  (when Yukawa on)
  Higgs                [independent]
Disintegration                   [requires Gravity, locks collision to Merge]
Barnes-Hut                       [independent]
Expansion                        [independent, in Engine tab]
```

Declarative `DEPS` array in `ui.js`, topological evaluation via `updateAllDeps()`.

Defaults on: gravity, coulomb, magnetic, gravitomag, 1PN, relativity, spin-orbit, radiation. Defaults off: Boson Gravity, Field Gravity, Yukawa, Axion, Higgs, Disintegration, Expansion, Barnes-Hut, Black Hole.

## UI

4-tab sidebar: Settings (mass/charge/spin, spawn mode, force/physics toggles), Engine (BH, collisions, boundary/topology, external fields, visuals, speed), Stats (energy/momentum/drift), Particle (selected details, force breakdown, phase plot, effective potential).

Topbar: Home | "No-Hair" | Pause/Step/Reset/Save/Load | Theme | Panel toggle.

19 presets in 4 groups: Gravity (6), Electromagnetism (3), Exotic (8), Cosmological (2). First 9 via keyboard `1`–`9`. Speed: 1–64, default 32.

## Backend Architecture

Two interchangeable backends selected at startup via `selectBackend()`. Backends conform to `PhysicsBackend`/`RenderBackend` interfaces defined in `backend-interface.js`.

- **CPU**: `CPUPhysics` (wraps `Physics` from integrator.js) + `CanvasRenderer` (wraps `Renderer`). Thin adapters — all logic stays in integrator.js and renderer.js.
- **GPU**: `GPUPhysics` (compute pipelines) + `GPURenderer` (instanced rendering). GPU overlays a separate `<canvas id="gpuCanvas">` with `alphaMode: 'premultiplied'`.

Falls back to CPU on WebGPU unavailability or device loss (with auto-save recovery). Force CPU via `?cpu=1` URL parameter.

## Renderer

### CPU Renderer (Canvas 2D)

Dark mode: additive blending (`lighter`). WORLD_SCALE = 16. Camera starts at zoom = WORLD_SCALE. Viewport culling: particles, photons, pions skip draw when outside camera bounds. Rendering batched: shadowBlur buckets, alpha buckets (photons), spin rings by sign, pion fills.

### GPU Renderer (WebGPU)

Instanced rendering via vertex shaders. Reads directly from GPU compute buffers (no readback). Separate render passes for particles, photons, pions, field overlays, heatmap, trails, force arrows, spin rings. All render pipelines have dual light/dark variants (premultiplied alpha vs additive blend).

### Visual Style (both backends)

- **Particles**: r = ∛(mass) (BH: r₊), glow in dark. Neutral=slate. Charged: RGB lerp red(+)/blue(-), intensity=|q|/5.
- **Trails**: circular Float32Array[256], wrap-detection for periodic boundaries
- **Force vectors**: gravity=red, coulomb=blue, magnetic=cyan, GM=rose, 1PN=orange, spin-curv=purple, radiation=yellow, yukawa=green, external=brown, higgs=lime, axion=indigo
- **Field overlays**: 64×64 offscreen, bilinear-upscaled. Higgs: purple(depleted)/lime(enhanced). Axion: indigo(+)/yellow(-).
- **Photons**: yellow (EM) / red (grav), alpha fades over PHOTON_LIFETIME=256
- **Pions**: green, glow in dark, constant alpha (decay is probabilistic)
- **V_eff plot**: 200-sample sidebar canvas

## GPU Acceleration (WebGPU)

### Architecture

`GPUPhysics` in `gpu-physics.js` orchestrates all compute passes. `GPURenderer` in `gpu-renderer.js` handles instanced rendering. Pipelines and bind group layouts are created in `gpu-pipelines.js`. Buffers allocated in `gpu-buffers.js`. Constants generated at compile time by `gpu-constants.js`.

**`gpu-constants.js`**: Single source of truth for JS→WGSL constant sharing. `buildWGSLConstants()` generates a WGSL `const` declaration block from `config.js` exports and `_PALETTE` hex→RGB conversions. Prepended to all shaders at pipeline creation. Includes physics constants, grid sizes, toggle bit definitions (FLAG_*, *_BIT), boundary/collision mode enums, and palette colors as `vec3f`.

**Dispatch sequence per substep** (all in one command encoder per substep):

1. Ghost generation (periodic boundary)
2. Tree build (Barnes-Hut, if enabled)
3. resetForces → cacheDerived → pairForce (or treeForce) → externalFields
4. Scalar field forces (Higgs gradient + mass mod, Axion gradient + axMod/yukMod) — BEFORE Boris
5. saveF1pn (save 1PN forces for velocity-Verlet correction)
6. Boris integrator: halfKick → rotate → halfKick → spinOrbit → applyTorques
7. Radiation reaction (Larmor, Hawking, pion emission)
8. borisDrift → expansion
9. 1PN velocity-Verlet correction (recompute + correction kick using saved f1pnOld)
10. Scalar field evolution (deposit → [self-grav] → KDK → gradients)
11. Collisions → field excitations → disintegration
12. Boson update (photon/pion drift, absorption, decay) → pair production
13. Boundary conditions

Post-substep (once per frame, separate encoder): heatmap, boson gravity, dead GC, history recording, updateColors, trail recording.

### Packed Struct Buffers

WebGPU limits storage buffers to `maxStorageBuffersPerShaderStage` (typically 10). To fit all pipelines within this limit, per-particle data is packed into struct buffers instead of individual SoA arrays:

| Struct | Size | Replaces | Fields |
|--------|------|----------|--------|
| `ParticleState` | 36B | posX/Y, velWX/Y, mass, charge, angW, baseMass, flags | 9 fields → 1 buffer |
| `ParticleAux` | 20B | radius, particleId, deathTime, deathMass, deathAngVel | 5 fields → 1 buffer |
| `ParticleDerived` | 32B | magMoment, angMomentum, invMass, radiusSq, velX/Y, angVel | 7+pad fields → 1 buffer |
| `AllForces` | 160B | 11 force vec2s, 3 torques, B-fields, B-gradients, totalForce | 10 vec4s → 1 buffer |
| `RadiationState` | 32B | jerkX/Y, radAccum, hawkAccum, yukawaRadAccum, radDisplayX/Y | 8 fields → 1 buffer |
| `Photon` | 32B | phPosX/Y, phVelX/Y, phEnergy, phEmitterId, phLifetime, phFlags | 8 fields → 1 buffer |
| `Pion` | 48B | piPosX/Y, piWX/Y, piMass, piCharge, piEnergy, piEmitterId, piAge, piFlags | 10+2pad fields → 1 buffer |

Worst-case pipeline (radiation) uses 10 storage buffers (was 42 before packing).

### Buffer Capacities

- `particleState`, `particleAux`, `derived`, `axYukMod`: `soaCapacity = MAX_PARTICLES × 2` (room for ghosts)
- `allForces`, `radiationState`, `color`, `f1pnOld`: `MAX_PARTICLES` (no ghost forces)
- `photonPool`: `MAX_PHOTONS (1024)`, `pionPool`: `MAX_PIONS (256)` — separate atomic counters (`phCount`, `piCount`)
- Ghost buffers: `ghostState`, `ghostAux`, `ghostDerived` — same structs, `MAX_PARTICLES` capacity
- History: `histPosX/Y/VelWX/VelWY/AngW/Time`: `MAX_PARTICLES × 256` ring buffers (lazy-allocated)
- Trail buffers: `trailX/Y`, `trailWriteIdx`, `trailCount` — lazy-allocated via `setTrailsEnabled()`

### Shader Organization

**Prepended shaders** (get `common.wgsl` concatenated before compilation): All Phase 2 shaders + `boundary.wgsl`. Use `SimUniforms`, `ParticleState`, etc. from common.wgsl.

**Field shaders** (get `field-common.wgsl` prepended): `field-deposit.wgsl`, `field-evolve.wgsl`, `field-forces.wgsl`, `field-selfgrav.wgsl`, `field-excitation.wgsl`, `field-render.wgsl`, `heatmap-render.wgsl`.

**Standalone shaders** (define own structs, NOT prepended): All Phase 3, Phase 4, `expansion.wgsl`, `heatmap.wgsl`, `disintegration.wgsl`, `pair-production.wgsl`, all render shaders. Must define `ParticleState`/`ParticleAux`/`Photon`/`Pion`/`SimUniforms` locally — keep in sync with `common.wgsl`.

**All shaders** receive the `buildWGSLConstants()` block (from `gpu-constants.js`) prepended at compile time — provides physics constants, toggle bit definitions, and palette colors without hardcoding.

### WGSL Gotchas

- **Operator precedence**: WGSL requires explicit parentheses when mixing `*` with `^` (XOR): `(a * b) ^ (c * d)`, not `a * b ^ c * d`.
- **Shared entry points**: When a shader has multiple entry points using the same module (e.g., `field-evolve.wgsl` with `computeLaplacian` + `computeGridGradients`), ALL bindings must use the most permissive access mode (`read_write` even if some entry points only read).
- **Buffer aliasing**: WebGPU disallows binding the same buffer twice in a dispatch. Radiation handles charge transfer by making `particleState` read-write in group 1 (not a separate binding).
- **Staging buffer mapping**: Readback staging buffers (`maxAccelStaging`, `ghostCountStaging`, `mergeCountStaging`) must not be copied to while still mapped from a previous `mapAsync`. Guard copies with `_pending` flags.
- **Uniform struct layout**: JS `writeUniforms`/`_writeFieldUniforms` field order must exactly match WGSL struct member order. WGSL structs use natural alignment (f32=4, u32=4, vec4=16). Off-by-one index errors silently corrupt all downstream fields.
- **Buffer initialization**: `addParticle()` must initialize ALL per-particle buffers. `axYukMod` defaults to (1.0, 1.0) not (0, 0) — zero would multiply all Yukawa/axion-modulated forces to zero. `radiationState` must be zeroed explicitly.

### GPU ↔ CPU Sync

- `addParticle()` writes packed `ParticleState` (36B) + `ParticleAux` (20B) + `color` (4B) + `axYukMod` (8B, initialized to 1.0/1.0) + `radiationState` (32B, zeroed) via `queue.writeBuffer()`
- `setToggles(physics)` packs CPU toggle booleans into `toggles0`/`toggles1` u32 bitfields. Called from `ui.js` `updateAllDeps()` on every toggle change AND from `_syncSlidersToGPU()` on every slider change. Heatmap state passed via `Object.create(sim.physics)` with `heatmapEnabled` added.
- `_writeFieldUniforms(dt)` writes `FieldUniforms` struct to shared field uniform buffer (used by field forces pass). Must match `field-common.wgsl` `FieldUniforms` struct layout exactly. `_writePerFieldUniforms(dt, fieldType)` writes to per-field dedicated uniform buffers (`_higgsUniformBuffer`/`_axionUniformBuffer`) with `currentFieldType` baked in — eliminates encoder split when both Higgs and Axion are active.
- Slider changes (yukawaMu, axionMass, higgsMass, external fields, bounceFriction, hubbleParam) sync to GPU via `_syncSlidersToGPU()` → `setToggles()` on every slider `input` event. Values are cached in `_yukawaMu`, `_higgsMass`, etc. and written to uniforms each substep.
- `serialize()`/`deserialize()` read/write full particle state via staging buffers for save/load. `deserialize()` initializes `axYukMod` to (1,1), zeroes `radiationState`, and restores slider parameters (`higgsMass`, `axionMass`, `yukawaMu`, `hubbleParam`).
- CPU-side `particles[]` array maintained in parallel for sidebar UI, presets, stats
- `device.lost` handler falls back to CPU mode, restores from periodic auto-save

### GPU Renderer

`GPURenderer` in `gpu-renderer.js` handles all visual output in GPU mode. All render pipelines have dual light/dark variants — light uses premultiplied alpha-over blend, dark uses additive blend (matching Canvas 2D `lighter`). Render passes (each in isolated command encoder for error containment):

1. **Particles**: Instanced quad rendering from `particleState` + `color` buffers. Color computed by `updateColors` compute pass (post-substep).
2. **Trails**: Line-strip per particle from ring buffer (`trailX`/`trailY`). Trail recording via `trails.wgsl` compute (post-substep). Lazy buffer allocation via `setTrailsEnabled()`.
3. **Field overlays**: Fullscreen triangle, bilinear-upscaled 64×64 grid. Per-field uniform buffers (higgs/axion) to avoid writeBuffer race. Lazy pipeline init via `initFieldOverlay()`.
4. **Heatmap overlay**: Fullscreen triangle, gravity/electric/Yukawa potential channels. Lazy pipeline init via `initHeatmapOverlay()`.
5. **Bosons**: Instanced point rendering for photons (yellow/red) and pions (green). Separate photon and pion pipelines.
6. **Spin rings**: Line-strip arcs around particles, arc length proportional to |angVel|. CW/CCW colors from `_PALETTE.spinPos`/`_PALETTE.spinNeg`.
7. **Force arrows**: Instanced arrow geometry, 11 force types with distinct colors.

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
- `sim.clearBosons()` releases photons/pions to pools and truncates arrays — use on preset load, clear, save-load
- PE accumulated inline in `pairForce()` via `_peAccum`; `potential.js` is fallback only (preset loads)
- **Object pooling**: `MasslessBoson.acquire()`/`.release()` and `Pion.acquire()`/`.release()` recycle dead instances. Module-level pool arrays, `_reset()` mutates existing Vec2s (no allocation). Pool caps (64 each) prevent unbounded growth. All creation sites use `.acquire()`, all removal sites call `.release()` before swap-and-pop.
- **Dirty flag**: `sim._dirty` skips entire render/stats path when paused with no interaction. Set by: physics update, camera, input, UI toggles, presets, theme, resize. `markDirty()` public method for external callers.
- **Batched force arrows**: Renderer accumulates arrows into pre-allocated `Float32Array` buffers, one `stroke()`+`fill()` per color. O(forces) canvas calls instead of O(particles×forces).
- **Batched spin rings**: Grouped by angular velocity sign, two passes (pos/neg) with one `stroke()`+`fill()` each. 4 canvas calls total instead of 9×N.
- **Photon alpha buckets**: 4 alpha bands (bright/med/dim/faint) → 4 `stroke()` calls instead of per-photon style changes.
- **shadowBlur bucketing**: Particles sorted into glow tiers, one `shadowBlur` set per tier instead of per-particle.
- **Inline torus minImage**: `pairForce()`, `_accum1PN()`, `calculateForce()`, `_compute1PNTreeWalk()` inline the torus fast path to eliminate cross-module function call overhead in O(N²) loops. Klein/RP² still dispatch to `minImage()`.
- **Yukawa cutoff**: `exp(-μr) < 0.002` ⟹ skip `Math.exp` when `μr > 6` in `pairForce()`.
- **Aberration pre-multiply**: Signal-delay aberration factor `(1 - n̂·v)^{-3}` pre-multiplied into `invR3a`/`invR5a` once per pair, reused across gravity/Coulomb/dipole/Yukawa terms.
- **Jerk guard**: Analytical jerk for Larmor radiation gated behind `radiationEnabled` flag — skips computation when radiation off.
- **Precomputed per-frame flags**: `_needAxMod`, conditional photon renorm flag computed once in `computeAllForces()` before pair loop.
- **Quadtree iterative insert**: Stack-based iterative insert replaces recursion. Direct quadrant child selection via `_childFor()` eliminates linear scan.
- **Lazy field init**: Higgs/Axion fields are `null` until first toggle-on (`ensureHiggsField()`/`ensureAxionField()` in main.js). Avoids 64×64 grid allocation + per-frame update when fields unused.
- **KaTeX render cache**: `ui.js` caches rendered KaTeX HTML by expression string, avoiding re-render on repeated info-tip opens.
- **KaTeX CSS preload**: Non-render-blocking `<link rel="preload" as="style">` pattern with `onload` swap.
- **V_eff dirty flag**: Hashes `(selId, refId, r_rounded, toggleKey)` to skip 200-sample curve recomputation when inputs unchanged.
- **Interior fast-path PQS**: `interpolate()`, `gradient()`, `interpolateWithGradient()` skip `_nb()` dispatch when stencil fully inside grid bounds. `_depositPQS()` and `_computeLaplacian()` also have interior fast paths.
- **Self-gravity early exit**: Skip O(SG⁴) coarse potential when field energy density below epsilon. `applyGravForces()` also exits early when field at vacuum (`hasEnergy` flag).
- **rAF-throttled hover**: `InputHandler` throttles mousemove tooltip updates to animation frame rate.
- **visibilitychange halt**: `main.js` pauses physics accumulator when tab hidden, preventing time spiral on refocus.
- **Accumulator cap**: ACCUMULATOR_CAP = 2 frames max per drain (prevents spiral of death).
- **textContent change detection**: `StatsDisplay` only writes DOM when value actually changed.
- Display throttles: STATS_THROTTLE_MASK=7 (energy, 8th frame), SIDEBAR_THROTTLE_MASK=1 (phase/effpot/selected, 2nd frame), FIELD_RENDER_INTERVAL=2, HEATMAP_INTERVAL=4
