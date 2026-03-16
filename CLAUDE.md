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
main.js                   838 lines  Simulation class, fixed-timestep loop, backend selection (CPU/GPU),
                                      pair production, pion loop, dirty-flag render, window.sim
index.html                498 lines  UI: 4-tab sidebar, reference overlay, zoom controls, field sliders
styles.css                295 lines  Project-specific CSS overrides, toggle/slider theme colors
colors.js                  18 lines  Project color tokens (extends shared-tokens.js)
src/
  integrator.js          1584 lines  CPU physics: Boris substep loop, radiation, pion emission/absorption,
                                      field excitations, tidal, GW quadrupole, expansion, Roche, external fields,
                                      Hertz bounce, scalar fields
  scalar-field.js         858 lines  ScalarField base: PQS grid, topology-aware deposition, Laplacian, C²
                                      gradients, field energy, excitations, particle-field gravity, self-gravity
  forces.js               803 lines  CPU pairForce(), computeAllForces(), calculateForce() (BH walk), compute1PN(),
                                      boson gravity, PE accumulator (resetPEAccum/getPEAccum), inline torus minImage
  ui.js                   716 lines  setupUI(), declarative dependency graph, info tips, reference overlay,
                                      shortcuts, dirty flag, KaTeX render cache, lazy field init triggers
  reference.js            714 lines  REFERENCE object: physics reference content (KaTeX math)
  renderer.js             707 lines  CPU Canvas 2D renderer (used as fallback when GPU unavailable)
  presets.js              680 lines  PRESETS (19 scenarios, 4 groups), loadPreset(), SLIDER_MAP, TOGGLE_MAP/TOGGLE_ORDER
  input.js                393 lines  InputHandler: mouse/touch, left/right-click symmetry (matter/antimatter),
                                      GPU deferred hit test (pollGPUHitResult), tree-accelerated CPU hit test
  quadtree.js             348 lines  QuadTreePool: SoA flat typed arrays, pool-based, zero GC, depth guard,
                                      iterative insert, direct quadrant child selection, boson distribution
  heatmap.js              315 lines  64x64 potential field overlay, signal-delayed positions, force-toggle-aware
  higgs-field.js          309 lines  HiggsField: Mexican hat potential, mass modulation, thermal phase transitions
  axion-field.js          299 lines  AxionField: quadratic potential, aF² EM coupling, PQ pseudoscalar coupling
  signal-delay.js         260 lines  getDelayedState() (3-phase light-cone solver, creationTime/deathTime guards)
  save-load.js            259 lines  saveState(), loadState(), downloadState(), uploadState(), quickSave/Load()
  stats-display.js        250 lines  Sidebar energy/momentum/drift readout, textContent change detection
  effective-potential.js  244 lines  V_eff(r) sidebar canvas, auto-scaling, axMod/yukMod modulation, dirty-flag skip
  pion.js                 236 lines  Massive Yukawa force carrier: proper velocity, (1+v²) GR deflection, decay, pool
  potential.js            211 lines  computePE(), treePE(), pairPE() (7 PE terms)
  energy.js               191 lines  KE, spin KE, PE, field energy, momentum, angular momentum
  config.js               166 lines  Named constants, mode enums (COL_*/BOUND_*/TORUS/KLEIN/RP²), helpers
  collisions.js           158 lines  handleCollisions(), resolveMerge(), annihilation, relativistic merge KE
  particle.js             142 lines  Particle: pos, vel, w, angw, baseMass, 11 force Vec2s, signal delay history
  phase-plot.js           137 lines  Phase space r-v_r plot (512-sample ring buffer)
  topology.js             131 lines  minImage(), wrapPosition() for Torus/Klein/RP²
  massless-boson.js        91 lines  MasslessBoson: pos, vel, energy, type ('em'/'grav'), BH tree lensing, pool
  vec2.js                  61 lines  Vec2 class: set, clone, add, sub, scale, mag, normalize, dist
  boson-utils.js           59 lines  treeDeflectBoson(): shared BH tree walk for photon/pion lensing
  backend-interface.js     57 lines  PhysicsBackend/RenderBackend typedefs, BACKEND_CPU/BACKEND_GPU constants
  cpu-physics.js           25 lines  CPUPhysics: thin adapter wrapping Physics (integrator.js) to PhysicsBackend
  relativity.js            25 lines  angwToAngVel(), setVelocity()
  canvas-renderer.js       20 lines  CanvasRenderer: thin adapter wrapping Renderer to RenderBackend
  gpu/
    gpu-physics.js       3791 lines  GPUPhysics: WebGPU compute pipeline orchestrator, addParticle/serialize,
                                      all dispatch methods, bind group creation, adaptive substepping, readback,
                                      per-field uniform buffers (Higgs/Axion), pre-allocated write buffers
    gpu-pipelines.js     1897 lines  Pipeline + bind group layout creation for all compute/render shaders,
                                      fetchShader() (single source of truth), getSharedPrefix() caching
    gpu-renderer.js      1215 lines  WebGPU instanced rendering: particles, bosons, field overlays, heatmap,
                                      trails, force arrows, spin rings, torque arcs, dashed rings
                                      (dual light/dark pipeline variants)
    gpu-buffers.js        564 lines  Buffer allocation: packed structs, quadtree, collision, field, history,
                                      trail buffers, staging
    gpu-constants.js      298 lines  buildWGSLConstants(): generates WGSL const block from config.js +
                                      _PALETTE colors, single source of truth for JS/WGSL constants
    shaders/               51 files  WGSL compute + render shaders (9199 lines total)
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

gpu-physics.js   <- gpu-buffers, gpu-pipelines (fetchShader + pipeline creators), gpu-constants
gpu-pipelines.js <- gpu-constants (buildWGSLConstants), exports fetchShader + getSharedPrefix
gpu-renderer.js  <- gpu-pipelines (fetchShader + render pipeline creators)
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
- Magnetic moment: `μ = 0.2qωr²` -- cached as `p.magMoment`
- Angular momentum: `L = Iω` -- cached as `p.angMomentum`
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

Independent toggle. `F = -g²m₁m₂e^{-μr}/r² · (1+μr)`. Parameters: `yukawaCoupling` (14), `yukawaMu` (0.15, slider 0.05-0.25). Includes analytical jerk for radiation. Emits pions as massive force carriers.

**Scalar Breit correction** (requires 1PN): O(v²/c²) from massive scalar boson exchange. `δH = g²m₁m₂e^{-μr}/(2r) · [v₁·v₂ + (r̂·v₁)(r̂·v₂)(1+μr)]`. Into `force1PN`, velocity-Verlet corrected.

### External Background Fields

No toggle -- controlled by slider values (default 0). Gravity (`F = mg`), Electric (`F = qE`), Magnetic (uniform Bz via Boris rotation). Direction angle sliders auto-show when strength > 0.

### Bounce (Hertz Contact)

`F = K · δ^{1.5}` (K=1). Tangential friction transfers torque. Quadtree-accelerated when BH on, O(n²) fallback when off -- do not early-return when `root < 0`.

## Scalar Fields

### Base Class (`ScalarField`)

PQS (cubic B-spline, order 3) grid on 64×64 (CPU) or 128×128 (GPU). 4×4 stencil per particle. C² interpolation and C² gradients (PQS-interpolated central-difference grid gradients). Pre-allocated weight arrays for zero-alloc hot path.

Key methods: `_nb()` (topology-aware neighbor, absolute coords), `_depositPQS()` (interior fast path + border fallback), `_computeLaplacian()` (interior fast path + border path), `_computeGridGradients()`, `_computeViscosity()`, `interpolate()`, `gradient()`, `interpolateWithGradient()` (fused, single stencil walk), `_fieldEnergy()`, `depositExcitation()`, `_computeEnergyDensity()`, `applyGravForces()`, `gravPE()`, `computeSelfGravity()`.

**Particle-field gravity** (requires Gravity -> Field Gravity toggle): F = -m·∇Φ via PQS interpolation of pre-computed potential gradients, O(N×16). Subclasses override `_addPotentialEnergy()` for V(φ). Default off.

**Field self-gravity** (requires Gravity -> Field Gravity toggle): Weak-field GR correction to Klein-Gordon. Φ computed via FFT convolution with Green's function G(r) = -1/√(r²+ε²) on the full grid, O(N² log N). Φ clamped to ±`SELFGRAV_PHI_MAX` (0.2) to prevent Laplacian sign-flip instability when `1+4Φ < 0`.

**Numerical viscosity**: `ν·∇²(ȧ)` in both KDK half-kicks, where `ν = 1/(2√(1/dx²+1/dy²))`. Gives Q=1 at Nyquist frequency, vanishes for physical (long-wavelength) modes. Prevents grid-scale noise from ringing indefinitely at high resolution.

Field arrays: `field`/`fieldDot` (not `phi`/`phiDot`). Field clamp: SCALAR_FIELD_MAX = 2. Merge excitation amplitude capped at `EXCITATION_MAX_AMPLITUDE` (1.0).

### Higgs Field

Independent toggle. Mexican hat `V(φ) = -½μ²φ² + ¼λφ⁴`. VEV=1; `λ = μ² = m_H²/2`. Slider: m_H 0.25-0.75 (default 0.50). Mass generation: `m_eff = baseMass · max(|φ(x)|, 0.05)`. Rate clamp: `HIGGS_MASS_MAX_DELTA = 4`. Gradient force into `forceHiggs`. Phase transitions: `μ²_eff = μ² - KE_local`.

### Axion Field

Independent toggle; requires Coulomb or Yukawa. Quadratic `V(a) = ½m_a²a²`, vacuum at a=0. Slider: m_a 0.01-0.10 (default 0.05).

**Scalar EM coupling (aF², when Coulomb on)**: Same for matter/antimatter. `α_eff = α(1+g·a)`. Per-particle `p.axMod`, geometric mean pairwise.

**PQ coupling (when Yukawa on)**: Flips sign for antimatter. `yukMod = 1+g·a` (matter) / `1-g·a` (antimatter). At vacuum: CP conserved.

## Pions (Massive Force Carriers)

Mass = `yukawaMu`. Proper velocity `w`: `vel = w/√(1+w²)`. GR deflection: `(1+v²)` factor.

**Emission**: Scalar Larmor `P = g²F_yuk²/3`. Species: π⁰ (50%), π⁺/π⁻ (25% each). MAX_PIONS = 256.

**Decay**: π⁰→2γ (half-life 32), π⁺→e⁺+γ (half-life 128), π⁻→e⁻+γ (half-life 128). Two-body kinematics in rest frame, Lorentz-boosted.

**Absorption**: Quadtree overlap query. Self-absorption permanently blocked by `emitterId`.

**Boson gravity** (requires Gravity + Barnes-Hut -> Boson Gravity toggle): Particle→boson and boson→boson via BH tree walks.

## Advanced Physics

### 1PN Corrections

Requires Relativity. Four O(v²/c²) sectors into `force1PN`:
- **EIH** (GM + 1PN): perihelion precession.
- **Darwin EM** (Magnetic + 1PN): EM remainder.
- **Bazanski** (GM + Magnetic + 1PN): mixed 1/r³.
- **Scalar Breit** (Yukawa + 1PN): massive scalar exchange.

NOT Newton's 3rd law. Velocity-Verlet corrected. `compute1PN()` zeroes `force1PN` before accumulating.

### Radiation

Requires Gravity, Coulomb, or Yukawa. Single toggle, four mechanisms:
- **Larmor dipole** (Coulomb): Landau-Lifshitz. Analytical jerk (gravity, Coulomb, Yukawa, dipoles, Bazanski, EIH position-only).
- **EM quadrupole** (Coulomb): `P = (1/180)|d³Q_ij/dt³|²`. Emits photons.
- **GW quadrupole** (Gravity): `P = (1/5)|d³I^TF_ij/dt³|²`. Emits gravitons (red).
- **Pion emission** (Yukawa): `P = g²F_yuk²/3`. Emits pions.

Self-absorption permanently blocked by `emitterId` for both photons and pions.

### Black Hole Mode

Requires Gravity + Relativity. Locks collision to Merge.
- **No hair**: Antimatter erased. Pair production disabled.
- **Kerr-Newman**: `r₊ = M + √(M² - a² - Q²)`, naked singularity floor.
- **Hawking** (requires Radiation): `T = κ/(2π)`, `P = σT⁴A`. Full evaporation emits final photon burst.

### Signal Delay

Auto-activates with Relativity. Per-particle circular history buffers (Float64Array[256], recorded every HISTORY_STRIDE=64 calls):
1. Newton-Raphson segment search (≤8 iterations)
2. Exact quadratic solve on converged segment
3. Constant-velocity extrapolation for early history (skipped for dead particles)

**Causality**: `creationTime` rejects extrapolation past creation. Dead particles continue exerting forces via signal delay until `simTime - deathTime > 2·domain_diagonal`.

**Liénard-Wiechert aberration**: `(1 - n̂·v_source)^{-3}`, clamped [0.01, 100]. Applied to gravity, Coulomb, Yukawa, dipole. Not 1PN (already O(v²)).

### Additional Physics

- **Spin-Orbit**: Stern-Gerlach `F = +μ·∇(Bz)`, Mathisson-Papapetrou `F = -L·∇(Bgz)`.
- **Disintegration & Roche**: Tidal + centrifugal + Coulomb stress. Eggleton (1983) Roche lobe.
- **Expansion**: `pos += H(pos - center)dt`, `w *= (1-Hdt)`. Default H = 0.001.
- **Antimatter & Pair Production**: Right-click spawns. Annihilation emits photons. Pair production from energetic photons near massive bodies.

## Sign Conventions (IMPORTANT)

All GEM interactions are **attractive** (gravity has one sign of "charge"):
- GM dipole: `+3L₁L₂/r⁴` (positive = attractive)
- GM Boris parameter: `+2Bgz` (co-moving masses attract)
- Bgz field: `-m_s(v_s × r̂)_z/r²` (negative sign in code)
- Frame-drag torque: positive drives co-rotation

**Angular velocity convention (y-down canvas)**: 2D cross product `rx·vy - ry·vx` gives positive for clockwise on screen.

## Energy, PE & Collisions

**PE**: Accumulated inline during `pairForce()` via `_peAccum`. 9 terms. `potential.js` kept as fallback for preset-load recomputation.

**Energy**: Relativistic KE = `wSq/(γ+1)·mass`. `pfiEnergy` = particle-field interaction from Higgs + Axion.

**Collisions**: Pass / bounce (Hertz) / merge. Always tree-accelerated broadphase. Merge kills both parents, creates new particle, retires parents for signal delay fade-out.

## Topology

Boundary "loop": Torus (TORUS=0), Klein bottle (KLEIN=1), RP² (RP2=2). `minImage()` zero-alloc via `out` parameter.

## Quadtree

`QuadTreePool`: SoA typed arrays, 512-node pool (grows). BH_THETA = 0.5, QUADTREE_CAPACITY = 4. Depth guard max 48. **Always built** every substep regardless of BH toggle -- used by collisions, hit testing, and optionally BH force computation.

## Toggle Dependencies

```
Forces:                        Physics:
  Gravity                        Relativity          [signal delay auto-activates]
    -> Gravitomagnetic             -> 1PN             [requires Magnetic, GM, or Yukawa]
    -> Field Gravity               -> Black Hole      [+Gravity, locks collision to Merge]
  Coulomb                        Spin-Orbit           [requires Magnetic or GM]
    -> Magnetic                  Radiation             [requires Gravity, Coulomb, or Yukawa]
  Gravity + Barnes-Hut
    -> Boson Gravity
  Yukawa               [independent]
  Axion                [requires Coulomb or Yukawa]
  Higgs                [independent]
Disintegration                   [requires Gravity, locks collision to Merge]
Barnes-Hut                       [independent]
Expansion                        [independent, in Engine tab]
```

Declarative `DEPS` array in `ui.js`, topological evaluation via `updateAllDeps()`.

Defaults on: gravity, coulomb, magnetic, gravitomag, 1PN, relativity, spin-orbit, radiation.
Defaults off: Boson Gravity, Field Gravity, Yukawa, Axion, Higgs, Disintegration, Expansion, Barnes-Hut, Black Hole.

## UI

4-tab sidebar: Settings (mass/charge/spin, spawn mode, force/physics toggles), Engine (GPU toggle, BH, collisions, boundary/topology, external fields, visuals, speed), Stats (energy/momentum/drift/mass), Particle (selected details, force breakdown, phase plot, effective potential).

19 presets in 4 groups: Gravity (6), Electromagnetism (3), Exotic (8), Cosmological (2). First 9 via keyboard `1`-`9`. Speed: 1-64, default 32.

## Backend Architecture

Two interchangeable backends via `selectBackend()`:

- **CPU**: `CPUPhysics` + `CanvasRenderer`. Thin adapters over integrator.js and renderer.js.
- **GPU**: `GPUPhysics` + `GPURenderer`. GPU renders to `<canvas id="gpuCanvas">` (z-index: -1). CPU canvas (z-index: 0) on top for 2D overlays.

Falls back to CPU on WebGPU unavailability or device loss. Force CPU via `?cpu=1`. Runtime toggle in Engine tab.

### GPU Packed Struct Buffers

| Struct | Size | Fields |
|--------|------|--------|
| `ParticleState` | 36B | posX/Y, velWX/Y, mass, charge, angW, baseMass, flags |
| `ParticleAux` | 20B | radius, particleId, deathTime, deathMass, deathAngVel |
| `ParticleDerived` | 32B | magMoment, angMomentum, invMass, radiusSq, velX/Y, angVel, bodyRSq |
| `AllForces` | 160B | 11 force vec2s, 3 torques, B-fields, B-gradients, totalForce, jerk |
| `RadiationState` | 48B | radAccum, hawkAccum, yukawaRadAccum, radDisplay, quadAccum, d3I/d3Q contrib |
| `Photon` | 32B | pos, vel, energy, emitterId, lifetime, flags |
| `Pion` | 48B | pos, w, mass, charge, energy, emitterId, age, flags |

### Shader Organization

**Shared includes** (prepended to ALL shaders via `getSharedPrefix()`):
- `shared-structs.wgsl`: All packed struct definitions (ParticleState, ParticleAux, ParticleDerived, AllForces, SimUniforms, RadiationState, Photon, Pion). Single source of truth — never redefine these in individual shaders.
- `shared-topology.wgsl`: `fullMinImageP(ox, oy, sx, sy, domW, domH, topo)` parameterized topology-aware minimum image function.
- `shared-rng.wgsl`: `pcgHash(seed)`/`pcgRand(seed)` PCG hash PRNG.

**Additional shared includes** (prepended selectively via `getTreePrefix()`):
- `shared-tree-nodes.wgsl`: Read-only BH node accessors (`getMinX`, `getComX`, `getTotalMass`, etc.). Used by forces-tree and collision. tree-build.wgsl has its own atomic versions.

**Prepend chains** (all start with `wgslConstants + shared-structs + shared-topology + shared-rng`):
- Phase 2 shaders + `boundary.wgsl`: + `common.wgsl` (toggle helpers, `fullMinImage` wrapper). Force shaders also get `signal-delay-common.wgsl`.
- Field shaders: + `field-common.wgsl` (FieldUniforms, PQS helpers).
- Tree-walk shaders (forces-tree, collision): + `shared-tree-nodes.wgsl` (node accessors).
- Signal delay shaders (forces-tree, onePN, heatmap): + `signal-delay-common.wgsl` (getDelayedStateGPU).
- All other standalone shaders: shared prefix only.
- `fetchShader()` exported from `gpu-pipelines.js` (single source of truth, imported by gpu-physics.js and gpu-renderer.js).

### GPU ↔ CPU Sync

- `addParticle()` writes packed structs via `queue.writeBuffer()`
- `setToggles()` packs booleans into `toggles0`/`toggles1` u32 bitfields, caches individual `_xxxEnabled` booleans
- `serialize()`/`deserialize()` for save/load and GPU→CPU toggle
- GPU hit test: `hitTest()`/`readHitResult()` with 1-frame latency via staging buffer
- GPU stats: `requestStats()`/`readStats()` at STATS_THROTTLE_MASK rate, 512-byte double-buffered staging
- `device.lost` handler falls back to CPU with auto-save recovery

### WGSL Gotchas

- WGSL requires explicit parentheses when mixing `*` with `^` (XOR)
- Multiple entry points sharing a module need `read_write` access mode on shared bindings
- WebGPU disallows binding the same buffer twice in a dispatch
- Staging buffers must not be copied to while mapped from previous `mapAsync`
- JS uniform write order must exactly match WGSL struct member order
- `addParticle()` must initialize ALL per-particle buffers. `axYukMod` defaults to (1.0, 1.0) not (0, 0)

### Numerical Stability (GPU)

- Division-by-zero: `select(0, 1/x, x > EPSILON)` or `max(x, EPSILON)`
- NaN barriers before writing to global memory (not after)
- `sqrt(max(x, 0))` on all discriminants; `exp(-μr)` guarded with `select(0, ..., μr < 80)`
- `deathTime` sentinel uses `FLT_MAX` (3.4028235e38) not `Infinity`
- Collision merge: `mass <= EPSILON` guards (not `== 0`) for race conditions
- Render: premultiplied alpha (`color.rgb * alpha`) with `srcFactor: 'one'`

## Renderer

### Visual Style (both backends)

- **Particles**: r = ∛(mass) (BH: r₊), no glow. Neutral=slate (BH light: text color). Charged: RGB lerp base→red(+)/blue(-).
- **Trails**: circular Float32Array[256], wrap-detection for periodic boundaries
- **Force vectors**: gravity=red, coulomb=blue, magnetic=cyan, GM=rose, 1PN=orange, spin-curv=purple, radiation=yellow, yukawa=green, external=brown, higgs=lime, axion=indigo
- **Field overlays**: 64×64 (CPU) / 128×128 (GPU), bilinear-upscaled. Higgs: purple/lime. Axion: indigo/yellow.
- **Heatmap**: 64×64 (CPU) / 128×128 (GPU), 3-channel (gravity/slate, electric/blue-red, Yukawa/green), mode selector (All/Grav/Elec/Yukawa)
- **Photons**: yellow (EM) / red (grav), alpha fades over PHOTON_LIFETIME=256
- **Pions**: green, constant alpha
- **V_eff plot**: 200-sample sidebar canvas

### GPU Renderer Passes

Particles, trails, field overlays, heatmap, bosons (photons + pions), spin rings, force arrows, torque arcs, dashed rings (ergosphere + antimatter). All with dual light/dark pipeline variants.

## Key Patterns

- `window.sim` for console debugging. `_PALETTE`/`_FONT` frozen by colors.js
- `Vec2.set(x,y)` in hot paths; `pairForce()` accumulates into `out` Vec2, zero alloc
- Module-level `_miOut` for zero-alloc `minImage()` output
- Particle constructor declares all properties upfront (V8 hidden class stability)
- World coordinates: `sim.domainW/H` (viewport / WORLD_SCALE), not pixels
- Theme: `data-theme` on `<html>` (not body)
- `.mode-toggles` sets `display: grid` overriding `hidden` -- use `style.display`
- External field trig cached once per frame via `_cacheExternalFields()`
- `forceRadiation` cleared for all particles before substep loop
- History recording counts `update()` calls, not substeps
- PE accumulated inline in `pairForce()` via `_peAccum`; `potential.js` is fallback only
- Object pooling: `MasslessBoson.acquire()`/`.release()` and `Pion.acquire()`/`.release()` with pool caps (64)
- Dirty flag: `sim._dirty` skips entire render/stats when paused with no interaction
- Batched rendering: force arrows, spin rings, photon alpha buckets -- O(forces) canvas calls not O(particles×forces)
- Inline torus minImage in pairForce/1PN loops; Klein/RP² dispatch to `minImage()`
- Yukawa cutoff: skip `Math.exp` when `μr > 6`
- Lazy field init: Higgs/Axion fields `null` until first toggle-on
- KaTeX CSS preload + render cache
- rAF-throttled hover, visibilitychange halt, accumulator cap = 2 frames
