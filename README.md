# Geon

A relativistic N-body simulation exploring how gravity, electromagnetism, scalar fields, and spin shape particle dynamics in 2D. Particles have mass, charge, and angular momentum; forces propagate at finite speed; scalar fields live on dynamical grids; matter and antimatter obey different symmetries.

**[Live Demo](https://a9l.im/physsim)** | Part of the [a9l.im](https://a9l.im) portfolio

## What It Simulates

Everything runs in natural units (c = 1, G = 1, h = 1). Particles store proper velocity **w** = yv as their state variable, so the speed of light is enforced automatically: coordinate velocity **v** = **w** / sqrt(1 + w^2) always satisfies |v| < c. The same trick applies to spin.

### Forces

- **Gravity** -- Newtonian 1/r^2 attraction between all massive particles, Plummer-softened.
- **Coulomb** -- 1/r^2 electrostatic force; like charges repel, opposites attract.
- **Magnetic dipole** -- 3u1u2/r^4 interaction between spinning charged particles. Magnetic moment u = qwr^2/5 (uniform charge sphere).
- **Lorentz** -- q(**v** x **B**) force from moving charges and spinning dipoles. Handled exactly by Boris rotation.
- **Gravitomagnetic dipole** -- 3L1L2/r^4 interaction between spinning masses. Co-rotating masses attract (GEM sign convention).
- **Frame-dragging** -- 4m(**v** x **B**_g) from moving/spinning masses, plus a torque that aligns neighboring spins.
- **Tidal locking** -- Dissipative tidal torque drives spin toward synchronous rotation with the orbital frequency. Always active when gravity is on.
- **Yukawa** -- Screened nuclear force V(r) = -g^2 e^{-ur}/r between massive particles, with configurable coupling and range parameter u (= pion mass). Includes analytical jerk for radiation reaction. When 1PN is enabled, receives O(v^2/c^2) **scalar Breit corrections** from massive scalar boson exchange.

### Relativistic Corrections

- **1PN (Einstein-Infeld-Hoffmann)** -- O(v^2/c^2) correction to gravity. Produces perihelion precession at the GR rate.
- **1PN (Darwin EM)** -- O(v^2/c^2) correction to electromagnetism from the Darwin Lagrangian.
- **1PN (Bazanski cross-term)** -- Mixed gravity-EM 1PN interaction. Position-dependent 1/r^3 force coupling mass and charge.
- **1PN (Scalar Breit)** -- O(v^2/c^2) correction for massive scalar boson exchange (Yukawa). Full Breit Hamiltonian with radial and tangential components. Velocity-Verlet corrected.
- **Signal delay** -- Forces use source positions from the past light cone, solved analytically with a three-phase algorithm (Newton-Raphson segment search, exact quadratic solve, constant-velocity extrapolation). Includes the Lienard-Wiechert (1 - n.v)^{-3} aberration factor. Newly created particles respect light-cone causality; deleted particles continue exerting forces until their signal fades past all observers.

### Scalar Fields

Two dynamical scalar fields live on 64x64 grids, sharing a common PQS (cubic B-spline) infrastructure for C^2-smooth interpolation and gradients. Both support topology-aware boundary conditions, receive energy from merge collisions as propagating wave packets, and gravitate particles and each other via weak-field GR corrections.

- **Higgs field** -- Mexican hat potential V(f) = -1/2 u^2 f^2 + 1/4 l f^4. The field spontaneously breaks symmetry to VEV = 1. Particles acquire effective mass m_eff = baseMass * |f(x)|. At VEV, particles have full mass; when kinetic energy drives the field to zero, particles become effectively massless -- a classical analog of the **electroweak phase transition**. Merge collisions excite oscillations around the VEV (**Higgs bosons**).

- **Axion field** -- Quadratic potential V(a) = 1/2 m_a^2 a^2 with vacuum at a = 0. Two coupling channels:

  - **Scalar EM (aF^2)** -- active when Coulomb is on. Makes the fine structure constant position-dependent: a_eff(x) = a(1 + g*a(x)). Charged particles source the field proportional to q^2 and feel gradient forces.

  - **Peccei-Quinn (aGG analog)** -- active when Yukawa is on. A pseudoscalar coupling that **flips sign for antimatter**, implementing the PQ mechanism for CP violation. At vacuum a = 0, matter and antimatter interact identically -- **CP is conserved**.

  Merge collisions excite oscillations around a = 0 (**axion particles**).

### Radiation and Force Carriers

- **Larmor radiation** -- Landau-Lifshitz force with analytical jerk from gravity + Coulomb + Yukawa, numerical backward-difference for residuals. Dipole photon emission with relativistic aberration.
- **EM quadrupole** -- d^3 Q_ij/dt^3 formula with TT-projected angular emission via rejection sampling.
- **GW quadrupole** -- Trace-free mass quadrupole d^3 I^TF_ij/dt^3 with COM-relative coordinates. Gravitons rendered red.
- **Pion emission (scalar Larmor)** -- Yukawa interactions radiate massive pions with P = g^2 m^2 a^2/3 (spin-0 angular factor 1/3 vs 2/3 for spin-1 EM). Pions travel at v < c, experience gravitational deflection with factor (1 + v^2), and **decay**: pi0 -> 2y (Lorentz-boosted), pi+ -> e+ + y, pi- -> e- + y (two-body kinematics in rest frame).
- **Photon & pion absorption** -- Quadtree overlap query transfers momentum (and charge for pi+/-) to absorbing particles. Self-absorption permanently blocked by emitter tracking.
- **Boson gravity** -- Photons and pions gravitate particles and each other via Barnes-Hut tree walks. GR deflection: 2x for photons (null geodesic), (1+v^2)x for pions (massive).
- **Field excitations** -- Inelastic merges deposit Gaussian wave packets into active scalar fields, propagated by the Klein-Gordon equation.

### Additional Physics

- **Spin-orbit coupling** -- Energy transfer between translational and rotational KE via Stern-Gerlach and Mathisson-Papapetrou kicks.
- **Disintegration** -- Roche-limit fragmentation with Eggleton (1983) Roche lobe formula and continuous L1 mass transfer.
- **Black hole mode** -- Kerr-Newman horizons r+ = M + sqrt(M^2 - a^2 - Q^2). Ergosphere visualization. Hawking radiation; extremal BHs stop radiating. Sub-threshold BHs evaporate with photon burst.
- **Cosmological expansion** -- Hubble flow from domain center with peculiar velocity redshift.
- **Antimatter & pair production** -- Right-click spawns antimatter. Matter-antimatter mergers annihilate with photon emission. Energetic photons near massive bodies produce particle-antiparticle pairs.
- **External fields** -- Uniform gravitational, electric, and magnetic background fields with configurable strength and direction.

### Integrator

Boris integrator (half-kick / rotate / half-kick / drift) with adaptive substepping based on acceleration and cyclotron frequency, capped at 32 substeps. Boris rotation handles magnetic forces exactly, preserving |v|. The four 1PN sectors use velocity-Verlet correction for second-order accuracy. Scalar fields evolve via Stormer-Verlet (KDK).

### Algorithms

- **Barnes-Hut** -- Toggleable O(N log N) quadtree with pool-based structure-of-arrays layout and zero per-frame allocation. Signal delay at leaf level; distant aggregates use current positions.
- **Collisions** -- Pass-through, elastic bounce (Hertz contact), or merge (conserves mass, charge, momentum, angular momentum; tracks relativistic KE for field excitations).
- **Topological boundaries** -- Periodic loop with torus, Klein bottle, or real projective plane identification. Minimum-image separation handles non-orientable crossings.
- **PQS grid** -- Cubic B-spline (order 3) infrastructure for scalar fields. 4x4 stencil, C^2-smooth interpolation/gradients, topology-aware deposition, interior fast-path Laplacian, zero-alloc hot paths.

## Controls

| Input | Action |
|-------|--------|
| Left click | Spawn particle / select matter / delete antimatter |
| Left drag | Spawn with velocity |
| Right click | Spawn antimatter / select antimatter / delete matter |
| Right drag | Spawn antimatter with velocity |
| Scroll | Zoom |
| Middle drag | Pan |
| `Space` | Pause / resume |
| `.` | Step forward one frame |
| `1`-`9` | Load preset directly |
| `V` / `F` / `C` | Toggle velocity / force / component vectors |
| `T` / `S` | Toggle theme / sidebar |
| `?` | Keyboard shortcut help |

### Sidebar Tabs

1. **Settings** -- Particle mass / charge / spin sliders, spawn mode, force toggles (gravity, Coulomb, magnetic, gravitomagnetic, Yukawa, axion, Higgs), physics toggles (relativity, 1PN, black hole, spin-orbit, radiation, disintegration). 19 presets across four groups.
2. **Engine** -- GPU toggle, Barnes-Hut, collision mode, bounce friction, boundary/topology, external fields (gravity/electric/magnetic with direction), visual overlays (trails, velocity/force/component vectors, potential heatmap, field overlays), sim speed (1-64x), cosmological expansion.
3. **Stats** -- Energy breakdown (linear KE, spin KE, PE, field energies, radiated, drift), conserved quantities (momentum, angular momentum, total mass).
4. **Particle** -- Selected particle details, per-force breakdown (11 force components + 4 torque types), phase space plot (r vs v_r), effective potential plot V_eff(r).

### Presets

19 scenarios across four groups:

| Group | Presets |
|-------|---------|
| **Gravity** | Kepler Orbits, Precession, Binary Inspiral, Tidal Lock, Roche Limit, Hawking Evaporation |
| **Electromagnetism** | Atom, Bremsstrahlung, Magnetic Dipoles |
| **Exotic** | Atomic Nucleus, Axion Field, Pion Exchange, Higgs Mechanism, Higgs Boson, Axion Burst, Peccei-Quinn, Phase Transition |
| **Cosmological** | Galaxy, Expanding Universe |

## Running Locally

```bash
# Serve from parent -- shared design files load via absolute paths
cd path/to/a9lim.github.io && python -m http.server
# -> http://localhost:8000/physsim/
```

No build step, no dependencies, no npm. ES6 modules require HTTP (no `file://`).

## Tech

Zero-dependency vanilla JavaScript. CPU backend uses Canvas 2D; GPU backend uses WebGPU compute + instanced rendering (auto-detected at startup, falls back to CPU). All physics and rendering code is hand-written -- no physics engines, no libraries. Structure-of-arrays quadtree, circular history buffers for signal delay, cubic B-spline field infrastructure, adaptive substepping -- all designed for zero per-frame allocation in hot paths.

~29,100 lines of code total across 92 source files.

## Architecture

```
main.js                   847 lines  Simulation class, fixed-timestep loop, backend selection (CPU/GPU),
                                      pair production, pion loop, dirty-flag render, window.sim
index.html                498 lines  UI: 4-tab sidebar, reference overlay, zoom controls, field sliders
styles.css                295 lines  Project-specific CSS overrides
colors.js                  18 lines  Project color tokens (extends shared-tokens.js)
src/
  integrator.js          1534 lines  Physics: Boris substep loop, radiation, pion emission/absorption, field
                                      excitations, tidal, GW quadrupole, expansion, Roche, external fields,
                                      Hertz bounce, scalar fields
  forces.js               832 lines  Pairwise + Barnes-Hut force accumulation, 1PN (4 sectors), boson gravity
  scalar-field.js         807 lines  ScalarField base: PQS grid, topology-aware deposition, Laplacian, C^2
                                      gradients, field energy, excitations, particle-field gravity, self-gravity
  ui.js                   724 lines  DOM setup, declarative toggle dependencies, info tips, shortcuts
  reference.js            715 lines  Physics reference content (KaTeX math)
  renderer.js             706 lines  Canvas 2D: particles, trails, spin rings, vectors, photons, pions,
                                      field overlays, batched draw calls
  presets.js              680 lines  19 preset scenarios (Gravity / EM / Exotic / Cosmological)
  input.js                397 lines  Mouse/touch, left/right-click symmetry (matter/antimatter),
                                      GPU deferred hit test
  quadtree.js             348 lines  SoA pool-based Barnes-Hut tree (zero GC), boson distribution
  higgs-field.js          318 lines  HiggsField: Mexican hat potential, mass modulation, thermal transitions
  heatmap.js              315 lines  Gravitational + electric + Yukawa potential field overlay
  axion-field.js          308 lines  AxionField: quadratic potential, scalar aF^2 coupling, PQ CP violation
  signal-delay.js         260 lines  Three-phase light-cone solver on circular history buffers
  save-load.js            259 lines  State serialization, quick save/load, file export/import
  stats-display.js        250 lines  Sidebar energy/momentum/drift/mass readout
  effective-potential.js  244 lines  V_eff(r) sidebar canvas with axMod/yukMod modulation
  pion.js                 236 lines  Massive Yukawa force carrier: proper velocity, GR deflection, decay
  potential.js            211 lines  PE computation (9 terms, pairwise + tree traversal)
  energy.js               191 lines  KE, PE, field energy, momentum, angular momentum
  config.js               164 lines  Named constants, mode enums, helpers
  collisions.js           158 lines  Merge, annihilation, baseMass conservation, relativistic KE tracking
  phase-plot.js           137 lines  Phase space r-v_r plot (512-sample ring buffer)
  particle.js             132 lines  Particle: 11 force Vec2s, axMod/yukMod, baseMass, signal delay history
  topology.js             131 lines  Torus / Klein / RP^2 min-image + wrapping
  fft.js                  100 lines  Cooley-Tukey radix-2 FFT for scalar field self-gravity
  massless-boson.js        91 lines  Radiation photon/graviton with BH tree lensing, object pool
  vec2.js                  61 lines  2D vector math
  boson-utils.js           59 lines  Shared BH tree walk for photon/pion gravitational lensing
  backend-interface.js     57 lines  PhysicsBackend/RenderBackend interface contracts
  cpu-physics.js           25 lines  CPUPhysics adapter (wraps integrator.js Physics class)
  relativity.js            25 lines  Proper velocity helpers
  canvas-renderer.js       20 lines  CanvasRenderer adapter (wraps renderer.js Renderer class)
  gpu/
    gpu-physics.js       3792 lines  WebGPU compute pipeline orchestrator, all dispatch methods
    gpu-pipelines.js     1897 lines  Pipeline + bind group layout creation for compute/render shaders
    gpu-renderer.js      1215 lines  WebGPU instanced rendering: particles, bosons, trails, arrows, spin
                                      rings, torque arcs, dashed rings (dual light/dark variants)
    gpu-buffers.js        564 lines  Buffer allocation: packed structs, quadtree, collision, field, trails
    gpu-constants.js      298 lines  Single-source JS->WGSL constant generation from config.js + palette
    shaders/               51 files  WGSL compute + render shaders (9199 lines total)
```

## Sibling Projects

- [Metabolism](https://github.com/a9lim/biosim) -- [a9l.im/biosim](https://a9l.im/biosim)
- [Redistricting](https://github.com/a9lim/gerry) -- [a9l.im/gerry](https://a9l.im/gerry)
- [Shoals](https://github.com/a9lim/finsim) -- [a9l.im/finsim](https://a9l.im/finsim)

## License

[AGPL-3.0](LICENSE)
