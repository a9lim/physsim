# Geon

A relativistic N-body simulation exploring how gravity, electromagnetism, scalar fields, and spin shape particle dynamics in 2D. Eleven distinct force types, two dynamical scalar fields, massive force carriers, and a full WebGPU compute backend -- all in 29,100 lines of zero-dependency vanilla JavaScript.

**[Live Demo](https://a9l.im/physsim)** | Part of the [a9l.im](https://a9l.im) portfolio

## Highlights

- **11 force types** -- Newtonian gravity, Coulomb, magnetic dipole, Lorentz, gravitomagnetic, frame-dragging, tidal locking, Yukawa (screened nuclear), spin-orbit, external fields, and scalar field gradients
- **4 relativistic corrections** -- Einstein-Infeld-Hoffmann, Darwin EM, Bazanski cross-term, and scalar Breit, all velocity-Verlet corrected
- **Signal delay** -- forces propagate at the speed of light via a three-phase light-cone solver (Newton-Raphson segment search, exact quadratic solve, constant-velocity extrapolation) on per-particle circular history buffers
- **Higgs field** -- Mexican hat potential with spontaneous symmetry breaking, mass generation (`m_eff = baseMass * |phi|`), and electroweak-like phase transitions
- **Axion field** -- quadratic potential with scalar EM coupling (position-dependent fine structure constant) and Peccei-Quinn CP violation that flips sign for antimatter
- **Pion force carriers** -- massive bosons emitted via scalar Larmor radiation, with species (pi0/pi+/pi-), Lorentz-boosted decay kinematics, GR deflection, and Coulomb interaction
- **Black holes** -- Kerr-Newman horizons, ergosphere visualization, Hawking radiation with evaporation, and pair production from energetic photons
- **4 radiation channels** -- Larmor dipole, EM quadrupole, gravitational wave quadrupole, and pion emission
- **3 non-trivial topologies** -- torus, Klein bottle, and real projective plane with topology-aware minimum image
- **WebGPU compute backend** -- 52 WGSL shaders (10,120 lines) with Barnes-Hut tree build, instanced rendering, and automatic CPU fallback
- **Boris integrator** -- exact energy-conserving magnetic rotation with adaptive substepping (up to 32 substeps per tick)
- **Barnes-Hut O(N log N)** -- pool-based structure-of-arrays quadtree with zero per-frame allocation

## What It Simulates

Everything runs in natural units (c = 1, G = 1, h = 1). Particles store proper velocity **w** = gamma * v as their state variable, so the speed of light is enforced automatically: coordinate velocity **v** = **w** / sqrt(1 + w^2) always satisfies |v| < c. The same trick applies to spin.

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

Two dynamical scalar fields live on 64x64 grids (128x128 on GPU), sharing a common PQS (cubic B-spline) infrastructure for C^2-smooth interpolation and gradients. Both support topology-aware boundary conditions, receive energy from merge collisions as propagating wave packets, and gravitate particles and each other via weak-field GR corrections.

- **Higgs field** -- Mexican hat potential V(f) = -1/2 u^2 f^2 + 1/4 l f^4. The field spontaneously breaks symmetry to VEV = 1. Particles acquire effective mass m_eff = baseMass * |f(x)|. At VEV, particles have full mass; when kinetic energy drives the field to zero, particles become effectively massless -- a classical analog of the **electroweak phase transition**. Merge collisions excite oscillations around the VEV (**Higgs bosons**). Portal coupling to the axion field.

- **Axion field** -- Quadratic potential V(a) = 1/2 m_a^2 a^2 with vacuum at a = 0. Scalar EM coupling (aF^2) makes the fine structure constant position-dependent. Peccei-Quinn coupling flips sign for antimatter, implementing CP violation that vanishes at vacuum. Portal coupling to the Higgs field.

### Radiation and Force Carriers

- **Larmor radiation** -- Landau-Lifshitz force with analytical jerk from gravity + Coulomb + Yukawa, numerical backward-difference for residuals.
- **EM quadrupole** -- d^3 Q_ij/dt^3 formula with TT-projected angular emission via rejection sampling.
- **GW quadrupole** -- Trace-free mass quadrupole d^3 I^TF_ij/dt^3 with COM-relative coordinates.
- **Pion emission** -- Scalar Larmor formula P = g^2 F^2/3. Three species (pi0, pi+, pi-) with Lorentz-boosted decay kinematics, GR deflection, Higgs-modulated mass, and pi+pi- annihilation.
- **Photon/pion lensing** -- gravitational deflection via Barnes-Hut tree walk (2x for null geodesics, (1+v^2) for massive bosons).

### Additional Physics

- **Spin-orbit coupling** -- Energy transfer between translational and rotational KE via Stern-Gerlach and Mathisson-Papapetrou kicks.
- **Disintegration** -- Roche-limit fragmentation with Eggleton (1983) Roche lobe formula and continuous L1 mass transfer.
- **Black hole mode** -- Kerr-Newman horizons r+ = M + sqrt(M^2 - a^2 - Q^2). Ergosphere visualization. Hawking radiation; extremal BHs stop radiating. Sub-threshold BHs evaporate with photon burst.
- **Cosmological expansion** -- Hubble flow from domain center with peculiar velocity redshift.
- **Antimatter & pair production** -- Right-click spawns antimatter. Matter-antimatter mergers annihilate with photon emission. Energetic photons near massive bodies produce particle-antiparticle pairs.

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
| `Ctrl+S` / `Ctrl+L` | Quick save / load |
| `Ctrl+Shift+S` / `Ctrl+Shift+L` | Download / upload state file |
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
