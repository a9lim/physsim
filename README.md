# No-Hair

A relativistic N-body simulation exploring how gravity, electromagnetism, scalar fields, and spin shape particle dynamics in 2D. Particles have mass, charge, and angular momentum; forces propagate at finite speed; scalar fields live on dynamical grids; matter and antimatter obey different symmetries.

**[Live Demo](https://a9l.im/physsim)** | Part of the [a9l.im](https://a9l.im) portfolio

## What It Simulates

Everything runs in natural units (c = 1, G = 1, h-bar = 1). Particles store proper velocity **w** = gamma \* **v** as their state variable, so the speed of light is enforced automatically: coordinate velocity **v** = **w** / sqrt(1 + w^2) always satisfies |v| < c. The same trick applies to spin.

### Forces

- **Gravity** -- Newtonian 1/r^2 attraction between all massive particles, Plummer-softened.
- **Coulomb** -- 1/r^2 electrostatic force; like charges repel, opposites attract.
- **Magnetic dipole** -- 3 mu\_1 mu\_2 / r^4 interaction between spinning charged particles. Magnetic moment mu = q \* omega \* r^2 / 5 (uniform charge sphere).
- **Lorentz** -- q(**v** x **B**) force from moving charges and spinning dipoles. Handled exactly by Boris rotation.
- **Gravitomagnetic dipole** -- 3 L\_1 L\_2 / r^4 interaction between spinning masses. Co-rotating masses attract (GEM sign convention).
- **Frame-dragging** -- 4m(**v** x **B**\_g) from moving/spinning masses, plus a torque that aligns neighboring spins.
- **Tidal locking** -- Dissipative tidal torque drives spin toward synchronous rotation with the orbital frequency. Always active when gravity is on.
- **Yukawa** -- Screened nuclear force V(r) = -g^2 exp(-mu\*r)/r between massive particles, with configurable coupling and range parameter mu (= pion mass). Includes analytical jerk for radiation reaction. When 1PN is enabled, receives O(v^2/c^2) **scalar Breit corrections** from massive scalar boson exchange.

### Relativistic Corrections

- **1PN (Einstein-Infeld-Hoffmann)** -- O(v^2/c^2) correction to gravity. Produces perihelion precession at the GR rate.
- **1PN (Darwin EM)** -- O(v^2/c^2) correction to electromagnetism from the Darwin Lagrangian.
- **1PN (Bazanski cross-term)** -- Mixed gravity-EM 1PN interaction. Position-dependent 1/r^3 force coupling mass and charge.
- **1PN (Scalar Breit)** -- O(v^2/c^2) correction for massive scalar boson exchange (Yukawa). Full Breit Hamiltonian with radial and tangential components. Velocity-Verlet corrected.
- **Signal delay** -- Forces use source positions from the past light cone, solved analytically with a three-phase algorithm (Newton-Raphson segment search, exact quadratic solve, constant-velocity extrapolation). Includes the Liénard-Wiechert `(1 - n̂·v)^{-3}` aberration factor. Newly created particles respect light-cone causality; deleted particles continue exerting forces until their signal fades past all observers.

### Scalar Fields

Two dynamical scalar fields live on 64x64 grids, sharing a common PQS (cubic B-spline) infrastructure for C^2-smooth interpolation and gradients. Both support topology-aware boundary conditions and receive energy from merge collisions as propagating wave packets (field excitations).

- **Higgs field** -- Mexican hat potential V(phi) = -1/2 mu^2 phi^2 + 1/4 lambda phi^4. The field spontaneously breaks symmetry to a vacuum expectation value (VEV) of 1. Particles acquire effective mass from the local field value: m\_eff = baseMass \* |phi(x)|. At VEV, particles have their full mass; when the field is driven to zero by high local kinetic energy, particles become effectively massless -- a classical analog of the **electroweak phase transition**. The field sources from particle mass via PQS deposition, exerts gradient forces, and is critically damped. Merge collisions excite propagating oscillations around the VEV -- the simulation's analog of **Higgs bosons**.

- **Axion field** -- Quadratic potential V(a) = 1/2 m\_a^2 a^2 with vacuum at a = 0 (no symmetry breaking). The field oscillates at frequency m\_a, exactly as cosmological axion dark matter does. Two independent coupling channels:

  - **Scalar EM coupling (aF^2)** -- active when Coulomb is on. Makes the fine structure constant position-dependent: alpha\_eff(x) = alpha \* (1 + g\*a(x)). Charged particles source the field proportional to q^2 and feel gradient forces. All electromagnetic interactions (Coulomb, magnetic dipole, Biot-Savart) use the local coupling.

  - **Peccei-Quinn coupling (aGG~ analog)** -- active when Yukawa is on. A pseudoscalar coupling that **flips sign for antimatter**, implementing the Peccei-Quinn mechanism for CP violation. Matter sources the field as +g\*m, antimatter as -g\*m. The Yukawa coupling is locally modulated: g^2\_eff = g^2(1 + g\*a) for matter, g^2(1 - g\*a) for antimatter. At the vacuum a = 0, both are identical -- **CP is conserved** (the PQ solution to the strong CP problem). When the field is displaced, matter and antimatter experience different nuclear binding strengths.

  Merge collisions excite propagating oscillations around a = 0 -- the simulation's analog of **axion particles**.

### Radiation and Force Carriers

- **Larmor radiation** -- Accelerating charges lose energy via the Landau-Lifshitz force (analytical jerk from gravity + Coulomb + Yukawa, numerical backward-difference for residual forces, power-dissipation terms with relativity). Photons are emitted in a dipole pattern with relativistic aberration.
- **EM quadrupole radiation** -- d^3 Q\_ij/dt^3 formula with TT-projected angular emission via rejection sampling.
- **Gravitational wave radiation** -- Trace-free mass quadrupole d^3 I^TF\_ij/dt^3 formula with COM-relative coordinates and per-particle energy extraction. Gravitons rendered red.
- **Pion emission (scalar Larmor)** -- Yukawa interactions radiate massive pions with power P = g^2 m^2 a^2 / 3. The scalar charge Q = g\*m (Yukawa couples to mass); the 1/3 angular factor reflects the single polarization of spin-0 radiation (vs 2/3 for spin-1 EM). Pions travel at v < c with proper velocity, experience gravitational deflection with the correct massive-particle factor (1 + v^2), and **decay into photons** (pi0 -> 2 gamma Lorentz-boosted from rest frame, pi+/- -> 1 gamma along flight). Probabilistic decay via half-life, not lifetime.
- **Photon & pion absorption** -- Quadtree overlap query transfers momentum (and charge for pi+/-) to absorbing particles. Self-absorption guards prevent immediate reabsorption.
- **Field excitations** -- Inelastic merge collisions deposit Gaussian wave packets into active scalar fields. The existing Klein-Gordon equation propagates them naturally as dispersive waves.

### Additional Physics

- **Spin-orbit coupling** -- Energy transfer between translational and rotational KE via Stern-Gerlach (mu \* grad(Bz)) and Mathisson-Papapetrou (L \* grad(Bgz)) kicks.
- **Disintegration** -- Roche-limit fragmentation when tidal + centrifugal + Coulomb stress exceeds self-gravity. Includes Roche lobe overflow using the full Eggleton (1983) formula with continuous L1 mass transfer.
- **Black hole mode** -- Kerr-Newman horizons: r+ = M + sqrt(M^2 - a^2 - Q^2) with spin parameter a = I\*|omega|/M. Ergosphere visualization. Hawking radiation at the surface gravity temperature; extremal BHs stop radiating. Sub-threshold BHs evaporate with a final photon burst.
- **Cosmological expansion** -- Hubble flow v\_H = H\*r from domain center with peculiar velocity redshift. Locks boundary to despawn.
- **Antimatter & pair production** -- Particles carry an antimatter flag. Matter-antimatter mergers annihilate the lesser mass with photon emission. Energetic photons near massive bodies spontaneously produce particle-antiparticle pairs.
- **External background fields** -- Uniform gravitational (F = mg), electric (F = qE), and magnetic (Bz) fields with configurable strength and direction. External Bz integrated exactly via Boris rotation.

### Integrator

Boris integrator (half-kick / rotate / half-kick / drift) with adaptive substepping. The Boris rotation handles velocity-dependent magnetic forces exactly, preserving |v| through each step. Substep count adapts to acceleration magnitude and cyclotron frequency, capped at 32 substeps per frame. The four 1PN sectors (EIH, Darwin EM, Bazanski, scalar Breit) use a velocity-Verlet correction pass for second-order accuracy. Scalar fields evolve via Störmer-Verlet (kick-drift-kick, O(dt²)) between the drift and force-recomputation steps.

### Algorithms

- **Barnes-Hut** -- Toggleable O(N log N) quadtree approximation with a pool-based structure-of-arrays layout and zero per-frame allocation. When off, exact O(N^2) pairwise forces preserve Newton's third law. Signal delay applied at leaf level; distant aggregates use current positions.
- **Collisions** -- Pass-through, elastic bounce (Hertz contact with configurable spin friction), or merge (conserves mass, charge, momentum, angular momentum; tracks relativistic KE loss for field excitations).
- **Topological boundaries** -- Periodic loop mode with torus, Klein bottle, or real projective plane identification. Minimum-image separation handles non-orientable crossings with correct velocity/spin flips.
- **PQS grid infrastructure** -- Shared cubic B-spline (order 3) framework for scalar fields. 4x4 = 16 node stencil per particle gives C^2-smooth interpolation and gradients. Topology-aware deposition, interior fast-path Laplacian, pre-allocated weight arrays for zero-alloc hot paths.

## Controls

| Input | Action |
|-------|--------|
| Left click | Spawn particle (Place / Shoot / Orbit mode) |
| Left drag | Set velocity (Shoot mode) |
| Right click | Remove particle |
| Scroll | Zoom |
| `Space` | Pause / resume |
| `.` | Step forward one frame |
| `1`--`9` | Load preset directly |
| `V` / `F` / `C` | Toggle velocity / force / component vectors |
| `T` / `S` | Toggle theme / sidebar |
| `A` | Toggle antimatter spawn mode |
| `?` | Keyboard shortcut help |

### Sidebar Tabs

1. **Settings** -- Particle mass / charge / spin sliders, spawn mode, force toggles (gravity, Coulomb, magnetic, gravitomagnetic, Yukawa, axion, Higgs), physics toggles (relativity, 1PN, black hole, spin-orbit, radiation, disintegration). Preset selector with 19 scenarios across four groups.
2. **Engine** -- Barnes-Hut, collision mode, bounce friction, boundary mode, topology, external fields (g, E, Bz), visual overlays (heatmap, vectors, trails), sim speed (1--128x), cosmological expansion.
3. **Stats** -- Energy breakdown (linear KE, spin KE, PE, Higgs field, axion field, radiated, drift), conserved quantities (momentum, angular momentum).
4. **Particle** -- Selected particle details (mass, charge, spin, speed, gamma, per-force breakdown with 11 component vectors), phase space plot, effective potential plot.

### Presets

19 scenarios across four groups:

| Group | Presets |
|-------|---------|
| **Gravity** | Kepler Orbit, Perihelion Precession, Inspiral, Tidal Locking, Roche Limit, Hawking Radiation |
| **Electromagnetism** | Hydrogen Atom, Bremsstrahlung, Magnetic Dipoles |
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

Zero-dependency vanilla JavaScript with Canvas 2D rendering. All physics and rendering code is hand-written -- no physics engines, no WebGL, no libraries. Structure-of-arrays quadtree, circular history buffers for signal delay, cubic B-spline field infrastructure, adaptive substepping -- all designed for zero per-frame allocation in hot paths.

## Architecture

```
main.js                     417 lines  Simulation class, fixed-timestep loop, pair production, pion loop, window.sim
index.html                  509 lines  UI: 4-tab sidebar, reference overlay, zoom controls, field sliders
styles.css                  269 lines  Project-specific CSS overrides
colors.js                    18 lines  Project color tokens (extends shared-tokens.js)
src/
  integrator.js            1382 lines  Physics: Boris substep loop, radiation, pion emission/absorption, field excitations,
                                       tidal, GW quadrupole, expansion, Roche, external fields, Hertz bounce, scalar fields
  reference.js              697 lines  Physics reference content (KaTeX math)
  presets.js                694 lines  19 preset scenarios (Gravity / EM / Exotic / Cosmological)
  ui.js                     529 lines  DOM setup, declarative toggle dependencies, info tips, shortcuts
  renderer.js               528 lines  Canvas 2D: particles, trails, spin rings, vectors, photons, pions, field overlays
  forces.js                 477 lines  Pairwise + Barnes-Hut force accumulation, 1PN (4 sectors), Yukawa + PQ modulation
  scalar-field.js           392 lines  ScalarField base: PQS grid, topology-aware deposition, Laplacian, C^2 gradients,
                                       field energy, field excitations
  quadtree.js               274 lines  SoA pool-based Barnes-Hut tree (zero GC)
  input.js                  270 lines  Mouse/touch, Place/Shoot/Orbit spawn modes
  signal-delay.js           255 lines  Three-phase light-cone solver on circular history buffers
  heatmap.js                248 lines  Gravitational + electric + Yukawa potential field overlay
  axion-field.js            223 lines  AxionField: quadratic potential, scalar aF^2 coupling, Peccei-Quinn CP violation
  higgs-field.js            209 lines  HiggsField: Mexican hat potential, mass modulation, thermal phase transitions
  save-load.js              205 lines  State serialization, quick save/load, file export/import
  effective-potential.js    204 lines  V_eff(r) sidebar canvas with axMod/yukMod modulation
  potential.js              164 lines  PE computation (7 terms, pairwise + tree traversal)
  energy.js                 160 lines  KE, PE, field energy, momentum, angular momentum
  collisions.js             138 lines  Merge, annihilation, baseMass conservation, relativistic merge KE tracking
  stats-display.js          131 lines  Sidebar energy/momentum/drift readout
  config.js                 131 lines  Named constants (softening, BH, numerical, pion, field, pair production)
  particle.js               128 lines  Particle: 11 force Vec2s, axMod/yukMod, baseMass, antimatter, signal delay history
  phase-plot.js             117 lines  Phase space r-v_r plot (512-sample ring buffer)
  topology.js               112 lines  Torus / Klein / RP^2 min-image + wrapping
  pion.js                   123 lines  Massive Yukawa force carrier: proper velocity, (1+v^2) GR deflection, Lorentz-boosted decay
  vec2.js                    61 lines  2D vector math
  boson-utils.js             58 lines  Shared BH tree walk for photon/pion gravitational lensing
  photon.js                  45 lines  Radiation photon with BH tree lensing
  relativity.js              22 lines  Proper velocity helpers
```

## Sibling Projects

- [Cellular Metabolism](https://github.com/a9lim/biosim) -- [a9l.im/biosim](https://a9l.im/biosim)
- [Redistricting Simulator](https://github.com/a9lim/gerry) -- [a9l.im/gerry](https://a9l.im/gerry)

## License

[AGPL-3.0](LICENSE)
