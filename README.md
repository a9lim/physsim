# No-Hair

A relativistic N-body simulation exploring how gravity, electromagnetism, scalar fields, and spin shape particle dynamics in 2D. Particles have mass, charge, and angular momentum; forces propagate at finite speed; scalar fields live on dynamical grids; matter and antimatter obey different symmetries.

**[Live Demo](https://a9l.im/physsim)** | Part of the [a9l.im](https://a9l.im) portfolio

## What It Simulates

Everything runs in natural units (c = 1, G = 1, ħ = 1). Particles store proper velocity **w** = γ**v** as their state variable, so the speed of light is enforced automatically: coordinate velocity **v** = **w** / √(1 + w²) always satisfies |v| < c. The same trick applies to spin.

### Forces

- **Gravity** — Newtonian 1/r² attraction between all massive particles, Plummer-softened.
- **Coulomb** — 1/r² electrostatic force; like charges repel, opposites attract.
- **Magnetic dipole** — 3μ₁μ₂/r⁴ interaction between spinning charged particles. Magnetic moment μ = qωr²/5 (uniform charge sphere).
- **Lorentz** — q(**v** × **B**) force from moving charges and spinning dipoles. Handled exactly by Boris rotation.
- **Gravitomagnetic dipole** — 3L₁L₂/r⁴ interaction between spinning masses. Co-rotating masses attract (GEM sign convention).
- **Frame-dragging** — 4m(**v** × **B**_g) from moving/spinning masses, plus a torque that aligns neighboring spins.
- **Tidal locking** — Dissipative tidal torque drives spin toward synchronous rotation with the orbital frequency. Always active when gravity is on.
- **Yukawa** — Screened nuclear force V(r) = -g²e^{-μr}/r between massive particles, with configurable coupling and range parameter μ (= pion mass). Includes analytical jerk for radiation reaction. When 1PN is enabled, receives O(v²/c²) **scalar Breit corrections** from massive scalar boson exchange.

### Relativistic Corrections

- **1PN (Einstein-Infeld-Hoffmann)** — O(v²/c²) correction to gravity. Produces perihelion precession at the GR rate.
- **1PN (Darwin EM)** — O(v²/c²) correction to electromagnetism from the Darwin Lagrangian.
- **1PN (Bazanski cross-term)** — Mixed gravity-EM 1PN interaction. Position-dependent 1/r³ force coupling mass and charge.
- **1PN (Scalar Breit)** — O(v²/c²) correction for massive scalar boson exchange (Yukawa). Full Breit Hamiltonian with radial and tangential components. Velocity-Verlet corrected.
- **Signal delay** — Forces use source positions from the past light cone, solved analytically with a three-phase algorithm (Newton-Raphson segment search, exact quadratic solve, constant-velocity extrapolation). Includes the Liénard-Wiechert (1 - n̂·v)⁻³ aberration factor. Newly created particles respect light-cone causality; deleted particles continue exerting forces until their signal fades past all observers.

### Scalar Fields

Two dynamical scalar fields live on 64×64 grids, sharing a common PQS (cubic B-spline) infrastructure for C²-smooth interpolation and gradients. Both support topology-aware boundary conditions, receive energy from merge collisions as propagating wave packets, and gravitate particles and each other via weak-field GR corrections.

- **Higgs field** — Mexican hat potential V(φ) = -½μ²φ² + ¼λφ⁴. The field spontaneously breaks symmetry to VEV = 1. Particles acquire effective mass m_eff = baseMass · |φ(x)|. At VEV, particles have full mass; when kinetic energy drives the field to zero, particles become effectively massless — a classical analog of the **electroweak phase transition**. Merge collisions excite oscillations around the VEV (**Higgs bosons**).

- **Axion field** — Quadratic potential V(a) = ½m_a²a² with vacuum at a = 0. Two coupling channels:

  - **Scalar EM (aF²)** — active when Coulomb is on. Makes the fine structure constant position-dependent: α_eff(x) = α(1 + g·a(x)). Charged particles source the field ∝ q² and feel gradient forces.

  - **Peccei-Quinn (aGG̃ analog)** — active when Yukawa is on. A pseudoscalar coupling that **flips sign for antimatter**, implementing the PQ mechanism for CP violation. At vacuum a = 0, matter and antimatter interact identically — **CP is conserved**.

  Merge collisions excite oscillations around a = 0 (**axion particles**).

### Radiation and Force Carriers

- **Larmor radiation** — Landau-Lifshitz force with analytical jerk from gravity + Coulomb + Yukawa, numerical backward-difference for residuals. Dipole photon emission with relativistic aberration.
- **EM quadrupole** — d³Q_ij/dt³ formula with TT-projected angular emission via rejection sampling.
- **GW quadrupole** — Trace-free mass quadrupole d³I^TF_ij/dt³ with COM-relative coordinates. Gravitons rendered red.
- **Pion emission (scalar Larmor)** — Yukawa interactions radiate massive pions with P = g²m²a²/3 (spin-0 angular factor 1/3 vs 2/3 for spin-1 EM). Pions travel at v < c, experience gravitational deflection with factor (1 + v²), and **decay**: π⁰ → 2γ (Lorentz-boosted), π⁺ → e⁺ + γ, π⁻ → e⁻ + γ (two-body kinematics in rest frame).
- **Photon & pion absorption** — Quadtree overlap query transfers momentum (and charge for π±) to absorbing particles.
- **Boson gravity** — Photons and pions gravitate particles and each other. GR deflection: 2× for photons (null geodesic), (1+v²)× for pions (massive).
- **Field excitations** — Inelastic merges deposit Gaussian wave packets into active scalar fields, propagated by the Klein-Gordon equation.

### Additional Physics

- **Spin-orbit coupling** — Energy transfer between translational and rotational KE via Stern-Gerlach and Mathisson-Papapetrou kicks.
- **Disintegration** — Roche-limit fragmentation with Eggleton (1983) Roche lobe formula and continuous L1 mass transfer.
- **Black hole mode** — Kerr-Newman horizons r₊ = M + √(M² - a² - Q²). Ergosphere visualization. Hawking radiation; extremal BHs stop radiating. Sub-threshold BHs evaporate with photon burst.
- **Cosmological expansion** — Hubble flow from domain center with peculiar velocity redshift.
- **Antimatter & pair production** — Right-click spawns antimatter. Matter-antimatter mergers annihilate with photon emission. Energetic photons near massive bodies produce particle-antiparticle pairs.
- **External fields** — Uniform gravitational, electric, and magnetic background fields with configurable strength and direction.

### Integrator

Boris integrator (half-kick / rotate / half-kick / drift) with adaptive substepping based on acceleration and cyclotron frequency, capped at 32 substeps. Boris rotation handles magnetic forces exactly, preserving |v|. The four 1PN sectors use velocity-Verlet correction for second-order accuracy. Scalar fields evolve via Störmer-Verlet (KDK).

### Algorithms

- **Barnes-Hut** — Toggleable O(N log N) quadtree with pool-based structure-of-arrays layout and zero per-frame allocation. Signal delay at leaf level; distant aggregates use current positions.
- **Collisions** — Pass-through, elastic bounce (Hertz contact), or merge (conserves mass, charge, momentum, angular momentum; tracks relativistic KE for field excitations).
- **Topological boundaries** — Periodic loop with torus, Klein bottle, or real projective plane identification. Minimum-image separation handles non-orientable crossings.
- **PQS grid** — Cubic B-spline (order 3) infrastructure for scalar fields. 4×4 stencil, C²-smooth interpolation/gradients, topology-aware deposition, interior fast-path Laplacian, zero-alloc hot paths.

## Controls

| Input | Action |
|-------|--------|
| Left click | Spawn particle / select matter / delete antimatter |
| Left drag | Spawn with velocity |
| Right click | Spawn antimatter / select antimatter / delete matter |
| Right drag | Spawn antimatter with velocity |
| Scroll | Zoom |
| `Space` | Pause / resume |
| `.` | Step forward one frame |
| `1`–`9` | Load preset directly |
| `V` / `F` / `C` | Toggle velocity / force / component vectors |
| `T` / `S` | Toggle theme / sidebar |
| `?` | Keyboard shortcut help |

### Sidebar Tabs

1. **Settings** — Particle mass / charge / spin sliders, spawn mode, force toggles (gravity, Coulomb, magnetic, gravitomagnetic, Yukawa, axion, Higgs), physics toggles (relativity, 1PN, black hole, spin-orbit, radiation, disintegration). 19 presets across four groups.
2. **Engine** — Barnes-Hut, collision mode, bounce friction, boundary/topology, external fields, visual overlays, sim speed (1–64×), cosmological expansion.
3. **Stats** — Energy breakdown (linear KE, spin KE, PE, field energies, radiated, drift), conserved quantities (momentum, angular momentum).
4. **Particle** — Selected particle details, per-force breakdown (11 components), phase space plot, effective potential plot.

### Presets

19 scenarios across four groups:

| Group | Presets |
|-------|---------|
| **Gravity** | Kepler Orbits, Precession, Binary Inspiral, Tidal Lock, Roche Limit, Hawking Evaporation |
| **Electromagnetism** | Atom, Bremsstrahlung, Magnetic Dipoles |
| **Exotic** | Atomic Nucleus, Axion Field, Pion Exchange, Higgs Mechanism, Higgs Boson, Axion Burst, Peccei–Quinn, Phase Transition |
| **Cosmological** | Galaxy, Expanding Universe |

## Running Locally

```bash
# Serve from parent — shared design files load via absolute paths
cd path/to/a9lim.github.io && python -m http.server
# -> http://localhost:8000/physsim/
```

No build step, no dependencies, no npm. ES6 modules require HTTP (no `file://`).

## Tech

Zero-dependency vanilla JavaScript with Canvas 2D rendering. All physics and rendering code is hand-written — no physics engines, no WebGL, no libraries. Structure-of-arrays quadtree, circular history buffers for signal delay, cubic B-spline field infrastructure, adaptive substepping — all designed for zero per-frame allocation in hot paths.

## Architecture

```
main.js                   419 lines  Simulation class, fixed-timestep loop, pair production, pion loop
index.html                476 lines  UI: 4-tab sidebar, reference overlay, zoom controls, field sliders
styles.css                269 lines  Project-specific CSS overrides
colors.js                  18 lines  Project color tokens (extends shared-tokens.js)
src/
  integrator.js          1500 lines  Physics: Boris substep loop, radiation, pion emission/absorption, field excitations,
                                     tidal, GW quadrupole, expansion, Roche, external fields, Hertz bounce, scalar fields
  forces.js               734 lines  Pairwise + Barnes-Hut force accumulation, 1PN (4 sectors), boson gravity
  reference.js            714 lines  Physics reference content (KaTeX math)
  presets.js              688 lines  19 preset scenarios (Gravity / EM / Exotic / Cosmological)
  scalar-field.js         638 lines  ScalarField base: PQS grid, topology-aware deposition, Laplacian, C² gradients,
                                     field energy, excitations, particle-field gravity, self-gravity
  renderer.js             532 lines  Canvas 2D: particles, trails, spin rings, vectors, photons, pions, field overlays
  ui.js                   521 lines  DOM setup, declarative toggle dependencies, info tips, shortcuts
  heatmap.js              309 lines  Gravitational + electric + Yukawa potential field overlay
  axion-field.js          297 lines  AxionField: quadratic potential, scalar aF² coupling, Peccei-Quinn CP violation
  higgs-field.js          296 lines  HiggsField: Mexican hat potential, mass modulation, thermal phase transitions
  quadtree.js             274 lines  SoA pool-based Barnes-Hut tree (zero GC)
  signal-delay.js         257 lines  Three-phase light-cone solver on circular history buffers
  input.js                249 lines  Mouse/touch, left/right-click symmetry (matter/antimatter)
  effective-potential.js  214 lines  V_eff(r) sidebar canvas with axMod/yukMod modulation
  potential.js            211 lines  PE computation (7 terms, pairwise + tree traversal)
  save-load.js            205 lines  State serialization, quick save/load, file export/import
  energy.js               191 lines  KE, PE, field energy, momentum, angular momentum
  pion.js                 187 lines  Massive Yukawa force carrier: proper velocity, (1+v²) GR deflection, decay channels
  config.js               152 lines  Named constants, mode enums, helpers
  collisions.js           142 lines  Merge, annihilation, baseMass conservation, relativistic merge KE tracking
  particle.js             132 lines  Particle: 11 force Vec2s, axMod/yukMod, baseMass, signal delay history
  topology.js             131 lines  Torus / Klein / RP² min-image + wrapping
  phase-plot.js           128 lines  Phase space r-v_r plot (512-sample ring buffer)
  stats-display.js        123 lines  Sidebar energy/momentum/drift readout
  vec2.js                  61 lines  2D vector math
  boson-utils.js           58 lines  Shared BH tree walk for photon/pion gravitational lensing
  massless-boson.js        45 lines  Radiation photon/graviton with BH tree lensing
  relativity.js            22 lines  Proper velocity helpers
```

## Sibling Projects

- [Cellular Metabolism](https://github.com/a9lim/biosim) — [a9l.im/biosim](https://a9l.im/biosim)
- [Redistricting Simulator](https://github.com/a9lim/gerry) — [a9l.im/gerry](https://a9l.im/gerry)

## License

[AGPL-3.0](LICENSE)
