# No-Hair

A relativistic N-body simulation exploring how gravity, electromagnetism, and spin shape particle dynamics in 2D. Particles have mass, charge, and angular momentum; forces propagate at finite speed.

**[Live Demo](https://a9l.im/physsim)** | Part of the [a9l.im](https://a9l.im) portfolio

## What It Simulates

Everything runs in natural units (c = 1, G = 1). Particles store proper velocity **w** = gamma * **v** as their state variable, so the speed of light is enforced automatically: coordinate velocity **v** = **w** / sqrt(1 + w^2) always satisfies |v| < c. The same trick applies to spin.

### Forces

- **Gravity** -- Newtonian 1/r^2 attraction between all massive particles, Plummer-softened.
- **Coulomb** -- 1/r^2 electrostatic force; like charges repel, opposites attract.
- **Magnetic dipole** -- 3 mu_1 mu_2 / r^4 interaction between spinning charged particles. Magnetic moment mu = q * omega * r^2 / 5 (uniform charge sphere).
- **Lorentz** -- q(v x B) force from moving charges and spinning dipoles. Handled exactly by Boris rotation.
- **Gravitomagnetic dipole** -- 3 L_1 L_2 / r^4 interaction between spinning masses. Co-rotating masses attract (GEM sign convention).
- **Frame-dragging** -- 4m(v x B_g) from moving/spinning masses, plus a torque that aligns neighboring spins.
- **1PN (Einstein-Infeld-Hoffmann)** -- O(v^2/c^2) correction to gravity. Produces perihelion precession at the GR rate.
- **1PN (Darwin EM)** -- O(v^2/c^2) correction to electromagnetism from the Darwin Lagrangian.
- **1PN (Bazanski cross-term)** -- Mixed gravity-EM 1PN interaction from the Bazanski Lagrangian. Position-dependent 1/r^3 force coupling mass and charge.
- **Larmor radiation** -- Accelerating charges lose energy via the full Landau-Lifshitz force (jerk + 1/c^2 power-dissipation terms) and emit photons.
- **Stern-Gerlach / Mathisson-Papapetrou** -- Translational forces from spin-field gradient coupling (EM and gravitational).
- **Yukawa** -- Screened force V(r) = -g^2 exp(-mu*r)/r between massive particles, with configurable coupling strength and range. Emits **pions** as massive force carriers via scalar Larmor radiation (P = g^2 m^2 a^2 / 3). Pions travel at v < c with proper velocity, experience gravitational deflection with the correct massive-particle factor (1 + v^2), and decay into photons (pi0 -> 2 gamma, pi+/- -> 1 gamma).
- **Axion dark matter** -- Dynamical axion-like scalar field on a 64x64 grid. Scalar aF² coupling makes the EM coupling constant position-dependent. Charged particles source the field (proportional to q²) and feel gradient forces. Field oscillates at frequency m_a, like real axion dark matter.
- **Higgs scalar field** -- Dynamical real scalar field on a 48x48 grid with Mexican hat potential V(phi) = -1/2 mu^2 phi^2 + 1/4 lambda phi^4. Particles source the field via weak CIC deposition (Yukawa coupling), acquire effective mass m = baseMass * |phi/v|, and feel gradient forces F = -(baseMass/v) * coupling * grad(phi). Analytical self-force subtraction prevents grid artifacts. Adaptive critical damping, thermal corrections restore symmetry at high energy density (phase transitions).

### Additional Physics

- **Signal delay** -- Forces use source positions from the past light cone, solved analytically with a three-phase algorithm (Newton-Raphson segment search, exact quadratic solve, constant-velocity extrapolation).
- **Spin-orbit coupling** -- Energy transfer between translational and rotational kinetic energy via field gradients.
- **Disintegration** -- Roche-limit fragmentation when tidal, centrifugal, and Coulomb stresses exceed self-gravity. Includes Roche lobe overflow with continuous mass transfer.
- **Tidal locking** -- Dissipative tidal torque drives spin toward synchronous rotation with the orbital frequency.
- **Photon emission & absorption** -- Accelerating charges emit photons in a Larmor dipole pattern with relativistic aberration. Photons carry energy and momentum, experience gravitational lensing (2x Newtonian deflection), and are absorbed on contact.
- **Field excitations** -- Merge collisions deposit Gaussian wave packets into active scalar fields (Higgs and axion), exciting oscillations that propagate via the Klein-Gordon equation. Higgs excitations ripple around VEV = 1 (Higgs boson analog); axion excitations ripple around vacuum a = 0.
- **Gravitational wave radiation** -- Mass and EM quadrupole formula with hybrid analytical+numerical jerk, orbital decay via tangential drag, and graviton emission (rendered red).
- **Black hole mode** -- Kerr-Newman horizons: r+ = M + sqrt(M^2 - a^2 - Q^2) with spin parameter a = J/M and charge Q. Collisions lock to merge. Hawking radiation at the Kerr-Newman surface gravity temperature; extremal black holes stop radiating. Sub-threshold black holes evaporate with a final photon burst.
- **Cosmological expansion** -- Hubble flow with peculiar velocity redshift. Locks boundary mode to despawn.
- **Antimatter & pair production** -- Particles carry an antimatter flag; matter-antimatter mergers annihilate with photon emission. Energetic photons near massive bodies can spontaneously produce particle-antiparticle pairs.
- **External background fields** -- Uniform gravitational (F=mg), electric (F=qE), and magnetic (Bz) fields with configurable strength and direction. External Bz integrated exactly via Boris rotation.

### Integrator

Boris integrator (half-kick / rotate / half-kick / drift) with adaptive substepping. The Boris rotation handles velocity-dependent magnetic forces exactly, preserving |v| through each step. Substep count adapts to acceleration magnitude and cyclotron frequency, capped at 32 substeps per frame. The three 1PN sectors (EIH, Darwin EM, Bazanski) use a velocity-Verlet correction pass for second-order accuracy.

### Algorithms

- **Barnes-Hut** -- Toggleable O(N log N) quadtree approximation with a pool-based structure-of-arrays layout and zero per-frame allocation. When off, exact O(N^2) pairwise forces preserve Newton's third law.
- **Collisions** -- Pass-through, elastic bounce (with configurable spin friction and relativistic Lorentz-boost resolution), or merge (conserves mass, charge, momentum, angular momentum).
- **Topological boundaries** -- Periodic loop mode with torus, Klein bottle, or real projective plane identification. Minimum-image separation handles non-orientable crossings with correct velocity/spin flips.

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

1. **Settings** -- Particle mass / charge / spin sliders, spawn mode, force toggles (gravity, Coulomb, magnetic, gravitomagnetic, Yukawa, axion, Higgs), physics toggles (relativity, 1PN, black hole, spin-orbit, radiation, disintegration). Preset selector with four category groups (Gravity, EM, Exotic, Cosmological) and 18 scenarios.
2. **Engine** -- Barnes-Hut, collision mode, bounce friction (visible only in bounce mode), boundary mode, topology, external fields (g, E, Bz), visual overlays (heatmap, vectors, trails), sim speed (1–128x, default 64x), cosmological expansion.
3. **Stats** -- Energy breakdown (linear KE, spin KE, PE, field, Higgs field, radiated, drift), conserved quantities (momentum, angular momentum).
4. **Particle** -- Selected particle details (mass, charge, spin, speed, gamma, per-force breakdown), phase space plot, effective potential plot.

## Running Locally

```bash
# Serve from parent -- shared design files load via absolute paths
cd path/to/a9lim.github.io && python -m http.server
# -> http://localhost:8000/physsim/
```

No build step, no dependencies, no npm. ES6 modules require HTTP (no `file://`).

## Tech

Zero-dependency vanilla JavaScript with Canvas 2D rendering. All physics and rendering code is hand-written. Structure-of-arrays quadtree, circular history buffers for signal delay, adaptive substepping -- all designed for zero per-frame allocation in hot paths.

## Architecture

```
main.js                     ~393 lines  Simulation class, fixed-timestep loop, Higgs field init, pion update loop, window.sim
index.html                  ~511 lines  UI structure, tab system, reference overlay, field sliders
styles.css                  ~245 lines  Project-specific CSS overrides, toggle/slider theme colors
colors.js                     18 lines  Project color tokens (extends shared-tokens.js)
src/
  integrator.js            ~1306 lines  Physics: adaptive Boris substep loop, radiation, pion emission/absorption, field excitations, tidal, GW, expansion, external fields, Hertz bounce
  ui.js                     ~527 lines  DOM setup, declarative toggle dependencies, info tips, shortcuts
  renderer.js               ~534 lines  Canvas 2D: particles, trails, vectors, photons, pions, glow, field overlays
  forces.js                 ~450 lines  Pairwise + Barnes-Hut force accumulation, 1PN, Yukawa
  presets.js                ~665 lines  Eighteen preset scenarios (Gravity / EM / Exotic / Cosmological)
  scalar-field.js           ~270 lines  ScalarField base class: PQS grid, topology-aware deposition, field excitations
  higgs-field.js            ~235 lines  HiggsField extends ScalarField: Mexican hat potential, mass modulation, phase transitions
  axion-field.js            ~214 lines  AxionField extends ScalarField: quadratic potential, scalar aF^2 coupling, EM modulation
  reference.js              ~697 lines  Extended physics reference (KaTeX math, pion/field excitation entries)
  quadtree.js               ~279 lines  SoA pool-based Barnes-Hut tree (zero GC)
  input.js                  ~262 lines  Mouse/touch, Place/Shoot/Orbit spawn modes
  signal-delay.js            249 lines  Light-cone solver on circular history buffers
  effective-potential.js    ~203 lines  V_eff(r) sidebar canvas, auto-scaling, current position marker
  save-load.js              ~205 lines  State serialization, quick save/load, file export/import
  heatmap.js                ~224 lines  Gravitational + electric + Yukawa potential field overlay
  potential.js              ~152 lines  PE computation (pairwise + tree traversal)
  energy.js                 ~153 lines  KE, PE, field energy, Higgs/Axion field energy, momentum, angular momentum
  stats-display.js          ~131 lines  Sidebar energy/momentum/drift readout, force breakdown
  particle.js               ~123 lines  Particle entity (incl. baseMass, forceHiggs, antimatter flag, _yukawaRadAccum)
  collisions.js             ~125 lines  Merge, antimatter annihilation, baseMass conservation, merge KE tracking
  pion.js                    121 lines  Pion: massive Yukawa force carrier, proper velocity, GR deflection, decay
  phase-plot.js              117 lines  Phase space plot (sidebar canvas)
  topology.js                112 lines  Torus / Klein / RP2 min-image + wrapping
  config.js                 ~136 lines  Named constants (softening, BH, numerical, pion, field excitation, pair production)
  vec2.js                     61 lines  2D vector math
  photon.js                   88 lines  Radiation photon entity with BH tree lensing
  relativity.js               22 lines  Proper velocity helpers
```

## Sibling Projects

- [Cellular Metabolism](https://github.com/a9lim/biosim) -- [a9l.im/biosim](https://a9l.im/biosim)
- [Redistricting Simulator](https://github.com/a9lim/gerry) -- [a9l.im/gerry](https://a9l.im/gerry)

## License

[AGPL-3.0](LICENSE)
