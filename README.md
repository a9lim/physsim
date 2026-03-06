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
- **Yukawa** -- Screened force V(r) = -g^2 exp(-mu*r)/r between massive particles, with configurable coupling strength and range.
- **Axion dark matter** -- Oscillating modulation of the electromagnetic coupling constant, simulating an axion-like background field.
- **Higgs scalar field** -- Dynamical real scalar field on a 48x48 grid with Mexican hat potential V(phi) = -1/2 mu^2 phi^2 + 1/4 lambda phi^4. Particles acquire effective mass m = baseMass * |phi/v| via Yukawa coupling to the field and feel gradient forces F = -(baseMass/v) * coupling * grad(phi). Thermal corrections restore symmetry at high energy density (phase transitions).

### Additional Physics

- **Signal delay** -- Forces use source positions from the past light cone, solved analytically with a three-phase algorithm (Newton-Raphson segment search, exact quadratic solve, constant-velocity extrapolation).
- **Spin-orbit coupling** -- Energy transfer between translational and rotational kinetic energy via field gradients.
- **Disintegration** -- Roche-limit fragmentation when tidal, centrifugal, and Coulomb stresses exceed self-gravity. Includes Roche lobe overflow with continuous mass transfer.
- **Tidal locking** -- Dissipative tidal torque drives spin toward synchronous rotation with the orbital frequency.
- **Photon emission & absorption** -- Accelerating charges emit photons in a Larmor dipole pattern with relativistic aberration. Photons carry energy and momentum, experience gravitational lensing (2x Newtonian deflection), and are absorbed on contact.
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

1. **Settings** -- Particle mass / charge / spin sliders, spawn mode, force toggles (gravity, Coulomb, magnetic, gravitomagnetic, Yukawa, axion, Higgs), physics toggles (relativity, 1PN, black hole, spin-orbit, radiation, tidal locking, disintegration). Preset selector with four category groups (Gravity, EM, Exotic, Cosmological).
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
main.js                     ~364 lines  Simulation class, fixed-timestep loop, Higgs field init, window.sim
index.html                  ~514 lines  UI structure, tab system, reference overlay, Higgs sliders
styles.css                  ~235 lines  Project-specific CSS overrides
colors.js                     18 lines  Project color tokens (extends shared-tokens.js)
src/
  integrator.js            ~1216 lines  Physics: adaptive Boris substep loop, radiation, tidal, GW, expansion, Higgs, external fields, Hertz bounce
  ui.js                     ~536 lines  DOM setup, declarative toggle dependencies, info tips, shortcuts, Higgs slider wiring
  renderer.js               ~494 lines  Canvas 2D: particles, trails, vectors, photons, glow, Higgs overlay
  forces.js                 ~461 lines  Pairwise + Barnes-Hut force accumulation, 1PN, Yukawa
  presets.js                ~586 lines  Fifteen preset scenarios (Gravity / EM / Exotic / Cosmological)
  higgs-field.js            ~393 lines  Higgs scalar field: Mexican hat potential, symplectic Euler, CIC deposition, mass modulation, gradient force, phase transitions
  reference.js              ~309 lines  Extended physics reference (KaTeX math)
  quadtree.js               ~280 lines  SoA pool-based Barnes-Hut tree (zero GC)
  input.js                  ~262 lines  Mouse/touch, Place/Shoot/Orbit spawn modes
  signal-delay.js            250 lines  Light-cone solver on circular history buffers
  effective-potential.js    ~207 lines  V_eff(r) sidebar canvas, auto-scaling, current position marker
  save-load.js              ~203 lines  State serialization, quick save/load, file export/import
  heatmap.js                ~190 lines  Gravitational + electric + Yukawa potential field overlay
  potential.js              ~160 lines  PE computation (pairwise + tree traversal)
  energy.js                 ~147 lines  KE, PE, field energy, Higgs field energy, momentum, angular momentum
  stats-display.js          ~126 lines  Sidebar energy/momentum/drift readout, force breakdown
  particle.js               ~124 lines  Particle entity (incl. baseMass, forceHiggs, antimatter flag)
  collisions.js             ~118 lines  Merge, antimatter annihilation, baseMass conservation
  phase-plot.js              116 lines  Phase space plot (sidebar canvas)
  topology.js                112 lines  Torus / Klein / RP2 min-image + wrapping
  config.js                 ~111 lines  Named constants (softening, BH, numerical, Higgs, pair production)
  vec2.js                     65 lines  2D vector math
  photon.js                   40 lines  Radiation photon entity
  relativity.js               34 lines  Proper velocity helpers
```

## Sibling Projects

- [Cellular Metabolism](https://github.com/a9lim/biosim) -- [a9l.im/biosim](https://a9l.im/biosim)
- [Redistricting Simulator](https://github.com/a9lim/gerry) -- [a9l.im/gerry](https://a9l.im/gerry)

## License

[AGPL-3.0](LICENSE)
