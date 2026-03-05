# No-Hair

A relativistic N-body simulation exploring how gravity, electromagnetism, and spin shape particle dynamics in 2D. Particles have mass, charge, and angular momentum; forces propagate at finite speed.

**[Live Demo](https://a9l.im/physsim)** | Part of the [a9l.im](https://a9l.im) portfolio

## What It Simulates

Everything runs in natural units (c = 1, G = 1). Particles store proper velocity **w** = gamma * **v** as their state variable, so the speed of light is enforced automatically: coordinate velocity **v** = **w** / sqrt(1 + w^2) always satisfies |v| < c. The same trick applies to spin.

### Forces

Ten distinct force types, all toggleable independently:

- **Gravity** -- Newtonian 1/r^2 attraction between all massive particles, Plummer-softened.
- **Coulomb** -- 1/r^2 electrostatic force; like charges repel, opposites attract.
- **Magnetic dipole** -- 3 mu_1 mu_2 / r^4 interaction between spinning charged particles. Magnetic moment mu = q * omega * r^2 / 5 (uniform charge sphere).
- **Lorentz** -- q(v x B) force from moving charges and spinning dipoles. Handled exactly by Boris rotation.
- **Gravitomagnetic dipole** -- 3 L_1 L_2 / r^4 interaction between spinning masses. Co-rotating masses attract (GEM sign convention).
- **Frame-dragging** -- 4m(v x B_g) from moving/spinning masses, plus a torque that aligns neighboring spins.
- **1PN (Einstein-Infeld-Hoffmann)** -- O(v^2/c^2) correction to gravity. Produces perihelion precession at the GR rate.
- **Larmor radiation** -- Accelerating charges lose energy via the Landau-Lifshitz force and emit photons.
- **Stern-Gerlach** -- Translational force from spin-field gradient coupling (EM).
- **Mathisson-Papapetrou** -- Gravitational analog of Stern-Gerlach for spinning masses.

### Additional Effects

- **Signal delay** -- Forces use source positions from the past light cone, solved analytically with a three-phase algorithm (Newton-Raphson segment search, exact quadratic solve, constant-velocity extrapolation).
- **Spin-orbit coupling** -- Energy transfer between translational and rotational kinetic energy via field gradients.
- **Tidal breakup** -- Roche-limit fragmentation when tidal, centrifugal, and Coulomb stresses exceed self-gravity.
- **Photon emission** -- Accelerating charges emit photons in a Larmor dipole pattern with relativistic aberration. Photons carry energy and momentum, and are absorbed on contact.
- **Black hole mode** -- All particles switch to Schwarzschild radius (r = 2M), collisions lock to merge, and each black hole emits Hawking radiation at the Planck-unit rate P = 1/(15360 pi M^2). Sub-threshold black holes evaporate with a final photon burst.

### Integrator

Boris integrator (half-kick / rotate / half-kick / drift) with adaptive substepping. The Boris rotation handles velocity-dependent magnetic forces exactly, preserving |v| through each step. Substep count adapts to acceleration magnitude and cyclotron frequency, capped at 16 substeps per frame. The 1PN correction uses a velocity-Verlet correction pass for second-order accuracy.

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
| `P` / `1`--`5` | Open presets / load preset directly |
| `V` / `F` / `C` | Toggle velocity / force / component vectors |
| `T` / `S` | Toggle theme / sidebar |
| `?` | Keyboard shortcut help |

### Sidebar Tabs

1. **Settings** -- Particle mass (0.05--5) / charge (-5 to 5) / spin sliders, spawn mode, force toggles, physics toggles (including Black Hole mode).
2. **Engine** -- Barnes-Hut, collision mode, bounce friction, boundary mode, topology, visual overlays, sim speed.
3. **Stats** -- Energy breakdown (linear KE, spin KE, PE, field, radiated, drift), conserved quantities (momentum with particle/field/radiated components, angular momentum with orbital/spin).
4. **Particle** -- Selected particle details (mass, charge, spin, speed, gamma, |F|) and phase space plot.

## Running Locally

```bash
# Serve from parent -- shared design files load via absolute paths
cd path/to/a9lim.github.io && python -m http.server
# -> http://localhost:8000/physsim/
```

No build step, no dependencies, no npm. ES6 modules require HTTP (no `file://`).

## Tech

Zero-dependency vanilla JavaScript with Canvas 2D rendering. All physics and rendering code is hand-written without libraries. Structure-of-arrays quadtree, circular history buffers for signal delay, adaptive substepping -- all designed for zero per-frame allocation in hot paths.

## Architecture

```
main.js                     212 lines  Simulation class, fixed-timestep loop, window.sim
index.html                  415 lines  UI structure, tab system, preset dialog
styles.css                  560 lines  Project-specific CSS
colors.js                    27 lines  Project color tokens (extends shared-tokens.js)
src/
  integrator.js             763 lines  Physics class: adaptive Boris substep loop
  forces.js                 335 lines  Pairwise + Barnes-Hut force accumulation
  signal-delay.js           315 lines  Light-cone solver on circular history buffers
  ui.js                     338 lines  DOM setup, toggles, info tips, keyboard shortcuts
  renderer.js               406 lines  Canvas 2D: particles, trails, vectors, glow
  input.js                  262 lines  Mouse/touch, Place/Shoot/Orbit spawn modes
  collisions.js             259 lines  Merge, bounce (relativistic + classical)
  quadtree.js               256 lines  SoA pool-based Barnes-Hut tree (zero GC)
  potential.js              158 lines  PE computation (pairwise + tree traversal)
  topology.js               129 lines  Torus / Klein / RP2 min-image + wrapping
  energy.js                 127 lines  KE, PE, field energy, momentum, angular momentum
  phase-plot.js             120 lines  Phase space plot (sidebar canvas)
  sankey.js                  98 lines  Energy bar chart (orphaned, not imported)
  stats-display.js           92 lines  Sidebar energy/momentum/drift readout
  presets.js                 87 lines  Five preset scenarios
  heatmap.js                 83 lines  Gravitational potential field overlay
  particle.js                79 lines  Particle entity definition
  vec2.js                    69 lines  2D vector math
  config.js                  54 lines  Named constants
  relativity.js              41 lines  Proper velocity helpers
  photon.js                  19 lines  Radiation photon entity
```

## Sibling Projects

- [Cellular Metabolism](https://github.com/a9lim/biosim) -- [a9l.im/biosim](https://a9l.im/biosim)
- [Redistricting Simulator](https://github.com/a9lim/gerry) -- [a9l.im/gerry](https://a9l.im/gerry)

## License

[AGPL-3.0](LICENSE)
