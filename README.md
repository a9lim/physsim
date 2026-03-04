# No-Hair

Relativistic N-body simulation with gravity, electromagnetism, and general-relativistic corrections. Particles have mass, charge, and spin; forces propagate at finite speed.

**[Live Demo →](https://a9l.im/physsim)** · Part of the [a9l.im](https://a9l.im) portfolio

## Physics

Natural units throughout: *c* = 1, *G* = 1.

**State variables** use proper velocity **w** = γ**v** (linear) and angular celerity *W* (rotational). Coordinate velocity **v** = **w**/√(1+w²) naturally enforces |**v**| < *c*. Surface speed is capped identically: ω = *W*/√(1+*W*²*r*²).

### Forces

| Force | Formula | Notes |
|-------|---------|-------|
| Gravity | *m*₁*m*₂/*r*² | Attractive, Plummer-softened |
| Coulomb | −*q*₁*q*₂/*r*² | Like charges repel |
| Magnetic dipole | −3μ₁μ₂/*r*⁴ | μ = *q*ω*r*²/5 (spinning charged sphere) |
| Lorentz | *q*(**v** × **B**) | Boris rotation; *B* from moving charges + dipoles |
| Gravitomagnetic dipole | +3*L*₁*L*₂/*r*⁴ | *L* = 2*m*ω*r*²/5; co-rotating masses attract |
| Frame-dragging | 4*m*(**v** × **B**_g) | Boris rotation; **B**_g from moving/spinning masses |
| 1PN (EIH) | O(*v*²/*c*²) correction | Perihelion precession ≈ 6π*M*/*a*(1−*e*²) rad/orbit |
| Larmor radiation | τ·d**F**/d*t* / γ³ | τ = 2*q*²/(3*m*); Landau-Lifshitz jerk term |
| Stern-Gerlach | +μ·∇*B* | Spin-curvature force on center of mass |
| Mathisson-Papapetrou | −*L*·∇*B*_g | Gravitational spin-curvature force |

### Additional effects

- **Signal delay** — Forces use source positions from the light cone, solved via Newton-Raphson on per-particle history buffers. Pairwise mode only.
- **Spin-orbit coupling** — Energy transfer between translational and rotational KE via field gradients.
- **Tidal breakup** — Roche-limit fragmentation when tidal + centrifugal + Coulomb self-stress exceeds self-gravity.
- **Radiation** — Accelerating charges emit photons (Larmor dipole pattern with relativistic aberration). Photons carry energy and momentum, and are absorbed on contact.

### Integrator

Boris integrator (half-kick / rotate / half-kick / drift) with adaptive substepping based on acceleration and cyclotron frequency, capped at 16 substeps per frame. 1PN uses a velocity-Verlet correction pass for second-order accuracy.

### Algorithms

- **Barnes-Hut** — Toggleable O(*N* log *N*) quadtree approximation. Pool-based SoA layout, zero per-frame allocation. When off, exact O(*N*²) pairwise forces preserve Newton's third law.
- **Collisions** — Pass-through, elastic bounce (with configurable spin friction), or merge (conserves mass, charge, momentum, angular momentum).
- **Boundaries** — Despawn, bounce, or periodic loop with topology selection: torus, Klein bottle, or real projective plane.

## Controls

| Input | Action |
|-------|--------|
| Left click | Spawn (Place / Shoot / Orbit mode) |
| Right click | Remove particle |
| Scroll | Zoom |
| `Space` | Pause / resume |
| `P` / `1`–`5` | Presets |
| `?` | Keyboard shortcut help |

## Running Locally

```bash
# Serve from parent — shared design files load via absolute paths
cd path/to/a9lim.github.io && python -m http.server
# → http://localhost:8000/physsim/
```

No build step, no dependencies. ES6 modules require HTTP (no `file://`).

## Architecture

```
main.js                    — Simulation loop, window.sim
├── src/integrator.js      — Boris substep loop, radiation, tidal breakup
│   ├── src/forces.js      — Pairwise + Barnes-Hut force accumulation
│   ├── src/collisions.js  — Merge, bounce, pass
│   ├── src/potential.js   — PE computation (pairwise + tree)
│   ├── src/signal-delay.js— Light-cone solve on history buffers
│   ├── src/topology.js    — Torus / Klein / RP² minimum-image + wrapping
│   ├── src/quadtree.js    — SoA pool-based Barnes-Hut tree
│   └── src/photon.js      — Radiation photon entity
├── src/energy.js          — KE, PE, field energy, momentum, angular momentum
├── src/relativity.js      — Proper velocity ↔ coordinate velocity
├── src/renderer.js        — Canvas 2D: particles, trails, forces, glow
├── src/input.js           — Mouse / touch, spawn modes
├── src/stats-display.js   — Sidebar energy / momentum / drift readout
├── src/heatmap.js         — Gravitational potential field overlay
├── src/phase-plot.js      — Phase space plot (selected particle)
├── src/sankey.js          — Energy breakdown bar chart
├── src/particle.js        — Entity definition
├── src/vec2.js            — 2D vector math
├── src/presets.js         — Scenario definitions
├── src/config.js          — Named constants
└── src/ui.js              — DOM setup, info tips, event binding
```

## Sibling Projects

- [Cellular Metabolism](https://github.com/a9lim/biosim) — [a9l.im/biosim](https://a9l.im/biosim)
- [Redistricting Simulator](https://github.com/a9lim/gerry) — [a9l.im/gerry](https://a9l.im/gerry)

## License

[AGPL-3.0](LICENSE)
