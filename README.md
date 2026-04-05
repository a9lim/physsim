# Geon

Interactive particle physics simulator running entirely in the browser. Spawn particles, fling them into orbit, toggle forces on and off, and watch the physics play out in real time. Covers gravity, electromagnetism, nuclear forces, scalar fields, relativistic corrections, and more -- all rendered live with a WebGPU compute backend (automatic CPU fallback).

**[Try it](https://a9l.im/geon)** | Part of the [a9l.im](https://a9l.im) portfolio

## What You Can Explore

**19 built-in presets** let you jump straight into curated scenarios:

| Category | Presets |
|----------|---------|
| Gravity | Kepler Orbits, Precession, Binary Inspiral, Tidal Lock, Roche Limit, Hawking Evaporation |
| Electromagnetism | Atom, Bremsstrahlung, Magnetic Dipoles |
| Exotic | Atomic Nucleus, Axion Field, Pion Exchange, Higgs Mechanism, Higgs Boson, Axion Burst, Peccei-Quinn, Phase Transition |
| Cosmological | Galaxy, Expanding Universe |

Each preset configures the relevant forces, boundary conditions, and initial particles so the physics is immediately visible. Every parameter is adjustable from the sidebar.

## Physics

Geon simulates 2D relativistic N-body dynamics in natural units (c = G = h = 1). Particles store proper velocity, so the speed of light is enforced automatically.

**11 force types** -- Newtonian gravity, Coulomb, magnetic dipole, Lorentz, gravitomagnetic, frame-dragging, tidal locking, Yukawa (screened nuclear), spin-orbit, external fields, and scalar field gradients.

**Relativistic corrections** -- First post-Newtonian corrections for gravity (Einstein-Infeld-Hoffmann), electromagnetism (Darwin), gravity-EM cross-terms (Bazanski), and Yukawa (scalar Breit). Finite-speed force propagation solves for the past light cone analytically.

**Scalar fields** -- A Higgs field with spontaneous symmetry breaking and mass generation, and an axion field with position-dependent coupling constants. Both live on smooth grids, interact with particles and each other, and produce visible wave dynamics on collisions.

**Radiation** -- Larmor dipole, EM quadrupole, gravitational wave quadrupole, and pion emission. Photons and pions are gravitationally lensed.

**Black holes** -- Kerr-Newman horizons, ergosphere visualization, Hawking radiation, and pair production from energetic photons.

**Topology** -- Boundaries can be open, periodic (torus), or non-orientable (Klein bottle, real projective plane).

## Controls

Click to spawn particles. Drag to launch them with velocity. Right-click for antimatter. Scroll to zoom, middle-drag to pan. Press `?` for the full shortcut list.

The sidebar has four tabs: **Settings** (forces, spawn parameters, presets), **Engine** (GPU toggle, collisions, boundaries, overlays, sim speed), **Stats** (energy, momentum, conserved quantities), and **Particle** (per-particle force breakdown, phase space plot, effective potential).

## Running Locally

```bash
cd path/to/a9lim.github.io && python -m http.server
# -> http://localhost:8000/geon/
```

Serve from the repository root -- shared design files load via absolute paths. No build step, no dependencies. ES6 modules require HTTP (not `file://`).

## Tech

Zero-dependency vanilla JavaScript (~29,000 lines across 92 files). CPU backend uses Canvas 2D; GPU backend uses WebGPU compute (52 WGSL shaders) with instanced rendering. Boris integrator with adaptive substepping, Barnes-Hut quadtree for O(N log N) force evaluation, cubic B-spline scalar field grids, and circular history buffers for signal delay -- all designed for zero per-frame allocation.

## Sibling Projects

- [Cyano](https://github.com/a9lim/cyano) -- [a9l.im/cyano](https://a9l.im/cyano)
- [Gerry](https://github.com/a9lim/gerry) -- [a9l.im/gerry](https://a9l.im/gerry)
- [Scripture](https://github.com/a9lim/scripture) -- [a9l.im/scripture](https://a9l.im/scripture)
- [Shoals](https://github.com/a9lim/shoals) -- [a9l.im/shoals](https://a9l.im/shoals)

## License

[AGPL-3.0](LICENSE)
