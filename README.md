# Relativistic N-Body Physics Simulation

A high-performance, interactive physics simulation that models gravity, electromagnetism, magnetic dipole interactions, and gravitomagnetic corrections with relativistic effects. Uses the **Barnes-Hut algorithm** for O(N log N) force calculation, enabling simulations with thousands of particles.

**[Live Demo →](https://a9l.im/physsim)** · Part of the [a9l.im](https://a9l.im) portfolio

## Features

- **Relativistic Physics** — Proper velocity integration naturally enforces the speed-of-light limit. Both linear and rotational state use the same pattern: `derived = state / √(1 + state² × scale²)`, so no particle can ever exceed *c*.
- **Four Force Types**
  - **Gravity** — Universal attraction between masses (inverse-square)
  - **Coulomb** — Electrostatic repulsion/attraction between charged particles
  - **Magnetic Dipole** — Interaction between spinning, charged particles
  - **Gravitomagnetic** — Relativistic correction coupling mass and angular momentum
- **Barnes-Hut Optimization** — QuadTree spatial partitioning approximates long-range forces at O(N log N). Toggleable — disable for exact O(N²) pairwise forces that preserve Newton's 3rd law exactly
- **Interaction Modes** — Place (spawn at rest), Shoot (drag to set velocity), Orbit (auto-calculates circular orbit around nearest massive body)
- **Collision Modes** — Pass-through, elastic bounce with spin-friction transfer (configurable friction), or merge (conserves mass, charge, momentum, and angular momentum)
- **Boundary Modes** — Despawn off-screen, toroidal wrap, or bounce off edges
- **Presets** — Solar System, Binary Star, Galaxy, Collision, Magnetic Spin
- **Visuals** — Particle trails, charge-based dynamic coloring, spin rings, additive glow in dark mode, light/dark theme toggle
- **Boris Integrator** — Splits E-like (radial) and B-like (velocity-dependent) forces; Boris rotation exactly preserves |v| for long-term magnetic stability
- **Independent Force Toggles** — Enable/disable gravity, Coulomb, magnetic, gravitomagnetic, and relativity independently via sidebar switches
- **Energy Conservation Display** — Real-time tracking of linear KE (relativistic or classical), rotational KE (I=(2/5)mr² solid sphere), gravitational PE, Coulomb PE, magnetic and gravitomagnetic dipole PE, total energy, and drift percentage
- **Conserved Quantities** — Real-time momentum (|Σmw|) and angular momentum (orbital + spin, computed about COM) tracking
- **Force Component Vectors** — Per-force-type arrows (gravity, Coulomb, magnetic, gravitomagnetic) in distinct colors alongside net force and velocity vectors
- **Particle Inspection** — Hover for compact tooltip (mass, charge, spin, speed); click to select and see live stats in sidebar (gamma, force breakdown)
- **Keyboard Shortcuts** — Space (pause), R (reset), `.` (step), P (presets), 1-5 (load preset), V (velocity vectors), F (force vectors), T (theme), S (sidebar); press `?` for help overlay
- **Info Tips** — Hover `?` icons next to controls for explanations of physics concepts and simulation parameters
- **Zoom & Pan** — Scroll to zoom

## Controls

| Input | Action |
|-------|--------|
| Left click | Spawn particle |
| Right click | Remove particle |
| Scroll wheel | Zoom in/out |
| Topbar buttons | Play/Pause, Step, Reset |
| Sidebar panel | All physics parameters (mass, charge, spin, collision mode, etc.) |

## Running Locally

Serve the directory to avoid CORS issues with ES6 modules:

```bash
python -m http.server
# Navigate to http://localhost:8000
```

No build step, no dependencies. Shared design system files (`shared-tokens.js`, `shared-base.css`) load from the root site — serve from the parent `a9lim.github.io/` directory for full functionality.

## Architecture

```
index.html
  ├── colors.js          — extends shared palette with particle hues, CSS vars
  └── main.js            — Simulation class (ES module entry)
        ├── src/physics.js    — force calculation, integration, collisions
        │     ├── src/quadtree.js  — Barnes-Hut spatial partitioning
        │     └── src/vec2.js      — 2D vector math
        ├── src/renderer.js   — Canvas 2D drawing, trails, themes
        ├── src/input.js      — mouse/touch interaction, particle spawning
        ├── src/relativity.js — relativistic helpers (proper velocity, spin derivation)
        ├── src/particle.js   — entity definition
        ├── src/presets.js    — preset definitions + loadPreset function
        └── src/ui.js         — setupUI, DOM cache, all event binding
```

Uses the shared design system from [a9lim.github.io](https://github.com/a9lim/a9lim.github.io) — glass panels, tool buttons, intro screen, preset dialog, slider values, and responsive breakpoints.

### Technical Details

The simulation uses the **Boris integrator** with proper velocity **w** = γ**v** (celerity) as the primary state variable. It separates position-dependent (E-like) forces from velocity-dependent (B-like) forces:

1. Half-kick with E-like forces: **w** += **F**_E/m · dt/2
2. Boris rotation for B-like forces: rotate **w** in the B+Bg field plane (preserves |**v**| exactly)
3. Half-kick with E-like forces: **w** += **F**_E/m · dt/2
4. Derive velocity: **v** = **w** / √(1 + w²), drift position: **x** += **v** · dt
5. Rebuild Barnes-Hut tree, handle collisions
6. Recalculate E-like forces and B/Bg fields

The Boris rotation uses combined parameter t = ((q/(2m))·B_z + 2·Bg_z)·dt/γ with s = 2t/(1+t²), giving exact area-preserving rotation. This handles Lorentz and gravitomagnetic forces without energy drift.

Spin uses the same proper-velocity pattern — `p.spin` stores proper angular velocity, angular velocity is derived via `ω = S / √(1 + S²r²)`, naturally capping surface velocity at *c*. Spin-orbit torques are position-dependent and integrated via half-kicks.

Natural units (c = 1, G = 1) throughout. The proper velocity approach provides inherent stability — γ = √(1 + w²) has no singularities unlike 1/√(1 − v²), and high-energy interactions cannot produce superluminal velocities. The Boris integrator exactly preserves kinetic energy through the magnetic rotation step, producing superior long-term stability for charged and spinning particles.

## Sibling Projects

- [Cellular Metabolism](https://github.com/a9lim/biosim) — [a9l.im/biosim](https://a9l.im/biosim)
- [Redistricting Simulator](https://github.com/a9lim/gerry) — [a9l.im/gerry](https://a9l.im/gerry)

## License

[AGPL-3.0](LICENSE)
