# Relativistic N-Body Physics Simulation

A high-performance, interactive physics simulation that models gravity, electromagnetism, magnetic dipole interactions, and gravitomagnetic corrections with relativistic effects. Uses the **Barnes-Hut algorithm** for O(N log N) force calculation, enabling simulations with thousands of particles.

**[Live Demo →](https://a9l.im/physsim)** · Part of the [a9l.im](https://a9l.im) portfolio

## Features

- **Relativistic Physics** — Momentum-based integration naturally enforces the speed-of-light limit. Velocity is derived via the Lorentz factor, so no particle can ever exceed *c*.
- **Four Force Types**
  - **Gravity** — Universal attraction between masses (inverse-square)
  - **Coulomb** — Electrostatic repulsion/attraction between charged particles
  - **Magnetic Dipole** — Interaction between spinning, charged particles
  - **Gravitomagnetic** — Relativistic correction coupling mass and angular momentum
- **Barnes-Hut Optimization** — QuadTree spatial partitioning approximates long-range forces at O(N log N)
- **Interaction Modes** — Place (spawn at rest), Shoot (drag to set velocity), Orbit (auto-calculates circular orbit around nearest massive body)
- **Collision Modes** — Pass-through, elastic bounce with spin-friction transfer, or merge (conserves mass/charge/momentum)
- **Boundary Modes** — Despawn off-screen, toroidal wrap, or bounce off edges
- **Presets** — Solar System, Binary Star, Galaxy, Collision, Magnetic Spin
- **Visuals** — Particle trails, charge-based dynamic coloring, spin rings, additive glow in dark mode, light/dark theme toggle
- **Velocity Verlet Integration** — Kick-drift-kick scheme for time-symmetric, energy-conserving integration
- **Independent Force Toggles** — Enable/disable gravity, Coulomb, magnetic, gravitomagnetic, and relativity independently via sidebar switches
- **Energy Conservation Display** — Real-time tracking of linear KE (relativistic or classical), rotational KE, gravitational and Coulomb PE, total energy, and drift percentage
- **Velocity & Force Vectors** — Optional arrow overlays showing particle velocity and net force direction/magnitude
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
        ├── src/particle.js   — entity definition
        ├── src/presets.js    — preset definitions + loadPreset function
        └── src/ui.js         — setupUI, DOM cache, all event binding
```

Uses the shared design system from [a9lim.github.io](https://github.com/a9lim/a9lim.github.io) — glass panels, tool buttons, intro screen, preset dialog, slider values, and responsive breakpoints.

### Technical Details

The simulation uses **Velocity Verlet** (kick-drift-kick) integration with relativistic momentum as the primary state variable:

1. Half-kick momentum with stored forces: **p** += **F** · dt/2
2. Derive velocity: **v** = **p** / (m · γ), where γ = √(1 + p²/m²)
3. Drift position: **x** += **v** · dt
4. Recalculate forces via Barnes-Hut tree traversal
5. Half-kick momentum with new forces: **p** += **F** · dt/2

Natural units (c = 1, G = 1) throughout. The momentum-based approach provides inherent stability — high-energy interactions cannot produce superluminal velocities. Velocity Verlet is time-symmetric and energy-conserving, producing significantly lower energy drift than forward Euler.

## Sibling Projects

- [Cellular Metabolism](https://github.com/a9lim/biosim) — [a9l.im/biosim](https://a9l.im/biosim)
- [Redistricting Simulator](https://github.com/a9lim/gerry) — [a9l.im/gerry](https://a9l.im/gerry)

## License

[AGPL-3.0](LICENSE)
