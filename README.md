# Relativistic N-Body Physics Simulation

A high-performance, interactive physics simulation that models gravity, electromagnetism, magnetic dipole interactions, and gravitomagnetic corrections with relativistic effects. Uses the **Barnes-Hut algorithm** for O(N log N) force calculation, enabling simulations with thousands of particles. Pure vanilla JS — no build system or dependencies.

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
- **Zoom & Pan** — Scroll to zoom

## Controls

| Action | Input |
|---|---|
| Spawn particle | Left click |
| Remove particle | Right click |
| Zoom | Scroll wheel |
| Play / Pause | Topbar button |
| Step (when paused) | Topbar button |
| Reset | Topbar button |

All physics parameters (mass, charge, angular momentum, collision mode, boundary mode, sim speed, trails) are in the sidebar panel, toggled from the topbar.

## Running Locally

Serve the directory to avoid CORS issues with ES6 modules:

```bash
python -m http.server
# Navigate to http://localhost:8000
```

## Technical Details

The simulation uses **relativistic Euler integration** with momentum as the primary state variable:

1. Forces calculated via Barnes-Hut tree traversal
2. Momentum update: **p** = **p** + **F** · dt
3. Velocity derived: **v** = **p** / (m · γ), where γ = √(1 + p²/m²)
4. Position update: **x** = **x** + **v** · dt

Natural units (c = 1, G = 1) throughout. The momentum-based approach provides inherent stability — high-energy interactions cannot produce superluminal velocities.

### Architecture

```
index.html
  ├── colors.js      — palette, fonts, CSS variable injection
  └── main.js        — Simulation class (ES module entry)
        ├── src/physics.js    — force calculation, integration, collisions
        │     ├── src/quadtree.js  — Barnes-Hut spatial partitioning
        │     └── src/vec2.js      — 2D vector math
        ├── src/renderer.js   — Canvas 2D drawing, trails, themes
        ├── src/input.js      — mouse/touch interaction, particle spawning
        └── src/particle.js   — entity definition
```
