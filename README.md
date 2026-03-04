# Relativistic N-Body Physics Simulation

Interactive physics simulation modeling gravity, electromagnetism, magnetic dipoles, and gravitomagnetic corrections with relativistic effects. Uses the **Barnes-Hut algorithm** for O(N log N) force calculation.

**[Live Demo →](https://a9l.im/physsim)** · Part of the [a9l.im](https://a9l.im) portfolio

## Features

- **Relativistic mechanics** — Proper velocity `w = γv` as state variable; velocity derived via `v = w/√(1+w²)`, naturally enforcing the speed-of-light limit. Same pattern for spin: angular celerity caps surface velocity below *c*.
- **Boris integrator** — Splits E-like (radial) and B-like (velocity-dependent) forces; Boris rotation exactly preserves |v| for long-term magnetic stability
- **6 force types** — Gravity, Coulomb, magnetic dipole, gravitomagnetic dipole, Lorentz, and linear gravitomagnetic (frame-dragging)
- **Larmor radiation** — Accelerating charges emit visible photons with orbital decay via Landau-Lifshitz force
- **Signal delay** — Finite-speed force propagation via retarded potentials (pairwise mode only)
- **Spin-orbit coupling** — Energy transfer between translational and rotational KE via B-field gradients
- **Tidal breakup** — Roche limit fragmentation when tidal/centrifugal/Coulomb stress exceeds self-gravity
- **Barnes-Hut** — Toggleable O(N log N) quadtree approximation vs exact O(N²) pairwise forces
- **Collisions** — Pass-through, elastic bounce with spin-friction transfer, or merge (conserves mass, charge, momentum, angular momentum)
- **5 presets** — Solar System, Binary Star, Galaxy, Collision, Magnetic Spin
- **Real-time diagnostics** — Energy breakdown (KE, spin KE, PE, field, radiated), momentum (particle + field + radiated), angular momentum (orbital + spin), all with drift tracking
- **Visuals** — Trails, force component vectors, charge-based coloring, spin rings, additive glow, light/dark theme

## Controls

| Input | Action |
|-------|--------|
| Left click | Spawn particle (Place/Shoot/Orbit modes) |
| Right click | Remove particle |
| Scroll wheel | Zoom in/out |
| `Space` | Pause/resume |
| `P` / `1-5` | Open presets / load preset directly |
| `?` | Keyboard shortcut help |

## Running Locally

```bash
# Serve from parent directory for shared design system files
cd path/to/a9lim.github.io && python -m http.server
# Navigate to http://localhost:8000/physsim/
```

No build step, no dependencies. ES6 modules require an HTTP server (no `file://`).

## Architecture

```
main.js                    — Simulation class (entry point)
├── src/integrator.js      — Physics class: adaptive Boris substep loop, radiation, tidal breakup
│     ├── src/forces.js        — force computation (pairwise + Barnes-Hut tree walk)
│     ├── src/collisions.js    — collision resolution (merge, bounce)
│     ├── src/potential.js     — potential energy computation
│     ├── src/signal-delay.js  — retarded potentials (signal delay)
│     ├── src/quadtree.js      — pool-based Barnes-Hut quadtree (zero per-frame GC)
│     └── src/photon.js        — radiation photon entity
├── src/stats-display.js   — energy/momentum/drift stats, selected particle info
├── src/energy.js          — energy, momentum, angular momentum computation
├── src/relativity.js      — proper velocity / angular celerity conversions
├── src/renderer.js        — Canvas 2D drawing, trails, themes
├── src/input.js           — mouse/touch interaction, particle spawning
├── src/particle.js        — entity definition
├── src/vec2.js            — 2D vector math
├── src/heatmap.js         — density heatmap overlay
├── src/phase-plot.js      — phase space visualization
├── src/sankey.js          — energy breakdown bar chart
├── src/presets.js         — preset scenario definitions
├── src/config.js          — named constants
└── src/ui.js              — DOM setup, event binding, info tips
```

### Technical Details

Natural units (c = 1, G = 1) throughout. The Boris integrator sequence per substep:

1. Half-kick: **w** += **F**_E/m · dt/2
2. Boris rotation: rotate **w** in B+Bg field plane (preserves |**v**| exactly)
3. Half-kick: **w** += **F**_E/m · dt/2
4. Derive **v** = **w**/√(1+w²), drift **x** += **v**·dt
5. Rebuild tree, handle collisions, recalculate forces

Spin uses the same proper-velocity pattern — `angw` (angular celerity) derives `angVel = angw/√(1+angw²r²)` via `angwToAngVel()`, capping surface velocity at *c*. Determines magnetic moment (μ = ⅕qωr²) and angular momentum (L = ⅖mωr²).

**Conservation note:** Velocity-dependent forces (Lorentz, linear gravitomagnetism) don't satisfy Newton's third law — the missing momentum is carried by fields not modeled here. Momentum and angular momentum are exactly conserved only with radial forces in pairwise mode (Barnes-Hut off).

## Sibling Projects

- [Cellular Metabolism](https://github.com/a9lim/biosim) — [a9l.im/biosim](https://a9l.im/biosim)
- [Redistricting Simulator](https://github.com/a9lim/gerry) — [a9l.im/gerry](https://a9l.im/gerry)

## License

[AGPL-3.0](LICENSE)
