# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Relativistic N-body physics simulation — interactive browser-based app using Barnes-Hut algorithm for O(N log N) force calculation. Pure vanilla JS with ES6 modules, no build system or dependencies.

## Running Locally

```bash
python -m http.server
# Navigate to http://localhost:8000
```

Opening `index.html` directly may fail due to CORS restrictions on ES6 module imports. There is no build step, test framework, or linter configured.

## Architecture

### Module Dependency Graph

```
index.html → main.js (Simulation class)
                ├── src/physics.js (force calculation, integration, collisions)
                │     ├── src/quadtree.js (Barnes-Hut spatial partitioning)
                │     └── src/vec2.js (2D vector math)
                ├── src/renderer.js (Canvas 2D drawing, trails, themes)
                │     └── src/vec2.js
                ├── src/input.js (mouse interaction, particle spawning)
                │     ├── src/particle.js
                │     └── src/vec2.js
                └── src/particle.js (entity definition)
                      └── src/vec2.js
```

### Key Design Decisions

- **Natural units**: c=1, G=1 throughout the physics engine. All equations use these conventions.
- **Momentum-based integration**: Physics uses relativistic momentum (not velocity) as the primary state variable. Velocity is derived via Lorentz factor: `v = p / (m * gamma)` where `gamma = sqrt(1 + p²/m²)`. This naturally enforces the speed-of-light limit.
- **Barnes-Hut approximation**: QuadTree stores aggregate mass, charge, spin, and center-of-mass per node. Theta parameter (0.5) controls accuracy vs. performance tradeoff.
- **Softening parameter**: `minDistSq = 25` prevents force singularities at close range.
- **No global state**: The `Simulation` instance owns all state. It's exposed as `window.sim` for console debugging.

### Rendering

- Dark mode uses additive blending (`globalCompositeOperation: 'lighter'`) for glow effects.
- Trail history: up to 200 positions per particle, stored as flat `[x, y, x, y, ...]` arrays in a Map keyed by particle ID.
- Particle color is computed from charge sign/magnitude using HSL (blue=positive, red=negative, grey=neutral).

### Force Types

The physics engine computes four force types per particle pair: gravitational, Coulomb (electrostatic), magnetic dipole-dipole, and gravitomagnetic correction. All are inverse-square with different coupling constants.

### Collision Modes

Three modes in physics.js: `pass` (no-op), `merge` (conserves mass/charge/momentum), `bounce` (elastic with spin-friction transfer, friction coefficient 0.4).

### Input Modes

Three placement modes in input.js: `place` (spawn at rest), `shoot` (drag distance sets velocity at 0.1x multiplier), `orbit` (calculates circular orbit velocity around nearest massive body). Touch events (touchstart/touchmove/touchend) delegate to mouse handlers for mobile support.

## Conventions

- All vector operations use the `Vec2` class. Use `vec.set(x, y)` for in-place mutation in hot paths; prefer `Vec2.add(a, b)` static methods elsewhere.
- Physics hot path avoids allocations: `calculateForce()` accumulates into an `out` Vec2 parameter; force array is reused across frames.
- DOM elements cached in `Simulation.dom`; UI mode state tracked in JS variables — no per-frame DOM queries.
- Particle visual radius scales as `sqrt(mass)`.
- Presets are defined inline in `Simulation.loadPreset()` in main.js.
- UI is styled with CSS custom properties (design tokens) for theme switching (dark/light). Accent: `#D97757` light / `#E89B80` dark.
- Fonts: Sora (section headings, uppercase), Instrument Serif (display/preset titles), Geist (body/controls). All loaded from Google Fonts CDN.
- UI architecture: floating frosted-glass topbar + non-blocking slide-in sidebar (right side, toggleable, does not block canvas interaction). Preset dialog is a centered modal card grid.
- Design system modeled after ~/Documents/antigravity/gerry — match its exact token values (colors, shadows, font sizes, spacing) when making UI changes.
- Sidebar section headings: Sora 0.68rem/600/uppercase/0.12em tracking with `border-bottom` underline. Panel header: Sora 0.72rem same style.
- Tool buttons (`.tool-btn`): 34x34, bare icons, transparent default, `--bg-hover` on hover. Ghost buttons (`.ghost-btn`): pill-shaped with border.
