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
index.html
  ├── colors.js (palette, fonts, CSS variable injection — non-module <script>)
  └── main.js (Simulation class, ES module)
        ├── src/physics.js (force calculation, integration, collisions)
        │     ├── src/quadtree.js (Barnes-Hut spatial partitioning)
        │     └── src/vec2.js (2D vector math)
        ├── src/renderer.js (Canvas 2D drawing, trails, themes — no imports, uses window._PALETTE)
        ├── src/input.js (mouse interaction, particle spawning)
        │     └── src/vec2.js
        └── src/particle.js (entity definition)
              └── src/vec2.js
```

### Key Design Decisions

- **Natural units**: c=1, G=1 throughout the physics engine. All equations use these conventions.
- **Momentum-based integration**: Physics uses relativistic momentum (not velocity) as the primary state variable. Velocity is derived via Lorentz factor: `v = p / (m * gamma)` where `gamma = sqrt(1 + p²/m²)`. This naturally enforces the speed-of-light limit.
- **Barnes-Hut approximation**: QuadTree stores aggregate mass, charge, spin, and center-of-mass per node. `BH_THETA` (0.5) controls accuracy vs. performance tradeoff.
- **Softening parameter**: `MIN_DIST_SQ` (25) prevents force singularities at close range. Other named constants: `BOUNCE_FRICTION` (0.4), `DESPAWN_MARGIN` (100).
- **Minimal global state**: The `Simulation` instance owns all runtime state (`window.sim` for console debugging). Design tokens (`window._PALETTE`, `window._FONT`, `window._r`) are frozen globals set by `colors.js` and consumed by ES modules via `window._PALETTE`.

### Color System

All colors and fonts are defined in `colors.js` (`_PALETTE` / `_FONT`), loaded as a plain `<script>` in `<head>` before the ES module graph. It serves two roles:
- **CSS injection**: An IIFE injects a `<style id="palette-vars">` element setting all CSS custom properties (`:root` for light, `[data-theme="dark"]` for dark).
- **JS globals**: Exposes `window._PALETTE`, `window._FONT`, `window._r` (alpha helper) for canvas drawing in ES modules. Modules alias as `const _PAL = window._PALETTE`.

Pattern matches `~/Documents/antigravity/biosim/colors.js` — single source of truth with `_r(hex, alpha)` helper, frozen palette object, and CSS variable injection IIFE.

### Rendering

- Dark mode uses additive blending (`globalCompositeOperation: 'lighter'`) for glow effects.
- Trail history: up to 200 positions per particle, stored as circular buffers (`{ data: Float32Array, len, start }`) in a Map keyed by particle ID.
- Spin ring colors are precomputed at module scope (`_spinColors`) — 4 hsla strings (2 hues × 2 themes).
- Particle color is computed from `_PALETTE` charge hues (`chargePos=220` blue, `chargeNeg=10` red, `neutral='#bdc3c7'` grey) with dynamic HSL saturation/lightness based on charge magnitude.

### Force Types

The physics engine computes four force types per particle pair: gravitational, Coulomb (electrostatic), magnetic dipole-dipole, and gravitomagnetic correction. All are inverse-square with different coupling constants.

### Collision Modes

Three modes in physics.js: `pass` (no-op), `merge` (conserves mass/charge/momentum), `bounce` (elastic with spin-friction transfer, friction coefficient 0.4).

### Input Modes

Three placement modes in input.js: `place` (spawn at rest), `shoot` (drag distance sets velocity at 0.1x multiplier), `orbit` (calculates circular orbit velocity around nearest massive body). Touch events (touchstart/touchmove/touchend) delegate to mouse handlers for mobile support.

## Conventions

### JS / Performance
- All vector operations use the `Vec2` class. Use `vec.set(x, y)` for in-place mutation in hot paths; prefer `Vec2.add(a, b)` static methods elsewhere.
- Physics hot path avoids allocations: `calculateForce()` accumulates into an `out` Vec2 parameter; force array is reused across frames.
- DOM elements cached in `Simulation.dom`; UI mode state tracked in JS variables — no per-frame DOM queries.
- `InputHandler` caches DOM refs (`massInput`, `chargeInput`, `spinInput`) and tracks `mode` state directly — no per-spawn DOM queries.

### Data & Presets
- Particle visual radius scales as `sqrt(mass)`.
- Presets are defined inline in `Simulation.loadPreset()` in main.js.

### UI Components
- Fonts: Instrument Serif (display/preset titles), Geist (body/controls/headings), Geist Mono (numeric values). All loaded from Google Fonts / jsDelivr CDN.
- UI architecture: floating frosted-glass topbar + non-blocking slide-in sidebar (right side, toggleable, does not block canvas interaction). Preset dialog is a centered modal card grid. Topbar right section layout: Presets | divider | Pause, Step, Reset | divider | Theme toggle (sun/moon), Settings. Playback/reset controls are icon-only `tool-btn`s, no text labels.
- Glass effect: `.glass` class (`background: var(--surface)`, `backdrop-filter: blur(24px) saturate(1.3)`, `border: 1px solid var(--border)`, `box-shadow: var(--shadow-md)`) applied to topbar, panel-body, preset-content, hint-bar. Panel-body and preset-content override to `--shadow-lg`; hint-bar overrides to `--shadow-sm`.
- Icon swaps (pause/play, sun/moon): both SVGs embedded in HTML with `hidden` attribute; JS toggles `hidden` instead of replacing `innerHTML`.

### Design System
- Design system modeled after ~/Documents/antigravity/gerry — match its exact token values (colors, shadows, font sizes, spacing) when making UI changes. Accent: `#FE3B01`, accent hover: `#FF6B3D`.
- Sidebar structure matches `~/Documents/antigravity/biosim` — section headings (Geist 0.68rem/600/uppercase/0.12em tracking with `border-bottom` underline), margin-based section spacing (no inter-section borders), no description text under headings. Panel header: Geist 0.72rem same style.
- Tool buttons (`.tool-btn`): 34x34, bare icons, transparent default, `--bg-hover` on hover. SVG defaults (`fill`, `stroke`, `stroke-width`, `stroke-linecap`, `stroke-linejoin`) set via `.tool-btn svg` CSS rule — individual SVGs only need `width`, `height`, `viewBox`. Exception: `#panelClose svg` overrides `stroke-width: 2.5`. Ghost buttons (`.ghost-btn`): pill-shaped with border. Topbar dividers (`.topbar-divider`): 1px wide, 18px tall, `--border` color.
