# CLAUDE.md

Part of the **a9l.im** portfolio. See parent `site-meta/CLAUDE.md` for the shared design system specification. Sibling projects: `biosim`, `gerry`.

## Overview

Relativistic N-body physics simulation — interactive browser-based app using Barnes-Hut algorithm for O(N log N) force calculation. Pure vanilla JS with ES6 modules, no build system or dependencies.

## Running Locally

```bash
python -m http.server
# Navigate to http://localhost:8000
```

Opening `index.html` directly may fail due to CORS restrictions on ES6 module imports. No build step, test framework, or linter configured.

## Architecture

### Module Dependency Graph

```
index.html
  ├── /shared-base.css (shared reset, layout tokens, .glass, .tool-btn, intro, keyframes, sim layout, toast, responsive)
  ├── styles.css (project-specific overrides + form controls, preset dialog, range sliders, mode toggles, checkboxes, ghost buttons)
  ├── /shared-tokens.js (_r, _FONT, _PALETTE base + extended — shared across all a9l.im sites)
  ├── /shared-utils.js (clamp, lerp, cubicBezier, showToast, debounce, throttle, escapeHtml)
  ├── /shared-camera.js (createCamera — viewport/zoom/pan module, bindZoomButtons)
  ├── colors.js (extends _PALETTE with particle hues, freezes, injects project CSS vars)
  └── main.js (Simulation class, ES module)
        ├── src/config.js (named constants: BH_THETA, ZOOM_MIN/MAX, physics params)
        ├── src/relativity.js (spinToAngVel, setVelocity)
        ├── src/physics.js (force calculation, integration, collisions)
        │     ├── src/quadtree.js (Barnes-Hut spatial partitioning)
        │     └── src/vec2.js (2D vector math)
        ├── src/renderer.js (Canvas 2D drawing, trails, themes — uses shared camera)
        ├── src/input.js (mouse interaction, particle spawning)
        │     └── src/vec2.js
        ├── src/particle.js (entity definition)
        │     └── src/vec2.js
        ├── src/presets.js (preset definitions + loadPreset function)
        └── src/ui.js (setupUI, DOM cache, all event binding)
```

### Key Design Decisions

- **Natural units**: c=1, G=1 throughout the physics engine. All equations use these conventions.
- **Proper velocity integration**: Physics uses proper velocity `w = γv` (celerity) as the primary linear state variable. Velocity is derived via `v = w / √(1 + w²)`, naturally enforcing the speed-of-light limit without mass in the derivation. Kicks use `Δw = F/m · Δt`. In the classical limit (`w ≈ v`), the derivation becomes identity. Spin uses the same pattern: `p.spin` stores proper angular velocity (unbounded), angular velocity is derived via `angVel = spin / √(1 + spin² · radius²)`, naturally capping surface velocity at c.
- **Velocity Verlet integration**: Kick-drift-kick scheme for time-symmetric, energy-conserving integration. Each step: half-kick proper velocity (old forces, F/m) → drift position → recalculate forces → half-kick proper velocity (new forces). Stored forces in `particle.force` Vec2. Lorentz factor computed as `γ = √(1 + w²)` — no NaN risk unlike `1/√(1-v²)`.
- **Force gating**: Each force type (gravity, Coulomb, magnetic, gravitomagnetic, spin-orbit) can be independently toggled via `Physics` boolean flags. Magnetic toggle gates both dipole and Lorentz forces; gravitomagnetic toggle gates both dipole and linear GM forces. Relativity toggle switches between relativistic (`1/√(1+w²)`/`spinToAngVel`) and classical (identity) proper-velocity-to-velocity and spin-to-angVel conversion.
- **Barnes-Hut approximation**: QuadTree stores aggregate mass, charge, angular velocity (magnetic moment, angular momentum), momentum, and center-of-mass per node. `BH_THETA` (0.5) controls accuracy vs. performance tradeoff. Aggregate nodes use average velocity (`totalMomentum/totalMass`) for velocity-dependent forces.
- **Softening parameter**: `MIN_DIST_SQ` (25) prevents force singularities at close range. Other named constants: `BOUNCE_FRICTION` (0.4), `DESPAWN_MARGIN` (100).
- **Zoom range**: Clamped to 1x–3x in all input paths (mouse wheel, pinch-to-zoom, and zoom buttons).
- **Minimal global state**: The `Simulation` instance owns all runtime state (`window.sim` for console debugging). Design tokens (`window._PALETTE`, `window._FONT`, `window._r`) are frozen globals set by `colors.js` and consumed by ES modules via `window._PALETTE`.

### Rendering

- Dark mode uses additive blending (`globalCompositeOperation: 'lighter'`) for glow effects.
- Trail history: up to 200 positions per particle, stored as circular buffers (`{ data: Float32Array, len, start }`) in a Map keyed by particle ID.
- Spin ring colors are precomputed at module scope (`_spinColors`) — 4 hsla strings (2 hues × 2 themes).
- Particle color is computed from `_PALETTE` charge hues (`chargePos=201` blue, `chargeNeg=7` red, `neutral` from `extended.slate`) with dynamic HSL saturation/lightness based on charge magnitude.
- **Velocity vectors**: Optional white arrows from particle center in velocity direction, scaled by speed.
- **Force vectors**: Optional accent-colored arrows from particle center in net force direction, scaled by magnitude.
- **Force component vectors**: Optional per-force-type arrows (gravity=slate, Coulomb=blue, magnetic=cyan, gravitomagnetic=purple) showing individual force contributions. Each particle stores `forceGravity`, `forceCoulomb`, `forceMagnetic`, `forceGravitomag` Vec2s accumulated during force calculation.
- **Torque arcs**: Curved arrows around particles showing spin-orbit torque. Displayed when force vectors or force components are toggled on. Net torque uses accent color; component mode shows magnetic (cyan) and gravitomagnetic (purple) torques separately at opposite angular offsets. Each particle stores `torqueMagnetic` and `torqueGravitomag` scalars.
- **Particle tooltip**: Hover over particles shows compact stats (mass, charge, spin, speed). Click to select and display live stats in a sidebar section (mass, charge, spin, speed, gamma, total force, torque).

### Energy Conservation

Energy stats computed per frame: linear KE (relativistic `(γ-1)mc²` or classical `½mv²`), rotational KE (`½m·spin²`), gravitational PE (`-Gm₁m₂/r`), Coulomb PE (`kq₁q₂/r`). Total energy and drift percentage displayed in sidebar.

### Force Types

Six force components per particle pair, organized under five toggles:

**Radial forces** (along separation vector):
- **Gravity** (`gravityEnabled`): `m₁m₂/r²`, attractive.
- **Coulomb** (`coulombEnabled`): `-q₁q₂/r²`, like-repels.
- **Magnetic dipole** (`magneticEnabled`): `(q₁ω₁)(q₂ω₂)/r⁴`, aligned-attracts. Uses derived angular velocity `angVel`.
- **Gravitomagnetic dipole** (`gravitomagEnabled`): `-(m₁ω₁)(m₂ω₂)/r⁴`, co-rotating-repels. Angular velocity is bounded by relativistic derivation (`angVel = spin/√(1+spin²r²)`), so GM dipole can never overwhelm gravity when relativity is on.

**Velocity-dependent forces** (perpendicular to velocity, do no work):
- **Lorentz force** (`magneticEnabled`): Moving charges create magnetic fields `B_z = q_s(v_s×r̂)_z/r²` that deflect other moving charges. `F = q·v×B`.
- **Linear gravitomagnetism** (`gravitomagEnabled`): Same structure for mass currents, opposite sign. Co-moving masses repel via `Bg_z = m_s(v_s×r̂)_z/r²`.

**Spin-orbit coupling** (`spinOrbitEnabled`): B and Bg fields from moving sources drive spin evolution. `d(spin)/dt = (q/m)·B_z - Bg_z`. Integrated via Verlet half-kicks alongside proper velocity. Torque displayed in selected particle stats.

### Collision Modes

Three modes in physics.js: `pass` (no-op), `merge` (conserves mass/charge/momentum), `bounce` (elastic with spin-friction transfer, friction coefficient 0.4).

### Input Modes

Three placement modes in input.js: `place` (spawn at rest), `shoot` (drag distance sets velocity at 0.1x multiplier), `orbit` (calculates circular orbit velocity around nearest massive body). Touch events delegate to mouse handlers for mobile support.

## Color System

Two-layer token system:
- **`/shared-tokens.js`** (shared): `_r`, color math helpers, `_FONT`, `_PALETTE` with shared tokens and `extended` sub-object.
- **`colors.js`** (project-specific): Extends `_PALETTE` with particle hues (`chargePos`, `chargeNeg` as hue integers, `neutral` from `_PALETTE.extended.slate`). Injects `--danger`/`--danger-subtle` CSS vars. Freezes all objects. Exposes `_PALETTE`, `_FONT`, `_r` on `window` for ES modules.

JS modules alias as `const _PAL = window._PALETTE`.

## UI & Layout

- **Topbar** (`#topbar.sim-toolbar`): floating frosted-glass bar with stat chips (left), tool buttons (right). Layout: Presets | divider | Pause, Step, Reset | divider | Theme toggle, Settings.
- **Control panel** (`#control-panel.sim-panel`): right-side slide-in sidebar, toggleable. Does not block canvas interaction. Sections with `.panel-section`, `.group-label` headings (Geist 0.68rem/600/uppercase/0.12em tracking).
- **Preset dialog** (`#preset-dialog.preset-dialog`): centered modal card grid. Uses shared `.preset-dialog`/`.preset-content`/`.preset-grid`/`.preset-card` classes from `shared-base.css`. Presets defined inline in `Simulation.loadPreset()`.
- **Hint bar** (`#hint-bar`): floating bottom pill with instruction text.
- **Intro screen**: themed splash, uses shared intro CSS.
- Icon swaps (pause/play, sun/moon): both SVGs embedded in HTML with `hidden` attribute; JS toggles `hidden` instead of replacing `innerHTML`.

### Responsive Breakpoints

- **900px**: `--inset: 8px`, `--panel-w: 100%`. Toolbar stats hidden. Control panel becomes bottom sheet (`translateY(100%)` → `translateY(0)` on open). Panel body gets drag handle (`::before`). Tool buttons shrink to 32×32. Shared rules handle `--toolbar-h: 48px`, `.sim-toolbar` positioning, `.sim-brand` sizing.
- **600px** (shared): brand shrinks, toolbar actions tighter, preset content padding reduces.
- **440px** (shared): `.hide-sm` hides elements, preset grid goes single-column.

### Project-Specific CSS (styles.css)

- Spacing tokens (`--sp-*`, `--inset`) — physsim-specific layout system
- `.panel-section` (has flex layout beyond basic margin)
- Global `label` styling (can't share without conflicts)
- Form controls (moved from shared-base.css): `.slider-value`, `input[type=range]` (WebKit + Moz), `.mode-toggles`/`.mode-btn`, `.checkbox-label`/`input[type="checkbox"]`, `.ghost-btn`
- Preset dialog (moved from shared-base.css): `.preset-dialog`, `.preset-backdrop`, `.preset-content`, `.preset-title`, `.preset-grid`, `.preset-card`, `.preset-name`, `.preset-desc` (with 600px/440px responsive overrides)
- `prefers-reduced-motion` backdrop-filter removal (supplements shared)
- Theme toggle icon CSS

## Key Patterns

### JS / Performance

- All vector operations use the `Vec2` class. Use `vec.set(x, y)` for in-place mutation in hot paths; prefer `Vec2.add(a, b)` static methods elsewhere.
- Physics hot path avoids allocations: `calculateForce()` accumulates into an `out` Vec2 parameter; force array is reused across frames.
- DOM elements cached in `Simulation.dom`; UI mode state tracked in JS variables — no per-frame DOM queries.
- `InputHandler` caches DOM refs (`massInput`, `chargeInput`, `spinInput`) and tracks `mode` state directly — no per-spawn DOM queries.

### Keyboard Shortcuts & Info Tips

- **Shortcuts** via `initShortcuts()` from `shared-shortcuts.js`: Space (pause), R (reset), `.` (step), P (presets), 1-5 (load preset), V (velocity vectors), F (force vectors), C (force components), T (theme), S (sidebar), O (spin-orbit toggle), Esc (close/deselect).
- **Info tips** via `createInfoTip()` from `shared-info.js`: `?` buttons next to Energy, Particle Properties, each force toggle, Interaction mode, Collision mode, Boundary mode. Data defined inline in `ui.js`.

### CSS Patterns

- **`.glass`** (from `shared-base.css`): applied to topbar, panel-body, preset-content, hint-bar. Panel-body and preset-content override to `--shadow-lg`; hint-bar overrides to `--shadow-sm`.
- **`.tool-btn`** (from `shared-base.css`): base 34×34. Exception: `#panelClose svg` overrides `stroke-width: 2.5`.
- **`.ghost-btn`** (from `shared-base.css`): pill-shaped with border.
- **Topbar dividers** (`.topbar-divider`): 1px wide, 18px tall, `--border` color.
- Theme toggle sets `data-theme` on `<html>` (not body). `<html>` has `data-theme="light"` in markup for FOUC prevention.
- Shared keyframes used: `slideDown` (toolbar entrance), `slideInRight` (panel slide-in), `paletteEnter` (hint bar entrance).

## Gotchas

- **CORS on local file://**: Must use an HTTP server — ES6 module imports fail on `file://` protocol.
- **Shared CSS at domain root** — `shared-base.css` is loaded via `/shared-base.css` (absolute path). When serving locally, serve from the parent `a9lim.github.io/` directory or the shared file won't resolve.
- **Preset dialog needs both ID and class** — `#preset-dialog` has `class="preset-dialog"` so both the shared CSS (`.preset-dialog`) and any JS targeting the ID work correctly.
- **`data-theme="light"` must be on `<html>`** — CSS theme rules depend on it before JS runs.
- Particle radius is `cbrt(mass)` (uniform density sphere with ρ = 3/(4π)). Both linear and rotational state variables use the same derivation pattern: `derived = state / √(1 + state² × scale²)`. Linear: `v = w / √(1 + w²)` where `p.w` is proper velocity (γv). Rotational: `angVel = spin / √(1 + spin² · r²)` where `p.spin` is proper angular velocity. Both naturally cap derived quantities below c. When relativity is off, derivation is identity (`v = w`, `angVel = spin`).
