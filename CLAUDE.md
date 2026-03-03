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
        ├── src/input.js (mouse interaction, particle/photon spawning)
        │     ├── src/vec2.js
        │     └── src/photon.js (for photon spawn mode)
        ├── src/particle.js (entity definition)
        │     └── src/vec2.js
        ├── src/presets.js (preset definitions + loadPreset function)
        └── src/ui.js (setupUI, DOM cache, all event binding)
```

### Key Design Decisions

- **Natural units**: c=1, G=1 throughout the physics engine. All equations use these conventions.
- **Proper velocity integration**: Physics uses proper velocity `w = γv` (celerity) as the primary linear state variable. Velocity is derived via `v = w / √(1 + w²)`, naturally enforcing the speed-of-light limit without mass in the derivation. Kicks use `Δw = F/m · Δt`. In the classical limit (`w ≈ v`), the derivation becomes identity. Spin uses the same pattern: `p.spin` stores proper angular velocity (unbounded), angular velocity is derived via `angVel = spin / √(1 + spin² · radius²)`, naturally capping surface velocity at c.
- **Boris integrator**: Splits forces into position-dependent (E-like: gravity, Coulomb, dipole) and velocity-dependent (B-like: Lorentz, linear GM). E-like forces use half-kick–half-kick; B-like forces use Boris rotation that exactly preserves |v|, giving superior long-term stability for magnetic/gravitomagnetic interactions. Sequence: half-kick(E) → Boris rotate(B) → half-kick(E) → drift → rebuild tree → collisions → new forces+fields. The Boris rotation parameter `t = ((q/(2m))·Bz + 2·Bgz)·dt/γ` combines EM and GM contributions; `s = 2t/(1+t²)` gives the exact rotation. In the proper velocity framework, γ = √(1 + w²) — no NaN risk unlike 1/√(1−v²). Stored forces in `particle.force` Vec2 contain only E-like forces (used by kicks); per-type display vectors (`forceMagnetic`, `forceGravitomag`) include both E-like and B-like contributions for rendering. Accumulated B/Bg field z-components stored in `particle.Bz` and `particle.Bgz`.
- **Force gating**: Each force type (gravity, Coulomb, magnetic, gravitomagnetic) can be independently toggled via `Physics` boolean flags. Magnetic toggle gates dipole forces and Lorentz B-field accumulation; gravitomagnetic toggle gates dipole forces and GM B-field accumulation. Both B-field accumulations feed into the Boris rotation step. Relativity toggle switches between relativistic (`1/√(1+w²)`/`spinToAngVel`) and classical (identity) proper-velocity-to-velocity and spin-to-angVel conversion.
- **Barnes-Hut approximation**: Toggleable via `barnesHutEnabled` (default on). QuadTree stores aggregate mass, charge, angular velocity (magnetic moment, angular momentum), momentum, and center-of-mass per node. `BH_THETA` (0.5) controls accuracy vs. performance tradeoff. Aggregate nodes use average velocity (`totalMomentum/totalMass`) for velocity-dependent forces. When off, computes exact O(N²) pairwise forces — preserves Newton's 3rd law exactly, improving conservation of momentum and angular momentum.
- **Plummer softening**: `SOFTENING_SQ` (25) prevents force singularities via additive softening `rSq_eff = r² + ε²`, keeping F = -dU/dr consistent (no PE-force mismatch at close range). Other named constants: `DESPAWN_MARGIN` (100). Bounce friction (`Physics.bounceFriction`, default 0.4) is an instance property adjustable via sidebar slider.
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
- **Particle tooltip**: Hover over particles shows compact stats (mass, charge, spin, speed). Click to select and display live stats in a sidebar section (mass, charge, spin as surface velocity in units of c, speed in c, gamma, total force).

### Energy Conservation

Energy stats computed per frame in the sidebar Energy section. Total energy (top-level row) = Linear KE + Spin KE + Potential + Field Energy + Radiated, with each component and drift shown as indented sub-rows (`.stat-sub`). Radiated energy tracks cumulative energy lost to Abraham-Lorentz radiation. Linear KE: relativistic `(γ-1)mc²` or classical `½mv²`. Spin KE: relativistic `m(√(1+L²/m²)-1)` where `L=I·S` or classical `½Iω²`, using `I = (2/5)mr²` uniform-density solid sphere via `INERTIA_K`. Potential: gravitational PE (`-Gm₁m₂/r`), Coulomb PE (`kq₁q₂/r`), magnetic dipole PE (`+(μ₁μ₂)/r³` with `μ=⅕qωr²`, aligned repels), gravitomagnetic dipole PE (`-(L₁L₂)/r³` with `L=Iω`, co-rotating attracts). All PE computed with Plummer-softened r = √(r²+ε²). Drift is percentage change from initial total energy.

### Conserved Quantities

Momentum and angular momentum stats computed per frame in the sidebar Conserved Quantities section. Momentum is `|Σ(mᵢwᵢ)|` (magnitude of total relativistic proper momentum), with drift sub-row showing percentage change from initial value. Angular momentum is computed about the center of mass with orbital `Σ(rᵢ × mᵢwᵢ)` and spin `Σ(IᵢSᵢ)` shown as separate sub-rows, plus a drift sub-row. `I = (2/5)mr²`. Conserved with gravity and Coulomb only. Velocity-dependent forces (Lorentz, linear GM) break Newton's 3rd law — see Force Types section. Turning off Barnes-Hut improves conservation by ensuring exact pairwise symmetry for radial forces. Initial values for drift tracking reset when particles are added or the simulation is cleared.

### Sign Conventions (IMPORTANT)

Gravitomagnetism is the gravitational analog of electromagnetism, but **all GEM interactions are attractive** (gravity has only one sign of "charge"):
- **GM dipole**: co-rotating masses **attract** (opposite sign from EM dipole, where aligned dipoles repel).
- **Linear GM**: co-moving masses **attract** (same qualitative behavior as parallel currents in EM, but the GEM coupling constant has an extra factor of 4 and opposite sign convention).

In code: the GM dipole force coefficient is `+3L₁L₂/r⁴` (positive = attractive along separation vector) and the GM Boris rotation parameter is `+2·Bgz`, which produces the same rotational sense as the EM Lorentz force for co-moving particles. Do NOT flip these signs — the intent is that gravitomagnetism always creates attractive corrections between masses.

### Force Types

Six force components per particle pair, organized under five toggles:

**Radial forces** (along separation vector):
- **Gravity** (`gravityEnabled`): `m₁m₂/r²`, attractive.
- **Coulomb** (`coulombEnabled`): `-q₁q₂/r²`, like-repels.
- **Magnetic dipole** (`magneticEnabled`): `-3μ₁μ₂/r⁴` where `μ = ⅕qωr²` (uniform charge density solid sphere, `MAG_MOMENT_K`), aligned ⊥-to-plane dipoles repel (standard 3D result).
- **Gravitomagnetic dipole** (`gravitomagEnabled`): `+3L₁L₂/r⁴` where `L = Iω = (2/5)mr²ω` (angular momentum), co-rotating masses **attract** (GEM flips EM sign). Angular velocity is bounded by relativistic derivation (`angVel = spin/√(1+spin²r²)`), so GM dipole can never overwhelm gravity when relativity is on.

**Velocity-dependent forces** (perpendicular to velocity, do no work — handled by Boris rotation):
- **Lorentz force** (`magneticEnabled`): Moving charges create magnetic fields `B_z = q_s(v_s×r̂)_z/r²` that deflect other moving charges. Accumulated as `p.Bz` and applied via Boris rotation with parameter `t_em = (q/(2m))·Bz·dt/γ`.
- **Linear gravitomagnetism** (`gravitomagEnabled`): Co-moving masses **attract** (frame-dragging). `Bg_z = m_s(v_s×r̂)_z/r²`. Accumulated as `p.Bgz` and applied via Boris rotation with parameter `t_gm = +2·Bgz·dt/γ` (factor of 4 from standard GEM, sign chosen so co-moving masses attract). Also accumulates `∇Bgz` for GM spin-orbit coupling and frame-dragging torque for spin alignment.

**Radiation reaction** (`radiationEnabled`):
- **Abraham-Lorentz radiation** via Landau-Lifshitz approximation: charged accelerating particles radiate energy. Force has two terms: jerk `τ·dF/dt` and Schott damping `τ·|F|²·v/m`, where `τ = 2·LARMOR_K·q²/m`. Relativistic correction divides by `γ³`. Clamped via `LL_FORCE_CLAMP` (max impulse as fraction of `|w|`). Radiated energy tracked in `sim.totalRadiated` and photons spawned when `dE > RADIATION_THRESHOLD`. Particle stores `prevForce` Vec2 for jerk computation across substeps.

**Tidal breakup** (`tidalEnabled`):
- Combined surface force disintegration: particles fragment when outward forces exceed self-gravity `m/r²`. Outward forces include: tidal stretching `TIDAL_STRENGTH·M·r/d³` from nearby bodies, centrifugal `ω²r` from spin, and Coulomb self-repulsion `q²/(4r²)`. Self-disruption (centrifugal + Coulomb) is checked first with early `continue` before the O(N) neighbor scan. Fragments into `FRAGMENT_COUNT` (3) pieces with tangential kick from parent spin.

**Known limitation:** Velocity-dependent forces (Lorentz, linear GM) do not satisfy Newton's 3rd law between particles — the force on A from B's field is not equal and opposite to the force on B from A's field. In real physics, the missing momentum/angular momentum is carried by the EM/GEM field. This particle-only simulation has no field degrees of freedom, so momentum and angular momentum are not exactly conserved when magnetic or gravitomagnetic forces are active. Radial forces (gravity, Coulomb, dipole) are central and conserve momentum/angular momentum exactly in pairwise mode.

**Spin-orbit coupling** (`magneticEnabled` + `relativityEnabled`): EM spin-orbit transfers energy between translational and spin KE via `dE = -μ·(v·∇Bz)·dt` where `μ = MAG_MOMENT_K·q·ω·r²`. **GM spin-orbit** (`gravitomagEnabled` + `relativityEnabled`): same pattern using angular momentum `L = I·ω` and `∇Bgz`. **Frame-dragging torque** (`gravitomagEnabled`): drives spins toward co-rotation via `τ = FRAME_DRAG_K·m_s·(ω_s - ω_p)/(r³)`. Spin also changes via collision angular momentum transfer (merge or bounce friction).

### Collision Modes

Three modes in physics.js: `pass` (no-op), `merge` (conserves mass, charge, momentum, and angular momentum — orbital angular momentum about the pair's COM plus spin angular momentum `I·spin` maps to merged particle's spin via `I = (2/5)mr²`), `bounce` (elastic with spin-friction transfer via `Δspin = J/(INERTIA_K·m·r)`, configurable friction coefficient via `Physics.bounceFriction` slider). Bounce uses relativistic elastic collision when relativity is on (Lorentz boost to COM frame, reversal, boost back — conserves both m·w momentum and m·γ energy), classical elastic collision when off (conserves m·v and ½mv²).

### Input Modes

Four placement modes in input.js: `place` (spawn at rest), `shoot` (drag distance sets velocity at 0.1x multiplier), `orbit` (calculates circular orbit velocity around nearest massive body), `photon` (spawn photon in mouse-movement direction; random direction if stationary). Touch events delegate to mouse handlers for mobile support.

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
- `.stat-sub` — indented stat sub-rows (smaller/muted text) for energy components under Total, angular momentum orbital/spin split, and drift values
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

- **Shortcuts** via `initShortcuts()` from `shared-shortcuts.js`: Space (pause), R (reset), `.` (step), P (presets), 1-5 (load preset), V (velocity vectors), F (force vectors), C (force components), T (theme), S (sidebar), Esc (close/deselect).
- **Info tips** via `createInfoTip()` from `shared-info.js`: `?` buttons next to Energy, Conserved Quantities, Particle Properties (spin), each force toggle (gravity, Coulomb, magnetic, gravitomagnetic, Barnes-Hut), Interaction mode, Collision mode, Boundary mode. Data defined inline in `ui.js`.

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
- Particle radius is `cbrt(mass)` (uniform density sphere with ρ = 3/(4π)). Moment of inertia `I = INERTIA_K·m·r²` with `INERTIA_K = 0.4` (solid sphere). Magnetic moment `μ = MAG_MOMENT_K·q·ω·r²` with `MAG_MOMENT_K = 0.2` (uniform charge density solid sphere). GM moment `L = Iω`. Both linear and rotational state variables use the same derivation pattern: `derived = state / √(1 + state² × scale²)`. Linear: `v = w / √(1 + w²)` where `p.w` is proper velocity (γv). Rotational: `angVel = spin / √(1 + spin² · r²)` where `p.spin` is proper angular velocity. Both naturally cap derived quantities below c. When relativity is off, derivation is identity (`v = w`, `angVel = spin`).
