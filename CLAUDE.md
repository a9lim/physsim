# CLAUDE.md

Part of the **a9l.im** portfolio. See parent `site-meta/CLAUDE.md` for the shared design system. Sibling projects: `biosim`, `gerry`.

## Running Locally

```bash
cd path/to/a9lim.github.io && python -m http.server
```

Must serve from parent `a9lim.github.io/` directory — shared files (`/shared-base.css`, `/shared-tokens.js`, etc.) load via absolute paths. ES6 modules require HTTP server (no `file://`). No build step, test framework, or linter.

## Module Graph

```
main.js (Simulation class, window.sim)
├── src/config.js          — named constants (BH_THETA, SOFTENING_SQ, PHYSICS_DT, INERTIA_K, MAG_MOMENT_K, LARMOR_K, etc.)
├── src/relativity.js      — angwToAngVel, angVelToAngw, setVelocity
├── src/energy.js          — computeEnergies(): KE, spin KE, momentum, angular momentum, Darwin field energy
├── src/integrator.js      — Physics class: adaptive Boris substep loop, spin-orbit, frame-drag, radiation, tidal breakup
│     ├── src/forces.js        — resetForces, computeAllForces, pairForce, calculateForce (BH walk)
│     ├── src/collisions.js    — handleCollisions, resolveMerge, resolveBounce
│     ├── src/potential.js     — computePE, treePE, pairPE
│     ├── src/signal-delay.js  — getDelayedState, interpolateHistory (retarded potentials)
│     ├── src/quadtree.js      — QuadTreePool: SoA pool-based Barnes-Hut quadtree (zero per-frame allocation)
│     └── src/photon.js        — radiation photon entity
├── src/stats-display.js   — StatsDisplay: energy/momentum/drift DOM updates, selected particle info
│     └── src/energy.js
├── src/renderer.js        — Canvas 2D, trails, glow, force vectors, themes
├── src/input.js           — mouse/touch, Place/Shoot/Orbit modes
├── src/particle.js        — entity (pos, vel, w, angw, angVel, mass, charge, radius, force vectors)
├── src/heatmap.js         — density heatmap
├── src/phase-plot.js      — phase space plot (sidebar canvas)
├── src/sankey.js          — energy breakdown bar chart (sidebar canvas)
├── src/presets.js         — preset definitions
└── src/ui.js              — setupUI, event binding, info tips (infoData object)
```

Shared scripts loaded in `<head>` before modules: `/shared-tokens.js` → `/shared-utils.js` → `/shared-camera.js` → `colors.js` → `/shared-info.js` → `/shared-shortcuts.js`.

## Physics Engine

### State Variables

Natural units: c = 1, G = 1. Both linear and rotational state use the same pattern:

| State | Derived | Formula | Cap |
|-------|---------|---------|-----|
| `p.w` (proper velocity, γv) | `p.vel` | `v = w / √(1 + w²)` | |v| < c |
| `p.angw` (angular celerity) | `p.angVel` | `ω = W / √(1 + W²r²)` | surface vel < c |

When relativity is off, derivation is identity (`v = w`, `angVel = angw`). Kicks use `Δw = F/m · Δt`. Particle radius = `cbrt(mass)` (ρ = 3/(4π)). `I = INERTIA_K·m·r²` (0.4, solid sphere). `μ = MAG_MOMENT_K·q·ω·r²` (0.2). `L = I·ω`.

### Boris Integrator

Per substep: half-kick(E) → Boris rotate(B) → half-kick(E) → drift → rebuild tree → collisions → new forces.

- **E-like forces** (position-dependent, stored in `p.force`): gravity, Coulomb, magnetic dipole, GM dipole
- **B-like forces** (velocity-dependent, Boris rotation): Lorentz `p.Bz`, linear GM `p.Bgz`
- Boris parameter: `t = ((q/(2m))·Bz + 2·Bgz)·dt/γ`, rotation `s = 2t/(1+t²)`
- Per-type display vectors (`forceMagnetic`, `forceGravitomag`) include both E-like and B-like contributions; `p.force` contains only E-like

### Force Types

**Radial** (along separation):
- **Gravity**: `+m₁m₂/r²` attractive
- **Coulomb**: `-q₁q₂/r²` like-repels
- **Magnetic dipole** (`magneticEnabled`): `-3μ₁μ₂/r⁴`, aligned ⊥-to-plane dipoles repel
- **GM dipole** (`gravitomagEnabled`): `+3L₁L₂/r⁴`, co-rotating masses **attract** (GEM flips EM sign)

**Velocity-dependent** (Boris rotation, perpendicular to v):
- **Lorentz** (`magneticEnabled`): `Bz = q_s·(v_s×r̂)_z/r²`
- **Linear GM** (`gravitomagEnabled`): `Bgz = m_s·(v_s×r̂)_z/r²`, `t_gm = +2·Bgz·dt/γ`. Also accumulates `∇Bgz` for spin-orbit and frame-dragging torque.

**Radiation** (`radiationEnabled`, requires Relativity): Landau-Lifshitz approximation. Larmor power P = 2q²a²/3. Force = `τ·(dF/dt - |F|²·v/m)` where `τ = 2q²/(3m)` (`LARMOR_K = 1/3`). Divided by γ³. Clamped by `LL_FORCE_CLAMP`. Photons spawned when `dE > RADIATION_THRESHOLD`, tracked in `sim.totalRadiated` and `sim.totalRadiatedPx/Py`.

**Signal Delay** (`signalDelayEnabled`, requires Relativity + BH off): Retarded potentials via Newton-Raphson light-cone solve on per-particle history buffers (`HISTORY_SIZE`).

**Spin-orbit** (`spinOrbitEnabled` + Relativity): `dE = -μ·(v·∇Bz)·dt` for EM (requires `magneticEnabled`), same with `L` and `∇Bgz` for GM. Gradient `∇Bz` has radial (`+3·Bz·r̂/r²`) and angular (`q_s·v_s⊥/r³`) terms. Frame-dragging torque: `τ = FRAME_DRAG_K·m_s·(ω_s - ω_p)/r³`.

**Tidal breakup** (`tidalEnabled`): fragments when tidal (`M·r/d³`) + centrifugal (`ω²r`) + Coulomb self-repulsion (`q²/4r²`) > self-gravity (`m/r²`). Splits into `FRAGMENT_COUNT` (3) pieces.

### Sign Conventions (IMPORTANT)

All GEM interactions are **attractive** (gravity has one sign of "charge"):
- GM dipole coefficient `+3L₁L₂/r⁴` (positive = attractive)
- GM Boris parameter `+2·Bgz` (co-moving masses attract)

Do NOT flip these signs.

### Potential Energy

Computed separately from forces via `Physics.computePE()` using the same BH theta criterion (tree traversal when BH on, exact pairwise when off). Includes gravitational, Coulomb, magnetic dipole, and GM dipole PE. All use Plummer softening `rSq + SOFTENING_SQ`.

### Energy & Momentum (`src/energy.js`)

`computeEnergies()` returns: `linearKE`, `spinKE`, `pe`, `fieldEnergy`, `fieldPx/Py`, `px/py`, `orbitalAngMom`, `spinAngMom`.

- **Spin KE**: relativistic `INERTIA_K·m·(√(1+W²r²)-1)`, classical `½Iω²`
- **Field energy**: EM + gravitational Darwin Lagrangian O(v²/c²) corrections
- **Momentum**: particle `Σ(mᵢwᵢ)` + Darwin field + `sim.totalRadiatedPx/Py`
- **Angular momentum**: orbital `Σ(rᵢ×mᵢwᵢ)` + spin `Σ(IᵢWᵢ)` about COM

Conserved exactly with gravity+Coulomb only, pairwise mode (BH off). Velocity-dependent forces break Newton's 3rd law — missing momentum carried by fields not modeled.

### Collisions

- **Pass**: no-op
- **Merge**: conserves mass, charge, momentum, angular momentum. Orbital L about pair COM + spin L → merged `angw` via `I = (2/5)mr²`.
- **Bounce**: elastic (relativistic: Lorentz boost to COM, classical: standard). Spin friction `Δω = J/I` where `I = INERTIA_K·m·r²`. Relativistic path converts through `angVelToAngw()`. Configurable friction via `Physics.bounceFriction` (0.4 default, sidebar slider).

### Barnes-Hut

Toggleable (`barnesHutEnabled`). QuadTreePool (SoA, pre-allocated, zero per-frame GC) aggregates mass, charge, angVel, magnetic moment, angular momentum, momentum, COM. `BH_THETA = 0.5`. When off: exact pairwise, better conservation. Adaptive substepping: `dtSafe = min(√(ε/a_max), T_cyclotron/8)`, `nSteps = min(ceil(dt/dtSafe), MAX_SUBSTEPS)`.

### Fixed-Timestep Loop

`PHYSICS_DT = 1/120`. Accumulator in `main.js` collects `rawDt * speedScale` per frame. While loop drains in fixed-size `PHYSICS_DT` steps. Capped by `MAX_SUBSTEPS * PHYSICS_DT * 4`. Photon updates and tidal breakup inside the fixed-step loop; energy/rendering/DOM outside.

## Toggle Dependencies

```
Gravity → Gravitomagnetic (sub-toggle)
Coulomb → Magnetic (sub-toggle)
Relativity → Radiation (sub-toggle)
            → Spin-Orbit (sub-toggle)
Relativity + BH off → Signal Delay
```

Disabled toggles get `.ctrl-disabled` (opacity 0.4, pointer-events none). Toggle colors: Gravity/GM = slate, Coulomb/Magnetic = blue, Relativity chain = yellow, Tidal = red.

## UI

- **4-tab sidebar**: Settings (particle props, interaction mode, forces, physics), Engine (BH, collision, boundary, visuals, speed), Stats (energy bar chart + numbers), Particle (selected particle details, phase plot)
- **Topbar**: Presets | Pause/Step/Reset | Theme/Settings
- **Preset dialog**: modal card grid, keyboard `P` or `1-5`
- **Intro screen**: themed splash with shared CSS
- **Theme**: `data-theme` on `<html>` (not body). Light default for FOUC prevention.
- **Responsive**: 900px → bottom sheet + 48px toolbar; 600px/440px shared breakpoints
- Phase plot and energy bar chart are always-on sidebar canvases (no toggle)
- Icon swaps (pause/play, sun/moon): toggle `hidden` attribute, not innerHTML

## Key Patterns

- `Vec2` for all vector math. `vec.set(x,y)` in hot paths; `Vec2.add(a,b)` elsewhere.
- Physics hot path: `pairForce()` in `forces.js` accumulates into `out` Vec2 parameter, no allocations. Toggle flags passed as reusable `_toggles` object (synced once per `update()`, not per-frame allocation).
- QuadTreePool: SoA flat typed arrays, pre-allocated 512 nodes. `pool.reset()` + `pool.build()` per substep, zero GC.
- DOM cached in `Simulation.dom` and `Simulation.selDom`. Shared by reference with `StatsDisplay`. No per-frame DOM queries.
- `InputHandler` caches DOM refs and tracks mode state directly.
- `window.sim` for console debugging. `window._PALETTE`/`window._FONT` frozen by `colors.js`.
- Shortcuts via `initShortcuts()`: Space, R, `.`, P, 1-5, V, F, C, T, S, Esc, `?`.
- Info tips via `createInfoTip()`: data defined in `infoData` object in `ui.js`.
- Dark mode: additive blending (`globalCompositeOperation: 'lighter'`).
- Particle color from charge hues (`chargePos=201`, `chargeNeg=7`, neutral from `extended.slate`).

## Gotchas

- Serve from `a9lim.github.io/` parent — `/shared-base.css` and `/shared-tokens.js` use absolute paths
- `#preset-dialog` needs both ID and `class="preset-dialog"` (shared CSS uses class, JS uses ID)
- `photon.js` is imported by `integrator.js` for radiation — not related to input modes
