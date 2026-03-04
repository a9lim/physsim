# CLAUDE.md

Part of the **a9l.im** portfolio. See parent `site-meta/CLAUDE.md` for the shared design system. Sibling projects: `biosim`, `gerry`.

## Running Locally

```bash
cd path/to/a9lim.github.io && python -m http.server
```

Serve from the parent `a9lim.github.io/` directory — shared files (`/shared-base.css`, `/shared-tokens.js`, etc.) load via absolute paths. ES6 modules require HTTP (no `file://`). No build step, test framework, or linter.

## Module Graph

```
main.js (Simulation class, window.sim)
├── src/config.js          — Named constants
├── src/relativity.js      — angwToAngVel, angVelToAngw, setVelocity
├── src/topology.js        — TORUS/KLEIN/RP2, minImage(), wrapPosition()
├── src/energy.js          — computeEnergies(): KE, spin KE, momentum, angular momentum, Darwin field energy
├── src/integrator.js      — Physics class: adaptive Boris substep loop
│   ├── src/forces.js      — resetForces, computeAllForces, compute1PNPairwise, pairForce, calculateForce (BH walk)
│   ├── src/collisions.js  — handleCollisions, resolveMerge, resolveBounce
│   ├── src/potential.js   — computePE, treePE, pairPE
│   ├── src/signal-delay.js— getDelayedState, interpolateHistory
│   ├── src/topology.js    — (shared) minimum-image separation and boundary wrapping
│   ├── src/quadtree.js    — QuadTreePool: SoA pool-based Barnes-Hut tree (zero per-frame alloc)
│   └── src/photon.js      — Radiation photon entity
├── src/stats-display.js   — StatsDisplay: energy/momentum/drift DOM updates, selected particle info
│   └── src/energy.js
├── src/renderer.js        — Canvas 2D, trails, glow, force/torque vectors, themes
├── src/input.js           — Mouse/touch, Place/Shoot/Orbit modes
├── src/particle.js        — Entity (pos, vel, w, angw, angVel, mass, charge, radius, per-type force vectors)
├── src/heatmap.js         — Gravitational potential field overlay (48×48 grid, updates every 6 frames)
├── src/phase-plot.js      — Phase space plot (sidebar canvas, selected particle)
├── src/sankey.js          — Energy breakdown bar chart (sidebar canvas)
├── src/presets.js         — Preset definitions (Solar System, Binary Star, Galaxy, Collision, Magnetic Spin)
└── src/ui.js              — setupUI, event binding, info tips (infoData object)
```

Shared scripts loaded in `<head>` before modules: `/shared-tokens.js` → `/shared-utils.js` → `/shared-camera.js` → `colors.js` → `/shared-info.js` → `/shared-shortcuts.js`.

## Physics Engine

### Natural Units

c = 1, G = 1 throughout. All velocities are fractions of c. All forces are dimensionless.

### State Variables

Both linear and rotational state use the same proper-velocity pattern:

| State variable | Derived | Formula | Cap |
|---|---|---|---|
| `p.w` (proper velocity, γv) | `p.vel` | **v** = **w** / √(1 + w²) | \|v\| < c |
| `p.angw` (angular celerity) | `p.angVel` | ω = W / √(1 + W²r²) | surface vel < c |

When relativity is off: `vel = w`, `angVel = angw` (identity).

Derived quantities from spin:
- Moment of inertia: `I = INERTIA_K · m · r²` (0.4 = 2/5, solid sphere)
- Magnetic moment: `μ = MAG_MOMENT_K · q · ω · r²` (0.2 = 1/5, uniform charge sphere)
- Angular momentum: `L = I · ω`
- Particle radius: `r = cbrt(mass)` (density ρ = 3/(4π))

### Per-Particle Force/Torque Display Vectors

Each particle stores per-type force vectors for component visualization: `forceGravity`, `forceCoulomb`, `forceMagnetic`, `forceGravitomag`, `force1PN`, `forceSpinCurv`, `forceRadiation`. All reset each substep. `forceSpinCurv` accumulates both Stern-Gerlach (+μ·∇Bz) and Mathisson-Papapetrou (−L·∇Bgz).

Torque display scalars: `torqueSpinOrbit` (EM + GM spin-orbit power) and `torqueFrameDrag`. Rendered as circular arc arrows around particles — orange for spin-orbit, purple for frame-drag.

### Boris Integrator

Per substep:

1. Store `_f1pnOld` (if 1PN enabled)
2. **Half-kick**: w += F/m · dt/2 (E-like forces only)
3. **Boris rotation**: rotate w in combined Bz + Bgz plane
   - `t = ((q/(2m))·Bz + 2·Bgz) · dt/γ`
   - `s = 2t/(1+t²)`
   - `w' = w + (w + w×t) × s` (preserves |v| exactly)
4. **Half-kick**: w += F/m · dt/2
5. Spin-orbit energy coupling
6. Stern-Gerlach / Mathisson-Papapetrou center-of-mass kicks
7. Frame-dragging torque
8. Radiation reaction (Landau-Lifshitz)
9. **Drift**: derive vel = w/√(1+w²), pos += vel · dt
10. Record signal-delay history
11. **1PN velocity-Verlet correction**: recompute 1PN at new positions, kick w += (F_new − F_old)·dt/(2m)
12. Rebuild quadtree
13. Handle collisions
14. Photon absorption
15. Compute forces for next substep

### Adaptive Substepping

- `dtSafe_accel = √(SOFTENING / a_max)`
- `dtSafe_cyclotron = (2π / ω_c) / 8` where ω_c = max(|q·Bz/m|, 4·|Bgz|)
- `dtSub = dtRemain / min(ceil(dtRemain / dtSafe), budget)`
- Capped at MAX_SUBSTEPS = 16 per frame

### Fixed-Timestep Loop (main.js)

`PHYSICS_DT = 1/120`. Accumulator collects `rawDt × speedScale` per animation frame. Drained in fixed-step chunks, capped at `MAX_SUBSTEPS × PHYSICS_DT × 4`. Photon updates and tidal breakup inside the loop; energy/rendering/DOM outside.

## Force Types

### E-like Forces (radial, position-dependent)

All use Plummer softening: r_eff = √(r² + SOFTENING_SQ), where SOFTENING = 10.

**Gravity**: `F = +m₁m₂ / r²` (attractive)
- PE: `U = −m₁m₂ / r`

**Coulomb**: `F = −q₁q₂ / r²` (like-repels, opposite-attracts)
- PE: `U = +q₁q₂ / r`

**Magnetic dipole** (requires Coulomb toggle): `F = −3μ₁μ₂ / r⁴` (aligned ⊥-to-plane dipoles repel)
- μ = MAG_MOMENT_K · q · ω · r² = q·ω·r²/5
- PE: `U = +μ₁μ₂ / r³`

**GM dipole** (requires Gravity toggle): `F = +3L₁L₂ / r⁴` (co-rotating masses attract; GEM sign flip)
- L = INERTIA_K · m · ω · r² = 2m·ω·r²/5
- PE: `U = −L₁L₂ / r³`

### B-like Forces (velocity-dependent, Boris rotation)

**Lorentz** (requires Coulomb + Magnetic toggles):
- Bz from moving charge: `q_s · (v_s × r̂)_z / r²`
- Bz from spinning dipole: `+μ_source / r³`
- Effect: `F = q(v × B)`, handled implicitly by Boris rotation

**Linear gravitomagnetic** (requires Gravity + GM toggles):
- Bgz from moving mass: `−m_s · (v_s × r̂)_z / r²`
- Bgz from spinning mass: `−2L_source / r³`
- Boris parameter: `t_gm = +2·Bgz·dt/γ` (positive → co-moving attract)
- Display: `F_GM = (4m·vel.y·Bgz, −4m·vel.x·Bgz)`

**Frame-dragging torque**: `τ = FRAME_DRAG_K · m_s · (ω_s − ω_p) / r³` = 0.1 · m_s · (ω_s − ω_p) / r³
- Applied as `angw += τ · dt / I`; drives spin alignment

### 1PN Correction (EIH)

Requires Gravity + Relativity. O(v²/c²) correction to gravity using coordinate velocities.

```
radial  = −v₁² − 2v₂² + 4(v₁·v₂) + 1.5(n̂·v₂)² + 5m₁/r + 4m₂/r
tangent = 4(n̂·v₁) − 3(n̂·v₂)
a_1PN   = (m₂/r²) · [n̂ · radial + (v₁−v₂) · tangent]
```

Velocity-Verlet: stores `_f1pnOld` before drift, recomputes after drift, applies correction kick `(F_new − F_old)·dt/(2m)`. Always pairwise (even in BH mode). Produces perihelion precession ~6πM/a(1−e²) rad/orbit.

1PN PE:
```
U_1PN = −(m₁m₂/r) · [1.5(v₁²+v₂²) − 3.5(v₁·v₂) − 0.5(v₁·n̂)(v₂·n̂) + m₁/r + m₂/r]
```

### Radiation

Requires Relativity.

**Larmor power**: P = 2q²a²/3

**Landau-Lifshitz force** (jerk term only, no Schott damping):
```
F_rad = τ · (F − F_prev) / dt / γ³
τ = 2q²/(3m)    (LARMOR_K = 1/3)
```
Clamped: |F_rad · dt/m| ≤ LL_FORCE_CLAMP · |w| = 0.5 · |w|

**Photon emission**: Energy accumulated in `_radAccum` per particle. Emits when ≥ RADIATION_THRESHOLD (0.01) and pool < MAX_PHOTONS (500). Emission angle sampled from sin²θ dipole pattern with relativistic aberration. Photon travels at c = 1.

**Photon absorption**: Quadtree query at photon position (radius SOFTENING). Self-absorption guard: emitter skipped for 2 substeps. On absorb: `target.w += ph.energy · ph.vel / target.mass`. Bookkeeping: totalRadiated decremented.

### Signal Delay

Requires Relativity + Barnes-Hut off (pairwise only).

Light-cone equation: |x_source(t_ret) − x_obs(now)| = now − t_ret (c = 1).

Newton-Raphson (3 iterations) on per-particle circular history buffers (Float64Array[HISTORY_SIZE=512] each for x, y, vx, vy, time). Linear interpolation at converged t_ret.

Visual: ghost circles at oldest recorded position with dashed line to current.

### Spin-Orbit Coupling

Requires Relativity + relevant force toggle (Magnetic for EM, GM for gravitational).

**Energy transfer**:
- EM: `dE = −μ · (v · ∇Bz) · dt`
- GM: `dE = −L · (v · ∇Bgz) · dt`
- Applied as `angw += dE / (I · ω)`

**Center-of-mass kicks** (spin-curvature forces):
- Stern-Gerlach (EM): `F = +μ · ∇Bz`
- Mathisson-Papapetrou (GM): `F = −L · ∇Bgz` (GEM sign flip)
- Both accumulate into `p.forceSpinCurv`

**Field gradients** (both radial + angular terms):
- `∇Bz`: radial `+3·Bz·r̂/r²`, angular `+q_s·v_s⊥/r³`, dipole `+3μr̂/r⁵`
- `∇Bgz`: radial `+3·Bgz·r̂/r²`, angular `−m_s·v_s⊥/r³`, dipole `−6Lr̂/r⁵`

### Tidal Breakup

Independent toggle. Fragments when any combination exceeds self-gravity:
```
tidal:       TIDAL_STRENGTH · M_other · r_body / r_sep³     (TIDAL_STRENGTH = 2.0)
centrifugal: ω² · r
coulomb:     q² / (4r²)
self-grav:   m / r²
```
Splits into FRAGMENT_COUNT (3) pieces at 120° intervals, radius×1.5 from original. Each gets mass/3, charge/3, tangential velocity from spin. Min mass to fragment: MIN_FRAGMENT_MASS × FRAGMENT_COUNT = 6.

## Sign Conventions (IMPORTANT)

All GEM interactions are **attractive** (gravity has one sign of "charge"):
- GM dipole: `+3L₁L₂/r⁴` (positive = attractive)
- GM Boris parameter: `+2·Bgz` (co-moving masses attract)
- Bgz field: `−m_s·(v_s×r̂)_z/r²` (negative sign from r̂ = source→observer)

Do NOT flip these signs.

## Potential Energy

Computed separately from forces via `Physics.computePE()`. Same BH theta criterion — tree traversal when BH on, exact pairwise when off. Four terms: gravitational (−m₁m₂/r), Coulomb (+q₁q₂/r), magnetic dipole (+μ₁μ₂/r³), GM dipole (−L₁L₂/r³). All Plummer-softened. 1PN PE when enabled.

## Energy & Momentum (`src/energy.js`)

`computeEnergies()` returns: `linearKE`, `spinKE`, `pe`, `fieldEnergy`, `fieldPx/Py`, `px/py`, `orbitalAngMom`, `spinAngMom`.

| Quantity | Relativistic | Classical |
|---|---|---|
| Linear KE | Σ(γ−1)·m | Σ ½m\|v\|² |
| Spin KE | Σ I·(√(1+W²r²)−1) / r² | Σ ½Iω² |
| Momentum | Σ m·w + field + radiated | same |
| Angular mom. | Σ(r×mw) + Σ(I·W) about COM | same |

**Darwin field corrections** (O(v²/c²)):
- EM field energy: `−½ Σ(qᵢqⱼ/r)·[(vᵢ·vⱼ) + (vᵢ·r̂)(vⱼ·r̂)]`
- GM field energy: opposite sign (mass replaces charge)
- Field momentum: analogous terms

Conservation: exact with gravity + Coulomb only, pairwise mode (BH off). Velocity-dependent forces break Newton's 3rd law — missing momentum carried by unmodeled fields.

## Collisions (`src/collisions.js`)

**Pass**: no-op.

**Merge**: conserves mass, charge, momentum, angular momentum. Orbital L about pair COM + spin L → merged angw via I = 2mr²/5.

**Bounce**: relativistic path Lorentz-boosts to COM frame along collision normal, reverses, boosts back. Tangential friction: `J = bounceFriction · (surfaceV₁ − surfaceV₂) · m_eff` where surfaceV = v_tangential + ω·r. Spin updated: `ω_new = ω_old − J/I`. Configurable friction (default 0.4).

## Topology (`src/topology.js`)

When boundary = "loop", topology selector chooses identification:

| Topology | Wrapping | min-image candidates |
|---|---|---|
| Torus (T²) | Both axes normal | 1 |
| Klein (K) | x normal; y-wrap mirrors x, negates w.x/vel.x/angw/angVel | 2 |
| RP² | x-wrap mirrors y; y-wrap mirrors x | 4 |

`minImage(ox, oy, sx, sy, topology, W, H, halfW, halfH, out)`: minimum-image separation. Klein/RP² need absolute positions (glide reflections depend on source coords). Zero-alloc via `out` parameter.

`wrapPosition(p, topology, W, H)`: wraps + applies velocity/spin flips for non-orientable crossings.

**Ghost generation** (`_generateGhosts` in integrator.js): topology-aware. `_addGhost()` accepts `flipVx`/`flipVy` flags.

`sim.topology` string ('torus'/'klein'/'rp2') → `physics._topologyConst` integer (TORUS=0/KLEIN=1/RP2=2).

## Barnes-Hut (`src/quadtree.js`)

QuadTreePool: SoA flat typed arrays, pre-allocated 512 nodes (doubles on overflow). `pool.reset()` + `pool.build()` per substep, zero GC.

Aggregates per node: totalMass, totalCharge, totalMagneticMoment, totalAngularMomentum, totalMomentumX/Y, comX/Y.

BH_THETA = 0.5. Off by default — exact pairwise gives better conservation.

## Toggle Dependencies

```
Gravity (red)
├── Gravitomagnetic (purple)
└── 1PN (rose)                  [also requires Relativity]

Coulomb (blue)
└── Magnetic (cyan)

Relativity (yellow)
├── Signal Delay (yellow)       [also requires BH off]
├── Spin-Orbit (orange)
└── Radiation (yellow)

Tidal (slate)                   [independent]
```

Disabled sub-toggles: `.ctrl-disabled` (opacity 0.4, pointer-events none). Toggle colors match force arrow colors: gravity=red, coulomb=blue, magnetic=cyan, GM=purple, 1PN=rose, spin-curvature=orange, radiation=yellow.

Default on load: all on except Radiation, Tidal, Barnes-Hut.

## UI

### 4-Tab Sidebar

1. **Settings**: particle mass/charge/spin sliders, interaction mode (Place/Shoot/Orbit), force toggles, physics toggles
2. **Engine**: Barnes-Hut, collision mode (Pass/Bounce/Merge), bounce friction slider, boundary mode (Despawn/Loop/Bounce), topology (Torus/Klein/RP²), visual toggles (trails, velocity/force/component vectors, potential field, acceleration scaling), sim speed
3. **Stats**: energy breakdown (total, linear KE, spin KE, PE, field, radiated, drift), conserved quantities (momentum with particle/field/radiated, angular momentum with orbital/spin, drift)
4. **Particle**: selected particle details (mass, charge, spin, speed, γ, |F|), phase space plot canvas

### Topbar

Presets (ghost button) | Pause / Step / Reset | Theme / Sidebar toggle

### Presets

| # | Name | Description |
|---|---|---|
| 1 | Solar System | Star (m=80) + 5 planets in circular orbits |
| 2 | Binary Stars | Two m=50 stars, spin=0.8c, counter-orbiting |
| 3 | Galaxy | Core (m=150) + 200 particles, circular orbits, random charge/spin |
| 4 | Collision | Two groups of 50 particles heading at each other (v=±0.5) |
| 5 | Magnetic Spin | 5×5 grid of charged spinning particles |

### Keyboard Shortcuts

Space (pause), R (reset), `.` (step), P (presets), 1–5 (load preset), V (velocity vectors), F (force vectors), C (force components), T (theme), S (sidebar), Esc (close dialogs), `?` (help overlay).

### Responsive

900px → bottom sheet + 48px toolbar. 600px/440px shared breakpoints from shared-base.css.

## Renderer

Canvas 2D. Dark mode uses additive blending (`globalCompositeOperation: 'lighter'`).

- **Particles**: filled circle at r=cbrt(mass), glow shadow in dark mode
- **Spin rings**: arc at radius+2, length ∝ |ω|, arrow shows CW/CCW, colored by spin sign
- **Trails**: circular Float32Array buffer (MAX_TRAIL_LENGTH=200), 4 opacity groups, wrap-detection for periodic boundaries
- **Force vectors**: scale=5 (÷mass if acceleration scaling on). Total (accent) or per-type components (colored by force type)
- **Torque arcs**: spin-orbit (orange, inner), frame-drag (purple, outer), total (accent). Arc length ∝ |power|
- **Photons**: yellow circles, size = 1.5 + energy×20 (cap 5px), glow in dark mode
- **Signal delay ghosts**: 30% alpha circles at oldest history position, dashed line to current

Particle color: neutral = `_PAL.neutral` (slate). Charged: hue from `chargePos` (201, blue) / `chargeNeg` (7, red), intensity from |q|/20.

## Input (`src/input.js`)

- **Left click** (< 5 world units drag): select particle or spawn at rest
- **Left drag**: spawn with velocity (Shoot: drag×0.02) or at rest (Place/Orbit)
- **Right click**: remove particle within radius+5
- **Orbit mode**: finds particle with max gravitational force on spawn point, spawns perpendicular at v = √(M/r)
- **Hover**: tooltip with m, q, spin, speed
- **Touch**: single=spawn, two-finger=pinch-zoom + pan (300ms guard prevents spawn after pinch)

## Key Patterns

- `Vec2` for all vector math. `vec.set(x,y)` in hot paths.
- `pairForce()`: accumulates into `out` Vec2 parameter, no allocations. Toggle flags via reusable `_toggles` object.
- QuadTreePool: SoA, pre-allocated, `reset()`+`build()` per substep. Zero GC.
- DOM cached in `Simulation.dom` and `Simulation.selDom`. Shared by reference with StatsDisplay.
- `window.sim` for console debugging. `_PALETTE`/`_FONT` frozen by colors.js.
- Dark mode: `globalCompositeOperation: 'lighter'` (additive blending).
- Icon swaps (pause/play, sun/moon): toggle `hidden` attribute, not innerHTML.
- Theme: `data-theme` on `<html>` (not body). Light default for FOUC prevention.

## Gotchas

- Serve from `a9lim.github.io/` parent — `/shared-base.css` and `/shared-tokens.js` use absolute paths
- `#preset-dialog` needs both ID and `class="preset-dialog"` (shared CSS uses class, JS uses ID)
- `photon.js` is imported by `integrator.js` for radiation — not related to input modes
- 1PN velocity-Verlet correction is always pairwise, even when BH is on
- Radiation force uses jerk term only (no Schott damping term `−τF²v/m²`)
- Shoot mode velocity scale is 0.02 (drag pixels × 0.02 = velocity)
