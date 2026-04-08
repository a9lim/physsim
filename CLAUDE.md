# CLAUDE.md

Part of the **a9l.im** portfolio. See root `CLAUDE.md` for the shared design system and shared code policy. Sibling projects: `shoals`, `cyano`, `gerry`, `scripture`.

## Rules

- Always prefer shared modules over project-specific reimplementations. Check `shared-*.js` files before adding utility code.
- Never use the phrase "retarded potential(s)" in code, comments, or user-facing text. Use "signal delay" or "finite-speed force propagation" instead.

## Running Locally

```bash
cd path/to/a9lim.github.io && python -m http.server
```

Serve from root — shared files load via absolute paths. No build step, test framework, or linter.

## Overview

Interactive particle physics simulator. Boris integrator, BH tree acceleration, Higgs/Axion scalar fields, WebGPU compute+render backend with CPU fallback, 19 presets across gravity/EM/exotic/cosmological scenarios. Zero dependencies, vanilla ES6 modules.

## Architecture

**`main.js`** (~940 lines): Simulation class, fixed-timestep loop (PHYSICS_DT = 1/128), backend selection (CPU/GPU), `window.sim` for debugging.

**Two interchangeable backends** via `selectBackend()`:
- **CPU**: `CPUPhysics` (wraps integrator.js) + `CanvasRenderer` (wraps renderer.js)
- **GPU**: `GPUPhysics` (WebGPU compute) + `GPURenderer` (WebGPU instanced rendering, dual light/dark pipelines)

Falls back to CPU on WebGPU unavailability or device loss. Force CPU via `?cpu=1`.

**Key modules**: `integrator.js` (Boris substep loop, all physics), `forces.js` (pairForce, BH tree walk, 1PN), `scalar-field.js` (PQS grid base), `higgs-field.js` / `axion-field.js` (field subclasses), `quadtree.js` (SoA flat typed arrays, pool-based), `ui.js` (declarative toggle dependency graph), `presets.js` (19 scenarios).

## Physics

### Units & State

c = G = ħ = 1. All velocities are fractions of c. Both linear (`p.w` = γv) and rotational (`p.angw`) state use proper velocity. When relativity off: `vel = w` identity.

### Boris Integrator (per substep)

Half-kick → Boris rotation (Bz + Bgz + extBz) → half-kick → spin-orbit/radiation/pion emission → drift → 1PN velocity-Verlet correction → scalar field evolution (Störmer-Verlet KDK) → quadtree rebuild + collisions → external/scalar field forces

Adaptive substepping: `dtSafe = min(√(softening/a_max), (2π/ω_c)/8)`. Max 32 substeps.

### Sign Conventions

All GEM interactions are **attractive** (gravity has one sign of "charge"):
- GM dipole: `+3L₁L₂/r⁴` (positive = attractive)
- GM Boris parameter: `+2Bgz` (co-moving masses attract)
- Angular velocity (y-down canvas): `rx·vy - ry·vx` gives positive for clockwise on screen

### Toggle Dependencies

```
Forces:                        Physics:
  Gravity                        Relativity          [signal delay auto-activates]
    -> Gravitomagnetic             -> 1PN             [requires Magnetic, GM, or Yukawa]
    (field gravity auto-on)        -> Black Hole      [+Gravity, locks collision to Merge]
  Coulomb                        Spin-Orbit           [requires Magnetic or GM]
    -> Magnetic                  Radiation             [requires Gravity, Coulomb, or Yukawa]
  Yukawa               [independent]  Boson Interaction [requires BH + (Gravity OR Coulomb)]
  Axion                [requires Coulomb or Yukawa]
  Higgs                [independent]
Disintegration                   [requires Gravity, locks collision to Merge] **HIDDEN**
Barnes-Hut                       [independent]
Expansion                        [independent, in Engine tab]
```

Declarative `DEPS` array in `ui.js`, topological evaluation via `updateAllDeps()`.

### Schwinger Discharge

Vacuum e⁺e⁻ pair production at BH horizons. Schwinger formula with Kerr-Newman spin correction: `Γ = (e²Q²)/(π²Σ) × exp(-πE_cr Σ/|Q|)`, where `Σ = r₊² + a²` (KN horizon area factor), `e = BOSON_CHARGE`, `m_e = ELECTRON_MASS`, `E_cr = m_e²/e`. Rate coefficient `e²/π²` bakes in area×prefactor (no separate 4πr₊² multiplication). Escaping lepton KE derived from horizon electrostatic potential: `KE = eΦ_H - m_e`, where `Φ_H = |Q|r₊/(r₊² + a²)` for KN. BH mass loss is `ELECTRON_MASS` per event (lepton KE not subtracted — prevents runaway drain). Threshold at `0.5·E_cr` (sub-critical but exponentially suppressed). Requires Black Hole + Coulomb + Radiation (shares evaporation cleanup with Hawking). Accumulates rate per substep into `_schwingerAccum` (CPU) / `RadiationState.schwingerAccum` (GPU); emits when accumulator reaches 1. Same-sign lepton escapes, opposite-sign falls back into BH — net: BH loses `BOSON_CHARGE` of charge and `ELECTRON_MASS` of mass per event. CPU in `integrator.js` (after Hawking radiation), GPU in `radiation.wgsl` (`schwingerDischarge` entry point). GPU leptons use the shared pion pool with `kind=1u`.

### Quantized Boson Charge

Pion and lepton charges are quantized in units of `BOSON_CHARGE` (config.js, default 0.1). Pion charge is `f32` in both CPU and GPU (WGSL Pion struct). Neutral pions have charge exactly `0.0`; charged pions/leptons carry `±BOSON_CHARGE`. Particle charge is also quantized: `addParticle()` (both CPU `main.js` and GPU `gpu-physics.js`) rounds charge to the nearest `BOSON_CHARGE` multiple. Charge conservation is maintained — emission subtracts `±BOSON_CHARGE` from emitter, absorption/decay adds it back. Disintegration (Roche transfer, tidal fragments) quantizes transferred charge. Annihilation checks use `abs(charge) < EPSILON` for neutral and sign comparison for opposite-charge (not integer equality).

### Higgs Mass Modulation

When Higgs enabled, Yukawa range parameter μ_eff = `yukawaMu · √(higgsMod_i · higgsMod_j)` where `higgsMod = max(|φ(x)|, HIGGS_MASS_FLOOR)` cached per particle. Geometric mean per pair. GPU: `higgsMod` in `axYukMod.z`.

### Scalar Field Base

PQS (cubic B-spline) grid: 64×64 (CPU), 128×128 (GPU). 4×4 stencil. C² interpolation and gradients. Field arrays: `field`/`fieldDot` (not `phi`/`phiDot`). Clamp: SCALAR_FIELD_MAX = 2.

Self-gravity via FFT convolution with Green's function. `computeSelfGravity(domainW, domainH, softeningSq, bcMode, topoConst)` — callers pass boundary mode directly, not a boolean. Called twice per KDK for O(dt²) accuracy.

## GPU

### Capacity Limits

| Resource | CPU | GPU |
|----------|-----|-----|
| Particles | 128 | 512 |
| Photons | 1024 | 4096 |
| Pions | 256 | 1024 |
| Leptons | 256 | Shares pion pool (1280) |

### Shader Organization

All shaders prepended with `wgslConstants + shared-structs.wgsl + shared-topology.wgsl + shared-rng.wgsl`. Tree-walk shaders add `shared-tree-nodes.wgsl`. `fetchShader()` in `gpu-pipelines.js` is single source of truth.

`SHADER_VERSION` in gpu-pipelines.js must be bumped after shader edits to invalidate browser cache.

### GPU Tree Build

4 dispatches (computeBounds, initRoot, insertParticles, computeAggregates). Lock-free CAS insertion. Visitor-flag bottom-up aggregation. Tree resets use `encoder.copyBufferToBuffer` (not `queue.writeBuffer`) because the tree may be built twice per substep; queue-level operations would execute before the encoder starts.

### GPU ↔ CPU Sync

- `addParticle()` must initialize ALL per-particle buffers. `axYukMod` defaults to `(1.0, 1.0, 1.0, 0.0)` not `(0, 0, 0, 0)`
- `queue.writeBuffer()` executes at queue time (before encoder starts), NOT inline with compute passes. Use `encoder.copyBufferToBuffer` for resets between dispatches within the same command buffer
- `_phase5Ready` flag guards field dispatches until async pipeline creation completes
- Async readback methods use try/catch/finally to clear pending flags on device loss

### Disintegration (Hidden)

UI toggle hidden via `style="display:none"` — still activatable via presets (Roche limit preset). Two mechanisms: tidal fragmentation (parent → SPAWN_COUNT children) and Roche lobe overflow (Eggleton 1983 mass transfer). Known bugs:

- **Charge cascade**: fragments inherit parent's charge/SPAWN_COUNT, but if Coulomb self-repulsion caused the breakup, fragments still exceed the threshold and cascade into dust. Needs a cooldown, charge redistribution fix, or minimum fragment mass floor.
- **GPU Roche**: source particle charge is not subtracted (only mass via `patchMass`). Minor charge non-conservation.
- **GPU readback latency**: disintegration events have 1-frame latency (same as merge events). Parent state in `DisintEvent` is snapshot from detection frame; by readback time, particle has evolved further.
- `DisintEvent` struct is 48 bytes (12 fields). GPU uses `atomicAdd` on event counter, capped at `MAX_DISINT_EVENTS` (64).

## Gotchas

### Will Cause Bugs

- `_peAccum` PE is accumulated inline during `pairForce()` — `potential.js` is a fallback only for preset-load recomputation
- `forceRadiation` cleared for all particles before substep loop
- History recording counts `update()` calls, not substeps (strided at HISTORY_STRIDE=64)
- `.mode-toggles` sets `display: grid` overriding `hidden` — use `style.display`
- External field trig cached once per frame via `_cacheExternalFields()`
- Lazy field init: Higgs/Axion fields are `null` until first toggle-on
- Bounce (Hertz) is always quadtree-accelerated when BH on, O(n²) when off — do not early-return when `root < 0`
- `magMoment`/`angMomentum` cached per particle at start of `computeAllForces()` using `bodyRadiusSq` (intrinsic body radius, not horizon radius in BH mode)
- Dead particles in GPU tree use `deathMass`/`deathAngVel` from `ParticleAux` for leaf data; CPU dead-particle path remains pairwise

### WGSL

- Explicit parentheses required when mixing `*` with `^` (XOR)
- Multiple entry points sharing a module need `read_write` access on shared bindings
- WebGPU disallows binding the same buffer twice in a dispatch
- Staging buffers must not be copied to while mapped from previous `mapAsync`
- JS uniform write order must exactly match WGSL struct member order
- `deathTime` sentinel: `FLT_MAX` (3.4028235e38), not `Infinity`
- Collision merge: `mass <= EPSILON` guards (not `== 0`) for race conditions
- Render: premultiplied alpha (`color.rgb * alpha`) with `srcFactor: 'one'`
- NaN barriers before writing to global memory (not after)

### Topology

- `minImage()` uses `out` parameter for zero-alloc
- RP² `wrapPosition()` uses iterative wrapping (max 2 passes) for simultaneous x+y out-of-bounds
- Periodic boundary interpolation in signal delay uses `minImage()` to wrap displacements

### Semantic

- 1PN does NOT obey Newton's 3rd law — velocity-Verlet corrected
- `compute1PN()` zeroes `force1PN` before accumulating
- Self-absorption permanently blocked by `emitterId` for both photons and pions
- Leptons on GPU share pion buffer — distinguished by `Pion.kind` field (0=pion, 1=lepton)
- GPU pion decay probability scaled by `1-(1-p)^N` to match CPU's per-tick rate
- World coordinates: `sim.domainW/H` (viewport / WORLD_SCALE), not pixels
- `_PALETTE`/`_FONT` frozen by colors.js
