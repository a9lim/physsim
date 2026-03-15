# GPU Signal Delay Parity — Design Spec

**Date:** 2026-03-15
**Status:** Approved
**Scope:** Bring GPU signal delay implementation to full parity with CPU, fix aberration gaps in both backends

## Problem Statement

The GPU physics backend records signal delay history (`history.wgsl`) and has a working light-cone solver (`getDelayedStateGPU`), but force computation shaders (`pair-force.wgsl`, `forces-tree.wgsl`, `onePN.wgsl`) do not use signal-delayed positions. They compute forces from current-time positions with only the aberration factor applied — physically incorrect since aberration is a correction on top of retarded positions, not a standalone effect.

Additionally:
- Dead/retired particles exert zero force in GPU mode (CPU fades them via signal delay history)
- GPU heatmap doesn't iterate dead particles
- No `creationTime` guard in GPU extrapolation (CPU rejects extrapolation past particle creation)
- CPU BH tree walk: ghosts skip signal delay, aggregates skip aberration — both should be fixed

## Design

### 1. Interleaved History Buffer

**Current:** 7 separate buffers — `histPosX`, `histPosY`, `histVelWX`, `histVelWY`, `histAngW`, `histTime`, `histMeta`.

**New:** 2 buffers total:

1. **`histData`** (f32): Interleaved `[posX, posY, velX, velY, angW, time]` per sample. Stride = 6. Indexed as `histData[particleIdx * HISTORY_LEN * 6 + sampleIdx * 6 + field]`. Size: `maxParticles × HISTORY_LEN × 6 × 4` bytes.

2. **`histMeta`** (u32): Expanded from 2 to 4 u32 per particle: `[writeIdx, count, creationTimeBits, _pad]`. `creationTimeBits = bitcast<u32>(simTime)` at spawn. Size: `maxParticles × 16` bytes.

**Rationale:** Reduces 7 buffer bindings to 2 (fits within `maxStorageBuffersPerShaderStage` limit of 10). Per-sample reads become contiguous 24-byte fetches (cache-line friendly). `creationTime` in histMeta lets the solver reject extrapolation past creation.

**Files changed:**
- `gpu-buffers.js` — replace 7 allocations with 2, expand histMeta stride
- `history.wgsl` — write interleaved format, initialize creationTime on FLAG_REBORN
- `gpu-physics.js` — `addParticle()`, `reset()`, `deserialize()` init histMeta with creationTime
- `gpu-pipelines.js` — update bind group layouts for history

### 2. Shared Signal Delay Lookup

**New file: `src/gpu/shaders/signal-delay-common.wgsl`**

Contains `getDelayedStateGPU()` rewritten for interleaved buffer format. Includes:
- NR light-cone solver (phase 1) + exact quadratic (phase 2) + backward extrapolation (phase 3)
- `minImageDisp()` for topology-aware periodic boundaries (Torus/Klein/RP²)
- `creationTime` rejection: reads `histMeta[srcIdx * 4 + 2]`, rejects extrapolation past creation
- Dead particle guard: `isDead=true` skips extrapolation (phase 3)

**Integration pattern:** Prepended to consuming shaders (like `common.wgsl`). Consumers declare `histData` and `histMeta` bindings at their chosen group/binding. The shared file references them by name.

**`heatmap.wgsl`:** Replaces its inline `getRetardedPosition()` with the shared function, gaining `creationTime` check and dead-particle guard it currently lacks.

### 3. pair-force.wgsl Signal Delay

**Bind group restructuring:**
- Group 0: uniforms (1 uniform)
- Group 1: particles + derived + axYukMod + particleAux (4 storage)
- Group 2: allForces + radState + maxAccel (3 storage)
- Group 3: histData + histMeta (2 storage)

Total: 9 storage + 1 uniform. Within limits.

**Tile approach change:** When signal delay is active, the shared-memory tile caches only observer-independent properties: `mass, charge, axMod, yukMod, radiusSq, srcIdx`. Each thread calls `getDelayedStateGPU(srcIdx, myPosX, myPosY, ...)` to get retarded position/velocity/angw per pair.

When signal delay is off (relativity disabled), tile caches position/velocity/angw as today — no performance regression.

**Retarded dipole recomputation (matching CPU `forces.js:105-109`):**
```
bodyRadiusSq = pow(mass, 2.0/3.0)  // NOT derived.radiusSq (BH horizon in BH mode)
sAngVel = retAngw / sqrt(1 + retAngw² * bodyRadiusSq)
sMagMoment = MAG_MOMENT_K * charge * sAngVel * bodyRadiusSq
sAngMomentum = INERTIA_K * mass * sAngVel * bodyRadiusSq
```

**Dead particle loop:** After tile loop, scan `[0, aliveCount)` for `FLAG_RETIRED & !FLAG_ALIVE`. Call `getDelayedStateGPU(ri, ..., isDead=true)`. Use `deathMass`/`deathAngVel` from `ParticleAux`. Apply aberration.

**Signal delay failure:** When `getDelayedStateGPU` returns `valid=false`, skip the source entirely (matching CPU `continue` at `forces.js:102`).

### 4. forces-tree.wgsl Signal Delay

**Bind groups:** Add group 3: histData + histMeta. Total: 10 storage + 1 uniform. At limit.

**Leaf nodes, non-ghost:** `getDelayedStateGPU(srcIdx, ...)`, recompute dipoles, apply aberration.

**Leaf nodes, ghost:** `getDelayedStateGPU(originalIdx, ...)` to get original's retarded position, then add periodic shift `(ghostPos - originalPos)`. Recompute dipoles. Apply aberration. The periodic shift is time-invariant (domain geometry doesn't change).

**Aggregate nodes:** Current-time CoM position (no signal delay), but apply aberration using `avgVx/avgVy` from `totalMomentumX/Y / totalMass`.

**Dead particle loop:** Replace stub (line 633: `// Phase 4 will add signal delay lookup here`) with proper `getDelayedStateGPU(ri, ..., isDead=true)`. Use `deathMass`/`deathAngVel` from `ParticleAux`. Apply aberration.

### 5. onePN.wgsl Signal Delay

**Bind groups:** Add group 3: histData + histMeta (2 storage). Total: 7 storage + 1 uniform. Well within limits.

**Changes:**
- Prepend `signal-delay-common.wgsl`
- Each thread calls `getDelayedStateGPU(srcIdx, ...)` per source
- If `valid=false`, skip source
- Use retarded position/velocity for all 1PN terms (EIH, Darwin, Bazanski, Scalar Breit)
- **No aberration** — 1PN is already O(v²/c²); aberration would be O(v³/c³)
- Dead particles excluded from 1PN (matching CPU behavior)

### 6. Heatmap Dead Particles

After the alive-particle loop, add a second loop scanning `[0, particleCount)` for `FLAG_RETIRED & !FLAG_ALIVE`. Call `getDelayedStateGPU(i, ..., isDead=true)`. Use `deathMass` for gravity/Yukawa, `charge` for electric potential. The heatmap already has history bindings (group 2) — just switch to interleaved format.

### 7. CPU-Side Fixes (forces.js)

**BH aggregate aberration:** Line 668 calls `pairForce()` without `signalDelayed`. Fix: pass `signalDelayed=true` when relativity is on. The aggregate's `avgVx/avgVy` provides the velocity for the aberration factor.

**Ghost signal delay:** Line 644 skips signal delay for ghosts (`!other.isGhost`). Fix: when signal delay is on, call `getDelayedState(real, particle, ...)` on the ghost's original, then add the periodic shift `(other.pos.x - real.pos.x, other.pos.y - real.pos.y)` to the retarded position. This gives the retarded ghost position. Apply aberration.

## Aberration Coverage Matrix

| Path | Delayed Positions | Aberration | Notes |
|------|-------------------|------------|-------|
| CPU pairwise | Yes | Yes | Already correct |
| CPU BH leaf (real) | Yes | Yes | Already correct |
| CPU BH leaf (ghost) | No → Yes | No → Yes | Fix: retarded original + shift |
| CPU BH aggregate | No (CoM) | No → Yes | Fix: add aberration with avg vel |
| CPU dead particles | Yes | Yes | Already correct |
| CPU 1PN | Yes | No | Correct — O(v²) already |
| GPU pairwise | No → Yes | Wrong → Yes | Fix: retarded positions + aberration |
| GPU BH leaf (real) | No → Yes | No → Yes | Fix both |
| GPU BH leaf (ghost) | No → Yes | No → Yes | Fix: retarded original + shift |
| GPU BH aggregate | No (CoM) | No → Yes | Fix: add aberration with avg vel |
| GPU dead (tree) | Stub → Yes | No → Yes | Fix: replace stub |
| GPU dead (pairwise) | Missing → Yes | Missing → Yes | Fix: add dead loop |
| GPU 1PN | No → Yes | No | Correct — no aberration needed |
| GPU heatmap alive | Yes | N/A | Already works (potential, not force) |
| GPU heatmap dead | Missing → Yes | N/A | Fix: add dead loop |

## Files Changed

### New files
- `src/gpu/shaders/signal-delay-common.wgsl` — shared getDelayedStateGPU for interleaved format

### GPU shader changes
- `src/gpu/shaders/history.wgsl` — write interleaved format, init creationTime
- `src/gpu/shaders/pair-force.wgsl` — signal delay lookup, dead particles, bind group restructure
- `src/gpu/shaders/forces-tree.wgsl` — signal delay at leaves/ghosts, aberration on aggregates, fix dead stub
- `src/gpu/shaders/onePN.wgsl` — signal delay lookup
- `src/gpu/shaders/heatmap.wgsl` — use shared function, add dead particle loop

### GPU infrastructure changes
- `src/gpu/gpu-buffers.js` — interleaved histData + expanded histMeta
- `src/gpu/gpu-pipelines.js` — bind group layouts for all affected shaders
- `src/gpu/gpu-physics.js` — buffer init, creationTime in addParticle/reset/deserialize, bind group creation, dispatch updates
- `src/gpu/gpu-constants.js` — HIST_STRIDE constant (6) for WGSL

### CPU changes
- `src/forces.js` — ghost signal delay + aberration, aggregate aberration

## Performance Considerations

- **Pairwise O(N²):** Each pair now requires ~20-30 global memory reads for NR convergence (vs 0 today). This is the most expensive change. Mitigated by: interleaved layout (cache-friendly), early exit on convergence, tile still caches observer-independent data.
- **BH tree walk:** Signal delay only at leaves (typically O(N log N) leaf visits). Aggregates unchanged except for cheap aberration factor.
- **1PN:** Already O(N²); signal delay adds proportional cost.
- **When relativity is off:** Zero overhead — all signal delay paths gated by toggle check.
