# Comprehensive Performance Audit — Implementation Spec

**Date**: 2026-03-19
**Scope**: 52 findings across 6 audit domains (CPU physics, scalar fields, renderer/UI, GPU backend, GPU shaders, dead code)
**Note**: Correctness bugs from the companion audit (`2026-03-19-comprehensive-audit.md`) are handled separately.
**Strategy**: File-clustered phases — each file edited once, phases ordered by dependency

---

## Phase 1 — Dead Code & Cleanup

**Files**: `vec2.js`, `massless-boson.js`, `pion.js`, `relativity.js`, `renderer.js`, `ui.js`, `heatmap.js`, `potential.js`, `collisions.js`, `cpu-physics.js`, `canvas-renderer.js`, `styles.css`, `forces.js`, `boson-utils.js`

| ID | Fix | File | Change |
|----|-----|------|--------|
| D1 | Delete `Vec2.scale()`, `mag()`, `normalize()` | `vec2.js:17-38` | Remove 3 methods. **Keep** `set`, `clone`, `add`, `sub`, `magSq`, `dist` (all used externally). |
| D2 | Delete `clearPool()` on MasslessBoson and Pion | `massless-boson.js:57`, `pion.js:66` | Remove 2 methods |
| D3 | Remove unused `PI` import | `renderer.js:1` | Edit import statement |
| D4 | Remove unused `EPSILON` import | `relativity.js:4` | Edit import statement |
| D5 | Remove unused `DEFAULT_SPEED_INDEX` import | `ui.js:4` | Edit import statement |
| D6 | Un-export `HEATMAP_MODES` (only used internally) | `heatmap.js:112` | Remove `export` keyword |
| D7 | Un-export `resolveMerge`, `treePE`, `pairPE` (only used internally) | `collisions.js:112`, `potential.js:66,142` | Remove `export` keywords |
| D8 | Delete unused `_engine` getters | `cpu-physics.js:24`, `canvas-renderer.js:19` | Remove getter methods |
| D9 | Delete `.tog-tidal` CSS rule | `styles.css:112` | Remove CSS rule |
| D10 | Remove unused params: `bounceFriction` + `relativityEnabled` from `handleCollisions`; `relativityEnabled` + `periodic` from `resolveMerge` (`periodic` IS used in `handleCollisions`, keep it there) | `collisions.js:24,112` | Remove params, update callers |
| D11 | Mark `potential.js` as fallback-only (document, don't delete yet — used in preset load) | `potential.js` | Add comment |
| D12 | Deduplicate `_walkBosonTree` / `_walkBosonTreeCharge` — keep as 2 functions but extract shared tree walk skeleton into a helper. Keep `treeDeflectBoson` separate (operates on particle tree, different output target). | `forces.js:710-887`, `boson-utils.js:20-73` | Extract `_walkBosonTreeCore(pool, root, pos, softeningSq, topology, extractFn)` shared skeleton; `_walkBosonTree` and `_walkBosonTreeCharge` become thin wrappers. `treeDeflectBoson` stays in `boson-utils.js` unchanged. |

**Risk**: Minimal — deletions and un-exports. D10 requires updating callers in `integrator.js`. D12 is a refactor but keeps inner-loop branches out of the hot path.

---

## Phase 2 — `quadtree.js`

**Files**: `quadtree.js`

| ID | Fix | Lines | Change |
|----|-----|-------|--------|
| C1 | Unroll `_subdivide()` — eliminate 3 array allocations | 117-119 | Replace `ids/xs/ys` arrays with 4 inline blocks initializing each child |
| C10 | Split polymorphic insert work stack into two typed stacks | 151-181 | Module-level `_workNodes: Int32Array(64)` + `_workParts: Array(64)` + `workTop` counter |
| H10 | Remove dead `rSq` read in `calculateMassDistribution` | 224 | Delete line |
| M5 | Replace `for...of` with indexed loop in `build()` | 360 | `for (let i = 0; i < particles.length; i++)` |

**Risk**: Low. `_subdivide` unrolling is mechanical. Insert stack split changes iteration order but not behavior.

---

## Phase 3 — `forces.js` + `boson-utils.js`

**Files**: `forces.js`, `boson-utils.js`

| ID | Fix | Lines | Change |
|----|-----|-------|--------|
| C4 | Replace `Math.cbrt(p.mass)` with `Math.sqrt(p.bodyRadiusSq)` in tidal locking | `forces.js:437` | `const bodyR = Math.sqrt(p.bodyRadiusSq); const ri5 = p.bodyRadiusSq * p.bodyRadiusSq * bodyR;` |
| C5 | Guard `Math.sqrt(yukMod)` behind `toggles.axionEnabled` | `forces.js:391`, `potential.js:155-156` | `const yukModPair = toggles.axionEnabled ? Math.sqrt(...) : 1;` same for `higgsModPair` |
| D12 | (continued from Phase 1) Extract shared tree walk skeleton for boson gravity/Coulomb | `forces.js:710-887` | `_walkBosonTreeCore` helper; `_walkBosonTree`/`_walkBosonTreeCharge` become thin wrappers |

**Risk**: Low-medium. C5 changes force computation guards — must verify axion/Yukawa toggle dependency ensures `yukMod === 1` when axion is off.

---

## Phase 4 — `integrator.js`

**Files**: `integrator.js`

| ID | Fix | Lines | Change |
|----|-----|-------|--------|
| C6 | Fuse spin-orbit loop (679-707) and torque loop (710-725) | 679-725 | Single loop with combined condition; one `angwToAngVel()` call at end |
| H2 | Defer `pion._syncVel()` — call once after both boson gravity + pion Coulomb | 822, 905 | Remove per-walk `_syncVel()`, add batch sync after both operations |
| H8 | Fuse adaptive dt estimation with half-kick loop | 597-676 | Compute `maxAccel` during half-kick (one substep behind — acceptable approximation) |

**Risk**: Medium. H8 changes substep sizing to use previous-substep's max acceleration. This is a valid approximation (forces are continuous) but subtly changes adaptive behavior. C6 is straightforward loop fusion.

---

## Phase 5 — `fft.js` + `scalar-field.js`

**Files**: `fft.js`, `scalar-field.js`

| ID | Fix | Lines | Change |
|----|-----|-------|--------|
| C7 | Pre-allocate `rowRe/rowIm/colRe/colIm` at module level | `fft.js:69-70,87-88` | Module-level `let _rowRe, _rowIm, _colRe, _colIm`; lazy-resize to max N |
| C14 | Skip `_sgPhiFull` copy — compute SG gradients directly from `_fftRe` | `scalar-field.js:631-635` | Point `_computeSelfGravGradients` at `_fftRe`, remove copy loop |
| C15 | Fuse `_computeEnergyDensity` + `_addPotentialEnergy` into single grid pass | `scalar-field.js:492-498` | Add `potentialFn` parameter to `_computeEnergyDensity`; subclasses pass their V(phi) |
| ~~C17~~ | ~~Fuse Laplacian + viscosity~~ | — | **Dropped**: C16 (Phase 6) inlines Laplacian into kick loops, which also subsumes viscosity. Fusing them as a standalone function would be immediately undone by C16. |

**Risk**: Medium. C14 requires verifying `_fftRe` isn't overwritten between IFFT and gradient computation.

---

## Phase 6 — `higgs-field.js` + `axion-field.js`

**Files**: `higgs-field.js`, `axion-field.js`

| ID | Fix | Lines | Change |
|----|-----|-------|--------|
| C8 | Fuse source + thermal deposition into single particle loop | `higgs-field.js:60-68` | Compute PQS coords once, deposit into both `_source` and `_thermal` arrays |
| ~~C9~~ | ~~Cache PQS interpolation in modulateMasses~~ | — | **Dropped**: Particles move between `modulateMasses` (pre-drift) and `applyForces` (post-drift), making cached gradients stale. Previously flagged as infeasible. |
| C16 | Inline Laplacian AND viscosity into kick loops | both files | Compute `(f[i-1]+f[i+1]-2f[i])*invCWSq + (f[i-G]+f[i+G]-2f[i])*invCHSq` inline for Laplacian; also inline `nu*(fDot stencil)` for viscosity. Interior fast path + border fallback. Eliminate `_laplacian` buffer and `_computeViscosity()` calls. Subsumes dropped C17. |
| C18 | Optimize polynomial evaluation | `higgs-field.js:99-133` | `const phiSq = phi*phi; const phiCu = phiSq*phi;` reuse in both kick branches |
| A2 | Early return in `interpolateAxMod` | `axion-field.js:209` | `if (!coulombEnabled && !yukawaEnabled) { for(p) p.axMod=1, p.yukMod=1; return; }` |

**Risk**: Medium-high. C16 is the most complex change — inlining both Laplacian and viscosity requires careful handling of border cells via `_nb()` and maintaining identical numerical results for both stencils.

---

## Phase 7 — `renderer.js` + `heatmap.js`

**Files**: `renderer.js`, `heatmap.js`

| ID | Fix | Lines | Change |
|----|-----|-------|--------|
| C11 | Fuse spin ring arc + arrowhead into single pass per sign | `renderer.js:321-361` | One loop per sign that draws arc then arrowhead inline, single `stroke()`+`fill()` |
| C12 | Fuse torque arc + arrowhead into single pass per type | `renderer.js:598-644` | Merge `_drawTorqueArc` arc and arrowhead loops; compute `getValue(p)` once |
| C19 | Pass length to `_batchArrowsDraw` instead of `subarray()` | `renderer.js:483-484,520` | Add `count` parameter; loop to `count` instead of `lines.length` |
| C20 | Set `imageSmoothingEnabled` once in constructor | `heatmap.js:305-314` | Move to constructor/init; remove per-frame set |
| C21 | Reduce trail alpha groups from 4 to 2 | `renderer.js:258-282` | 2 groups instead of 4; halves `ctx.stroke()` calls |
| C22 | Remove redundant `threshold` check | `renderer.js:503` | Delete `if (mag < threshold) continue;` — `minLen` check is strictly tighter |

**Risk**: Low. C21 changes visual appearance slightly (coarser alpha gradient). All others are mechanical.

---

## Phase 8 — `main.js` + `ui.js` + `stats-display.js`

**Files**: `main.js`, `ui.js`, `stats-display.js`

| ID | Fix | Lines | Change |
|----|-----|-------|--------|
| C2 | Pre-allocate `renderOpts`, `enabledForces`, `heatmapOpts` on Simulation | `main.js:731-746` | Create once in constructor; mutate fields in `_render()` |
| C3 | Pre-allocate `_gpuToggleProxy` instead of `Object.create()` | `ui.js:260,272` | Module-level object; copy 2 properties before each `setToggles()` |
| C13 | Gate sidebar plots behind panel visibility + active tab | `main.js:823-836` | Check `panel.classList.contains('open') && activeTab === 'particle'` before drawing |

**Risk**: Minimal. C3 requires the proxy object to have all physics properties — use `Object.assign(_proxy, sim.physics)` then override the 2 GPU-specific properties.

---

## Phase 9 — `gpu-buffers.js` + `gpu-constants.js`

**Files**: `gpu-buffers.js`, `gpu-constants.js`, `shared-structs.wgsl`

| ID | Fix | Lines | Change |
|----|-----|-------|--------|
| G6 | Split uniform buffer: per-frame (full 256B) + per-substep (20B) | `gpu-buffers.js:558` | New `writeSubstepUniforms(dt, simTime, alive, count, frame)` writes only 20 bytes. `writeUniforms()` becomes `writeFrameUniforms()` called once per frame. WGSL: add `SubstepUniforms` struct at new binding. |
| G14 | Cache external field cos/sin in `setToggles()` | `gpu-buffers.js:558-561` | Precompute in `setToggles()`, store as `_cachedExtGravDirX/Y` etc., pass to `writeFrameUniforms()` |
| G17 | Add precomputed cell dimensions to `FieldUniforms` | `gpu-constants.js` | Add `invCellWSq`, `invCellHSq`, `cellW`, `cellH` to struct; set in field uniform write |

**Risk**: Medium. G6 changes the WGSL binding layout — every shader that reads uniforms must bind the new substep buffer. This is a structural change that must be coordinated with shader updates in Phase 12.

**Dependency**: Phase 12 (shader updates) depends on this phase for the new uniform struct.

---

## Phase 10 — `gpu-physics.js`

**Files**: `gpu-physics.js`

| ID | Fix | Lines | Change |
|----|-----|-------|--------|
| G5 | Batch deterministic passes within each substep into single encoder | 3200 | **Constraint**: Adaptive substepping reads `_maxAccel` from the GPU, preventing full cross-substep batching. Instead: (a) batch all compute passes *within* each substep into one encoder (forces, integrate, collisions, field evolve — currently already done), and (b) reduce per-substep overhead by using `encoder.copyBufferToBuffer` from pre-staged uniform rings instead of `queue.writeBuffer` for the substep-varying fields (dt, simTime, counts). This eliminates the queue-time writeBuffer penalty while preserving adaptive substepping. |
| G12 | Size boson dispatches to actual count | 1207, 1307 | Track `_photonCount`/`_pionCount` (1-frame latency readback or CPU tracking of emit/absorb). Dispatch `ceil(count/64)` workgroups. Skip entirely when count === 0. |
| G13 | Pre-compute FFT butterfly params | 2306-2312 | At init: build `Float32Array(4 * 2 * log2(GRID))` with all (N, stage, direction) triples. Single `writeBuffer` at init. Index by `stageIdx` uniform or use buffer offset. |
| G15 | Use `clearBuffer` for zero-filled field buffers | 1869-1888 | Replace 9 `writeBuffer(zeros)` calls with `encoder.clearBuffer(buf)`. Only `writeBuffer` the Higgs vacuum (1.0 values). |
| G16 | Batch `deserialize` uploads | 3580-3635 | Build full packed `Float32Array` per buffer type CPU-side, issue 7 `writeBuffer` calls total. |

**Risk**: Medium. G5 is scoped to intra-substep batching + uniform staging ring (not cross-substep batching, which is infeasible due to adaptive substepping's GPU readback dependency). The uniform staging ring must pre-write all possible substep values; `copyBufferToBuffer` selects the correct slot per substep.

**Dependency**: Depends on Phase 9 (G6 uniform split).

---

## Phase 11 — `gpu-renderer.js` + `gpu-pipelines.js`

**Files**: `gpu-renderer.js`, `gpu-pipelines.js`

| ID | Fix | Lines | Change |
|----|-----|-------|--------|
| G4 | Batch render passes into 2-3 submits | `gpu-renderer.js:548-843` | Group: (1) trails+particles+overlays, (2) bosons+rings, (3) arrows+torques. 3 submits instead of 20+. |
| H5 | Dynamic uniform buffer for arrows/torques | `gpu-renderer.js:873-964` | Allocate uniform buffer with slots for all 15 draw types (11 forces + velocity + 3 torques). One render pass with per-draw dynamic offsets. |
| M5 | Cache concatenated shared prefix | `gpu-pipelines.js:12-19` | Key by `wgslConstants` string, cache the full `prefix + structs + topo + rng` concatenation |
| M6 | Share shader module for light/dark pipeline pairs | `gpu-renderer.js:261-264` | Fetch once, create one `device.createShaderModule()`, pass to both pipeline creations |

**Risk**: Medium. G4 requires render passes to be compatible (same framebuffer, load/store ops). H5 requires dynamic offset support in bind groups.

---

## Phase 12 — GPU Shaders: Micro-optimizations

**Files**: `forces-tree.wgsl`, `pair-force.wgsl`, `field-fft.wgsl`, `tree-build.wgsl`, `collision.wgsl`, `onePN.wgsl`, `field-evolve.wgsl` (+ Higgs/Axion variants), `field-deposit.wgsl`

| ID | Fix | File | Change |
|----|-----|------|--------|
| G8 | Pack `accumulateForce` params into vec4 groups | `forces-tree.wgsl:42-56` | Group (px,py,pvx,pvy), (pMass,pCharge,pAngVel,pMagMom), etc. ~12 params instead of 27 |
| G9 | Hoist toggle reads before tree walk loop | `forces-tree.wgsl:58-316` | Read `toggles0/toggles1` once, store in local `let` vars |
| G10 | Precompute FFT twiddle factors into lookup buffer | `field-fft.wgsl:85-88` | New `@group(0) @binding(N)` read-only buffer with 128 complex twiddle values. Index by butterfly position. |
| G11 | Non-atomic stores for new tree node init | `tree-build.wgsl:223-276` | Replace `atomicStore` with plain store for all fields except child pointers on parent |
| G17 | Read precomputed cell dims from `FieldUniforms` | `field-evolve.wgsl:54-59` | Replace `domainW / f32(GRID)` with `fieldUniforms.cellW` etc. |
| G18 | Replace `pow(mass, 5.0/3.0)` with derived values | `pair-force.wgsl:431` | `pRi5 = pBodyRadiusSq * pBodyRadiusSq * sqrt(pBodyRadiusSq)` |
| G19 | Check `NONE` before pushing children in collision walk | `collision.wgsl:114-119` | Add `if (childIdx != NONE)` guard per child push |
| G20 | Read `derived.bodyRSq` instead of `pow(mass, 2/3)` | `pair-force.wgsl:502` | `tile[localIdx].bodyRadSq = derived[tileSrcIdx].bodyRSq` |
| G21 | Hoist `axYukMod[i]` before inner loop in pairwise 1PN | `onePN.wgsl:207-213` | Read `axYukMod[i].y` and `.z` once before `for j` loop |

**Risk**: Low-medium. G10 adds a new buffer binding which must be wired through `gpu-pipelines.js`. G11 requires careful reasoning about memory ordering — safe because only the allocating thread writes before publishing child pointers.

**Dependency**: G17 depends on Phase 9 (FieldUniforms changes). G6 uniform split may require updating `@group(0)` bindings across all shaders.

---

## Phase 13 — GPU Shaders: Major Refactors

**Files**: `compute-stats.wgsl`, `bosons.wgsl`, `bosons-tree-walk.wgsl`, `heatmap.wgsl`, `pair-force.wgsl`, `signal-delay-common.wgsl`

| ID | Fix | File | Change |
|----|-----|------|--------|
| G1 | Parallelize `compute-stats.wgsl` | `compute-stats.wgsl` | **Passes 1-2 (O(N))**: One thread per particle, `workgroupBarrier()` + shared memory reduction for KE/momentum/COM. **Pass 3 (O(N^2) PE)**: Tiled pairwise with `TILE_SIZE` shared memory (same pattern as `pair-force.wgsl`), accumulate into shared `peTile`, final reduction. **Pass 4 (O(GRID^2))**: One thread per cell, shared memory reduction. **Pass 5-6 (O(N))**: Keep single-threaded (PQS interpolation + selected particle copy — small N). |
| G2 | Parallelize boson absorption | `bosons.wgsl`, `bosons-tree-walk.wgsl` | **Pass 1 (parallel)**: One thread per boson detects absorption candidates, writes `(bosonIdx, particleIdx, priority)` to append buffer via `atomicAdd` on counter. **Pass 2 (serial)**: Single-thread resolves conflicts (same particle absorbing multiple bosons), applies momentum/charge transfer. |
| G3 | Tree-accelerate GPU heatmap | `heatmap.wgsl` | Replace all-pairs loop with BH tree walk per grid cell. Add `getDelayedStateGPU` for dead particles (tree walk uses current-time aggregate nodes, leaf-level signal delay). Add uniform `deadCount` to skip dead loop when 0. |
| G7 | Mitigate signal delay warp divergence | `pair-force.wgsl`, `signal-delay-common.wgsl` | When `signalDelayed`: (1) Skip shared memory tile preload (delayed positions come from history buffer anyway). (2) Prefetch history ring metadata (bufferStart, sampleCount, creationTime) into per-tile shared memory to reduce global reads in NR solver. |

**Risk**: High. G1 requires careful atomic/reduction patterns and must produce identical results to single-threaded version. G2 changes absorption semantics slightly (conflict resolution order may differ). G3 adds a full BH tree walk to heatmap shader. G7 changes the tiling strategy conditionally.

---

## Cross-Phase Dependencies

```
Phase 1  (dead code)     — no deps
Phase 2  (quadtree)      — no deps
Phase 3  (forces)        — depends on Phase 1 (D12 boson walk dedup)
Phase 4  (integrator)    — depends on Phase 3 (forces API stable)
Phase 5  (fft/fields)    — no deps
Phase 6  (higgs/axion)   — no deps (C16 inlines Laplacian+viscosity directly, independent of Phase 5)
Phase 7  (renderer)      — no deps
Phase 8  (main/ui)       — no deps
Phase 9  (gpu-buffers)   — no deps
Phase 10 (gpu-physics)   — depends on Phase 9 (uniform split)
Phase 11 (gpu-renderer)  — no deps
Phase 12 (gpu-shaders)   — depends on Phase 9 (FieldUniforms), Phase 11 (pipeline changes)
Phase 13 (gpu-refactors) — depends on Phase 12 (shader structure stable)
```

Independent chains that can be parallelized:
- **CPU chain**: 1 → 2 → 3 → 4 (and independently 5 → 6)
- **Renderer chain**: 7, 8 (independent)
- **GPU chain**: 9 → 10 → 11 → 12 → 13

## Verification Strategy

Each phase produces a working commit. Verification per phase:
1. Load each of the 19 presets — no crashes, no NaN in stats
2. Toggle all force/physics combinations — no regressions
3. Switch CPU ↔ GPU backend — consistent behavior
4. Test all 3 topologies (Torus/Klein/RP2)
5. For GPU phases: test with `?cpu=1` fallback still working
6. For shader phases: compare stats output (energy, momentum) between old and new within floating-point tolerance
