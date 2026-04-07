# GPU Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all real bugs and performance issues from the GPU shader audit, prioritized by severity.

**Architecture:** 6 tasks covering: tree OOB guard, collision merge race fix, shader fetch parallelization, per-substep encoder batching, expansion shader flag constant, and collision bounce race guard. No new files — all changes are to existing shaders and JS modules.

**Tech Stack:** WGSL shaders, vanilla ES6 modules, WebGPU API

**Testing:** No test framework. Verify manually: load page, check console for errors, run presets with Barnes-Hut + collisions enabled, toggle Higgs/expansion, resize window. GPU indicator badge should show. `?cpu=1` forces CPU fallback.

---

## Triage Summary

| # | Finding | Verdict | Task |
|---|---------|---------|------|
| 1 | Collision merge race condition | **Fix** — mass duplication under contention | Task 2 |
| 2 | aliveCount naming misleading | **Skip** — comment-only, no behavioral impact |
| 3 | Preset races GPU init | **Skip** — drawDragOverlay clears CPU canvas every frame; GPU canvas shows through correctly |
| 4 | addParticle stale derived on reuse | **Skip** — cacheDerived runs before forces, safe |
| 5 | Pair-force dead loop not tiled | **Skip** — low particle count makes this acceptable |
| 6 | Expansion shader workgroup_size(256) | **Skip** — harmless inconsistency |
| 7 | Field evolve 11 bindings | **Skip** — 10 storage + 1 uniform, within spec |
| 8 | Per-substep encoder overhead | **Fix** — batch substeps into single encoder | Task 4 |
| 9 | writeSubstepUniforms per substep | **Skip** — already writes only 5 fields |
| 10 | Aberration sign | **Skip** — verified correct |
| 11 | 1PN EIH terms | **Skip** — verified correct |
| 12 | Boris rotation signs | **Skip** — verified correct |
| 13 | Tidal torque sign | **Skip** — verified correct |
| 14 | Yukawa Higgs modulation | **Skip** — higgsMod always positive |
| 15 | 30+ shader fetches on cold load | **Fix** — parallelize pipeline creation | Task 3 |
| 16 | Phase 5 fire-and-forget | **Skip** — 1-2 frame gap is acceptable |
| 17 | _PALETTE must exist before GPU init | **Skip** — enforced by script tag order |
| 18 | _PALETTE at module load time | **Skip** — same as 17 |
| 19 | Expansion hardcodes FLAG_ALIVE as 1u | **Fix** — use constant | Task 5 |
| 20 | computeBounds particleCount | **Skip** — verified consistent |
| 21 | Tree insert depth limit | **Skip** — MAX_DEPTH=48 guard confirmed present |
| 22 | allocNode() no OOB guard | **Fix** — can corrupt GPU memory | Task 1 |

Additionally:
| - | Bounce collision concurrent writes | **Fix** — same race pattern as merge | Task 6 |

---

### Task 1: Tree allocNode() OOB guard

**Files:**
- Modify: `src/gpu/shaders/tree-build.wgsl:10-12` (add MAX_NODES constant)
- Modify: `src/gpu/shaders/tree-build.wgsl:120-122` (guard allocNode)
- Modify: `src/gpu/shaders/tree-build.wgsl:223-276` (guard subdivide)
- Modify: `src/gpu/gpu-constants.js:60-256` (add QT_MAX_NODES to WGSL constants)

The tree node buffer is sized `maxParticles * 6 = 3072` nodes in gpu-buffers.js:122. `allocNode()` does `atomicAdd(&nodeCounter, 1u)` with no bounds check. If pathological particle placement causes excessive subdivision, writes go OOB and corrupt adjacent GPU buffers.

- [ ] **Step 1: Add QT_MAX_NODES to WGSL constants**

In `src/gpu/gpu-constants.js`, inside the `buildWGSLConstants()` template string, after the `MAX_DEPTH` line (around line 221), add:

```js
const QT_MAX_NODES: u32 = ${GPU_MAX_PARTICLES * 6}u;
```

This requires importing `GPU_MAX_PARTICLES` which is already imported at line 28.

- [ ] **Step 2: Guard allocNode() return value**

In `src/gpu/shaders/tree-build.wgsl`, replace `allocNode` (lines 120-122):

```wgsl
fn allocNode() -> u32 {
    return atomicAdd(&nodeCounter, 1u);
}
```

with:

```wgsl
fn allocNode() -> u32 {
    let idx = atomicAdd(&nodeCounter, 1u);
    if (idx >= QT_MAX_NODES) { return 0u; } // OOB → alias root (safe no-op)
    return idx;
}
```

Returning 0 (root) on overflow means the child writes overwrite the root node. This is benign — the tree is already corrupt at this point, and the frame will produce garbage forces but no memory corruption. The tree is rebuilt from scratch every substep.

- [ ] **Step 3: Guard subdivide() against overflow**

In `src/gpu/shaders/tree-build.wgsl`, in `subdivide()` (line 223), add an early-out after the 4 allocations:

Replace lines 231-234:

```wgsl
    let nw = allocNode();
    let ne = allocNode();
    let sw = allocNode();
    let se = allocNode();
```

with:

```wgsl
    let nw = allocNode();
    let ne = allocNode();
    let sw = allocNode();
    let se = allocNode();

    // OOB guard: if any child aliased root, skip subdivision
    if (nw == 0u || ne == 0u || sw == 0u || se == 0u) { return; }
```

- [ ] **Step 4: Bump SHADER_VERSION**

In `src/gpu/gpu-pipelines.js`, increment the `SHADER_VERSION` constant (line 12):

```js
const SHADER_VERSION = 58;
```

- [ ] **Step 5: Verify**

Serve locally, load with Barnes-Hut enabled preset (e.g., "Binary Star" or any preset with gravity + BH toggle). Confirm no console errors. Spawn 100+ particles to stress the tree.

- [ ] **Step 6: Commit**

```
git add src/gpu/shaders/tree-build.wgsl src/gpu/gpu-constants.js src/gpu/gpu-pipelines.js
git commit -m "fix: add OOB guard to GPU quadtree allocNode — prevents buffer overflow on pathological particle placement"
```

---

### Task 2: Collision merge race condition — atomic claim

**Files:**
- Modify: `src/gpu/shaders/collision.wgsl:130-310` (resolveCollisions entry point)

Two threads processing pairs (A,B) and (B,C) can both read B as alive, both merge it, duplicating B's mass. Fix: use `atomicExchange` on the flags field to atomically claim the right to consume a particle. Only the thread that successfully transitions `FLAG_ALIVE → FLAG_RETIRED` gets to use that particle's mass.

The `particleState` array uses `ParticleState` structs (not atomic), but we can't atomicExchange on a struct field. Instead, use a two-phase approach: first atomically zero the mass of the consumed particle (the "loser"), and only proceed if the CAS succeeds.

Actually, WGSL struct fields aren't individually atomic. The cleanest approach: change the alive check to use the mass itself as the claim signal. After reading both particles, immediately zero the loser's mass with a store. If another thread also zeroed it, the second reader's copy will still have the old mass — but that's the race we're trying to prevent.

The real fix: since `particleState` is `read_write` storage but not atomic, we need a separate atomic claim buffer. But that adds complexity. A simpler approach that's sufficient for 512 max particles: **sort collision pairs so each particle appears in at most one pair per frame**.

Actually, the simplest correct fix: add a per-particle `atomic<u32>` claim buffer. Each thread atomicExchanges the claim for both particles. If either exchange returns non-zero (already claimed), skip the pair.

- [ ] **Step 1: Add collision claim buffer**

In `src/gpu/gpu-buffers.js`, after the `mergeResultCounter` buffer (around line 198), add:

```js
    const collisionClaims = device.createBuffer({
        label: 'collisionClaims',
        size: 4 * maxParticles,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
```

Add `collisionClaims` to the return object (around line 284).

- [ ] **Step 2: Add claim buffer binding to collision shader**

In `src/gpu/shaders/collision.wgsl`, add a new binding in group 1 after `allForces` (after line 31):

```wgsl
@group(1) @binding(4) var<storage, read_write> collisionClaims: array<atomic<u32>>;
```

This goes in group 1 (not group 2) to stay within the `maxStorageBuffersPerShaderStage = 10` limit. Group 0 has 1 storage, group 1 has 4→5 storage, group 2 has 4 storage = 10 total.

- [ ] **Step 3: Add atomic claim to resolveCollisions**

In `src/gpu/shaders/collision.wgsl`, in `resolveCollisions` (after the alive checks around line 155), add claim logic:

Replace:

```wgsl
    if ((ps1.flags & FLAG_ALIVE) == 0u || (ps2.flags & FLAG_ALIVE) == 0u) { return; }
    if (ps1.mass <= EPSILON || ps2.mass <= EPSILON) { return; }
```

with:

```wgsl
    if ((ps1.flags & FLAG_ALIVE) == 0u || (ps2.flags & FLAG_ALIVE) == 0u) { return; }
    if (ps1.mass <= EPSILON || ps2.mass <= EPSILON) { return; }

    // Atomic claim: prevent concurrent merge of same particle by multiple pairs.
    // atomicExchange returns previous value — if non-zero, another thread already claimed.
    let claim1 = atomicExchange(&collisionClaims[idx1], 1u);
    let claim2 = atomicExchange(&collisionClaims[idx2], 1u);
    if (claim1 != 0u || claim2 != 0u) {
        // Release any claim we took (so particle can be merged next frame)
        if (claim1 == 0u) { atomicStore(&collisionClaims[idx1], 0u); }
        if (claim2 == 0u) { atomicStore(&collisionClaims[idx2], 0u); }
        return;
    }
```

- [ ] **Step 4: Update collision pipeline bind group layout**

In `src/gpu/gpu-pipelines.js`, find `createCollisionPipelines` (line 337). The `group1Layout` currently has 4 entries (lines 352-358). Add a 5th entry for collisionClaims:

```js
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // collisionClaims (atomic claim)
```

- [ ] **Step 5: Update collision bind group creation in gpu-physics.js**

In `src/gpu/gpu-physics.js`, find `_createCollisionBindGroups`. Add the `collisionClaims` buffer as binding 4 in group 1. The bind group creation for collision group 1 needs `b.collisionClaims` appended to its entries array.

- [ ] **Step 6: Clear claims buffer before collision dispatch**

In `src/gpu/gpu-physics.js`, in `_dispatchCollisions`, add a `clearBuffer` call for `collisionClaims` before the collision resolve pass (after detection, before resolve). Find where `collisionPairCounter` is cleared (it's already cleared before detection) and add nearby:

```js
encoder.clearBuffer(this.buffers.collisionClaims, 0, this.aliveCount * 4);
```

- [ ] **Step 7: Bump SHADER_VERSION**

Already bumped in Task 1. If implementing separately, bump to 59.

- [ ] **Step 8: Verify**

Load any preset with collision mode set to "Merge". Spawn overlapping particles rapidly. Observe that total mass is conserved (check Energy sidebar — mass energy should be stable after merges, no duplication).

- [ ] **Step 9: Commit**

```
git add src/gpu/shaders/collision.wgsl src/gpu/gpu-buffers.js src/gpu/gpu-pipelines.js src/gpu/gpu-physics.js
git commit -m "fix: atomic collision claim prevents concurrent merge consuming same particle twice"
```

---

### Task 3: Parallelize shader fetch on cold load

**Files:**
- Modify: `src/gpu/gpu-physics.js:387-552` (init method)

Currently, `init()` calls 7 pipeline-creation functions sequentially with `await`. Each function internally fetches 1-6 shader files. On cold load this creates a ~20-request waterfall. Fix: call all pipeline-creation functions in parallel via `Promise.all`.

The only dependency is that `_ensureSharedCache()` must complete before pipelines that use `getSharedPrefix()`. But `getSharedPrefix` already calls `_ensureSharedCache` internally, so every pipeline function is independently safe to call in parallel.

- [ ] **Step 1: Parallelize Phase 2-4 pipeline creation**

In `src/gpu/gpu-physics.js`, replace the sequential pipeline creation in `init()` (lines ~432-460):

```js
        // --- Phase 2 pipelines ---
        this._phase2 = await createPhase2Pipelines(this.device, wgslConstants);
        this._createPhase2BindGroups();

        // --- Phase 3: Ghost generation pipeline ---
        const ghostGen = await createGhostGenPipeline(this.device, wgslConstants);
        ...
        // --- Phase 4: Advanced physics pipelines ---
        this._phase4 = await createPhase4Pipelines(this.device, wgslConstants);
        ...
```

with:

```js
        // --- Pipelines: fetch all shaders in parallel ---
        const [phase2, ghostGen, treeBuild, treeForce, collisionPipelines, deadGC, phase4, updateColors, trailRecord, hitTest, computeStats] =
            await Promise.all([
                createPhase2Pipelines(this.device, wgslConstants),
                createGhostGenPipeline(this.device, wgslConstants),
                createTreeBuildPipelines(this.device, wgslConstants),
                createTreeForcePipeline(this.device, wgslConstants),
                createCollisionPipelines(this.device, wgslConstants),
                createDeadGCPipeline(this.device, wgslConstants),
                createPhase4Pipelines(this.device, wgslConstants),
                createUpdateColorsPipeline(this.device, wgslConstants),
                createTrailRecordPipeline(this.device, wgslConstants),
                createHitTestPipeline(this.device, wgslConstants),
                createComputeStatsPipeline(this.device, wgslConstants),
            ]);

        this._phase2 = phase2;
        this._createPhase2BindGroups();

        this._ghostGenPipeline = ghostGen.pipeline;
        this._createGhostGenBindGroups(ghostGen.bindGroupLayouts);

        this._treeBuild = treeBuild;
        this._createTreeBuildBindGroups(this._treeBuild.bindGroupLayouts);

        const tfResult = treeForce;
        this._treeForcePipeline = tfResult.pipeline;
        this._createTreeForceBindGroups(tfResult.bindGroupLayouts);

        this._collisionPipelines = collisionPipelines;
        this._createCollisionBindGroups(this._collisionPipelines.bindGroupLayouts);

        this._deadGCPipeline = deadGC.pipeline;
        this._createDeadGCBindGroup(deadGC.bindGroupLayouts);

        this._phase4 = phase4;
        this._createPhase4BindGroups();
```

Then handle updateColors, trailRecord, hitTest, and computeStats the same way — assign their results from the destructured array and create bind groups. Preserve the exact same bind group creation code that currently follows each `await`, just move the `await` into the single `Promise.all`.

- [ ] **Step 2: Also pre-fetch shared includes in init()**

The 5 fetches at lines 390-396 (`sharedStructs`, `sharedTopo`, `sharedRng`, `commonSrc`, `boundaryWGSL`) are already parallel via `Promise.all`. No change needed there.

But these shared files are ALSO fetched by `_ensureSharedCache()` inside the pipeline functions. Since all pipeline functions now start simultaneously, `_ensureSharedCache()` will be called ~11 times before its cache is populated. The internal `if (!_sharedCache)` check means only the first call fetches; the rest see `_sharedCache` already being populated. But there's a TOCTOU race: multiple calls see `_sharedCache === null`, all start fetching.

Fix: make `_ensureSharedCache` use a promise lock:

In `src/gpu/gpu-pipelines.js`, replace:

```js
let _sharedCache = null;

async function _ensureSharedCache() {
    if (!_sharedCache) {
        const [structs, topo, treeNodes, rng] = await Promise.all([
            fetchShader('shared-structs.wgsl'),
            fetchShader('shared-topology.wgsl'),
            fetchShader('shared-tree-nodes.wgsl'),
            fetchShader('shared-rng.wgsl'),
        ]);
        _sharedCache = { structs, topo, treeNodes, rng };
    }
    return _sharedCache;
}
```

with:

```js
let _sharedCache = null;
let _sharedCachePromise = null;

async function _ensureSharedCache() {
    if (_sharedCache) return _sharedCache;
    if (!_sharedCachePromise) {
        _sharedCachePromise = Promise.all([
            fetchShader('shared-structs.wgsl'),
            fetchShader('shared-topology.wgsl'),
            fetchShader('shared-tree-nodes.wgsl'),
            fetchShader('shared-rng.wgsl'),
        ]).then(([structs, topo, treeNodes, rng]) => {
            _sharedCache = { structs, topo, treeNodes, rng };
            return _sharedCache;
        });
    }
    return _sharedCachePromise;
}
```

- [ ] **Step 3: Verify**

Open DevTools Network tab. Reload page with cache disabled. Confirm shader requests are batched (many start at the same time) rather than waterfall'd. GPU init should complete faster (roughly 1-2 network round-trips instead of ~20).

- [ ] **Step 4: Commit**

```
git add src/gpu/gpu-physics.js src/gpu/gpu-pipelines.js
git commit -m "perf: parallelize all shader fetches on GPU init — eliminates cold-load waterfall"
```

---

### Task 4: Batch substeps into single queue.submit() call

**Files:**
- Modify: `src/gpu/gpu-physics.js:3129-3435` (update + _dispatchSubstep)

Currently each substep creates its own command encoder and calls `queue.submit()`. With up to 32 substeps per frame, that's 32 GPU→driver context switches. Fix: have `_dispatchSubstep` return the command buffer instead of submitting, then batch-submit all at once.

The substep uniform fields are scattered across the uniform buffer (offsets 0, 4, 68, 120, 128), so using `encoder.copyBufferToBuffer` from staging would need 5 separate copies per substep. The simpler approach: keep `writeSubstepUniforms` (which uses `queue.writeBuffer`) as-is, since all `queue.writeBuffer` calls execute at queue time before any encoder commands. This means we need one encoder per substep (each encoder depends on its uniform state), but we batch-submit all command buffers in one call.

- [ ] **Step 1: Modify _dispatchSubstep to return command buffer**

In `src/gpu/gpu-physics.js`, change `_dispatchSubstep(dtSub)` (line 3270):

Replace the last line:

```js
        this.device.queue.submit([encoder.finish()]);
```

with:

```js
        return encoder.finish();
```

- [ ] **Step 2: Modify update() to batch-submit**

In `src/gpu/gpu-physics.js`, in `update()` (around line 3179), replace:

```js
        for (let step = 0; step < numSubsteps; step++) {
            this.simTime += dtSub;
            this._dispatchSubstep(dtSub);
        }
```

with:

```js
        const commandBuffers = [];
        for (let step = 0; step < numSubsteps; step++) {
            this.simTime += dtSub;
            commandBuffers.push(this._dispatchSubstep(dtSub));
        }
        this.device.queue.submit(commandBuffers);
```

This is safe because `writeSubstepUniforms` uses `queue.writeBuffer` which is sequenced by the WebGPU spec: each `queue.writeBuffer` happens-before the next `queue.submit`, and `queue.submit` with multiple command buffers executes them in order.

- [ ] **Step 3: Verify**

Load page, check simulation runs identically. Adaptive substepping (high-acceleration scenarios) should still produce correct physics. Use DevTools Performance panel to confirm fewer GPU submit calls per frame.

- [ ] **Step 4: Commit**

```
git add src/gpu/gpu-physics.js
git commit -m "perf: batch substep command buffers into single queue.submit call"
```

---

### Task 5: Expansion shader — use FLAG_ALIVE constant

**Files:**
- Modify: `src/gpu/shaders/expansion.wgsl:28`

Trivial fix. The shader hardcodes `1u` instead of using the `FLAG_ALIVE` constant. Since expansion.wgsl is prepended with the shared prefix (which includes the constants block defining `FLAG_ALIVE`), the constant is available.

- [ ] **Step 1: Replace hardcoded flag**

In `src/gpu/shaders/expansion.wgsl`, line 28, replace:

```wgsl
    if ((flag & 1u) == 0u) { return; }
```

with:

```wgsl
    if ((flag & FLAG_ALIVE) == 0u) { return; }
```

- [ ] **Step 2: Verify the constant is available**

Check that expansion.wgsl is prepended with the wgslConstants block. Look at `createExpansionPipeline` in `gpu-pipelines.js` — it should use `getSharedPrefix(wgslConstants)` or equivalent. If it uses a custom prefix without the constants block, the `FLAG_ALIVE` symbol won't be defined.

If expansion.wgsl uses its own `ExpansionUniforms` struct and is NOT prepended with the shared prefix (the file header says "NOT prepended with common.wgsl"), then `FLAG_ALIVE` won't be available. In that case, add the constant locally:

```wgsl
const FLAG_ALIVE: u32 = 1u;
```

at the top of expansion.wgsl (after the struct definition, before the entry point).

- [ ] **Step 3: Bump SHADER_VERSION if not already bumped**

- [ ] **Step 4: Commit**

```
git add src/gpu/shaders/expansion.wgsl src/gpu/gpu-pipelines.js
git commit -m "fix: expansion shader uses FLAG_ALIVE constant instead of hardcoded 1u"
```

---

### Task 6: Bounce collision race guard

**Files:**
- Modify: `src/gpu/shaders/collision.wgsl:416-534` (resolveBouncePairwise entry point)

The bounce resolution has the same race as merge: two threads can apply impulse to the same particle simultaneously, with both reading stale velocities and writing conflicting updates. Unlike merge (where mass duplication is the concern), bounce races cause impulse loss or doubling.

For bounce, the fix is simpler than merge: since bounce doesn't kill particles, we only need to prevent concurrent modification. Reuse the same `collisionClaims` buffer from Task 2.

- [ ] **Step 1: Add atomic claim to resolveBouncePairwise**

In `src/gpu/shaders/collision.wgsl`, in `resolveBouncePairwise` (after the alive check at line 428), add the same claim pattern:

```wgsl
    // Atomic claim: prevent concurrent impulse on same particle
    let claim1 = atomicExchange(&collisionClaims[idx1], 1u);
    let claim2 = atomicExchange(&collisionClaims[idx2], 1u);
    if (claim1 != 0u || claim2 != 0u) {
        if (claim1 == 0u) { atomicStore(&collisionClaims[idx1], 0u); }
        if (claim2 == 0u) { atomicStore(&collisionClaims[idx2], 0u); }
        return;
    }
```

The `collisionClaims` buffer is already bound (from Task 2) and cleared before collision dispatch. For bounce mode, `resolveCollisions` won't run (collision mode selects one or the other), so the claims buffer is exclusively used by bounce.

- [ ] **Step 2: Verify**

Load with collision mode "Bounce". Spawn overlapping particles. Confirm impulses are applied correctly (particles repel, no visibly frozen pairs or double-speed ejections).

- [ ] **Step 3: Commit**

```
git add src/gpu/shaders/collision.wgsl
git commit -m "fix: atomic claim prevents concurrent bounce impulse on same particle"
```

---

## Execution order

Tasks 1, 2, 5, 6 are shader changes that can be done in parallel (different files or different entry points in collision.wgsl). Task 3 is JS-only. Task 4 is JS-only.

Recommended order: 1 → 2+6 → 5 → 3 → 4.
