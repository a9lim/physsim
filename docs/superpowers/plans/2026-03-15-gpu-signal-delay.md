# GPU Signal Delay Parity — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring GPU signal delay to full CPU parity — retarded positions in force computation, dead particle forces, creationTime guards, and aberration on all force paths.

**Architecture:** Interleave 7 history buffers into 1 (`histData`, stride=6) + expand `histMeta` (stride 2→4 with creationTime). Extract shared `signal-delay-common.wgsl` for NR light-cone solver. Move jerk from `RadiationState` to `AllForces._pad` to free a storage binding for forces-tree. Add signal delay + dead particle support to `pair-force.wgsl`, `forces-tree.wgsl`, `onePN.wgsl`, and `heatmap.wgsl`. Fix CPU aggregate aberration and ghost signal delay.

**Tech Stack:** WebGPU (WGSL compute shaders), vanilla JS (no build step)

**Spec:** `docs/superpowers/specs/2026-03-15-gpu-signal-delay-design.md`

## Review Fixes (post-review amendments)

1. **Atomic histMeta migration**: Tasks 1, 3, 4, 5, 6 (buffer restructuring, history.wgsl rewrite, signal-delay-common.wgsl creation, gpu-physics.js histMeta init, gpu-pipelines.js layout) must all land in a **single commit** at the end of Task 6. Individual commit steps within Tasks 1-6 are removed — commit once after Task 6 step 2 with message: `refactor: interleave history buffers, shared signal delay solver, creationTime in histMeta`.

2. **pair-force bind group layout**: Task 7 (jerk migration) must NOT create an intermediate pair-force layout. Instead, Task 7 step 6 should set pair-force to its **final** layout (matching Task 8 step 1): `Group 1: 4 storage (particles, derived, axYukMod, particleAux), Group 2: 2 storage (allForces, maxAccel), Group 3: 2 storage (histData, histMeta)`. This means Task 7 and Task 8 pipeline layout changes for pair-force are combined. Task 8 step 1 then only needs to update the WGSL shader code, not the pipeline layout.

3. **Dead particle loop bound**: All dead particle loops (pair-force Task 8 step 5, forces-tree Task 9 step 5, heatmap Task 11 step 4) must scan `[0, uniforms.particleCount)` NOT `[0, uniforms.aliveCount)`. Also fix the **pre-existing bug** in `forces-tree.wgsl` line 628 which uses `uniforms.aliveCount` — change to `uniforms.particleCount`.

4. **Heatmap particleAux binding**: Task 11 must add `particleAux` as a binding (group 0, binding 1) in heatmap.wgsl. Add the `ParticleAux` struct definition. Update `createHeatmapPipelines()` in gpu-pipelines.js to include `particleAux` in group 0. Use `particleAux[di].deathMass` for dead particle gravity/Yukawa in the dead loop. Remove the "for now use frozen mass" workaround comment.

5. **compute-stats.wgsl**: Add to Task 7 step 2 and step 8 commit. Rename `_pad: vec2<f32>` → `jerk: vec2<f32>` in its local `AllForces` struct definition.

---

## Chunk 1: Infrastructure — Buffer Restructuring & Shared Solver

### Task 1: Interleave history buffers in gpu-buffers.js

**Files:**
- Modify: `src/gpu/gpu-buffers.js:314-364` (history buffer allocation)

- [ ] **Step 1: Replace 7 history buffer fields with 2**

Replace the 7 individual buffer properties (`histPosX`, `histPosY`, `histVelWX`, `histVelWY`, `histAngW`, `histTime`) and the existing `histMeta` with 2 new buffers:

```javascript
// In the field declarations (around line 314-321):
// Replace:
//   this.histPosX = null; this.histPosY = null;
//   this.histVelWX = null; this.histVelWY = null;
//   this.histAngW = null; this.histTime = null;
//   this.histMeta = null;
// With:
this.histData = null;    // interleaved [posX, posY, velX, velY, angW, time] per sample
this.histMeta = null;    // [writeIdx, count, creationTimeBits, _pad] per particle (stride 4)
```

- [ ] **Step 2: Update allocateHistoryBuffers()**

Replace the allocation method (around lines 328-364):

```javascript
allocateHistoryBuffers(dev) {
    if (this.historyAllocated) return;
    const HIST_STRIDE = 6; // posX, posY, velX, velY, angW, time
    const dataSize = this.soaCapacity * HISTORY_SIZE * HIST_STRIDE * 4; // f32 each
    this.histData = dev.createBuffer({
        label: 'histData',
        size: dataSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // histMeta: 4 u32 per particle (writeIdx, count, creationTimeBits, _pad)
    const metaSize = this.soaCapacity * 4 * 4; // 4 u32 per particle
    this.histMeta = dev.createBuffer({
        label: 'histMeta',
        size: metaSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.historyAllocated = true;
}
```

- [ ] **Step 3: Update destroy() to clean up new buffer names**

In the `destroy()` method, replace the 7 individual buffer destroy calls with:

```javascript
if (this.histData) { this.histData.destroy(); this.histData = null; }
if (this.histMeta) { this.histMeta.destroy(); this.histMeta = null; }
```

Remove the old `histPosX`, `histPosY`, etc. destroy calls.

- [ ] **Step 4: Commit**

```bash
git add src/gpu/gpu-buffers.js
git commit -m "refactor: interleave history buffers — 7 separate → histData + histMeta"
```

---

### Task 2: Add HIST_STRIDE constant to gpu-constants.js

**Files:**
- Modify: `src/gpu/gpu-constants.js:173-178` (signal delay constants section)

- [ ] **Step 1: Add HIST_STRIDE and HIST_META_STRIDE constants**

After the `NR_TOLERANCE` line (around line 177), add:

```javascript
// In the WGSL constants template string:
const HIST_STRIDE: u32 = 6u;       // interleaved: posX, posY, velX, velY, angW, time
const HIST_META_STRIDE: u32 = 4u;  // writeIdx, count, creationTimeBits, _pad
```

Also export a JS-side constant for use in gpu-physics.js:

```javascript
// Near the other JS-side exports (around line 230+):
export const HIST_STRIDE = 6;
export const HIST_META_STRIDE = 4;
```

- [ ] **Step 2: Commit**

```bash
git add src/gpu/gpu-constants.js
git commit -m "feat: add HIST_STRIDE/HIST_META_STRIDE constants for interleaved history"
```

---

### Task 3: Rewrite history.wgsl for interleaved format

**Files:**
- Modify: `src/gpu/shaders/history.wgsl` (entire file — recording + remove old getDelayedStateGPU)

- [ ] **Step 1: Update buffer bindings**

Replace the 7 history buffer bindings (group 1, bindings 0-6) with 2:

```wgsl
// Group 1: history buffers (interleaved)
@group(1) @binding(0) var<storage, read_write> histData: array<f32>;
@group(1) @binding(1) var<storage, read_write> histMeta: array<u32>;
```

- [ ] **Step 2: Rewrite recordHistory to write interleaved format**

```wgsl
@compute @workgroup_size(64)
fn recordHistory(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= arrayLength(&particles)) { return; }
    if ((particles[i].flags & FLAG_ALIVE) == 0u) { return; }

    let metaBase = i * HIST_META_STRIDE;

    // Reborn particles: clear stale history, set creationTime, clear flag
    if ((particles[i].flags & FLAG_REBORN) != 0u) {
        histMeta[metaBase] = 0u;              // writeIdx
        histMeta[metaBase + 1u] = 0u;         // count
        histMeta[metaBase + 2u] = bitcast<u32>(u.simTime); // creationTime
        histMeta[metaBase + 3u] = 0u;         // _pad
        particles[i].flags &= ~FLAG_REBORN;
    }

    var writeIdx = histMeta[metaBase];
    var count = histMeta[metaBase + 1u];

    let sampleBase = i * HISTORY_LEN * HIST_STRIDE
                   + (writeIdx & HISTORY_MASK) * HIST_STRIDE;

    histData[sampleBase + 0u] = particles[i].posX;
    histData[sampleBase + 1u] = particles[i].posY;

    // Store coordinate velocity (vel = w / sqrt(1 + w²))
    let wx = particles[i].velWX;
    let wy = particles[i].velWY;
    let gamma = sqrt(1.0 + wx * wx + wy * wy);
    let invG = 1.0 / gamma;
    histData[sampleBase + 2u] = wx * invG;
    histData[sampleBase + 3u] = wy * invG;

    histData[sampleBase + 4u] = particles[i].angW;
    histData[sampleBase + 5u] = u.simTime;

    writeIdx = (writeIdx + 1u) & HISTORY_MASK;
    count = min(count + 1u, HISTORY_LEN);
    histMeta[metaBase] = writeIdx;
    histMeta[metaBase + 1u] = count;
}
```

- [ ] **Step 3: Remove getDelayedStateGPU and supporting functions**

Delete the `DelayedState` struct, `minImageDisp()` function, and `getDelayedStateGPU()` function (lines 97-419 of old file). These move to `signal-delay-common.wgsl` in Task 4.

- [ ] **Step 4: Commit**

```bash
git add src/gpu/shaders/history.wgsl
git commit -m "refactor: history.wgsl writes interleaved histData, removes getDelayedStateGPU"
```

---

### Task 4: Create signal-delay-common.wgsl

**Files:**
- Create: `src/gpu/shaders/signal-delay-common.wgsl`

- [ ] **Step 1: Write the shared signal delay solver**

This file is prepended to consuming shaders. It references `histData` and `histMeta` by name — consumers declare the bindings. It also references `HIST_STRIDE`, `HIST_META_STRIDE`, `HISTORY_LEN`, `HISTORY_MASK`, `NR_TOLERANCE`, `NR_MAX_ITER`, `EPSILON`, `TOPO_TORUS`, `TOPO_KLEIN` from the generated constants block.

```wgsl
// ─── Signal Delay Common ───
// Shared Newton-Raphson light-cone solver for interleaved history buffers.
// Prepended to consuming shaders. Callers declare histData/histMeta bindings.

struct DelayedState {
    x: f32, y: f32,
    vx: f32, vy: f32,
    angw: f32,
    valid: bool,
};

// Full topology-aware minimum image displacement (Torus/Klein/RP²)
fn sdMinImageDisp(ox: f32, oy: f32, sx: f32, sy: f32,
                  domW: f32, domH: f32, topo: u32) -> vec2f {
    // [Copy the minImageDisp function from old history.wgsl lines 106-163,
    //  renamed to sdMinImageDisp to avoid name collisions with common.wgsl's
    //  fullMinImage which uses uniforms.domainW directly]
}

// Helper: read interleaved history sample fields
fn histSampleBase(srcIdx: u32, sampleIdx: u32) -> u32 {
    return srcIdx * HISTORY_LEN * HIST_STRIDE + sampleIdx * HIST_STRIDE;
}

fn getDelayedStateGPU(
    srcIdx: u32,
    obsX: f32, obsY: f32,
    simTime: f32,
    periodic: bool,
    domW: f32, domH: f32,
    topoMode: u32,
    isDead: bool,
) -> DelayedState {
    var result: DelayedState;
    result.valid = false;

    let metaBase = srcIdx * HIST_META_STRIDE;
    let writeIdx = histMeta[metaBase];
    let count = histMeta[metaBase + 1u];
    if (count < 2u) { return result; }

    let start = (writeIdx - count + HISTORY_LEN) & HISTORY_MASK;
    let newest = (writeIdx - 1u + HISTORY_LEN) & HISTORY_MASK;

    // Read oldest/newest timestamps from interleaved data
    let oldestBase = histSampleBase(srcIdx, start);
    let newestBase = histSampleBase(srcIdx, newest);
    let tOldest = histData[oldestBase + 5u]; // time field at offset 5
    let tNewest = histData[newestBase + 5u];
    let timeSpan = simTime - tOldest;
    if (timeSpan < NR_TOLERANCE) { return result; }

    // Current distance to newest sample
    let nxPos = histData[newestBase + 0u];
    let nyPos = histData[newestBase + 1u];
    var cdx: f32; var cdy: f32;
    if (periodic) {
        let d = sdMinImageDisp(obsX, obsY, nxPos, nyPos, domW, domH, topoMode);
        cdx = d.x; cdy = d.y;
    } else {
        cdx = nxPos - obsX; cdy = nyPos - obsY;
    }
    let distSq = cdx * cdx + cdy * cdy;

    // ─── Phase 1: Newton-Raphson segment search ───
    if (distSq <= 4.0 * timeSpan * timeSpan) {
        var t = simTime - sqrt(distSq);
        t = clamp(t, tOldest, tNewest);

        let histSpan = tNewest - tOldest;
        var segK: i32;
        if (histSpan > NR_TOLERANCE) {
            segK = i32(floor((t - tOldest) / histSpan * f32(count - 1u)));
        } else { segK = 0; }
        segK = clamp(segK, 0, i32(count) - 2);

        // Walk to correct segment
        for (var w = 0; w < 256; w++) {
            if (segK >= i32(count) - 2) { break; }
            let nextBase = histSampleBase(srcIdx, (start + u32(segK + 1)) & HISTORY_MASK);
            if (histData[nextBase + 5u] > t) { break; }
            segK++;
        }
        for (var w = 0; w < 256; w++) {
            if (segK <= 0) { break; }
            let curBase = histSampleBase(srcIdx, (start + u32(segK)) & HISTORY_MASK);
            if (histData[curBase + 5u] <= t) { break; }
            segK--;
        }

        var prevSegK: i32 = -1;
        for (var iter = 0u; iter < NR_MAX_ITER; iter++) {
            if (segK == prevSegK) { break; }
            prevSegK = segK;

            let loBase = histSampleBase(srcIdx, (start + u32(segK)) & HISTORY_MASK);
            let hiBase = histSampleBase(srcIdx, ((start + u32(segK)) + 1u) & HISTORY_MASK);
            let tLo = histData[loBase + 5u];
            let segDt = histData[hiBase + 5u] - tLo;
            if (segDt < NR_TOLERANCE) {
                if (segK < i32(count) - 2) { segK++; prevSegK = -1; continue; }
                break;
            }

            let xLo = histData[loBase]; let yLo = histData[loBase + 1u];
            var vxEff: f32; var vyEff: f32;
            if (periodic) {
                let d = sdMinImageDisp(xLo, yLo, histData[hiBase], histData[hiBase + 1u], domW, domH, topoMode);
                vxEff = d.x / segDt; vyEff = d.y / segDt;
            } else {
                vxEff = (histData[hiBase] - xLo) / segDt;
                vyEff = (histData[hiBase + 1u] - yLo) / segDt;
            }

            let s = t - tLo;
            let sx_interp = xLo + vxEff * s;
            let sy_interp = yLo + vyEff * s;

            var dx: f32; var dy: f32;
            if (periodic) {
                let d = sdMinImageDisp(obsX, obsY, sx_interp, sy_interp, domW, domH, topoMode);
                dx = d.x; dy = d.y;
            } else {
                dx = sx_interp - obsX; dy = sy_interp - obsY;
            }

            let dSq = dx * dx + dy * dy;
            if (dSq < NR_TOLERANCE * NR_TOLERANCE) { break; }
            let dist = sqrt(dSq);

            let g = dist - (simTime - t);
            let gp = (dx * vxEff + dy * vyEff) / dist + 1.0;
            if (abs(gp) < NR_TOLERANCE) { break; }

            t -= g / gp;
            t = clamp(t, tOldest, tNewest);

            for (var w2 = 0; w2 < 64; w2++) {
                if (segK >= i32(count) - 2) { break; }
                let ni = histSampleBase(srcIdx, (start + u32(segK + 1)) & HISTORY_MASK);
                if (histData[ni + 5u] > t) { break; }
                segK++;
            }
            for (var w2 = 0; w2 < 64; w2++) {
                if (segK <= 0) { break; }
                let ci = histSampleBase(srcIdx, (start + u32(segK)) & HISTORY_MASK);
                if (histData[ci + 5u] <= t) { break; }
                segK--;
            }
        }

        // ─── Phase 2: Exact quadratic on converged segment (+/- 1 neighbor) ───
        let center = segK;
        for (var offset = 0; offset <= 1; offset++) {
            for (var dir = select(-1, 1, offset == 0); dir <= 1; dir += 2) {
                let k = center + offset * dir;
                if (k < 0 || k > i32(count) - 2) { continue; }

                let loBase = histSampleBase(srcIdx, (start + u32(k)) & HISTORY_MASK);
                let hiBase = histSampleBase(srcIdx, ((start + u32(k)) + 1u) & HISTORY_MASK);
                let tLo = histData[loBase + 5u];
                let segDt = histData[hiBase + 5u] - tLo;
                if (segDt < NR_TOLERANCE) { continue; }

                let xLo = histData[loBase]; let yLo = histData[loBase + 1u];
                let xHi = histData[hiBase]; let yHi = histData[hiBase + 1u];

                var dx: f32; var dy: f32;
                var vx: f32; var vy: f32;
                if (periodic) {
                    let d0 = sdMinImageDisp(obsX, obsY, xLo, yLo, domW, domH, topoMode);
                    dx = d0.x; dy = d0.y;
                    let d1 = sdMinImageDisp(xLo, yLo, xHi, yHi, domW, domH, topoMode);
                    vx = d1.x / segDt; vy = d1.y / segDt;
                } else {
                    dx = xLo - obsX; dy = yLo - obsY;
                    vx = (xHi - xLo) / segDt; vy = (yHi - yLo) / segDt;
                }

                let rSq = dx * dx + dy * dy;
                let vSq = vx * vx + vy * vy;
                let dDotV = dx * vx + dy * vy;
                let T = simTime - tLo;

                let a = vSq - 1.0;
                let h = dDotV + T;
                let c = rSq - T * T;
                let disc = h * h - a * c;
                if (disc < 0.0) { continue; }

                let sqrtDisc = sqrt(max(disc, 0.0));
                var s_sol: f32;
                if (abs(a) < NR_TOLERANCE) {
                    if (abs(h) < NR_TOLERANCE) { continue; }
                    s_sol = -c / (2.0 * h);
                } else {
                    let s1 = (-h + sqrtDisc) / a;
                    let s2 = (-h - sqrtDisc) / a;
                    let ok1 = s1 >= -EPSILON && s1 <= segDt + EPSILON;
                    let ok2 = s2 >= -EPSILON && s2 <= segDt + EPSILON;
                    if (ok1 && ok2) { s_sol = max(s1, s2); }
                    else if (ok1) { s_sol = s1; }
                    else if (ok2) { s_sol = s2; }
                    else { continue; }
                }

                s_sol = clamp(s_sol, 0.0, segDt);
                let frac = s_sol / segDt;

                result.x = xLo + frac * (xHi - xLo);
                result.y = yLo + frac * (yHi - yLo);
                // Interpolate velocity from interleaved data
                let loVx = histData[loBase + 2u]; let hiVx = histData[hiBase + 2u];
                let loVy = histData[loBase + 3u]; let hiVy = histData[hiBase + 3u];
                result.vx = loVx + frac * (hiVx - loVx);
                result.vy = loVy + frac * (hiVy - loVy);
                // Interpolate angw
                let loAngw = histData[loBase + 4u]; let hiAngw = histData[hiBase + 4u];
                result.angw = loAngw + frac * (hiAngw - loAngw);
                result.valid = true;
                return result;
            }
        }
    }

    // Dead particles: don't extrapolate past buffer
    if (isDead) { return result; }

    // ─── Phase 3: Extrapolation from oldest sample ───
    {
        let xStart = histData[oldestBase];
        let yStart = histData[oldestBase + 1u];
        var dx: f32; var dy: f32;
        if (periodic) {
            let d = sdMinImageDisp(obsX, obsY, xStart, yStart, domW, domH, topoMode);
            dx = d.x; dy = d.y;
        } else {
            dx = xStart - obsX; dy = yStart - obsY;
        }

        let vx = histData[oldestBase + 2u];
        let vy = histData[oldestBase + 3u];
        let rSq = dx * dx + dy * dy;
        let vSq = vx * vx + vy * vy;
        let dDotV = dx * vx + dy * vy;
        let T = timeSpan;

        let a = vSq - 1.0;
        let h = dDotV + T;
        let c = rSq - T * T;
        let disc = h * h - a * c;
        if (disc < 0.0) { return result; }

        let sqrtDisc = sqrt(disc);
        var s_sol: f32;
        if (abs(a) < NR_TOLERANCE) {
            if (abs(h) < NR_TOLERANCE) { return result; }
            s_sol = -c / (2.0 * h);
        } else {
            let s1 = (-h + sqrtDisc) / a;
            let s2 = (-h - sqrtDisc) / a;
            let ok1 = s1 <= EPSILON;
            let ok2 = s2 <= EPSILON;
            if (ok1 && ok2) { s_sol = max(s1, s2); }
            else if (ok1) { s_sol = s1; }
            else if (ok2) { s_sol = s2; }
            else { return result; }
        }
        if (s_sol > 0.0) { s_sol = 0.0; }

        // Reject extrapolation past particle creation
        let creationTimeBits = histMeta[srcIdx * HIST_META_STRIDE + 2u];
        let creationTime = bitcast<f32>(creationTimeBits);
        if (tOldest + s_sol < creationTime) { return result; }

        result.x = xStart + vx * s_sol;
        result.y = yStart + vy * s_sol;
        result.vx = vx;
        result.vy = vy;
        result.angw = histData[oldestBase + 4u];
        result.valid = true;
        return result;
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/gpu/shaders/signal-delay-common.wgsl
git commit -m "feat: create signal-delay-common.wgsl with interleaved getDelayedStateGPU"
```

---

### Task 5: Update gpu-physics.js history initialization

**Files:**
- Modify: `src/gpu/gpu-physics.js` (addParticle, reset, deserialize, dispatchHistory, bind groups)

- [ ] **Step 1: Update addParticle() to initialize histMeta with creationTime**

In `addParticle()` (around line 1370), after the existing histMeta initialization, add creationTime. The current code writes 2 u32s (writeIdx=0, count=0). Change to write 4 u32s:

```javascript
// Initialize history metadata (4 u32: writeIdx, count, creationTimeBits, _pad)
if (this._buffers.historyAllocated) {
    const metaBuf = new ArrayBuffer(16);
    const metaU32 = new Uint32Array(metaBuf);
    const metaF32 = new Float32Array(metaBuf);
    metaU32[0] = 0;  // writeIdx
    metaU32[1] = 0;  // count
    metaF32[2] = this._simTime; // creationTime as f32 bits
    metaU32[3] = 0;  // _pad
    this._device.queue.writeBuffer(
        this._buffers.histMeta,
        idx * 16, // 4 u32 per particle
        metaBuf
    );
}
```

- [ ] **Step 2: Update reset() to clear histMeta with stride 4**

If `reset()` clears history metadata, update the stride from 2 to 4 u32 per particle.

- [ ] **Step 3: Update deserialize() histMeta initialization**

In `deserialize()`, after uploading particles, initialize histMeta for each particle with stride 4 (writeIdx=0, count=0, creationTime=-Infinity bits, _pad=0). Use `-Infinity` for deserialized particles (they're treated as always existing, matching CPU `loadState`).

- [ ] **Step 4: Update dispatchHistory() bind group creation**

The history recording dispatch creates a bind group for group 1. Update it to bind the new `histData` and `histMeta` buffers instead of the 7 separate ones:

```javascript
// Group 1: histData + histMeta (was 7 separate buffers)
const histBindGroup = this._device.createBindGroup({
    layout: this._historyPipeline.bindGroupLayouts[1],
    entries: [
        { binding: 0, resource: { buffer: b.histData } },
        { binding: 1, resource: { buffer: b.histMeta } },
    ],
});
```

- [ ] **Step 5: Commit**

```bash
git add src/gpu/gpu-physics.js
git commit -m "feat: gpu-physics histMeta stride 4, creationTime init, interleaved bind groups"
```

---

### Task 6: Update gpu-pipelines.js history pipeline layout

**Files:**
- Modify: `src/gpu/gpu-pipelines.js` (history pipeline bind group layout)

- [ ] **Step 1: Find the history recording pipeline creation**

This is in `createPhase4Pipelines()` (around line 369). The history pipeline's group 1 layout currently has 7 bindings (one per history buffer). Change to 2 bindings:

```javascript
// Group 1: histData (rw) + histMeta (rw) = 2 storage
// Was: 7 separate buffers (histPosX, histPosY, histVelWX, histVelWY, histAngW, histTime, histMeta)
```

Update the layout entries array from 7 `'storage'` entries to 2.

- [ ] **Step 2: Commit**

```bash
git add src/gpu/gpu-pipelines.js
git commit -m "refactor: history pipeline layout — 7 bindings → 2 (histData + histMeta)"
```

---

### Task 7: Rename AllForces._pad → AllForces.jerk

**Files:**
- Modify: `src/gpu/shaders/common.wgsl:66` (AllForces struct)
- Modify: All standalone shaders that define their own copy of AllForces

- [ ] **Step 1: Rename in common.wgsl**

```wgsl
// In AllForces struct (line 66):
// Replace:
//     _pad: vec2<f32>,
// With:
    jerk: vec2<f32>,  // analytical jerk for Larmor radiation (was _pad)
```

- [ ] **Step 2: Find and update all standalone AllForces definitions**

Search for `_pad: vec2` in all `.wgsl` files. Standalone shaders (not prepended with common.wgsl) that define their own `AllForces` struct must also rename `_pad` → `jerk`. These include at minimum: `forces-tree.wgsl`, `radiation.wgsl`, `compute-stats.wgsl`.

Run: `grep -rn "_pad: vec2" src/gpu/shaders/` to find all instances.

- [ ] **Step 3: Update pair-force.wgsl to write jerk to AllForces.jerk**

In pair-force.wgsl, replace the radiationState jerk write (around line 440-443):

```wgsl
// Replace:
//     var rs = radState[idx];
//     rs.jerkX = select(accJerkX, 0.0, accJerkX != accJerkX);
//     rs.jerkY = select(accJerkY, 0.0, accJerkY != accJerkY);
//     radState[idx] = rs;
// With:
    af.jerk = vec2(
        select(accJerkX, 0.0, accJerkX != accJerkX),
        select(accJerkY, 0.0, accJerkY != accJerkY)
    );
```

Move this BEFORE the `allForces[idx] = af;` write (since `af` is already being constructed).

Remove the `radState` binding declaration from pair-force.wgsl (group 3, binding 0). Update group 3 to only have `maxAccel`.

- [ ] **Step 4: Update forces-tree.wgsl to write jerk to AllForces.jerk**

Same pattern: replace `radiationState[pIdx].jerkX/jerkY` writes with `localAF.jerk = vec2(...)`. Remove the `radiationState` binding. Move `maxAccel` from group 2 binding 2 to group 2 binding 1.

- [ ] **Step 5: Update radiation.wgsl to read jerk from AllForces**

In `radiation.wgsl` (around lines 186-187), replace:

```wgsl
// Replace:
//     var jerkXVal = radState[i].jerkX;
//     var jerkYVal = radState[i].jerkY;
// With:
    var jerkXVal = allForces[i].jerk.x;
    var jerkYVal = allForces[i].jerk.y;
```

- [ ] **Step 6: Update gpu-pipelines.js bind group layouts**

For `pairForce` in `createPhase2Pipelines()` (line 71): remove `radiationState` from group 3. New layout:
```javascript
// Group 0: uniform
// Group 1: 3 storage (particles, derived, axYukMod)
// Group 2: 1 storage (allForces)
// Group 3: 1 storage (maxAccel)
const pairForce = await makePipeline('pairForce', 'pair-force.wgsl', [
    ['uniform'],
    ['storage', 'storage', 'storage'],
    ['storage'],
    ['storage'],
]);
```

For `treeForce` in `createTreeForcePipeline()` (line 221): remove `radiationState` from group 2. New layout:
```javascript
// Group 0: nodes (storage) + uniforms
// Group 1: particleState + particleAux + derived + axYukMod + ghostOriginalIdx = 5 storage
// Group 2: allForces + maxAccel = 2 storage
```

- [ ] **Step 7: Update gpu-physics.js bind group creation for force dispatches**

Update bind group creation to stop binding `radiationState` for pair-force and tree-force dispatches. Update radiation dispatch to read jerk from allForces (already bound).

- [ ] **Step 8: Commit**

```bash
git add src/gpu/shaders/common.wgsl src/gpu/shaders/pair-force.wgsl src/gpu/shaders/forces-tree.wgsl src/gpu/shaders/radiation.wgsl src/gpu/gpu-pipelines.js src/gpu/gpu-physics.js
git commit -m "refactor: move jerk from RadiationState to AllForces.jerk, free storage binding"
```

---

## Chunk 2: Force Shader Signal Delay Integration

### Task 8: Add signal delay to pair-force.wgsl

**Files:**
- Modify: `src/gpu/shaders/pair-force.wgsl` (tile restructure, signal delay lookup, dead particles)
- Modify: `src/gpu/gpu-pipelines.js` (pair-force bind group layout)
- Modify: `src/gpu/gpu-physics.js` (pair-force bind group creation)

- [ ] **Step 1: Add history bindings and prepend signal-delay-common.wgsl**

Update `createPhase2Pipelines()` in gpu-pipelines.js to prepend `signal-delay-common.wgsl` to pair-force.wgsl (after `common.wgsl`). Add new bind group layout:

```javascript
// New pair-force layout:
// Group 0: uniform
// Group 1: particles + derived + axYukMod + particleAux = 4 storage
// Group 2: allForces + maxAccel = 2 storage
// Group 3: histData + histMeta = 2 storage
const pairForce = await makePipeline('pairForce', 'pair-force.wgsl', [
    ['uniform'],
    ['storage', 'storage', 'storage', 'storage'],
    ['storage', 'storage'],
    ['storage', 'storage'],
]);
```

Note: the `makePipeline` helper in createPhase2Pipelines must be updated to support prepending signal-delay-common.wgsl for pair-force specifically (or create pair-force separately from the helper).

- [ ] **Step 2: Add bindings in pair-force.wgsl**

Add `particleAux` to group 1 and history to group 3:

```wgsl
@group(1) @binding(3) var<storage, read_write> particleAux: array<ParticleAux>;

// Group 3: signal delay history (interleaved)
@group(3) @binding(0) var<storage, read_write> histData: array<f32>;
@group(3) @binding(1) var<storage, read_write> histMeta: array<u32>;
```

Add the `ParticleAux` struct definition (posX/posY not needed — just the aux fields):
```wgsl
struct ParticleAux {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
};
```

- [ ] **Step 3: Modify TileParticle for signal delay mode**

Add `srcIdx: u32` to `TileParticle`. When signal delay is active, tile loads `srcIdx` and observer-independent properties; each thread calls `getDelayedStateGPU` per source:

```wgsl
struct TileParticle {
    posX: f32, posY: f32,     // current pos (used when signal delay off)
    velX: f32, velY: f32,     // current vel (used when signal delay off)
    mass: f32, charge: f32,
    angVel: f32,              // current angVel (used when signal delay off)
    magMoment: f32,           // current (used when signal delay off)
    angMomentum: f32,         // current (used when signal delay off)
    axMod: f32, yukMod: f32,
    radiusSq: f32,
    srcIdx: u32,              // NEW: particle index for signal delay lookup
};
```

- [ ] **Step 4: Add signal delay lookup in inner loop**

In the force accumulation loop (after loading from tile), when `signalDelayed` is true:

```wgsl
// After: let s = tile[j]; / if (s.mass < EPSILON) { continue; }
// Add signal delay lookup:
var sx = s.posX; var sy = s.posY;
var svx = s.velX; var svy = s.velY;
var sAngVel_sd = s.angVel;
var sMagMoment_sd = s.magMoment;
var sAngMomentum_sd = s.angMomentum;

if (signalDelayed) {
    let delayed = getDelayedStateGPU(
        s.srcIdx, pPosX, pPosY, uniforms.simTime,
        isPeriodic, uniforms.domainW, uniforms.domainH,
        uniforms.topologyMode, false
    );
    if (!delayed.valid) { continue; } // skip — outside light cone
    sx = delayed.x; sy = delayed.y;
    svx = delayed.vx; svy = delayed.vy;
    // Recompute dipoles from retarded angw
    let bodyRadSq = pow(s.mass, 2.0 / 3.0);
    let retAngwSq = delayed.angw * delayed.angw;
    sAngVel_sd = delayed.angw / sqrt(1.0 + retAngwSq * bodyRadSq);
    sMagMoment_sd = MAG_MOMENT_K * s.charge * sAngVel_sd * bodyRadSq;
    sAngMomentum_sd = INERTIA_K * s.mass * sAngVel_sd * bodyRadSq;
}
```

Then use `sx, sy, svx, svy, sAngVel_sd, sMagMoment_sd, sAngMomentum_sd` in place of `s.posX, s.posY, s.velX, s.velY, s.angVel, s.magMoment, s.angMomentum` throughout the force computation.

- [ ] **Step 5: Add dead particle loop after tile loop**

After the tiled force accumulation, add:

```wgsl
// Dead particle forces (signal delay fade-out)
if (alive && signalDelayed) {
    let maxSlots = uniforms.aliveCount;
    for (var ri = 0u; ri < maxSlots; ri++) {
        let rPs = particles[ri];
        if ((rPs.flags & FLAG_RETIRED) == 0u) { continue; }
        if ((rPs.flags & FLAG_ALIVE) != 0u) { continue; }

        let rAux = particleAux[ri];
        let delayed = getDelayedStateGPU(
            ri, pPosX, pPosY, uniforms.simTime,
            isPeriodic, uniforms.domainW, uniforms.domainH,
            uniforms.topologyMode, true  // isDead = true
        );
        if (!delayed.valid) { continue; }

        // Recompute dipoles from retarded state using deathMass
        let bodyRadSq = pow(rAux.deathMass, 2.0 / 3.0);
        let retAngwSq = delayed.angw * delayed.angw;
        let dAngVel = delayed.angw / sqrt(1.0 + retAngwSq * bodyRadSq);
        let dMagMom = MAG_MOMENT_K * rPs.charge * dAngVel * bodyRadSq;
        let dAngMom = INERTIA_K * rAux.deathMass * dAngVel * bodyRadSq;

        // Accumulate forces (same logic as tile loop, with signalDelayed=true)
        // ... [call the same force accumulation code with:
        //      sx=delayed.x, sy=delayed.y, svx=delayed.vx, svy=delayed.vy,
        //      mass=rAux.deathMass, charge=rPs.charge,
        //      angVel=dAngVel, magMoment=dMagMom, angMomentum=dAngMom,
        //      axMod=1.0, yukMod=1.0, aberration=true]
    }
}
```

Consider extracting the force accumulation into a helper function to avoid duplicating the gravity/Coulomb/dipole/Yukawa code between the tile loop and dead particle loop.

- [ ] **Step 6: Update gpu-physics.js bind group creation**

Update the pair-force dispatch to create bind groups matching the new layout (add particleAux to group 1, histData+histMeta to group 3).

- [ ] **Step 7: Commit**

```bash
git add src/gpu/shaders/pair-force.wgsl src/gpu/gpu-pipelines.js src/gpu/gpu-physics.js
git commit -m "feat: pair-force.wgsl signal delay — retarded positions, dead particles, aberration"
```

---

### Task 9: Add signal delay to forces-tree.wgsl

**Files:**
- Modify: `src/gpu/shaders/forces-tree.wgsl` (leaf signal delay, ghost handling, aggregate aberration, dead stub)
- Modify: `src/gpu/gpu-pipelines.js` (tree-force bind group layout)
- Modify: `src/gpu/gpu-physics.js` (tree-force bind group creation)

- [ ] **Step 1: Add history bindings (group 3)**

After the jerk migration (Task 7), forces-tree has 8 storage. Add group 3:

```wgsl
// Group 3: signal delay history (interleaved)
@group(3) @binding(0) var<storage, read_write> histData: array<f32>;
@group(3) @binding(1) var<storage, read_write> histMeta: array<u32>;
```

Total: 10 storage + 1 uniform. At limit.

Update `createTreeForcePipeline()` in gpu-pipelines.js to add group 3 layout and prepend `signal-delay-common.wgsl`.

- [ ] **Step 2: Add signal delay at leaf nodes (non-ghost)**

In the leaf iteration (around line 570-608 of current file), after checking `other !== particle`:

```wgsl
// When signal delay is active and particle is not a ghost:
if (hasSignalDelay && (otherFlags & FLAG_GHOST) == 0u) {
    let delayed = getDelayedStateGPU(
        srcParticleIdx, px, py, uniforms.simTime,
        isPeriodic, uniforms.domainW, uniforms.domainH,
        uniforms.topologyMode, false
    );
    if (!delayed.valid) { continue; }
    // Use delayed.x, delayed.y, delayed.vx, delayed.vy
    // Recompute dipoles from delayed.angw
    // Set signalDelayed = true for aberration
}
```

- [ ] **Step 3: Add signal delay for ghost leaf nodes**

When the source is a ghost:

```wgsl
if (hasSignalDelay && (otherFlags & FLAG_GHOST) != 0u) {
    let origIdx = ghostOriginalIdx[srcParticleIdx];
    let origPs = particleState[origIdx];
    let delayed = getDelayedStateGPU(
        origIdx, px, py, uniforms.simTime,
        isPeriodic, uniforms.domainW, uniforms.domainH,
        uniforms.topologyMode, false
    );
    if (!delayed.valid) { continue; }
    // Periodic shift: ghost.pos - original.currentPos
    let shiftX = otherPosX - origPs.posX;
    let shiftY = otherPosY - origPs.posY;
    // Retarded ghost position = retarded original + shift
    let gsx = delayed.x + shiftX;
    let gsy = delayed.y + shiftY;
    // Use gsx, gsy, delayed.vx, delayed.vy for force computation
    // Recompute dipoles from delayed.angw
}
```

- [ ] **Step 4: Add aberration to aggregate nodes**

In the aggregate force computation (around line 663), add aberration using the aggregate's average velocity:

```wgsl
// After computing forces from aggregate CoM:
// Add aberration factor
if (hasSignalDelay) {
    let avgVx = getTotalMomX(nodeIdx) / nodeMass;
    let avgVy = getTotalMomY(nodeIdx) / nodeMass;
    // ... compute aberration and apply to accumulated force from this aggregate
}
```

This requires modifying `accumulateForce()` to accept a `signalDelayed` flag and apply aberration when true.

- [ ] **Step 5: Fix dead particle stub**

Replace the stub at line 633 (`// Phase 4 will add signal delay lookup here`) with proper signal delay:

```wgsl
// Dead particles: signal delay lookup
if (hasSignalDelay) {
    let delayed = getDelayedStateGPU(
        ri, px, py, uniforms.simTime,
        isPeriodic, uniforms.domainW, uniforms.domainH,
        uniforms.topologyMode, true  // isDead
    );
    if (!delayed.valid) { continue; }
    let rAux = particleAux[ri];
    // Use delayed position/velocity, rAux.deathMass, rPs.charge
    // Recompute dipoles from delayed.angw and deathMass
    // Apply aberration
}
```

- [ ] **Step 6: Update gpu-pipelines.js and gpu-physics.js**

Add group 3 (histData + histMeta) to tree-force pipeline layout and bind group creation.

- [ ] **Step 7: Commit**

```bash
git add src/gpu/shaders/forces-tree.wgsl src/gpu/gpu-pipelines.js src/gpu/gpu-physics.js
git commit -m "feat: forces-tree.wgsl signal delay — leaves, ghosts, aggregate aberration, dead particles"
```

---

### Task 10: Add signal delay to onePN.wgsl

**Files:**
- Modify: `src/gpu/shaders/onePN.wgsl` (signal delay lookup per pair)
- Modify: `src/gpu/gpu-pipelines.js` (1PN bind group layout)
- Modify: `src/gpu/gpu-physics.js` (1PN bind group creation)

- [ ] **Step 1: Add history bindings and prepend signal-delay-common.wgsl**

Add group 3 to onePN.wgsl:

```wgsl
@group(3) @binding(0) var<storage, read_write> histData: array<f32>;
@group(3) @binding(1) var<storage, read_write> histMeta: array<u32>;
```

Update pipeline creation to prepend `signal-delay-common.wgsl`.

- [ ] **Step 2: Add signal delay lookup in 1PN pair loop**

For each source particle in the O(N²) loop, call `getDelayedStateGPU`. Use retarded position/velocity. Skip source on `valid=false`. No aberration (1PN is already O(v²/c²)). Dead particles excluded from 1PN (matching CPU).

- [ ] **Step 3: Update gpu-pipelines.js and gpu-physics.js**

Add group 3 layout and bind group creation for 1PN dispatches.

- [ ] **Step 4: Commit**

```bash
git add src/gpu/shaders/onePN.wgsl src/gpu/gpu-pipelines.js src/gpu/gpu-physics.js
git commit -m "feat: onePN.wgsl signal delay — retarded positions for all 1PN terms"
```

---

## Chunk 3: Heatmap, Boundary, CPU Fixes

### Task 11: Update heatmap.wgsl to use shared solver + dead particles

**Files:**
- Modify: `src/gpu/shaders/heatmap.wgsl` (replace inline solver, add dead loop)
- Modify: `src/gpu/gpu-pipelines.js` (heatmap bind group layout — already has history bindings)
- Modify: `src/gpu/gpu-physics.js` (heatmap bind group creation — update for new buffer names)

- [ ] **Step 1: Remove inline getRetardedPosition()**

Delete the `RetardedPos` struct and `getRetardedPosition()` function from heatmap.wgsl. These are replaced by `getDelayedStateGPU` from `signal-delay-common.wgsl`.

- [ ] **Step 2: Prepend signal-delay-common.wgsl**

Update `createHeatmapPipelines()` in gpu-pipelines.js to prepend `signal-delay-common.wgsl` to heatmap.wgsl.

- [ ] **Step 3: Update computeHeatmap to use getDelayedStateGPU**

Replace the `getRetardedPosition(i, wx, wy, hu.simTime)` call with:

```wgsl
let ret = getDelayedStateGPU(
    i, wx, wy, hu.simTime,
    hu.periodic != 0u, hu.domainW, hu.domainH,
    hu.topologyMode, false
);
if (ret.valid) {
    srcX = ret.x; srcY = ret.y;
}
```

- [ ] **Step 4: Add dead particle loop**

After the alive-particle loop, add:

```wgsl
// Dead particles: signal delay fade-out
if (useDelay) {
    for (var di = 0u; di < hu.particleCount; di++) {
        let dp = particles[di];
        if ((dp.flags & FLAG_RETIRED) == 0u) { continue; }
        if ((dp.flags & FLAG_ALIVE) != 0u) { continue; }

        let ret = getDelayedStateGPU(
            di, wx, wy, hu.simTime,
            hu.periodic != 0u, hu.domainW, hu.domainH,
            hu.topologyMode, true  // isDead
        );
        if (!ret.valid) { continue; }

        // Need deathMass — add particleAux binding or read mass from ParticleState
        // (deathMass stored in ParticleAux, need to add binding)
        // For now, use the frozen mass in ParticleState (already zeroed for merge kills)
        // Better: add particleAux to heatmap bindings
        // ... compute potential from retarded position
    }
}
```

Note: The heatmap needs access to `deathMass` from `ParticleAux` for dead particles. Add `particleAux` as a binding in group 0 alongside `particleState`. Check storage buffer limits — current heatmap has 8 storage + 1 uniform. Adding `particleAux` → 9 storage. Within limits.

- [ ] **Step 5: Update heatmap bind groups for interleaved history**

The heatmap pipeline already has group 2 for history (4 bindings: histPosX, histPosY, histTime, histMeta). Change to 2 bindings (histData, histMeta). Update gpu-pipelines.js and gpu-physics.js.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/shaders/heatmap.wgsl src/gpu/gpu-pipelines.js src/gpu/gpu-physics.js
git commit -m "feat: heatmap uses shared signal delay solver, adds dead particle potential"
```

---

### Task 12: Fix boundary.wgsl to write deathTime

**Files:**
- Modify: `src/gpu/shaders/boundary.wgsl` (write deathTime + deathMass + deathAngVel before FLAG_RETIRED)
- Modify: `src/gpu/gpu-pipelines.js` (boundary bind group — add particleAux if not already bound)

- [ ] **Step 1: Check if boundary.wgsl has access to particleAux**

Current bindings: group 0 has uniforms + particleState. Need to add particleAux.

```wgsl
@group(0) @binding(2) var<storage, read_write> particleAux: array<ParticleAux>;
```

Add ParticleAux struct definition to boundary.wgsl (standalone shader).

- [ ] **Step 2: Write death metadata before setting FLAG_RETIRED**

At the despawn point (line 107):

```wgsl
// Replace:
//     ps.flags = (ps.flags & ~FLAG_ALIVE) | FLAG_RETIRED;
// With:
    // Save death metadata for signal delay fade-out
    var aux = particleAux[idx];
    aux.deathTime = uniforms.simTime;
    aux.deathMass = ps.mass;
    // Compute deathAngVel from angW
    let sr = ps.angW * aux.radius;
    let relOn = (uniforms.toggles0 & RELATIVITY_BIT) != 0u;
    aux.deathAngVel = select(ps.angW, ps.angW / sqrt(1.0 + sr * sr), relOn);
    particleAux[idx] = aux;
    ps.flags = (ps.flags & ~FLAG_ALIVE) | FLAG_RETIRED;
```

- [ ] **Step 3: Update boundary pipeline layout in gpu-pipelines.js**

Add `particleAux` storage binding to the boundary pipeline's bind group layout. Update gpu-physics.js to pass the buffer.

- [ ] **Step 4: Commit**

```bash
git add src/gpu/shaders/boundary.wgsl src/gpu/gpu-pipelines.js src/gpu/gpu-physics.js
git commit -m "feat: boundary.wgsl writes deathTime/deathMass for signal delay fade-out"
```

---

### Task 13: CPU fixes — aggregate aberration + ghost signal delay

**Files:**
- Modify: `src/forces.js:661-668` (aggregate aberration), `src/forces.js:644-660` (ghost signal delay)

- [ ] **Step 1: Add aberration to aggregate node pairForce call**

At line 668 in forces.js, the aggregate `pairForce()` call doesn't pass `signalDelayed`. Fix:

```javascript
// Replace line 668:
// pairForce(particle, pool.comX[nodeIdx], pool.comY[nodeIdx], avgVx, avgVy,
//     nodeMass, pool.totalCharge[nodeIdx], 0,
//     pool.totalMagneticMoment[nodeIdx], pool.totalAngularMomentum[nodeIdx],
//     out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology, 1, 1);
// With (add useSignalDelay as last arg):
pairForce(particle, pool.comX[nodeIdx], pool.comY[nodeIdx], avgVx, avgVy,
    nodeMass, pool.totalCharge[nodeIdx], 0,
    pool.totalMagneticMoment[nodeIdx], pool.totalAngularMomentum[nodeIdx],
    out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology, 1, 1, useSignalDelay);
```

- [ ] **Step 2: Add signal delay for ghost particles**

Replace the ghost exclusion at line 644 (`if (useSignalDelay && !other.isGhost)`) with handling for both cases:

```javascript
if (useSignalDelay) {
    if (real.histCount < 2) continue;
    const ret = getDelayedState(real, particle, simTime, periodic, domW, domH, halfDomW, halfDomH, topology);
    if (!ret) continue;
    if (other.isGhost) {
        // Retarded original position + periodic shift
        const shiftX = other.pos.x - real.pos.x;
        const shiftY = other.pos.y - real.pos.y;
        sx = ret.x + shiftX;
        sy = ret.y + shiftY;
    } else {
        sx = ret.x;
        sy = ret.y;
    }
    svx = ret.vx; svy = ret.vy;
    const retAngwSq = ret.angw * ret.angw;
    const retRadiusSq = real.bodyRadiusSq;
    sAngVel = ret.angw / Math.sqrt(1 + retAngwSq * retRadiusSq);
    sMagMom = MAG_MOMENT_K * other.charge * sAngVel * retRadiusSq;
    sAngMom = INERTIA_K * other.mass * sAngVel * retRadiusSq;
    delayed = true;
} else {
    sx = other.pos.x; sy = other.pos.y; svx = other.vel.x; svy = other.vel.y;
    sAngVel = other.angVel; sMagMom = other.magMoment; sAngMom = other.angMomentum;
    delayed = false;
}
```

- [ ] **Step 3: Commit**

```bash
git add src/forces.js
git commit -m "fix: CPU aggregate aberration + ghost signal delay in BH tree walk"
```

---

### Task 14: Verification and CLAUDE.md update

- [ ] **Step 1: Manual verification**

Serve from `a9lim.github.io/` and test in browser:
1. Load a preset with multiple particles (e.g., "Binary Star")
2. Enable Relativity toggle → verify signal delay activates (forces use retarded positions)
3. Toggle GPU mode on/off → verify behavior is consistent between backends
4. Enable Barnes-Hut → verify tree walk uses signal delay at leaves
5. Enable Black Hole mode → let particles merge → verify dead particle forces fade via signal delay
6. Switch topology to Klein/RP² → verify periodic signal delay works
7. Check heatmap overlay with Relativity on → verify retarded potential field
8. Boundary mode = Despawn → push particle off edge → verify it continues exerting force briefly

- [ ] **Step 2: Update CLAUDE.md**

Update the following sections in `physsim/CLAUDE.md`:
- **GPU Acceleration / Packed Struct Buffers**: Update history buffer description (7 → 2 buffers)
- **GPU Acceleration / Dispatch sequence**: Note that force dispatches now include history bind groups
- **AllForces struct**: Document `jerk: vec2<f32>` replacing `_pad`
- **Signal Delay**: Note GPU parity achieved
- **Key Patterns**: Note `signal-delay-common.wgsl` prepend pattern

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for GPU signal delay parity, interleaved history, jerk migration"
```
