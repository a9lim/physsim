# Physsim Stability & Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve simulation stability via fixed-timestep physics, fix physics correctness issues, eliminate GC pressure from QuadTree allocation, and split the monolithic Physics class into focused modules.

**Architecture:** Fixed-timestep accumulator in the main loop feeds constant-dt ticks to the physics engine. QuadTree uses a pre-allocated node pool instead of per-frame construction. Physics.js splits into integrator/forces/collisions/potential/signal-delay modules. Stats display extracts from Simulation into its own module.

**Tech Stack:** Vanilla JS ES6 modules, Canvas 2D, no build step, no test framework. Verify via browser console (`window.sim`) and visual inspection. Serve from parent `a9lim.github.io/` with `python -m http.server`.

---

## Verification Protocol

Since there's no test framework, each task includes manual verification steps:
- **Console check**: Open browser devtools, run commands against `window.sim`
- **Visual check**: Observe simulation behavior (particles move, collisions work, etc.)
- **Energy check**: Watch the Stats tab drift percentages — should stay small for gravity-only scenarios
- **Regression check**: Load each preset (keys 1-5), let run 10 seconds, confirm no NaN/freezes

---

### Task 1: Fixed-Timestep Physics Loop

**Files:**
- Modify: `src/config.js` (add PHYSICS_DT constant)
- Modify: `main.js:196-273` (rewrite loop method)

**Step 1: Add PHYSICS_DT constant**

In `src/config.js`, add after the `MAX_SUBSTEPS` line:

```js
// Fixed physics timestep (seconds, before speedScale)
export const PHYSICS_DT = 1 / 120;
```

**Step 2: Import PHYSICS_DT in main.js**

Update the import from config.js in `main.js:9`:

```js
import { ZOOM_MIN, ZOOM_MAX, WHEEL_ZOOM_IN, DEFAULT_SPEED_SCALE, INERTIA_K, PHOTON_LIFETIME, FRAGMENT_COUNT, PHYSICS_DT, MAX_SUBSTEPS } from './src/config.js';
```

**Step 3: Add accumulator state to Simulation constructor**

In `main.js`, add to the constructor (after `this.running = true;` around line 43):

```js
this.accumulator = 0;
```

**Step 4: Rewrite the loop method**

Replace `main.js` loop method (lines 196-273) with:

```js
loop(timestamp) {
    const rawDt = Math.min((timestamp - this.lastTime) / 1000, 0.1);
    this.lastTime = timestamp;

    if (this.running) {
        this.accumulator += rawDt * this.speedScale;
        const maxAccum = PHYSICS_DT * MAX_SUBSTEPS * 4;
        if (this.accumulator > maxAccum) this.accumulator = maxAccum;

        const cam = this.camera;
        const halfW = this.width / (2 * cam.zoom);
        const halfH = this.height / (2 * cam.zoom);

        while (this.accumulator >= PHYSICS_DT) {
            this.physics.update(this.particles, PHYSICS_DT, this.collisionMode, this.boundaryMode, halfW * 2, halfH * 2, cam.x - halfW, cam.y - halfH);

            // Update photons (inside fixed step for time consistency)
            for (let i = this.photons.length - 1; i >= 0; i--) {
                const ph = this.photons[i];
                ph.update(PHYSICS_DT);

                if (this.physics.radiationEnabled) {
                    for (const p of this.particles) {
                        const dx = ph.pos.x - p.pos.x, dy = ph.pos.y - p.pos.y;
                        const distSq = dx * dx + dy * dy;
                        if (distSq < p.radius * p.radius) {
                            const impulse = ph.energy / p.mass;
                            p.w.x += ph.vel.x * impulse;
                            p.w.y += ph.vel.y * impulse;
                            ph.alive = false;
                            this.totalRadiated = Math.max(0, this.totalRadiated - ph.energy);
                            this.totalRadiatedPx -= ph.vel.x * ph.energy;
                            this.totalRadiatedPy -= ph.vel.y * ph.energy;
                            break;
                        }
                    }
                }

                if (!ph.alive || ph.lifetime > PHOTON_LIFETIME) {
                    this.photons.splice(i, 1);
                }
            }

            // Tidal breakup (inside fixed step)
            const toFragment = this.physics.checkTidalBreakup(this.particles);
            for (const p of toFragment) {
                const idx = this.particles.indexOf(p);
                if (idx === -1) continue;
                this.particles.splice(idx, 1);

                const n = FRAGMENT_COUNT;
                const fragMass = p.mass / n;
                const fragCharge = p.charge / n;

                for (let i = 0; i < n; i++) {
                    const angle = (2 * Math.PI * i) / n;
                    const offset = p.radius * 1.5;
                    const fx = p.pos.x + Math.cos(angle) * offset;
                    const fy = p.pos.y + Math.sin(angle) * offset;
                    const tangVx = -Math.sin(angle) * p.angVel * offset;
                    const tangVy = Math.cos(angle) * p.angVel * offset;
                    this.addParticle(fx, fy, p.vel.x + tangVx, p.vel.y + tangVy, {
                        mass: fragMass, charge: fragCharge, spin: p.angw
                    });
                }
            }

            this.accumulator -= PHYSICS_DT;
        }
    }

    this.heatmap.update(this.particles, this.camera, this.width, this.height);
    this.phasePlot.update(this.particles, this.selectedParticle);
    this.renderer.render(this.particles, PHYSICS_DT, this.camera, this.photons);
    this.phasePlot.draw(this.renderer.isLight);
    this.sankey.draw(this.renderer.isLight);
    if (this.running) this.computeEnergy();
    this.updateSelectedParticle();

    requestAnimationFrame((t) => this.loop(t));
}
```

**Step 5: Update step button in ui.js**

In `src/ui.js:199-208`, the step button currently uses `0.1 * sim.speedScale` as dt. Change to use `PHYSICS_DT`:

Add import at top of ui.js:
```js
import { PHYSICS_DT } from './config.js';
```

Replace the step button dt calculation (line 204):
```js
const dt = PHYSICS_DT;
```

And the same for the keyboard step handler (line 242):
```js
const dt = PHYSICS_DT;
```

**Step 6: Verify**

- Serve site, load Solar System preset (key 1)
- Watch energy drift in Stats tab — should be small and stable
- Change speed slider to extreme values (1, 500) — simulation should remain stable
- Throttle CPU in devtools (6x slowdown) — simulation should slow down but not explode
- Console: `sim.accumulator` should be a small positive number near 0

**Step 7: Commit**

```bash
git add src/config.js main.js src/ui.js
git commit -m "feat: fixed-timestep physics loop with accumulator"
```

---

### Task 2: Time-Based Photon Lifetime

**Files:**
- Modify: `src/config.js` (change PHOTON_LIFETIME value)
- Modify: `src/photon.js:12` (accumulate dt instead of ++)

**Step 1: Change PHOTON_LIFETIME to time-based**

In `src/config.js`, change:
```js
export const PHOTON_LIFETIME = 300;          // frames before despawn
```
to:
```js
export const PHOTON_LIFETIME = 30;           // sim-time-units before despawn
```

**Step 2: Change Photon.update to accumulate dt**

In `src/photon.js`, replace the `update` method:

```js
update(dt) {
    this.pos.x += this.vel.x * dt;
    this.pos.y += this.vel.y * dt;
    this.lifetime += dt;
}
```

**Step 3: Verify**

- Load Magnetic preset (key 5), enable Radiation toggle
- Photons should appear and fade over a consistent duration regardless of speed setting
- Change speed slider — photon visual lifetime should stay proportional to simulation time

**Step 4: Commit**

```bash
git add src/config.js src/photon.js
git commit -m "fix: time-based photon lifetime instead of frame-based"
```

---

### Task 3: Lazy History Buffers

**Files:**
- Modify: `src/particle.js:34-41` (remove eager allocation)
- Modify: `src/physics.js:283-293` (lazy init before recording)
- Modify: `src/physics.js:401` (guard histCount check)

**Step 1: Remove eager allocation from Particle constructor**

In `src/particle.js`, replace lines 34-41:

```js
// History buffers for signal delay
this.histX = new Float64Array(HISTORY_SIZE);
this.histY = new Float64Array(HISTORY_SIZE);
this.histVx = new Float64Array(HISTORY_SIZE);
this.histVy = new Float64Array(HISTORY_SIZE);
this.histTime = new Float64Array(HISTORY_SIZE);
this.histHead = 0;
this.histCount = 0;
```

with:

```js
// History buffers for signal delay (lazy-allocated)
this.histX = null;
this.histY = null;
this.histVx = null;
this.histVy = null;
this.histTime = null;
this.histHead = 0;
this.histCount = 0;
```

Remove the `HISTORY_SIZE` import from particle.js since it's no longer used there.

**Step 2: Add _initHistory method to Particle**

Add to `src/particle.js`, in the Particle class (after `updateColor`):

```js
_initHistory() {
    if (this.histX) return;
    this.histX = new Float64Array(HISTORY_SIZE);
    this.histY = new Float64Array(HISTORY_SIZE);
    this.histVx = new Float64Array(HISTORY_SIZE);
    this.histVy = new Float64Array(HISTORY_SIZE);
    this.histTime = new Float64Array(HISTORY_SIZE);
    this.histHead = 0;
    this.histCount = 0;
}
```

Re-add the `HISTORY_SIZE` import to particle.js for this method.

**Step 3: Lazy-init in physics.js signal delay recording**

In `src/physics.js`, in the drift step (around line 284), wrap the history recording with lazy init:

```js
// Record history for signal delay
if (this.signalDelayEnabled) {
    p._initHistory();
    const h = p.histHead;
    p.histX[h] = p.pos.x;
    p.histY[h] = p.pos.y;
    p.histVx[h] = p.vel.x;
    p.histVy[h] = p.vel.y;
    p.histTime[h] = this.simTime;
    p.histHead = (h + 1) % HISTORY_SIZE;
    if (p.histCount < HISTORY_SIZE) p.histCount++;
}
```

**Step 4: Guard histCount checks against null buffers**

In `src/physics.js` `_computeAllForces` (line 401), the check `o.histCount >= 2` is already safe since `histCount` defaults to 0 and is only incremented after init. But `_interpolateHistory` accesses `p.histTime[start]` — add a null guard:

In `_interpolateHistory` (line 769), add at top:
```js
if (!p.histX) return null;
```

Similarly in `_getDelayedState`, the call to `_interpolateHistory` will return null if buffers aren't initialized, which is already handled.

**Step 5: Verify**

- Load any preset — simulation should work normally (signal delay is off by default)
- Console: `sim.particles[0].histX` should be `null`
- Enable signal delay toggle, wait a moment
- Console: `sim.particles[0].histX` should now be a Float64Array
- Disable signal delay, add new particle — new particle should have `histX === null`

**Step 6: Commit**

```bash
git add src/particle.js src/physics.js
git commit -m "perf: lazy-allocate signal delay history buffers"
```

---

### Task 4: Tidal Breakup Softening

**Files:**
- Modify: `src/physics.js:916-919` (add SOFTENING_SQ to tidal distSq)

**Step 1: Add softening**

In `src/physics.js` `checkTidalBreakup`, change lines 916-919:

```js
const dx = other.pos.x - p.pos.x, dy = other.pos.y - p.pos.y;
const distSq = dx * dx + dy * dy;
const r = Math.sqrt(distSq);
const tidalAccel = TIDAL_STRENGTH * other.mass * p.radius / (r * distSq);
```

to:

```js
const dx = other.pos.x - p.pos.x, dy = other.pos.y - p.pos.y;
const distSq = dx * dx + dy * dy + SOFTENING_SQ;
const r = Math.sqrt(distSq);
const tidalAccel = TIDAL_STRENGTH * other.mass * p.radius / (r * distSq);
```

**Step 2: Verify**

- Enable tidal forces, create two massive particles near each other
- They should fragment smoothly without NaN or explosions
- Console: check no `NaN` values in particle positions after tidal events

**Step 3: Commit**

```bash
git add src/physics.js
git commit -m "fix: add softening to tidal breakup distance calculation"
```

---

### Task 5: Scale Bounce Overlap Push-out

**Files:**
- Modify: `src/physics.js:615` (scale overlap constant)

**Step 1: Replace hardcoded constant**

In `src/physics.js` `resolveBounce`, change line 615:

```js
const overlap = (minDist - safeDist) / 2 + 0.25;
```

to:

```js
const overlap = (minDist - safeDist) / 2 + minDist * 0.01;
```

**Step 2: Verify**

- Set collision mode to Bounce, spawn several particles
- Particles should bounce cleanly without sticking or tunneling
- Test with very small (mass=1) and very large (mass=100) particles

**Step 3: Commit**

```bash
git add src/physics.js
git commit -m "fix: scale bounce overlap push-out by particle size"
```

---

### Task 6: Write-Pointer Compaction for Despawn

**Files:**
- Modify: `src/physics.js:334-362` (rewrite boundary handling to use compaction)

**Step 1: Rewrite boundary despawn**

Replace the boundary handling block in `physics.js` (lines 334-362). Currently it uses `splice` inside a reverse loop for despawn. Rewrite to separate boundary effects from removal:

```js
// Step 8: Handle boundaries (once per frame, after all substeps)
let writeIdx = 0;
for (let i = 0; i < particles.length; i++) {
    const p = particles[i];
    const left = offX, top = offY;
    const right = offX + width, bottom = offY + height;

    if (boundaryMode === 'despawn') {
        if (p.pos.x < left - DESPAWN_MARGIN || p.pos.x > right + DESPAWN_MARGIN ||
            p.pos.y < top - DESPAWN_MARGIN || p.pos.y > bottom + DESPAWN_MARGIN) {
            continue; // skip — don't copy to output
        }
    } else if (boundaryMode === 'loop') {
        if (p.pos.x < left) p.pos.x += width;
        else if (p.pos.x > right) p.pos.x -= width;
        if (p.pos.y < top) p.pos.y += height;
        else if (p.pos.y > bottom) p.pos.y -= height;
    } else if (boundaryMode === 'bounce') {
        let bounced = false;
        if (p.pos.x < left + p.radius) { p.pos.x = left + p.radius; p.w.x *= -1; bounced = true; }
        else if (p.pos.x > right - p.radius) { p.pos.x = right - p.radius; p.w.x *= -1; bounced = true; }
        if (p.pos.y < top + p.radius) { p.pos.y = top + p.radius; p.w.y *= -1; bounced = true; }
        else if (p.pos.y > bottom - p.radius) { p.pos.y = bottom - p.radius; p.w.y *= -1; bounced = true; }

        if (bounced) {
            const invG = relOn ? 1 / Math.sqrt(1 + p.w.magSq()) : 1;
            p.vel.x = p.w.x * invG;
            p.vel.y = p.w.y * invG;
        }
    }

    particles[writeIdx++] = p;
}
particles.length = writeIdx;
```

**Step 2: Verify**

- Set boundary to Despawn, spawn particles near edges — they should disappear normally
- Set boundary to Loop — particles should wrap
- Set boundary to Bounce — particles should reflect
- Console: `sim.particles.length` should decrease when particles despawn

**Step 3: Commit**

```bash
git add src/physics.js
git commit -m "perf: write-pointer compaction for boundary despawn"
```

---

### Task 7: Cyclotron Frequency in Adaptive Substepping

**Files:**
- Modify: `src/physics.js:69-78` (add cyclotron criterion)

**Step 1: Add cyclotron frequency check**

After the existing acceleration-based substep calculation (lines 69-78), add cyclotron frequency criterion. Replace the block:

```js
// ─── Determine substep count from max acceleration ───
let maxAccelSq = 0;
for (let i = 0; i < n; i++) {
    const p = particles[i];
    const aSq = p.force.magSq() / (p.mass * p.mass);
    if (aSq > maxAccelSq) maxAccelSq = aSq;
}
const aMax = Math.sqrt(maxAccelSq);
const dtSafe = aMax > 0 ? Math.sqrt(SOFTENING / aMax) : dt;
const nSteps = Math.min(Math.ceil(dt / dtSafe), MAX_SUBSTEPS);
const dtSub = dt / nSteps;
```

with:

```js
// ─── Determine substep count from max acceleration + cyclotron frequency ───
let maxAccelSq = 0;
let maxCyclotron = 0;
for (let i = 0; i < n; i++) {
    const p = particles[i];
    const aSq = p.force.magSq() / (p.mass * p.mass);
    if (aSq > maxAccelSq) maxAccelSq = aSq;
    // Cyclotron frequency: ω_c = |qBz/m| for EM, |4Bgz| for GM
    if (hasMagnetic && Math.abs(p.Bz) > 0) {
        const wc = Math.abs(p.charge * p.Bz / p.mass);
        if (wc > maxCyclotron) maxCyclotron = wc;
    }
    if (hasGM && Math.abs(p.Bgz) > 0) {
        const wc = 4 * Math.abs(p.Bgz);
        if (wc > maxCyclotron) maxCyclotron = wc;
    }
}
const aMax = Math.sqrt(maxAccelSq);
let dtSafe = aMax > 0 ? Math.sqrt(SOFTENING / aMax) : dt;
// Ensure at least 8 steps per cyclotron orbit
if (maxCyclotron > 0) {
    const dtCyclotron = (2 * Math.PI / maxCyclotron) / 8;
    if (dtCyclotron < dtSafe) dtSafe = dtCyclotron;
}
const nSteps = Math.min(Math.ceil(dt / dtSafe), MAX_SUBSTEPS);
const dtSub = dt / nSteps;
```

Note: `hasMagnetic` and `hasGM` are already defined at lines 80-81 but are used here before those lines. Move the declarations before the substep block:

```js
const hasMagnetic = this.magneticEnabled;
const hasGM = this.gravitomagEnabled;
```

**Step 2: Verify**

- Load Magnetic preset (key 5), observe — particles with strong B-fields should orbit smoothly
- Console: temporarily log nSteps to confirm it increases in strong B-field scenarios
- Energy drift should improve for magnetic scenarios

**Step 3: Commit**

```bash
git add src/physics.js
git commit -m "feat: cyclotron frequency criterion in adaptive substepping"
```

---

### Task 8: QuadTree Node Pooling

**Files:**
- Rewrite: `src/quadtree.js` (pool-based architecture)
- Modify: `src/physics.js` (use pool, update tree-walk code)

**Step 1: Rewrite quadtree.js with pool**

Replace `src/quadtree.js` entirely:

```js
import { INERTIA_K, MAG_MOMENT_K } from './config.js';

export class Rect {
    constructor(x, y, w, h) {
        this.x = x; this.y = y; this.w = w; this.h = h;
    }

    contains(point) {
        return (point.x >= this.x - this.w &&
            point.x <= this.x + this.w &&
            point.y >= this.y - this.h &&
            point.y <= this.y + this.h);
    }

    intersects(range) {
        return !(range.x - range.w > this.x + this.w ||
            range.x + range.w < this.x - this.w ||
            range.y - range.h > this.y + this.h ||
            range.y + range.h < this.y - this.h);
    }
}

const NONE = -1;

export default class QuadTreePool {
    constructor(capacity = 4, maxNodes = 512) {
        this.nodeCapacity = capacity;
        this.maxNodes = maxNodes;

        // Pre-allocate flat arrays for all node fields
        this.bx = new Float64Array(maxNodes);
        this.by = new Float64Array(maxNodes);
        this.bw = new Float64Array(maxNodes);
        this.bh = new Float64Array(maxNodes);

        this.totalMass = new Float64Array(maxNodes);
        this.totalCharge = new Float64Array(maxNodes);
        this.totalMagneticMoment = new Float64Array(maxNodes);
        this.totalAngularMomentum = new Float64Array(maxNodes);
        this.totalMomentumX = new Float64Array(maxNodes);
        this.totalMomentumY = new Float64Array(maxNodes);
        this.comX = new Float64Array(maxNodes);
        this.comY = new Float64Array(maxNodes);

        // Children indices (NONE = no child)
        this.nw = new Int32Array(maxNodes).fill(NONE);
        this.ne = new Int32Array(maxNodes).fill(NONE);
        this.sw = new Int32Array(maxNodes).fill(NONE);
        this.se = new Int32Array(maxNodes).fill(NONE);

        // Leaf point storage: fixed-size per node
        // points[node * capacity + i] = particle ref (stored in separate JS array)
        this.points = new Array(maxNodes * capacity).fill(null);
        this.pointCount = new Uint8Array(maxNodes);

        this.divided = new Uint8Array(maxNodes);
        this.count = 0;
    }

    reset() {
        this.count = 0;
    }

    alloc(bx, by, bw, bh) {
        const idx = this.count++;
        if (idx >= this.maxNodes) {
            // Fallback: double the pool (rare)
            this._grow();
        }
        this.bx[idx] = bx;
        this.by[idx] = by;
        this.bw[idx] = bw;
        this.bh[idx] = bh;
        this.totalMass[idx] = 0;
        this.totalCharge[idx] = 0;
        this.totalMagneticMoment[idx] = 0;
        this.totalAngularMomentum[idx] = 0;
        this.totalMomentumX[idx] = 0;
        this.totalMomentumY[idx] = 0;
        this.comX[idx] = bx;
        this.comY[idx] = by;
        this.nw[idx] = NONE;
        this.ne[idx] = NONE;
        this.sw[idx] = NONE;
        this.se[idx] = NONE;
        this.pointCount[idx] = 0;
        this.divided[idx] = 0;
        // Clear point slots
        const base = idx * this.nodeCapacity;
        for (let i = 0; i < this.nodeCapacity; i++) this.points[base + i] = null;
        return idx;
    }

    _grow() {
        const newMax = this.maxNodes * 2;
        const copyF64 = (old) => { const a = new Float64Array(newMax); a.set(old); return a; };
        const copyI32 = (old, fill) => { const a = new Int32Array(newMax); a.set(old); a.fill(fill, this.maxNodes); return a; };
        const copyU8 = (old) => { const a = new Uint8Array(newMax); a.set(old); return a; };

        this.bx = copyF64(this.bx); this.by = copyF64(this.by);
        this.bw = copyF64(this.bw); this.bh = copyF64(this.bh);
        this.totalMass = copyF64(this.totalMass); this.totalCharge = copyF64(this.totalCharge);
        this.totalMagneticMoment = copyF64(this.totalMagneticMoment);
        this.totalAngularMomentum = copyF64(this.totalAngularMomentum);
        this.totalMomentumX = copyF64(this.totalMomentumX);
        this.totalMomentumY = copyF64(this.totalMomentumY);
        this.comX = copyF64(this.comX); this.comY = copyF64(this.comY);
        this.nw = copyI32(this.nw, NONE); this.ne = copyI32(this.ne, NONE);
        this.sw = copyI32(this.sw, NONE); this.se = copyI32(this.se, NONE);
        this.pointCount = copyU8(this.pointCount);
        this.divided = copyU8(this.divided);

        const newPoints = new Array(newMax * this.nodeCapacity).fill(null);
        for (let i = 0; i < this.maxNodes * this.nodeCapacity; i++) newPoints[i] = this.points[i];
        this.points = newPoints;

        this.maxNodes = newMax;
    }

    // ─── Tree operations (take node index as first arg) ───

    _contains(idx, px, py) {
        return (px >= this.bx[idx] - this.bw[idx] &&
            px <= this.bx[idx] + this.bw[idx] &&
            py >= this.by[idx] - this.bh[idx] &&
            py <= this.by[idx] + this.bh[idx]);
    }

    _intersects(idx, rx, ry, rw, rh) {
        return !(rx - rw > this.bx[idx] + this.bw[idx] ||
            rx + rw < this.bx[idx] - this.bw[idx] ||
            ry - rh > this.by[idx] + this.bh[idx] ||
            ry + rh < this.by[idx] - this.bh[idx]);
    }

    _subdivide(idx) {
        const x = this.bx[idx], y = this.by[idx];
        const hw = this.bw[idx] / 2, hh = this.bh[idx] / 2;
        this.nw[idx] = this.alloc(x - hw, y - hh, hw, hh);
        this.ne[idx] = this.alloc(x + hw, y - hh, hw, hh);
        this.sw[idx] = this.alloc(x - hw, y + hh, hw, hh);
        this.se[idx] = this.alloc(x + hw, y + hh, hw, hh);
        this.divided[idx] = 1;
    }

    insert(idx, particle) {
        if (!this._contains(idx, particle.pos.x, particle.pos.y)) return false;

        const cap = this.nodeCapacity;
        if (this.pointCount[idx] < cap && !this.divided[idx]) {
            this.points[idx * cap + this.pointCount[idx]] = particle;
            this.pointCount[idx]++;
            return true;
        }

        if (!this.divided[idx]) {
            this._subdivide(idx);
            const base = idx * cap;
            for (let i = 0; i < this.pointCount[idx]; i++) {
                const p = this.points[base + i];
                this.insert(this.nw[idx], p) ||
                    this.insert(this.ne[idx], p) ||
                    this.insert(this.sw[idx], p) ||
                    this.insert(this.se[idx], p);
                this.points[base + i] = null;
            }
            this.pointCount[idx] = 0;
        }

        return this.insert(this.nw[idx], particle) ||
            this.insert(this.ne[idx], particle) ||
            this.insert(this.sw[idx], particle) ||
            this.insert(this.se[idx], particle);
    }

    calculateMassDistribution(idx) {
        if (!this.divided[idx]) {
            const cnt = this.pointCount[idx];
            if (cnt === 0) return;

            let mass = 0, charge = 0, magMom = 0, angMom = 0;
            let cx = 0, cy = 0, momX = 0, momY = 0;
            const base = idx * this.nodeCapacity;

            for (let i = 0; i < cnt; i++) {
                const p = this.points[base + i];
                const rSq = p.radius * p.radius;
                mass += p.mass;
                charge += p.charge;
                magMom += MAG_MOMENT_K * p.charge * p.angVel * rSq;
                angMom += INERTIA_K * p.mass * p.angVel * rSq;
                cx += p.pos.x * p.mass;
                cy += p.pos.y * p.mass;
                momX += p.mass * p.w.x;
                momY += p.mass * p.w.y;
            }

            this.totalMass[idx] = mass;
            this.totalCharge[idx] = charge;
            this.totalMagneticMoment[idx] = magMom;
            this.totalAngularMomentum[idx] = angMom;
            this.totalMomentumX[idx] = momX;
            this.totalMomentumY[idx] = momY;
            if (mass > 0) { this.comX[idx] = cx / mass; this.comY[idx] = cy / mass; }
        } else {
            const children = [this.nw[idx], this.ne[idx], this.sw[idx], this.se[idx]];
            for (const c of children) this.calculateMassDistribution(c);

            let mass = 0, charge = 0, magMom = 0, angMom = 0;
            let cx = 0, cy = 0, momX = 0, momY = 0;

            for (const c of children) {
                mass += this.totalMass[c];
                charge += this.totalCharge[c];
                magMom += this.totalMagneticMoment[c];
                angMom += this.totalAngularMomentum[c];
                cx += this.comX[c] * this.totalMass[c];
                cy += this.comY[c] * this.totalMass[c];
                momX += this.totalMomentumX[c];
                momY += this.totalMomentumY[c];
            }

            this.totalMass[idx] = mass;
            this.totalCharge[idx] = charge;
            this.totalMagneticMoment[idx] = magMom;
            this.totalAngularMomentum[idx] = angMom;
            this.totalMomentumX[idx] = momX;
            this.totalMomentumY[idx] = momY;
            if (mass > 0) { this.comX[idx] = cx / mass; this.comY[idx] = cy / mass; }
        }
    }

    query(idx, rx, ry, rw, rh, found) {
        if (!found) found = [];
        if (!this._intersects(idx, rx, ry, rw, rh)) return found;

        if (!this.divided[idx]) {
            const base = idx * this.nodeCapacity;
            for (let i = 0; i < this.pointCount[idx]; i++) {
                const p = this.points[base + i];
                if (p.pos.x >= rx - rw && p.pos.x <= rx + rw &&
                    p.pos.y >= ry - rh && p.pos.y <= ry + rh) {
                    found.push(p);
                }
            }
        } else {
            this.query(this.nw[idx], rx, ry, rw, rh, found);
            this.query(this.ne[idx], rx, ry, rw, rh, found);
            this.query(this.sw[idx], rx, ry, rw, rh, found);
            this.query(this.se[idx], rx, ry, rw, rh, found);
        }

        return found;
    }

    // ─── Convenience: build full tree from particles ───
    build(bx, by, bw, bh, particles) {
        this.reset();
        const root = this.alloc(bx, by, bw, bh);
        for (const p of particles) this.insert(root, p);
        this.calculateMassDistribution(root);
        return root;
    }
}
```

**Step 2: Update physics.js to use pool**

This requires updating every reference to the old QuadTree API. Key changes:

- Import `QuadTreePool` instead of `QuadTree, { Rect }`; still import `{ Rect }` for collision range queries
- Add `this.pool = new QuadTreePool(QUADTREE_CAPACITY)` to Physics constructor
- Replace `new QuadTree(boundary, cap)` + insert loop + calculateMassDistribution with `this.pool.build(bx, by, bw, bh, particles)` returning a root index
- Update `calculateForce` tree walk to use `pool.fieldName[nodeIdx]` instead of `node.fieldName`
- Update `_treePE` similarly
- Update `handleCollisions` to use `pool.query(root, rx, ry, rw, rh)`
- Update `_computeAllForces` Barnes-Hut path to pass root index

The changes are mechanical — same logic, different access syntax (pool arrays vs object properties).

**Step 3: Verify**

- Load all presets, toggle Barnes-Hut on/off — behavior should be identical
- Set collision mode to Bounce and Merge — collisions should work
- Console: `sim.physics.pool.count` should show node count per build (varies, typically 4N)
- Performance: open devtools Performance tab, record 5 seconds — GC pauses should be reduced vs before

**Step 4: Commit**

```bash
git add src/quadtree.js src/physics.js
git commit -m "perf: pool-based QuadTree eliminates per-frame allocations"
```

---

### Task 9: Physics Module Split

**Files:**
- Create: `src/forces.js`
- Create: `src/collisions.js`
- Create: `src/potential.js`
- Create: `src/signal-delay.js`
- Rename: `src/physics.js` → `src/integrator.js`
- Modify: `main.js` (update import path)

**Step 1: Create src/forces.js**

Extract from physics.js:
- `resetForces(particles)` (was `_resetForces`)
- `computeAllForces(particles, toggles, pool, root, barnesHutEnabled, signalDelayEnabled, relativityEnabled, simTime)` (was `_computeAllForces`)
- `pairForce(p, sx, sy, svx, svy, sMass, sCharge, sAngVel, sMagMoment, sAngMomentum, out, toggles)` (was `_pairForce`)
- `calculateForce(particle, pool, nodeIdx, theta, out, toggles)` (was `calculateForce`)

All become exported functions. `toggles` is an object reference `{ gravityEnabled, coulombEnabled, magneticEnabled, gravitomagEnabled }` passed from Physics instance.

**Step 2: Create src/collisions.js**

Extract from physics.js:
- `handleCollisions(particles, pool, root, mode, bounceFriction, relativityEnabled)`
- `resolveMerge(p1, p2, relativityEnabled)`
- `resolveBounce(p1, p2, minDist, dist, bounceFriction, relativityEnabled)`

All exported functions. Import `INERTIA_K`, `MAG_MOMENT_K` from config, `setVelocity, angwToAngVel, angVelToAngw` from relativity.

**Step 3: Create src/potential.js**

Extract from physics.js:
- `computePE(particles, toggles, pool, root, barnesHutEnabled, bhTheta)`
- `treePE(particle, pool, nodeIdx, theta, toggles)`
- `pairPE(p, sx, sy, sMass, sCharge, sAngVel, sMagMoment, sAngMomentum, toggles)`

All exported functions.

**Step 4: Create src/signal-delay.js**

Extract from physics.js:
- `getDelayedState(source, observer, simTime)`
- `interpolateHistory(p, t)`

Exported functions. Import `HISTORY_SIZE` from config.

**Step 5: Slim down physics.js → integrator.js**

Rename file. The Physics class keeps:
- Constructor (toggles, pool, sim ref, simTime, potentialEnergy, _forcesInit)
- `update()` method (substep loop, Boris rotation, spin-orbit, frame-drag, radiation, drift, boundary)
- `checkTidalBreakup()`
- Imports from the new modules

**Step 6: Update main.js import**

```js
import Physics from './src/integrator.js';
```

**Step 7: Verify**

- Full regression: load all 5 presets, toggle every force on/off, all collision/boundary modes
- Energy conservation should be identical to before the split
- Console: `sim.physics` should still work as before

**Step 8: Commit**

```bash
git add src/forces.js src/collisions.js src/potential.js src/signal-delay.js src/integrator.js main.js
git rm src/physics.js
git commit -m "refactor: split physics.js into integrator/forces/collisions/potential/signal-delay modules"
```

---

### Task 10: Stats Display Extraction

**Files:**
- Create: `src/stats-display.js`
- Modify: `main.js` (remove inline energy/stats methods, use new module)

**Step 1: Create src/stats-display.js**

```js
import { computeEnergies } from './energy.js';

export default class StatsDisplay {
    constructor(dom, selDom, sankey) {
        this.dom = dom;
        this.selDom = selDom;
        this.sankey = sankey;
        this.initialEnergy = null;
        this.initialMomentum = null;
        this.initialAngMom = null;
    }

    resetBaseline() {
        this.initialEnergy = null;
        this.initialMomentum = null;
        this.initialAngMom = null;
    }

    updateEnergy(particles, physics, sim) {
        const e = computeEnergies(particles, physics, sim);
        const angMom = e.orbitalAngMom + e.spinAngMom;

        const totalPx = e.px + e.fieldPx + sim.totalRadiatedPx;
        const totalPy = e.py + e.fieldPy + sim.totalRadiatedPy;
        const pMag = Math.sqrt(totalPx * totalPx + totalPy * totalPy);
        const total = e.linearKE + e.spinKE + e.pe + e.fieldEnergy + sim.totalRadiated;

        if (this.initialEnergy === null && particles.length > 0) {
            this.initialEnergy = total;
            this.initialMomentum = pMag;
            this.initialAngMom = angMom;
        }

        const eDrift = this.initialEnergy !== null && this.initialEnergy !== 0
            ? ((total - this.initialEnergy) / Math.abs(this.initialEnergy) * 100) : 0;
        const pDrift = this.initialMomentum !== null && this.initialMomentum !== 0
            ? ((pMag - this.initialMomentum) / Math.abs(this.initialMomentum) * 100) : 0;
        const aDrift = this.initialAngMom !== null && this.initialAngMom !== 0
            ? ((angMom - this.initialAngMom) / Math.abs(this.initialAngMom) * 100) : 0;

        const fmt = (v) => Math.abs(v) < 0.01 ? '0' : Math.abs(v) > 999 ? v.toExponential(1) : v.toFixed(1);
        const fmtDrift = (v) => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';

        this.dom.linearKE.textContent = fmt(e.linearKE);
        this.dom.spinKE.textContent = fmt(e.spinKE);
        this.dom.potentialE.textContent = fmt(e.pe);
        this.dom.totalE.textContent = fmt(total);
        this.dom.energyDrift.textContent = fmtDrift(eDrift);
        this.dom.fieldE.textContent = fmt(e.fieldEnergy);
        this.dom.radiatedE.textContent = fmt(sim.totalRadiated);
        this.sankey.update(e.linearKE, e.spinKE, e.pe, e.fieldEnergy, sim.totalRadiated);
        this.dom.momentum.textContent = fmt(pMag);
        this.dom.fieldMom.textContent = fmt(Math.sqrt(e.fieldPx * e.fieldPx + e.fieldPy * e.fieldPy));
        this.dom.radiatedMom.textContent = fmt(Math.sqrt(sim.totalRadiatedPx * sim.totalRadiatedPx + sim.totalRadiatedPy * sim.totalRadiatedPy));
        this.dom.momentumDrift.textContent = fmtDrift(pDrift);
        this.dom.angularMomentum.textContent = fmt(angMom);
        this.dom.orbitalAngMom.textContent = fmt(e.orbitalAngMom);
        this.dom.spinAngMom.textContent = fmt(e.spinAngMom);
        this.dom.angMomDrift.textContent = fmtDrift(aDrift);
    }

    updateSelected(particle, particles, physics) {
        const p = particle;
        const dom = this.selDom;

        if (!p || !particles.includes(p)) {
            dom.details.hidden = true;
            dom.hint.hidden = false;
            dom.phaseSection.hidden = true;
            return null; // signal to caller to clear selection
        }

        dom.details.hidden = false;
        dom.hint.hidden = true;
        dom.phaseSection.hidden = false;
        const fmt = (v) => Math.abs(v) < 0.01 ? '0' : Math.abs(v) > 999 ? v.toExponential(1) : v.toFixed(2);
        const speed = Math.sqrt(p.vel.x * p.vel.x + p.vel.y * p.vel.y);
        const gamma = physics.relativityEnabled ? Math.sqrt(1 + p.w.magSq()) : 1;
        const totalFx = p.forceGravity.x + p.forceCoulomb.x + p.forceMagnetic.x + p.forceGravitomag.x;
        const totalFy = p.forceGravity.y + p.forceCoulomb.y + p.forceMagnetic.y + p.forceGravitomag.y;
        const forceMag = Math.sqrt(totalFx * totalFx + totalFy * totalFy);

        dom.id.textContent = p.id;
        dom.mass.textContent = fmt(p.mass);
        dom.charge.textContent = fmt(p.charge);
        const surfaceV = p.angVel * p.radius;
        dom.spin.textContent = surfaceV.toFixed(4) + 'c';
        dom.speed.textContent = speed.toFixed(4) + 'c';
        dom.gamma.textContent = gamma.toFixed(3);
        dom.force.textContent = fmt(forceMag);
        return p; // still valid
    }
}
```

**Step 2: Update main.js**

- Import `StatsDisplay` from `./src/stats-display.js`
- Create `this.stats = new StatsDisplay(this.dom, this.selDom, this.sankey)` in constructor
- Remove `this.dom` and `this.selDom` from constructor (moved to StatsDisplay)
- Remove `this.initialEnergy`, `this.initialMomentum`, `this.initialAngMom`
- Remove `computeEnergy()` and `updateSelectedParticle()` methods
- In `loop()`: replace `this.computeEnergy()` with `this.stats.updateEnergy(this.particles, this.physics, this)`
- In `loop()`: replace `this.updateSelectedParticle()` with:
  ```js
  const sel = this.stats.updateSelected(this.selectedParticle, this.particles, this.physics);
  if (!sel && this.selectedParticle) this.selectedParticle = null;
  ```
- In `addParticle()`: replace `this.initialEnergy = null; this.initialMomentum = null; this.initialAngMom = null;` with `this.stats.resetBaseline()`
- In clear button handler (ui.js): replace the three null assignments with `sim.stats.resetBaseline()`
- Keep `this.dom.speedInput` reference in Simulation since it's used by the speed slider (or pass it to StatsDisplay too)

**Step 3: Verify**

- All stats should display correctly in the sidebar
- Energy drift should track properly
- Adding particles / clearing should reset baseline
- Selected particle info should update

**Step 4: Commit**

```bash
git add src/stats-display.js main.js src/ui.js
git commit -m "refactor: extract stats display into dedicated module"
```

---

### Task 11: Final Regression Test

**Files:** None (verification only)

**Step 1: Full preset regression**

Load each preset (keys 1-5), run for 30 seconds each:
- Solar System: stable orbits, energy drift < 1%
- Binary Stars: stable orbit, spinning particles
- Galaxy: no explosions, particles orbit core
- Collision: clean bounce/merge depending on mode
- Magnetic: charged particles interact, no NaN

**Step 2: Toggle regression**

For each preset, toggle every force on/off, every collision mode, every boundary mode. No crashes or NaN.

**Step 3: Feature regression**

- Signal delay: enable with relativity on, BH off. Ghost circles appear.
- Radiation: enable with relativity on. Photons spawn and fade.
- Tidal: enable, create large + small particles nearby. Fragmentation works.
- Heatmap: enable potential field overlay. Colors render.
- All visual toggles: velocity vectors, force vectors, force components, trails.

**Step 4: Performance check**

- DevTools Performance tab: record 10 seconds with Galaxy preset
- GC pauses should be minimal (< 5ms)
- Frame rate should stay near 60fps

**Step 5: Commit docs update**

Update `CLAUDE.md` module graph to reflect new file structure.

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md module graph for new file structure"
```
