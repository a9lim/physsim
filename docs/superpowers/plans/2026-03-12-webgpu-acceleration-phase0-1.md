# WebGPU Acceleration — Phase 0 & 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish backend abstraction layer (Phase 0) and build the minimal GPU scaffold — SoA buffers, WebGPU device init, simplest compute shaders, instanced particle rendering (Phase 1).

**Architecture:** Phase 0 creates `CPUPhysics` and `CanvasRenderer` wrappers around the existing code, with backend detection in `main.js`. No file moves yet — the wrappers delegate to existing classes in-place. Phase 1 builds the GPU backend alongside: `GPUPhysics` (buffer management + compute dispatches) and `GPURenderer` (instanced WebGPU draws). By end of Phase 1, GPU path shows particles that drift in straight lines, wrap at boundaries, and render as circles.

**Tech Stack:** WebGPU API, WGSL shaders, vanilla JS ES6 modules. No build tools, no npm, no bundler.

**Spec:** `docs/superpowers/specs/2026-03-12-webgpu-acceleration-design.md`

**Testing:** This project has no test framework (zero-dependency vanilla JS). Verification is browser-based: open `python -m http.server` from `a9lim.github.io/`, navigate to `/physsim/`, check browser console for errors, compare visual behavior.

**Deviation from spec:** The spec's Phase 0 says "move CPU files to `src/cpu/` with `cpu-` prefix." This plan defers the file move to after the GPU path is working. Reason: moving 17 files and updating 50+ import paths before validating the GPU path risks a large, hard-to-debug breakage. Instead, Phase 0 creates thin wrappers around the existing classes in-place. The file move happens in a future phase once both backends are validated side-by-side.

---

## Chunk 1: Phase 0 — Backend Abstraction Layer

### Task 1: Create shared backend interface definition

**Files:**
- Create: `src/backend-interface.js`

This file documents the contract that both `CPUPhysics`/`GPUPhysics` and `CanvasRenderer`/`GPURenderer` must implement. It exports no runtime code — just JSDoc type definitions that serve as documentation.

- [ ] **Step 1: Create the interface file**

Create `src/backend-interface.js`:

```js
/**
 * @fileoverview Backend interface contracts for physics and rendering.
 *
 * Both CPU and GPU backends implement these interfaces.
 * The UI layer (ui.js, input.js, save-load.js) interacts only through these methods.
 */

/**
 * @typedef {Object} ParticleState
 * @property {number} x
 * @property {number} y
 * @property {number} vx
 * @property {number} vy
 * @property {number} mass
 * @property {number} charge
 * @property {number} angVel
 * @property {number} radius
 * @property {boolean} antimatter
 */

/**
 * @typedef {Object} SimStats
 * @property {number} ke - Kinetic energy
 * @property {number} pe - Potential energy
 * @property {number} px - Momentum x
 * @property {number} py - Momentum y
 * @property {number} angL - Angular momentum
 * @property {number} drift - Energy drift from initial
 * @property {number} particleCount
 */

/**
 * @typedef {Object} PhysicsBackend
 * @property {function(number): void} update - Advance simulation by dt
 * @property {function(): number} getParticleCount
 * @property {function(number): ParticleState} getParticleState
 * @property {function(): SimStats} getStats
 * @property {function(Object): number} addParticle - Returns slot index
 * @property {function(number): void} removeParticle
 * @property {function(number): Array<{label: string, x: number, y: number}>} getSelectedForceBreakdown - 11 force vectors
 * @property {function(string): Float32Array} getFieldData - Field overlay data
 * @property {function(Object): void} setUniforms - Update toggles/sliders
 * @property {function(): Object} serialize
 * @property {function(Object): void} deserialize
 * @property {function(): void} reset
 * @property {function(string): void} loadPreset
 */

/**
 * @typedef {Object} RenderBackend
 * @property {function(): void} render
 * @property {function(number, number): void} resize
 * @property {function(boolean): void} setTheme
 */

export const BACKEND_CPU = 'cpu';
export const BACKEND_GPU = 'gpu';
```

- [ ] **Step 2: Verify file loads without errors**

Open browser console, check: `import('./src/backend-interface.js')` resolves without errors.

- [ ] **Step 3: Commit**

```bash
git add src/backend-interface.js
git commit -m "feat: add backend interface contract for CPU/GPU abstraction"
```

---

### Task 2: Create CPUPhysics wrapper

**Files:**
- Create: `src/cpu-physics.js`
- Read (not modify): `src/integrator.js`, `main.js`

The wrapper delegates to the existing `Physics` class without modifying it. It adapts the current scattered-state API (where `main.js` owns `particles[]`, `photons[]`, etc.) into the shared interface. For Phase 0, this wrapper is intentionally thin — it holds references to the existing simulation state.

- [ ] **Step 1: Create `src/cpu-physics.js`**

```js
/**
 * @fileoverview CPUPhysics — wraps existing Physics class (integrator.js)
 * to conform to the shared PhysicsBackend interface.
 *
 * This is a thin adapter. The actual physics code stays in integrator.js unchanged.
 * The wrapper exists so main.js can swap between CPU and GPU backends.
 */
import Physics from './integrator.js';

export default class CPUPhysics {
    /**
     * @param {Physics} engine - Existing Physics instance to wrap.
     */
    constructor(engine) {
        /** @type {Physics} */
        this.engine = engine;
    }

    /**
     * Expose the underlying Physics engine for code that still needs direct access
     * during the migration period (ui.js, save-load.js, etc.).
     * This will be removed once all callers use the shared interface.
     */
    get _engine() { return this.engine; }
}
```

Note: this is intentionally minimal. The wrapper accepts an existing `Physics` instance rather than creating a new one, avoiding a duplicate allocation. The full interface methods (`update`, `getStats`, etc.) will be added iteratively as `main.js` is refactored to use them. For Phase 0, the wrapper just provides `._engine` passthrough so existing code keeps working.

- [ ] **Step 2: Verify import**

In browser console: `import('./src/cpu-physics.js').then(m => console.log(m.default))` — should log the class.

- [ ] **Step 3: Commit**

```bash
git add src/cpu-physics.js
git commit -m "feat: add CPUPhysics wrapper for backend abstraction"
```

---

### Task 3: Create CanvasRenderer wrapper

**Files:**
- Create: `src/canvas-renderer.js`
- Read (not modify): `src/renderer.js`

Same pattern as CPUPhysics — thin wrapper around existing Renderer.

- [ ] **Step 1: Create `src/canvas-renderer.js`**

```js
/**
 * @fileoverview CanvasRenderer — wraps existing Renderer class (renderer.js)
 * to conform to the shared RenderBackend interface.
 */
import Renderer from './renderer.js';

export default class CanvasRenderer {
    /**
     * @param {CanvasRenderingContext2D} ctx
     * @param {number} width
     * @param {number} height
     */
    constructor(ctx, width, height) {
        /** @type {Renderer} */
        this.engine = new Renderer(ctx, width, height);
    }

    /** Expose underlying Renderer during migration period. */
    get _engine() { return this.engine; }
}
```

- [ ] **Step 2: Verify import**

In browser console: `import('./src/canvas-renderer.js').then(m => console.log(m.default))` — should log the class.

- [ ] **Step 3: Commit**

```bash
git add src/canvas-renderer.js
git commit -m "feat: add CanvasRenderer wrapper for backend abstraction"
```

---

### Task 4: Add backend detection to main.js

**Files:**
- Modify: `main.js`

Add WebGPU feature detection. For Phase 0, always selects CPU — GPU path doesn't exist yet. The detection function is async and will be awaited in the constructor.

- [ ] **Step 1: Add `selectBackend()` function**

Add at the top of `main.js`, after the existing imports and before the `Simulation` class:

```js
import { BACKEND_CPU, BACKEND_GPU } from './src/backend-interface.js';
import CPUPhysics from './src/cpu-physics.js';
import CanvasRenderer from './src/canvas-renderer.js';

/**
 * Detect WebGPU support and return the best available backend.
 * @returns {Promise<{backend: string, device?: GPUDevice}>}
 */
async function selectBackend() {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
        return { backend: BACKEND_CPU };
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return { backend: BACKEND_CPU };

        const device = await adapter.requestDevice({
            // Only request the WebGPU-guaranteed minimums for detection.
            // Phase 1 shaders need ≤8 storage buffers per stage.
            // Actual limits will be validated when pipelines are created.
        });
        if (!device) return { backend: BACKEND_CPU };

        return { backend: BACKEND_GPU, device };
    } catch (e) {
        console.warn('WebGPU detection failed:', e);
        return { backend: BACKEND_CPU };
    }
}
```

- [ ] **Step 2: Wire backend detection into Simulation constructor**

In the `Simulation` class constructor, after `this.physics = new Physics();` and `this.renderer = new Renderer(...)`, add:

```js
        // Backend detection (async, completes after first frame)
        this.backend = BACKEND_CPU;
        this._cpuPhysics = new CPUPhysics(this.physics);
        this._canvasRenderer = new CanvasRenderer(this.ctx, this.width, this.height);
        // GPU backend will be initialized in Phase 1
```

Note: for Phase 0, the existing `this.physics` and `this.renderer` continue to be used directly. The wrappers are created alongside but not yet wired into the loop. This is intentional — we're establishing the scaffolding without disrupting anything.

- [ ] **Step 3: Add backend indicator to console**

At the end of the constructor, after all initialization:

```js
        selectBackend().then(({ backend, device }) => {
            this.backend = backend;
            this._gpuDevice = device || null;
            console.log(`[physsim] Backend: ${backend}${device ? ' (WebGPU available)' : ''}`);
        });
```

- [ ] **Step 4: Verify**

Serve site: `cd path/to/a9lim.github.io && python -m http.server`
Open `/physsim/` in Chrome. Check console for:
- `[physsim] Backend: gpu (WebGPU available)` on Chrome 113+
- `[physsim] Backend: cpu` on Firefox/Safari without WebGPU
- Simulation works identically to before (no behavioral changes)

- [ ] **Step 5: Commit**

```bash
git add main.js
git commit -m "feat: add WebGPU backend detection and backend abstraction wrappers"
```

---

### Task 5: Verify Phase 0 complete

- [ ] **Step 1: Full regression check**

Open the site. Verify each of these works:
1. Default preset loads and runs
2. All 19 presets load correctly (press 1-9, click preset menu)
3. Theme toggle (light/dark) works
4. Pause/play/step/reset work
5. Click to spawn particles, right-click for antimatter
6. Tab switching (Settings/Engine/Stats/Particle) works
7. Save/load (quick save Ctrl+S, quick load Ctrl+L) works
8. Camera zoom/pan works
9. Console shows `[physsim] Backend: gpu (WebGPU available)` in Chrome
10. No new console errors

- [ ] **Step 2: Commit Phase 0 complete tag**

```bash
git commit --allow-empty -m "chore: Phase 0 complete — backend abstraction layer established"
```

---

## Chunk 2: Phase 1 — GPU Scaffold

### Task 6: Create GPU directory structure and buffer allocation module

**Files:**
- Create: `src/gpu/gpu-buffers.js`
- Create: `src/gpu/shaders/` (directory)

This module handles SoA buffer creation, the particle pool's free stack, and staging buffers for readback. It's the GPU equivalent of the scattered particle properties in `Particle` class.

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/gpu/shaders
```

- [ ] **Step 2: Create `src/gpu/gpu-buffers.js`**

```js
/**
 * @fileoverview GPU buffer allocation for SoA particle state.
 *
 * Creates and manages all GPUBuffer instances for the particle system.
 * Buffers are fixed-capacity (MAX_PARTICLES), indexed by particle slot.
 */

/** @param {GPUDevice} device */
export function createParticleBuffers(device, maxParticles) {
    const FLOAT_SIZE = 4;
    const UINT_SIZE = 4;

    /**
     * Helper: create a storage buffer with COPY_SRC for readback.
     * @param {string} label
     * @param {number} elementSize - bytes per element
     * @param {number} count
     */
    function storageBuffer(label, elementSize, count) {
        return device.createBuffer({
            label,
            size: elementSize * count,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
    }

    // Core state (f32 per particle)
    const posX = storageBuffer('posX', FLOAT_SIZE, maxParticles);
    const posY = storageBuffer('posY', FLOAT_SIZE, maxParticles);
    const velWX = storageBuffer('velWX', FLOAT_SIZE, maxParticles);
    const velWY = storageBuffer('velWY', FLOAT_SIZE, maxParticles);
    const angW = storageBuffer('angW', FLOAT_SIZE, maxParticles);
    const mass = storageBuffer('mass', FLOAT_SIZE, maxParticles);
    const baseMass = storageBuffer('baseMass', FLOAT_SIZE, maxParticles);
    const charge = storageBuffer('charge', FLOAT_SIZE, maxParticles);

    // Derived/cached
    const radius = storageBuffer('radius', FLOAT_SIZE, maxParticles);
    const gamma = storageBuffer('gamma', FLOAT_SIZE, maxParticles);

    // Particle metadata
    const flags = storageBuffer('flags', UINT_SIZE, maxParticles);
    const color = storageBuffer('color', UINT_SIZE, maxParticles);

    // Pool management: aliveCount + freeStack + freeTop
    // Packed into one buffer: [aliveCount: u32, freeTop: u32, freeStack: u32[maxParticles]]
    const poolMgmt = storageBuffer('poolMgmt', UINT_SIZE, maxParticles + 2);

    // Stats readback buffer (small, double-buffered)
    const statsBuffer = device.createBuffer({
        label: 'statsBuffer',
        size: 64, // 16 floats
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const statsStagingA = device.createBuffer({
        label: 'statsStagingA',
        size: 64,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const statsStagingB = device.createBuffer({
        label: 'statsStagingB',
        size: 64,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    return {
        maxParticles,
        // Core state
        posX, posY, velWX, velWY, angW, mass, baseMass, charge,
        // Derived
        radius, gamma,
        // Metadata
        flags, color,
        // Pool
        poolMgmt,
        // Stats
        statsBuffer, statsStagingA, statsStagingB,

        /** Destroy all buffers */
        destroy() {
            for (const key of Object.keys(this)) {
                if (this[key] && typeof this[key].destroy === 'function') {
                    this[key].destroy();
                }
            }
        }
    };
}

/**
 * Create the uniform buffer for simulation parameters.
 * Layout matches SimUniforms struct in common.wgsl.
 */
export function createUniformBuffer(device) {
    // 256 bytes is enough for all uniforms with padding
    return device.createBuffer({
        label: 'simUniforms',
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
}

/**
 * Write simulation uniforms to the GPU buffer.
 * @param {GPUDevice} device
 * @param {GPUBuffer} buffer
 * @param {Object} params - Simulation parameters
 */
export function writeUniforms(device, buffer, params) {
    // Pack into Float32Array matching WGSL struct layout
    const data = new ArrayBuffer(256);
    const f = new Float32Array(data);
    const u = new Uint32Array(data);

    f[0] = params.dt || 0;
    f[1] = params.simTime || 0;
    f[2] = params.domainW || 1;
    f[3] = params.domainH || 1;
    f[4] = params.speedScale || 1;
    f[5] = params.softening || 8;
    f[6] = params.softeningSq || 64;
    u[7] = params.toggles0 || 0;     // toggle bitfield 0
    u[8] = params.toggles1 || 0;     // toggle bitfield 1
    f[9] = params.yukawaCoupling || 14;
    f[10] = params.yukawaMu || 0.15;
    f[11] = params.higgsMass || 0.5;
    f[12] = params.axionMass || 0.05;
    u[13] = params.boundaryMode || 0;
    u[14] = params.topologyMode || 0;
    u[15] = params.collisionMode || 0;
    u[16] = params.maxParticles || 4096;
    u[17] = params.aliveCount || 0;

    device.queue.writeBuffer(buffer, 0, data);
}
```

- [ ] **Step 3: Verify module import**

In browser console: `import('./src/gpu/gpu-buffers.js').then(m => console.log(Object.keys(m)))` — should log `['createParticleBuffers', 'createUniformBuffer', 'writeUniforms']`.

- [ ] **Step 4: Commit**

```bash
git add src/gpu/gpu-buffers.js
git commit -m "feat: add GPU SoA buffer allocation module"
```

---

### Task 7: Create common WGSL shader and drift compute shader

**Files:**
- Create: `src/gpu/shaders/common.wgsl`
- Create: `src/gpu/shaders/drift.wgsl`

These are the first WGSL shaders. `common.wgsl` defines the uniform struct and shared constants. `drift.wgsl` is the simplest compute shader — it moves particles by `vel * dt` (no forces, no Boris rotation, just position update). This lets us verify the entire compute pipeline works end-to-end.

- [ ] **Step 1: Create `src/gpu/shaders/common.wgsl`**

```wgsl
// Common structs and constants shared across all compute/render shaders.
// This file is prepended to other shaders before compilation.

struct SimUniforms {
    dt: f32,
    simTime: f32,
    domainW: f32,
    domainH: f32,
    speedScale: f32,
    softening: f32,
    softeningSq: f32,
    toggles0: u32,
    toggles1: u32,
    yukawaCoupling: f32,
    yukawaMu: f32,
    higgsMass: f32,
    axionMass: f32,
    boundaryMode: u32,
    topologyMode: u32,
    collisionMode: u32,
    maxParticles: u32,
    aliveCount: u32,
};

// Toggle bit constants (toggles0)
const GRAVITY_BIT: u32       = 1u;
const COULOMB_BIT: u32       = 2u;
const MAGNETIC_BIT: u32      = 4u;
const GRAVITOMAG_BIT: u32    = 8u;
const ONE_PN_BIT: u32        = 16u;
const RELATIVITY_BIT: u32    = 32u;
const SPIN_ORBIT_BIT: u32    = 64u;
const RADIATION_BIT: u32     = 128u;
const BLACK_HOLE_BIT: u32    = 256u;
const DISINTEGRATION_BIT: u32 = 512u;
const EXPANSION_BIT: u32     = 1024u;
const YUKAWA_BIT: u32        = 2048u;
const HIGGS_BIT: u32         = 4096u;
const AXION_BIT: u32         = 8192u;
const BARNES_HUT_BIT: u32    = 16384u;
const BOSON_GRAV_BIT: u32    = 32768u;

// Toggle bit constants (toggles1)
const FIELD_GRAV_BIT: u32    = 1u;
const HERTZ_BOUNCE_BIT: u32  = 2u;

// Particle flag bits
const FLAG_ALIVE: u32    = 1u;
const FLAG_RETIRED: u32  = 2u;
const FLAG_ANTIMATTER: u32 = 4u;
const FLAG_BH: u32       = 8u;
const FLAG_GHOST: u32    = 16u;

// Boundary modes
const BOUND_DESPAWN: u32 = 0u;
const BOUND_BOUNCE: u32  = 1u;
const BOUND_LOOP: u32    = 2u;

// Topology modes
const TOPO_TORUS: u32 = 0u;
const TOPO_KLEIN: u32 = 1u;
const TOPO_RP2: u32   = 2u;
```

- [ ] **Step 2: Create `src/gpu/shaders/drift.wgsl`**

```wgsl
// Minimal drift shader: pos += vel * dt
// Used in Phase 1 before Boris integrator is ported.
// Velocity is NOT updated (no forces yet) — particles drift in straight lines.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> posX: array<f32>;
@group(0) @binding(2) var<storage, read_write> posY: array<f32>;
@group(0) @binding(3) var<storage, read> velWX: array<f32>;
@group(0) @binding(4) var<storage, read> velWY: array<f32>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    // Skip dead particles
    let flag = flags[idx];
    if ((flag & FLAG_ALIVE) == 0u) { return; }

    // For Phase 1: vel = w (no relativistic correction yet)
    let vx = velWX[idx];
    let vy = velWY[idx];

    posX[idx] = posX[idx] + vx * uniforms.dt;
    posY[idx] = posY[idx] + vy * uniforms.dt;
}
```

- [ ] **Step 3: Verify WGSL syntax**

These files will be validated when compiled into shader modules in Task 9. For now, visually confirm the `@group`/`@binding` indices match the buffer layout from `gpu-buffers.js`.

- [ ] **Step 4: Commit**

```bash
git add src/gpu/shaders/common.wgsl src/gpu/shaders/drift.wgsl
git commit -m "feat: add common WGSL structs and minimal drift compute shader"
```

---

### Task 7b: Create resetForces and cacheDerived compute shaders

**Files:**
- Create: `src/gpu/shaders/reset-forces.wgsl`
- Create: `src/gpu/shaders/cache-derived.wgsl`

These are trivial stubs for Phase 1 — no forces exist yet, but the spec requires them as Phase 1 deliverables. They will be expanded in Phase 2.

- [ ] **Step 1: Create `src/gpu/shaders/reset-forces.wgsl`**

```wgsl
// Zero all force/torque/bField accumulators.
// Phase 1 stub — no forces computed yet, but maintains pipeline structure.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
// Force buffers would be bound here in Phase 2+.
// For Phase 1, this is a no-op placeholder.

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // No-op in Phase 1. Forces will be added in Phase 2.
}
```

- [ ] **Step 2: Create `src/gpu/shaders/cache-derived.wgsl`**

```wgsl
// Compute derived particle properties: radius, gamma.
// Phase 1: radius = cbrt(mass), gamma = sqrt(1 + wSq).

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read> mass: array<f32>;
@group(0) @binding(2) var<storage, read> velWX: array<f32>;
@group(0) @binding(3) var<storage, read> velWY: array<f32>;
@group(0) @binding(4) var<storage, read_write> radius: array<f32>;
@group(0) @binding(5) var<storage, read_write> gamma: array<f32>;
@group(0) @binding(6) var<storage, read> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    let flag = flags[idx];
    if ((flag & FLAG_ALIVE) == 0u) { return; }

    let m = mass[idx];
    radius[idx] = pow(m, 1.0 / 3.0);  // cbrt

    let wx = velWX[idx];
    let wy = velWY[idx];
    let wSq = wx * wx + wy * wy;
    gamma[idx] = sqrt(1.0 + wSq);
}
```

- [ ] **Step 3: Commit**

```bash
git add src/gpu/shaders/reset-forces.wgsl src/gpu/shaders/cache-derived.wgsl
git commit -m "feat: add resetForces stub and cacheDerived compute shaders"
```

---

### Task 8: Create boundary wrap compute shader

**Files:**
- Create: `src/gpu/shaders/boundary.wgsl`

Handles despawn/bounce/loop for particles that leave the domain. For Phase 1, supports torus wrapping only (Klein/RP2 deferred to Phase 3).

- [ ] **Step 1: Create `src/gpu/shaders/boundary.wgsl`**

```wgsl
// Boundary wrap/bounce/despawn shader.
// Phase 1: torus wrap only. Klein/RP2 added in Phase 3.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> posX: array<f32>;
@group(0) @binding(2) var<storage, read_write> posY: array<f32>;
@group(0) @binding(3) var<storage, read_write> velWX: array<f32>;
@group(0) @binding(4) var<storage, read_write> velWY: array<f32>;
@group(0) @binding(5) var<storage, read_write> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    let flag = flags[idx];
    if ((flag & FLAG_ALIVE) == 0u) { return; }

    var x = posX[idx];
    var y = posY[idx];
    let w = uniforms.domainW;
    let h = uniforms.domainH;

    if (uniforms.boundaryMode == BOUND_LOOP) {
        // Torus wrap (Phase 1 — periodic only)
        if (x < 0.0) { x += w; }
        else if (x >= w) { x -= w; }
        if (y < 0.0) { y += h; }
        else if (y >= h) { y -= h; }
        posX[idx] = x;
        posY[idx] = y;

    } else if (uniforms.boundaryMode == BOUND_BOUNCE) {
        var vx = velWX[idx];
        var vy = velWY[idx];
        if (x < 0.0) { x = -x; vx = abs(vx); }
        else if (x >= w) { x = 2.0 * w - x; vx = -abs(vx); }
        if (y < 0.0) { y = -y; vy = abs(vy); }
        else if (y >= h) { y = 2.0 * h - y; vy = -abs(vy); }
        posX[idx] = x;
        posY[idx] = y;
        velWX[idx] = vx;
        velWY[idx] = vy;

    } else {
        // Despawn: mark particles outside domain as dead
        if (x < 0.0 || x >= w || y < 0.0 || y >= h) {
            flags[idx] = flag & ~FLAG_ALIVE;
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/gpu/shaders/boundary.wgsl
git commit -m "feat: add boundary wrap/bounce/despawn compute shader"
```

---

### Task 9: Create GPUPhysics — device init, pipeline creation, dispatch

**Files:**
- Create: `src/gpu/gpu-physics.js`

This is the GPU physics orchestrator. It creates compute pipelines from WGSL shaders, manages bind groups, and encodes command buffers for dispatch. Phase 1 only dispatches drift + boundary.

- [ ] **Step 1: Create `src/gpu/gpu-physics.js`**

```js
/**
 * @fileoverview GPUPhysics — WebGPU compute pipeline orchestrator.
 *
 * Phase 1: drift + boundary only. Forces, Boris, fields added in later phases.
 */
import { createParticleBuffers, createUniformBuffer, writeUniforms } from './gpu-buffers.js';

const MAX_PARTICLES = 4096;

export default class GPUPhysics {
    /**
     * @param {GPUDevice} device
     * @param {number} domainW
     * @param {number} domainH
     */
    constructor(device, domainW, domainH) {
        this.device = device;
        this.domainW = domainW;
        this.domainH = domainH;
        this.simTime = 0;
        this.aliveCount = 0;

        // Defaults matching CPU path
        this.boundaryMode = 0;  // despawn
        this.topologyMode = 0;  // torus

        this.buffers = createParticleBuffers(device, MAX_PARTICLES);
        this.uniformBuffer = createUniformBuffer(device);

        // Pipelines will be created in init()
        this._driftPipeline = null;
        this._boundaryPipeline = null;
        this._driftBindGroup = null;
        this._boundaryBindGroup = null;

        this._ready = false;
    }

    /** Load WGSL shaders and create compute pipelines. Must be called before update(). */
    async init() {
        const commonWGSL = await fetchShader('common.wgsl');
        const driftWGSL = await fetchShader('drift.wgsl');
        const boundaryWGSL = await fetchShader('boundary.wgsl');

        // --- Drift pipeline ---
        const driftModule = this.device.createShaderModule({
            label: 'drift',
            code: commonWGSL + '\n' + driftWGSL,
        });

        const driftBindGroupLayout = this.device.createBindGroupLayout({
            label: 'drift',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ],
        });

        this._driftPipeline = this.device.createComputePipeline({
            label: 'drift',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [driftBindGroupLayout] }),
            compute: { module: driftModule, entryPoint: 'main' },
        });

        this._driftBindGroup = this.device.createBindGroup({
            label: 'drift',
            layout: driftBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.buffers.posX } },
                { binding: 2, resource: { buffer: this.buffers.posY } },
                { binding: 3, resource: { buffer: this.buffers.velWX } },
                { binding: 4, resource: { buffer: this.buffers.velWY } },
                { binding: 5, resource: { buffer: this.buffers.flags } },
            ],
        });

        // --- Boundary pipeline (same bind group layout + flags is read_write) ---
        const boundaryModule = this.device.createShaderModule({
            label: 'boundary',
            code: commonWGSL + '\n' + boundaryWGSL,
        });

        const boundaryBindGroupLayout = this.device.createBindGroupLayout({
            label: 'boundary',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });

        this._boundaryPipeline = this.device.createComputePipeline({
            label: 'boundary',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [boundaryBindGroupLayout] }),
            compute: { module: boundaryModule, entryPoint: 'main' },
        });

        this._boundaryBindGroup = this.device.createBindGroup({
            label: 'boundary',
            layout: boundaryBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.buffers.posX } },
                { binding: 2, resource: { buffer: this.buffers.posY } },
                { binding: 3, resource: { buffer: this.buffers.velWX } },
                { binding: 4, resource: { buffer: this.buffers.velWY } },
                { binding: 5, resource: { buffer: this.buffers.flags } },
            ],
        });

        this._ready = true;
    }

    /**
     * Add a particle to the GPU buffers.
     * Phase 1: writes directly via queue.writeBuffer (no free stack yet).
     */
    addParticle({ x, y, vx = 0, vy = 0, mass: m = 1, charge: q = 0 }) {
        const idx = this.aliveCount;
        if (idx >= MAX_PARTICLES) return -1;

        const f32 = new Float32Array([0]);
        const u32 = new Uint32Array([0]);

        f32[0] = x; this.device.queue.writeBuffer(this.buffers.posX, idx * 4, f32);
        f32[0] = y; this.device.queue.writeBuffer(this.buffers.posY, idx * 4, f32);
        f32[0] = vx; this.device.queue.writeBuffer(this.buffers.velWX, idx * 4, f32);
        f32[0] = vy; this.device.queue.writeBuffer(this.buffers.velWY, idx * 4, f32);
        f32[0] = 0; this.device.queue.writeBuffer(this.buffers.angW, idx * 4, f32);
        f32[0] = m; this.device.queue.writeBuffer(this.buffers.mass, idx * 4, f32);
        f32[0] = m; this.device.queue.writeBuffer(this.buffers.baseMass, idx * 4, f32);
        f32[0] = q; this.device.queue.writeBuffer(this.buffers.charge, idx * 4, f32);
        f32[0] = Math.cbrt(m); this.device.queue.writeBuffer(this.buffers.radius, idx * 4, f32);
        const wSq = vx * vx + vy * vy;
        f32[0] = Math.sqrt(1 + wSq); this.device.queue.writeBuffer(this.buffers.gamma, idx * 4, f32);
        u32[0] = FLAG_ALIVE; this.device.queue.writeBuffer(this.buffers.flags, idx * 4, u32);

        // Pack color: neutral slate = #8A7E72 → RGBA
        u32[0] = 0xFF727E8A; // ABGR packed
        this.device.queue.writeBuffer(this.buffers.color, idx * 4, u32);

        this.aliveCount++;
        return idx;
    }

    /**
     * Run one substep: drift + boundary.
     * Phase 1 only — no forces, no Boris rotation.
     */
    update(dt) {
        if (!this._ready || this.aliveCount === 0) return;

        this.simTime += dt;

        // Upload uniforms
        writeUniforms(this.device, this.uniformBuffer, {
            dt,
            simTime: this.simTime,
            domainW: this.domainW,
            domainH: this.domainH,
            speedScale: 1,
            softening: 8,
            softeningSq: 64,
            boundaryMode: this.boundaryMode,
            topologyMode: this.topologyMode,
            maxParticles: MAX_PARTICLES,
            aliveCount: this.aliveCount,
        });

        const workgroups = Math.ceil(this.aliveCount / 64);

        const encoder = this.device.createCommandEncoder({ label: 'physics' });

        // Drift pass
        const driftPass = encoder.beginComputePass({ label: 'drift' });
        driftPass.setPipeline(this._driftPipeline);
        driftPass.setBindGroup(0, this._driftBindGroup);
        driftPass.dispatchWorkgroups(workgroups);
        driftPass.end();

        // Boundary pass
        const boundaryPass = encoder.beginComputePass({ label: 'boundary' });
        boundaryPass.setPipeline(this._boundaryPipeline);
        boundaryPass.setBindGroup(0, this._boundaryBindGroup);
        boundaryPass.dispatchWorkgroups(workgroups);
        boundaryPass.end();

        this.device.queue.submit([encoder.finish()]);
    }

    reset() {
        this.aliveCount = 0;
        this.simTime = 0;
    }

    destroy() {
        this.buffers.destroy();
        this.uniformBuffer.destroy();
    }
}

// Flag constants (must match common.wgsl)
const FLAG_ALIVE = 1;

/** Fetch a WGSL shader file relative to src/gpu/shaders/ */
async function fetchShader(filename) {
    const resp = await fetch(`src/gpu/shaders/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    return resp.text();
}
```

- [ ] **Step 2: Verify module loads**

In browser console: `import('./src/gpu/gpu-physics.js').then(m => console.log(m.default))` — should log the class.

- [ ] **Step 3: Commit**

```bash
git add src/gpu/gpu-physics.js
git commit -m "feat: add GPUPhysics orchestrator with drift + boundary pipelines"
```

---

### Task 10: Create instanced particle render shader

**Files:**
- Create: `src/gpu/shaders/particle.wgsl`

Vertex shader reads position/radius/color from SoA storage buffers, applies camera transform, outputs a quad. Fragment shader draws a circle with soft edge falloff.

- [ ] **Step 1: Create `src/gpu/shaders/particle.wgsl`**

```wgsl
// Instanced particle rendering — vertex + fragment.
// Each instance = one particle. Renders as a screen-aligned quad, fragment discards outside circle.

struct CameraUniforms {
    viewMatrix: mat4x4<f32>,
    invViewMatrix: mat4x4<f32>,
    zoom: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> posX: array<f32>;
@group(0) @binding(2) var<storage, read> posY: array<f32>;
@group(0) @binding(3) var<storage, read> radius: array<f32>;
@group(0) @binding(4) var<storage, read> color: array<u32>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,       // -1..+1 within quad
    @location(1) particleColor: vec4<f32>,
    @location(2) softness: f32,       // for edge falloff
};

// Quad vertices: 2 triangles forming a [-1,1] square
const QUAD_POS = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32,
) -> VertexOut {
    var out: VertexOut;

    let flag = flags[instanceIndex];
    // Skip non-alive particles by pushing off-screen
    if ((flag & 1u) == 0u) {
        out.position = vec4(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let px = posX[instanceIndex];
    let py = posY[instanceIndex];
    let r = radius[instanceIndex];

    // Transform world position to clip space via camera
    let worldPos = camera.viewMatrix * vec4(px, py, 0.0, 1.0);

    // Quad corner offset in pixels, then to clip space
    let quadCorner = QUAD_POS[vertexIndex];
    let pixelRadius = max(r * camera.zoom, 2.0); // minimum 2px
    let offsetPx = quadCorner * pixelRadius;

    let clipX = worldPos.x + offsetPx.x * 2.0 / camera.canvasWidth;
    let clipY = worldPos.y + offsetPx.y * 2.0 / camera.canvasHeight;

    out.position = vec4(clipX, clipY, 0.0, 1.0);
    out.uv = quadCorner;

    // Unpack RGBA from u32 (ABGR packed)
    let packed = color[instanceIndex];
    out.particleColor = vec4(
        f32(packed & 0xFFu) / 255.0,
        f32((packed >> 8u) & 0xFFu) / 255.0,
        f32((packed >> 16u) & 0xFFu) / 255.0,
        f32((packed >> 24u) & 0xFFu) / 255.0,
    );

    out.softness = 1.0 / max(pixelRadius, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Circle with soft edge
    let dist = length(in.uv);
    if (dist > 1.0) { discard; }

    // Smooth falloff at edge (replaces Canvas2D shadowBlur)
    let alpha = smoothstep(1.0, 0.7, dist) * in.particleColor.a;

    // Premultiplied alpha output (required by alphaMode: 'premultiplied')
    return vec4(in.particleColor.rgb * alpha, alpha);
}
```

- [ ] **Step 2: Commit**

```bash
git add src/gpu/shaders/particle.wgsl
git commit -m "feat: add instanced particle render shader (vertex + fragment)"
```

---

### Task 11: Create GPURenderer

**Files:**
- Create: `src/gpu/gpu-renderer.js`

Sets up the WebGPU render pipeline for instanced particle drawing. Reads position/radius/color directly from the compute storage buffers.

- [ ] **Step 1: Create `src/gpu/gpu-renderer.js`**

```js
/**
 * @fileoverview GPURenderer — WebGPU instanced rendering for particles.
 *
 * Phase 1: particles only. Trails, fields, bosons, arrows added later.
 */

export default class GPURenderer {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {GPUDevice} device
     * @param {Object} particleBuffers - from gpu-buffers.js
     */
    constructor(canvas, device, particleBuffers) {
        this.canvas = canvas;
        this.device = device;
        this.buffers = particleBuffers;

        this.context = canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device,
            format: this.format,
            alphaMode: 'premultiplied',
        });

        // Camera state (updated from shared-camera.js)
        this.cameraX = 0;
        this.cameraY = 0;
        this.zoom = 16; // WORLD_SCALE default
        this.canvasWidth = canvas.width;
        this.canvasHeight = canvas.height;
        this.isLight = true;

        // Uniform buffer for camera
        this.cameraBuffer = device.createBuffer({
            label: 'cameraUniforms',
            size: 256, // 2 × mat4x4 + 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._pipeline = null;
        this._bindGroup = null;
        this._ready = false;
    }

    /** Create render pipeline. Must be called after GPUPhysics.init(). */
    async init() {
        const shaderCode = await fetchShader('particle.wgsl');

        const module = this.device.createShaderModule({
            label: 'particle render',
            code: shaderCode,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            label: 'particle render',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            ],
        });

        this._pipeline = this.device.createRenderPipeline({
            label: 'particle render',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module,
                entryPoint: 'vs_main',
            },
            fragment: {
                module,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.format,
                    blend: this.isLight
                        ? {
                            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        }
                        : {
                            // Additive blending for dark mode
                            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                        },
                }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        this._bindGroup = this.device.createBindGroup({
            label: 'particle render',
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.cameraBuffer } },
                { binding: 1, resource: { buffer: this.buffers.posX } },
                { binding: 2, resource: { buffer: this.buffers.posY } },
                { binding: 3, resource: { buffer: this.buffers.radius } },
                { binding: 4, resource: { buffer: this.buffers.color } },
                { binding: 5, resource: { buffer: this.buffers.flags } },
            ],
        });

        this._ready = true;
    }

    /** Update camera uniform buffer. Call before render(). */
    updateCamera(camera) {
        this.cameraX = camera.x;
        this.cameraY = camera.y;
        this.zoom = camera.zoom;
        this.canvasWidth = this.canvas.width;
        this.canvasHeight = this.canvas.height;

        // Build 2D view matrix (world → clip)
        // clip.x = (worldX - cameraX) * zoom * 2 / canvasWidth
        // clip.y = -(worldY - cameraY) * zoom * 2 / canvasHeight  (y-flip for clip space)
        const sx = this.zoom * 2 / this.canvasWidth;
        const sy = -this.zoom * 2 / this.canvasHeight;
        const tx = -this.cameraX * sx;
        const ty = -this.cameraY * sy;

        // mat4x4 column-major
        const view = new Float32Array([
            sx, 0,  0, 0,
            0,  sy, 0, 0,
            0,  0,  1, 0,
            tx, ty, 0, 1,
        ]);

        // Inverse: world = (clip - translate) / scale
        const isx = 1 / sx;
        const isy = 1 / sy;
        const inv = new Float32Array([
            isx, 0,   0, 0,
            0,   isy, 0, 0,
            0,   0,   1, 0,
            -tx * isx, -ty * isy, 0, 1,
        ]);

        const data = new ArrayBuffer(256);
        const f = new Float32Array(data);
        f.set(view, 0);        // viewMatrix at offset 0 (64 bytes)
        f.set(inv, 16);        // invViewMatrix at offset 64 (64 bytes)
        f[32] = this.zoom;     // offset 128
        f[33] = this.canvasWidth;
        f[34] = this.canvasHeight;
        f[35] = 0; // pad

        this.device.queue.writeBuffer(this.cameraBuffer, 0, data);
    }

    /** Render one frame. */
    render(aliveCount) {
        if (!this._ready || aliveCount === 0) return;

        const textureView = this.context.getCurrentTexture().createView();

        const encoder = this.device.createCommandEncoder({ label: 'render' });

        const pass = encoder.beginRenderPass({
            label: 'particle render',
            colorAttachments: [{
                view: textureView,
                clearValue: this.isLight
                    ? { r: 0.941, g: 0.922, b: 0.894, a: 1 }  // --bg-canvas light: #F0EBE4
                    : { r: 0.047, g: 0.043, b: 0.035, a: 1 },  // --bg-canvas dark: #0C0B09
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bindGroup);
        // 6 vertices per quad (2 triangles), aliveCount instances
        pass.draw(6, aliveCount);
        pass.end();

        this.device.queue.submit([encoder.finish()]);
    }

    setTheme(isLight) {
        this.isLight = isLight;
        // KNOWN LIMITATION (Phase 1): Blend mode baked at init time — only clear color changes.
        // Spec requires two pre-built pipelines (additive dark / alpha light) swapped on theme change.
        // Will be implemented in Phase 2 when the render pipeline is expanded.
    }

    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.canvasWidth = width;
        this.canvasHeight = height;
    }

    destroy() {
        this.cameraBuffer.destroy();
    }
}

async function fetchShader(filename) {
    const resp = await fetch(`src/gpu/shaders/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load shader: ${filename}`);
    return resp.text();
}
```

- [ ] **Step 2: Commit**

```bash
git add src/gpu/gpu-renderer.js
git commit -m "feat: add GPURenderer with instanced particle render pipeline"
```

---

### Task 12: Wire GPU backend into main.js and verify end-to-end

**Files:**
- Modify: `main.js`

Connect GPUPhysics and GPURenderer to the main loop. Add a test mode that spawns a few particles with random velocities on the GPU path to verify everything works end-to-end.

- [ ] **Step 1: Add GPU imports to main.js**

Add after the existing backend imports (from Task 4):

```js
import GPUPhysics from './src/gpu/gpu-physics.js';
import GPURenderer from './src/gpu/gpu-renderer.js';
```

- [ ] **Step 2: Add GPU initialization in selectBackend result handler**

Replace the `selectBackend().then(...)` block from Task 4 with. Note: the dual-canvas setup must happen **before** creating `GPURenderer`, since a canvas can only have one context type (`'2d'` vs `'webgpu'`):

```js
        selectBackend().then(async ({ backend, device }) => {
            this.backend = backend;
            this._gpuDevice = device || null;
            console.log(`[physsim] Backend: ${backend}${device ? ' (WebGPU available)' : ''}`);

            if (backend === BACKEND_GPU && device) {
                try {
                    // Create a separate canvas for GPU rendering (overlaid on CPU canvas).
                    // Cannot reuse simCanvas — it already has a '2d' context.
                    const gpuCanvas = document.createElement('canvas');
                    gpuCanvas.id = 'gpuCanvas';
                    gpuCanvas.width = this.width;
                    gpuCanvas.height = this.height;
                    gpuCanvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;z-index:1;';
                    this.canvas.parentElement.appendChild(gpuCanvas);

                    this._gpuPhysics = new GPUPhysics(device, this.domainW, this.domainH);
                    this._gpuRenderer = new GPURenderer(gpuCanvas, device, this._gpuPhysics.buffers);
                    await this._gpuPhysics.init();
                    await this._gpuRenderer.init();
                    this._gpuReady = true;
                    console.log('[physsim] GPU backend initialized');

                    // Register device.lost handler for error recovery
                    device.lost.then((info) => {
                        console.error('[physsim] GPU device lost:', info.message);
                        this._gpuReady = false;
                        gpuCanvas.remove();
                        // Full CPU fallback with state restore deferred to Phase 6
                    });

                    // Test: spawn a few particles with random velocities
                    for (let i = 0; i < 8; i++) {
                        this._gpuPhysics.addParticle({
                            x: this.domainW * (0.2 + 0.6 * Math.random()),
                            y: this.domainH * (0.2 + 0.6 * Math.random()),
                            vx: (Math.random() - 0.5) * 2,
                            vy: (Math.random() - 0.5) * 2,
                            mass: 0.5 + Math.random() * 2,
                        });
                    }
                } catch (e) {
                    console.error('[physsim] GPU init failed, falling back to CPU:', e);
                    this._gpuReady = false;
                }
            }
        });
```

- [ ] **Step 3: Add GPU update + render to the loop**

In the `loop()` method, after the existing physics update block but before the dirty-flag render block, add:

```js
        // GPU path: run compute + render independently of CPU path (Phase 1 test)
        if (this._gpuReady && this.running) {
            this._gpuPhysics.update(PHYSICS_DT * this.speedScale);
            this._gpuRenderer.updateCamera(this.camera);
            // For Phase 1: render GPU particles on top of CPU canvas
            // (both paths active simultaneously for comparison)
            this._gpuRenderer.render(this._gpuPhysics.aliveCount);
        }
```

Note: In Phase 1, BOTH backends render simultaneously — CPU particles on Canvas 2D, GPU particles on the WebGPU canvas overlay. This is intentional for comparison. In Phase 2+, the CPU path will be disabled when GPU is active.

**Known limitation**: `aliveCount` on the JS side is not decremented by GPU-side boundary despawns. This means the drift/boundary shaders iterate over some dead slots (skipped by `FLAG_ALIVE` check — correct but slightly wasteful). A readback mechanism to sync `aliveCount` will be added in a later phase.

- [ ] **Step 5: Verify end-to-end**

Serve site, open in Chrome. Expected:
1. CPU simulation runs normally (existing particles with forces)
2. Console shows `[physsim] GPU backend initialized`
3. 8 additional circles appear on the GPU canvas (overlaid, no forces, drifting in straight lines)
4. GPU particles wrap/bounce at boundaries
5. No console errors

- [ ] **Step 6: Commit**

```bash
git add main.js
git commit -m "feat: wire GPU backend into main loop with test particles"
```

---

### Task 13: Add camera synchronization for GPU renderer

**Files:**
- Modify: `main.js`

When the user zooms/pans (via shared-camera.js), the GPU renderer needs to update its camera matrix. The camera object is already available in the loop.

- [ ] **Step 1: Sync camera in render block**

The `updateCamera` call in Step 3 of Task 12 already passes `this.camera`. Verify that zoom, pan, and resize events properly update the GPU canvas:

```js
        // In the resize handler (search for 'resize' in main.js):
        if (this._gpuRenderer) {
            const gpuCanvas = document.getElementById('gpuCanvas');
            if (gpuCanvas) {
                gpuCanvas.width = this.width;
                gpuCanvas.height = this.height;
            }
            this._gpuRenderer.resize(this.width, this.height);
        }
```

- [ ] **Step 2: Verify camera sync**

Zoom in/out with scroll wheel. GPU particles should zoom in sync with CPU particles. Pan with middle-click — GPU particles should track.

- [ ] **Step 3: Commit**

```bash
git add main.js
git commit -m "feat: sync camera zoom/pan/resize with GPU renderer"
```

---

### Task 14: Verify Phase 1 complete

- [ ] **Step 1: Full Phase 1 verification checklist**

Open site in Chrome with WebGPU support. Verify:
1. Console shows `[physsim] Backend: gpu (WebGPU available)`
2. Console shows `[physsim] GPU backend initialized`
3. CPU simulation runs normally with all features
4. 8 GPU test particles appear as colored circles
5. GPU particles drift in straight lines (no forces)
6. GPU particles wrap at boundaries (switch boundary to Loop)
7. GPU particles bounce at boundaries (switch boundary to Bounce)
8. GPU particles despawn when leaving domain (boundary = Despawn)
9. Camera zoom affects both CPU and GPU particles in sync
10. Camera pan affects both CPU and GPU particles in sync
11. Window resize updates both canvases
12. No console errors or warnings
13. Performance: no visible frame drops from running both backends
14. UI elements (toolbar, sidebar panel) are not obscured by GPU canvas overlay
15. Theme toggle: GPU particles still visible in both light and dark modes (note: blend mode is light-only in Phase 1 — additive dark-mode blending deferred to Phase 2)

- [ ] **Step 2: Open in Firefox (no WebGPU)**

Verify:
1. Console shows `[physsim] Backend: cpu`
2. No GPU canvas created
3. Simulation runs normally on CPU path
4. No errors

- [ ] **Step 3: Commit Phase 1 complete**

```bash
git commit --allow-empty -m "chore: Phase 1 complete — GPU scaffold with drift + boundary + instanced rendering"
```

---

## What Phase 2 Covers (not in this plan)

Phase 2 will be planned in a separate document once Phase 0+1 are verified. It covers:

- Port pairwise force computation to WGSL (all 11 force types)
- Full Boris integrator (half-kick, rotation, half-kick, drift)
- Spin-orbit kick + torque application
- Radiation reaction (Larmor)
- Force display rendering (arrows)
- Disabling CPU path when GPU is active
- Particle spawning via click (input.js → GPU addParticle)
