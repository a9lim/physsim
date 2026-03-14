// ─── Ghost Particle Generation ───
// One thread per alive particle. Checks proximity to domain edges.
// Appends ghost copies at positions beyond the boundary edge.
// Ghost flag (bit 4 in flags) marks them for tree insertion but skips self-accumulation.

// Topology constants
const TOPO_TORUS: u32 = 0u;
const TOPO_KLEIN: u32 = 1u;
const TOPO_RP2:   u32 = 2u;

// Flag bits
const FLAG_ALIVE:  u32 = 1u;
const FLAG_GHOST:  u32 = 16u;

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
    extGravity: f32,
    extGravityAngle: f32,
    extElectric: f32,
    extElectricAngle: f32,
    extBz: f32,
    bounceFriction: f32,
    extGx: f32,
    extGy: f32,
    extEx: f32,
    extEy: f32,
    axionCoupling: f32,
    higgsCoupling: f32,
    particleCount: u32,
    bhTheta: f32,
    _pad3: u32,
    _pad4: u32,
};

@group(0) @binding(0) var<storage, read> posX: array<f32>;
@group(0) @binding(1) var<storage, read> posY: array<f32>;
@group(0) @binding(2) var<storage, read> velWX: array<f32>;
@group(0) @binding(3) var<storage, read> velWY: array<f32>;
@group(0) @binding(4) var<storage, read> angW_in: array<f32>;
@group(0) @binding(5) var<storage, read> mass_in: array<f32>;
@group(0) @binding(6) var<storage, read> charge_in: array<f32>;
@group(0) @binding(7) var<storage, read> flags_in: array<u32>;

// Ghost output: written into the same SoA arrays starting at offset = aliveCount
@group(1) @binding(0) var<storage, read_write> ghostPosX: array<f32>;
@group(1) @binding(1) var<storage, read_write> ghostPosY: array<f32>;
@group(1) @binding(2) var<storage, read_write> ghostVelWX: array<f32>;
@group(1) @binding(3) var<storage, read_write> ghostVelWY: array<f32>;
@group(1) @binding(4) var<storage, read_write> ghostAngW: array<f32>;
@group(1) @binding(5) var<storage, read_write> ghostMass: array<f32>;
@group(1) @binding(6) var<storage, read_write> ghostCharge: array<f32>;
@group(1) @binding(7) var<storage, read_write> ghostFlags: array<u32>;

@group(2) @binding(0) var<storage, read_write> ghostCounter: atomic<u32>;
@group(2) @binding(1) var<uniform> uniforms: SimUniforms;

// Additional per-particle data needed for tree aggregates
@group(1) @binding(8) var<storage, read> radius_in: array<f32>;
@group(1) @binding(9) var<storage, read> magAngMom_in: array<vec2<f32>>;   // packed magMoment, angMomentum
@group(1) @binding(10) var<storage, read_write> ghostRadius: array<f32>;
@group(1) @binding(11) var<storage, read_write> ghostMagAngMom: array<vec2<f32>>; // packed ghost magMoment, angMomentum
@group(1) @binding(12) var<storage, read> particleId_in: array<u32>;
@group(1) @binding(13) var<storage, read_write> ghostParticleId: array<u32>;

// Store original particle index for ghost->original mapping
@group(2) @binding(2) var<storage, read_write> ghostOriginalIdx: array<u32>;

const MAX_GHOSTS: u32 = 4096u; // Must match MAX_PARTICLES

fn appendGhost(
    px: f32, py: f32, wx: f32, wy: f32, aw: f32,
    m: f32, q: f32, r: f32, mm: f32, am: f32,
    origIdx: u32, pid: u32
) {
    let slot = atomicAdd(&ghostCounter, 1u);
    if (slot >= MAX_GHOSTS) {
        return; // overflow guard
    }
    ghostPosX[slot] = px;
    ghostPosY[slot] = py;
    ghostVelWX[slot] = wx;
    ghostVelWY[slot] = wy;
    ghostAngW[slot] = aw;
    ghostMass[slot] = m;
    ghostCharge[slot] = q;
    ghostFlags[slot] = FLAG_ALIVE | FLAG_GHOST;
    ghostRadius[slot] = r;
    ghostMagAngMom[slot] = vec2(mm, am);
    ghostOriginalIdx[slot] = origIdx;
    ghostParticleId[slot] = pid;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) {
        return;
    }
    if ((flags_in[idx] & FLAG_ALIVE) == 0u) {
        return;
    }

    let W = uniforms.domainW;
    let H = uniforms.domainH;
    let margin = max(W, H) * uniforms.bhTheta;
    let topo = uniforms.topologyMode;

    let x = posX[idx];
    let y = posY[idx];
    let wx = velWX[idx];
    let wy = velWY[idx];
    let aw = angW_in[idx];
    let m = mass_in[idx];
    let q = charge_in[idx];
    let r = radius_in[idx];
    let mamIn = magAngMom_in[idx];
    let mm = mamIn.x;
    let am = mamIn.y;
    let pid = particleId_in[idx];

    let nearL = x < margin;
    let nearR = x > W - margin;
    let nearT = y < margin;
    let nearB = y > H - margin;

    if (topo == TOPO_TORUS) {
        // Torus: simple wrap, no velocity flip
        if (nearL) { appendGhost(x + W, y, wx, wy, aw, m, q, r, mm, am, idx, pid); }
        if (nearR) { appendGhost(x - W, y, wx, wy, aw, m, q, r, mm, am, idx, pid); }
        if (nearT) { appendGhost(x, y + H, wx, wy, aw, m, q, r, mm, am, idx, pid); }
        if (nearB) { appendGhost(x, y - H, wx, wy, aw, m, q, r, mm, am, idx, pid); }
        if (nearL && nearT) { appendGhost(x + W, y + H, wx, wy, aw, m, q, r, mm, am, idx, pid); }
        if (nearL && nearB) { appendGhost(x + W, y - H, wx, wy, aw, m, q, r, mm, am, idx, pid); }
        if (nearR && nearT) { appendGhost(x - W, y + H, wx, wy, aw, m, q, r, mm, am, idx, pid); }
        if (nearR && nearB) { appendGhost(x - W, y - H, wx, wy, aw, m, q, r, mm, am, idx, pid); }
    } else if (topo == TOPO_KLEIN) {
        // Klein: x wraps normally; y-wrap flips x and negates vx, angw
        if (nearL) { appendGhost(x + W, y, wx, wy, aw, m, q, r, mm, am, idx, pid); }
        if (nearR) { appendGhost(x - W, y, wx, wy, aw, m, q, r, mm, am, idx, pid); }
        // y-glide: mirror x, negate vx and angw
        let fmm_k = -mm; // magMoment flips sign with angVel flip
        let fam_k = -am; // angMomentum flips sign
        if (nearT) { appendGhost(W - x, y + H, -wx, wy, -aw, m, q, r, fmm_k, fam_k, idx, pid); }
        if (nearB) { appendGhost(W - x, y - H, -wx, wy, -aw, m, q, r, fmm_k, fam_k, idx, pid); }
        // Corners: combine x-wrap + y-glide
        if (nearL && nearT) { appendGhost(W - x + W, y + H, -wx, wy, -aw, m, q, r, fmm_k, fam_k, idx, pid); }
        if (nearL && nearB) { appendGhost(W - x + W, y - H, -wx, wy, -aw, m, q, r, fmm_k, fam_k, idx, pid); }
        if (nearR && nearT) { appendGhost(W - x - W, y + H, -wx, wy, -aw, m, q, r, fmm_k, fam_k, idx, pid); }
        if (nearR && nearB) { appendGhost(W - x - W, y - H, -wx, wy, -aw, m, q, r, fmm_k, fam_k, idx, pid); }
    } else {
        // RP2: x-wrap flips y and negates vy, angw; y-wrap flips x and negates vx, angw
        let fmm_vy = -mm;
        let fam_vy = -am;
        let fmm_vx = -mm;
        let fam_vx = -am;
        // x-wrap: mirror y, negate vy and angw
        if (nearL) { appendGhost(x + W, H - y, wx, -wy, -aw, m, q, r, fmm_vy, fam_vy, idx, pid); }
        if (nearR) { appendGhost(x - W, H - y, wx, -wy, -aw, m, q, r, fmm_vy, fam_vy, idx, pid); }
        // y-wrap: mirror x, negate vx and angw
        if (nearT) { appendGhost(W - x, y + H, -wx, wy, -aw, m, q, r, fmm_vx, fam_vx, idx, pid); }
        if (nearB) { appendGhost(W - x, y - H, -wx, wy, -aw, m, q, r, fmm_vx, fam_vx, idx, pid); }
        // Corners: y-glide takes precedence (matches CPU _addGhost for RP2 corners)
        if (nearL && nearT) { appendGhost(W - x + W, y + H, -wx, wy, -aw, m, q, r, fmm_vx, fam_vx, idx, pid); }
        if (nearL && nearB) { appendGhost(W - x + W, y - H, -wx, wy, -aw, m, q, r, fmm_vx, fam_vx, idx, pid); }
        if (nearR && nearT) { appendGhost(W - x - W, y + H, -wx, wy, -aw, m, q, r, fmm_vx, fam_vx, idx, pid); }
        if (nearR && nearB) { appendGhost(W - x - W, y - H, -wx, wy, -aw, m, q, r, fmm_vx, fam_vx, idx, pid); }
    }
}
