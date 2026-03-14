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

// ── Packed buffer structs (standalone — common.wgsl not prepended) ──

struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct ParticleAux {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
};

struct ParticleDerived {
    magMoment: f32,
    angMomentum: f32,
    invMass: f32,
    radiusSq: f32,
    velX: f32,
    velY: f32,
    angVel: f32,
    _pad: f32,
};

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
    frameCount: u32,
    _pad4: u32,
};

// Group 0: packed particle state (read_write for encoder compat)
@group(0) @binding(0) var<storage, read_write> particleState: array<ParticleState>;

// Group 1: ghost outputs + derived + particleAux (read_write for encoder compat)
@group(1) @binding(0) var<storage, read_write> ghostState: array<ParticleState>;
@group(1) @binding(1) var<storage, read_write> ghostAux: array<ParticleAux>;
@group(1) @binding(2) var<storage, read_write> derived_in: array<ParticleDerived>;
@group(1) @binding(3) var<storage, read_write> ghostDerived: array<ParticleDerived>;
@group(1) @binding(4) var<storage, read_write> particleAux_in: array<ParticleAux>;

// Group 2: ghostCounter + uniforms + ghostOriginalIdx
@group(2) @binding(0) var<storage, read_write> ghostCounter: atomic<u32>;
@group(2) @binding(1) var<uniform> uniforms: SimUniforms;
@group(2) @binding(2) var<storage, read_write> ghostOriginalIdx: array<u32>;

const MAX_GHOSTS: u32 = 4096u; // Must match MAX_PARTICLES

fn appendGhost(
    gs: ParticleState, ga: ParticleAux, dd: ParticleDerived,
    origIdx: u32
) {
    let slot = atomicAdd(&ghostCounter, 1u);
    if (slot >= MAX_GHOSTS) {
        return; // overflow guard
    }
    ghostState[slot] = gs;
    ghostAux[slot] = ga;
    ghostDerived[slot] = dd;
    ghostOriginalIdx[slot] = origIdx;
}

fn makeGhostState(px: f32, py: f32, wx: f32, wy: f32, aw: f32, m: f32, q: f32, bm: f32) -> ParticleState {
    var gs: ParticleState;
    gs.posX = px;
    gs.posY = py;
    gs.velWX = wx;
    gs.velWY = wy;
    gs.mass = m;
    gs.charge = q;
    gs.angW = aw;
    gs.baseMass = bm;
    gs.flags = FLAG_ALIVE | FLAG_GHOST;
    return gs;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) {
        return;
    }

    let ps = particleState[idx];
    if ((ps.flags & FLAG_ALIVE) == 0u) {
        return;
    }

    let W = uniforms.domainW;
    let H = uniforms.domainH;
    let margin = max(W, H) * uniforms.bhTheta;
    let topo = uniforms.topologyMode;

    let x = ps.posX;
    let y = ps.posY;
    let wx = ps.velWX;
    let wy = ps.velWY;
    let aw = ps.angW;
    let m = ps.mass;
    let q = ps.charge;
    let bm = ps.baseMass;
    let aux = particleAux_in[idx];
    let r = aux.radius;
    let pid = aux.particleId;
    let dd = derived_in[idx];
    let mm = dd.magMoment;
    let am = dd.angMomentum;

    // Build ghost aux (copy radius and particleId, zero death fields)
    var ga: ParticleAux;
    ga.radius = r;
    ga.particleId = pid;
    ga.deathTime = 0.0;
    ga.deathMass = 0.0;
    ga.deathAngVel = 0.0;

    let nearL = x < margin;
    let nearR = x > W - margin;
    let nearT = y < margin;
    let nearB = y > H - margin;

    // Build flipped derived struct for Klein/RP2 glide reflections.
    // Must also flip velocity fields in the derived struct to match flipped velWX/velWY,
    // since tree force shader reads velocity from derived_in[].velX/velY.

    // Klein y-glide: flip vx and angvel
    var ddKleinY = dd;
    ddKleinY.velX = -dd.velX;
    ddKleinY.angVel = -dd.angVel;
    ddKleinY.magMoment = -mm;
    ddKleinY.angMomentum = -am;

    // RP2 x-glide: flip vy and angvel
    var ddRP2X = dd;
    ddRP2X.velY = -dd.velY;
    ddRP2X.angVel = -dd.angVel;
    ddRP2X.magMoment = -mm;
    ddRP2X.angMomentum = -am;

    // RP2 y-glide: flip vx and angvel (same as Klein y-glide)
    let ddRP2Y = ddKleinY;

    if (topo == TOPO_TORUS) {
        // Torus: simple wrap, no velocity flip
        if (nearL) { appendGhost(makeGhostState(x + W, y, wx, wy, aw, m, q, bm), ga, dd, idx); }
        if (nearR) { appendGhost(makeGhostState(x - W, y, wx, wy, aw, m, q, bm), ga, dd, idx); }
        if (nearT) { appendGhost(makeGhostState(x, y + H, wx, wy, aw, m, q, bm), ga, dd, idx); }
        if (nearB) { appendGhost(makeGhostState(x, y - H, wx, wy, aw, m, q, bm), ga, dd, idx); }
        if (nearL && nearT) { appendGhost(makeGhostState(x + W, y + H, wx, wy, aw, m, q, bm), ga, dd, idx); }
        if (nearL && nearB) { appendGhost(makeGhostState(x + W, y - H, wx, wy, aw, m, q, bm), ga, dd, idx); }
        if (nearR && nearT) { appendGhost(makeGhostState(x - W, y + H, wx, wy, aw, m, q, bm), ga, dd, idx); }
        if (nearR && nearB) { appendGhost(makeGhostState(x - W, y - H, wx, wy, aw, m, q, bm), ga, dd, idx); }
    } else if (topo == TOPO_KLEIN) {
        // Klein: x wraps normally; y-wrap flips x and negates vx, angw
        if (nearL) { appendGhost(makeGhostState(x + W, y, wx, wy, aw, m, q, bm), ga, dd, idx); }
        if (nearR) { appendGhost(makeGhostState(x - W, y, wx, wy, aw, m, q, bm), ga, dd, idx); }
        // y-glide: mirror x, negate vx and angw — derived velX/angVel also flipped
        if (nearT) { appendGhost(makeGhostState(W - x, y + H, -wx, wy, -aw, m, q, bm), ga, ddKleinY, idx); }
        if (nearB) { appendGhost(makeGhostState(W - x, y - H, -wx, wy, -aw, m, q, bm), ga, ddKleinY, idx); }
        // Corners: combine x-wrap + y-glide
        if (nearL && nearT) { appendGhost(makeGhostState(W - x + W, y + H, -wx, wy, -aw, m, q, bm), ga, ddKleinY, idx); }
        if (nearL && nearB) { appendGhost(makeGhostState(W - x + W, y - H, -wx, wy, -aw, m, q, bm), ga, ddKleinY, idx); }
        if (nearR && nearT) { appendGhost(makeGhostState(W - x - W, y + H, -wx, wy, -aw, m, q, bm), ga, ddKleinY, idx); }
        if (nearR && nearB) { appendGhost(makeGhostState(W - x - W, y - H, -wx, wy, -aw, m, q, bm), ga, ddKleinY, idx); }
    } else {
        // RP2: x-wrap flips y and negates vy, angw; y-wrap flips x and negates vx, angw
        // x-wrap: mirror y, negate vy and angw — derived velY/angVel flipped
        if (nearL) { appendGhost(makeGhostState(x + W, H - y, wx, -wy, -aw, m, q, bm), ga, ddRP2X, idx); }
        if (nearR) { appendGhost(makeGhostState(x - W, H - y, wx, -wy, -aw, m, q, bm), ga, ddRP2X, idx); }
        // y-wrap: mirror x, negate vx and angw — derived velX/angVel flipped
        if (nearT) { appendGhost(makeGhostState(W - x, y + H, -wx, wy, -aw, m, q, bm), ga, ddRP2Y, idx); }
        if (nearB) { appendGhost(makeGhostState(W - x, y - H, -wx, wy, -aw, m, q, bm), ga, ddRP2Y, idx); }
        // Corners: y-glide takes precedence (matches CPU _addGhost for RP2 corners)
        if (nearL && nearT) { appendGhost(makeGhostState(W - x + W, y + H, -wx, wy, -aw, m, q, bm), ga, ddRP2Y, idx); }
        if (nearL && nearB) { appendGhost(makeGhostState(W - x + W, y - H, -wx, wy, -aw, m, q, bm), ga, ddRP2Y, idx); }
        if (nearR && nearT) { appendGhost(makeGhostState(W - x - W, y + H, -wx, wy, -aw, m, q, bm), ga, ddRP2Y, idx); }
        if (nearR && nearB) { appendGhost(makeGhostState(W - x - W, y - H, -wx, wy, -aw, m, q, bm), ga, ddRP2Y, idx); }
    }
}
