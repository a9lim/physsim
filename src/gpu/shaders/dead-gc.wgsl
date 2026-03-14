// ─── Dead Particle Garbage Collection ───
// Transitions RETIRED particles to FREE when their signal delay history expires.
// Runs once per frame (not per substep).

const FLAG_ALIVE:   u32 = 1u;
const FLAG_RETIRED: u32 = 2u;

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

@group(0) @binding(0) var<storage, read_write> particleState: array<ParticleState>;
@group(0) @binding(1) var<storage, read_write> particleAux: array<ParticleAux>;
@group(0) @binding(2) var<uniform> uniforms: SimUniforms;
@group(0) @binding(3) var<storage, read_write> freeStack: array<u32>;
@group(0) @binding(4) var<storage, read_write> freeTop: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    // Scan all particle slots (not just alive count — retired particles may be anywhere)
    let maxSlots = uniforms.aliveCount; // High-water mark: aliveCount is never decremented, so all ever-allocated slots (alive, retired, or freed) live in [0, aliveCount)
    if (idx >= maxSlots) { return; }

    var ps = particleState[idx];

    // Skip alive particles
    if ((ps.flags & FLAG_ALIVE) != 0u) { return; }
    // Skip already-freed particles (flags == 0)
    if (ps.flags == 0u) { return; }

    let isRetired = (ps.flags & FLAG_RETIRED) != 0u;

    if (isRetired) {
        let deathT = particleAux[idx].deathTime;
        // Boundary-killed particles have deathTime = Infinity (never set by boundary shader)
        // → free immediately. Properly retired particles have finite deathTime → wait for expiry.
        if (deathT < 1e30) {
            let domainDiag = sqrt(uniforms.domainW * uniforms.domainW + uniforms.domainH * uniforms.domainH);
            let expiry = 2.0 * domainDiag;
            if (uniforms.simTime - deathT <= expiry) { return; }
        }
    }

    // Transition to FREE: clear all flags
    ps.flags = 0u;
    particleState[idx] = ps;

    // Push slot to free stack
    let slot = atomicAdd(&freeTop, 1u);
    freeStack[slot] = idx;
}
