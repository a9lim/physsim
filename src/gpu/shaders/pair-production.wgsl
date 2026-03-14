// ─── Pair Production ───
// Photon → particle-antiparticle pair near massive bodies.
// One thread per photon. Writes spawn events to append buffer.

// Packed photon struct (matches common.wgsl Photon)
struct Photon {
    posX: f32, posY: f32,
    velX: f32, velY: f32,
    energy: f32,
    emitterId: u32, age: u32, flags: u32,
};

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState_PP {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct PairProdUniforms {
    minEnergy: f32,       // 0.5
    proximity: f32,       // 8.0
    probability: f32,     // 0.005
    minAge: u32,          // 64
    maxParticles: u32,    // 32
    currentParticleCount: u32,
    photonCount: u32,
    blackHoleEnabled: u32,
    simTime: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct PairEvent {
    photonIdx: u32,
    nearestParticleIdx: u32,
    photonEnergy: f32,
    photonPosX: f32,
    photonPosY: f32,
    photonVelX: f32,
    photonVelY: f32,
    _pad: f32,
};

// Group 0: photonPool + phCount (read_write for encoder compat)
@group(0) @binding(0) var<storage, read_write> photonPool: array<Photon>;
@group(0) @binding(1) var<storage, read_write> phCount: array<u32>;

// Group 1: particleState (read_write for encoder compat)
@group(1) @binding(0) var<storage, read_write> particles: array<ParticleState_PP>;

@group(2) @binding(0) var<storage, read_write> pairEvents: array<PairEvent>;
@group(2) @binding(1) var<storage, read_write> pairCounter: atomic<u32>;
@group(2) @binding(2) var<uniform> pu: PairProdUniforms;

// Simple hash-based PRNG (per-thread deterministic from photon index + simTime)
fn pcgHash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn randomFloat(seed: u32) -> f32 {
    return f32(pcgHash(seed)) / 4294967296.0;
}

const MAX_PAIR_EVENTS: u32 = 32u;

@compute @workgroup_size(256)
fn checkPairProduction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let phIdx = gid.x;
    if (phIdx >= pu.photonCount) { return; }
    if (pu.blackHoleEnabled != 0u) { return; }

    let ph = photonPool[phIdx];
    if ((ph.flags & 1u) == 0u) { return; }
    if (ph.energy < pu.minEnergy) { return; }
    if (ph.age < pu.minAge) { return; }
    if (pu.currentParticleCount >= pu.maxParticles) { return; }

    // Check probability
    let seed = phIdx * 12345u + bitcast<u32>(pu.simTime);
    if (randomFloat(seed) > pu.probability) { return; }

    // Find nearest massive body within proximity
    let phX = ph.posX;
    let phY = ph.posY;
    let proxSq = pu.proximity * pu.proximity;
    var minDistSq: f32 = proxSq;
    var nearestIdx: u32 = 0xFFFFFFFFu;

    for (var i = 0u; i < pu.currentParticleCount; i++) {
        let p = particles[i];
        let pflag = p.flags;
        if ((pflag & 1u) == 0u) { continue; }
        if (p.mass < 1e-9) { continue; }

        let dx = p.posX - phX;
        let dy = p.posY - phY;
        let dSq = dx * dx + dy * dy;
        // CPU checks dSq < PAIR_PROD_RADIUS² * p.mass (larger bodies have larger cross-section)
        if (dSq < proxSq * p.mass) {
            minDistSq = dSq;
            nearestIdx = i;
        }
    }

    if (nearestIdx == 0xFFFFFFFFu) { return; }

    // Write pair production event and kill the photon
    let slot = atomicAdd(&pairCounter, 1u);
    if (slot < MAX_PAIR_EVENTS) {
        var evt: PairEvent;
        evt.photonIdx = phIdx;
        evt.nearestParticleIdx = nearestIdx;
        evt.photonEnergy = ph.energy;
        evt.photonPosX = phX;
        evt.photonPosY = phY;
        evt.photonVelX = ph.velX;
        evt.photonVelY = ph.velY;
        pairEvents[slot] = evt;

        // Kill the photon (matches CPU behavior)
        var phDead = photonPool[phIdx];
        phDead.flags = phDead.flags & ~1u;
        photonPool[phIdx] = phDead;
    }
}
