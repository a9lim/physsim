// trails.wgsl — Append current particle position to trail ring buffer.
//
// Dispatched once per frame (after all substeps), not per substep.
// Ring buffer length = TRAIL_LEN (256). Write index wraps around.

const TRAIL_LEN: u32 = 256u;

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

@group(0) @binding(0) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(1) var<storage, read_write> trailX: array<f32>;   // [MAX_PARTICLES * TRAIL_LEN]
@group(0) @binding(2) var<storage, read_write> trailY: array<f32>;   // [MAX_PARTICLES * TRAIL_LEN]
@group(0) @binding(3) var<storage, read_write> trailWriteIdx: array<u32>;  // [MAX_PARTICLES]
@group(0) @binding(4) var<storage, read_write> trailCount: array<u32>;     // [MAX_PARTICLES]

const ALIVE_BIT: u32 = 1u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= arrayLength(&particles)) { return; }
    let p = particles[idx];
    if ((p.flags & ALIVE_BIT) == 0u) { return; }

    let writeIdx = trailWriteIdx[idx];
    let base = idx * TRAIL_LEN;

    trailX[base + writeIdx] = p.posX;
    trailY[base + writeIdx] = p.posY;

    trailWriteIdx[idx] = (writeIdx + 1u) % TRAIL_LEN;
    trailCount[idx] = min(trailCount[idx] + 1u, TRAIL_LEN);
}
