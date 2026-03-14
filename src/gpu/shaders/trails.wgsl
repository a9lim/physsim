// trails.wgsl — Append current particle position to trail ring buffer.
//
// Dispatched once per frame (after all substeps), not per substep.
// Ring buffer length = TRAIL_LEN (256). Write index wraps around.

const TRAIL_LEN: u32 = 256u;

@group(0) @binding(0) var<storage, read> posX: array<f32>;
@group(0) @binding(1) var<storage, read> posY: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> trailX: array<f32>;   // [MAX_PARTICLES * TRAIL_LEN]
@group(0) @binding(4) var<storage, read_write> trailY: array<f32>;   // [MAX_PARTICLES * TRAIL_LEN]
@group(0) @binding(5) var<storage, read_write> trailWriteIdx: array<u32>;  // [MAX_PARTICLES]
@group(0) @binding(6) var<storage, read_write> trailCount: array<u32>;     // [MAX_PARTICLES]

const ALIVE_BIT: u32 = 1u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= arrayLength(&posX)) { return; }
    if ((flags[idx] & ALIVE_BIT) == 0u) { return; }

    let writeIdx = trailWriteIdx[idx];
    let base = idx * TRAIL_LEN;

    trailX[base + writeIdx] = posX[idx];
    trailY[base + writeIdx] = posY[idx];

    trailWriteIdx[idx] = (writeIdx + 1u) % TRAIL_LEN;
    trailCount[idx] = min(trailCount[idx] + 1u, TRAIL_LEN);
}
