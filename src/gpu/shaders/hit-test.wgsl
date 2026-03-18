// hit-test.wgsl — GPU quadtree point query for particle selection
//
// Dispatched as (1, 1, 1) — single thread. Walks the quadtree to find the
// particle closest to the click point within its radius.
// Writes the particle index + state data to hitResult for tooltip display.
//
// Standalone shader — common.wgsl not prepended.
// Constants (FLAG_ALIVE, etc.) provided by generated wgslConstants block.

struct HitUniforms {
    clickX: f32,
    clickY: f32,
    aliveCount: u32,
    _pad: u32,
};

// Struct definitions (ParticleState, ParticleAux, ParticleDerived) provided by shared-structs.wgsl.

// Hit result layout (12 f32 = 48 bytes):
//   [0]: hitIndex (as bitcast i32)
//   [1]: mass
//   [2]: charge
//   [3]: radius
//   [4]: velX  (coordinate velocity)
//   [5]: velY
//   [6]: angVel
//   [7]: posX
//   [8]: posY
//   [9..11]: reserved

// Quadtree node layout: flat array<u32>, 20 u32 per node (matches tree-build.wgsl)
const NODE_STRIDE: u32 = 20u;
fn nodeOffset(idx: u32) -> u32 { return idx * NODE_STRIDE; }
fn getMinX(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx)]); }
fn getMinY(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 1u]); }
fn getMaxX(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 2u]); }
fn getMaxY(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 3u]); }
fn getNW(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 12u]); }
fn getNE(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 13u]); }
fn getSW(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 14u]); }
fn getSE(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 15u]); }
fn getParticleIndex(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 16u]); }

const NONE: i32 = -1;

@group(0) @binding(0) var<uniform> hit: HitUniforms;
@group(0) @binding(1) var<storage, read> nodes: array<u32>;
@group(0) @binding(2) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(3) var<storage, read> particleAux: array<ParticleAux>;
@group(0) @binding(4) var<storage, read_write> hitResult: array<u32>;
@group(0) @binding(5) var<storage, read> derived: array<ParticleDerived>;

@compute @workgroup_size(1)
fn main() {
    let cx = hit.clickX;
    let cy = hit.clickY;

    var bestIdx: i32 = -1;
    var bestDistSq: f32 = 1e30;

    // Stack-based tree walk — check all particles within their radius
    var stack: array<u32, 48>;
    var top: i32 = 0;
    stack[0] = 0u;

    loop {
        if (top < 0) { break; }
        let nodeIdx = stack[top];
        top -= 1;

        // Expand search box by max particle radius (conservative: ∛(max_mass))
        let expand = 4.0;
        if (cx < getMinX(nodeIdx) - expand || cx > getMaxX(nodeIdx) + expand ||
            cy < getMinY(nodeIdx) - expand || cy > getMaxY(nodeIdx) + expand) {
            continue;
        }

        let isLeaf = getNW(nodeIdx) == NONE;
        let pi = getParticleIndex(nodeIdx);

        if (isLeaf && pi >= 0) {
            let pIdx = u32(pi);
            let p = particles[pIdx];
            if ((p.flags & FLAG_ALIVE) != 0u) {
                let dx = cx - p.posX;
                let dy = cy - p.posY;
                let distSq = dx * dx + dy * dy;
                let r = particleAux[pIdx].radius;
                if (distSq < r * r && distSq < bestDistSq) {
                    bestDistSq = distSq;
                    bestIdx = pi;
                }
            }
        } else if (!isLeaf) {
            let nw = getNW(nodeIdx);
            let ne = getNE(nodeIdx);
            let sw = getSW(nodeIdx);
            let se = getSE(nodeIdx);
            if (nw != NONE && top + 4 <= 48) { top += 1; stack[top] = u32(nw); }
            if (ne != NONE && top + 4 <= 48) { top += 1; stack[top] = u32(ne); }
            if (sw != NONE && top + 4 <= 48) { top += 1; stack[top] = u32(sw); }
            if (se != NONE && top + 4 <= 48) { top += 1; stack[top] = u32(se); }
        }
    }

    // Write index (as bitcast u32)
    hitResult[0] = bitcast<u32>(bestIdx);

    // Write particle data if hit found
    if (bestIdx >= 0) {
        let idx = u32(bestIdx);
        let p = particles[idx];
        let aux = particleAux[idx];
        let der = derived[idx];
        hitResult[1] = bitcast<u32>(p.mass);
        hitResult[2] = bitcast<u32>(p.charge);
        hitResult[3] = bitcast<u32>(aux.radius);
        hitResult[4] = bitcast<u32>(der.velX);
        hitResult[5] = bitcast<u32>(der.velY);
        hitResult[6] = bitcast<u32>(der.angVel);
        hitResult[7] = bitcast<u32>(p.posX);
        hitResult[8] = bitcast<u32>(p.posY);
    }
}
