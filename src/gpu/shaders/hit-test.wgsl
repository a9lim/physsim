// hit-test.wgsl — GPU quadtree point query for particle selection
//
// Dispatched as (1, 1, 1) — single thread. Walks the quadtree built by
// tree-build.wgsl to find the particle closest to the click point within
// its radius. Writes the particle index (or -1) to hitResult.
//
// Standalone shader — common.wgsl not prepended.
// Constants (FLAG_ALIVE, etc.) provided by generated wgslConstants block.

struct HitUniforms {
    clickX: f32,
    clickY: f32,
    aliveCount: u32,
    _pad: u32,
};

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

// Packed auxiliary struct (matches common.wgsl ParticleAux)
struct ParticleAux {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
};

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
@group(0) @binding(4) var<storage, read_write> hitResult: array<i32>;

@compute @workgroup_size(1)
fn main() {
    let cx = hit.clickX;
    let cy = hit.clickY;
    let count = hit.aliveCount;

    var bestIdx: i32 = -1;
    var bestDistSq: f32 = 1e30;

    // When tree is available (root node exists), use tree walk
    // The tree root is always node 0; if NW child == NONE and particleIndex == NONE,
    // the tree is empty or not built — fall back to linear scan.
    let rootNW = getNW(0u);
    let rootPI = getParticleIndex(0u);
    let treeAvailable = rootNW != NONE || rootPI != NONE;

    if (treeAvailable) {
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
                if (nw != NONE && top < 46) { top += 1; stack[top] = u32(nw); }
                if (ne != NONE && top < 46) { top += 1; stack[top] = u32(ne); }
                if (sw != NONE && top < 46) { top += 1; stack[top] = u32(sw); }
                if (se != NONE && top < 46) { top += 1; stack[top] = u32(se); }
            }
        }
    } else {
        // Linear scan fallback (no tree built this frame)
        for (var i = 0u; i < count; i++) {
            let p = particles[i];
            if ((p.flags & FLAG_ALIVE) == 0u) { continue; }
            let dx = cx - p.posX;
            let dy = cy - p.posY;
            let distSq = dx * dx + dy * dy;
            let r = particleAux[i].radius;
            if (distSq < r * r && distSq < bestDistSq) {
                bestDistSq = distSq;
                bestIdx = i32(i);
            }
        }
    }

    hitResult[0] = bestIdx;
}
