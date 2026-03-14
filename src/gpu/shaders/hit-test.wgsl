// hit-test.wgsl — GPU quadtree point query for particle selection
//
// Dispatched as (1, 1, 1) — single thread. Walks the quadtree built by
// tree-build.wgsl to find the particle closest to the click point within
// its radius. Writes the particle index (or -1) to hitResult.

struct HitUniforms {
    clickX: f32,
    clickY: f32,
    _pad0: f32,
    _pad1: f32,
};

struct QTNode {
    minX: f32, minY: f32, maxX: f32, maxY: f32,
    comX: f32, comY: f32,
    totalMass: f32, totalCharge: f32,
    totalMagMoment: f32, totalAngMomentum: f32,
    totalMomentumX: f32, totalMomentumY: f32,
    nw: i32, ne: i32, sw: i32, se: i32,
    particleIndex: i32,
    particleCount: u32,
};

@group(0) @binding(0) var<uniform> hit: HitUniforms;
@group(0) @binding(1) var<storage, read> tree: array<QTNode>;
@group(0) @binding(2) var<storage, read> posX: array<f32>;
@group(0) @binding(3) var<storage, read> posY: array<f32>;
@group(0) @binding(4) var<storage, read> radius: array<f32>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> hitResult: array<i32>;  // single element

const ALIVE_BIT: u32 = 1u;

fn pointInBox(px: f32, py: f32, node: QTNode) -> bool {
    return px >= node.minX && px <= node.maxX && py >= node.minY && py <= node.maxY;
}

@compute @workgroup_size(1)
fn main() {
    let cx = hit.clickX;
    let cy = hit.clickY;

    var bestIdx: i32 = -1;
    var bestDistSq: f32 = 1e30;

    // Stack-based tree walk — check all particles within their radius
    var stack: array<i32, 48>;
    var top: i32 = 0;
    stack[0] = 0;  // root node index

    loop {
        if (top < 0) { break; }
        let nodeIdx = stack[top];
        top -= 1;

        if (nodeIdx < 0) { continue; }
        let node = tree[nodeIdx];

        // Skip nodes whose bounding box doesn't contain click point
        // (expanded by max possible radius — conservative)
        if (cx < node.minX - 4.0 || cx > node.maxX + 4.0 ||
            cy < node.minY - 4.0 || cy > node.maxY + 4.0) {
            continue;
        }

        // Leaf node — check particle
        if (node.particleIndex >= 0) {
            let pi = u32(node.particleIndex);
            if ((flags[pi] & ALIVE_BIT) != 0u) {
                let dx = cx - posX[pi];
                let dy = cy - posY[pi];
                let distSq = dx * dx + dy * dy;
                let r = radius[pi];
                if (distSq < r * r && distSq < bestDistSq) {
                    bestDistSq = distSq;
                    bestIdx = node.particleIndex;
                }
            }
            continue;
        }

        // Internal node — push children
        if (node.nw >= 0) { top += 1; stack[top] = node.nw; }
        if (node.ne >= 0) { top += 1; stack[top] = node.ne; }
        if (node.sw >= 0) { top += 1; stack[top] = node.sw; }
        if (node.se >= 0) { top += 1; stack[top] = node.se; }
    }

    hitResult[0] = bestIdx;
}
