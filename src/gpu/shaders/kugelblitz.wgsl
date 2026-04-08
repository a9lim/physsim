// ─── Kugelblitz Collapse ───
// Detect boson energy concentrations exceeding the hoop conjecture threshold
// (E > r/2 in natural units) and condense them into a new massive particle.
//
// Runs after boson tree aggregation. One thread per tree node checks the
// collapse condition. The deepest (smallest) qualifying node wins via
// child-disqualification: a node only collapses if none of its children
// also qualify. Max 1 event per substep.
//
// Prepended with wgslConstants + shared-structs.wgsl + shared-topology.wgsl + shared-rng.wgsl.

// Boson tree node layout (same as boson-tree.wgsl)
const NODE_WORDS: u32 = 20u;
const NONE: i32 = -1;

const N_MIN_X: u32 = 0u;
const N_MIN_Y: u32 = 1u;
const N_MAX_X: u32 = 2u;
const N_MAX_Y: u32 = 3u;
const N_COM_X: u32 = 4u;
const N_COM_Y: u32 = 5u;
const N_TOTAL_MASS: u32 = 6u;
const N_TOTAL_CHARGE: u32 = 7u;
const N_PARTICLE_COUNT: u32 = 12u;
const N_DIVIDED: u32 = 13u;
const N_NW: u32 = 14u;
const N_NE: u32 = 15u;
const N_SW: u32 = 16u;
const N_SE: u32 = 17u;
const N_PARTICLE_IDX: u32 = 18u;

// Group 0: uniforms + boson tree (read_write for shared layout compat)
@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read_write> bosonTree: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> bosonNodeCounter: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> bosonVisitorFlags: array<atomic<u32>>;

// Group 1: photon pool
@group(1) @binding(0) var<storage, read_write> photons: array<Photon>;
@group(1) @binding(1) var<storage, read_write> phCount: atomic<u32>;

// Group 2: pion pool
@group(2) @binding(0) var<storage, read_write> pions: array<Pion>;
@group(2) @binding(1) var<storage, read_write> piCount: atomic<u32>;
@group(2) @binding(2) var<storage, read_write> pionClaims: array<atomic<u32>>;

// Group 3: kugelblitz event output
@group(3) @binding(0) var<storage, read_write> kbEvents: array<f32>;
@group(3) @binding(1) var<storage, read_write> kbCounter: atomic<u32>;

fn nodeF32(nodeIdx: u32, field: u32) -> f32 {
    return bitcast<f32>(atomicLoad(&bosonTree[nodeIdx * NODE_WORDS + field]));
}

fn nodeU32(nodeIdx: u32, field: u32) -> u32 {
    return atomicLoad(&bosonTree[nodeIdx * NODE_WORDS + field]);
}

// Check if a node meets the collapse condition.
// N_PARTICLE_COUNT on GPU is a visitor-flag protocol value, not a subtree boson count,
// so we skip the count check here and rely on energy + size (the physics condition).
fn qualifies(nodeIdx: u32) -> bool {
    let energy = nodeF32(nodeIdx, N_TOTAL_MASS);
    if (energy < MIN_KUGELBLITZ_ENERGY) { return false; }
    let sizeX = nodeF32(nodeIdx, N_MAX_X) - nodeF32(nodeIdx, N_MIN_X);
    let sizeY = nodeF32(nodeIdx, N_MAX_Y) - nodeF32(nodeIdx, N_MIN_Y);
    let nodeSize = max(sizeX, sizeY);
    return energy > nodeSize * 0.5;
}

// Check if any child of a divided node also qualifies (if so, this node defers)
fn childQualifies(nodeIdx: u32) -> bool {
    let nw = nodeU32(nodeIdx, N_NW);
    if (bitcast<i32>(nw) == NONE) { return false; }
    if (qualifies(nw)) { return true; }
    if (qualifies(nodeU32(nodeIdx, N_NE))) { return true; }
    if (qualifies(nodeU32(nodeIdx, N_SW))) { return true; }
    if (qualifies(nodeU32(nodeIdx, N_SE))) { return true; }
    return false;
}

@compute @workgroup_size(64)
fn checkKugelblitz(@builtin(global_invocation_id) gid: vec3u) {
    let nodeIdx = gid.x;
    let totalNodes = atomicLoad(&bosonNodeCounter);
    if (nodeIdx >= totalNodes) { return; }

    // Only the deepest qualifying node collapses
    if (!qualifies(nodeIdx)) { return; }
    if (childQualifies(nodeIdx)) { return; }

    // Claim the single event slot (max 1 per substep)
    let slot = atomicAdd(&kbCounter, 1u);
    if (slot > 0u) { return; } // another node won

    let phN = atomicLoad(&phCount);
    let piN = atomicLoad(&piCount);

    // Pass 1: Collect energy, momentum, charge — walk subtree
    var totalEnergy: f32 = 0.0;
    var totalPx: f32 = 0.0;
    var totalPy: f32 = 0.0;
    var totalCharge: f32 = 0.0;
    var comX: f32 = 0.0;
    var comY: f32 = 0.0;
    var bosonCount: f32 = 0.0;

    var stack: array<u32, 64>;
    var top: i32 = 0;
    stack[0] = nodeIdx;
    top = 1;

    while (top > 0) {
        top--;
        let nIdx = stack[top];
        let isDivided = bitcast<i32>(nodeU32(nIdx, N_NW)) != NONE;

        if (isDivided) {
            if (top + 4 <= 64) {
                stack[top] = nodeU32(nIdx, N_NW); top++;
                stack[top] = nodeU32(nIdx, N_NE); top++;
                stack[top] = nodeU32(nIdx, N_SW); top++;
                stack[top] = nodeU32(nIdx, N_SE); top++;
            }
        } else {
            // Leaf node — check each boson stored here
            let pidx = bitcast<i32>(nodeU32(nIdx, N_PARTICLE_IDX));
            if (pidx < 0) { continue; }
            let ui = u32(pidx);

            if (ui < phN) {
                if ((photons[ui].flags & 1u) != 0u) {
                    let e = photons[ui].energy;
                    totalEnergy += e;
                    comX += photons[ui].posX * e;
                    comY += photons[ui].posY * e;
                    totalPx += e * photons[ui].velX;
                    totalPy += e * photons[ui].velY;
                    bosonCount += 1.0;
                }
            } else {
                let pi = ui - phN;
                if (pi < piN && (pions[pi].flags & 1u) != 0u) {
                    let wx = pions[pi].wX; let wy = pions[pi].wY;
                    let gamma = sqrt(1.0 + wx * wx + wy * wy);
                    let e = pions[pi].mass * gamma;
                    totalEnergy += e;
                    comX += pions[pi].posX * e;
                    comY += pions[pi].posY * e;
                    totalPx += pions[pi].mass * wx;
                    totalPy += pions[pi].mass * wy;
                    totalCharge += pions[pi].charge;
                    bosonCount += 1.0;
                }
            }
        }
    }

    if (totalEnergy < MIN_KUGELBLITZ_ENERGY) { return; }

    // COM position
    let invE = 1.0 / totalEnergy;
    comX *= invE;
    comY *= invE;

    // Pass 2: Angular momentum about COM + kill bosons
    var totalAngL: f32 = 0.0;
    top = 0;
    stack[0] = nodeIdx;
    top = 1;

    while (top > 0) {
        top--;
        let nIdx = stack[top];
        let isDivided = bitcast<i32>(nodeU32(nIdx, N_NW)) != NONE;

        if (isDivided) {
            if (top + 4 <= 64) {
                stack[top] = nodeU32(nIdx, N_NW); top++;
                stack[top] = nodeU32(nIdx, N_NE); top++;
                stack[top] = nodeU32(nIdx, N_SW); top++;
                stack[top] = nodeU32(nIdx, N_SE); top++;
            }
        } else {
            let pidx = bitcast<i32>(nodeU32(nIdx, N_PARTICLE_IDX));
            if (pidx < 0) { continue; }
            let ui = u32(pidx);

            if (ui < phN) {
                if ((photons[ui].flags & 1u) != 0u) {
                    let e = photons[ui].energy;
                    let dx = photons[ui].posX - comX;
                    let dy = photons[ui].posY - comY;
                    totalAngL += dx * (e * photons[ui].velY) - dy * (e * photons[ui].velX);
                    // Kill photon
                    photons[ui].flags &= ~1u;
                }
            } else {
                let pi = ui - phN;
                if (pi < piN && (pions[pi].flags & 1u) != 0u) {
                    let dx = pions[pi].posX - comX;
                    let dy = pions[pi].posY - comY;
                    totalAngL += dx * (pions[pi].mass * pions[pi].wY) - dy * (pions[pi].mass * pions[pi].wX);
                    // Kill pion/lepton
                    pions[pi].flags &= ~1u;
                }
            }
        }
    }

    // Write event: x, y, px, py, energy, charge, angL, count
    kbEvents[0] = comX;
    kbEvents[1] = comY;
    kbEvents[2] = totalPx;
    kbEvents[3] = totalPy;
    kbEvents[4] = totalEnergy;
    kbEvents[5] = totalCharge;
    kbEvents[6] = totalAngL;
    kbEvents[7] = bosonCount;
}
