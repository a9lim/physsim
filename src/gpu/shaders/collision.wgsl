// ─── Collision Detection & Resolution ───
// detectCollisions: tree-accelerated broadphase, writes (i,j) pairs to append buffer
// resolveCollisions: processes pairs — merge or annihilation

const NONE: i32 = -1;
const MAX_STACK: u32 = 48u;
const EPSILON: f32 = 1e-9;
const INERTIA_K: f32 = 0.4;
const FLAG_ALIVE: u32 = 1u;
const FLAG_RETIRED: u32 = 2u;
const FLAG_ANTIMATTER: u32 = 4u;
const FLAG_GHOST: u32 = 16u;
const COL_MERGE: u32 = 1u;

// Merge event types
const MERGE_ANNIHILATION: u32 = 0u;
const MERGE_INELASTIC: u32 = 1u;

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

@group(0) @binding(0) var<storage, read> nodes: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: SimUniforms;

@group(1) @binding(0) var<storage, read_write> posX: array<f32>;
@group(1) @binding(1) var<storage, read_write> posY: array<f32>;
@group(1) @binding(2) var<storage, read_write> velWX: array<f32>;
@group(1) @binding(3) var<storage, read_write> velWY: array<f32>;
@group(1) @binding(4) var<storage, read_write> angW_buf: array<f32>;
@group(1) @binding(5) var<storage, read_write> mass_buf: array<f32>;
@group(1) @binding(6) var<storage, read_write> baseMass_buf: array<f32>;
@group(1) @binding(7) var<storage, read_write> charge_buf: array<f32>;
@group(1) @binding(8) var<storage, read_write> flags_buf: array<atomic<u32>>;
@group(1) @binding(9) var<storage, read> radius_buf: array<f32>;
@group(1) @binding(10) var<storage, read> particleId_buf: array<u32>;

// Ghost->original mapping
@group(1) @binding(11) var<storage, read> ghostOriginalIdx: array<u32>;

@group(2) @binding(0) var<storage, read_write> collisionPairs: array<u32>;
@group(2) @binding(1) var<storage, read_write> pairCounter: atomic<u32>;
@group(2) @binding(2) var<storage, read_write> mergeResults: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read_write> mergeCounter: atomic<u32>;

// ─── detectCollisions ───
// Tree query: for each alive particle, find overlapping particles via tree walk
// Write unique pairs (i < j) to collision pair buffer

fn nodeIntersects(nodeIdx: u32, rx: f32, ry: f32, rw: f32, rh: f32) -> bool {
    return !(rx - rw > getMaxX(nodeIdx) ||
             rx + rw < getMinX(nodeIdx) ||
             ry - rh > getMaxY(nodeIdx) ||
             ry + rh < getMinY(nodeIdx));
}

@compute @workgroup_size(64)
fn detectCollisions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pIdx = gid.x;
    if (pIdx >= uniforms.aliveCount) { return; }
    if (uniforms.collisionMode != COL_MERGE) { return; }

    let f = atomicLoad(&flags_buf[pIdx]);
    if ((f & FLAG_ALIVE) == 0u) { return; }
    if ((f & FLAG_GHOST) != 0u) { return; }

    let px = posX[pIdx];
    let py = posY[pIdx];
    let p1Radius = radius_buf[pIdx];
    let searchR = p1Radius * 2.0;
    let p1Id = particleId_buf[pIdx];

    // Stack-based tree query (overlap search)
    var stack: array<u32, 48>;
    var stackTop: u32 = 0u;
    stack[0] = 0u;
    stackTop = 1u;

    loop {
        if (stackTop == 0u) { break; }
        stackTop -= 1u;
        let nodeIdx = stack[stackTop];

        if (!nodeIntersects(nodeIdx, px, py, searchR, searchR)) { continue; }

        let isLeaf = getNW(nodeIdx) == NONE;
        let pi = getParticleIndex(nodeIdx);

        if (isLeaf && pi >= 0) {
            let sIdx = u32(pi);

            // Resolve ghost -> original
            var realIdx: u32 = sIdx;
            let sFlags = atomicLoad(&flags_buf[sIdx]);
            if ((sFlags & FLAG_GHOST) != 0u && sIdx >= uniforms.aliveCount) {
                realIdx = ghostOriginalIdx[sIdx - uniforms.aliveCount];
            }

            if (pIdx == realIdx) { continue; }
            if ((sFlags & FLAG_ALIVE) == 0u) { continue; }
            if (mass_buf[realIdx] == 0.0) { continue; }

            // Only process pair once: lower ID writes
            if (pIdx >= realIdx) { continue; }

            let sx = posX[sIdx];
            let sy = posY[sIdx];
            let dx = sx - px;
            let dy = sy - py;
            let distSq = dx * dx + dy * dy;
            let minDist = p1Radius + radius_buf[realIdx];

            if (distSq < minDist * minDist) {
                let slot = atomicAdd(&pairCounter, 1u);
                if (slot < arrayLength(&collisionPairs) / 2u) {
                    collisionPairs[slot * 2u] = pIdx;
                    collisionPairs[slot * 2u + 1u] = realIdx;
                }
            }
        } else if (!isLeaf) {
            if (stackTop + 4u <= MAX_STACK) {
                stack[stackTop] = u32(getNW(nodeIdx)); stackTop += 1u;
                stack[stackTop] = u32(getNE(nodeIdx)); stackTop += 1u;
                stack[stackTop] = u32(getSW(nodeIdx)); stackTop += 1u;
                stack[stackTop] = u32(getSE(nodeIdx)); stackTop += 1u;
            }
        }
    }
}

// ─── resolveCollisions ───
// One thread per detected pair. Resolves merge or annihilation.
// Uses atomicExchange on flags to prevent double-merging.

fn particleKE(idx: u32) -> f32 {
    let wx = velWX[idx];
    let wy = velWY[idx];
    let wSq = wx * wx + wy * wy;
    return wSq / (sqrt(1.0 + wSq) + 1.0) * mass_buf[idx];
}

@compute @workgroup_size(64)
fn resolveCollisions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pairIdx = gid.x;
    let numPairs = atomicLoad(&pairCounter);
    if (pairIdx >= numPairs) { return; }

    let idx1 = collisionPairs[pairIdx * 2u];
    let idx2 = collisionPairs[pairIdx * 2u + 1u];

    // Check both particles are still alive (another pair may have consumed one)
    let f1 = atomicLoad(&flags_buf[idx1]);
    let f2 = atomicLoad(&flags_buf[idx2]);
    if ((f1 & FLAG_ALIVE) == 0u || (f2 & FLAG_ALIVE) == 0u) { return; }
    if (mass_buf[idx1] == 0.0 || mass_buf[idx2] == 0.0) { return; }

    let isAntimatter1 = (f1 & FLAG_ANTIMATTER) != 0u;
    let isAntimatter2 = (f2 & FLAG_ANTIMATTER) != 0u;

    let relOn = (uniforms.toggles0 & 32u) != 0u; // RELATIVITY_BIT

    if (isAntimatter1 != isAntimatter2) {
        // Annihilation: matter + antimatter -> energy
        let annihilated = min(mass_buf[idx1], mass_buf[idx2]);
        let cx = (posX[idx1] + posX[idx2]) * 0.5;
        let cy = (posY[idx1] + posY[idx2]) * 0.5;

        let fraction1 = annihilated / mass_buf[idx1];
        let fraction2 = annihilated / mass_buf[idx2];
        let keAnn = fraction1 * particleKE(idx1) + fraction2 * particleKE(idx2);

        // Store merge event for photon burst emission (handled by JS readback)
        let slot = atomicAdd(&mergeCounter, 1u);
        mergeResults[slot] = vec4<f32>(cx, cy, 2.0 * annihilated + keAnn, f32(MERGE_ANNIHILATION));

        mass_buf[idx1] -= annihilated;
        mass_buf[idx2] -= annihilated;
        if (mass_buf[idx1] > 0.0) {
            baseMass_buf[idx1] *= mass_buf[idx1] / (mass_buf[idx1] + annihilated);
        }
        if (mass_buf[idx2] > 0.0) {
            baseMass_buf[idx2] *= mass_buf[idx2] / (mass_buf[idx2] + annihilated);
        }

        // Retire particles with zero mass
        if (mass_buf[idx1] == 0.0) {
            atomicAnd(&flags_buf[idx1], ~FLAG_ALIVE);
            atomicOr(&flags_buf[idx1], FLAG_RETIRED);
        }
        if (mass_buf[idx2] == 0.0) {
            atomicAnd(&flags_buf[idx2], ~FLAG_ALIVE);
            atomicOr(&flags_buf[idx2], FLAG_RETIRED);
        }
    } else {
        // Inelastic merge: p2 merges into p1
        let keBefore = particleKE(idx1) + particleKE(idx2);

        let totalMass = mass_buf[idx1] + mass_buf[idx2];
        let newWx = (mass_buf[idx1] * velWX[idx1] + mass_buf[idx2] * velWX[idx2]) / totalMass;
        let newWy = (mass_buf[idx1] * velWY[idx1] + mass_buf[idx2] * velWY[idx2]) / totalMass;
        let dx = posX[idx2] - posX[idx1];
        let dy = posY[idx2] - posY[idx1];
        let newX = (posX[idx1] * mass_buf[idx1] + posX[idx2] * mass_buf[idx2]) / totalMass;
        let newY = (posY[idx1] * mass_buf[idx1] + posY[idx2] * mass_buf[idx2]) / totalMass;

        // Angular momentum conservation
        let dx1 = posX[idx1] - newX;
        let dy1 = posY[idx1] - newY;
        let dx2 = posX[idx2] - newX;
        let dy2 = posY[idx2] - newY;
        let Lorb = dx1 * (mass_buf[idx1] * velWY[idx1]) - dy1 * (mass_buf[idx1] * velWX[idx1])
                 + dx2 * (mass_buf[idx2] * velWY[idx2]) - dy2 * (mass_buf[idx2] * velWX[idx2]);
        let r1 = radius_buf[idx1];
        let r2 = radius_buf[idx2];
        let Lspin = INERTIA_K * mass_buf[idx1] * r1 * r1 * angW_buf[idx1]
                   + INERTIA_K * mass_buf[idx2] * r2 * r2 * angW_buf[idx2];

        // Update p1 (winner)
        mass_buf[idx1] = totalMass;
        baseMass_buf[idx1] += baseMass_buf[idx2];
        charge_buf[idx1] += charge_buf[idx2];
        velWX[idx1] = newWx;
        velWY[idx1] = newWy;
        posX[idx1] = newX;
        posY[idx1] = newY;

        let newRadius = pow(totalMass, 1.0 / 3.0);
        let newI = INERTIA_K * totalMass * newRadius * newRadius;
        if (newI > EPSILON) {
            angW_buf[idx1] = (Lorb + Lspin) / newI;
        } else {
            angW_buf[idx1] = 0.0;
        }

        // Retire p2 (loser)
        mass_buf[idx2] = 0.0;
        baseMass_buf[idx2] = 0.0;
        atomicAnd(&flags_buf[idx2], ~FLAG_ALIVE);
        atomicOr(&flags_buf[idx2], FLAG_RETIRED);

        // Compute KE lost for field excitation
        let keAfter = particleKE(idx1);
        let keLost = max(0.0, keBefore - keAfter);
        if (keLost > 0.0) {
            let slot = atomicAdd(&mergeCounter, 1u);
            mergeResults[slot] = vec4<f32>(newX, newY, keLost, f32(MERGE_INELASTIC));
        }
    }
}
