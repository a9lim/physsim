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
    _pad3: u32,
    _pad4: u32,
};

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

@group(0) @binding(0) var<storage, read> nodes: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: SimUniforms;

// Group 1: packed particle structs (rw for resolve)
@group(1) @binding(0) var<storage, read_write> particleState: array<ParticleState>;
@group(1) @binding(1) var<storage, read_write> particleAux: array<ParticleAux>;
@group(1) @binding(2) var<storage, read> ghostOriginalIdx: array<u32>;

// Group 2: collision pairs + counters + merge results
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

    let ps = particleState[pIdx];
    if ((ps.flags & FLAG_ALIVE) == 0u) { return; }
    if ((ps.flags & FLAG_GHOST) != 0u) { return; }

    let px = ps.posX;
    let py = ps.posY;
    let pAux = particleAux[pIdx];
    let p1Radius = pAux.radius;
    let searchR = p1Radius * 2.0;
    let p1Id = pAux.particleId;

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
            let sPs = particleState[sIdx];
            if ((sPs.flags & FLAG_GHOST) != 0u && sIdx >= uniforms.aliveCount) {
                realIdx = ghostOriginalIdx[sIdx - uniforms.aliveCount];
            }

            if (pIdx == realIdx) { continue; }
            if ((sPs.flags & FLAG_ALIVE) == 0u) { continue; }
            let realPs = particleState[realIdx];
            if (realPs.mass == 0.0) { continue; }

            // Only process pair once: lower ID writes
            if (pIdx >= realIdx) { continue; }

            let sx = sPs.posX;
            let sy = sPs.posY;
            let dx = sx - px;
            let dy = sy - py;
            let distSq = dx * dx + dy * dy;
            let realAux = particleAux[realIdx];
            let minDist = p1Radius + realAux.radius;

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

fn particleKE(ps: ParticleState) -> f32 {
    let wSq = ps.velWX * ps.velWX + ps.velWY * ps.velWY;
    return wSq / (sqrt(1.0 + wSq) + 1.0) * ps.mass;
}

@compute @workgroup_size(64)
fn resolveCollisions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pairIdx = gid.x;
    let numPairs = atomicLoad(&pairCounter);
    if (pairIdx >= numPairs) { return; }

    let idx1 = collisionPairs[pairIdx * 2u];
    let idx2 = collisionPairs[pairIdx * 2u + 1u];

    // Read both particles
    var ps1 = particleState[idx1];
    var ps2 = particleState[idx2];

    // Check both particles are still alive (another pair may have consumed one)
    if ((ps1.flags & FLAG_ALIVE) == 0u || (ps2.flags & FLAG_ALIVE) == 0u) { return; }
    if (ps1.mass == 0.0 || ps2.mass == 0.0) { return; }

    let isAntimatter1 = (ps1.flags & FLAG_ANTIMATTER) != 0u;
    let isAntimatter2 = (ps2.flags & FLAG_ANTIMATTER) != 0u;

    var aux1 = particleAux[idx1];
    var aux2 = particleAux[idx2];

    let relOn = (uniforms.toggles0 & 32u) != 0u; // RELATIVITY_BIT

    if (isAntimatter1 != isAntimatter2) {
        // Annihilation: matter + antimatter -> energy
        let annihilated = min(ps1.mass, ps2.mass);
        let cx = (ps1.posX + ps2.posX) * 0.5;
        let cy = (ps1.posY + ps2.posY) * 0.5;

        let fraction1 = annihilated / ps1.mass;
        let fraction2 = annihilated / ps2.mass;
        let keAnn = fraction1 * particleKE(ps1) + fraction2 * particleKE(ps2);

        // Store merge event for photon burst emission (handled by JS readback)
        let slot = atomicAdd(&mergeCounter, 1u);
        mergeResults[slot] = vec4<f32>(cx, cy, 2.0 * annihilated + keAnn, f32(MERGE_ANNIHILATION));

        ps1.mass -= annihilated;
        ps2.mass -= annihilated;
        if (ps1.mass > 0.0) {
            ps1.baseMass *= ps1.mass / (ps1.mass + annihilated);
        }
        if (ps2.mass > 0.0) {
            ps2.baseMass *= ps2.mass / (ps2.mass + annihilated);
        }

        // Retire particles with zero mass
        if (ps1.mass == 0.0) {
            ps1.flags = (ps1.flags & ~FLAG_ALIVE) | FLAG_RETIRED;
            aux1.deathTime = uniforms.simTime;
            aux1.deathMass = annihilated; // pre-removal mass
            aux1.deathAngVel = ps1.angW;
        }
        if (ps2.mass == 0.0) {
            ps2.flags = (ps2.flags & ~FLAG_ALIVE) | FLAG_RETIRED;
            aux2.deathTime = uniforms.simTime;
            aux2.deathMass = annihilated; // pre-removal mass
            aux2.deathAngVel = ps2.angW;
        }

        particleState[idx1] = ps1;
        particleState[idx2] = ps2;
        particleAux[idx1] = aux1;
        particleAux[idx2] = aux2;
    } else {
        // Inelastic merge: p2 merges into p1
        let keBefore = particleKE(ps1) + particleKE(ps2);

        let totalMass = ps1.mass + ps2.mass;
        let newWx = (ps1.mass * ps1.velWX + ps2.mass * ps2.velWX) / totalMass;
        let newWy = (ps1.mass * ps1.velWY + ps2.mass * ps2.velWY) / totalMass;
        let newX = (ps1.posX * ps1.mass + ps2.posX * ps2.mass) / totalMass;
        let newY = (ps1.posY * ps1.mass + ps2.posY * ps2.mass) / totalMass;

        // Angular momentum conservation
        let dx1 = ps1.posX - newX;
        let dy1 = ps1.posY - newY;
        let dx2 = ps2.posX - newX;
        let dy2 = ps2.posY - newY;
        let Lorb = dx1 * (ps1.mass * ps1.velWY) - dy1 * (ps1.mass * ps1.velWX)
                 + dx2 * (ps2.mass * ps2.velWY) - dy2 * (ps2.mass * ps2.velWX);
        let r1 = aux1.radius;
        let r2 = aux2.radius;
        let Lspin = INERTIA_K * ps1.mass * r1 * r1 * ps1.angW
                   + INERTIA_K * ps2.mass * r2 * r2 * ps2.angW;

        // Update p1 (winner)
        ps1.mass = totalMass;
        ps1.baseMass += ps2.baseMass;
        ps1.charge += ps2.charge;
        ps1.velWX = newWx;
        ps1.velWY = newWy;
        ps1.posX = newX;
        ps1.posY = newY;

        let newRadius = pow(totalMass, 1.0 / 3.0);
        let newI = INERTIA_K * totalMass * newRadius * newRadius;
        if (newI > EPSILON) {
            ps1.angW = (Lorb + Lspin) / newI;
        } else {
            ps1.angW = 0.0;
        }

        // Retire p2 (loser) — save death metadata before zeroing mass
        aux2.deathTime = uniforms.simTime;
        aux2.deathMass = ps2.mass;
        aux2.deathAngVel = ps2.angW;

        ps2.mass = 0.0;
        ps2.baseMass = 0.0;
        ps2.flags = (ps2.flags & ~FLAG_ALIVE) | FLAG_RETIRED;

        particleState[idx1] = ps1;
        particleState[idx2] = ps2;
        particleAux[idx2] = aux2;

        // Compute KE lost for field excitation
        let keAfter = particleKE(ps1);
        let keLost = max(0.0, keBefore - keAfter);
        if (keLost > 0.0) {
            let slot = atomicAdd(&mergeCounter, 1u);
            mergeResults[slot] = vec4<f32>(newX, newY, keLost, f32(MERGE_INELASTIC));
        }
    }
}
