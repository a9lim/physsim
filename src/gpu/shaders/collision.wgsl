// ─── Collision Detection & Resolution ───
// detectCollisions: tree-accelerated broadphase, writes (i,j) pairs to append buffer
// resolveCollisions: processes pairs — merge or annihilation
// detectCollisionsPairwise: O(N²) tiled broadphase (no tree required)
// resolveBouncePairwise: Hertz contact + friction impulse for bounce mode

// Constants provided by generated wgslConstants block.
// Shader-specific constants:
const NONE: i32 = -1;
const MAX_STACK: u32 = 48u;

// Merge event types
const MERGE_ANNIHILATION: u32 = 0u;
const MERGE_INELASTIC: u32 = 1u;

// Full topology-aware minimum-image displacement (inlined from common.wgsl)
fn fullMinImageCol(ox: f32, oy: f32, sx: f32, sy: f32, w: f32, h: f32, topo: u32) -> vec2<f32> {
    let halfW = w * 0.5;
    let halfH = h * 0.5;
    if (topo == 0u) {
        // Torus: simple periodic
        var rx = sx - ox;
        if (rx > halfW) { rx -= w; } else if (rx < -halfW) { rx += w; }
        var ry = sy - oy;
        if (ry > halfH) { ry -= h; } else if (ry < -halfH) { ry += h; }
        return vec2(rx, ry);
    }
    // Klein/RP²: evaluate glide-reflection candidates.
    // Candidate 0: only torus-wrap axes with translational (not glide) periodicity.
    var dx0 = sx - ox;
    var dy0 = sy - oy;
    if (topo == 1u) {
        if (dx0 > halfW) { dx0 -= w; } else if (dx0 < -halfW) { dx0 += w; }
    }
    var bestSq = dx0 * dx0 + dy0 * dy0;
    var bestDx = dx0;
    var bestDy = dy0;
    if (topo == 1u) {
        // Klein: y-wrap glide reflection
        let gx = w - sx;
        var dx1 = gx - ox;
        if (dx1 > halfW) { dx1 -= w; } else if (dx1 < -halfW) { dx1 += w; }
        var dy1 = (sy + h) - oy;
        if (dy1 > h) { dy1 -= 2.0 * h; } else if (dy1 < -h) { dy1 += 2.0 * h; }
        let dSq1 = dx1 * dx1 + dy1 * dy1;
        if (dSq1 < bestSq) { bestDx = dx1; bestDy = dy1; bestSq = dSq1; }
        var dy1b = (sy - h) - oy;
        if (dy1b > h) { dy1b -= 2.0 * h; } else if (dy1b < -h) { dy1b += 2.0 * h; }
        let dSq1b = dx1 * dx1 + dy1b * dy1b;
        if (dSq1b < bestSq) { bestDx = dx1; bestDy = dy1b; }
    } else {
        // RP²: both axes glide reflections (translational periods 2W, 2H)
        // Candidate 1: y-glide  (x,y) ~ (W-x, y+H) — x not wrapped
        let gx = w - sx;
        let dxG = gx - ox;
        var dyG = (sy + h) - oy;
        if (dyG > h) { dyG -= 2.0 * h; } else if (dyG < -h) { dyG += 2.0 * h; }
        let dSqG = dxG * dxG + dyG * dyG;
        if (dSqG < bestSq) { bestDx = dxG; bestDy = dyG; bestSq = dSqG; }
        // Candidate 2: x-glide  (x,y) ~ (x+W, H-y) — y not wrapped
        let gy = h - sy;
        var dxH = (sx + w) - ox;
        if (dxH > w) { dxH -= 2.0 * w; } else if (dxH < -w) { dxH += 2.0 * w; }
        let dyH = gy - oy;
        let dSqH = dxH * dxH + dyH * dyH;
        if (dSqH < bestSq) { bestDx = dxH; bestDy = dyH; bestSq = dSqH; }
        // Candidate 3: both glides  (x,y) ~ (2W-x, 2H-y)
        var dxC = (2.0 * w - sx) - ox;
        if (dxC > w) { dxC -= 2.0 * w; } else if (dxC < -w) { dxC += 2.0 * w; }
        var dyC = (2.0 * h - sy) - oy;
        if (dyC > h) { dyC -= 2.0 * h; } else if (dyC < -h) { dyC += 2.0 * h; }
        let dSqC = dxC * dxC + dyC * dyC;
        if (dSqC < bestSq) { bestDx = dxC; bestDy = dyC; }
    }
    return vec2(bestDx, bestDy);
}

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

// Packed AllForces struct (mirrors common.wgsl, only torques.w used here)
struct AllForces_Col {
    f0: vec4<f32>,
    f1: vec4<f32>,
    f2: vec4<f32>,
    f3: vec4<f32>,
    f4: vec4<f32>,
    f5: vec4<f32>,
    torques: vec4<f32>,
    bFields: vec4<f32>,
    bFieldGrads: vec4<f32>,
    totalForce: vec2<f32>,
    jerk: vec2<f32>,
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

@group(0) @binding(0) var<storage, read_write> nodes: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: SimUniforms;

// Group 1: packed particle structs (rw for resolve + encoder compat)
@group(1) @binding(0) var<storage, read_write> particleState: array<ParticleState>;
@group(1) @binding(1) var<storage, read_write> particleAux: array<ParticleAux>;
@group(1) @binding(2) var<storage, read_write> ghostOriginalIdx: array<u32>;

// Group 1 continued: force accumulators (for contact torque display)
@group(1) @binding(3) var<storage, read_write> allForces: array<AllForces_Col>;

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

    // Check both particles are still alive (another pair may have consumed one).
    // Race condition guard: use <= to catch partially-consumed particles from
    // concurrent merge threads modifying the same particle.
    if ((ps1.flags & FLAG_ALIVE) == 0u || (ps2.flags & FLAG_ALIVE) == 0u) { return; }
    if (ps1.mass <= EPSILON || ps2.mass <= EPSILON) { return; }

    let isAntimatter1 = (ps1.flags & FLAG_ANTIMATTER) != 0u;
    let isAntimatter2 = (ps2.flags & FLAG_ANTIMATTER) != 0u;

    var aux1 = particleAux[idx1];
    var aux2 = particleAux[idx2];

    let relOn = (uniforms.toggles0 & 32u) != 0u; // RELATIVITY_BIT

    // Minimum-image displacement from p1 to p2 (handles all topologies).
    let boundLoop = uniforms.boundaryMode == BOUND_LOOP;
    var dx12 = ps2.posX - ps1.posX;
    var dy12 = ps2.posY - ps1.posY;
    if (boundLoop) {
        let mi = fullMinImageCol(ps1.posX, ps1.posY, ps2.posX, ps2.posY,
                                 uniforms.domainW, uniforms.domainH, uniforms.topologyMode);
        dx12 = mi.x;
        dy12 = mi.y;
    }

    if (isAntimatter1 != isAntimatter2) {
        // Annihilation: matter + antimatter -> energy
        let annihilated = min(ps1.mass, ps2.mass);
        // Use minimum-image midpoint: p1 + dx12 * 0.5
        let cx = ps1.posX + dx12 * 0.5;
        let cy = ps1.posY + dy12 * 0.5;

        let fraction1 = annihilated / ps1.mass;
        let fraction2 = annihilated / ps2.mass;
        let keAnn = fraction1 * particleKE(ps1) + fraction2 * particleKE(ps2);

        // Store merge event for photon burst emission (handled by JS readback)
        let slot = atomicAdd(&mergeCounter, 1u);
        mergeResults[slot] = vec4<f32>(cx, cy, 2.0 * annihilated + keAnn, f32(MERGE_ANNIHILATION));

        ps1.mass -= annihilated;
        ps2.mass -= annihilated;
        if (ps1.mass > EPSILON) {
            ps1.baseMass *= ps1.mass / (ps1.mass + annihilated);
        } else {
            ps1.baseMass = 0.0;
        }
        if (ps2.mass > EPSILON) {
            ps2.baseMass *= ps2.mass / (ps2.mass + annihilated);
        } else {
            ps2.baseMass = 0.0;
        }

        // Retire particles with zero mass
        // Store pre-annihilation mass and coordinate angular velocity
        if (ps1.mass == 0.0) {
            ps1.flags = (ps1.flags & ~FLAG_ALIVE) | FLAG_RETIRED;
            aux1.deathTime = uniforms.simTime;
            aux1.deathMass = annihilated; // full pre-annihilation mass (was all consumed)
            let sr1 = ps1.angW * aux1.radius;
            aux1.deathAngVel = select(ps1.angW, ps1.angW / sqrt(1.0 + sr1 * sr1), relOn);
        }
        if (ps2.mass == 0.0) {
            ps2.flags = (ps2.flags & ~FLAG_ALIVE) | FLAG_RETIRED;
            aux2.deathTime = uniforms.simTime;
            aux2.deathMass = annihilated; // full pre-annihilation mass (was all consumed)
            let sr2 = ps2.angW * aux2.radius;
            aux2.deathAngVel = select(ps2.angW, ps2.angW / sqrt(1.0 + sr2 * sr2), relOn);
        }

        particleState[idx1] = ps1;
        particleState[idx2] = ps2;
        particleAux[idx1] = aux1;
        particleAux[idx2] = aux2;
    } else {
        // Inelastic merge: both particles die, new particle created in idx1's slot
        let keBefore = particleKE(ps1) + particleKE(ps2);

        let totalMass = ps1.mass + ps2.mass;
        // Guard against zero total mass (shouldn't happen, but protects against concurrent races)
        if (totalMass <= EPSILON) { return; }
        let invTotalMass = 1.0 / totalMass;
        let newWx = (ps1.mass * ps1.velWX + ps2.mass * ps2.velWX) * invTotalMass;
        let newWy = (ps1.mass * ps1.velWY + ps2.mass * ps2.velWY) * invTotalMass;
        // Use minimum-image displacement: newPos = p1 + dx12 * (m2 / totalMass)
        let newX = ps1.posX + dx12 * (ps2.mass * invTotalMass);
        let newY = ps1.posY + dy12 * (ps2.mass * invTotalMass);

        // Angular momentum conservation — arms relative to new CoM, using minimum-image positions.
        // p2_miPos = p1 + dx12 (minimum-image position of p2 relative to p1's frame)
        let p2miX = ps1.posX + dx12;
        let p2miY = ps1.posY + dy12;
        let dx1 = ps1.posX - newX;
        let dy1 = ps1.posY - newY;
        let dx2 = p2miX - newX;
        let dy2 = p2miY - newY;
        let Lorb = dx1 * (ps1.mass * ps1.velWY) - dy1 * (ps1.mass * ps1.velWX)
                 + dx2 * (ps2.mass * ps2.velWY) - dy2 * (ps2.mass * ps2.velWX);
        let r1 = aux1.radius;
        let r2 = aux2.radius;
        let Lspin = INERTIA_K * ps1.mass * r1 * r1 * ps1.angW
                   + INERTIA_K * ps2.mass * r2 * r2 * ps2.angW;

        // Retire p2 — save death metadata before zeroing mass
        aux2.deathTime = uniforms.simTime;
        aux2.deathMass = ps2.mass;
        let sr2 = ps2.angW * r2;
        aux2.deathAngVel = select(ps2.angW, ps2.angW / sqrt(1.0 + sr2 * sr2), relOn);

        // Write new merged particle to idx1's slot (fresh identity, both parents die)
        let newRadius = pow(totalMass, 1.0 / 3.0);
        let newI = INERTIA_K * totalMass * newRadius * newRadius;
        var newAngW = 0.0;
        if (newI > EPSILON) {
            newAngW = (Lorb + Lspin) / newI;
        }

        var newPs: ParticleState;
        newPs.posX = newX;
        newPs.posY = newY;
        newPs.velWX = newWx;
        newPs.velWY = newWy;
        newPs.mass = totalMass;
        newPs.charge = ps1.charge + ps2.charge;
        newPs.angW = newAngW;
        newPs.baseMass = ps1.baseMass + ps2.baseMass;
        // Preserve antimatter flag from p1 unless BH mode (no hair); FLAG_REBORN signals history/trail reset
        let bhOn = (uniforms.toggles0 & 256u) != 0u; // BLACK_HOLE_BIT
        let amFlag = select(ps1.flags & FLAG_ANTIMATTER, 0u, bhOn);
        newPs.flags = FLAG_ALIVE | FLAG_REBORN | amFlag;

        // Fresh aux for new particle (unique ID, clean death state)
        aux1.radius = newRadius;
        aux1.particleId = 0x80000000u | (uniforms.frameCount * MAX_PARTICLES + idx1);
        aux1.deathTime = 1e30; // large sentinel — alive
        aux1.deathMass = 0.0;
        aux1.deathAngVel = 0.0;

        // Retire p2
        ps2.mass = 0.0;
        ps2.baseMass = 0.0;
        ps2.flags = (ps2.flags & ~FLAG_ALIVE) | FLAG_RETIRED;

        particleState[idx1] = newPs;
        particleState[idx2] = ps2;
        particleAux[idx1] = aux1;
        particleAux[idx2] = aux2;

        // Compute KE lost for field excitation
        let keAfter = particleKE(newPs);
        let keLost = max(0.0, keBefore - keAfter);
        if (keLost > 0.0) {
            let slot = atomicAdd(&mergeCounter, 1u);
            mergeResults[slot] = vec4<f32>(newX, newY, keLost, f32(MERGE_INELASTIC));
        }
    }
}

// ─── detectCollisionsPairwise ───
// O(N²) tiled broadphase — no tree required.
// One thread per particle (observer). Scans all j > i to find overlapping pairs.
// Shared TILE_SIZE tile loaded collaboratively per tile (same pattern as pair-force.wgsl).
// Works for both COL_MERGE and COL_BOUNCE.

const TILE_SIZE_COL: u32 = 64u;

struct ColTileEntry {
    posX: f32,
    posY: f32,
    radius: f32,
    valid: u32,   // 1 = alive real particle, 0 = padding
};

var<workgroup> colTile: array<ColTileEntry, TILE_SIZE_COL>;

@compute @workgroup_size(TILE_SIZE_COL)
fn detectCollisionsPairwise(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let pIdx = gid.x;
    let localIdx = lid.x;
    let N = uniforms.aliveCount;

    // Load observer (this thread's particle)
    var alive = false;
    var pPosX: f32 = 0.0;
    var pPosY: f32 = 0.0;
    var pRadius: f32 = 0.0;
    if (pIdx < N) {
        let ps = particleState[pIdx];
        if ((ps.flags & FLAG_ALIVE) != 0u && (ps.flags & FLAG_GHOST) == 0u && ps.mass > 0.0) {
            alive = true;
            pPosX = ps.posX;
            pPosY = ps.posY;
            pRadius = particleAux[pIdx].radius;
        }
    }

    let boundLoop = uniforms.boundaryMode == 2u; // BOUND_LOOP
    let hw = uniforms.domainW * 0.5;
    let hh = uniforms.domainH * 0.5;

    let numTiles = (N + TILE_SIZE_COL - 1u) / TILE_SIZE_COL;

    for (var t: u32 = 0u; t < numTiles; t++) {
        // Collaborative tile load
        let srcIdx = t * TILE_SIZE_COL + localIdx;
        if (srcIdx < N) {
            let sps = particleState[srcIdx];
            let isReal = (sps.flags & FLAG_ALIVE) != 0u
                      && (sps.flags & FLAG_GHOST) == 0u
                      && sps.mass > 0.0;
            colTile[localIdx].posX   = sps.posX;
            colTile[localIdx].posY   = sps.posY;
            colTile[localIdx].radius = particleAux[srcIdx].radius;
            colTile[localIdx].valid  = select(0u, 1u, isReal);
        } else {
            colTile[localIdx].valid = 0u;
        }

        workgroupBarrier();

        // Scan tile: only emit pair when pIdx < srcIdx (unique pairs, lower index emits)
        if (alive) {
            for (var j: u32 = 0u; j < TILE_SIZE_COL; j++) {
                let sIdx = t * TILE_SIZE_COL + j;
                if (sIdx >= N || sIdx <= pIdx) { continue; }
                if (colTile[j].valid == 0u) { continue; }

                var dx = colTile[j].posX - pPosX;
                var dy = colTile[j].posY - pPosY;
                if (boundLoop) {
                    let mi = fullMinImageCol(pPosX, pPosY, colTile[j].posX, colTile[j].posY,
                                             uniforms.domainW, uniforms.domainH, uniforms.topologyMode);
                    dx = mi.x;
                    dy = mi.y;
                }

                let distSq = dx * dx + dy * dy;
                let minDist = pRadius + colTile[j].radius;

                if (distSq < minDist * minDist) {
                    let slot = atomicAdd(&pairCounter, 1u);
                    if (slot < arrayLength(&collisionPairs) / 2u) {
                        collisionPairs[slot * 2u]      = pIdx;
                        collisionPairs[slot * 2u + 1u] = sIdx;
                    }
                }
            }
        }

        workgroupBarrier();
    }
}

// ─── resolveBouncePairwise ───
// One thread per detected collision pair.
// Applies Hertz contact impulse + tangential friction to proper velocities (w) and angW.
// Matches CPU _repelPair() — F = delta^1.5, tangential friction with clamped vRel.
// NOTE: The CPU applies this as a force accumulated into the force step.  On the GPU the
// collision step runs AFTER Boris drift, so we apply the impulse directly to velW/angW
// (equivalent to a single-substep impulse p = F·dt with dt = uniforms.dt).

@compute @workgroup_size(64)
fn resolveBouncePairwise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pairIdx = gid.x;
    let numPairs = atomicLoad(&pairCounter);
    if (pairIdx >= numPairs) { return; }

    let idx1 = collisionPairs[pairIdx * 2u];
    let idx2 = collisionPairs[pairIdx * 2u + 1u];

    var ps1 = particleState[idx1];
    var ps2 = particleState[idx2];

    if ((ps1.flags & FLAG_ALIVE) == 0u || (ps2.flags & FLAG_ALIVE) == 0u) { return; }
    if (ps1.mass <= EPSILON || ps2.mass <= EPSILON) { return; }

    let aux1 = particleAux[idx1];
    let aux2 = particleAux[idx2];
    let r1 = max(aux1.radius, EPSILON);
    let r2 = max(aux2.radius, EPSILON);

    let boundLoop = uniforms.boundaryMode == BOUND_LOOP;
    var dx = ps2.posX - ps1.posX;
    var dy = ps2.posY - ps1.posY;
    if (boundLoop) {
        let mi = fullMinImageCol(ps1.posX, ps1.posY, ps2.posX, ps2.posY,
                                 uniforms.domainW, uniforms.domainH, uniforms.topologyMode);
        dx = mi.x;
        dy = mi.y;
    }

    let distSq = dx * dx + dy * dy;
    let minDist = r1 + r2;
    if (distSq >= minDist * minDist) { return; }

    let dist = sqrt(distSq);
    let delta = minDist - dist;
    if (delta <= 0.0) { return; }

    // Hertz normal force magnitude: F = delta^1.5
    let Fn = delta * sqrt(delta);
    let safeDist = max(dist, EPSILON);
    let nx = dx / safeDist;
    let ny = dy / safeDist;

    // Tangential direction (perpendicular to normal, y-down convention)
    let tx = -ny;
    let ty =  nx;

    let relOn = (uniforms.toggles0 & 32u) != 0u; // RELATIVITY_BIT

    // Coordinate velocities (vel = w / sqrt(1 + w²) when relativity on, else w)
    let w1Sq = ps1.velWX * ps1.velWX + ps1.velWY * ps1.velWY;
    let w2Sq = ps2.velWX * ps2.velWX + ps2.velWY * ps2.velWY;
    let inv1 = select(1.0, 1.0 / sqrt(1.0 + w1Sq), relOn);
    let inv2 = select(1.0, 1.0 / sqrt(1.0 + w2Sq), relOn);
    let v1x = ps1.velWX * inv1;
    let v1y = ps1.velWY * inv1;
    let v2x = ps2.velWX * inv2;
    let v2y = ps2.velWY * inv2;

    // Angular velocity: angVel = angW / sqrt(1 + (angW*r)²) when relativity on
    let sr1 = ps1.angW * r1;
    let sr2 = ps2.angW * r2;
    let av1 = select(ps1.angW, ps1.angW / sqrt(1.0 + sr1 * sr1), relOn);
    let av2 = select(ps2.angW, ps2.angW / sqrt(1.0 + sr2 * sr2), relOn);

    // Relative tangential velocity at contact point (surface velocity contribution)
    // v1t = v1·t + av1·r1,  v2t = v2·t - av2·r2  (sign: CW spin adds to contact surface vel)
    let v1t = v1x * tx + v1y * ty + av1 * r1;
    let v2t = v2x * tx + v2y * ty - av2 * r2;
    let vRel = v1t - v2t;

    let friction = uniforms.bounceFriction;
    let Ft = -friction * Fn * clamp(vRel * 10.0, -1.0, 1.0);

    // Total force on p1 in world space
    let F1x = -nx * Fn + tx * Ft;
    let F1y = -ny * Fn + ty * Ft;

    // Apply as impulse: Δw ≈ F·dt / m
    // (same as CPU which accumulates into force vectors then integrates with dt)
    let dt = uniforms.dt;
    let invM1 = select(0.0, 1.0 / ps1.mass, ps1.mass > EPSILON);
    let invM2 = select(0.0, 1.0 / ps2.mass, ps2.mass > EPSILON);

    ps1.velWX += F1x * dt * invM1;
    ps1.velWY += F1y * dt * invM1;
    ps2.velWX -= F1x * dt * invM2;  // Newton's 3rd law
    ps2.velWY -= F1y * dt * invM2;

    // Angular impulse: τ = r × F_t → ΔangW ≈ τ·dt / I,  I = INERTIA_K·m·r²
    // Contact torque on p1: +r1 * Ft,  on p2: -r2 * Ft
    if (friction > 0.0) {
        let I1 = INERTIA_K * ps1.mass * r1 * r1;
        let I2 = INERTIA_K * ps2.mass * r2 * r2;
        if (I1 > EPSILON) { ps1.angW += r1 * Ft * dt / I1; }
        if (I2 > EPSILON) { ps2.angW -= r2 * Ft * dt / I2; }
        // Record contact torque for display (torques.w)
        var af1 = allForces[idx1];
        af1.torques.w += r1 * Ft;
        allForces[idx1] = af1;
        var af2 = allForces[idx2];
        af2.torques.w -= r2 * Ft;
        allForces[idx2] = af2;
    }

    particleState[idx1] = ps1;
    particleState[idx2] = ps2;
}
