// ─── Boson Gravity Tree ───
// Build BH tree from alive photons + pions for boson gravity.
// Lightweight: only totalMass + CoM (no charge/dipole aggregates).
// Traversal shaders: particle<-boson gravity and boson<->boson mutual gravity.
//
// Standalone shader — defines own structs (NOT prepended with common.wgsl).

const MAX_PHOTONS: u32 = 512u;
const MAX_PIONS: u32 = 256u;
const BOSON_SOFTENING_SQ: f32 = 4.0;
const BH_THETA_SQ: f32 = 0.25; // 0.5^2
const EPSILON: f32 = 1e-9;
const MAX_DEPTH: u32 = 48u;

// ── Packed struct definitions ──

struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct AllForces {
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
    _pad: vec2<f32>,
};

struct Photon {
    posX: f32, posY: f32,
    velX: f32, velY: f32,
    energy: f32,
    emitterId: u32, age: u32, flags: u32,
};

struct Pion {
    posX: f32, posY: f32,
    wX: f32, wY: f32,
    mass: f32, charge: i32, energy: f32,
    emitterId: u32, age: u32, flags: u32,
    _pad0: u32, _pad1: u32,
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
};

// Boson tree node: same 20-word layout as particle tree but only mass+CoM used
const NODE_WORDS: u32 = 20u;

// Node field offsets
const N_MIN_X: u32 = 0u;
const N_MIN_Y: u32 = 1u;
const N_MAX_X: u32 = 2u;
const N_MAX_Y: u32 = 3u;
const N_COM_X: u32 = 4u;
const N_COM_Y: u32 = 5u;
const N_TOTAL_MASS: u32 = 6u;
const N_PARTICLE_COUNT: u32 = 12u;
const N_DIVIDED: u32 = 13u;
const N_NW: u32 = 14u;
const N_NE: u32 = 15u;
const N_SW: u32 = 16u;
const N_SE: u32 = 17u;
const N_PARTICLE_IDX: u32 = 18u;
const N_DEPTH: u32 = 19u;

// Particle flag bits
const FLAG_ALIVE: u32 = 1u;

@group(0) @binding(0) var<uniform> u: SimUniforms;

// Boson tree nodes (read-write for build, read-only for traversal)
@group(1) @binding(0) var<storage, read_write> bosonTree: array<u32>;
@group(1) @binding(1) var<storage, read_write> bosonNodeCounter: atomic<u32>;

// Photon pool (packed)
@group(2) @binding(0) var<storage, read_write> photons: array<Photon>;
@group(2) @binding(1) var<storage, read_write> phCount: atomic<u32>;

// Pion pool (packed)
@group(3) @binding(0) var<storage, read_write> pions: array<Pion>;
@group(3) @binding(1) var<storage, read_write> piCount: atomic<u32>;

// Particle state + force accumulators
@group(4) @binding(0) var<storage, read> particles: array<ParticleState>;
@group(4) @binding(1) var<storage, read_write> allForces: array<AllForces>;

// Helper: read f32 from tree node
fn nodeF32(nodeIdx: u32, field: u32) -> f32 {
    return bitcast<f32>(bosonTree[nodeIdx * NODE_WORDS + field]);
}

// Helper: read u32 from tree node
fn nodeU32(nodeIdx: u32, field: u32) -> u32 {
    return bosonTree[nodeIdx * NODE_WORDS + field];
}

// Helper: write f32 to tree node
fn setNodeF32(nodeIdx: u32, field: u32, val: f32) {
    bosonTree[nodeIdx * NODE_WORDS + field] = bitcast<u32>(val);
}

// Helper: write u32 to tree node
fn setNodeU32(nodeIdx: u32, field: u32, val: u32) {
    bosonTree[nodeIdx * NODE_WORDS + field] = val;
}

// Allocate a new tree node, returns index
fn allocNode() -> u32 {
    return atomicAdd(&bosonNodeCounter, 1u);
}

// Initialize a node with bounds
fn initNode(idx: u32, minX: f32, minY: f32, maxX: f32, maxY: f32, depth: u32) {
    let base = idx * NODE_WORDS;
    bosonTree[base + N_MIN_X] = bitcast<u32>(minX);
    bosonTree[base + N_MIN_Y] = bitcast<u32>(minY);
    bosonTree[base + N_MAX_X] = bitcast<u32>(maxX);
    bosonTree[base + N_MAX_Y] = bitcast<u32>(maxY);
    bosonTree[base + N_COM_X] = bitcast<u32>(0.0);
    bosonTree[base + N_COM_Y] = bitcast<u32>(0.0);
    bosonTree[base + N_TOTAL_MASS] = bitcast<u32>(0.0);
    bosonTree[base + N_PARTICLE_COUNT] = 0u;
    bosonTree[base + N_DIVIDED] = 0u;
    bosonTree[base + N_NW] = 0u;
    bosonTree[base + N_NE] = 0u;
    bosonTree[base + N_SW] = 0u;
    bosonTree[base + N_SE] = 0u;
    bosonTree[base + N_PARTICLE_IDX] = 0xFFFFFFFFu;
    bosonTree[base + N_DEPTH] = depth;
}

// Insert a boson into the tree (iterative)
@compute @workgroup_size(64)
fn insertBosonsIntoTree(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let phN = atomicLoad(&phCount);
    let piN = atomicLoad(&piCount);
    let total = phN + piN;
    if (i >= total) { return; }

    var bx: f32; var by: f32; var bMass: f32;
    if (i < phN) {
        // Photon: source mass = energy
        if ((photons[i].flags & 1u) == 0u) { return; }
        bx = photons[i].posX; by = photons[i].posY;
        bMass = photons[i].energy;
    } else {
        let pi = i - phN;
        if ((pions[pi].flags & 1u) == 0u) { return; }
        bx = pions[pi].posX; by = pions[pi].posY;
        // Pion source mass = gravMass = m * gamma
        let wx = pions[pi].wX; let wy = pions[pi].wY;
        let gamma = sqrt(1.0 + wx * wx + wy * wy);
        bMass = pions[pi].mass * gamma;
    }

    // Standard iterative tree insert (walk from root, subdivide on collision)
    var nodeIdx = 0u;
    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        let minX = nodeF32(nodeIdx, N_MIN_X);
        let minY = nodeF32(nodeIdx, N_MIN_Y);
        let maxX = nodeF32(nodeIdx, N_MAX_X);
        let maxY = nodeF32(nodeIdx, N_MAX_Y);
        let midX = (minX + maxX) * 0.5;
        let midY = (minY + maxY) * 0.5;

        // Determine quadrant
        let isEast = bx >= midX;
        let isSouth = by >= midY;

        if (nodeU32(nodeIdx, N_DIVIDED) != 0u) {
            // Already subdivided — descend into correct child
            var childField: u32;
            if (!isEast && !isSouth) { childField = N_NW; }
            else if (isEast && !isSouth) { childField = N_NE; }
            else if (!isEast && isSouth) { childField = N_SW; }
            else { childField = N_SE; }
            nodeIdx = nodeU32(nodeIdx, childField);
            continue;
        }

        let pCount = nodeU32(nodeIdx, N_PARTICLE_COUNT);
        if (pCount == 0u) {
            // Empty leaf — insert here
            setNodeU32(nodeIdx, N_PARTICLE_IDX, i);
            setNodeU32(nodeIdx, N_PARTICLE_COUNT, 1u);
            setNodeF32(nodeIdx, N_COM_X, bx);
            setNodeF32(nodeIdx, N_COM_Y, by);
            setNodeF32(nodeIdx, N_TOTAL_MASS, bMass);
            return;
        }

        // Leaf with one particle — subdivide
        let nw = allocNode();
        let ne = allocNode();
        let sw = allocNode();
        let se = allocNode();
        initNode(nw, minX, minY, midX, midY, depth + 1u);
        initNode(ne, midX, minY, maxX, midY, depth + 1u);
        initNode(sw, minX, midY, midX, maxY, depth + 1u);
        initNode(se, midX, midY, maxX, maxY, depth + 1u);
        setNodeU32(nodeIdx, N_NW, nw);
        setNodeU32(nodeIdx, N_NE, ne);
        setNodeU32(nodeIdx, N_SW, sw);
        setNodeU32(nodeIdx, N_SE, se);
        setNodeU32(nodeIdx, N_DIVIDED, 1u);

        // Re-insert the existing particle into the correct child
        let existIdx = nodeU32(nodeIdx, N_PARTICLE_IDX);
        let exX = nodeF32(nodeIdx, N_COM_X);
        let exY = nodeF32(nodeIdx, N_COM_Y);
        let exMass = nodeF32(nodeIdx, N_TOTAL_MASS);

        var exChild: u32;
        let exEast = exX >= midX;
        let exSouth = exY >= midY;
        if (!exEast && !exSouth) { exChild = nw; }
        else if (exEast && !exSouth) { exChild = ne; }
        else if (!exEast && exSouth) { exChild = sw; }
        else { exChild = se; }
        setNodeU32(exChild, N_PARTICLE_IDX, existIdx);
        setNodeU32(exChild, N_PARTICLE_COUNT, 1u);
        setNodeF32(exChild, N_COM_X, exX);
        setNodeF32(exChild, N_COM_Y, exY);
        setNodeF32(exChild, N_TOTAL_MASS, exMass);

        // Clear parent leaf data
        setNodeU32(nodeIdx, N_PARTICLE_IDX, 0xFFFFFFFFu);

        // Now descend for the new boson
        var newChildField: u32;
        if (!isEast && !isSouth) { newChildField = N_NW; }
        else if (isEast && !isSouth) { newChildField = N_NE; }
        else if (!isEast && isSouth) { newChildField = N_SW; }
        else { newChildField = N_SE; }
        nodeIdx = nodeU32(nodeIdx, newChildField);
    }
}

// Bottom-up aggregate: only totalMass + comX/comY
@compute @workgroup_size(64)
fn computeBosonAggregates(@builtin(global_invocation_id) gid: vec3u) {
    let nodeIdx = gid.x;
    let nodeCount = atomicLoad(&bosonNodeCounter);
    if (nodeIdx >= nodeCount) { return; }
    if (nodeU32(nodeIdx, N_DIVIDED) == 0u) { return; } // leaf, already has data

    // Aggregate from children
    var totalM: f32 = 0.0;
    var comX: f32 = 0.0;
    var comY: f32 = 0.0;

    let children = array<u32, 4>(
        nodeU32(nodeIdx, N_NW), nodeU32(nodeIdx, N_NE),
        nodeU32(nodeIdx, N_SW), nodeU32(nodeIdx, N_SE)
    );

    for (var c = 0u; c < 4u; c++) {
        let childIdx = children[c];
        let cMass = nodeF32(childIdx, N_TOTAL_MASS);
        if (cMass > EPSILON) {
            comX += cMass * nodeF32(childIdx, N_COM_X);
            comY += cMass * nodeF32(childIdx, N_COM_Y);
            totalM += cMass;
        }
    }

    if (totalM > EPSILON) {
        let invM = 1.0 / totalM;
        setNodeF32(nodeIdx, N_COM_X, comX * invM);
        setNodeF32(nodeIdx, N_COM_Y, comY * invM);
    }
    setNodeF32(nodeIdx, N_TOTAL_MASS, totalM);
    setNodeU32(nodeIdx, N_PARTICLE_COUNT, 1u); // mark as having data
}

// Gravitational force from bosons onto particles via BH tree walk.
@compute @workgroup_size(64)
fn computeBosonGravity(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((particles[i].flags & FLAG_ALIVE) == 0u) { return; }

    let px = particles[i].posX; let py = particles[i].posY;
    let pMass = particles[i].mass;

    var fx: f32 = 0.0; var fy: f32 = 0.0;

    // Stack-based BH tree walk (boson tree)
    var stack: array<u32, 48>;
    var top: i32 = 0;
    stack[0] = 0u; // root
    top = 1;

    while (top > 0) {
        top--;
        let nIdx = stack[top];
        let nodeMass = nodeF32(nIdx, N_TOTAL_MASS);
        if (nodeMass < EPSILON) { continue; }

        let cx = nodeF32(nIdx, N_COM_X);
        let cy = nodeF32(nIdx, N_COM_Y);
        let dx = cx - px;
        let dy = cy - py;
        let dSq = dx * dx + dy * dy;
        let size = nodeF32(nIdx, N_MAX_X) - nodeF32(nIdx, N_MIN_X);
        let isDivided = nodeU32(nIdx, N_DIVIDED) != 0u;

        if (!isDivided) {
            // Leaf: direct gravity with BOSON_SOFTENING_SQ
            let rSq = dSq + BOSON_SOFTENING_SQ;
            let invRSq = 1.0 / rSq;
            let f = pMass * nodeMass * sqrt(invRSq) * invRSq;
            fx += dx * f;
            fy += dy * f;
        } else if (size * size < BH_THETA_SQ * dSq) {
            // Distant: use aggregate
            let rSq = dSq + BOSON_SOFTENING_SQ;
            let invRSq = 1.0 / rSq;
            let f = pMass * nodeMass * sqrt(invRSq) * invRSq;
            fx += dx * f;
            fy += dy * f;
        } else {
            // Open: push children
            stack[top] = nodeU32(nIdx, N_NW); top++;
            stack[top] = nodeU32(nIdx, N_NE); top++;
            stack[top] = nodeU32(nIdx, N_SW); top++;
            stack[top] = nodeU32(nIdx, N_SE); top++;
        }
    }

    // Add to gravity force accumulators (allForces.f0.xy = gravity)
    var af = allForces[i];
    af.f0.x += fx;
    af.f0.y += fy;
    allForces[i] = af;
}

// Mutual gravitational interaction between bosons via BH tree walk.
// GR receiver factors: 2 for photons (null geodesic), 1+v^2 for pions (massive).
@compute @workgroup_size(64)
fn applyBosonBosonGravity(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let phN = atomicLoad(&phCount);
    let piN = atomicLoad(&piCount);
    let total = phN + piN;
    if (i >= total) { return; }

    let dt = u.dt;
    var bx: f32; var by: f32; var grFactor: f32;

    if (i < phN) {
        if ((photons[i].flags & 1u) == 0u) { return; }
        bx = photons[i].posX; by = photons[i].posY;
        grFactor = 2.0; // photon: null geodesic
    } else {
        let pi = i - phN;
        if ((pions[pi].flags & 1u) == 0u) { return; }
        bx = pions[pi].posX; by = pions[pi].posY;
        let wx = pions[pi].wX; let wy = pions[pi].wY;
        let wSq = wx * wx + wy * wy;
        let vSq = wSq / (1.0 + wSq);
        grFactor = 1.0 + vSq; // massive: 1+v^2
    }

    // BH tree walk of boson tree
    var kx: f32 = 0.0; var ky: f32 = 0.0;

    var stack: array<u32, 48>;
    var top: i32 = 0;
    stack[0] = 0u;
    top = 1;

    while (top > 0) {
        top--;
        let nIdx = stack[top];
        let nodeMass = nodeF32(nIdx, N_TOTAL_MASS);
        if (nodeMass < EPSILON) { continue; }

        let cx = nodeF32(nIdx, N_COM_X);
        let cy = nodeF32(nIdx, N_COM_Y);
        let dx = cx - bx;
        let dy = cy - by;
        let dSq = dx * dx + dy * dy;
        let size = nodeF32(nIdx, N_MAX_X) - nodeF32(nIdx, N_MIN_X);
        let isDivided = nodeU32(nIdx, N_DIVIDED) != 0u;

        if (!isDivided) {
            let rSq = dSq + BOSON_SOFTENING_SQ;
            let invRSq = 1.0 / rSq;
            let f = grFactor * nodeMass * sqrt(invRSq) * invRSq * dt;
            kx += dx * f;
            ky += dy * f;
        } else if (size * size < BH_THETA_SQ * dSq) {
            let rSq = dSq + BOSON_SOFTENING_SQ;
            let invRSq = 1.0 / rSq;
            let f = grFactor * nodeMass * sqrt(invRSq) * invRSq * dt;
            kx += dx * f;
            ky += dy * f;
        } else {
            stack[top] = nodeU32(nIdx, N_NW); top++;
            stack[top] = nodeU32(nIdx, N_NE); top++;
            stack[top] = nodeU32(nIdx, N_SW); top++;
            stack[top] = nodeU32(nIdx, N_SE); top++;
        }
    }

    // Apply impulse
    if (i < phN) {
        photons[i].velX += kx;
        photons[i].velY += ky;
        // Renormalize photon to c=1
        let vSq = photons[i].velX * photons[i].velX + photons[i].velY * photons[i].velY;
        if (abs(vSq - 1.0) > 1e-6) {
            let v = sqrt(vSq);
            if (v > EPSILON) { photons[i].velX /= v; photons[i].velY /= v; }
        }
    } else {
        let pi = i - phN;
        pions[pi].wX += kx;
        pions[pi].wY += ky;
    }
}
