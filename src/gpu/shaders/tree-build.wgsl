// ─── GPU Quadtree Build (GraphWaGu-style) ───
// Three dispatches:
//   1. computeBounds — parallel min/max reduction → root node bounds
//   2. initRoot — single-thread: read bounds, write root node
//   3. insertParticles — one thread per particle, CAS lock insertion
//   4. computeAggregates — bottom-up aggregation via visitor flags

// Constants provided by generated wgslConstants block.
// Shader-specific constants:
const NONE: i32 = -1;
const QT_CAPACITY: u32 = 1u; // Leaves hold 1 particle (GPU simplification for lock-free)
const LOCK_BIT: i32 = -2147483648; // 0x80000000 (MSB)

// Fixed-point scale for atomic min/max (f32 → i32: multiply by 2^16)
const FP_SCALE: f32 = 65536.0;
const FP_INV_SCALE: f32 = 0.0000152587890625; // 1/65536

// Struct definitions (ParticleState, ParticleDerived, SimUniforms) provided by shared-structs.wgsl.

// Node layout in flat buffer: 20 u32 words per node
// [0..3]: minX, minY, maxX, maxY (as f32 reinterpreted)
// [4..5]: comX, comY
// [6]: totalMass
// [7]: totalCharge
// [8]: totalMagMoment
// [9]: totalAngMomentum
// [10]: totalMomentumX
// [11]: totalMomentumY
// [12..15]: nw, ne, sw, se (as i32)
// [16]: particleIndex (i32, with LOCK_BIT for CAS)
// [17]: particleCount (u32)
// [18]: parentIndex (i32)
// [19]: padding

@group(0) @binding(0) var<storage, read_write> nodes: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> nodeCounter: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> bounds: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> visitorFlags: array<atomic<u32>>;

// Group 1: packed particle state + derived + aux (read_write for encoder compat)
@group(1) @binding(0) var<storage, read_write> particleState: array<ParticleState>;
@group(1) @binding(1) var<storage, read_write> derived_in: array<ParticleDerived>;
@group(1) @binding(2) var<storage, read_write> particleAux: array<ParticleAux>;

@group(2) @binding(0) var<uniform> uniforms: SimUniforms;

// ── Node accessors ──
const NODE_STRIDE: u32 = 20u;

fn nodeOffset(idx: u32) -> u32 { return idx * NODE_STRIDE; }

fn getMinX(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx)])); }
fn getMinY(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx) + 1u])); }
fn getMaxX(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx) + 2u])); }
fn getMaxY(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx) + 3u])); }

fn setMinX(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx)], bitcast<u32>(v)); }
fn setMinY(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx) + 1u], bitcast<u32>(v)); }
fn setMaxX(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx) + 2u], bitcast<u32>(v)); }
fn setMaxY(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx) + 3u], bitcast<u32>(v)); }

fn getComX(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx) + 4u])); }
fn getComY(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx) + 5u])); }
fn setComX(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx) + 4u], bitcast<u32>(v)); }
fn setComY(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx) + 5u], bitcast<u32>(v)); }

fn getTotalMass(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx) + 6u])); }
fn setTotalMass(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx) + 6u], bitcast<u32>(v)); }

fn getTotalCharge(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx) + 7u])); }
fn setTotalCharge(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx) + 7u], bitcast<u32>(v)); }

fn getTotalMagMoment(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx) + 8u])); }
fn setTotalMagMoment(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx) + 8u], bitcast<u32>(v)); }

fn getTotalAngMomentum(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx) + 9u])); }
fn setTotalAngMomentum(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx) + 9u], bitcast<u32>(v)); }

fn getTotalMomX(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx) + 10u])); }
fn setTotalMomX(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx) + 10u], bitcast<u32>(v)); }

fn getTotalMomY(idx: u32) -> f32 { return bitcast<f32>(atomicLoad(&nodes[nodeOffset(idx) + 11u])); }
fn setTotalMomY(idx: u32, v: f32) { atomicStore(&nodes[nodeOffset(idx) + 11u], bitcast<u32>(v)); }

fn getNW(idx: u32) -> i32 { return bitcast<i32>(atomicLoad(&nodes[nodeOffset(idx) + 12u])); }
fn getNE(idx: u32) -> i32 { return bitcast<i32>(atomicLoad(&nodes[nodeOffset(idx) + 13u])); }
fn getSW(idx: u32) -> i32 { return bitcast<i32>(atomicLoad(&nodes[nodeOffset(idx) + 14u])); }
fn getSE(idx: u32) -> i32 { return bitcast<i32>(atomicLoad(&nodes[nodeOffset(idx) + 15u])); }

fn setNW(idx: u32, v: i32) { atomicStore(&nodes[nodeOffset(idx) + 12u], bitcast<u32>(v)); }
fn setNE(idx: u32, v: i32) { atomicStore(&nodes[nodeOffset(idx) + 13u], bitcast<u32>(v)); }
fn setSW(idx: u32, v: i32) { atomicStore(&nodes[nodeOffset(idx) + 14u], bitcast<u32>(v)); }
fn setSE(idx: u32, v: i32) { atomicStore(&nodes[nodeOffset(idx) + 15u], bitcast<u32>(v)); }

fn getParticleIndex(idx: u32) -> i32 { return bitcast<i32>(atomicLoad(&nodes[nodeOffset(idx) + 16u])); }
fn getParticleCount(idx: u32) -> u32 { return atomicLoad(&nodes[nodeOffset(idx) + 17u]); }

fn getParentIndex(idx: u32) -> i32 { return bitcast<i32>(atomicLoad(&nodes[nodeOffset(idx) + 18u])); }
fn setParentIndex(idx: u32, v: i32) { atomicStore(&nodes[nodeOffset(idx) + 18u], bitcast<u32>(v)); }

// Atomic CAS on particleIndex field (for lock protocol)
fn casParticleIndex(idx: u32, expected: i32, desired: i32) -> i32 {
    let offset = nodeOffset(idx) + 16u;
    let result = atomicCompareExchangeWeak(&nodes[offset], bitcast<u32>(expected), bitcast<u32>(desired));
    return bitcast<i32>(result.old_value);
}

fn setParticleIndex(idx: u32, v: i32) {
    atomicStore(&nodes[nodeOffset(idx) + 16u], bitcast<u32>(v));
}

fn setParticleCount(idx: u32, v: u32) {
    atomicStore(&nodes[nodeOffset(idx) + 17u], v);
}

fn atomicAddParticleCount(idx: u32, delta: u32) -> u32 {
    return atomicAdd(&nodes[nodeOffset(idx) + 17u], delta);
}

fn allocNode() -> u32 {
    return atomicAdd(&nodeCounter, 1u);
}

// ─── Dispatch 1: computeBounds ───
// Parallel reduction over alive particles → fixed-point atomic min/max.
// bounds[0..3] = (minX, minY, maxX, maxY) as fixed-point i32.
// Initialize: bounds[0,1] = +MAX, bounds[2,3] = -MAX before dispatch.

var<workgroup> wgMinX: atomic<i32>;
var<workgroup> wgMinY: atomic<i32>;
var<workgroup> wgMaxX: atomic<i32>;
var<workgroup> wgMaxY: atomic<i32>;

@compute @workgroup_size(256)
fn computeBounds(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = gid.x;
    let n = uniforms.particleCount; // totalCount = aliveCount + ghostCount

    // Initialize workgroup atomics
    if (lid.x == 0u) {
        atomicStore(&wgMinX, 2147483647);  // i32 max
        atomicStore(&wgMinY, 2147483647);
        atomicStore(&wgMaxX, -2147483647); // i32 min
        atomicStore(&wgMaxY, -2147483647);
    }
    workgroupBarrier();

    if (idx < n) {
        let ps = particleState[idx];
        let isAliveOrGhost = (ps.flags & (FLAG_ALIVE | FLAG_GHOST)) != 0u;
        let isRetired = (ps.flags & FLAG_RETIRED) != 0u && (ps.flags & FLAG_ALIVE) == 0u;
        if (isAliveOrGhost || isRetired) {
            let px = i32(ps.posX * FP_SCALE);
            let py = i32(ps.posY * FP_SCALE);
            atomicMin(&wgMinX, px);
            atomicMin(&wgMinY, py);
            atomicMax(&wgMaxX, px);
            atomicMax(&wgMaxY, py);
        }
    }

    workgroupBarrier();

    // Workgroup leader writes to global atomics
    if (lid.x == 0u) {
        atomicMin(&bounds[0], atomicLoad(&wgMinX));
        atomicMin(&bounds[1], atomicLoad(&wgMinY));
        atomicMax(&bounds[2], atomicLoad(&wgMaxX));
        atomicMax(&bounds[3], atomicLoad(&wgMaxY));
    }
}

// ─── Dispatch 2: initRoot ───
// Single-thread: read fixed-point bounds, convert to f32, write root node.

@compute @workgroup_size(1)
fn initRoot() {
    let minX = f32(atomicLoad(&bounds[0])) * FP_INV_SCALE;
    let minY = f32(atomicLoad(&bounds[1])) * FP_INV_SCALE;
    let maxX = f32(atomicLoad(&bounds[2])) * FP_INV_SCALE;
    let maxY = f32(atomicLoad(&bounds[3])) * FP_INV_SCALE;

    // Add padding (10% margin like CPU boundary)
    let padX = (maxX - minX) * 0.1 + 1.0;
    let padY = (maxY - minY) * 0.1 + 1.0;

    setMinX(0u, minX - padX);
    setMinY(0u, minY - padY);
    setMaxX(0u, maxX + padX);
    setMaxY(0u, maxY + padY);
    setNW(0u, NONE);
    setNE(0u, NONE);
    setSW(0u, NONE);
    setSE(0u, NONE);
    setParticleIndex(0u, NONE);
    setParticleCount(0u, 0u);
    setParentIndex(0u, NONE);
    setTotalMass(0u, 0.0);
    setTotalCharge(0u, 0.0);
    setTotalMagMoment(0u, 0.0);
    setTotalAngMomentum(0u, 0.0);
    setTotalMomX(0u, 0.0);
    setTotalMomY(0u, 0.0);
}

// ─── Dispatch 3: insertParticles ───
// One thread per particle. Walk from root to appropriate leaf.
// CAS lock protocol for concurrent insertion.

fn childFor(nodeIdx: u32, px: f32, py: f32) -> u32 {
    let cx = (getMinX(nodeIdx) + getMaxX(nodeIdx)) * 0.5;
    let cy = (getMinY(nodeIdx) + getMaxY(nodeIdx)) * 0.5;
    if (py <= cy) {
        if (px <= cx) { return u32(getNW(nodeIdx)); }
        else { return u32(getNE(nodeIdx)); }
    } else {
        if (px <= cx) { return u32(getSW(nodeIdx)); }
        else { return u32(getSE(nodeIdx)); }
    }
}

fn subdivide(nodeIdx: u32) {
    let minX = getMinX(nodeIdx);
    let minY = getMinY(nodeIdx);
    let maxX = getMaxX(nodeIdx);
    let maxY = getMaxY(nodeIdx);
    let cx = (minX + maxX) * 0.5;
    let cy = (minY + maxY) * 0.5;

    let nw = allocNode();
    let ne = allocNode();
    let sw = allocNode();
    let se = allocNode();

    // Initialize NW
    setMinX(nw, minX); setMinY(nw, minY); setMaxX(nw, cx); setMaxY(nw, cy);
    setNW(nw, NONE); setNE(nw, NONE); setSW(nw, NONE); setSE(nw, NONE);
    setParticleIndex(nw, NONE); setParticleCount(nw, 0u);
    setParentIndex(nw, i32(nodeIdx));
    setTotalMass(nw, 0.0); setTotalCharge(nw, 0.0);
    setTotalMagMoment(nw, 0.0); setTotalAngMomentum(nw, 0.0);
    setTotalMomX(nw, 0.0); setTotalMomY(nw, 0.0);

    // Initialize NE
    setMinX(ne, cx); setMinY(ne, minY); setMaxX(ne, maxX); setMaxY(ne, cy);
    setNW(ne, NONE); setNE(ne, NONE); setSW(ne, NONE); setSE(ne, NONE);
    setParticleIndex(ne, NONE); setParticleCount(ne, 0u);
    setParentIndex(ne, i32(nodeIdx));
    setTotalMass(ne, 0.0); setTotalCharge(ne, 0.0);
    setTotalMagMoment(ne, 0.0); setTotalAngMomentum(ne, 0.0);
    setTotalMomX(ne, 0.0); setTotalMomY(ne, 0.0);

    // Initialize SW
    setMinX(sw, minX); setMinY(sw, cy); setMaxX(sw, cx); setMaxY(sw, maxY);
    setNW(sw, NONE); setNE(sw, NONE); setSW(sw, NONE); setSE(sw, NONE);
    setParticleIndex(sw, NONE); setParticleCount(sw, 0u);
    setParentIndex(sw, i32(nodeIdx));
    setTotalMass(sw, 0.0); setTotalCharge(sw, 0.0);
    setTotalMagMoment(sw, 0.0); setTotalAngMomentum(sw, 0.0);
    setTotalMomX(sw, 0.0); setTotalMomY(sw, 0.0);

    // Initialize SE
    setMinX(se, cx); setMinY(se, cy); setMaxX(se, maxX); setMaxY(se, maxY);
    setNW(se, NONE); setNE(se, NONE); setSW(se, NONE); setSE(se, NONE);
    setParticleIndex(se, NONE); setParticleCount(se, 0u);
    setParentIndex(se, i32(nodeIdx));
    setTotalMass(se, 0.0); setTotalCharge(se, 0.0);
    setTotalMagMoment(se, 0.0); setTotalAngMomentum(se, 0.0);
    setTotalMomX(se, 0.0); setTotalMomY(se, 0.0);

    setNW(nodeIdx, i32(nw));
    setNE(nodeIdx, i32(ne));
    setSW(nodeIdx, i32(sw));
    setSE(nodeIdx, i32(se));
}

@compute @workgroup_size(64)
fn insertParticles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pIdx = gid.x;
    let n = uniforms.particleCount; // totalCount
    if (pIdx >= n) { return; }

    let ps = particleState[pIdx];
    let isAliveOrGhost = (ps.flags & (FLAG_ALIVE | FLAG_GHOST)) != 0u;
    let isRetired = (ps.flags & FLAG_RETIRED) != 0u && (ps.flags & FLAG_ALIVE) == 0u;
    if (!isAliveOrGhost && !isRetired) { return; }

    let px = ps.posX;
    let py = ps.posY;
    var cur: u32 = 0u; // root is always node 0
    var depth: u32 = 0u;

    loop {
        if (depth >= MAX_DEPTH) {
            break; // depth guard — particle accepted at max depth
        }

        // Try to CAS particleIndex from NONE to our particle index (claim empty leaf)
        let prev = casParticleIndex(cur, NONE, i32(pIdx));

        if (prev == NONE) {
            // Successfully claimed empty leaf.
            // Walk up to root, incrementing ancestor leaf-descendant counts.
            // computeAggregates uses these counts to know how many visitors
            // to expect before it is safe to aggregate an internal node.
            setParticleCount(cur, 1u);
            var ancestor: i32 = getParentIndex(cur);
            loop {
                if (ancestor < 0) { break; }
                atomicAddParticleCount(u32(ancestor), 1u);
                ancestor = getParentIndex(u32(ancestor));
            }
            break;
        }

        // Node is occupied or being subdivided
        let prevUnlocked = prev & ~LOCK_BIT;

        if (prevUnlocked >= 0 && getNW(cur) == NONE) {
            // Occupied leaf with no children — try to lock for subdivision
            let lockResult = casParticleIndex(cur, prev, prev | LOCK_BIT);
            if (lockResult == prev) {
                // We got the lock — subdivide and reinsert displaced particle
                subdivide(cur);
                let displacedIdx = u32(prev);
                let displacedPs = particleState[displacedIdx];

                // Transition cur from leaf to internal: reset particleCount to 0.
                // The old particleCount=1 (from leaf claim) is stale — internal
                // nodes use particleCount as leaf-descendant count for aggregation.
                setParticleCount(cur, 0u);

                // Reinsert displaced particle into correct child.
                let childForDisplaced = childFor(cur, displacedPs.posX, displacedPs.posY);
                setParticleIndex(childForDisplaced, i32(displacedIdx));
                setParticleCount(childForDisplaced, 1u);
                // Count displaced particle as a leaf descendant of cur (and all
                // of cur's ancestors). Cur's ancestors were already incremented
                // by the displaced particle's original insertion — but that was
                // for a leaf at cur's level, not one level deeper. The net effect
                // is the same count (cur's ancestors still have +1 from the
                // original), so we only need to increment cur itself.
                atomicAddParticleCount(cur, 1u);

                // Mark current node as internal (clear particleIndex).
                setParticleIndex(cur, NONE);

                // Now descend into correct child for our particle
                cur = childFor(cur, px, py);
                depth += 1u;
                continue;
            }
            // Failed to lock — someone else is subdividing. Spin/retry.
        }

        // Node has children (internal node) or is being subdivided — descend
        if (getNW(cur) != NONE) {
            cur = childFor(cur, px, py);
            depth += 1u;
        }
        // If NW is still NONE, another thread is subdividing — spin
        // (bounded by depth guard)
    }
}

// ─── Dispatch 4: computeAggregates ───
// Each leaf thread walks to root via parentIndex.
// First visitor to an internal node sets flag and exits.
// Second visitor computes aggregate from children.

@compute @workgroup_size(64)
fn computeAggregates(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pIdx = gid.x;
    let n = uniforms.particleCount; // totalCount
    if (pIdx >= n) { return; }

    let ps = particleState[pIdx];
    let isAliveOrGhost = (ps.flags & (FLAG_ALIVE | FLAG_GHOST)) != 0u;
    let isRetired = (ps.flags & FLAG_RETIRED) != 0u && (ps.flags & FLAG_ALIVE) == 0u;
    if (!isAliveOrGhost && !isRetired) { return; }

    // Find the leaf node containing this particle
    // Walk from root to leaf to find our node
    let px = ps.posX;
    let py = ps.posY;
    var leafNode: u32 = 0u;
    var depth: u32 = 0u;
    loop {
        if (depth >= MAX_DEPTH) { break; }
        if (getNW(leafNode) == NONE) { break; } // leaf
        leafNode = childFor(leafNode, px, py);
        depth += 1u;
    }

    // Write leaf aggregates (single particle)
    // Retired particles use deathMass/deathAngVel from aux buffer
    // (their mass is zeroed on merge, so ps.mass == 0)
    var m: f32;
    var q: f32;
    var mm: f32;
    var am: f32;
    var wx: f32;
    var wy: f32;

    if (isRetired) {
        let aux = particleAux[pIdx];
        m = aux.deathMass;
        q = ps.charge;
        let bodyRadSq = pow(m, 2.0 / 3.0);
        let dAngW = aux.deathAngVel;
        let dAngWSq = dAngW * dAngW;
        let dAngVel = dAngW / sqrt(1.0 + dAngWSq * bodyRadSq);
        mm = MAG_MOMENT_K * q * dAngVel * bodyRadSq;
        am = INERTIA_K * m * dAngVel * bodyRadSq;
        wx = 0.0;
        wy = 0.0;
    } else {
        m = ps.mass;
        q = ps.charge;
        let drvd = derived_in[pIdx];
        mm = drvd.magMoment;
        am = drvd.angMomentum;
        wx = ps.velWX;
        wy = ps.velWY;
    }

    setTotalMass(leafNode, m);
    setTotalCharge(leafNode, q);
    setTotalMagMoment(leafNode, mm);
    setTotalAngMomentum(leafNode, am);
    setTotalMomX(leafNode, m * wx);
    setTotalMomY(leafNode, m * wy);
    if (m > 0.0) {
        setComX(leafNode, px);
        setComY(leafNode, py);
    }

    // Walk up to root via parent indices
    var curNode: i32 = getParentIndex(leafNode);
    loop {
        if (curNode < 0) { break; } // reached root's parent (NONE)

        let nodeU = u32(curNode);
        // Each visitor increments the flag.  Only the last visitor (the one
        // that brings the count up to the number of populated children) is
        // allowed to aggregate and continue climbing.  All earlier visitors
        // must exit here so they do not read partially-written child data.
        //
        // `particleCount` on an internal node was repurposed during the
        // insertParticles pass to count how many of its 4 children received
        // at least one particle.  That tells us exactly how many threads will
        // walk through this node, so we can use it as the threshold.
        let expectedVisitors = getParticleCount(nodeU); // populated child count
        let prev = atomicAdd(&visitorFlags[nodeU], 1u);
        if (prev < expectedVisitors - 1u) {
            // Not the last visitor — exit, let the last one aggregate
            break;
        }

        // Last visitor — all populated children have written their data, aggregate
        let c0 = u32(getNW(nodeU));
        let c1 = u32(getNE(nodeU));
        let c2 = u32(getSW(nodeU));
        let c3 = u32(getSE(nodeU));

        let m0 = getTotalMass(c0);
        let m1 = getTotalMass(c1);
        let m2 = getTotalMass(c2);
        let m3 = getTotalMass(c3);
        let totalM = m0 + m1 + m2 + m3;

        setTotalMass(nodeU, totalM);
        setTotalCharge(nodeU, getTotalCharge(c0) + getTotalCharge(c1) + getTotalCharge(c2) + getTotalCharge(c3));
        setTotalMagMoment(nodeU, getTotalMagMoment(c0) + getTotalMagMoment(c1) + getTotalMagMoment(c2) + getTotalMagMoment(c3));
        setTotalAngMomentum(nodeU, getTotalAngMomentum(c0) + getTotalAngMomentum(c1) + getTotalAngMomentum(c2) + getTotalAngMomentum(c3));
        setTotalMomX(nodeU, getTotalMomX(c0) + getTotalMomX(c1) + getTotalMomX(c2) + getTotalMomX(c3));
        setTotalMomY(nodeU, getTotalMomY(c0) + getTotalMomY(c1) + getTotalMomY(c2) + getTotalMomY(c3));

        if (totalM > 0.0) {
            setComX(nodeU, (getComX(c0) * m0 + getComX(c1) * m1 + getComX(c2) * m2 + getComX(c3) * m3) / totalM);
            setComY(nodeU, (getComY(c0) * m0 + getComY(c1) * m1 + getComY(c2) * m2 + getComY(c3) * m3) / totalM);
        }

        curNode = getParentIndex(nodeU);
    }
}
