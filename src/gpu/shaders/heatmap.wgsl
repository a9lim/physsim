// ─── Potential Field Heatmap ───
// 128x128 grid potential from particles. Gravity, Coulomb, Yukawa contributions.
// Two entry points:
//   computeHeatmap:     O(N*GRID²) pairwise, with signal delay support
//   computeHeatmapTree: O(log(N)*GRID²) BH tree walk, no signal delay (non-periodic only)
// Dead/retired particles always use pairwise scan with signal delay history.

struct HeatmapUniforms {
    // Camera/viewport for world-space grid
    viewLeft: f32,
    viewTop: f32,
    cellW: f32,
    cellH: f32,
    // Physics params
    softeningSq: f32,
    yukawaCoupling: f32,
    yukawaMu: f32,
    simTime: f32,
    // Domain
    domainW: f32,
    domainH: f32,
    // Toggle bits
    doGravity: u32,
    doCoulomb: u32,
    doYukawa: u32,
    useDelay: u32,
    periodic: u32,
    topologyMode: u32,
    particleCount: u32,
    useTree: u32,   // 1 when BH tree available
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

// Group 0: particleState + particleAux + treeNodes (read_write for encoder compat)
@group(0) @binding(0) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(1) var<storage, read_write> particleAux: array<ParticleAux>;
@group(0) @binding(2) var<storage, read_write> nodes: array<u32>;

@group(1) @binding(0) var<storage, read_write> gravPotential: array<f32>;
@group(1) @binding(1) var<storage, read_write> elecPotential: array<f32>;
@group(1) @binding(2) var<storage, read_write> yukawaPotential: array<f32>;
@group(1) @binding(3) var<uniform> hu: HeatmapUniforms;

// Group 2: signal delay history (interleaved format, only used when useDelay != 0)
@group(2) @binding(0) var<storage, read_write> histData: array<f32>;
@group(2) @binding(1) var<storage, read_write> histMeta: array<u32>;

// Constants provided by generated wgslConstants block.
// Node accessors from shared-tree-nodes.wgsl (prepended).

// Yukawa cutoff: exp(-mu*r) < 0.002 when mu*r > 6
fn yukawaCutoffSq(mu: f32) -> f32 {
    let cutoff = 6.0 / mu;
    return cutoff * cutoff;
}

// ─── Helper: accumulate potential from a single source position ───
fn accumulatePotential(
    wx: f32, wy: f32,
    srcX: f32, srcY: f32,
    mass: f32, charge: f32,
    doG: bool, doC: bool, doY: bool,
    softeningSq: f32, yCutSq: f32,
    yukCoupling: f32, yukMu: f32,
    periodic: bool,
    gPhi: ptr<function, f32>,
    ePhi: ptr<function, f32>,
    yPhi: ptr<function, f32>,
) {
    var dx: f32; var dy: f32;
    if (periodic) {
        let d = fullMinImageP(wx, wy, srcX, srcY, hu.domainW, hu.domainH, hu.topologyMode);
        dx = d.x; dy = d.y;
    } else {
        dx = srcX - wx; dy = srcY - wy;
    }

    let rSq = dx * dx + dy * dy + softeningSq;
    let invR = 1.0 / sqrt(rSq);

    if (doG) { *gPhi -= mass * invR; }
    if (doC) { *ePhi += charge * invR; }
    if (doY && rSq < yCutSq) {
        let r = 1.0 / invR;
        *yPhi -= yukCoupling * mass * exp(-yukMu * r) * invR;
    }
}

// ─── Entry 1: Pairwise with signal delay support ───
@compute @workgroup_size(8, 8)
fn computeHeatmap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if (gx >= HGRID || gy >= HGRID) { return; }

    let wx = hu.viewLeft + (f32(gx) + 0.5) * hu.cellW;
    let wy = hu.viewTop + (f32(gy) + 0.5) * hu.cellH;

    var gPhi: f32 = 0.0;
    var ePhi: f32 = 0.0;
    var yPhi: f32 = 0.0;

    let doG = hu.doGravity != 0u;
    let doC = hu.doCoulomb != 0u;
    let doY = hu.doYukawa != 0u;
    let useDelay = hu.useDelay != 0u;
    let isPeriodic = hu.periodic != 0u;
    let yCutSq = select(1e30, yukawaCutoffSq(hu.yukawaMu), doY);

    // Alive particles
    for (var i = 0u; i < hu.particleCount; i++) {
        let p = particles[i];
        let flag = p.flags;
        if ((flag & FLAG_ALIVE) == 0u) { continue; }

        var srcX = p.posX;
        var srcY = p.posY;

        // Signal delay: solve for retarded position
        if (useDelay) {
            let ret = getDelayedStateGPU(i, wx, wy, hu.simTime,
                isPeriodic, hu.domainW, hu.domainH,
                hu.topologyMode, false);
            if (!ret.valid) { continue; } // outside past light cone — particle not yet visible
            srcX = ret.x;
            srcY = ret.y;
        }

        accumulatePotential(wx, wy, srcX, srcY, p.mass, p.charge,
            doG, doC, doY, hu.softeningSq, yCutSq, hu.yukawaCoupling, hu.yukawaMu,
            isPeriodic, &gPhi, &ePhi, &yPhi);
    }

    // Dead/retired particles: signal delay fade-out
    if (useDelay) {
        for (var di = 0u; di < hu.particleCount; di++) {
            let dp = particles[di];
            if ((dp.flags & FLAG_RETIRED) == 0u) { continue; }
            if ((dp.flags & FLAG_ALIVE) != 0u) { continue; }

            let ret = getDelayedStateGPU(di, wx, wy, hu.simTime,
                isPeriodic, hu.domainW, hu.domainH,
                hu.topologyMode, true);
            if (!ret.valid) { continue; }

            let dAux = particleAux[di];
            accumulatePotential(wx, wy, ret.x, ret.y, dAux.deathMass, dp.charge,
                doG, doC, doY, hu.softeningSq, yCutSq, hu.yukawaCoupling, hu.yukawaMu,
                isPeriodic, &gPhi, &ePhi, &yPhi);
        }
    }

    let idx = gy * HGRID + gx;
    gravPotential[idx] = gPhi;
    elecPotential[idx] = ePhi;
    yukawaPotential[idx] = yPhi;
}

// ─── Entry 2: BH tree walk (no signal delay) ───
// Uses theta criterion: if node_size/distance < theta, use aggregate.
// Falls back to leaf-level evaluation at individual particles.

const HM_MAX_STACK: u32 = 48u;
const HM_NONE: i32 = -1;

@compute @workgroup_size(8, 8)
fn computeHeatmapTree(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if (gx >= HGRID || gy >= HGRID) { return; }

    let wx = hu.viewLeft + (f32(gx) + 0.5) * hu.cellW;
    let wy = hu.viewTop + (f32(gy) + 0.5) * hu.cellH;

    var gPhi: f32 = 0.0;
    var ePhi: f32 = 0.0;
    var yPhi: f32 = 0.0;

    let doG = hu.doGravity != 0u;
    let doC = hu.doCoulomb != 0u;
    let doY = hu.doYukawa != 0u;
    let softeningSq = hu.softeningSq;
    let yCutSq = select(1e30, yukawaCutoffSq(hu.yukawaMu), doY);
    let yukCoupling = hu.yukawaCoupling;
    let yukMu = hu.yukawaMu;

    // Stack-based BH tree traversal
    var stack: array<u32, HM_MAX_STACK>;
    var top: i32 = 0;
    stack[0] = 0u;
    top = 1;

    while (top > 0) {
        top--;
        let nIdx = stack[u32(top)];

        let isLeaf = getNW(nIdx) == HM_NONE;

        if (isLeaf) {
            // Leaf node: evaluate individual particle
            let pIdx = getParticleIndex(nIdx);
            if (pIdx < 0) { continue; }
            let j = u32(pIdx);
            let pj = particles[j];
            if ((pj.flags & FLAG_ALIVE) == 0u) { continue; }

            let dx = pj.posX - wx;
            let dy = pj.posY - wy;
            let rSq = dx * dx + dy * dy + softeningSq;
            let invR = 1.0 / sqrt(rSq);

            if (doG) { gPhi -= pj.mass * invR; }
            if (doC) { ePhi += pj.charge * invR; }
            if (doY && rSq < yCutSq) {
                let r = 1.0 / invR;
                gPhi; // no-op, just continue
                yPhi -= yukCoupling * pj.mass * exp(-yukMu * r) * invR;
            }
        } else {
            // Internal node: check theta criterion
            let comX = getComX(nIdx);
            let comY = getComY(nIdx);
            let dx = comX - wx;
            let dy = comY - wy;
            let distSq = dx * dx + dy * dy + softeningSq;

            let sizeX = getMaxX(nIdx) - getMinX(nIdx);
            let sizeY = getMaxY(nIdx) - getMinY(nIdx);
            let sizeSq = max(sizeX * sizeX, sizeY * sizeY);

            if (sizeSq < BH_THETA_SQ * distSq) {
                // Use aggregate: mass and charge from tree node
                let aggMass = getTotalMass(nIdx);
                let aggCharge = getTotalCharge(nIdx);
                let invR = 1.0 / sqrt(distSq);

                if (doG) { gPhi -= aggMass * invR; }
                if (doC) { ePhi += aggCharge * invR; }
                if (doY && distSq < yCutSq) {
                    let r = 1.0 / invR;
                    yPhi -= yukCoupling * aggMass * exp(-yukMu * r) * invR;
                }
            } else if (top + 4 <= i32(HM_MAX_STACK)) {
                // Open node: push children
                let nw = getNW(nIdx); let ne = getNE(nIdx);
                let sw = getSW(nIdx); let se = getSE(nIdx);
                if (nw != HM_NONE) { stack[u32(top)] = u32(nw); top++; }
                if (ne != HM_NONE) { stack[u32(top)] = u32(ne); top++; }
                if (sw != HM_NONE) { stack[u32(top)] = u32(sw); top++; }
                if (se != HM_NONE) { stack[u32(top)] = u32(se); top++; }
            }
        }
    }

    // Dead/retired particles: always pairwise (signal delay needed)
    let useDelay = hu.useDelay != 0u;
    let isPeriodic = hu.periodic != 0u;
    if (useDelay) {
        for (var di = 0u; di < hu.particleCount; di++) {
            let dp = particles[di];
            if ((dp.flags & FLAG_RETIRED) == 0u) { continue; }
            if ((dp.flags & FLAG_ALIVE) != 0u) { continue; }

            let ret = getDelayedStateGPU(di, wx, wy, hu.simTime,
                isPeriodic, hu.domainW, hu.domainH,
                hu.topologyMode, true);
            if (!ret.valid) { continue; }

            let dAux = particleAux[di];
            accumulatePotential(wx, wy, ret.x, ret.y, dAux.deathMass, dp.charge,
                doG, doC, doY, softeningSq, yCutSq, yukCoupling, yukMu,
                isPeriodic, &gPhi, &ePhi, &yPhi);
        }
    }

    let idx = gy * HGRID + gx;
    gravPotential[idx] = gPhi;
    elecPotential[idx] = ePhi;
    yukawaPotential[idx] = yPhi;
}

// ─── 3x3 Separable Box Blur ───
@group(0) @binding(0) var<storage, read_write> arr: array<f32>;
@group(0) @binding(1) var<storage, read_write> blurTemp: array<f32>;

@compute @workgroup_size(8, 8)
fn blurHorizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= HGRID || y >= HGRID) { return; }

    let row = y * HGRID;
    let l = select(arr[row], arr[row + x - 1u], x > 0u);
    let c = arr[row + x];
    let r = select(arr[row + HGRID - 1u], arr[row + x + 1u], x < HGRID - 1u);
    blurTemp[row + x] = (l + c + r) * (1.0 / 3.0);
}

@compute @workgroup_size(8, 8)
fn blurVertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= HGRID || y >= HGRID) { return; }

    let t = select(blurTemp[x], blurTemp[(y - 1u) * HGRID + x], y > 0u);
    let c = blurTemp[y * HGRID + x];
    let b = select(blurTemp[(HGRID - 1u) * HGRID + x], blurTemp[(y + 1u) * HGRID + x], y < HGRID - 1u);
    arr[y * HGRID + x] = (t + c + b) * (1.0 / 3.0);
}
