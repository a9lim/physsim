// ─── 1PN Force Recomputation (Velocity-Verlet Correction) ───
// Four O(v²/c²) sectors:
//   EIH (GM + 1PN): perihelion precession
//   Darwin EM (Magnetic + 1PN): EM remainder
//   Bazanski (GM + Magnetic + 1PN): mixed 1/r³
//   Scalar Breit (Yukawa + 1PN): massive scalar exchange
//
// Three entry points:
//   compute1PN     — recomputes 1PN forces at post-drift positions (O(N²) pairwise)
//   compute1PNTree — recomputes 1PN forces via Barnes-Hut tree walk (O(N log N))
//   vvKick1PN      — applies VV correction kick: w += (f1pn_new - f1pn_old) * dt/(2m)
//
// Prepended with: wgslConstants + shared-structs.wgsl + shared-topology.wgsl + shared-rng.wgsl
//   + signal-delay-common.wgsl
//   + shared-tree-nodes.wgsl (for compute1PNTree only — tree node accessors)
// Shared structs (ParticleState, ParticleDerived, AllForces, etc.) provided by shared-structs.wgsl.
// Topology helpers (fullMinImageP) provided by shared-topology.wgsl.
// Signal delay (getDelayedStateGPU, DelayedState) provided by signal-delay-common.wgsl.
// Signal delay: when relativity is on, uses retarded positions/velocities for all 1PN terms.
// No aberration — 1PN is already O(v²/c²). Dead particles excluded from 1PN.

const NONE: i32 = -1;
const MAX_STACK: u32 = 48u;

// Must match SimUniforms byte layout in common.wgsl / writeUniforms() exactly.
// Fields we don't use are kept as padding to preserve alignment.
struct Uniforms {
    dt: f32,                // [0] dt
    simTime: f32,           // [1] simTime
    domainW: f32,           // [2] domainW
    domainH: f32,           // [3] domainH
    _speedScale: f32,       // [4] speedScale (unused here)
    _softening: f32,        // [5] softening (unused here)
    softeningSq: f32,       // [6] softeningSq
    toggles0: u32,          // [7] toggles0
    _toggles1: u32,         // [8] toggles1 (unused here)
    yukawaCoupling: f32,    // [9] yukawaCoupling
    yukawaMu: f32,          // [10] yukawaMu
    _higgsMass: f32,        // [11] higgsMass (unused here)
    _axionMass: f32,        // [12] axionMass (unused here)
    boundaryMode: u32,      // [13] boundaryMode
    topologyMode: u32,      // [14] topologyMode
    _collisionMode: u32,    // [15] collisionMode (unused here)
    _maxParticles: u32,     // [16] maxParticles (unused here)
    aliveCount: u32,        // [17] aliveCount
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Group 1: particle state (read_write for encoder compat)
@group(1) @binding(0) var<storage, read_write> particles: array<ParticleState>;
@group(1) @binding(1) var<storage, read_write> derived: array<ParticleDerived>;
@group(1) @binding(2) var<storage, read_write> axYukMod: array<vec2<f32>>;  // packed: axMod, yukMod

// Group 2: force outputs (particleState accessed via group 1 to avoid aliasing)
@group(2) @binding(0) var<storage, read_write> allForces: array<AllForces>;
@group(2) @binding(1) var<storage, read_write> f1pnOld: array<f32>; // rw for encoder compat

// Group 3: signal delay history (interleaved) — used by getDelayedStateGPU()
@group(3) @binding(0) var<storage, read_write> histData: array<f32>;
@group(3) @binding(1) var<storage, read_write> histMeta: array<u32>;

// ── Tree walk bindings (used by compute1PNTree only) ──
// Group 1 binding 3: ghost original index mapping
@group(1) @binding(3) var<storage, read_write> ghostOriginalIdx: array<u32>;
// Group 3 binding 2: BH tree nodes (accessor fns from shared-tree-nodes.wgsl reference 'nodes')
@group(3) @binding(2) var<storage, read_write> nodes: array<u32>;

// Per-source 1PN accumulation (shared between tree and pairwise paths)
fn accum1PN(
    px: f32, py: f32, pvx: f32, pvy: f32, pMass: f32, pCharge: f32,
    sx: f32, sy: f32, svx: f32, svy: f32, sMass: f32, sCharge: f32,
    sYukMod: f32, softeningSq: f32,
    periodic: bool, domW: f32, domH: f32, topo: u32,
    gmOn: bool, magOn: bool, yukOn: bool, yukawaMu: f32,
    yukawaCoupling: f32, pYukMod: f32,
) -> vec2f {
    var rx = sx - px;
    var ry = sy - py;
    if (periodic) {
        let d = fullMinImageP(px, py, sx, sy, domW, domH, topo);
        rx = d.x;
        ry = d.y;
    }
    let rSq = rx * rx + ry * ry + softeningSq;
    let invRSq = 1.0 / rSq;
    let invR = sqrt(invRSq);
    let r_val = 1.0 / invR;
    let nx = rx * invR;
    let ny = ry * invR;

    var fx: f32 = 0.0;
    var fy: f32 = 0.0;

    // EIH (gravitomagnetic + 1PN)
    if (gmOn) {
        let v1Sq = pvx * pvx + pvy * pvy;
        let v2Sq = svx * svx + svy * svy;
        let nDotV1 = nx * pvx + ny * pvy;
        let nDotV2 = nx * svx + ny * svy;
        let radial = -v1Sq - 2.0 * v2Sq
            + 1.5 * nDotV2 * nDotV2
            + 5.0 * pMass * invR + 4.0 * sMass * invR;
        let v1Coeff = 4.0 * nDotV1 - 3.0 * nDotV2;
        let v2Coeff = 3.0 * nDotV2;
        let base = pMass * sMass * invRSq * invR;
        fx += base * (rx * radial + (pvx * v1Coeff + svx * v2Coeff) * r_val);
        fy += base * (ry * radial + (pvy * v1Coeff + svy * v2Coeff) * r_val);
    }

    // Darwin EM (magnetic + 1PN)
    if (magOn) {
        let v2DotN = svx * nx + svy * ny;
        let v1DotN = pvx * nx + pvy * ny;
        let coeff = 0.5 * pCharge * sCharge * invRSq;
        fx += coeff * (pvx * v2DotN - 3.0 * nx * v1DotN * v2DotN);
        fy += coeff * (pvy * v2DotN - 3.0 * ny * v1DotN * v2DotN);
    }

    // Bazanski (GM + Magnetic + 1PN)
    if (gmOn && magOn) {
        let crossCoeff = pCharge * sCharge * (pMass + sMass)
            - (pCharge * pCharge * sMass + sCharge * sCharge * pMass);
        let fDir = crossCoeff * invRSq * invRSq;
        fx += rx * fDir;
        fy += ry * fDir;
    }

    // Scalar Breit (Yukawa + 1PN)
    if (yukOn) {
        let mu = yukawaMu;
        let muR_val = mu * r_val;
        let expMuR = select(0.0, exp(-muR_val), muR_val < 80.0);
        let nDotV1 = nx * pvx + ny * pvy;
        let nDotV2 = nx * svx + ny * svy;
        let v1DotV2 = pvx * svx + pvy * svy;
        let alpha = 1.0 + mu * r_val;
        let beta = 0.5 * yukawaCoupling * sqrt(max(pYukMod * sYukMod, 0.0))
                   * pMass * sMass * expMuR * invRSq;
        let radial = -(alpha * v1DotV2
                       + (alpha * alpha + alpha + 1.0) * nDotV1 * nDotV2);
        fx += beta * (radial * nx + alpha * (nDotV2 * pvx + nDotV1 * svx));
        fy += beta * (radial * ny + alpha * (nDotV2 * pvy + nDotV1 * svy));
    }

    return vec2f(fx, fy);
}

@compute @workgroup_size(64)
fn compute1PN(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((particles[i].flags & FLAG_ALIVE) == 0u) { return; }

    let gmOn = (u.toggles0 & GRAVITOMAG_BIT) != 0u;
    let magOn = (u.toggles0 & MAGNETIC_BIT) != 0u;
    let yukOn = (u.toggles0 & YUKAWA_BIT) != 0u;
    let periodic = u.boundaryMode == BOUND_LOOP;
    let signalDelayed = (u.toggles0 & RELATIVITY_BIT) != 0u;

    // Zero 1PN forces before accumulation (preserve spinCurv in f2.zw)
    var af = allForces[i];
    af.f2.x = 0.0;
    af.f2.y = 0.0;
    allForces[i] = af;

    let px = particles[i].posX; let py = particles[i].posY;
    // Use current velocity (post-drift) for 1PN — this is the VV correction
    let wx = particles[i].velWX; let wy = particles[i].velWY;
    let gamma = sqrt(1.0 + wx * wx + wy * wy);
    let invG = 1.0 / gamma;
    let pvx = wx * invG; let pvy = wy * invG;
    let pMass = particles[i].mass; let pCharge = particles[i].charge;

    var f1pnX: f32 = 0.0;
    var f1pnY: f32 = 0.0;

    // Pairwise loop (or tree walk when BH on — dispatched differently)
    let n = u.aliveCount;
    for (var j = 0u; j < n; j++) {
        if (j == i) { continue; }
        if ((particles[j].flags & FLAG_ALIVE) == 0u) { continue; }

        var sx: f32; var sy: f32;
        var svx: f32; var svy: f32;

        if (signalDelayed) {
            // Signal delay: use retarded position/velocity from history
            // No aberration — 1PN is already O(v²/c²)
            let delayed = getDelayedStateGPU(
                j, px, py, u.simTime,
                periodic, u.domainW, u.domainH,
                u.topologyMode, false
            );
            if (!delayed.valid) { continue; }
            sx = delayed.x; sy = delayed.y;
            // Signal delay history stores coordinate velocity directly
            svx = delayed.vx; svy = delayed.vy;
        } else {
            // No signal delay: use current positions/velocities (post-drift)
            sx = particles[j].posX; sy = particles[j].posY;
            let swx = particles[j].velWX; let swy = particles[j].velWY;
            let sg = sqrt(1.0 + swx * swx + swy * swy);
            svx = swx / sg; svy = swy / sg;
        }

        let f = accum1PN(px, py, pvx, pvy, pMass, pCharge,
                         sx, sy, svx, svy,
                         particles[j].mass, particles[j].charge, axYukMod[j].y,
                         u.softeningSq, periodic, u.domainW, u.domainH, u.topologyMode,
                         gmOn, magOn, yukOn, u.yukawaMu,
                         u.yukawaCoupling, axYukMod[i].y);
        f1pnX += f.x;
        f1pnY += f.y;
    }

    var afOut = allForces[i];
    afOut.f2.x = f1pnX;
    afOut.f2.y = f1pnY;
    allForces[i] = afOut;
}

@compute @workgroup_size(64)
fn compute1PNTree(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((particles[i].flags & FLAG_ALIVE) == 0u) { return; }
    if ((particles[i].flags & FLAG_GHOST) != 0u) { return; } // Ghosts don't receive forces

    let gmOn = (u.toggles0 & GRAVITOMAG_BIT) != 0u;
    let magOn = (u.toggles0 & MAGNETIC_BIT) != 0u;
    let yukOn = (u.toggles0 & YUKAWA_BIT) != 0u;
    let periodic = u.boundaryMode == BOUND_LOOP;
    let signalDelayed = (u.toggles0 & RELATIVITY_BIT) != 0u;

    // Zero 1PN forces before accumulation (preserve spinCurv in f2.zw)
    var af = allForces[i];
    af.f2.x = 0.0;
    af.f2.y = 0.0;
    allForces[i] = af;

    let px = particles[i].posX; let py = particles[i].posY;
    let wx = particles[i].velWX; let wy = particles[i].velWY;
    let gamma = sqrt(1.0 + wx * wx + wy * wy);
    let invG = 1.0 / gamma;
    let pvx = wx * invG; let pvy = wy * invG;
    let pMass = particles[i].mass; let pCharge = particles[i].charge;

    var f1pnX: f32 = 0.0;
    var f1pnY: f32 = 0.0;

    // Stack-based BH tree walk
    var stack: array<u32, 48>;
    var stackTop: u32 = 1u;
    stack[0] = 0u; // root

    loop {
        if (stackTop == 0u) { break; }
        stackTop -= 1u;
        let nodeIdx = stack[stackTop];

        let nodeMass = getTotalMass(nodeIdx);
        if (nodeMass < EPSILON) { continue; }

        let comX = getComX(nodeIdx);
        let comY = getComY(nodeIdx);
        var dx = comX - px;
        var dy = comY - py;
        if (periodic) {
            let d = fullMinImageP(px, py, comX, comY, u.domainW, u.domainH, u.topologyMode);
            dx = d.x;
            dy = d.y;
        }
        let dSq = dx * dx + dy * dy;
        let size = getMaxX(nodeIdx) - getMinX(nodeIdx);

        let isLeaf = getNW(nodeIdx) == NONE;
        let particleIdx = getParticleIndex(nodeIdx);

        if (isLeaf && particleIdx >= 0) {
            // Leaf node: accumulate from individual particle
            let sIdx = u32(particleIdx);
            let sPs = particles[sIdx];

            // Skip self
            if (sIdx == i) { continue; }

            // Ghost handling: skip if original is self
            let isGhost = (sPs.flags & FLAG_GHOST) != 0u;
            var origIdx: u32 = sIdx;
            if (isGhost && sIdx >= u.aliveCount) {
                origIdx = ghostOriginalIdx[sIdx - u.aliveCount];
            }
            if (origIdx == i) { continue; }

            // Skip dead/non-alive (dead particles excluded from 1PN)
            let sIsRetired = (sPs.flags & FLAG_RETIRED) != 0u;
            if ((sPs.flags & FLAG_ALIVE) == 0u) { continue; }
            if (sIsRetired) { continue; }

            var sx: f32; var sy: f32;
            var svx: f32; var svy: f32;

            if (signalDelayed && !isGhost) {
                // Non-ghost leaf: signal delay from own history
                let delayed = getDelayedStateGPU(
                    sIdx, px, py, u.simTime,
                    periodic, u.domainW, u.domainH,
                    u.topologyMode, false
                );
                if (!delayed.valid) { continue; }
                sx = delayed.x; sy = delayed.y;
                svx = delayed.vx; svy = delayed.vy;
            } else if (signalDelayed && isGhost) {
                // Ghost leaf: signal delay from original + periodic shift
                let origPs = particles[origIdx];
                let delayed = getDelayedStateGPU(
                    origIdx, px, py, u.simTime,
                    periodic, u.domainW, u.domainH,
                    u.topologyMode, false
                );
                if (!delayed.valid) { continue; }
                let shiftX = sPs.posX - origPs.posX;
                let shiftY = sPs.posY - origPs.posY;
                sx = delayed.x + shiftX; sy = delayed.y + shiftY;
                svx = delayed.vx; svy = delayed.vy;
            } else {
                // No signal delay: use current positions/velocities
                sx = sPs.posX; sy = sPs.posY;
                let swx = sPs.velWX; let swy = sPs.velWY;
                let sg = sqrt(1.0 + swx * swx + swy * swy);
                svx = swx / sg; svy = swy / sg;
            }

            let f = accum1PN(px, py, pvx, pvy, pMass, pCharge,
                             sx, sy, svx, svy,
                             sPs.mass, sPs.charge, axYukMod[sIdx].y,
                             u.softeningSq, periodic, u.domainW, u.domainH, u.topologyMode,
                             gmOn, magOn, yukOn, u.yukawaMu,
                             u.yukawaCoupling, axYukMod[i].y);
            f1pnX += f.x;
            f1pnY += f.y;

        } else if (!isLeaf && (size * size < BH_THETA_SQ * dSq)) {
            // Distant node: use aggregate data
            let avgVx = getTotalMomX(nodeIdx) / nodeMass;
            let avgVy = getTotalMomY(nodeIdx) / nodeMass;

            let f = accum1PN(px, py, pvx, pvy, pMass, pCharge,
                             comX, comY, avgVx, avgVy,
                             nodeMass, getTotalCharge(nodeIdx), 1.0,
                             u.softeningSq, periodic, u.domainW, u.domainH, u.topologyMode,
                             gmOn, magOn, yukOn, u.yukawaMu,
                             u.yukawaCoupling, axYukMod[i].y);
            f1pnX += f.x;
            f1pnY += f.y;

        } else if (!isLeaf) {
            // Near field: push 4 children
            if (stackTop + 4u <= MAX_STACK) {
                let nw = getNW(nodeIdx);
                let ne = getNE(nodeIdx);
                let sw = getSW(nodeIdx);
                let se = getSE(nodeIdx);
                if (nw != NONE) { stack[stackTop] = u32(nw); stackTop += 1u; }
                if (ne != NONE) { stack[stackTop] = u32(ne); stackTop += 1u; }
                if (sw != NONE) { stack[stackTop] = u32(sw); stackTop += 1u; }
                if (se != NONE) { stack[stackTop] = u32(se); stackTop += 1u; }
            }
        }
    }

    var afOut = allForces[i];
    afOut.f2.x = f1pnX;
    afOut.f2.y = f1pnY;
    allForces[i] = afOut;
}

@compute @workgroup_size(64)
fn vvKick1PN(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((particles[i].flags & FLAG_ALIVE) == 0u) { return; }

    let halfDtOverM = u.dt * 0.5 * derived[i].invMass;
    let af1pn = allForces[i].f2;
    let newF = vec2f(af1pn.x, af1pn.y);
    let oldF = vec2f(f1pnOld[i * 2u], f1pnOld[i * 2u + 1u]);
    particles[i].velWX += (newF.x - oldF.x) * halfDtOverM;
    particles[i].velWY += (newF.y - oldF.y) * halfDtOverM;

    // NaN guard
    if (particles[i].velWX != particles[i].velWX || particles[i].velWY != particles[i].velWY) {
        particles[i].velWX = 0.0; particles[i].velWY = 0.0;
    }
}
