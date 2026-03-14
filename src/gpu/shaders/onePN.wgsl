// ─── 1PN Force Recomputation (Velocity-Verlet Correction) ───
// Four O(v²/c²) sectors:
//   EIH (GM + 1PN): perihelion precession
//   Darwin EM (Magnetic + 1PN): EM remainder
//   Bazanski (GM + Magnetic + 1PN): mixed 1/r³
//   Scalar Breit (Yukawa + 1PN): massive scalar exchange
//
// Two entry points:
//   compute1PN — recomputes 1PN forces at post-drift positions (pairwise)
//   vvKick1PN — applies VV correction kick: w += (f1pn_new - f1pn_old) * dt/(2m)
//
// Standalone shader — defines own structs (NOT prepended with common.wgsl).

const ALIVE_BIT: u32 = 1u;

// ── Packed struct definitions ──

struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct ParticleDerived {
    magMoment: f32,
    angMomentum: f32,
    invMass: f32,
    radiusSq: f32,
    velX: f32,
    velY: f32,
    angVel: f32,
    _pad: f32,
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

// Must match SimUniforms byte layout in common.wgsl / writeUniforms() exactly.
// Fields we don't use are kept as padding to preserve alignment.
struct Uniforms {
    dt: f32,                // [0] dt
    _simTime: f32,          // [1] simTime (unused here)
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
    _topologyMode: u32,     // [14] topologyMode (unused here)
    _collisionMode: u32,    // [15] collisionMode (unused here)
    _maxParticles: u32,     // [16] maxParticles (unused here)
    aliveCount: u32,        // [17] aliveCount
};

// Toggle bit constants
const GRAVITOMAG_BIT: u32    = 8u;
const MAGNETIC_BIT: u32      = 4u;
const YUKAWA_BIT: u32        = 2048u;
const RELATIVITY_BIT: u32    = 32u;

const EPSILON: f32 = 1e-9;
const BOUND_LOOP: u32 = 2u;

@group(0) @binding(0) var<uniform> u: Uniforms;

// Group 1: particle state (read-only)
@group(1) @binding(0) var<storage, read> particles: array<ParticleState>;
@group(1) @binding(1) var<storage, read> derived: array<ParticleDerived>;
@group(1) @binding(2) var<storage, read> axYukMod: array<vec2<f32>>;  // packed: axMod, yukMod

// Group 2: force outputs + VV kick
@group(2) @binding(0) var<storage, read_write> allForces: array<AllForces>;
@group(2) @binding(1) var<storage, read> f1pnOld: array<f32>;
@group(2) @binding(2) var<storage, read_write> particlesRW: array<ParticleState>; // rw for VV kick

// Per-source 1PN accumulation (shared between tree and pairwise paths)
fn accum1PN(
    px: f32, py: f32, pvx: f32, pvy: f32, pMass: f32, pCharge: f32,
    sx: f32, sy: f32, svx: f32, svy: f32, sMass: f32, sCharge: f32,
    sYukMod: f32, softeningSq: f32,
    periodic: bool, domW: f32, domH: f32,
    gmOn: bool, magOn: bool, yukOn: bool, yukawaMu: f32,
    yukawaCoupling: f32, pYukMod: f32,
) -> vec2f {
    var rx = sx - px;
    var ry = sy - py;
    if (periodic) {
        let halfW = domW * 0.5;
        let halfH = domH * 0.5;
        if (rx > halfW) { rx -= domW; } else if (rx < -halfW) { rx += domW; }
        if (ry > halfH) { ry -= domH; } else if (ry < -halfH) { ry += domH; }
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
        let base = sMass * invRSq * invR;
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
        let expMuR = exp(-mu * r_val);
        let nDotV1 = nx * pvx + ny * pvy;
        let nDotV2 = nx * svx + ny * svy;
        let v1DotV2 = pvx * svx + pvy * svy;
        let alpha = 1.0 + mu * r_val;
        let beta = 0.5 * yukawaCoupling * sqrt(pYukMod * sYukMod)
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
    if ((particles[i].flags & ALIVE_BIT) == 0u) { return; }

    let gmOn = (u.toggles0 & GRAVITOMAG_BIT) != 0u;
    let magOn = (u.toggles0 & MAGNETIC_BIT) != 0u;
    let yukOn = (u.toggles0 & YUKAWA_BIT) != 0u;
    let periodic = u.boundaryMode == BOUND_LOOP;

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
        if ((particles[j].flags & ALIVE_BIT) == 0u) { continue; }

        // Use current positions/velocities (post-drift)
        let swx = particles[j].velWX; let swy = particles[j].velWY;
        let sg = sqrt(1.0 + swx * swx + swy * swy);
        let sinvG = 1.0 / sg;
        let svx = swx * sinvG; let svy = swy * sinvG;

        let f = accum1PN(px, py, pvx, pvy, pMass, pCharge,
                         particles[j].posX, particles[j].posY, svx, svy,
                         particles[j].mass, particles[j].charge, axYukMod[j].y,
                         u.softeningSq, periodic, u.domainW, u.domainH,
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
fn vvKick1PN(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((particles[i].flags & ALIVE_BIT) == 0u) { return; }

    let halfDtOverM = u.dt * 0.5 * derived[i].invMass;
    let af1pn = allForces[i].f2;
    let newF = vec2f(af1pn.x, af1pn.y);
    let oldF = vec2f(f1pnOld[i * 2u], f1pnOld[i * 2u + 1u]);
    particlesRW[i].velWX += (newF.x - oldF.x) * halfDtOverM;
    particlesRW[i].velWY += (newF.y - oldF.y) * halfDtOverM;

    // NaN guard
    if (particlesRW[i].velWX != particlesRW[i].velWX || particlesRW[i].velWY != particlesRW[i].velWY) {
        particlesRW[i].velWX = 0.0; particlesRW[i].velWY = 0.0;
    }
}
