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

const ALIVE_BIT: u32 = 1u;

struct Uniforms {
    dt: f32,
    halfDt: f32,
    simTime: f32,
    domainW: f32,
    domainH: f32,
    softeningSq: f32,
    yukawaCoupling: f32,
    yukawaMu: f32,
    toggles0: u32,
    aliveCount: u32,
    boundaryMode: u32,
    _pad0: u32,
};

// Toggle bit constants
const GRAVITOMAG_BIT: u32    = 8u;
const MAGNETIC_BIT: u32      = 4u;
const YUKAWA_BIT: u32        = 2048u;
const RELATIVITY_BIT: u32    = 32u;

const EPSILON: f32 = 1e-9;
const BOUND_LOOP: u32 = 2u;

@group(0) @binding(0) var<uniform> u: Uniforms;

// Particle state (read)
@group(1) @binding(0) var<storage, read> posX: array<f32>;
@group(1) @binding(1) var<storage, read> posY: array<f32>;
@group(1) @binding(2) var<storage, read> velWX: array<f32>;
@group(1) @binding(3) var<storage, read> velWY: array<f32>;
@group(1) @binding(4) var<storage, read> mass: array<f32>;
@group(1) @binding(5) var<storage, read> charge: array<f32>;
@group(1) @binding(6) var<storage, read> flags: array<u32>;
@group(1) @binding(7) var<storage, read> yukMod: array<f32>;
@group(1) @binding(8) var<storage, read> invMass: array<f32>;

// Force accumulators (read_write) — forces2.xy = f1pn
@group(2) @binding(0) var<storage, read_write> forces2: array<vec4<f32>>;

// f1pnOld buffer (read, for VV kick)
@group(2) @binding(1) var<storage, read> f1pnOld: array<f32>;

// Proper velocity (read_write, for VV kick)
@group(2) @binding(2) var<storage, read_write> velWX_rw: array<f32>;
@group(2) @binding(3) var<storage, read_write> velWY_rw: array<f32>;

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
    if ((flags[i] & ALIVE_BIT) == 0u) { return; }

    let gmOn = (u.toggles0 & GRAVITOMAG_BIT) != 0u;
    let magOn = (u.toggles0 & MAGNETIC_BIT) != 0u;
    let yukOn = (u.toggles0 & YUKAWA_BIT) != 0u;
    let periodic = u.boundaryMode == BOUND_LOOP;

    // Zero 1PN forces before accumulation
    forces2[i] = vec4f(0.0, 0.0, forces2[i].z, forces2[i].w); // f1pn.xy = 0

    let px = posX[i]; let py = posY[i];
    // Use current velocity (post-drift) for 1PN — this is the VV correction
    let wx = velWX[i]; let wy = velWY[i];
    let gamma = sqrt(1.0 + wx * wx + wy * wy);
    let invG = 1.0 / gamma;
    let pvx = wx * invG; let pvy = wy * invG;
    let pMass = mass[i]; let pCharge = charge[i];

    var f1pnX: f32 = 0.0;
    var f1pnY: f32 = 0.0;

    // Pairwise loop (or tree walk when BH on — dispatched differently)
    let n = u.aliveCount;
    for (var j = 0u; j < n; j++) {
        if (j == i) { continue; }
        if ((flags[j] & ALIVE_BIT) == 0u) { continue; }

        // Use current positions/velocities (post-drift)
        let swx = velWX[j]; let swy = velWY[j];
        let sg = sqrt(1.0 + swx * swx + swy * swy);
        let sinvG = 1.0 / sg;
        let svx = swx * sinvG; let svy = swy * sinvG;

        let f = accum1PN(px, py, pvx, pvy, pMass, pCharge,
                         posX[j], posY[j], svx, svy,
                         mass[j], charge[j], yukMod[j],
                         u.softeningSq, periodic, u.domainW, u.domainH,
                         gmOn, magOn, yukOn, u.yukawaMu,
                         u.yukawaCoupling, yukMod[i]);
        f1pnX += f.x;
        f1pnY += f.y;
    }

    forces2[i] = vec4f(f1pnX, f1pnY, forces2[i].z, forces2[i].w);
}

@compute @workgroup_size(64)
fn vvKick1PN(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((flags[i] & ALIVE_BIT) == 0u) { return; }

    let halfDtOverM = u.halfDt * invMass[i];
    let newF = vec2f(forces2[i].x, forces2[i].y);
    let oldF = vec2f(f1pnOld[i * 2u], f1pnOld[i * 2u + 1u]);
    velWX_rw[i] += (newF.x - oldF.x) * halfDtOverM;
    velWY_rw[i] += (newF.y - oldF.y) * halfDtOverM;

    // NaN guard
    if (velWX_rw[i] != velWX_rw[i] || velWY_rw[i] != velWY_rw[i]) {
        velWX_rw[i] = 0.0; velWY_rw[i] = 0.0;
    }
}
