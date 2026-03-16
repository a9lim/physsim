// ─── Barnes-Hut Tree Walk Force Computation ───
// One thread per alive particle. Stack-based iterative traversal.
// Uses theta=0.5 opening angle criterion.
// Accumulates into the same force buffers as pairwise path.

// Constants provided by generated wgslConstants block.
// Shader-specific constants:
const NONE: i32 = -1;
const MAX_STACK: u32 = 48u;

// Minimum-image displacement (inlined from common.wgsl since standalone shader)
fn torusMinImage(ox: f32, oy: f32, sx: f32, sy: f32, w: f32, h: f32) -> vec2<f32> {
    let halfW = w * 0.5;
    let halfH = h * 0.5;
    var rx = sx - ox;
    if (rx > halfW) { rx -= w; } else if (rx < -halfW) { rx += w; }
    var ry = sy - oy;
    if (ry > halfH) { ry -= h; } else if (ry < -halfH) { ry += h; }
    return vec2(rx, ry);
}

fn fullMinImage(ox: f32, oy: f32, sx: f32, sy: f32, w: f32, h: f32, topo: u32) -> vec2<f32> {
    if (topo == TOPO_TORUS) {
        return torusMinImage(ox, oy, sx, sy, w, h);
    }
    let halfW = w * 0.5;
    let halfH = h * 0.5;
    // Candidate 0: only torus-wrap axes with translational (not glide) periodicity.
    var dx0 = sx - ox;
    var dy0 = sy - oy;
    if (topo == TOPO_KLEIN) {
        if (dx0 > halfW) { dx0 -= w; } else if (dx0 < -halfW) { dx0 += w; }
    }
    var bestSq = dx0 * dx0 + dy0 * dy0;
    var bestDx = dx0;
    var bestDy = dy0;
    if (topo == TOPO_KLEIN) {
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

// Node layout accessors (same as tree-build.wgsl)
const NODE_STRIDE: u32 = 20u;
fn nodeOffset(idx: u32) -> u32 { return idx * NODE_STRIDE; }

fn getMinX(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx)]); }
fn getMinY(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 1u]); }
fn getMaxX(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 2u]); }
fn getMaxY(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 3u]); }
fn getComX(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 4u]); }
fn getComY(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 5u]); }
fn getTotalMass(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 6u]); }
fn getTotalCharge(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 7u]); }
fn getTotalMagMoment(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 8u]); }
fn getTotalAngMomentum(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 9u]); }
fn getTotalMomX(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 10u]); }
fn getTotalMomY(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 11u]); }
fn getNW(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 12u]); }
fn getNE(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 13u]); }
fn getSW(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 14u]); }
fn getSE(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 15u]); }
fn getParticleIndex(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 16u]); }
fn getParticleCount(idx: u32) -> u32 { return nodes[nodeOffset(idx) + 17u]; }

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

@group(0) @binding(0) var<storage, read_write> nodes: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: SimUniforms;

// Group 1: packed particle structs (read_write for encoder compat)
@group(1) @binding(0) var<storage, read_write> particleState: array<ParticleState>;
@group(1) @binding(1) var<storage, read_write> particleAux: array<ParticleAux>;
@group(1) @binding(2) var<storage, read_write> derived_in: array<ParticleDerived>;
@group(1) @binding(3) var<storage, read_write> axYukMod_in: array<vec2<f32>>; // packed: axMod, yukMod
@group(1) @binding(4) var<storage, read_write> ghostOriginalIdx: array<u32>;

// Group 2: force accumulators + maxAccel — 2 bindings (radiationState removed, jerk now in AllForces)
@group(2) @binding(0) var<storage, read_write> allForces: array<AllForces>;
@group(2) @binding(1) var<storage, read_write> maxAccel: array<atomic<u32>>;

// Group 3: signal delay history (interleaved) — 2 bindings
// Bindings declared for pipeline layout; used by signal delay lookups (Task 9)
@group(3) @binding(0) var<storage, read_write> histData: array<f32>;
@group(3) @binding(1) var<storage, read_write> histMeta: array<u32>;

// Shared pairForce function (from pair-force.wgsl, imported or inlined)
// This function accumulates E-like forces and B-field contributions
// for one source acting on one receiver particle.
// For the tree walk, sources are either individual leaf particles
// or aggregate node data (mass, charge, CoM, etc.).

// Aberration constants
const ABERRATION_CLAMP_MIN: f32 = 0.01;
const ABERRATION_CLAMP_MAX: f32 = 100.0;

// Inline the core force accumulation (matches pairForce in forces.js):
// jerkOut: function-scope pointer for accumulating analytical jerk (radiation)
fn accumulateForce(
    af: ptr<function, AllForces>,
    px: f32, py: f32,
    pMass: f32, pCharge: f32,
    pMagMoment: f32, pAngMomentum: f32,
    pAngVel: f32, pVelX: f32, pVelY: f32,
    pAxMod: f32, pYukMod: f32,
    sx: f32, sy: f32,
    svx: f32, svy: f32,
    sMass: f32, sCharge: f32,
    sAngVel: f32, sMagMoment: f32, sAngMomentum: f32,
    sAxMod: f32, sYukMod: f32,
    pBodyRadiusSq: f32,
    jerkOut: ptr<function, vec2<f32>>,
    useAberration: bool,
) {
    let toggles = uniforms.toggles0;
    let softeningSq = uniforms.softeningSq;

    // Displacement (with minimum-image for periodic boundaries)
    let periodic = uniforms.boundaryMode == BOUND_LOOP;
    var disp = vec2<f32>(sx - px, sy - py);
    if (periodic) {
        disp = fullMinImage(px, py, sx, sy, uniforms.domainW, uniforms.domainH, uniforms.topologyMode);
    }
    let rx = disp.x;
    let ry = disp.y;
    let rawRSq = rx * rx + ry * ry;
    let rSq = rawRSq + softeningSq;
    let invRSq = 1.0 / rSq;
    let invR = sqrt(invRSq);
    let invR3 = invR * invRSq;
    let invR5 = invR3 * invRSq;

    // Lienard-Wiechert aberration: (1 - n_hat dot v_source)^{-3}
    var aberr: f32 = 1.0;
    if (useAberration) {
        let nDotV = -(rx * svx + ry * svy) * invR;
        let denom = max(1.0 - nDotV, ABERRATION_CLAMP_MIN);
        aberr = min(1.0 / (denom * denom * denom), ABERRATION_CLAMP_MAX);
    }
    let invR3a = select(invR3, invR3 * aberr, useAberration);
    let invR5a = select(invR5, invR5 * aberr, useAberration);

    let needAxMod = ((toggles & COULOMB_BIT) != 0u || (toggles & MAGNETIC_BIT) != 0u) && (toggles & AXION_BIT) != 0u;
    var axModPair: f32 = 1.0;
    if (needAxMod) {
        axModPair = sqrt(max(pAxMod * sAxMod, 0.0));
    }

    // Relative velocity for jerk computation
    let radOn = (toggles & RADIATION_BIT) != 0u;
    let vrx = svx - pVelX;
    let vry = svy - pVelY;
    let rDotVr = rx * vrx + ry * vry;

    // Gravity: +m1*m2/r^2 (attractive)
    if ((toggles & GRAVITY_BIT) != 0u) {
        let k = pMass * sMass;
        let fDir = k * invR3a;
        let fx = rx * fDir;
        let fy = ry * fDir;
        (*af).f0.x += fx;
        (*af).f0.y += fy;
        (*af).totalForce.x += fx;
        (*af).totalForce.y += fy;

        // Analytical jerk for Larmor radiation
        if (radOn) {
            let jRadial = -3.0 * rDotVr * k * invRSq * invR3a;
            (*jerkOut).x += vrx * fDir + rx * jRadial;
            (*jerkOut).y += vry * fDir + ry * jRadial;
        }

        // Tidal locking torque
        let crossRV = rx * (svy - pVelY) - ry * (svx - pVelX);
        let wOrbit = crossRV * invRSq;
        let dw = pAngVel - wOrbit;
        var coupling = sMass;
        if ((toggles & COULOMB_BIT) != 0u && pMass > EPSILON) {
            coupling += pCharge * sCharge / pMass;
        }
        let ri5 = pBodyRadiusSq * pBodyRadiusSq * pow(pMass, 1.0/3.0);
        let invR6 = invRSq * invRSq * invRSq;
        (*af).torques.z += -TIDAL_STRENGTH * coupling * coupling * ri5 * invR6 * dw;
    }

    // Coulomb: -q1*q2/r^2 (like repels)
    if ((toggles & COULOMB_BIT) != 0u) {
        let k = -(pCharge * sCharge) * axModPair;
        let fDir = k * invR3a;
        let fx = rx * fDir;
        let fy = ry * fDir;
        (*af).f0.z += fx;
        (*af).f0.w += fy;
        (*af).totalForce.x += fx;
        (*af).totalForce.y += fy;

        // Analytical jerk for Larmor radiation
        if (radOn) {
            let jRadial = -3.0 * rDotVr * k * invRSq * invR3a;
            (*jerkOut).x += vrx * fDir + rx * jRadial;
            (*jerkOut).y += vry * fDir + ry * jRadial;
        }
    }

    // Cross product (vs x r)_z for Biot-Savart
    let crossSV = svx * ry - svy * rx;

    // Magnetic: dipole-dipole + Bz field
    if ((toggles & MAGNETIC_BIT) != 0u) {
        let axMod = axModPair;
        // Dipole-dipole radial: F = +3*mu1*mu2/r^4
        let fDir = 3.0 * (pMagMoment * sMagMoment) * invR5a * axMod;
        let fx = rx * fDir;
        let fy = ry * fDir;
        (*af).f1.x += fx;
        (*af).f1.y += fy;
        (*af).totalForce.x += fx;
        (*af).totalForce.y += fy;

        // Analytical jerk: F = 3μ₁μ₂·r/r⁵, d/dt(1/r⁵) = -5·rDotVr/r⁷
        if (radOn) {
            let invR7a = invR5a * invRSq;
            let jRadial = -15.0 * (pMagMoment * sMagMoment) * rDotVr * invR7a * axMod;
            (*jerkOut).x += vrx * fDir + rx * jRadial;
            (*jerkOut).y += vry * fDir + ry * jRadial;
        }

        // Bz from moving charge
        let BzMoving = sCharge * crossSV * invR3 * axMod;
        (*af).bFields.x += BzMoving;
        // dBz gradients from moving charge
        (*af).bFieldGrads.x += 3.0 * BzMoving * rx * invRSq + sCharge * svy * invR3 * axMod;
        (*af).bFieldGrads.y += 3.0 * BzMoving * ry * invRSq - sCharge * svx * invR3 * axMod;

        // Bz from dipole: -mu/r^3
        (*af).bFields.x -= sMagMoment * invR3 * axMod;
        // dBz gradients from dipole
        (*af).bFieldGrads.x -= 3.0 * sMagMoment * rx * invR5 * axMod;
        (*af).bFieldGrads.y -= 3.0 * sMagMoment * ry * invR5 * axMod;
    }

    // Gravitomagnetic: dipole + Bgz field
    if ((toggles & GRAVITOMAG_BIT) != 0u) {
        let fDir = 3.0 * (pAngMomentum * sAngMomentum) * invR5a;
        let fx = rx * fDir;
        let fy = ry * fDir;
        (*af).f1.z += fx;
        (*af).f1.w += fy;
        (*af).totalForce.x += fx;
        (*af).totalForce.y += fy;

        // Analytical jerk: F = 3L₁L₂·r/r⁵, d/dt(1/r⁵) = -5·rDotVr/r⁷
        if (radOn) {
            let invR7a = invR5a * invRSq;
            let jRadial = -15.0 * (pAngMomentum * sAngMomentum) * rDotVr * invR7a;
            (*jerkOut).x += vrx * fDir + rx * jRadial;
            (*jerkOut).y += vry * fDir + ry * jRadial;
        }

        // Bgz from moving mass
        let BgzMoving = -sMass * crossSV * invR3;
        (*af).bFields.y += BgzMoving;
        // dBgz gradients from moving mass
        (*af).bFieldGrads.z += 3.0 * BgzMoving * rx * invRSq - sMass * svy * invR3;
        (*af).bFieldGrads.w += 3.0 * BgzMoving * ry * invRSq + sMass * svx * invR3;

        // Bgz from spin: -2L/r^3
        (*af).bFields.y -= 2.0 * sAngMomentum * invR3;
        // dBgz gradients from spin
        (*af).bFieldGrads.z -= 6.0 * sAngMomentum * rx * invR5;
        (*af).bFieldGrads.w -= 6.0 * sAngMomentum * ry * invR5;

        // Frame-dragging torque
        let fdTorque = 2.0 * sAngMomentum * (sAngVel - pAngVel) * invR3;
        (*af).torques.y += fdTorque;
    }

    // Yukawa
    if ((toggles & YUKAWA_BIT) != 0u) {
        let mu = uniforms.yukawaMu;
        let cutoffSq = (6.0 / mu) * (6.0 / mu);
        if (rawRSq < cutoffSq) {
            let r = 1.0 / invR;
            let muR = mu * r;
            let expMuR = select(0.0, exp(-muR), muR < 80.0);
            let yukModPair = sqrt(max(pYukMod * sYukMod, 0.0));
            let yukInvRa = select(invR, invR * aberr, useAberration);
            let fDir = uniforms.yukawaCoupling * yukModPair * pMass * sMass * expMuR * (invRSq + mu * invR) * yukInvRa;
            let fx = rx * fDir;
            let fy = ry * fDir;
            (*af).f3.z += fx;
            (*af).f3.w += fy;
            (*af).totalForce.x += fx;
            (*af).totalForce.y += fy;

            // Analytical jerk for pion emission radiation
            if (radOn) {
                let jRadial = -(3.0 * invRSq + 3.0 * mu * invR + mu * mu)
                              * rDotVr * uniforms.yukawaCoupling * yukModPair * pMass * sMass
                              * expMuR * invRSq * yukInvRa;
                (*jerkOut).x += vrx * fDir + rx * jRadial;
                (*jerkOut).y += vry * fDir + ry * jRadial;
            }

            // Scalar Breit 1PN correction
            if ((toggles & ONE_PN_BIT) != 0u) {
                let nx = rx * invR;
                let ny = ry * invR;
                let nDotV1 = nx * pVelX + ny * pVelY;
                let nDotV2 = nx * svx + ny * svy;
                let v1DotV2 = pVelX * svx + pVelY * svy;
                let alpha = 1.0 + mu * r;
                let beta = 0.5 * uniforms.yukawaCoupling * yukModPair * pMass * sMass * expMuR * invRSq;
                let radial = -(alpha * v1DotV2 + (alpha * alpha + alpha + 1.0) * nDotV1 * nDotV2);
                let sbX = beta * (radial * nx + alpha * (nDotV2 * pVelX + nDotV1 * svx));
                let sbY = beta * (radial * ny + alpha * (nDotV2 * pVelY + nDotV1 * svy));
                (*af).f2.x += sbX;
                (*af).f2.y += sbY;
                (*af).totalForce.x += sbX;
                (*af).totalForce.y += sbY;
            }
        }
    }

    // 1PN EIH (gravitomagnetic + 1PN): perihelion precession
    if ((toggles & ONE_PN_BIT) != 0u && (toggles & GRAVITOMAG_BIT) != 0u) {
        let r_val = 1.0 / invR;
        let nx = rx * invR;
        let ny = ry * invR;
        let v1Sq = pVelX * pVelX + pVelY * pVelY;
        let v2Sq = svx * svx + svy * svy;
        let nDotV1 = nx * pVelX + ny * pVelY;
        let nDotV2 = nx * svx + ny * svy;
        let radial = -v1Sq - 2.0 * v2Sq
            + 1.5 * nDotV2 * nDotV2
            + 5.0 * pMass * invR + 4.0 * sMass * invR;
        let v1Coeff = 4.0 * nDotV1 - 3.0 * nDotV2;
        let v2Coeff = 3.0 * nDotV2;
        let base = sMass * invR3;
        let eihX = base * (rx * radial + (pVelX * v1Coeff + svx * v2Coeff) * r_val);
        let eihY = base * (ry * radial + (pVelY * v1Coeff + svy * v2Coeff) * r_val);
        (*af).f2.x += eihX;
        (*af).f2.y += eihY;
        (*af).totalForce.x += eihX;
        (*af).totalForce.y += eihY;

        // Analytical jerk for position-only EIH term: F = m₂(5m₁+4m₂)·r/r⁴
        if (radOn) {
            let kEIH = sMass * (5.0 * pMass + 4.0 * sMass);
            let fDirEIH = kEIH * invRSq * invRSq;
            let jRadialEIH = -4.0 * kEIH * rDotVr * invRSq * invRSq * invRSq;
            (*jerkOut).x += vrx * fDirEIH + rx * jRadialEIH;
            (*jerkOut).y += vry * fDirEIH + ry * jRadialEIH;
        }
    }

    // 1PN Darwin EM (magnetic + 1PN)
    if ((toggles & ONE_PN_BIT) != 0u && (toggles & MAGNETIC_BIT) != 0u) {
        let nx = rx * invR;
        let ny = ry * invR;
        let v2DotN = svx * nx + svy * ny;
        let v1DotN = pVelX * nx + pVelY * ny;
        let coeff = 0.5 * pCharge * sCharge * invRSq;
        let darX = coeff * (pVelX * v2DotN - 3.0 * nx * v1DotN * v2DotN);
        let darY = coeff * (pVelY * v2DotN - 3.0 * ny * v1DotN * v2DotN);
        (*af).f2.x += darX;
        (*af).f2.y += darY;
        (*af).totalForce.x += darX;
        (*af).totalForce.y += darY;
    }

    // 1PN Bazanski (GM + Magnetic + 1PN): mixed gravity+EM
    if ((toggles & ONE_PN_BIT) != 0u && (toggles & GRAVITOMAG_BIT) != 0u && (toggles & MAGNETIC_BIT) != 0u) {
        let crossCoeff = pCharge * sCharge * (pMass + sMass)
            - (pCharge * pCharge * sMass + sCharge * sCharge * pMass);
        let fDir = crossCoeff * invRSq * invRSq;
        let bazX = rx * fDir;
        let bazY = ry * fDir;
        (*af).f2.x += bazX;
        (*af).f2.y += bazY;
        (*af).totalForce.x += bazX;
        (*af).totalForce.y += bazY;

        // Analytical jerk: F = crossCoeff·r/r⁴, d/dt(1/r⁴) = -4·rDotVr/r⁶
        if (radOn) {
            let jRadial = -4.0 * crossCoeff * rDotVr * invRSq * invRSq * invRSq;
            (*jerkOut).x += vrx * fDir + rx * jRadial;
            (*jerkOut).y += vry * fDir + ry * jRadial;
        }
    }

}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pIdx = gid.x;
    if (pIdx >= uniforms.aliveCount) { return; }

    let ps = particleState[pIdx];
    if ((ps.flags & FLAG_ALIVE) == 0u) { return; }
    if ((ps.flags & FLAG_GHOST) != 0u) { return; } // Ghosts don't receive forces

    let px = ps.posX;
    let py = ps.posY;
    let pMass = ps.mass;
    let pCharge = ps.charge;
    let pDerived = derived_in[pIdx];
    let pMagMoment = pDerived.magMoment;
    let pAngMomentum = pDerived.angMomentum;
    let pAux = particleAux[pIdx];
    let pRadius = pAux.radius;
    let pBodyRadiusSq = pRadius * pRadius; // Approximation; BH mode uses different formula
    let pAngW = ps.angW;
    let pAxMod = axYukMod_in[pIdx].x;
    let pYukMod = axYukMod_in[pIdx].y;

    // Derive velocity from proper velocity
    let wSq = ps.velWX * ps.velWX + ps.velWY * ps.velWY;
    let relOn = (uniforms.toggles0 & RELATIVITY_BIT) != 0u;
    let invGamma = select(1.0, 1.0 / sqrt(1.0 + wSq), relOn);
    let pVelX = ps.velWX * invGamma;
    let pVelY = ps.velWY * invGamma;
    let pAngVel = select(pAngW, pAngW / sqrt(1.0 + pAngW * pAngW * pBodyRadiusSq), relOn);

    let pInvMass = pDerived.invMass;

    let thetaSq = uniforms.bhTheta * uniforms.bhTheta;
    let pid = pAux.particleId;

    // Local jerk accumulator for radiation analytical jerk
    var localJerk = vec2<f32>(0.0, 0.0);

    // Load AllForces struct ONCE from global memory (already zeroed by resetForces pass)
    var localAF = allForces[pIdx];

    let isPeriodic = uniforms.boundaryMode == BOUND_LOOP;
    let hasSignalDelay = (uniforms.toggles0 & RELATIVITY_BIT) != 0u;

    // Pre-drift signal delay time: forces are computed at pre-drift positions,
    // so use simTime BEFORE the dt increment to match the particle's actual
    // spacetime location (CPU increments simTime after drift, before recompute).
    let sdTime = uniforms.simTime - uniforms.dt;

    // Stack-based iterative tree walk
    var stack: array<u32, 48>;
    var stackTop: u32 = 0u;
    stack[0] = 0u; // push root
    stackTop = 1u;

    loop {
        if (stackTop == 0u) { break; }
        stackTop -= 1u;
        let nodeIdx = stack[stackTop];

        let nodeMass = getTotalMass(nodeIdx);
        if (nodeMass < EPSILON) { continue; }

        let comX = getComX(nodeIdx);
        let comY = getComY(nodeIdx);
        var comDisp = vec2<f32>(comX - px, comY - py);
        if (isPeriodic) {
            comDisp = fullMinImage(px, py, comX, comY, uniforms.domainW, uniforms.domainH, uniforms.topologyMode);
        }
        let dx = comDisp.x;
        let dy = comDisp.y;
        let dSq = dx * dx + dy * dy;
        let size = getMaxX(nodeIdx) - getMinX(nodeIdx); // node width

        let isLeaf = getNW(nodeIdx) == NONE;
        let particleIdx = getParticleIndex(nodeIdx);

        if (isLeaf && particleIdx >= 0) {
            // Leaf node: accumulate from the individual particle
            let sIdx = u32(particleIdx);

            // Skip self-interaction
            let sPs = particleState[sIdx];
            let sAux = particleAux[sIdx];
            // Ghost particles: skip if original is self
            let isGhost = (sPs.flags & FLAG_GHOST) != 0u;
            var origIdx: u32 = sIdx;
            if (isGhost && sIdx >= uniforms.aliveCount) {
                origIdx = ghostOriginalIdx[sIdx - uniforms.aliveCount];
            }
            if (origIdx == pIdx) { continue; } // skip self
            if (sIdx == pIdx) { continue; }
            if ((sPs.flags & FLAG_ALIVE) == 0u) { continue; } // skip dead/retired particles

            let sAYM = axYukMod_in[sIdx];

            // Signal delay: use retarded positions/velocities for leaf particles
            if (hasSignalDelay && !isGhost) {
                // Non-ghost leaf: signal delay from own history
                let delayed = getDelayedStateGPU(
                    sIdx, px, py, sdTime,
                    isPeriodic, uniforms.domainW, uniforms.domainH,
                    uniforms.topologyMode, false
                );
                if (!delayed.valid) { continue; }
                // Recompute dipoles from retarded angw
                let bodyRadSq = pow(sPs.mass, 2.0 / 3.0);
                let retAngwSq = delayed.angw * delayed.angw;
                let sAngVelRet = delayed.angw / sqrt(1.0 + retAngwSq * bodyRadSq);
                let sMagMomRet = MAG_MOMENT_K * sPs.charge * sAngVelRet * bodyRadSq;
                let sAngMomRet = INERTIA_K * sPs.mass * sAngVelRet * bodyRadSq;
                accumulateForce(
                    &localAF, px, py, pMass, pCharge,
                    pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY,
                    pAxMod, pYukMod,
                    delayed.x, delayed.y,
                    delayed.vx, delayed.vy,
                    sPs.mass, sPs.charge,
                    sAngVelRet, sMagMomRet, sAngMomRet,
                    sAYM.x, sAYM.y,
                    pBodyRadiusSq,
                    &localJerk,
                    true, // useAberration
                );
            } else if (hasSignalDelay && isGhost) {
                // Ghost leaf: signal delay from original particle's history + periodic shift
                let origPs = particleState[origIdx];
                let delayed = getDelayedStateGPU(
                    origIdx, px, py, sdTime,
                    isPeriodic, uniforms.domainW, uniforms.domainH,
                    uniforms.topologyMode, false
                );
                if (!delayed.valid) { continue; }
                // Periodic shift: ghostPos - originalCurrentPos
                let shiftX = sPs.posX - origPs.posX;
                let shiftY = sPs.posY - origPs.posY;
                // Retarded ghost position
                let gsx = delayed.x + shiftX;
                let gsy = delayed.y + shiftY;
                // Recompute dipoles from retarded angw
                let bodyRadSq = pow(sPs.mass, 2.0 / 3.0);
                let retAngwSq = delayed.angw * delayed.angw;
                let sAngVelRet = delayed.angw / sqrt(1.0 + retAngwSq * bodyRadSq);
                let sMagMomRet = MAG_MOMENT_K * sPs.charge * sAngVelRet * bodyRadSq;
                let sAngMomRet = INERTIA_K * sPs.mass * sAngVelRet * bodyRadSq;
                accumulateForce(
                    &localAF, px, py, pMass, pCharge,
                    pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY,
                    pAxMod, pYukMod,
                    gsx, gsy,
                    delayed.vx, delayed.vy,
                    sPs.mass, sPs.charge,
                    sAngVelRet, sMagMomRet, sAngMomRet,
                    sAYM.x, sAYM.y,
                    pBodyRadiusSq,
                    &localJerk,
                    true, // useAberration
                );
            } else {
                // No signal delay: use current positions
                let sDerived = derived_in[sIdx];
                accumulateForce(
                    &localAF, px, py, pMass, pCharge,
                    pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY,
                    pAxMod, pYukMod,
                    sPs.posX, sPs.posY,
                    sDerived.velX, sDerived.velY,
                    sPs.mass, sPs.charge,
                    sDerived.angVel,
                    sDerived.magMoment, sDerived.angMomentum,
                    sAYM.x, sAYM.y,
                    pBodyRadiusSq,
                    &localJerk,
                    false, // no aberration
                );
            }
        } else if (!isLeaf && (size * size < thetaSq * dSq)) {
            // Distant node: use aggregate multipole data
            let avgVx = getTotalMomX(nodeIdx) / nodeMass;
            let avgVy = getTotalMomY(nodeIdx) / nodeMass;

            accumulateForce(
                &localAF, px, py, pMass, pCharge,
                pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY,
                pAxMod, pYukMod,
                comX, comY,
                avgVx, avgVy,
                nodeMass, getTotalCharge(nodeIdx),
                0.0, // sAngVel = 0 for aggregate
                getTotalMagMoment(nodeIdx), getTotalAngMomentum(nodeIdx),
                1.0, 1.0, // axMod/yukMod = 1 for aggregate
                pBodyRadiusSq,
                &localJerk,
                hasSignalDelay, // aberration on aggregates when signal delay active
            );
        } else if (!isLeaf) {
            // Push children (only valid ones; NONE = -1 would become garbage u32)
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

    // After tree walk, iterate retired particles (pairwise, matching CPU behavior).
    // Only relevant when relativity is on (dead particles exert forces via signal delay).
    // Skip entirely otherwise — the O(N) scan per thread dominates BH O(N log N) cost.
    // Uses particleCount (not aliveCount) because retired slots can be beyond aliveCount.
    for (var ri = 0u; ri < select(0u, uniforms.particleCount, hasSignalDelay); ri++) {
        let rPs = particleState[ri];
        if ((rPs.flags & FLAG_RETIRED) == 0u) { continue; }
        if ((rPs.flags & FLAG_ALIVE) != 0u) { continue; }

        let delayed = getDelayedStateGPU(
            ri, px, py, sdTime,
            isPeriodic, uniforms.domainW, uniforms.domainH,
            uniforms.topologyMode, true // isDead
        );
        if (!delayed.valid) { continue; }

        let rAux = particleAux[ri];
        let deadMass = rAux.deathMass;
        let deadCharge = rPs.charge;
        // Recompute dipoles from retarded angw using deathMass
        let bodyRadSq = pow(deadMass, 2.0 / 3.0);
        let retAngwSq = delayed.angw * delayed.angw;
        let sAngVelRet = delayed.angw / sqrt(1.0 + retAngwSq * bodyRadSq);
        let sMagMomRet = MAG_MOMENT_K * deadCharge * sAngVelRet * bodyRadSq;
        let sAngMomRet = INERTIA_K * deadMass * sAngVelRet * bodyRadSq;
        let deadAxYuk = axYukMod_in[ri];
        accumulateForce(
            &localAF, px, py, pMass, pCharge,
            pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY,
            pAxMod, pYukMod,
            delayed.x, delayed.y,
            delayed.vx, delayed.vy,
            deadMass, deadCharge,
            sAngVelRet, sMagMomRet, sAngMomRet,
            select(deadAxYuk.x, 1.0, deadAxYuk.x == 0.0),
            select(deadAxYuk.y, 1.0, deadAxYuk.y == 0.0),
            pBodyRadiusSq,
            &localJerk,
            true, // useAberration
        );
    }

    // NaN guard on total force — must happen BEFORE writing to global memory
    if (localAF.totalForce.x != localAF.totalForce.x) { localAF.totalForce = vec2(0.0, localAF.totalForce.y); }
    if (localAF.totalForce.y != localAF.totalForce.y) { localAF.totalForce = vec2(localAF.totalForce.x, 0.0); }

    // Write accumulated jerk to AllForces (NaN guard)
    localAF.jerk = vec2(
        select(localJerk.x, 0.0, localJerk.x != localJerk.x),
        select(localJerk.y, 0.0, localJerk.y != localJerk.y)
    );

    // Write accumulated forces back to global memory ONCE
    allForces[pIdx] = localAF;

    // Adaptive substepping: atomicMax of |F/m| as fixed-point u32
    let totalFSq = localAF.totalForce.x * localAF.totalForce.x + localAF.totalForce.y * localAF.totalForce.y;
    let accelSq = totalFSq * pInvMass * pInvMass;
    if (accelSq == accelSq && accelSq < 1e20) {
        let accelBits = bitcast<u32>(sqrt(accelSq));
        atomicMax(&maxAccel[0], accelBits);
    }
}
