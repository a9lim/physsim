// ─── Barnes-Hut Tree Walk Force Computation ───
// One thread per alive particle. Stack-based iterative traversal.
// Uses theta=0.5 opening angle criterion.
// Accumulates into the same force buffers as pairwise path.

const NONE: i32 = -1;
const MAX_STACK: u32 = 48u;
const EPSILON: f32 = 1e-9;
const FLAG_ALIVE:   u32 = 1u;
const FLAG_RETIRED: u32 = 2u;
const FLAG_GHOST:   u32 = 16u;

const GRAVITY_BIT:     u32 = 1u;
const COULOMB_BIT:     u32 = 2u;
const MAGNETIC_BIT:    u32 = 4u;
const GRAVITOMAG_BIT:  u32 = 8u;
const ONEPN_BIT:       u32 = 16u;
const RELATIVITY_BIT:  u32 = 32u;
const YUKAWA_BIT:      u32 = 2048u;
const AXION_BIT:       u32 = 8192u;
const RADIATION_BIT:   u32 = 128u;

const MAG_MOMENT_K: f32 = 0.2;
const INERTIA_K: f32 = 0.4;
const TIDAL_STRENGTH: f32 = 0.3;

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
    _pad: vec2<f32>,
};

struct RadiationState {
    jerkX: f32, jerkY: f32,
    radAccum: f32, hawkAccum: f32, yukawaRadAccum: f32,
    radDisplayX: f32, radDisplayY: f32,
    _pad: f32,
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

// Group 2: force accumulators
@group(2) @binding(0) var<storage, read_write> allForces: array<AllForces>;
@group(2) @binding(1) var<storage, read_write> radiationState: array<RadiationState>;
@group(2) @binding(2) var<storage, read_write> maxAccel: array<atomic<u32>>;

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
) {
    let toggles = uniforms.toggles0;
    let softeningSq = uniforms.softeningSq;

    // Displacement
    let rx = sx - px;
    let ry = sy - py;
    let rawRSq = rx * rx + ry * ry;
    let rSq = rawRSq + softeningSq;
    let invRSq = 1.0 / rSq;
    let invR = sqrt(invRSq);
    let invR3 = invR * invRSq;
    let invR5 = invR3 * invRSq;

    // Lienard-Wiechert aberration: (1 - n_hat dot v_source)^{-3}
    let signalDelayed = (toggles & RELATIVITY_BIT) != 0u;
    var aberr: f32 = 1.0;
    if (signalDelayed) {
        let nDotV = -(rx * svx + ry * svy) * invR;
        let denom = max(1.0 - nDotV, ABERRATION_CLAMP_MIN);
        aberr = min(1.0 / (denom * denom * denom), ABERRATION_CLAMP_MAX);
    }
    let invR3a = select(invR3, invR3 * aberr, signalDelayed);
    let invR5a = select(invR5, invR5 * aberr, signalDelayed);

    let needAxMod = ((toggles & COULOMB_BIT) != 0u || (toggles & MAGNETIC_BIT) != 0u) && (toggles & AXION_BIT) != 0u;
    var axModPair: f32 = 1.0;
    if (needAxMod) {
        axModPair = sqrt(pAxMod * sAxMod);
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
            let expMuR = exp(-mu * r);
            let yukModPair = sqrt(pYukMod * sYukMod);
            let yukInvRa = select(invR, invR * aberr, signalDelayed);
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
            if ((toggles & ONEPN_BIT) != 0u) {
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
    if ((toggles & ONEPN_BIT) != 0u && (toggles & GRAVITOMAG_BIT) != 0u) {
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
    }

    // 1PN Darwin EM (magnetic + 1PN)
    if ((toggles & ONEPN_BIT) != 0u && (toggles & MAGNETIC_BIT) != 0u) {
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
    if ((toggles & ONEPN_BIT) != 0u && (toggles & GRAVITOMAG_BIT) != 0u && (toggles & MAGNETIC_BIT) != 0u) {
        let crossCoeff = pCharge * sCharge * (pMass + sMass)
            - (pCharge * pCharge * sMass + sCharge * sCharge * pMass);
        let fDir = crossCoeff * invRSq * invRSq;
        let bazX = rx * fDir;
        let bazY = ry * fDir;
        (*af).f2.x += bazX;
        (*af).f2.y += bazY;
        (*af).totalForce.x += bazX;
        (*af).totalForce.y += bazY;
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
        let dx = comX - px;
        let dy = comY - py;
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

            let sDerived = derived_in[sIdx];
            let sAYM = axYukMod_in[sIdx];
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
            );
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

    // After tree walk, iterate retired particles (pairwise, matching CPU behavior)
    for (var ri = 0u; ri < uniforms.aliveCount; ri++) {
        let rPs = particleState[ri];
        if ((rPs.flags & FLAG_RETIRED) == 0u) { continue; }
        if ((rPs.flags & FLAG_ALIVE) != 0u) { continue; }
        let rAux = particleAux[ri];
        // Phase 4 will add signal delay lookup here
        // For now, use current (frozen) position as approximation
        accumulateForce(
            &localAF, px, py, pMass, pCharge,
            pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY,
            pAxMod, pYukMod,
            rPs.posX, rPs.posY, 0.0, 0.0, // vel = 0 for dead
            rAux.deathMass, rPs.charge,
            0.0, 0.0, 0.0, // no spin data from dead
            1.0, 1.0,
            pBodyRadiusSq,
            &localJerk,
        );
    }

    // Write accumulated forces back to global memory ONCE
    allForces[pIdx] = localAF;

    // Write accumulated jerk for radiation reaction
    var rs = radiationState[pIdx];
    rs.jerkX = localJerk.x;
    rs.jerkY = localJerk.y;
    radiationState[pIdx] = rs;

    // Adaptive substepping: atomicMax of |F/m| as fixed-point u32
    let totalFSq = localAF.totalForce.x * localAF.totalForce.x + localAF.totalForce.y * localAF.totalForce.y;
    let accelSq = totalFSq * pInvMass * pInvMass;
    let accelBits = bitcast<u32>(sqrt(accelSq));
    atomicMax(&maxAccel[0], accelBits);
}
