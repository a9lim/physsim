// ─── Barnes-Hut Tree Walk Force Computation ───
// One thread per alive particle. Stack-based iterative traversal.
// Uses theta=0.5 opening angle criterion.
// Accumulates into the same force buffers as pairwise path.

// Constants provided by generated wgslConstants block.
// Shader-specific constants:
const NONE: i32 = -1;
const MAX_STACK: u32 = 48u;

// Node layout accessors from shared-tree-nodes.wgsl (prepended)

@group(0) @binding(0) var<storage, read_write> nodes: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: SimUniforms;

// Group 1: packed particle structs (read_write for encoder compat)
@group(1) @binding(0) var<storage, read_write> particleState: array<ParticleState>;
@group(1) @binding(1) var<storage, read_write> particleAux: array<ParticleAux>;
@group(1) @binding(2) var<storage, read_write> derived_in: array<ParticleDerived>;
@group(1) @binding(3) var<storage, read_write> axYukMod_in: array<vec4<f32>>; // packed: axMod, yukMod, higgsMod, pad
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

// Aberration constants (ABERRATION_CLAMP_MIN/MAX) from wgslConstants

// Inline the core force accumulation (matches pairForce in forces.js):
// jerkOut: function-scope pointer for accumulating analytical jerk (radiation)
fn accumulateForce(
    af: ptr<function, AllForces>,
    px: f32, py: f32,
    pMass: f32, pCharge: f32,
    pMagMoment: f32, pAngMomentum: f32,
    pAngVel: f32, pVelX: f32, pVelY: f32,
    pAxMod: f32, pYukMod: f32, pHiggsMod: f32,
    sx: f32, sy: f32,
    svx: f32, svy: f32,
    sMass: f32, sCharge: f32,
    sAngVel: f32, sMagMoment: f32, sAngMomentum: f32,
    sAxMod: f32, sYukMod: f32, sHiggsMod: f32,
    pRi5: f32,
    jerkOut: ptr<function, vec2<f32>>,
    useAberration: bool,
) {
    let toggles = uniforms.toggles0;
    let softeningSq = uniforms.softeningSq;

    // Displacement (with minimum-image for periodic boundaries)
    let periodic = uniforms.boundaryMode == BOUND_LOOP;
    var disp = vec2<f32>(sx - px, sy - py);
    if (periodic) {
        disp = fullMinImageP(px, py, sx, sy, uniforms.domainW, uniforms.domainH, uniforms.topologyMode);
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
        let invR6 = invRSq * invRSq * invRSq;
        (*af).torques.z += -TIDAL_STRENGTH * coupling * coupling * pRi5 * invR6 * dw;
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
        // Dipole-dipole radial: F = -3*mu1*mu2/r^4 (aligned dipoles repel)
        let fDir = -3.0 * (pMagMoment * sMagMoment) * invR5a * axMod;
        let fx = rx * fDir;
        let fy = ry * fDir;
        (*af).f1.x += fx;
        (*af).f1.y += fy;
        (*af).totalForce.x += fx;
        (*af).totalForce.y += fy;

        // Analytical jerk: F = 3μ₁μ₂·r/r⁵, d/dt(1/r⁵) = -5·rDotVr/r⁷
        if (radOn) {
            let invR7a = invR5a * invRSq;
            let jRadial = 15.0 * (pMagMoment * sMagMoment) * rDotVr * invR7a * axMod;
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

    // Yukawa (Higgs-modulated μ when both enabled)
    if ((toggles & YUKAWA_BIT) != 0u) {
        let higgsOn = (toggles & HIGGS_BIT) != 0u;
        let mu = select(uniforms.yukawaMu, uniforms.yukawaMu * sqrt(pHiggsMod * sHiggsMod), higgsOn);
        let cutoffSq = select((6.0 / uniforms.yukawaMu) * (6.0 / uniforms.yukawaMu), (6.0 / max(mu, EPSILON)) * (6.0 / max(mu, EPSILON)), higgsOn);
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
        let base = pMass * sMass * invR3;
        let eihX = base * (rx * radial + (pVelX * v1Coeff + svx * v2Coeff) * r_val);
        let eihY = base * (ry * radial + (pVelY * v1Coeff + svy * v2Coeff) * r_val);
        (*af).f2.x += eihX;
        (*af).f2.y += eihY;
        (*af).totalForce.x += eihX;
        (*af).totalForce.y += eihY;

        // Analytical jerk for position-only EIH term: F = m₂(5m₁+4m₂)·r/r⁴
        if (radOn) {
            let kEIH = pMass * sMass * (5.0 * pMass + 4.0 * sMass);
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
    let pBodyRadiusSq = pDerived.bodyRSq; // true body radius² from cache-derived
    // Pre-hoist ri5 = mass^(5/3) for tidal locking (avoids pow() per pair in accumulateForce)
    let pBodyRadius = sqrt(pBodyRadiusSq);
    let pRi5 = pBodyRadiusSq * pBodyRadiusSq * pBodyRadius;
    let pAngW = ps.angW;
    let pAym = axYukMod_in[pIdx];
    let pAxMod = pAym.x;
    let pYukMod = pAym.y;
    let pHiggsMod = pAym.z;

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
            comDisp = fullMinImageP(px, py, comX, comY, uniforms.domainW, uniforms.domainH, uniforms.topologyMode);
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
            let sIsRetired = (sPs.flags & FLAG_RETIRED) != 0u && (sPs.flags & FLAG_ALIVE) == 0u;
            if ((sPs.flags & FLAG_ALIVE) == 0u && !sIsRetired) { continue; }

            // Retired leaf particle: use signal delay with isDead=true
            if (sIsRetired) {
                let delayed = getDelayedStateGPU(
                    sIdx, px, py, sdTime,
                    isPeriodic, uniforms.domainW, uniforms.domainH,
                    uniforms.topologyMode, true // isDead
                );
                if (!delayed.valid) { continue; }
                let sAuxR = particleAux[sIdx];
                let deadMass = sAuxR.deathMass;
                let deadCharge = sPs.charge;
                let bodyRadSq = pow(deadMass, 2.0 / 3.0);
                let retAngwSq = delayed.angw * delayed.angw;
                let sAngVelRet = delayed.angw / sqrt(1.0 + retAngwSq * bodyRadSq);
                let sMagMomRet = MAG_MOMENT_K * deadCharge * sAngVelRet * bodyRadSq;
                let sAngMomRet = INERTIA_K * deadMass * sAngVelRet * bodyRadSq;
                let deadAxYuk = axYukMod_in[sIdx];
                accumulateForce(
                    &localAF, px, py, pMass, pCharge,
                    pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY,
                    pAxMod, pYukMod, pHiggsMod,
                    delayed.x, delayed.y,
                    delayed.vx, delayed.vy,
                    deadMass, deadCharge,
                    sAngVelRet, sMagMomRet, sAngMomRet,
                    select(deadAxYuk.x, 1.0, abs(deadAxYuk.x) < EPSILON),
                    select(deadAxYuk.y, 1.0, abs(deadAxYuk.y) < EPSILON),
                    select(deadAxYuk.z, 1.0, abs(deadAxYuk.z) < EPSILON),
                    pRi5,
                    &localJerk,
                    true, // useAberration
                );
                continue;
            }

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
                // Recompute dipoles from retarded angw (bodyRSq cached in derived)
                let bodyRadSq = derived_in[sIdx].bodyRSq;
                let retAngwSq = delayed.angw * delayed.angw;
                let sAngVelRet = delayed.angw / sqrt(1.0 + retAngwSq * bodyRadSq);
                let sMagMomRet = MAG_MOMENT_K * sPs.charge * sAngVelRet * bodyRadSq;
                let sAngMomRet = INERTIA_K * sPs.mass * sAngVelRet * bodyRadSq;
                accumulateForce(
                    &localAF, px, py, pMass, pCharge,
                    pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY,
                    pAxMod, pYukMod, pHiggsMod,
                    delayed.x, delayed.y,
                    delayed.vx, delayed.vy,
                    sPs.mass, sPs.charge,
                    sAngVelRet, sMagMomRet, sAngMomRet,
                    sAYM.x, sAYM.y, sAYM.z,
                    pRi5,
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
                // Recompute dipoles from retarded angw (bodyRSq cached in derived)
                let bodyRadSq = derived_in[sIdx].bodyRSq;
                let retAngwSq = delayed.angw * delayed.angw;
                let sAngVelRet = delayed.angw / sqrt(1.0 + retAngwSq * bodyRadSq);
                let sMagMomRet = MAG_MOMENT_K * sPs.charge * sAngVelRet * bodyRadSq;
                let sAngMomRet = INERTIA_K * sPs.mass * sAngVelRet * bodyRadSq;
                accumulateForce(
                    &localAF, px, py, pMass, pCharge,
                    pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY,
                    pAxMod, pYukMod, pHiggsMod,
                    gsx, gsy,
                    delayed.vx, delayed.vy,
                    sPs.mass, sPs.charge,
                    sAngVelRet, sMagMomRet, sAngMomRet,
                    sAYM.x, sAYM.y, sAYM.z,
                    pRi5,
                    &localJerk,
                    true, // useAberration
                );
            } else {
                // No signal delay: use current positions
                let sDerived = derived_in[sIdx];
                accumulateForce(
                    &localAF, px, py, pMass, pCharge,
                    pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY,
                    pAxMod, pYukMod, pHiggsMod,
                    sPs.posX, sPs.posY,
                    sDerived.velX, sDerived.velY,
                    sPs.mass, sPs.charge,
                    sDerived.angVel,
                    sDerived.magMoment, sDerived.angMomentum,
                    sAYM.x, sAYM.y, sAYM.z,
                    pRi5,
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
                pAxMod, pYukMod, pHiggsMod,
                comX, comY,
                avgVx, avgVy,
                nodeMass, getTotalCharge(nodeIdx),
                0.0, // sAngVel = 0 for aggregate
                getTotalMagMoment(nodeIdx), getTotalAngMomentum(nodeIdx),
                1.0, 1.0, 1.0, // axMod/yukMod/higgsMod = 1 for aggregate
                pRi5,
                &localJerk,
                false, // no aberration on aggregate nodes — velocities are not retarded
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

    // Dead (retired) particles are now inserted into the BH tree and handled
    // at leaf level during the tree walk above, replacing the former O(N²) pairwise scan.
    // The pairwise scan in pair-force.wgsl is still needed when BH is off.

    // NaN guard on total force — must happen BEFORE writing to global memory
    localAF.totalForce = vec2(
        select(localAF.totalForce.x, 0.0, localAF.totalForce.x != localAF.totalForce.x),
        select(localAF.totalForce.y, 0.0, localAF.totalForce.y != localAF.totalForce.y)
    );

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
