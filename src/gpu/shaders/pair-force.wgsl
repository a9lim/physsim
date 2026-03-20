// O(N^2) tiled pairwise force computation.
// Each workgroup loads TILE_SIZE source particles into shared memory,
// then each thread accumulates forces from all sources onto its particle.
//
// Ports CPU pairForce() from forces.js. Signal delay: when relativity is on,
// each source uses retarded position/velocity from history via getDelayedStateGPU().
// Dead/retired particles exert forces via signal delay fade-out (scanned after tile loop).
// Aberration factor (1 - n_hat . v_source)^{-3} applied with retarded velocity.
//
// No Barnes-Hut (Phase 3). All force types gated by toggle bits.
//
// Uses packed ParticleState + ParticleDerived + ParticleAux structs. Jerk written to AllForces.
// Requires signal-delay-common.wgsl prepended (provides getDelayedStateGPU, DelayedState).

const TILE_SIZE: u32 = 64u;

// Source particle data loaded into shared memory per tile
struct TileParticle {
    posX: f32,
    posY: f32,
    velX: f32,
    velY: f32,
    mass: f32,
    charge: f32,
    angVel: f32,
    magMoment: f32,
    angMomentum: f32,
    axMod: f32,
    yukMod: f32,
    higgsMod: f32,
    bodyRadSq: f32,  // pow(mass, 2/3) — true body radius squared (not BH horizon)
    srcIdx: u32,     // global index for signal delay lookup
};

var<workgroup> tile: array<TileParticle, TILE_SIZE>;

// Bind group 0: uniforms
@group(0) @binding(0) var<uniform> uniforms: SimUniforms;

// Bind group 1: particle state (read_write for encoder compat) — 4 bindings
@group(1) @binding(0) var<storage, read_write> particles: array<ParticleState>;
@group(1) @binding(1) var<storage, read_write> derived: array<ParticleDerived>;
@group(1) @binding(2) var<storage, read_write> axYukMod: array<vec4<f32>>;  // packed: axMod, yukMod, higgsMod, pad
@group(1) @binding(3) var<storage, read_write> particleAux: array<ParticleAux>;

// Bind group 2: force accumulators + maxAccel — 2 bindings
@group(2) @binding(0) var<storage, read_write> allForces: array<AllForces>;
@group(2) @binding(1) var<storage, read_write> maxAccel: array<atomic<u32>>;

// Bind group 3: signal delay history (interleaved) — 2 bindings
// Bindings declared for pipeline layout; used by signal delay lookups (Task 8)
@group(3) @binding(0) var<storage, read_write> histData: array<f32>;
@group(3) @binding(1) var<storage, read_write> histMeta: array<u32>;

// Aberration constants (ABERRATION_CLAMP_MIN/MAX, ABERRATION_THRESHOLD) from wgslConstants

// ── Source data struct for force computation (used by both tile loop and dead loop) ──
struct SourceData {
    posX: f32, posY: f32,
    velX: f32, velY: f32,
    mass: f32, charge: f32,
    angVel: f32, magMoment: f32, angMomentum: f32,
    axMod: f32, yukMod: f32, higgsMod: f32,
    useAberration: bool,
};

// ── Force accumulator struct (passed by pointer to helper) ──
struct ForceAccum {
    gravX: f32, gravY: f32,
    coulX: f32, coulY: f32,
    magX: f32, magY: f32,
    gmX: f32, gmY: f32,
    pnX: f32, pnY: f32,
    yukX: f32, yukY: f32,
    totalX: f32, totalY: f32,
    jerkX: f32, jerkY: f32,
    bz: f32, bgz: f32,
    dbzdx: f32, dbzdy: f32,
    dbgzdx: f32, dbgzdy: f32,
    frameDrag: f32, tidal: f32,
};

// ── Accumulate pairwise forces from one source onto the observer ──
// Modifies accum in-place. All toggle/config params are module-scope or passed explicitly.
fn accumulatePairForce(
    pPosX: f32, pPosY: f32, pVelX: f32, pVelY: f32,
    pMass: f32, pCharge: f32, pAngVel: f32,
    pMagMom: f32, pAngMom: f32,
    pAxMod: f32, pYukMod: f32, pHiggsMod: f32,
    pRi5: f32,
    src: SourceData,
    softeningSq: f32,
    yukMu: f32, yukCutoffSq: f32,
    gravOn: bool, coulOn: bool, magOn: bool, gmOn: bool,
    onePNOn: bool, yukawaOn: bool, higgsOn: bool, radOn: bool,
    needAxMod: bool, isPeriodic: bool,
    accum: ptr<function, ForceAccum>,
) {
    // Minimum image displacement (full topology: Torus/Klein/RP2)
    var rx: f32;
    var ry: f32;
    if (isPeriodic) {
        let mi = fullMinImage(pPosX, pPosY, src.posX, src.posY);
        rx = mi.x;
        ry = mi.y;
    } else {
        rx = src.posX - pPosX;
        ry = src.posY - pPosY;
    }

    let rawRSq = rx * rx + ry * ry;
    let rSq = rawRSq + softeningSq;
    let invRSq = 1.0 / rSq;
    let invR = sqrt(invRSq);
    let invR3 = invR * invRSq;
    let invR5 = invR3 * invRSq;

    // Lienard-Wiechert aberration: (1 - n_hat . v_source)^{-3}
    var aberr: f32 = 1.0;
    if (src.useAberration) {
        let nDotV = -(rx * src.velX + ry * src.velY) * invR;
        let denom = max(1.0 - nDotV, ABERRATION_CLAMP_MIN);
        aberr = min(1.0 / (denom * denom * denom), ABERRATION_CLAMP_MAX);
    }
    let invR3a = select(invR3, invR3 * aberr, src.useAberration);
    let invR5a = select(invR5, invR5 * aberr, src.useAberration);

    // Relative velocity for jerk computation
    let vrx = src.velX - pVelX;
    let vry = src.velY - pVelY;
    let rDotVr = rx * vrx + ry * vry;

    // (v_s x r)_z for Biot-Savart
    let crossSV = src.velX * ry - src.velY * rx;

    // Axion modulation (geometric mean, guarded against negative products)
    let axModPair = select(1.0, sqrt(max(pAxMod * src.axMod, 0.0)), needAxMod);

    // -- Gravity --
    if (gravOn) {
        let k = pMass * src.mass;
        let fDir = k * invR3a;
        (*accum).gravX += rx * fDir;
        (*accum).gravY += ry * fDir;
        (*accum).totalX += rx * fDir;
        (*accum).totalY += ry * fDir;

        if (radOn) {
            let jRadial = -3.0 * rDotVr * k * invRSq * invR3a;
            (*accum).jerkX += vrx * fDir + rx * jRadial;
            (*accum).jerkY += vry * fDir + ry * jRadial;
        }

        // Tidal locking torque
        let crossRV = rx * (src.velY - pVelY) - ry * (src.velX - pVelX);
        let wOrbit = crossRV * invRSq;
        let dw = pAngVel - wOrbit;
        var coupling = src.mass;
        if (coulOn && pMass > EPSILON) {
            coupling += pCharge * src.charge / pMass;
        }
        let invR6 = invRSq * invRSq * invRSq;
        (*accum).tidal -= TIDAL_STRENGTH * coupling * coupling * pRi5 * invR6 * dw;
    }

    // -- Coulomb --
    if (coulOn) {
        let k = -(pCharge * src.charge) * axModPair;
        let fDir = k * invR3a;
        (*accum).coulX += rx * fDir;
        (*accum).coulY += ry * fDir;
        (*accum).totalX += rx * fDir;
        (*accum).totalY += ry * fDir;

        if (radOn) {
            let jRadial = -3.0 * rDotVr * k * invRSq * invR3a;
            (*accum).jerkX += vrx * fDir + rx * jRadial;
            (*accum).jerkY += vry * fDir + ry * jRadial;
        }
    }

    // -- 1PN EIH (gravity) --
    if (onePNOn && gmOn) {
        let r = 1.0 / invR;
        let nx = rx * invR;
        let ny = ry * invR;
        let v1Sq = pVelX * pVelX + pVelY * pVelY;
        let v2Sq = src.velX * src.velX + src.velY * src.velY;
        let nDotV1 = nx * pVelX + ny * pVelY;
        let nDotV2 = nx * src.velX + ny * src.velY;
        let radial = -v1Sq - 2.0 * v2Sq
            + 1.5 * nDotV2 * nDotV2
            + 5.0 * pMass * invR + 4.0 * src.mass * invR;
        let v1Coeff = 4.0 * nDotV1 - 3.0 * nDotV2;
        let v2Coeff = 3.0 * nDotV2;
        let base = pMass * src.mass * invR3;
        let fx = base * (rx * radial + (pVelX * v1Coeff + src.velX * v2Coeff) * r);
        let fy = base * (ry * radial + (pVelY * v1Coeff + src.velY * v2Coeff) * r);
        (*accum).pnX += fx;
        (*accum).pnY += fy;
        (*accum).totalX += fx;
        (*accum).totalY += fy;

        // Analytical jerk for position-only EIH term: F = m₂(5m₁+4m₂)·r/r⁴
        if (radOn) {
            let kEIH = pMass * src.mass * (5.0 * pMass + 4.0 * src.mass);
            let fDirEIH = kEIH * invRSq * invRSq;
            let jRadialEIH = -4.0 * kEIH * rDotVr * invRSq * invRSq * invRSq;
            (*accum).jerkX += vrx * fDirEIH + rx * jRadialEIH;
            (*accum).jerkY += vry * fDirEIH + ry * jRadialEIH;
        }
    }

    // -- 1PN Darwin EM --
    if (onePNOn && magOn) {
        let nx = rx * invR;
        let ny = ry * invR;
        let v2DotN = src.velX * nx + src.velY * ny;
        let v1DotN = pVelX * nx + pVelY * ny;
        let coeff = 0.5 * pCharge * src.charge * invRSq;
        let fx = coeff * (pVelX * v2DotN - 3.0 * nx * v1DotN * v2DotN);
        let fy = coeff * (pVelY * v2DotN - 3.0 * ny * v1DotN * v2DotN);
        (*accum).pnX += fx;
        (*accum).pnY += fy;
        (*accum).totalX += fx;
        (*accum).totalY += fy;
    }

    // -- 1PN Bazanski (mixed gravity+EM) --
    if (onePNOn && gmOn && magOn) {
        let crossCoeff = pCharge * src.charge * (pMass + src.mass)
            - (pCharge * pCharge * src.mass + src.charge * src.charge * pMass);
        let fDir = crossCoeff * invRSq * invRSq;
        (*accum).pnX += rx * fDir;
        (*accum).pnY += ry * fDir;
        (*accum).totalX += rx * fDir;
        (*accum).totalY += ry * fDir;

        // Analytical jerk: F = crossCoeff·r/r⁴, d/dt(1/r⁴) = -4·rDotVr/r⁶
        if (radOn) {
            let jRadial = -4.0 * crossCoeff * rDotVr * invRSq * invRSq * invRSq;
            (*accum).jerkX += vrx * fDir + rx * jRadial;
            (*accum).jerkY += vry * fDir + ry * jRadial;
        }
    }

    // -- Magnetic dipole-dipole --
    if (magOn) {
        let fDir = -3.0 * (pMagMom * src.magMoment) * invR5a * axModPair;
        (*accum).magX += rx * fDir;
        (*accum).magY += ry * fDir;
        (*accum).totalX += rx * fDir;
        (*accum).totalY += ry * fDir;

        // Analytical jerk: F = 3μ₁μ₂·r/r⁵, d/dt(1/r⁵) = -5·rDotVr/r⁷
        if (radOn) {
            let invR7a = invR5a * invRSq;
            let jRadial = 15.0 * (pMagMom * src.magMoment) * rDotVr * invR7a * axModPair;
            (*accum).jerkX += vrx * fDir + rx * jRadial;
            (*accum).jerkY += vry * fDir + ry * jRadial;
        }

        // Bz from moving charge (Biot-Savart)
        let BzMoving = src.charge * crossSV * invR3 * axModPair;
        (*accum).bz += BzMoving;

        // dBz gradients for spin-orbit
        (*accum).dbzdx += 3.0 * BzMoving * rx * invRSq + src.charge * src.velY * invR3 * axModPair;
        (*accum).dbzdy += 3.0 * BzMoving * ry * invRSq - src.charge * src.velX * invR3 * axModPair;

        // Dipole-sourced Bz: -mu/r^3
        (*accum).bz -= src.magMoment * invR3 * axModPair;
        (*accum).dbzdx -= 3.0 * src.magMoment * rx * invR5 * axModPair;
        (*accum).dbzdy -= 3.0 * src.magMoment * ry * invR5 * axModPair;
    }

    // -- Gravitomagnetic dipole-dipole --
    if (gmOn) {
        let fDir = 3.0 * (pAngMom * src.angMomentum) * invR5a;
        (*accum).gmX += rx * fDir;
        (*accum).gmY += ry * fDir;
        (*accum).totalX += rx * fDir;
        (*accum).totalY += ry * fDir;

        // Analytical jerk: F = 3L₁L₂·r/r⁵, d/dt(1/r⁵) = -5·rDotVr/r⁷
        if (radOn) {
            let invR7a = invR5a * invRSq;
            let jRadial = -15.0 * (pAngMom * src.angMomentum) * rDotVr * invR7a;
            (*accum).jerkX += vrx * fDir + rx * jRadial;
            (*accum).jerkY += vry * fDir + ry * jRadial;
        }

        // Bgz from moving mass: -m_s(v_s x r_hat)_z / r^2
        let BgzMoving = -src.mass * crossSV * invR3;
        (*accum).bgz += BgzMoving;

        // dBgz gradients for spin-orbit
        (*accum).dbgzdx += 3.0 * BgzMoving * rx * invRSq - src.mass * src.velY * invR3;
        (*accum).dbgzdy += 3.0 * BgzMoving * ry * invRSq + src.mass * src.velX * invR3;

        // Spin-sourced Bgz: -2L/r^3
        (*accum).bgz -= 2.0 * src.angMomentum * invR3;
        (*accum).dbgzdx -= 6.0 * src.angMomentum * rx * invR5;
        (*accum).dbgzdy -= 6.0 * src.angMomentum * ry * invR5;

        // Frame-dragging torque: aligns spins toward co-rotation
        (*accum).frameDrag += 2.0 * src.angMomentum * (src.angVel - pAngVel) * invR3;
    }

    // -- Yukawa --
    if (yukawaOn && rawRSq < yukCutoffSq) {
        let r_dist = 1.0 / invR;
        let mu = select(yukMu, yukMu * sqrt(pHiggsMod * src.higgsMod), higgsOn);
        let muR = mu * r_dist;
        let expMuR = select(0.0, exp(-muR), muR < 80.0);
        let yukModPair = sqrt(max(pYukMod * src.yukMod, 0.0));
        let yukCoupling = uniforms.yukawaCoupling;
        let yukInvRa = select(invR, invR * aberr, src.useAberration);
        let fDir = yukCoupling * yukModPair * pMass * src.mass * expMuR
                   * (invRSq + mu * invR) * yukInvRa;
        (*accum).yukX += rx * fDir;
        (*accum).yukY += ry * fDir;
        (*accum).totalX += rx * fDir;
        (*accum).totalY += ry * fDir;

        if (radOn) {
            let jRadial = -(3.0 * invRSq + 3.0 * mu * invR + mu * mu)
                          * rDotVr * yukCoupling * yukModPair * pMass * src.mass
                          * expMuR * invRSq * yukInvRa;
            (*accum).jerkX += vrx * fDir + rx * jRadial;
            (*accum).jerkY += vry * fDir + ry * jRadial;
        }

        // Scalar Breit 1PN correction
        if (onePNOn) {
            let nx = rx * invR;
            let ny = ry * invR;
            let nDotV1 = nx * pVelX + ny * pVelY;
            let nDotV2 = nx * src.velX + ny * src.velY;
            let v1DotV2 = pVelX * src.velX + pVelY * src.velY;
            let alpha = 1.0 + mu * r_dist;
            let beta = 0.5 * yukCoupling * yukModPair * pMass * src.mass * expMuR * invRSq;
            let radial = -(alpha * v1DotV2 + (alpha * alpha + alpha + 1.0) * nDotV1 * nDotV2);
            let fx = beta * (radial * nx + alpha * (nDotV2 * pVelX + nDotV1 * src.velX));
            let fy = beta * (radial * ny + alpha * (nDotV2 * pVelY + nDotV1 * src.velY));
            (*accum).pnX += fx;
            (*accum).pnY += fy;
            (*accum).totalX += fx;
            (*accum).totalY += fy;
        }
    }
}

// ── Build a SourceData from signal-delayed state ──
fn makeDelayedSource(
    delayed: DelayedState,
    sMass: f32, sCharge: f32,
    sAxMod: f32, sYukMod: f32, sHiggsMod: f32,
    bodyRadSq: f32,
) -> SourceData {
    var src: SourceData;
    src.posX = delayed.x;
    src.posY = delayed.y;
    src.velX = delayed.vx;
    src.velY = delayed.vy;
    src.mass = sMass;
    src.charge = sCharge;
    // Recompute dipole moments from retarded angw
    let retAngwSq = delayed.angw * delayed.angw;
    let sAngVel = delayed.angw / sqrt(1.0 + retAngwSq * bodyRadSq);
    src.angVel = sAngVel;
    src.magMoment = MAG_MOMENT_K * sCharge * sAngVel * bodyRadSq;
    src.angMomentum = INERTIA_K * sMass * sAngVel * bodyRadSq;
    src.axMod = sAxMod;
    src.yukMod = sYukMod;
    src.higgsMod = sHiggsMod;
    src.useAberration = true;
    return src;
}

@compute @workgroup_size(TILE_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = gid.x;
    let localIdx = lid.x;
    let N = uniforms.aliveCount;

    // Load this thread's particle (observer)
    let alive = idx < N && (particles[idx].flags & FLAG_ALIVE) != 0u;

    var pPosX: f32 = 0.0;
    var pPosY: f32 = 0.0;
    var pVelX: f32 = 0.0;
    var pVelY: f32 = 0.0;
    var pMass: f32 = 0.0;
    var pCharge: f32 = 0.0;
    var pAngVel: f32 = 0.0;
    var pMagMom: f32 = 0.0;
    var pAngMom: f32 = 0.0;
    var pAxMod: f32 = 1.0;
    var pYukMod: f32 = 1.0;
    var pHiggsMod: f32 = 1.0;
    var pInvMass: f32 = 0.0;
    var pRadiusSq: f32 = 0.0;
    // pRi5 = mass^(5/3) pre-hoisted for tidal locking

    // Pre-hoisted tidal locking constant: ri5 = mass^(5/3) = bodyRadiusSq^2 * bodyRadius
    var pRi5: f32 = 0.0;

    if (alive) {
        let p = particles[idx];
        pPosX = p.posX;
        pPosY = p.posY;
        pMass = p.mass;
        pCharge = p.charge;
        let d = derived[idx];
        pVelX = d.velX;
        pVelY = d.velY;
        pAngVel = d.angVel;
        pMagMom = d.magMoment;
        pAngMom = d.angMomentum;
        pInvMass = d.invMass;
        pRadiusSq = d.radiusSq;
        let aym = axYukMod[idx];
        pAxMod = aym.x;
        pYukMod = aym.y;
        pHiggsMod = aym.z;
        // G18: Hoist ri5 = mass^(5/3) = bodyRSq^2 * sqrt(bodyRSq) — avoids pow() per pair
        let pBodyRSq = d.bodyRSq;
        let pBodyR = sqrt(pBodyRSq);
        pRi5 = pBodyRSq * pBodyRSq * pBodyR;
    }

    // Read toggle bits
    let gravOn = hasToggle0(GRAVITY_BIT);
    let coulOn = hasToggle0(COULOMB_BIT);
    let magOn = hasToggle0(MAGNETIC_BIT);
    let gmOn = hasToggle0(GRAVITOMAG_BIT);
    let onePNOn = hasToggle0(ONE_PN_BIT);
    let relOn = hasToggle0(RELATIVITY_BIT);
    let yukawaOn = hasToggle0(YUKAWA_BIT);
    let higgsOn = hasToggle0(HIGGS_BIT);
    let radOn = hasToggle0(RADIATION_BIT);
    let isPeriodic = uniforms.boundaryMode == BOUND_LOOP;

    let softeningSq = uniforms.softeningSq;

    // Axion EM modulation flag
    let needAxMod = (coulOn || magOn) && hasToggle0(AXION_BIT);

    // Yukawa cutoff: exp(-mu*r) < 0.002 when mu*r > 6
    // When Higgs enabled, muEff can be as small as yukawaMu * HIGGS_MASS_FLOOR — widen cutoff
    let yukMu = uniforms.yukawaMu;
    let muMin = select(yukMu, yukMu * HIGGS_MASS_FLOOR, higgsOn);
    let yukCutoffSq = select(1e30, (6.0 / muMin) * (6.0 / muMin), yukawaOn && muMin > EPSILON);

    // Signal delay is active when relativity is on
    let signalDelayed = relOn;

    // Pre-drift signal delay time: forces are computed at pre-drift positions,
    // so use simTime BEFORE the dt increment to match the particle's actual
    // spacetime location (CPU increments simTime after drift, before recompute).
    let sdTime = uniforms.simTime - uniforms.dt;

    // Per-thread force accumulator
    var accum: ForceAccum;
    accum.gravX = 0.0; accum.gravY = 0.0;
    accum.coulX = 0.0; accum.coulY = 0.0;
    accum.magX = 0.0; accum.magY = 0.0;
    accum.gmX = 0.0; accum.gmY = 0.0;
    accum.pnX = 0.0; accum.pnY = 0.0;
    accum.yukX = 0.0; accum.yukY = 0.0;
    accum.totalX = 0.0; accum.totalY = 0.0;
    accum.jerkX = 0.0; accum.jerkY = 0.0;
    accum.bz = 0.0; accum.bgz = 0.0;
    accum.dbzdx = 0.0; accum.dbzdy = 0.0;
    accum.dbgzdx = 0.0; accum.dbgzdy = 0.0;
    accum.frameDrag = 0.0; accum.tidal = 0.0;

    // Number of tiles needed to cover all particles
    let numTiles = (N + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < numTiles; t++) {
        // Collaborative tile load: each thread loads one source particle
        let tileSrcIdx = t * TILE_SIZE + localIdx;
        if (tileSrcIdx < N && (particles[tileSrcIdx].flags & FLAG_ALIVE) != 0u) {
            let sp = particles[tileSrcIdx];
            tile[localIdx].posX = sp.posX;
            tile[localIdx].posY = sp.posY;
            let sd = derived[tileSrcIdx];
            tile[localIdx].velX = sd.velX;
            tile[localIdx].velY = sd.velY;
            tile[localIdx].mass = sp.mass;
            tile[localIdx].charge = sp.charge;
            tile[localIdx].angVel = sd.angVel;
            tile[localIdx].magMoment = sd.magMoment;
            tile[localIdx].angMomentum = sd.angMomentum;
            let saym = axYukMod[tileSrcIdx];
            tile[localIdx].axMod = saym.x;
            tile[localIdx].yukMod = saym.y;
            tile[localIdx].higgsMod = saym.z;
            tile[localIdx].bodyRadSq = derived[tileSrcIdx].bodyRSq; // G20: read precomputed value
            tile[localIdx].srcIdx = tileSrcIdx;
        } else {
            // Mark as invalid (zero mass = no force contribution)
            tile[localIdx].mass = 0.0;
            tile[localIdx].srcIdx = 0xFFFFFFFFu;
        }

        workgroupBarrier();

        // Each thread accumulates forces from all sources in this tile
        if (alive) {
            for (var j: u32 = 0u; j < TILE_SIZE; j++) {
                let sIdx = t * TILE_SIZE + j;
                if (sIdx >= N || sIdx == idx) { continue; }
                let s = tile[j];
                if (s.mass < EPSILON) { continue; }

                // Build SourceData for this source particle
                var src: SourceData;
                if (signalDelayed) {
                    // Look up signal-delayed state for this source
                    let delayed = getDelayedStateGPU(
                        s.srcIdx, pPosX, pPosY, sdTime,
                        isPeriodic, uniforms.domainW, uniforms.domainH,
                        uniforms.topologyMode, false
                    );
                    if (!delayed.valid) { continue; }
                    src = makeDelayedSource(delayed, s.mass, s.charge, s.axMod, s.yukMod, s.higgsMod, s.bodyRadSq);
                } else {
                    src.posX = s.posX;
                    src.posY = s.posY;
                    src.velX = s.velX;
                    src.velY = s.velY;
                    src.mass = s.mass;
                    src.charge = s.charge;
                    src.angVel = s.angVel;
                    src.magMoment = s.magMoment;
                    src.angMomentum = s.angMomentum;
                    src.axMod = s.axMod;
                    src.yukMod = s.yukMod;
                    src.higgsMod = s.higgsMod;
                    src.useAberration = false;
                }

                accumulatePairForce(
                    pPosX, pPosY, pVelX, pVelY,
                    pMass, pCharge, pAngVel,
                    pMagMom, pAngMom,
                    pAxMod, pYukMod, pHiggsMod, pRi5,
                    src, softeningSq,
                    yukMu, yukCutoffSq,
                    gravOn, coulOn, magOn, gmOn,
                    onePNOn, yukawaOn, higgsOn, radOn,
                    needAxMod, isPeriodic,
                    &accum,
                );
            }
        }

        workgroupBarrier();
    }

    // ── Dead particle loop: signal delay fade-out ──
    // Dead/retired particles continue exerting forces via their signal delay history
    if (alive && signalDelayed) {
        let totalCount = uniforms.particleCount;
        for (var ri: u32 = 0u; ri < totalCount; ri++) {
            let flags = particles[ri].flags;
            // Retired but not alive = dead particle
            if ((flags & FLAG_RETIRED) == 0u || (flags & FLAG_ALIVE) != 0u) { continue; }

            let delayed = getDelayedStateGPU(
                ri, pPosX, pPosY, sdTime,
                isPeriodic, uniforms.domainW, uniforms.domainH,
                uniforms.topologyMode, true
            );
            if (!delayed.valid) { continue; }

            // Use death state from particleAux
            let aux = particleAux[ri];
            let deadMass = aux.deathMass;
            let deadCharge = particles[ri].charge;
            let bodyRadSq = pow(deadMass, 2.0 / 3.0);
            let deadAxYuk = axYukMod[ri];

            let src = makeDelayedSource(
                delayed, deadMass, deadCharge,
                select(deadAxYuk.x, 1.0, abs(deadAxYuk.x) < EPSILON),
                select(deadAxYuk.y, 1.0, abs(deadAxYuk.y) < EPSILON),
                select(deadAxYuk.z, 1.0, abs(deadAxYuk.z) < EPSILON),
                bodyRadSq
            );

            accumulatePairForce(
                pPosX, pPosY, pVelX, pVelY,
                pMass, pCharge, pAngVel,
                pMagMom, pAngMom,
                pAxMod, pYukMod, pHiggsMod, pRi5,
                src, softeningSq,
                yukMu, yukCutoffSq,
                gravOn, coulOn, magOn, gmOn,
                onePNOn, yukawaOn, higgsOn, radOn,
                needAxMod, isPeriodic,
                &accum,
            );
        }
    }

    // Write accumulated forces to packed AllForces struct
    if (alive) {
        var af: AllForces;
        af.f0 = vec4(accum.gravX, accum.gravY, accum.coulX, accum.coulY);
        af.f1 = vec4(accum.magX, accum.magY, accum.gmX, accum.gmY);
        af.f2 = vec4(accum.pnX, accum.pnY, 0.0, 0.0);  // spinCurv filled by spin-orbit pass
        af.f3 = vec4(0.0, 0.0, accum.yukX, accum.yukY);   // radiation.xy filled by Phase 4
        af.f4 = vec4(0.0, 0.0, 0.0, 0.0);           // external/higgs filled by later passes
        af.f5 = vec4(0.0, 0.0, 0.0, 0.0);           // axion filled by later passes
        af.torques = vec4(0.0, accum.frameDrag, accum.tidal, 0.0);
        af.bFields = vec4(accum.bz, accum.bgz, 0.0, 0.0);  // extBz added by external fields pass
        af.bFieldGrads = vec4(accum.dbzdx, accum.dbzdy, accum.dbgzdx, accum.dbgzdy);
        // NaN guard on total force and jerk — must happen BEFORE writing to global memory
        af.totalForce = vec2(
            select(accum.totalX, 0.0, accum.totalX != accum.totalX),
            select(accum.totalY, 0.0, accum.totalY != accum.totalY)
        );
        af.jerk = vec2(
            select(accum.jerkX, 0.0, accum.jerkX != accum.jerkX),
            select(accum.jerkY, 0.0, accum.jerkY != accum.jerkY)
        );

        allForces[idx] = af;

        // Adaptive substepping: atomicMax of |F/m|^2 as fixed-point u32
        let totalFSq = af.totalForce.x * af.totalForce.x + af.totalForce.y * af.totalForce.y;
        let accelSq = totalFSq * pInvMass * pInvMass;
        // Guard against NaN/Inf which would corrupt adaptive stepping
        if (accelSq == accelSq && accelSq < 1e20) {
            let accelBits = bitcast<u32>(sqrt(accelSq));
            atomicMax(&maxAccel[0], accelBits);
        }
    }
}
