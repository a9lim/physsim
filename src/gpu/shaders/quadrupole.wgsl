// ─── Quadrupole Radiation Shaders ───
// Three entry points dispatched once per frame (after all substeps):
//   quadrupoleCoM    — workgroup-reduce center of mass + totalKE
//   quadrupoleContrib — compute per-particle d³I/d³Q + update residual force history, workgroup-reduce global sums
//   quadrupoleApply  — finalize power, apply drag, accumulate energy, emit photons/gravitons
//
// Standalone shader — defines own structs (NOT prepended with common.wgsl).
// Constants provided by generated wgslConstants block.

// ── RNG ──
fn pcgHash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn pcgRand(seed: u32) -> f32 {
    return f32(pcgHash(seed)) / 4294967296.0;
}

// ── Packed struct definitions (must match common.wgsl / writeUniforms() byte layout) ──

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
    f0: vec4<f32>,          // gravity.xy, coulomb.xy
    f1: vec4<f32>,          // magnetic.xy, gravitomag.xy
    f2: vec4<f32>,          // f1pn.xy, spinCurv.xy
    f3: vec4<f32>,          // radiation.xy, yukawa.xy
    f4: vec4<f32>,          // external.xy, higgs.xy
    f5: vec4<f32>,          // axion.xy, pad, pad
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
    qResFx0: f32,
    qResFy0: f32,
    qResFx1: f32,
    qResFy1: f32,
    qResCount: f32,
    quadAccum: f32,
    emQuadAccum: f32,
    d3IContrib: f32,
    d3QContrib: f32,
    otherFx0: f32,
    otherFy0: f32,
    otherFx1: f32,
    otherFy1: f32,
    otherCount: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct Photon {
    posX: f32, posY: f32,
    velX: f32, velY: f32,
    energy: f32,
    emitterId: u32, lifetime: f32, flags: u32,
};

struct Uniforms {
    dt: f32,
    simTime: f32,
    domainW: f32,
    domainH: f32,
    _speedScale: f32,
    _softening: f32,
    _softeningSq: f32,
    toggles0: u32,
    _toggles1: u32,
    yukawaCoupling: f32,
    _yukawaMu: f32,
    _higgsMass: f32,
    _axionMass: f32,
    _boundaryMode: u32,
    _topologyMode: u32,
    _collisionMode: u32,
    _maxParticles: u32,
    aliveCount: u32,
    _extGravity: f32,
    _extGravityAngle: f32,
    _extElectric: f32,
    _extElectricAngle: f32,
    _extBz: f32,
    _bounceFriction: f32,
    _extGx: f32,
    _extGy: f32,
    _extEx: f32,
    _extEy: f32,
    _axionCoupling: f32,
    _higgsCoupling: f32,
    _particleCount: u32,
    _bhTheta: f32,
    frameCount: u32,
    _pad4: u32,
};

// ── Bind groups (shared by all 3 entry points) ──

@group(0) @binding(0) var<uniform> u: Uniforms;

// Group 1: particle data
@group(1) @binding(0) var<storage, read_write> particles: array<ParticleState>;
@group(1) @binding(1) var<storage, read_write> particleAux: array<ParticleAux>;
@group(1) @binding(2) var<storage, read_write> derived: array<ParticleDerived>;
@group(1) @binding(3) var<storage, read_write> allForces: array<AllForces>;
@group(1) @binding(4) var<storage, read_write> radState: array<RadiationState>;

// Group 2: photon pool
@group(2) @binding(0) var<storage, read_write> photons: array<Photon>;
@group(2) @binding(1) var<storage, read_write> phCount: atomic<u32>;

// Group 3: reduction buffer
// Layout: [0..MAX_QUAD_WG*4): CoM partials, [MAX_QUAD_WG*4..MAX_QUAD_WG*12): d³ partials
@group(3) @binding(0) var<storage, read_write> reductionBuf: array<f32>;

// ── Workgroup shared memory ──
var<workgroup> sh_comXw: array<f32, 64>;
var<workgroup> sh_comYw: array<f32, 64>;
var<workgroup> sh_mass: array<f32, 64>;
var<workgroup> sh_ke: array<f32, 64>;

// For contrib pass: 8 accumulators
var<workgroup> sh_d3Ixx: array<f32, 64>;
var<workgroup> sh_d3Ixy: array<f32, 64>;
var<workgroup> sh_d3Iyy: array<f32, 64>;
var<workgroup> sh_d3Qxx: array<f32, 64>;
var<workgroup> sh_d3Qxy: array<f32, 64>;
var<workgroup> sh_d3Qyy: array<f32, 64>;
var<workgroup> sh_totalD3I: array<f32, 64>;
var<workgroup> sh_totalD3Q: array<f32, 64>;

// ── Helper: workgroup tree reduction (power-of-2 stride) ──
// After this, lane 0 holds the sum. All arrays of same size share the barrier.
fn workgroupReduce4(lid: u32) {
    // Reduce 4 arrays in lockstep (CoM pass)
    for (var stride: u32 = 32u; stride > 0u; stride >>= 1u) {
        workgroupBarrier();
        if (lid < stride) {
            sh_comXw[lid] += sh_comXw[lid + stride];
            sh_comYw[lid] += sh_comYw[lid + stride];
            sh_mass[lid] += sh_mass[lid + stride];
            sh_ke[lid] += sh_ke[lid + stride];
        }
    }
    workgroupBarrier();
}

fn workgroupReduce8(lid: u32) {
    // Reduce 8 arrays in lockstep (contrib pass)
    for (var stride: u32 = 32u; stride > 0u; stride >>= 1u) {
        workgroupBarrier();
        if (lid < stride) {
            sh_d3Ixx[lid] += sh_d3Ixx[lid + stride];
            sh_d3Ixy[lid] += sh_d3Ixy[lid + stride];
            sh_d3Iyy[lid] += sh_d3Iyy[lid + stride];
            sh_d3Qxx[lid] += sh_d3Qxx[lid + stride];
            sh_d3Qxy[lid] += sh_d3Qxy[lid + stride];
            sh_d3Qyy[lid] += sh_d3Qyy[lid + stride];
            sh_totalD3I[lid] += sh_totalD3I[lid + stride];
            sh_totalD3Q[lid] += sh_totalD3Q[lid + stride];
        }
    }
    workgroupBarrier();
}

// ═══════════════════════════════════════════════════════════════════
// Entry Point 1: quadrupoleCoM
// Workgroup-reduce CoM + totalKE. History shift deferred to pass 2 (after jerk uses it).
// ═══════════════════════════════════════════════════════════════════
@compute @workgroup_size(64)
fn quadrupoleCoM(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let i = gid.x;
    let localId = lid.x;
    let alive = i < u.aliveCount && (particles[i].flags & FLAG_ALIVE) != 0u;

    // Load per-particle values
    var mXw: f32 = 0.0;
    var mYw: f32 = 0.0;
    var m: f32 = 0.0;
    var ke: f32 = 0.0;

    if (alive) {
        let mass = particles[i].mass;
        mXw = particles[i].posX * mass;
        mYw = particles[i].posY * mass;
        m = mass;

        // KE = mass * wSq / (gamma + 1)
        let wx = particles[i].velWX;
        let wy = particles[i].velWY;
        let wSq = wx * wx + wy * wy;
        if (wSq > EPSILON_SQ) {
            ke = mass * wSq / (sqrt(1.0 + wSq) + 1.0);
        }
        // NOTE: residual force history shift is done in quadrupoleContrib (pass 2)
        // AFTER the backward-difference jerk computation uses the old history values.
    }

    // Store into shared memory
    sh_comXw[localId] = mXw;
    sh_comYw[localId] = mYw;
    sh_mass[localId] = m;
    sh_ke[localId] = ke;

    // Workgroup reduction
    workgroupReduce4(localId);

    // Lane 0 writes partial sums to reduction buffer
    if (localId == 0u) {
        let base = wgid.x * 4u;
        reductionBuf[base + 0u] = sh_comXw[0];
        reductionBuf[base + 1u] = sh_comYw[0];
        reductionBuf[base + 2u] = sh_mass[0];
        reductionBuf[base + 3u] = sh_ke[0];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Entry Point 2: quadrupoleContrib
// Finalize CoM from partial sums. Compute per-particle d³I/d³Q contributions.
// Update residual force history AFTER jerk computation (matches CPU order).
// Workgroup-reduce to global d³ sums.
// ═══════════════════════════════════════════════════════════════════
@compute @workgroup_size(64)
fn quadrupoleContrib(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let i = gid.x;
    let localId = lid.x;
    let numWG = (u.aliveCount + 63u) / 64u;

    // Finalize CoM from all workgroup partial sums (each thread reads all — max 64 iterations)
    var comXw: f32 = 0.0;
    var comYw: f32 = 0.0;
    var totalMass: f32 = 0.0;
    for (var wg: u32 = 0u; wg < numWG; wg++) {
        let base = wg * 4u;
        comXw += reductionBuf[base + 0u];
        comYw += reductionBuf[base + 1u];
        totalMass += reductionBuf[base + 2u];
    }
    var comX: f32 = 0.0;
    var comY: f32 = 0.0;
    if (totalMass > EPSILON) {
        comX = comXw / totalMass;
        comY = comYw / totalMass;
    }

    let alive = i < u.aliveCount && (particles[i].flags & FLAG_ALIVE) != 0u;
    let gravOn = (u.toggles0 & GRAVITY_BIT) != 0u;
    let coulombOn = (u.toggles0 & COULOMB_BIT) != 0u;
    let gwQuad = gravOn;
    let emQuad = coulombOn;

    // Per-particle d³ contribution
    var d3I_xx: f32 = 0.0; var d3I_xy: f32 = 0.0; var d3I_yy: f32 = 0.0;
    var d3Q_xx: f32 = 0.0; var d3Q_xy: f32 = 0.0; var d3Q_yy: f32 = 0.0;
    var contribI: f32 = 0.0;
    var contribQ: f32 = 0.0;

    if (alive) {
        // Coordinate velocity from proper velocity
        let wx = particles[i].velWX;
        let wy = particles[i].velWY;
        let wSq = wx * wx + wy * wy;
        let invGamma = 1.0 / sqrt(1.0 + wSq);
        let vx = wx * invGamma;
        let vy = wy * invGamma;

        // CoM-relative position
        let x = particles[i].posX - comX;
        let y = particles[i].posY - comY;

        // Total force
        let Fx = allForces[i].totalForce.x;
        let Fy = allForces[i].totalForce.y;

        // Total jerk = analytical (from force pass) + backward-difference residual
        var Jx = radState[i].jerkX;
        var Jy = radState[i].jerkY;

        let rs = radState[i];
        let count = u32(rs.qResCount);
        // Use PHYSICS_DT (fixed timestep constant) for backward-difference spacing,
        // matching CPU where quadrupole runs once per update() with dt ≈ PHYSICS_DT.
        let dt = PHYSICS_DT;

        if (count >= 2u && dt > EPSILON) {
            // O(dt²) 3-point backward derivative
            let invDt = 1.0 / dt;
            let c0 = 0.5 * invDt;
            let c1 = -2.0 * invDt;
            let c2 = 1.5 * invDt;
            // Current residual = totalForce - grav - coulomb - yukawa
            let af = allForces[i];
            let resFx = af.totalForce.x - af.f0.x - af.f0.z - af.f3.z;
            let resFy = af.totalForce.y - af.f0.y - af.f0.w - af.f3.w;
            Jx += c0 * rs.qResFx0 + c1 * rs.qResFx1 + c2 * resFx;
            Jy += c0 * rs.qResFy0 + c1 * rs.qResFy1 + c2 * resFy;
        } else if (count >= 1u && dt > EPSILON) {
            // O(dt) 2-point backward fallback
            let af = allForces[i];
            let resFx = af.totalForce.x - af.f0.x - af.f0.z - af.f3.z;
            let resFy = af.totalForce.y - af.f0.y - af.f0.w - af.f3.w;
            Jx += (resFx - rs.qResFx1) / dt;
            Jy += (resFy - rs.qResFy1) / dt;
        }
        // else: jerk not ready for this particle, use analytical only

        // Mass quadrupole d³I_ij/dt³
        if (gwQuad) {
            let d3I_xx_i = 6.0 * vx * Fx + 2.0 * x * Jx;
            let d3I_xy_i = Jx * y + 3.0 * Fx * vy + 3.0 * vx * Fy + x * Jy;
            let d3I_yy_i = 6.0 * vy * Fy + 2.0 * y * Jy;
            d3I_xx = d3I_xx_i;
            d3I_xy = d3I_xy_i;
            d3I_yy = d3I_yy_i;
            contribI = d3I_xx_i * d3I_xx_i + 2.0 * d3I_xy_i * d3I_xy_i + d3I_yy_i * d3I_yy_i;
        }

        // Charge quadrupole d³Q_ij/dt³
        if (emQuad) {
            let qm = particles[i].charge * derived[i].invMass;
            let d3Q_xx_i = qm * (6.0 * vx * Fx + 2.0 * x * Jx);
            let d3Q_xy_i = qm * (Jx * y + 3.0 * Fx * vy + 3.0 * vx * Fy + x * Jy);
            let d3Q_yy_i = qm * (6.0 * vy * Fy + 2.0 * y * Jy);
            d3Q_xx = d3Q_xx_i;
            d3Q_xy = d3Q_xy_i;
            d3Q_yy = d3Q_yy_i;
            contribQ = d3Q_xx_i * d3Q_xx_i + 2.0 * d3Q_xy_i * d3Q_xy_i + d3Q_yy_i * d3Q_yy_i;
        }

        // Store per-particle contribution norms in scratch fields
        // AND shift residual force history (AFTER jerk computation used old values)
        var rsW = radState[i];
        rsW.d3IContrib = contribI;
        rsW.d3QContrib = contribQ;
        // Shift history: t-1 → t-2, current → t-1 (matches CPU order in integrator.js:1241-1243)
        let af2 = allForces[i];
        let curResFx = af2.totalForce.x - af2.f0.x - af2.f0.z - af2.f3.z;
        let curResFy = af2.totalForce.y - af2.f0.y - af2.f0.w - af2.f3.w;
        rsW.qResFx0 = rsW.qResFx1;
        rsW.qResFy0 = rsW.qResFy1;
        rsW.qResFx1 = curResFx;
        rsW.qResFy1 = curResFy;
        if (rsW.qResCount < 2.0) {
            rsW.qResCount += 1.0;
        }
        radState[i] = rsW;
    }

    // Load into shared memory for reduction
    sh_d3Ixx[localId] = d3I_xx;
    sh_d3Ixy[localId] = d3I_xy;
    sh_d3Iyy[localId] = d3I_yy;
    sh_d3Qxx[localId] = d3Q_xx;
    sh_d3Qxy[localId] = d3Q_xy;
    sh_d3Qyy[localId] = d3Q_yy;
    sh_totalD3I[localId] = contribI;
    sh_totalD3Q[localId] = contribQ;

    // Workgroup reduction
    workgroupReduce8(localId);

    // Lane 0 writes partial sums
    if (localId == 0u) {
        let base = MAX_QUAD_WG * 4u + wgid.x * 8u;
        reductionBuf[base + 0u] = sh_d3Ixx[0];
        reductionBuf[base + 1u] = sh_d3Ixy[0];
        reductionBuf[base + 2u] = sh_d3Iyy[0];
        reductionBuf[base + 3u] = sh_d3Qxx[0];
        reductionBuf[base + 4u] = sh_d3Qxy[0];
        reductionBuf[base + 5u] = sh_d3Qyy[0];
        reductionBuf[base + 6u] = sh_totalD3I[0];
        reductionBuf[base + 7u] = sh_totalD3Q[0];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Entry Point 3: quadrupoleApply
// Finalize global power, apply tangential drag, accumulate energy, emit photons.
// ═══════════════════════════════════════════════════════════════════
@compute @workgroup_size(64)
fn quadrupoleApply(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((particles[i].flags & FLAG_ALIVE) == 0u) { return; }

    let numWG = (u.aliveCount + 63u) / 64u;
    let gravOn = (u.toggles0 & GRAVITY_BIT) != 0u;
    let coulombOn = (u.toggles0 & COULOMB_BIT) != 0u;
    let gwQuad = gravOn;
    let emQuad = coulombOn;

    // Finalize CoM partial sums (need totalKE)
    var totalKE: f32 = 0.0;
    for (var wg: u32 = 0u; wg < numWG; wg++) {
        totalKE += reductionBuf[wg * 4u + 3u];
    }

    // Finalize d³ partial sums
    var d3Ixx: f32 = 0.0; var d3Ixy: f32 = 0.0; var d3Iyy: f32 = 0.0;
    var d3Qxx: f32 = 0.0; var d3Qxy: f32 = 0.0; var d3Qyy: f32 = 0.0;
    var totalD3I: f32 = 0.0;
    var totalD3Q: f32 = 0.0;

    for (var wg: u32 = 0u; wg < numWG; wg++) {
        let base = MAX_QUAD_WG * 4u + wg * 8u;
        d3Ixx += reductionBuf[base + 0u];
        d3Ixy += reductionBuf[base + 1u];
        d3Iyy += reductionBuf[base + 2u];
        d3Qxx += reductionBuf[base + 3u];
        d3Qxy += reductionBuf[base + 4u];
        d3Qyy += reductionBuf[base + 5u];
        totalD3I += reductionBuf[base + 6u];
        totalD3Q += reductionBuf[base + 7u];
    }

    // GW power: trace-free reduced quadrupole
    // I^TF_ij = I_ij - (1/3)δ_ij·trace. For 2D motion in 3D (I_zz=0): trace = I_xx + I_yy
    let trI = d3Ixx + d3Iyy;
    let d3Ixx_tf = d3Ixx - trI / 3.0;
    let d3Iyy_tf = d3Iyy - trI / 3.0;
    let gwPower = select(0.0, 0.2 * (d3Ixx_tf * d3Ixx_tf + 2.0 * d3Ixy * d3Ixy + d3Iyy_tf * d3Iyy_tf), gwQuad);

    // EM power: NOT trace-free
    let emPower = select(0.0, (1.0 / 180.0) * (d3Qxx * d3Qxx + 2.0 * d3Qxy * d3Qxy + d3Qyy * d3Qyy), emQuad);

    let quadPower = gwPower + emPower;
    if (quadPower <= 0.0 || totalKE <= EPSILON_SQ) { return; }

    // Use PHYSICS_DT constant (matches CPU where quadrupole uses this._dt ≈ PHYSICS_DT)
    let dt = PHYSICS_DT;
    var dE = quadPower * dt;
    dE = min(dE, QUADRUPOLE_POWER_CLAMP * totalKE);

    // Split proportionally between GW and EM channels
    let gwFrac = gwPower / quadPower;
    let gwDE = dE * gwFrac;
    let emDE = dE - gwDE;

    // Tangential drag: scale all proper velocities by (1 - f)
    let f = min(0.5 * dE / totalKE, 1.0);
    let scale = 1.0 - f;
    let fOverDt = f / dt;

    let wx = particles[i].velWX;
    let wy = particles[i].velWY;

    // Update display force (drag direction).
    // Start from allForces.f3.xy (Larmor value, or 0 for uncharged) — NOT radState.radDisplayX
    // which would accumulate frame-over-frame for uncharged particles (never reset by Larmor).
    // Matches CPU: forceRadiation zeroed at start of update(), Larmor sets it, quadrupole subtracts.
    var rs = radState[i];
    let afBase = allForces[i];
    rs.radDisplayX = afBase.f3.x - particles[i].mass * wx * fOverDt;
    rs.radDisplayY = afBase.f3.y - particles[i].mass * wy * fOverDt;

    // Apply drag (NaN guard per project conventions)
    let newWx = wx * scale;
    let newWy = wy * scale;
    if (newWx == newWx && newWy == newWy) {
        particles[i].velWX = newWx;
        particles[i].velWY = newWy;
    }

    // Distribute energy proportional to per-particle contribution
    let invD3I = select(0.0, 1.0 / totalD3I, totalD3I > EPSILON_SQ);
    let invD3Q = select(0.0, 1.0 / totalD3Q, totalD3Q > EPSILON_SQ);

    if (invD3I > 0.0) { rs.quadAccum += gwDE * rs.d3IContrib * invD3I; }
    if (invD3Q > 0.0) { rs.emQuadAccum += emDE * rs.d3QContrib * invD3Q; }

    // ── Emit GW graviton ──
    if (rs.quadAccum >= MIN_MASS) {
        let phIdx = atomicAdd(&phCount, 1u);
        if (phIdx < MAX_PHOTONS) {
            // Quadrupole angular pattern: power ∝ (Axx·cos2φ + Axy·sin2φ)²
            let angle = quadSample(d3Ixx, d3Ixy, i, 0u);
            let cosA = cos(angle);
            let sinA = sin(angle);
            let offset = max(particleAux[i].radius * 1.5, 1.0);
            var ph: Photon;
            ph.posX = particles[i].posX + cosA * offset;
            ph.posY = particles[i].posY + sinA * offset;
            ph.velX = cosA;
            ph.velY = sinA;
            ph.energy = rs.quadAccum;
            ph.emitterId = particleAux[i].particleId;
            ph.lifetime = 0.0;
            ph.flags = 3u; // FLAG_ALIVE(1) | FLAG_GRAV(2)
            photons[phIdx] = ph;
            rs.quadAccum = 0.0;
        } else {
            atomicSub(&phCount, 1u);
        }
    }

    // ── Emit EM quadrupole photon ──
    if (rs.emQuadAccum >= MIN_MASS) {
        let phIdx = atomicAdd(&phCount, 1u);
        if (phIdx < MAX_PHOTONS) {
            let angle = quadSample(d3Qxx, d3Qxy, i, 1u);
            let cosA = cos(angle);
            let sinA = sin(angle);
            let offset = max(particleAux[i].radius * 1.5, 1.0);
            var ph: Photon;
            ph.posX = particles[i].posX + cosA * offset;
            ph.posY = particles[i].posY + sinA * offset;
            ph.velX = cosA;
            ph.velY = sinA;
            ph.energy = rs.emQuadAccum;
            ph.emitterId = particleAux[i].particleId;
            ph.lifetime = 0.0;
            ph.flags = 1u; // FLAG_ALIVE only (EM photon)
            photons[phIdx] = ph;
            rs.emQuadAccum = 0.0;
        } else {
            atomicSub(&phCount, 1u);
        }
    }

    radState[i] = rs;

    // Update allForces.f3.xy with final display force (Larmor + quadrupole combined)
    var afRad = allForces[i];
    afRad.f3.x = rs.radDisplayX;
    afRad.f3.y = rs.radDisplayY;
    allForces[i] = afRad;
}

// ── Quadrupole rejection sampling ──
// Power ∝ (Axx·cos2φ + Axy·sin2φ)² where peak² = Axx² + Axy².
fn quadSample(Axx: f32, Axy: f32, particleIdx: u32, channel: u32) -> f32 {
    let peak2 = Axx * Axx + Axy * Axy;
    if (peak2 < EPSILON_SQ) {
        return pcgRand((particleIdx * 2654435761u) ^ (u.frameCount * 1664525u) ^ (channel * 999u)) * TWO_PI;
    }
    var seedBase = (particleIdx * 2246822519u) ^ (u.frameCount * 2654435769u) ^ (channel * 12345u);
    for (var t: u32 = 0u; t < 8u; t++) {
        let phi = pcgRand(seedBase ^ (t * 1234567u)) * TWO_PI;
        let c2 = cos(2.0 * phi);
        let s2 = sin(2.0 * phi);
        let h = Axx * c2 + Axy * s2;
        if (pcgRand(seedBase ^ (t * 7654321u + 1u)) * peak2 <= h * h) {
            return phi;
        }
    }
    // Fallback: random angle
    return pcgRand(seedBase ^ 999999u) * TWO_PI;
}
