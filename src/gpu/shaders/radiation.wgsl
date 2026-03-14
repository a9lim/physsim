// ─── Radiation Reaction Shaders ───
// Three entry points:
//   lamrorRadiation — Landau-Lifshitz Larmor radiation (requires Coulomb + Radiation)
//   hawkingRadiation — Kerr-Newman BH evaporation (requires Black Hole + Radiation)
//   pionEmission — Scalar Larmor pion emission (requires Yukawa + Radiation)
//
// All kernels accumulate energy into per-particle accumulators and emit
// photons/pions via atomic append to boson pool buffers.
//
// Standalone shader — defines own structs (NOT prepended with common.wgsl).

const ALIVE_BIT: u32 = 1u;
const LL_FORCE_CLAMP: f32 = 0.5;
const MIN_MASS: f32 = 0.05;
const EPSILON: f32 = 1e-9;
const MAX_PHOTONS: u32 = 512u;
const MAX_PIONS: u32 = 256u;
const MAX_SPEED_RATIO: f32 = 0.9999;
const INERTIA_K: f32 = 0.4;

// PCG hash RNG (high quality, replaces sin-based LCG)
fn pcgHash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn pcgRand(seed: u32) -> f32 {
    return f32(pcgHash(seed)) / 4294967296.0;
}

// Toggle bit constants
const COULOMB_BIT: u32    = 2u;
const RELATIVITY_BIT: u32 = 32u;
const RADIATION_BIT: u32  = 128u;
const BLACK_HOLE_BIT: u32 = 256u;
const YUKAWA_BIT: u32     = 2048u;

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

struct Photon {
    posX: f32, posY: f32,
    velX: f32, velY: f32,
    energy: f32,
    emitterId: u32, age: u32, flags: u32,
};

struct Pion {
    posX: f32, posY: f32,
    wX: f32, wY: f32,
    mass: f32, charge: i32, energy: f32,
    emitterId: u32, age: u32, flags: u32,
    _pad0: u32, _pad1: u32,
};

// Must match SimUniforms byte layout in common.wgsl / writeUniforms() exactly.
// Fields we don't use are kept as padding to preserve alignment.
struct Uniforms {
    dt: f32,                // [0] dt
    simTime: f32,           // [1] simTime
    domainW: f32,           // [2] domainW
    domainH: f32,           // [3] domainH
    _speedScale: f32,       // [4] speedScale (unused here)
    _softening: f32,        // [5] softening (unused here)
    _softeningSq: f32,      // [6] softeningSq (unused here)
    toggles0: u32,          // [7] toggles0
    _toggles1: u32,         // [8] toggles1 (unused here)
    yukawaCoupling: f32,    // [9] yukawaCoupling
    yukawaMu: f32,          // [10] yukawaMu
    _higgsMass: f32,        // [11] higgsMass (unused here)
    _axionMass: f32,        // [12] axionMass (unused here)
    _boundaryMode: u32,     // [13] boundaryMode (unused here)
    _topologyMode: u32,     // [14] topologyMode (unused here)
    _collisionMode: u32,    // [15] collisionMode (unused here)
    _maxParticles: u32,     // [16] maxParticles (unused here)
    aliveCount: u32,        // [17] aliveCount
    _extGravity: f32,       // [18]
    _extGravityAngle: f32,  // [19]
    _extElectric: f32,      // [20]
    _extElectricAngle: f32, // [21]
    _extBz: f32,            // [22]
    _bounceFriction: f32,   // [23]
    _extGx: f32,            // [24]
    _extGy: f32,            // [25]
    _extEx: f32,            // [26]
    _extEy: f32,            // [27]
    _axionCoupling: f32,    // [28]
    _higgsCoupling: f32,    // [29]
    _particleCount: u32,    // [30]
    _bhTheta: f32,          // [31]
    frameCount: u32,        // [32] _pad3 in common.wgsl, used as frameCount
    _pad4: u32,             // [33]
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Group 1: packed particle data
@group(1) @binding(0) var<storage, read_write> particles: array<ParticleState>;
@group(1) @binding(1) var<storage, read> particleAux: array<ParticleAux>;
@group(1) @binding(2) var<storage, read_write> derived: array<ParticleDerived>;
@group(1) @binding(3) var<storage, read> allForces: array<AllForces>;
@group(1) @binding(4) var<storage, read_write> radState: array<RadiationState>;
@group(1) @binding(5) var<storage, read> axYukMod: array<vec2<f32>>;

// Group 2: photon pool
@group(2) @binding(0) var<storage, read_write> photons: array<Photon>;
@group(2) @binding(1) var<storage, read_write> phCount: atomic<u32>;

// Group 3: pion pool
@group(3) @binding(0) var<storage, read_write> pions: array<Pion>;
@group(3) @binding(1) var<storage, read_write> piCount: atomic<u32>;

// ─── Landau-Lifshitz Larmor Radiation ───
// Ports integrator.js Larmor radiation. Requires Coulomb + Radiation.
@compute @workgroup_size(64)
fn lamrorRadiation(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((particles[i].flags & ALIVE_BIT) == 0u) { return; }

    let coulombOn = (u.toggles0 & COULOMB_BIT) != 0u;
    let radiationOn = (u.toggles0 & RADIATION_BIT) != 0u;
    if (!coulombOn || !radiationOn) { return; }
    if (abs(particles[i].charge) < EPSILON) { return; }

    let wx = particles[i].velWX; let wy = particles[i].velWY;
    let wMagSq = wx * wx + wy * wy;
    if (wMagSq < EPSILON * EPSILON) { return; }

    let gamma = sqrt(1.0 + wMagSq);
    let qSq = particles[i].charge * particles[i].charge;
    let mInv = derived[i].invMass;
    let tau = 2.0 / 3.0 * qSq * mInv;

    // Term 1: analytical jerk (pre-accumulated in force pass)
    let jerkXVal = radState[i].jerkX;
    let jerkYVal = radState[i].jerkY;

    var fRadX = tau * jerkXVal;
    var fRadY = tau * jerkYVal;

    let relativityOn = (u.toggles0 & RELATIVITY_BIT) != 0u;
    if (relativityOn && gamma > 1.0) {
        let invG3 = 1.0 / (gamma * gamma * gamma);
        fRadX *= invG3;
        fRadY *= invG3;

        let invGamma = 1.0 / gamma;
        let vx = wx * invGamma; let vy = wy * invGamma;
        let ftv = allForces[i].totalForce; let fx = ftv.x; let fy = ftv.y;
        let fSq = fx * fx + fy * fy;
        let vDotF = vx * fx + vy * fy;

        // Terms 2+3: power-dissipation along v
        let t23 = -tau * gamma * (fSq - vDotF * vDotF) * mInv;
        fRadX += t23 * vx;
        fRadY += t23 * vy;
    }

    // LL force clamp: |F_rad| <= 0.5 * |F_ext|
    let fRadMag = sqrt(fRadX * fRadX + fRadY * fRadY);
    let ftv2 = allForces[i].totalForce;
    let fExtMag = sqrt(ftv2.x * ftv2.x + ftv2.y * ftv2.y);
    let maxFRad = LL_FORCE_CLAMP * fExtMag;
    if (fRadMag > maxFRad && fRadMag > EPSILON * EPSILON) {
        let scale = maxFRad / fRadMag;
        fRadX *= scale;
        fRadY *= scale;
    }

    // Apply radiation reaction to proper velocity
    let dt = u.dt;
    let keBefore = wMagSq / (gamma + 1.0) * particles[i].mass;
    particles[i].velWX += fRadX * dt * mInv;
    particles[i].velWY += fRadY * dt * mInv;

    // Store display force
    var rs = radState[i];
    rs.radDisplayX = fRadX;
    rs.radDisplayY = fRadY;

    // Compute energy lost
    let wx2 = particles[i].velWX; let wy2 = particles[i].velWY;
    let wMagSqAfter = wx2 * wx2 + wy2 * wy2;
    let gammaAfter = sqrt(1.0 + wMagSqAfter);
    let keAfter = wMagSqAfter / (gammaAfter + 1.0) * particles[i].mass;
    let dE = max(0.0, keBefore - keAfter);

    // Accumulate for photon emission
    rs.radAccum += dE;

    // Emit photon when threshold reached
    if (rs.radAccum >= MIN_MASS) {
        let phIdx = atomicAdd(&phCount, 1u);
        if (phIdx < MAX_PHOTONS) {
            // Emit along -acceleration direction (simplified dipole pattern)
            let ftv3 = allForces[i].totalForce;
            let ax = ftv3.x * mInv;
            let ay = ftv3.y * mInv;
            let aMag = sqrt(ax * ax + ay * ay);
            var cosA: f32; var sinA: f32;
            if (aMag > EPSILON) {
                cosA = -ax / aMag; sinA = -ay / aMag;
            } else {
                cosA = 1.0; sinA = 0.0;
            }
            let offset = max(particleAux[i].radius * 1.5, 1.0);
            var ph: Photon;
            ph.posX = particles[i].posX + cosA * offset;
            ph.posY = particles[i].posY + sinA * offset;
            ph.velX = cosA; ph.velY = sinA;
            ph.energy = rs.radAccum;
            ph.emitterId = particleAux[i].particleId;
            ph.age = 0u; ph.flags = 1u;
            photons[phIdx] = ph;
            rs.radAccum = 0.0;
        } else {
            atomicSub(&phCount, 1u);
        }
    }
    radState[i] = rs;
}

// ─── Hawking Radiation ───
// Kerr-Newman BH evaporation. Requires Black Hole + Radiation.
@compute @workgroup_size(64)
fn hawkingRadiation(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((particles[i].flags & ALIVE_BIT) == 0u) { return; }

    let blackHoleOn = (u.toggles0 & BLACK_HOLE_BIT) != 0u;
    let radiationOn = (u.toggles0 & RADIATION_BIT) != 0u;
    if (!blackHoleOn || !radiationOn) { return; }
    if (particles[i].mass <= MIN_MASS) { return; }

    let M = particles[i].mass;
    let bodyRSq = pow(M, 2.0 / 3.0); // cbrt(M)^2
    let angw = particles[i].angW;
    let angvel = angw / sqrt(1.0 + angw * angw * bodyRSq);
    let a = INERTIA_K * bodyRSq * abs(angvel); // I_K * r² * |ω|
    let Q = particles[i].charge;
    let disc = M * M - a * a - Q * Q;

    var power: f32 = 0.0;
    if (disc > EPSILON) {
        let rPlus = M + sqrt(disc);
        let kappa = sqrt(disc) / (2.0 * M * rPlus);
        let T = kappa / 6.2831853; // 2*PI
        let A = 4.0 * 3.14159265 * (rPlus * rPlus + a * a);
        let sigma = 3.14159265 * 3.14159265 / 60.0;
        power = sigma * T * T * T * T * A;
    }
    // else extremal: no radiation

    let dt = u.dt;
    let dE = min(power * dt, particles[i].mass);
    if (dE <= 0.0) { return; }

    particles[i].mass -= dE;
    var drd = derived[i];
    drd.invMass = 1.0 / particles[i].mass;
    derived[i] = drd;
    particles[i].baseMass *= 1.0 - dE / (particles[i].mass + dE);

    var rs = radState[i];
    rs.hawkAccum += dE;

    if (rs.hawkAccum >= MIN_MASS) {
        let phIdx = atomicAdd(&phCount, 1u);
        if (phIdx < MAX_PHOTONS) {
            // Isotropic emission with pseudo-random angle
            let angle = pcgRand((i * 12345u) ^ u.frameCount) * 6.2831853;
            let cosA = cos(angle); let sinA = sin(angle);
            let offset = max(particleAux[i].radius * 1.5, 1.0);
            var ph: Photon;
            ph.posX = particles[i].posX + cosA * offset;
            ph.posY = particles[i].posY + sinA * offset;
            ph.velX = cosA; ph.velY = sinA;
            ph.energy = rs.hawkAccum;
            ph.emitterId = particleAux[i].particleId;
            ph.age = 0u; ph.flags = 1u;
            photons[phIdx] = ph;
            rs.hawkAccum = 0.0;
        } else { atomicSub(&phCount, 1u); }
    }
    radState[i] = rs;
}

// ─── Pion Emission (Scalar Larmor) ───
// P = g² * F_yuk² / 3. Requires Yukawa + Radiation.
@compute @workgroup_size(64)
fn pionEmission(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((particles[i].flags & ALIVE_BIT) == 0u) { return; }

    let yukawaOn = (u.toggles0 & YUKAWA_BIT) != 0u;
    let radiationOn = (u.toggles0 & RADIATION_BIT) != 0u;
    if (!yukawaOn || !radiationOn) { return; }

    // Read Yukawa force from packed allForces.f3.zw
    let fYukX = allForces[i].f3.z; let fYukY = allForces[i].f3.w;
    let fYukSq = fYukX * fYukX + fYukY * fYukY;
    if (fYukSq < EPSILON * EPSILON) { return; }

    let dt = u.dt;
    let coupling = u.yukawaCoupling;
    var dE = coupling / 3.0 * fYukSq * dt;

    // LL clamp
    let fYukMag = sqrt(fYukSq);
    let wx = particles[i].velWX; let wy = particles[i].velWY;
    let wSqCl = wx * wx + wy * wy;
    let vMag = sqrt(wSqCl / (1.0 + wSqCl));
    let maxDE = LL_FORCE_CLAMP * fYukMag * vMag * dt;
    dE = min(dE, maxDE);

    var rs = radState[i];
    rs.yukawaRadAccum += dE;
    let pionMass = u.yukawaMu;

    if (rs.yukawaRadAccum >= pionMass + MIN_MASS) {
        let ke = rs.yukawaRadAccum - pionMass;
        if (ke > 0.0) {
            let piIdx = atomicAdd(&piCount, 1u);
            if (piIdx < MAX_PIONS) {
                // Scalar dipole emission angle (along Yukawa force direction)
                let angle = atan2(fYukY, fYukX);
                let speed = min(sqrt(ke * (ke + 2.0 * pionMass)) / (ke + pionMass), MAX_SPEED_RATIO);
                let gammaPI = 1.0 / sqrt(1.0 - speed * speed);
                let piWx = gammaPI * speed * cos(angle);
                let piWy = gammaPI * speed * sin(angle);

                // Species: 50% pi0, 25% pi+, 25% pi-
                let rng = pcgRand((i * 98765u) ^ (u.frameCount * 4321u));
                var piChg: i32 = 0;
                if (rng > 0.5) {
                    let rng2 = pcgRand((i * 54321u) ^ (u.frameCount * 6789u));
                    piChg = select(-1, 1, rng2 < 0.5);
                    // Transfer charge from emitter (particleState is rw)
                    particles[i].charge -= f32(piChg);
                }

                let offset = max(particleAux[i].radius * 1.5, 1.0);
                var pi: Pion;
                pi.posX = particles[i].posX + cos(angle) * offset;
                pi.posY = particles[i].posY + sin(angle) * offset;
                pi.wX = piWx; pi.wY = piWy;
                pi.mass = pionMass;
                pi.charge = piChg;
                pi.energy = rs.yukawaRadAccum;
                pi.emitterId = particleAux[i].particleId;
                pi.age = 0u; pi.flags = 1u;
                pi._pad0 = 0u; pi._pad1 = 0u;
                pions[piIdx] = pi;

                // Radiation reaction: rescale emitter w
                let wSq = wx * wx + wy * wy;
                if (wSq > EPSILON * EPSILON) {
                    let gam = sqrt(1.0 + wSq);
                    let pKE = (gam - 1.0) * particles[i].mass;
                    if (pKE > rs.yukawaRadAccum) {
                        let keNew = pKE - rs.yukawaRadAccum;
                        let gammaNew = 1.0 + keNew / particles[i].mass;
                        let wSqNew = gammaNew * gammaNew - 1.0;
                        if (wSqNew > EPSILON * EPSILON) {
                            let sc = sqrt(wSqNew / wSq);
                            particles[i].velWX *= sc;
                            particles[i].velWY *= sc;
                        }
                    }
                }

                rs.yukawaRadAccum = 0.0;
            } else { atomicSub(&piCount, 1u); }
        }
    }
    radState[i] = rs;
}
