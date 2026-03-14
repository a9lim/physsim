// ─── Radiation Reaction Shaders ───
// Three entry points:
//   lamrorRadiation — Landau-Lifshitz Larmor radiation (requires Coulomb + Radiation)
//   hawkingRadiation — Kerr-Newman BH evaporation (requires Black Hole + Radiation)
//   pionEmission — Scalar Larmor pion emission (requires Yukawa + Radiation)
//
// All kernels accumulate energy into per-particle accumulators and emit
// photons/pions via atomic append to boson pool buffers.

const ALIVE_BIT: u32 = 1u;
const LL_FORCE_CLAMP: f32 = 0.5;
const MIN_MASS: f32 = 0.05;
const EPSILON: f32 = 1e-9;
const MAX_PHOTONS: u32 = 512u;
const MAX_PIONS: u32 = 256u;
const MAX_SPEED_RATIO: f32 = 0.9999;
const INERTIA_K: f32 = 0.4;

// Toggle bit constants
const COULOMB_BIT: u32    = 2u;
const RELATIVITY_BIT: u32 = 32u;
const RADIATION_BIT: u32  = 128u;
const BLACK_HOLE_BIT: u32 = 256u;
const YUKAWA_BIT: u32     = 2048u;

struct Uniforms {
    dt: f32,
    simTime: f32,
    domainW: f32,
    domainH: f32,
    softeningSq: f32,
    yukawaCoupling: f32,
    yukawaMu: f32,
    toggles0: u32,
    aliveCount: u32,
    frameCount: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Particle state (read)
@group(1) @binding(0) var<storage, read> posX: array<f32>;
@group(1) @binding(1) var<storage, read> posY: array<f32>;
@group(1) @binding(2) var<storage, read_write> velWX: array<f32>;
@group(1) @binding(3) var<storage, read_write> velWY: array<f32>;
@group(1) @binding(4) var<storage, read_write> mass: array<f32>;
@group(1) @binding(5) var<storage, read> charge_buf: array<f32>;
@group(1) @binding(6) var<storage, read> flags: array<u32>;
@group(1) @binding(7) var<storage, read_write> invMassRadSq: array<vec2<f32>>; // packed: invMass, radiusSq
@group(1) @binding(8) var<storage, read_write> baseMass: array<f32>;
@group(1) @binding(9) var<storage, read> radius: array<f32>;
@group(1) @binding(10) var<storage, read> angW_buf: array<f32>;
@group(1) @binding(11) var<storage, read> particleId: array<u32>;

// Force/jerk inputs (read)
@group(1) @binding(12) var<storage, read> force_total: array<vec2<f32>>;   // packed totalForce
@group(1) @binding(13) var<storage, read> jerk_buf: array<f32>;  // interleaved [x0, y0, x1, y1, ...]
@group(1) @binding(14) var<storage, read> yukForceX: array<f32>;
@group(1) @binding(15) var<storage, read> yukForceY: array<f32>;

// Radiation accumulators (read_write)
@group(2) @binding(0) var<storage, read_write> radAccum: array<f32>;
@group(2) @binding(1) var<storage, read_write> hawkAccum: array<f32>;
@group(2) @binding(2) var<storage, read_write> yukawaRadAccum: array<f32>;

// Display force output (write)
@group(2) @binding(3) var<storage, read_write> radDisplayX: array<f32>;
@group(2) @binding(4) var<storage, read_write> radDisplayY: array<f32>;

// Photon pool (write via atomic append)
@group(3) @binding(0) var<storage, read_write> phPosX: array<f32>;
@group(3) @binding(1) var<storage, read_write> phPosY: array<f32>;
@group(3) @binding(2) var<storage, read_write> phVelX: array<f32>;
@group(3) @binding(3) var<storage, read_write> phVelY: array<f32>;
@group(3) @binding(4) var<storage, read_write> phEnergy: array<f32>;
@group(3) @binding(5) var<storage, read_write> phEmitterId: array<u32>;
@group(3) @binding(6) var<storage, read_write> phAge: array<u32>;
@group(3) @binding(7) var<storage, read_write> phFlags: array<u32>;
@group(3) @binding(8) var<storage, read_write> phCount: atomic<u32>;

// Pion pool (write via atomic append)
@group(3) @binding(9) var<storage, read_write> piPosX: array<f32>;
@group(3) @binding(10) var<storage, read_write> piPosY: array<f32>;
@group(3) @binding(11) var<storage, read_write> piWX: array<f32>;
@group(3) @binding(12) var<storage, read_write> piWY: array<f32>;
@group(3) @binding(13) var<storage, read_write> piMass: array<f32>;
@group(3) @binding(14) var<storage, read_write> piCharge_buf: array<i32>;
@group(3) @binding(15) var<storage, read_write> piEnergy: array<f32>;
@group(3) @binding(16) var<storage, read_write> piEmitterId: array<u32>;
@group(3) @binding(17) var<storage, read_write> piAge: array<u32>;
@group(3) @binding(18) var<storage, read_write> piFlags: array<u32>;
@group(3) @binding(19) var<storage, read_write> piCount: atomic<u32>;

// Charge buffer (read_write for pion charge transfer)
@group(3) @binding(20) var<storage, read_write> charge_rw: array<f32>;

// ─── Landau-Lifshitz Larmor Radiation ───
// Ports integrator.js Larmor radiation. Requires Coulomb + Radiation.
@compute @workgroup_size(64)
fn lamrorRadiation(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((flags[i] & ALIVE_BIT) == 0u) { return; }

    let coulombOn = (u.toggles0 & COULOMB_BIT) != 0u;
    let radiationOn = (u.toggles0 & RADIATION_BIT) != 0u;
    if (!coulombOn || !radiationOn) { return; }
    if (abs(charge_buf[i]) < EPSILON) { return; }

    let wx = velWX[i]; let wy = velWY[i];
    let wMagSq = wx * wx + wy * wy;
    if (wMagSq < EPSILON * EPSILON) { return; }

    let gamma = sqrt(1.0 + wMagSq);
    let qSq = charge_buf[i] * charge_buf[i];
    let mInv = invMassRadSq[i].x;
    let tau = 2.0 / 3.0 * qSq * mInv;

    // Term 1: analytical jerk (pre-accumulated in force pass)
    var jerkXVal = jerk_buf[i * 2u];
    var jerkYVal = jerk_buf[i * 2u + 1u];

    var fRadX = tau * jerkXVal;
    var fRadY = tau * jerkYVal;

    let relativityOn = (u.toggles0 & RELATIVITY_BIT) != 0u;
    if (relativityOn && gamma > 1.0) {
        let invG3 = 1.0 / (gamma * gamma * gamma);
        fRadX *= invG3;
        fRadY *= invG3;

        let invGamma = 1.0 / gamma;
        let vx = wx * invGamma; let vy = wy * invGamma;
        let ftv = force_total[i]; let fx = ftv.x; let fy = ftv.y;
        let fSq = fx * fx + fy * fy;
        let vDotF = vx * fx + vy * fy;

        // Terms 2+3: power-dissipation along v
        let t23 = -tau * gamma * (fSq - vDotF * vDotF) * mInv;
        fRadX += t23 * vx;
        fRadY += t23 * vy;
    }

    // LL force clamp: |F_rad| <= 0.5 * |F_ext|
    let fRadMag = sqrt(fRadX * fRadX + fRadY * fRadY);
    let ftv2 = force_total[i];
    let fExtMag = sqrt(ftv2.x * ftv2.x + ftv2.y * ftv2.y);
    let maxFRad = LL_FORCE_CLAMP * fExtMag;
    if (fRadMag > maxFRad && fRadMag > EPSILON * EPSILON) {
        let scale = maxFRad / fRadMag;
        fRadX *= scale;
        fRadY *= scale;
    }

    // Apply radiation reaction to proper velocity
    let dt = u.dt;
    let keBefore = wMagSq / (gamma + 1.0) * mass[i];
    velWX[i] += fRadX * dt * mInv;
    velWY[i] += fRadY * dt * mInv;

    // Store display force
    radDisplayX[i] = fRadX;
    radDisplayY[i] = fRadY;

    // Compute energy lost
    let wx2 = velWX[i]; let wy2 = velWY[i];
    let wMagSqAfter = wx2 * wx2 + wy2 * wy2;
    let gammaAfter = sqrt(1.0 + wMagSqAfter);
    let keAfter = wMagSqAfter / (gammaAfter + 1.0) * mass[i];
    let dE = max(0.0, keBefore - keAfter);

    // Accumulate for photon emission
    radAccum[i] += dE;

    // Emit photon when threshold reached
    if (radAccum[i] >= MIN_MASS) {
        let phIdx = atomicAdd(&phCount, 1u);
        if (phIdx < MAX_PHOTONS) {
            // Emit along -acceleration direction (simplified dipole pattern)
            let ftv3 = force_total[i];
            let ax = ftv3.x * mInv;
            let ay = ftv3.y * mInv;
            let aMag = sqrt(ax * ax + ay * ay);
            var cosA: f32; var sinA: f32;
            if (aMag > EPSILON) {
                cosA = -ax / aMag; sinA = -ay / aMag;
            } else {
                cosA = 1.0; sinA = 0.0;
            }
            let offset = max(radius[i] * 1.5, 1.0);
            phPosX[phIdx] = posX[i] + cosA * offset;
            phPosY[phIdx] = posY[i] + sinA * offset;
            phVelX[phIdx] = cosA; phVelY[phIdx] = sinA;
            phEnergy[phIdx] = radAccum[i];
            phEmitterId[phIdx] = particleId[i];
            phAge[phIdx] = 0u; phFlags[phIdx] = 1u;
            radAccum[i] = 0.0;
        } else {
            atomicSub(&phCount, 1u);
        }
    }
}

// ─── Hawking Radiation ───
// Kerr-Newman BH evaporation. Requires Black Hole + Radiation.
@compute @workgroup_size(64)
fn hawkingRadiation(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((flags[i] & ALIVE_BIT) == 0u) { return; }

    let blackHoleOn = (u.toggles0 & BLACK_HOLE_BIT) != 0u;
    let radiationOn = (u.toggles0 & RADIATION_BIT) != 0u;
    if (!blackHoleOn || !radiationOn) { return; }
    if (mass[i] <= MIN_MASS) { return; }

    let M = mass[i];
    let bodyRSq = pow(M, 2.0 / 3.0); // cbrt(M)^2
    let angvel = angW_buf[i] / sqrt(1.0 + angW_buf[i] * angW_buf[i] * bodyRSq);
    let a = INERTIA_K * bodyRSq * abs(angvel); // I_K * r² * |ω|
    let Q = charge_buf[i];
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
    let dE = min(power * dt, mass[i]);
    if (dE <= 0.0) { return; }

    mass[i] -= dE;
    var imrs = invMassRadSq[i];
    imrs.x = 1.0 / mass[i];
    invMassRadSq[i] = imrs;
    baseMass[i] *= 1.0 - dE / (mass[i] + dE);
    hawkAccum[i] += dE;

    if (hawkAccum[i] >= MIN_MASS) {
        let phIdx = atomicAdd(&phCount, 1u);
        if (phIdx < MAX_PHOTONS) {
            // Isotropic emission with pseudo-random angle
            let angle = fract(sin(f32(i * 12345u ^ u.frameCount)) * 43758.5453) * 6.2831853;
            let cosA = cos(angle); let sinA = sin(angle);
            let offset = max(radius[i] * 1.5, 1.0);
            phPosX[phIdx] = posX[i] + cosA * offset;
            phPosY[phIdx] = posY[i] + sinA * offset;
            phVelX[phIdx] = cosA; phVelY[phIdx] = sinA;
            phEnergy[phIdx] = hawkAccum[i];
            phEmitterId[phIdx] = particleId[i];
            phAge[phIdx] = 0u; phFlags[phIdx] = 1u;
            hawkAccum[i] = 0.0;
        } else { atomicSub(&phCount, 1u); }
    }
}

// ─── Pion Emission (Scalar Larmor) ───
// P = g² * F_yuk² / 3. Requires Yukawa + Radiation.
@compute @workgroup_size(64)
fn pionEmission(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.aliveCount) { return; }
    if ((flags[i] & ALIVE_BIT) == 0u) { return; }

    let yukawaOn = (u.toggles0 & YUKAWA_BIT) != 0u;
    let radiationOn = (u.toggles0 & RADIATION_BIT) != 0u;
    if (!yukawaOn || !radiationOn) { return; }

    let fYukX = yukForceX[i]; let fYukY = yukForceY[i];
    let fYukSq = fYukX * fYukX + fYukY * fYukY;
    if (fYukSq < EPSILON * EPSILON) { return; }

    let dt = u.dt;
    let coupling = u.yukawaCoupling;
    var dE = coupling / 3.0 * fYukSq * dt;

    // LL clamp
    let fYukMag = sqrt(fYukSq);
    let wx = velWX[i]; let wy = velWY[i];
    let wSqCl = wx * wx + wy * wy;
    let vMag = sqrt(wSqCl / (1.0 + wSqCl));
    let maxDE = LL_FORCE_CLAMP * fYukMag * vMag * dt;
    dE = min(dE, maxDE);

    yukawaRadAccum[i] += dE;
    let pionMass = u.yukawaMu;

    if (yukawaRadAccum[i] >= pionMass + MIN_MASS) {
        let ke = yukawaRadAccum[i] - pionMass;
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
                let rng = fract(sin(f32(i * 98765u ^ u.frameCount * 4321u)) * 43758.5453);
                var piChg: i32 = 0;
                if (rng > 0.5) {
                    let rng2 = fract(rng * 7.3);
                    piChg = select(-1, 1, rng2 < 0.5);
                    // Transfer charge from emitter
                    charge_rw[i] -= f32(piChg);
                }

                let offset = max(radius[i] * 1.5, 1.0);
                piPosX[piIdx] = posX[i] + cos(angle) * offset;
                piPosY[piIdx] = posY[i] + sin(angle) * offset;
                piWX[piIdx] = piWx; piWY[piIdx] = piWy;
                piMass[piIdx] = pionMass;
                piCharge_buf[piIdx] = piChg;
                piEnergy[piIdx] = yukawaRadAccum[i];
                piEmitterId[piIdx] = particleId[i];
                piAge[piIdx] = 0u; piFlags[piIdx] = 1u;

                // Radiation reaction: rescale emitter w
                let wSq = wx * wx + wy * wy;
                if (wSq > EPSILON * EPSILON) {
                    let gam = sqrt(1.0 + wSq);
                    let pKE = (gam - 1.0) * mass[i];
                    if (pKE > yukawaRadAccum[i]) {
                        let keNew = pKE - yukawaRadAccum[i];
                        let gammaNew = 1.0 + keNew / mass[i];
                        let wSqNew = gammaNew * gammaNew - 1.0;
                        if (wSqNew > EPSILON * EPSILON) {
                            let scale = sqrt(wSqNew / wSq);
                            velWX[i] *= scale;
                            velWY[i] *= scale;
                        }
                    }
                }

                yukawaRadAccum[i] = 0.0;
            } else { atomicSub(&piCount, 1u); }
        }
    }
}
