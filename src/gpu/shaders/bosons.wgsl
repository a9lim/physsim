// ─── Boson Update Shaders ───
// Photon/pion drift, gravitational lensing, absorption, pion decay.

const MAX_PHOTONS: u32 = 512u;
const MAX_PIONS: u32 = 256u;
const BOSON_SOFTENING_SQ: f32 = 4.0;
const BOSON_MIN_AGE: u32 = 4u;
const PHOTON_LIFETIME: f32 = 256.0;
const EPSILON: f32 = 1e-9;

// Pion decay probabilities (pre-computed from half-lives)
const PION_DECAY_PROB: f32 = 0.00054; // 1 - exp(-ln2/32 * 1/128)
const CHARGED_PION_DECAY_PROB: f32 = 0.0000423; // 1 - exp(-ln2/128 * 1/128)
const ELECTRON_MASS: f32 = 0.05;
const MAX_SPEED_RATIO: f32 = 0.99;
const MAX_PARTICLES: u32 = 4096u;

// Particle flag bits
const ALIVE_BIT: u32 = 1u;
const ANTIMATTER_BIT: u32 = 4u;

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
};

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read_write> aliveCountAtomic: atomic<u32>;

// Particle SoA (read_write: absorption writes velW, decay spawns electrons)
@group(1) @binding(0) var<storage, read_write> posX: array<f32>;
@group(1) @binding(1) var<storage, read_write> posY: array<f32>;
@group(1) @binding(2) var<storage, read_write> mass: array<f32>;
@group(1) @binding(3) var<storage, read> radius: array<f32>;
@group(1) @binding(4) var<storage, read_write> flags: array<u32>;
@group(1) @binding(5) var<storage, read> particleId: array<u32>;
@group(1) @binding(6) var<storage, read_write> velWX: array<f32>;
@group(1) @binding(7) var<storage, read_write> velWY: array<f32>;
@group(1) @binding(8) var<storage, read_write> charge_buf: array<f32>;
@group(1) @binding(9) var<storage, read_write> baseMass: array<f32>;
@group(1) @binding(10) var<storage, read_write> angW_buf: array<f32>;

// Photon pool (SoA)
@group(2) @binding(0) var<storage, read_write> phPosX: array<f32>;
@group(2) @binding(1) var<storage, read_write> phPosY: array<f32>;
@group(2) @binding(2) var<storage, read_write> phVelX: array<f32>;
@group(2) @binding(3) var<storage, read_write> phVelY: array<f32>;
@group(2) @binding(4) var<storage, read_write> phEnergy: array<f32>;
@group(2) @binding(5) var<storage, read_write> phEmitterId: array<u32>;
@group(2) @binding(6) var<storage, read_write> phAge: array<u32>;
@group(2) @binding(7) var<storage, read_write> phFlags: array<u32>;
@group(2) @binding(8) var<storage, read_write> phCount: atomic<u32>;

// Pion pool (SoA)
@group(3) @binding(0) var<storage, read_write> piPosX: array<f32>;
@group(3) @binding(1) var<storage, read_write> piPosY: array<f32>;
@group(3) @binding(2) var<storage, read_write> piWX: array<f32>;
@group(3) @binding(3) var<storage, read_write> piWY: array<f32>;
@group(3) @binding(4) var<storage, read_write> piMass: array<f32>;
@group(3) @binding(5) var<storage, read_write> piCharge: array<i32>;
@group(3) @binding(6) var<storage, read_write> piEnergy: array<f32>;
@group(3) @binding(7) var<storage, read_write> piEmitterId: array<u32>;
@group(3) @binding(8) var<storage, read_write> piAge: array<u32>;
@group(3) @binding(9) var<storage, read_write> piFlags: array<u32>;
@group(3) @binding(10) var<storage, read_write> piCount: atomic<u32>;

// Photon drift: move at c=1, apply gravitational lensing via pairwise fallback
@compute @workgroup_size(64)
fn updatePhotons(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= atomicLoad(&phCount)) { return; }
    if ((phFlags[i] & 1u) == 0u) { return; } // not alive

    let dt = u.dt;
    let aliveN = u.aliveCount;

    // Gravitational deflection: GR gives 2x Newtonian for null geodesic
    // Pairwise fallback (tree walk added when boson gravity is wired)
    for (var j = 0u; j < aliveN; j++) {
        if ((flags[j] & ALIVE_BIT) == 0u) { continue; }
        let dx = posX[j] - phPosX[i];
        let dy = posY[j] - phPosY[i];
        let rSq = dx * dx + dy * dy + BOSON_SOFTENING_SQ;
        let invR3 = 1.0 / (rSq * sqrt(rSq));
        phVelX[i] += 2.0 * mass[j] * dx * invR3 * dt;
        phVelY[i] += 2.0 * mass[j] * dy * invR3 * dt;
    }

    // Renormalize to c=1
    let vSq = phVelX[i] * phVelX[i] + phVelY[i] * phVelY[i];
    if (abs(vSq - 1.0) > 1e-6) {
        let v = sqrt(vSq);
        if (v > EPSILON) {
            phVelX[i] /= v;
            phVelY[i] /= v;
        }
    }

    // Drift
    phPosX[i] += phVelX[i] * dt;
    phPosY[i] += phVelY[i] * dt;

    // Age + lifetime despawn
    phAge[i] += 1u;
    if (f32(phAge[i]) * u.dt > PHOTON_LIFETIME) {
        phFlags[i] &= ~1u; // mark dead
    }
}

// Pion drift: massive particle with proper velocity w, v = w/sqrt(1+w^2)
@compute @workgroup_size(64)
fn updatePions(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= atomicLoad(&piCount)) { return; }
    if ((piFlags[i] & 1u) == 0u) { return; } // not alive

    let dt = u.dt;
    let aliveN = u.aliveCount;
    let wx = piWX[i]; let wy = piWY[i];
    let wSq = wx * wx + wy * wy;
    let gamma = sqrt(1.0 + wSq);
    let vSq = wSq / (gamma * gamma);

    // Gravitational deflection: (1+v^2) factor for massive particle
    let grFactor = 1.0 + vSq;
    for (var j = 0u; j < aliveN; j++) {
        if ((flags[j] & ALIVE_BIT) == 0u) { continue; }
        let dx = posX[j] - piPosX[i];
        let dy = posY[j] - piPosY[i];
        let rSq = dx * dx + dy * dy + BOSON_SOFTENING_SQ;
        let invR3 = 1.0 / (rSq * sqrt(rSq));
        piWX[i] += grFactor * mass[j] * dx * invR3 * dt;
        piWY[i] += grFactor * mass[j] * dy * invR3 * dt;
    }

    // Sync vel from w
    let wx2 = piWX[i]; let wy2 = piWY[i];
    let gamma2 = sqrt(1.0 + wx2 * wx2 + wy2 * wy2);
    let velX = wx2 / gamma2;
    let velY = wy2 / gamma2;

    // Drift
    piPosX[i] += velX * dt;
    piPosY[i] += velY * dt;
    piAge[i] += 1u;
}

// Photon absorption: transfer momentum to nearby particles
@compute @workgroup_size(64)
fn absorbPhotons(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let count = atomicLoad(&phCount);
    if (i >= count) { return; }
    if ((phFlags[i] & 1u) == 0u) { return; }
    if (phAge[i] < BOSON_MIN_AGE) { return; }

    let phX = phPosX[i]; let phY = phPosY[i];
    let aliveN = u.aliveCount;

    // Check all alive particles for overlap
    for (var j = 0u; j < aliveN; j++) {
        if ((flags[j] & ALIVE_BIT) == 0u) { continue; }
        if (particleId[j] == phEmitterId[i]) { continue; } // self-absorption blocked

        let dx = phX - posX[j];
        let dy = phY - posY[j];
        let distSq = dx * dx + dy * dy;
        let rSq = radius[j] * radius[j];
        if (distSq < rSq) {
            // Absorb: transfer momentum to proper velocity
            let impulse = phEnergy[i];
            let invTM = select(0.0, 1.0 / mass[j], mass[j] > EPSILON);
            velWX[j] += impulse * phVelX[i] * invTM;
            velWY[j] += impulse * phVelY[i] * invTM;
            phFlags[i] &= ~1u; // mark dead
            break;
        }
    }
}

// Pion absorption: transfer momentum + charge to nearby particles
@compute @workgroup_size(64)
fn absorbPions(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let count = atomicLoad(&piCount);
    if (i >= count) { return; }
    if ((piFlags[i] & 1u) == 0u) { return; }
    if (piAge[i] < BOSON_MIN_AGE) { return; }

    let piX = piPosX[i]; let piY = piPosY[i];
    let aliveN = u.aliveCount;

    for (var j = 0u; j < aliveN; j++) {
        if ((flags[j] & ALIVE_BIT) == 0u) { continue; }
        if (particleId[j] == piEmitterId[i]) { continue; }

        let dx = piX - posX[j];
        let dy = piY - posY[j];
        if (dx * dx + dy * dy < radius[j] * radius[j]) {
            // Transfer momentum
            let wx = piWX[i]; let wy = piWY[i];
            let gamma = sqrt(1.0 + wx * wx + wy * wy);
            let invG = 1.0 / gamma;
            let impulse = piEnergy[i];
            let invTM = select(0.0, 1.0 / mass[j], mass[j] > EPSILON);
            velWX[j] += impulse * (wx * invG) * invTM;
            velWY[j] += impulse * (wy * invG) * invTM;
            // Transfer charge
            charge_buf[j] += f32(piCharge[i]);
            piFlags[i] &= ~1u;
            break;
        }
    }
}

// Pion decay: pi0 -> 2 photons, pi+/- -> electron/positron + photon
@compute @workgroup_size(64)
fn decayPions(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let count = atomicLoad(&piCount);
    if (i >= count) { return; }
    if ((piFlags[i] & 1u) == 0u) { return; }

    // Decay probability depends on charge
    let isNeutral = piCharge[i] == 0;
    let prob = select(CHARGED_PION_DECAY_PROB, PION_DECAY_PROB, isNeutral);

    // Pseudo-random from pion index + age (deterministic per-frame)
    let rng = fract(sin(f32(i * 73856093u ^ piAge[i] * 19349663u)) * 43758.5453);
    if (rng > prob) { return; }

    let mPi = piMass[i];
    let wx = piWX[i]; let wy = piWY[i];
    let wSq = wx * wx + wy * wy;
    let gamma = sqrt(1.0 + wSq);
    let invG = 1.0 / gamma;
    let vx = wx * invG; let vy = wy * invG;
    let vSq = vx * vx + vy * vy;

    if (isNeutral) {
        // pi0 -> 2 photons: back-to-back in rest frame, Lorentz-boosted
        let restAngle = rng * 6.2831853; // 2*PI
        let cosR = cos(restAngle); let sinR = sin(restAngle);
        let eRest = mPi * 0.5;

        for (var s = 0; s < 2; s++) {
            let sign = select(-1.0, 1.0, s == 0);
            var pxR = sign * eRest * cosR;
            var pyR = sign * eRest * sinR;

            // Lorentz boost along pion velocity
            if (vSq > 1e-12) {
                let v = sqrt(vSq);
                let clampedVSq = min(vSq, MAX_SPEED_RATIO * MAX_SPEED_RATIO);
                let gammaB = 1.0 / sqrt(1.0 - clampedVSq);
                let nx = vx / v; let ny = vy / v;
                let pPar = pxR * nx + pyR * ny;
                let pPerpX = pxR - pPar * nx;
                let pPerpY = pyR - pPar * ny;
                let pParB = gammaB * (pPar + v * eRest);
                pxR = pParB * nx + pPerpX;
                pyR = pParB * ny + pPerpY;
            }

            let pMag = sqrt(pxR * pxR + pyR * pyR);
            if (pMag < EPSILON) { continue; }
            let cosA = pxR / pMag; let sinA = pyR / pMag;

            // Atomic append to photon pool
            let phIdx = atomicAdd(&phCount, 1u);
            if (phIdx < MAX_PHOTONS) {
                let offset = max(mPi * 1.5, 1.0);
                phPosX[phIdx] = piPosX[i] + cosA * offset;
                phPosY[phIdx] = piPosY[i] + sinA * offset;
                phVelX[phIdx] = cosA;
                phVelY[phIdx] = sinA;
                phEnergy[phIdx] = pMag;
                phEmitterId[phIdx] = piEmitterId[i]; // inherit emitterId
                phAge[phIdx] = 0u;
                phFlags[phIdx] = 1u; // alive, type=em
            } else {
                atomicSub(&phCount, 1u); // rollback
            }
        }
    } else {
        // pi+/- -> electron/positron + photon
        // Two-body kinematics in rest frame
        let mE = ELECTRON_MASS;
        if (mPi <= mE) {
            // Not enough rest energy — emit photon only
            let angle = atan2(vy, vx);
            let cosA = cos(angle); let sinA = sin(angle);
            let phIdx = atomicAdd(&phCount, 1u);
            if (phIdx < MAX_PHOTONS) {
                let offset = max(mPi * 1.5, 1.0);
                phPosX[phIdx] = piPosX[i] + cosA * offset;
                phPosY[phIdx] = piPosY[i] + sinA * offset;
                phVelX[phIdx] = cosA; phVelY[phIdx] = sinA;
                phEnergy[phIdx] = piEnergy[i];
                phEmitterId[phIdx] = piEmitterId[i];
                phAge[phIdx] = 0u; phFlags[phIdx] = 1u;
            } else { atomicSub(&phCount, 1u); }
        } else {
            // Rest-frame energies
            let ePhRest = (mPi * mPi - mE * mE) / (2.0 * mPi);
            let eElRest = mPi - ePhRest;
            let pRest = ePhRest;

            let restAngle2 = fract(rng * 7.3) * 6.2831853;
            let cosR = cos(restAngle2); let sinR = sin(restAngle2);

            // Photon momentum in rest frame
            var phPxR = pRest * cosR; var phPyR = pRest * sinR;
            // Electron momentum (opposite)
            var elPxR = -pRest * cosR; var elPyR = -pRest * sinR;
            var elELab = eElRest;

            // Lorentz boost
            if (vSq > 1e-12) {
                let v = sqrt(vSq);
                let clampedVSq = min(vSq, MAX_SPEED_RATIO * MAX_SPEED_RATIO);
                let gammaB = 1.0 / sqrt(1.0 - clampedVSq);
                let nx = vx / v; let ny = vy / v;
                // Boost photon
                let phPar = phPxR * nx + phPyR * ny;
                let phPerpX = phPxR - phPar * nx;
                let phPerpY = phPyR - phPar * ny;
                let phParB = gammaB * (phPar + v * ePhRest);
                phPxR = phParB * nx + phPerpX;
                phPyR = phParB * ny + phPerpY;
                // Boost electron
                let elPar = elPxR * nx + elPyR * ny;
                let elPerpX = elPxR - elPar * nx;
                let elPerpY = elPyR - elPar * ny;
                let elParB = gammaB * (elPar + v * eElRest);
                elPxR = elParB * nx + elPerpX;
                elPyR = elParB * ny + elPerpY;
                elELab = gammaB * (eElRest + v * elPar);
            }

            // Emit photon
            let phMag = sqrt(phPxR * phPxR + phPyR * phPyR);
            if (phMag > EPSILON) {
                let phCos = phPxR / phMag; let phSin = phPyR / phMag;
                let phIdx = atomicAdd(&phCount, 1u);
                if (phIdx < MAX_PHOTONS) {
                    let offset = max(mPi * 1.5, 1.0);
                    phPosX[phIdx] = piPosX[i] + phCos * offset;
                    phPosY[phIdx] = piPosY[i] + phSin * offset;
                    phVelX[phIdx] = phCos; phVelY[phIdx] = phSin;
                    phEnergy[phIdx] = phMag;
                    phEmitterId[phIdx] = piEmitterId[i];
                    phAge[phIdx] = 0u; phFlags[phIdx] = 1u;
                } else { atomicSub(&phCount, 1u); }
            }

            // Spawn electron/positron via atomic append to particle pool
            // pi+ -> positron (antimatter, charge=+1)
            // pi- -> electron (charge=-1)
            if (elELab > EPSILON) {
                let pIdx = atomicAdd(&aliveCountAtomic, 1u);
                if (pIdx < MAX_PARTICLES) {
                    let elVx = elPxR / elELab;
                    let elVy = elPyR / elELab;
                    let offset = max(mPi * 1.5, 1.0);
                    posX[pIdx] = piPosX[i] - (phPxR / max(phMag, EPSILON)) * offset;
                    posY[pIdx] = piPosY[i] - (phPyR / max(phMag, EPSILON)) * offset;
                    let elGamma = 1.0 / sqrt(max(1.0 - elVx * elVx - elVy * elVy, 0.01));
                    velWX[pIdx] = elVx * elGamma;
                    velWY[pIdx] = elVy * elGamma;
                    mass[pIdx] = mE;
                    baseMass[pIdx] = mE;
                    charge_buf[pIdx] = f32(piCharge[i]); // +1 or -1
                    angW_buf[pIdx] = 0.0;
                    flags[pIdx] = ALIVE_BIT; // alive
                    // Set antimatter flag for pi+ decay
                    if (piCharge[i] > 0) { flags[pIdx] |= ANTIMATTER_BIT; }
                } else { atomicSub(&aliveCountAtomic, 1u); }
            }
        }
    }

    // Mark pion as dead
    piFlags[i] &= ~1u;
}
