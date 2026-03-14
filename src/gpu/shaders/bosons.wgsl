// ─── Boson Update Shaders ───
// Photon/pion drift, gravitational lensing, absorption, pion decay.
//
// Standalone shader — defines own structs (NOT prepended with common.wgsl).

const MAX_PHOTONS: u32 = 1024u;
const MAX_PIONS: u32 = 256u;
const BOSON_SOFTENING_SQ: f32 = 4.0;
const BOSON_MIN_AGE: u32 = 4u;
const PHOTON_LIFETIME: f32 = 256.0;
const EPSILON: f32 = 1e-9;

// Pion decay probabilities (pre-computed from half-lives)
// PION_DECAY_PROB    = 1 - exp(-ln2 / 32  * (1/128)) = 1 - exp(-0.000216) ≈ 0.0001695
// CHARGED_PION_DECAY_PROB = 1 - exp(-ln2 / 128 * (1/128)) = 1 - exp(-0.0000540) ≈ 0.0000424
const PION_DECAY_PROB: f32 = 0.0001695; // pi0 (PION_HALF_LIFE=32, PHYSICS_DT=1/128)
const CHARGED_PION_DECAY_PROB: f32 = 0.0000423; // pi+/- (CHARGED_PION_HALF_LIFE=128, PHYSICS_DT=1/128)
const ELECTRON_MASS: f32 = 0.05;
const MAX_SPEED_RATIO: f32 = 0.99;
const MAX_PARTICLES: u32 = 4096u;

// Particle flag bits
const ALIVE_BIT: u32 = 1u;
const ANTIMATTER_BIT: u32 = 4u;

// PCG hash RNG (high quality, replaces sin-based LCG)
fn pcgHash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn pcgRand(seed: u32) -> f32 {
    return f32(pcgHash(seed)) / 4294967296.0;
}

// ── Packed struct definitions ──

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

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read_write> aliveCountAtomic: atomic<u32>;

// Packed particle state (rw: absorption writes velW, decay spawns electrons)
@group(1) @binding(0) var<storage, read_write> particles: array<ParticleState>;
@group(1) @binding(1) var<storage, read_write> particleAux: array<ParticleAux>;

// Photon pool (packed)
@group(2) @binding(0) var<storage, read_write> photons: array<Photon>;
@group(2) @binding(1) var<storage, read_write> phCount: atomic<u32>;

// Pion pool (packed)
@group(3) @binding(0) var<storage, read_write> pions: array<Pion>;
@group(3) @binding(1) var<storage, read_write> piCount: atomic<u32>;

// Photon drift: move at c=1, apply gravitational lensing via pairwise fallback
@compute @workgroup_size(64)
fn updatePhotons(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= atomicLoad(&phCount)) { return; }
    if ((photons[i].flags & 1u) == 0u) { return; } // not alive

    let dt = u.dt;
    let aliveN = u.aliveCount;

    // Load photon state into locals for accumulation
    var phVX = photons[i].velX;
    var phVY = photons[i].velY;

    // Gravitational deflection: GR gives 2x Newtonian for null geodesic
    for (var j = 0u; j < aliveN; j++) {
        if ((particles[j].flags & ALIVE_BIT) == 0u) { continue; }
        let dx = particles[j].posX - photons[i].posX;
        let dy = particles[j].posY - photons[i].posY;
        let rSq = dx * dx + dy * dy + BOSON_SOFTENING_SQ;
        let invR3 = 1.0 / (rSq * sqrt(rSq));
        phVX += 2.0 * particles[j].mass * dx * invR3 * dt;
        phVY += 2.0 * particles[j].mass * dy * invR3 * dt;
    }

    // Renormalize to c=1
    let vSq = phVX * phVX + phVY * phVY;
    if (abs(vSq - 1.0) > 1e-6) {
        let v = sqrt(vSq);
        if (v > EPSILON) {
            phVX /= v;
            phVY /= v;
        }
    }

    // Write back velocity + drift
    var ph = photons[i];
    ph.velX = phVX; ph.velY = phVY;
    ph.posX += phVX * dt;
    ph.posY += phVY * dt;

    // Age + lifetime despawn
    ph.age += 1u;
    if (f32(ph.age) * u.dt > PHOTON_LIFETIME) {
        ph.flags &= ~1u; // mark dead
    }
    photons[i] = ph;
}

// Pion drift: massive particle with proper velocity w, v = w/sqrt(1+w^2)
@compute @workgroup_size(64)
fn updatePions(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= atomicLoad(&piCount)) { return; }
    if ((pions[i].flags & 1u) == 0u) { return; } // not alive

    let dt = u.dt;
    let aliveN = u.aliveCount;
    var piWX = pions[i].wX; var piWY = pions[i].wY;
    let wSq = piWX * piWX + piWY * piWY;
    let gamma = sqrt(1.0 + wSq);
    let vSq = wSq / (gamma * gamma);

    // Gravitational deflection: (1+v^2) factor for massive particle
    let grFactor = 1.0 + vSq;
    for (var j = 0u; j < aliveN; j++) {
        if ((particles[j].flags & ALIVE_BIT) == 0u) { continue; }
        let dx = particles[j].posX - pions[i].posX;
        let dy = particles[j].posY - pions[i].posY;
        let rSq = dx * dx + dy * dy + BOSON_SOFTENING_SQ;
        let invR3 = 1.0 / (rSq * sqrt(rSq));
        piWX += grFactor * particles[j].mass * dx * invR3 * dt;
        piWY += grFactor * particles[j].mass * dy * invR3 * dt;
    }

    // Sync vel from w
    let gamma2 = sqrt(1.0 + piWX * piWX + piWY * piWY);
    let velX = piWX / gamma2;
    let velY = piWY / gamma2;

    // Write back
    var pi = pions[i];
    pi.wX = piWX; pi.wY = piWY;
    pi.posX += velX * dt;
    pi.posY += velY * dt;
    pi.age += 1u;
    pions[i] = pi;
}

// Photon absorption: transfer momentum to nearby particles
@compute @workgroup_size(64)
fn absorbPhotons(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let count = atomicLoad(&phCount);
    if (i >= count) { return; }
    if ((photons[i].flags & 1u) == 0u) { return; }
    if (photons[i].age < BOSON_MIN_AGE) { return; }

    let phX = photons[i].posX; let phY = photons[i].posY;
    let aliveN = u.aliveCount;

    // Check all alive particles for overlap
    for (var j = 0u; j < aliveN; j++) {
        if ((particles[j].flags & ALIVE_BIT) == 0u) { continue; }
        if (particleAux[j].particleId == photons[i].emitterId) { continue; } // self-absorption blocked

        let dx = phX - particles[j].posX;
        let dy = phY - particles[j].posY;
        let distSq = dx * dx + dy * dy;
        let rSq = particleAux[j].radius * particleAux[j].radius;
        if (distSq < rSq) {
            // Absorb: transfer momentum to proper velocity
            let impulse = photons[i].energy;
            let invTM = select(0.0, 1.0 / particles[j].mass, particles[j].mass > EPSILON);
            particles[j].velWX += impulse * photons[i].velX * invTM;
            particles[j].velWY += impulse * photons[i].velY * invTM;
            photons[i].flags &= ~1u; // mark dead
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
    if ((pions[i].flags & 1u) == 0u) { return; }
    if (pions[i].age < BOSON_MIN_AGE) { return; }

    let piX = pions[i].posX; let piY = pions[i].posY;
    let aliveN = u.aliveCount;

    for (var j = 0u; j < aliveN; j++) {
        if ((particles[j].flags & ALIVE_BIT) == 0u) { continue; }
        if (particleAux[j].particleId == pions[i].emitterId) { continue; }

        let dx = piX - particles[j].posX;
        let dy = piY - particles[j].posY;
        if (dx * dx + dy * dy < particleAux[j].radius * particleAux[j].radius) {
            // Transfer momentum
            let wx = pions[i].wX; let wy = pions[i].wY;
            let gamma = sqrt(1.0 + wx * wx + wy * wy);
            let invG = 1.0 / gamma;
            let impulse = pions[i].energy;
            let invTM = select(0.0, 1.0 / particles[j].mass, particles[j].mass > EPSILON);
            particles[j].velWX += impulse * (wx * invG) * invTM;
            particles[j].velWY += impulse * (wy * invG) * invTM;
            // Transfer charge
            particles[j].charge += f32(pions[i].charge);
            pions[i].flags &= ~1u;
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
    if ((pions[i].flags & 1u) == 0u) { return; }

    // Decay probability depends on charge
    let isNeutral = pions[i].charge == 0;
    let prob = select(CHARGED_PION_DECAY_PROB, PION_DECAY_PROB, isNeutral);

    // Pseudo-random from pion index + age (deterministic per-frame)
    let rng = pcgRand((i * 73856093u) ^ (pions[i].age * 19349663u));
    if (rng > prob) { return; }

    let mPi = pions[i].mass;
    let wx = pions[i].wX; let wy = pions[i].wY;
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
                var ph: Photon;
                ph.posX = pions[i].posX + cosA * offset;
                ph.posY = pions[i].posY + sinA * offset;
                ph.velX = cosA; ph.velY = sinA;
                ph.energy = pMag;
                ph.emitterId = pions[i].emitterId; // inherit emitterId
                ph.age = 0u; ph.flags = 1u; // alive, type=em
                photons[phIdx] = ph;
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
                var ph: Photon;
                ph.posX = pions[i].posX + cosA * offset;
                ph.posY = pions[i].posY + sinA * offset;
                ph.velX = cosA; ph.velY = sinA;
                ph.energy = pions[i].energy;
                ph.emitterId = pions[i].emitterId;
                ph.age = 0u; ph.flags = 1u;
                photons[phIdx] = ph;
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
                    var ph: Photon;
                    ph.posX = pions[i].posX + phCos * offset;
                    ph.posY = pions[i].posY + phSin * offset;
                    ph.velX = phCos; ph.velY = phSin;
                    ph.energy = phMag;
                    ph.emitterId = pions[i].emitterId;
                    ph.age = 0u; ph.flags = 1u;
                    photons[phIdx] = ph;
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
                    let elGamma = 1.0 / sqrt(max(1.0 - elVx * elVx - elVy * elVy, 0.01));
                    var p: ParticleState;
                    p.posX = pions[i].posX - (phPxR / max(phMag, EPSILON)) * offset;
                    p.posY = pions[i].posY - (phPyR / max(phMag, EPSILON)) * offset;
                    p.velWX = elVx * elGamma;
                    p.velWY = elVy * elGamma;
                    p.mass = mE;
                    p.charge = f32(pions[i].charge); // +1 or -1
                    p.angW = 0.0;
                    p.baseMass = mE;
                    p.flags = ALIVE_BIT; // alive
                    // Set antimatter flag for pi+ decay
                    if (pions[i].charge > 0) { p.flags |= ANTIMATTER_BIT; }
                    particles[pIdx] = p;

                    // Initialize particleAux: radius=cbrt(mE), deathTime=+Inf, deathMass=0, deathAngVel=0
                    // particleId=0 is acceptable (pion decay products are anonymous / not tracked for self-absorption)
                    var aux: ParticleAux;
                    aux.radius = pow(mE, 1.0 / 3.0);
                    aux.particleId = 0xFFFFFFFFu; // sentinel: no emitter identity
                    aux.deathTime = bitcast<f32>(0x7F800000u); // +Infinity
                    aux.deathMass = 0.0;
                    aux.deathAngVel = 0.0;
                    particleAux[pIdx] = aux;
                } else { atomicSub(&aliveCountAtomic, 1u); }
            }
        }
    }

    // Mark pion as dead
    pions[i].flags &= ~1u;
}
