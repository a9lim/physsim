// ─── Boson Update Shaders ───
// Photon/pion drift, gravitational lensing, absorption, pion decay.
//
// Standalone shader — defines own structs (NOT prepended with common.wgsl).

// Constants provided by generated wgslConstants block.

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
    emitterId: u32, lifetime: f32, flags: u32,
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

    // Load photon state into local once (avoid redundant global memory reads)
    var ph = photons[i];
    if ((ph.flags & 1u) == 0u) { return; } // not alive

    let dt = u.dt;
    let aliveN = u.aliveCount;
    let phPosX = ph.posX;
    let phPosY = ph.posY;
    var phVX = ph.velX;
    var phVY = ph.velY;

    // Gravitational deflection: GR gives 2x Newtonian for null geodesic
    for (var j = 0u; j < aliveN; j++) {
        let pj = particles[j];
        if ((pj.flags & FLAG_ALIVE) == 0u) { continue; }
        let dx = pj.posX - phPosX;
        let dy = pj.posY - phPosY;
        let rSq = dx * dx + dy * dy + BOSON_SOFTENING_SQ;
        let invRSq = 1.0 / rSq;
        let invR3 = invRSq * sqrt(invRSq);
        phVX += 2.0 * pj.mass * dx * invR3 * dt;
        phVY += 2.0 * pj.mass * dy * invR3 * dt;
    }

    // Renormalize to c=1
    let vSq = phVX * phVX + phVY * phVY;
    if (vSq > EPSILON) {
        let invV = 1.0 / sqrt(vSq);
        phVX *= invV;
        phVY *= invV;
    }

    // NaN guard
    if (phVX != phVX || phVY != phVY) { phVX = 1.0; phVY = 0.0; }

    // Write back velocity + drift
    ph.velX = phVX; ph.velY = phVY;
    ph.posX += phVX * dt;
    ph.posY += phVY * dt;

    // Accumulate lifetime + despawn check
    ph.lifetime += dt;
    if (ph.lifetime > PHOTON_LIFETIME) {
        ph.flags &= ~1u; // mark dead
    }
    photons[i] = ph;
}

// Pion drift: massive particle with proper velocity w, v = w/sqrt(1+w^2)
@compute @workgroup_size(64)
fn updatePions(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= atomicLoad(&piCount)) { return; }

    // Load pion state once
    var pi = pions[i];
    if ((pi.flags & 1u) == 0u) { return; } // not alive

    let dt = u.dt;
    let aliveN = u.aliveCount;
    let piPosX = pi.posX;
    let piPosY = pi.posY;
    var piWX = pi.wX;
    var piWY = pi.wY;
    let wSq = piWX * piWX + piWY * piWY;
    let gamma = sqrt(1.0 + wSq);
    let vSq = wSq / max(gamma * gamma, EPSILON);

    // Gravitational deflection: (1+v^2) factor for massive particle
    let grFactor = 1.0 + vSq;
    for (var j = 0u; j < aliveN; j++) {
        let pj = particles[j];
        if ((pj.flags & FLAG_ALIVE) == 0u) { continue; }
        let dx = pj.posX - piPosX;
        let dy = pj.posY - piPosY;
        let rSq2 = dx * dx + dy * dy + BOSON_SOFTENING_SQ;
        let invRSq2 = 1.0 / rSq2;
        let invR3 = invRSq2 * sqrt(invRSq2);
        piWX += grFactor * pj.mass * dx * invR3 * dt;
        piWY += grFactor * pj.mass * dy * invR3 * dt;
    }

    // NaN guard
    if (piWX != piWX || piWY != piWY) { piWX = 0.0; piWY = 0.0; }

    // Sync vel from w
    let gamma2 = sqrt(1.0 + piWX * piWX + piWY * piWY);
    let invGamma2 = 1.0 / gamma2;
    let velX = piWX * invGamma2;
    let velY = piWY * invGamma2;

    // Write back
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

    // Load photon state once
    let ph = photons[i];
    if ((ph.flags & 1u) == 0u) { return; }
    if (ph.lifetime < BOSON_MIN_AGE_TIME) { return; }

    let phX = ph.posX;
    let phY = ph.posY;
    let phEmitterId = ph.emitterId;
    let phEnergy = ph.energy;
    let phVelX = ph.velX;
    let phVelY = ph.velY;
    let aliveN = u.aliveCount;

    // Check all alive particles for overlap
    for (var j = 0u; j < aliveN; j++) {
        let pj = particles[j];
        if ((pj.flags & FLAG_ALIVE) == 0u) { continue; }
        let auxJ = particleAux[j];
        if (auxJ.particleId == phEmitterId) { continue; } // self-absorption blocked

        let dx = phX - pj.posX;
        let dy = phY - pj.posY;
        let distSq = dx * dx + dy * dy;
        let rSq = auxJ.radius * auxJ.radius;
        if (distSq < rSq) {
            // Absorb: transfer momentum to proper velocity
            let invTM = select(0.0, 1.0 / pj.mass, pj.mass > EPSILON);
            particles[j].velWX += phEnergy * phVelX * invTM;
            particles[j].velWY += phEnergy * phVelY * invTM;
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

    // Load pion state once
    let pi = pions[i];
    if ((pi.flags & 1u) == 0u) { return; }
    if (pi.age < BOSON_MIN_AGE) { return; }

    let piX = pi.posX;
    let piY = pi.posY;
    let piEmitterId = pi.emitterId;
    let piWX = pi.wX;
    let piWY = pi.wY;
    let piEnergy = pi.energy;
    let piCharge = pi.charge;
    let aliveN = u.aliveCount;

    // Precompute pion velocity direction
    let gamma = sqrt(1.0 + piWX * piWX + piWY * piWY);
    let invG = 1.0 / gamma;

    for (var j = 0u; j < aliveN; j++) {
        let pj = particles[j];
        if ((pj.flags & FLAG_ALIVE) == 0u) { continue; }
        let auxJ = particleAux[j];
        if (auxJ.particleId == piEmitterId) { continue; }

        let dx = piX - pj.posX;
        let dy = piY - pj.posY;
        if (dx * dx + dy * dy < auxJ.radius * auxJ.radius) {
            // Transfer momentum
            let invTM = select(0.0, 1.0 / pj.mass, pj.mass > EPSILON);
            particles[j].velWX += piEnergy * (piWX * invG) * invTM;
            particles[j].velWY += piEnergy * (piWY * invG) * invTM;
            // Transfer charge
            particles[j].charge += f32(piCharge);
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

    // Load pion state once
    let piState = pions[i];
    if ((piState.flags & 1u) == 0u) { return; }

    // Decay probability depends on charge
    let isNeutral = piState.charge == 0;
    let prob = select(CHARGED_PION_DECAY_PROB, PION_DECAY_PROB, isNeutral);

    // Pseudo-random from pion index + age (deterministic per-frame)
    let rng = pcgRand((i * 73856093u) ^ (piState.age * 19349663u));
    if (rng > prob) { return; }

    let mPi = piState.mass;
    let wx = piState.wX; let wy = piState.wY;
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
        let piDecayPosX = piState.posX;
        let piDecayPosY = piState.posY;
        let piDecayEmitter = piState.emitterId;

        for (var s = 0; s < 2; s++) {
            let sign = select(-1.0, 1.0, s == 0);
            var pxR = sign * eRest * cosR;
            var pyR = sign * eRest * sinR;

            // Lorentz boost along pion velocity
            if (vSq > 1e-12) {
                let v = sqrt(vSq);
                let clampedVSq = min(vSq, MAX_SPEED_RATIO * MAX_SPEED_RATIO);
                let gammaB = 1.0 / sqrt(max(1.0 - clampedVSq, EPSILON));
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
            let invPMag = 1.0 / pMag;
            let cosA = pxR * invPMag; let sinA = pyR * invPMag;

            // Atomic append to photon pool
            let phIdx = atomicAdd(&phCount, 1u);
            if (phIdx < MAX_PHOTONS) {
                let emitOffset = max(mPi * 1.5, 1.0);
                var ph: Photon;
                ph.posX = piDecayPosX + cosA * emitOffset;
                ph.posY = piDecayPosY + sinA * emitOffset;
                ph.velX = cosA; ph.velY = sinA;
                ph.energy = pMag;
                ph.emitterId = piDecayEmitter;
                ph.lifetime = 0.0; ph.flags = 1u;
                photons[phIdx] = ph;
            } else {
                atomicSub(&phCount, 1u);
            }
        }
    } else {
        // pi+/- -> electron/positron + photon
        // Two-body kinematics in rest frame
        let mE = ELECTRON_MASS;
        let piDecayPosX2 = piState.posX;
        let piDecayPosY2 = piState.posY;
        let piDecayEmitter2 = piState.emitterId;
        let piDecayEnergy = piState.energy;
        let piDecayCharge = piState.charge;
        if (mPi <= mE) {
            // Not enough rest energy — emit photon only
            let angle = atan2(vy, vx);
            let cosA = cos(angle); let sinA = sin(angle);
            let phIdx = atomicAdd(&phCount, 1u);
            if (phIdx < MAX_PHOTONS) {
                let emitOffset = max(mPi * 1.5, 1.0);
                var ph: Photon;
                ph.posX = piDecayPosX2 + cosA * emitOffset;
                ph.posY = piDecayPosY2 + sinA * emitOffset;
                ph.velX = cosA; ph.velY = sinA;
                ph.energy = piDecayEnergy;
                ph.emitterId = piDecayEmitter2;
                ph.lifetime = 0.0; ph.flags = 1u;
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
                let gammaB = 1.0 / sqrt(max(1.0 - clampedVSq, EPSILON));
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
                let invPhMag = 1.0 / phMag;
                let phCos = phPxR * invPhMag; let phSin = phPyR * invPhMag;
                let phIdx = atomicAdd(&phCount, 1u);
                if (phIdx < MAX_PHOTONS) {
                    let emitOffset = max(mPi * 1.5, 1.0);
                    var ph: Photon;
                    ph.posX = piDecayPosX2 + phCos * emitOffset;
                    ph.posY = piDecayPosY2 + phSin * emitOffset;
                    ph.velX = phCos; ph.velY = phSin;
                    ph.energy = phMag;
                    ph.emitterId = piDecayEmitter2;
                    ph.lifetime = 0.0; ph.flags = 1u;
                    photons[phIdx] = ph;
                } else { atomicSub(&phCount, 1u); }
            }

            // Spawn electron/positron via atomic append to particle pool
            // pi+ -> positron (antimatter, charge=+1)
            // pi- -> electron (charge=-1)
            if (elELab > EPSILON) {
                let pIdx = atomicAdd(&aliveCountAtomic, 1u);
                if (pIdx < MAX_PARTICLES) {
                    let invElELab = 1.0 / max(elELab, EPSILON);
                    let elVx = elPxR * invElELab;
                    let elVy = elPyR * invElELab;
                    let emitOffset = max(mPi * 1.5, 1.0);
                    let elGamma = 1.0 / sqrt(max(1.0 - elVx * elVx - elVy * elVy, 0.01));
                    var p: ParticleState;
                    let phDir = select(vec2f(1.0, 0.0), vec2f(phPxR, phPyR) / max(phMag, EPSILON), phMag > EPSILON);
                    p.posX = piDecayPosX2 - phDir.x * emitOffset;
                    p.posY = piDecayPosY2 - phDir.y * emitOffset;
                    p.velWX = elVx * elGamma;
                    p.velWY = elVy * elGamma;
                    p.mass = mE;
                    p.charge = f32(piDecayCharge); // +1 or -1
                    p.angW = 0.0;
                    p.baseMass = mE;
                    p.flags = FLAG_ALIVE; // alive
                    // Set antimatter flag for pi+ decay
                    if (piDecayCharge > 0) { p.flags |= FLAG_ANTIMATTER; }
                    particles[pIdx] = p;

                    // Initialize particleAux: radius=cbrt(mE), deathTime=+Inf, deathMass=0, deathAngVel=0
                    // particleId=0 is acceptable (pion decay products are anonymous / not tracked for self-absorption)
                    var aux: ParticleAux;
                    aux.radius = pow(mE, 1.0 / 3.0);
                    aux.particleId = 0xFFFFFFFFu; // sentinel: no emitter identity
                    aux.deathTime = 1e30; // large sentinel (WGSL disallows Inf)
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
