// O(N^2) tiled pairwise force computation.
// Each workgroup loads TILE_SIZE source particles into shared memory,
// then each thread accumulates forces from all sources onto its particle.
//
// Ports CPU pairForce() from forces.js. No signal delay (Phase 4).
// No Barnes-Hut (Phase 3). All force types gated by toggle bits.

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
    radiusSq: f32,  // body radius squared for tidal locking
};

var<workgroup> tile: array<TileParticle, TILE_SIZE>;

// Bind group 0: uniforms
@group(0) @binding(0) var<uniform> uniforms: SimUniforms;

// Bind group 1: particle state (read-only)
@group(1) @binding(0) var<storage, read> posX: array<f32>;
@group(1) @binding(1) var<storage, read> posY: array<f32>;
@group(1) @binding(2) var<storage, read> velX: array<f32>;
@group(1) @binding(3) var<storage, read> velY: array<f32>;
@group(1) @binding(4) var<storage, read> mass: array<f32>;
@group(1) @binding(5) var<storage, read> charge: array<f32>;
@group(1) @binding(6) var<storage, read> angVel: array<f32>;
@group(1) @binding(7) var<storage, read> magMoment: array<f32>;
@group(1) @binding(8) var<storage, read> angMomentum: array<f32>;
@group(1) @binding(9) var<storage, read> axMod: array<f32>;
@group(1) @binding(10) var<storage, read> yukMod: array<f32>;
@group(1) @binding(11) var<storage, read> flags: array<u32>;
@group(1) @binding(12) var<storage, read> radiusSq: array<f32>;
@group(1) @binding(13) var<storage, read> velWX: array<f32>;
@group(1) @binding(14) var<storage, read> velWY: array<f32>;

// Bind group 2: force accumulators (read_write)
@group(2) @binding(0) var<storage, read_write> forces0: array<vec4<f32>>;   // gravity.xy, coulomb.xy
@group(2) @binding(1) var<storage, read_write> forces1: array<vec4<f32>>;   // magnetic.xy, gravitomag.xy
@group(2) @binding(2) var<storage, read_write> forces2: array<vec4<f32>>;   // f1pn.xy, spinCurv.xy
@group(2) @binding(3) var<storage, read_write> forces3: array<vec4<f32>>;   // radiation.xy, yukawa.xy
@group(2) @binding(4) var<storage, read_write> torques: array<vec4<f32>>;   // spinOrbit, frameDrag, tidal, contact
@group(2) @binding(5) var<storage, read_write> bFields: array<vec4<f32>>;   // Bz, Bgz, extBz, pad
@group(2) @binding(6) var<storage, read_write> bFieldGrads: array<vec4<f32>>; // dBzdx, dBzdy, dBgzdx, dBgzdy
@group(2) @binding(7) var<storage, read_write> totalForceX: array<f32>;
@group(2) @binding(8) var<storage, read_write> totalForceY: array<f32>;

@compute @workgroup_size(TILE_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = gid.x;
    let localIdx = lid.x;
    let N = uniforms.aliveCount;

    // Load this thread's particle (observer)
    let alive = idx < N && (flags[idx] & FLAG_ALIVE) != 0u;

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
    var pInvMass: f32 = 0.0;
    var pRadiusSq: f32 = 0.0;
    var pBodyRadiusSq: f32 = 0.0;  // cbrt(mass)^2

    if (alive) {
        pPosX = posX[idx];
        pPosY = posY[idx];
        pVelX = velX[idx];
        pVelY = velY[idx];
        pMass = mass[idx];
        pCharge = charge[idx];
        pAngVel = angVel[idx];
        pMagMom = magMoment[idx];
        pAngMom = angMomentum[idx];
        pAxMod = axMod[idx];
        pYukMod = yukMod[idx];
        pRadiusSq = radiusSq[idx];
        pInvMass = select(0.0, 1.0 / pMass, pMass > EPSILON);
        // Body radius squared for tidal locking: cbrt(mass)^2
        pBodyRadiusSq = pow(pMass, 2.0 / 3.0);
    }

    // Read toggle bits
    let gravOn = hasToggle0(GRAVITY_BIT);
    let coulOn = hasToggle0(COULOMB_BIT);
    let magOn = hasToggle0(MAGNETIC_BIT);
    let gmOn = hasToggle0(GRAVITOMAG_BIT);
    let onePNOn = hasToggle0(ONE_PN_BIT);
    let relOn = hasToggle0(RELATIVITY_BIT);
    let yukawaOn = hasToggle0(YUKAWA_BIT);
    let axionOn = hasToggle0(AXION_BIT);
    let radOn = hasToggle0(RADIATION_BIT);
    let isPeriodic = uniforms.boundaryMode == BOUND_LOOP;

    let softeningSq = uniforms.softeningSq;

    // Axion EM modulation flag
    let needAxMod = (coulOn || magOn) && axionOn;

    // Yukawa cutoff: exp(-mu*r) < 0.002 when mu*r > 6
    let yukMu = uniforms.yukawaMu;
    let yukCutoffSq = select(1e30, (6.0 / yukMu) * (6.0 / yukMu), yukawaOn && yukMu > EPSILON);

    // Per-thread accumulators
    var accGravX: f32 = 0.0; var accGravY: f32 = 0.0;
    var accCoulX: f32 = 0.0; var accCoulY: f32 = 0.0;
    var accMagX: f32 = 0.0; var accMagY: f32 = 0.0;
    var accGMX: f32 = 0.0; var accGMY: f32 = 0.0;
    var acc1PNX: f32 = 0.0; var acc1PNY: f32 = 0.0;
    var accYukX: f32 = 0.0; var accYukY: f32 = 0.0;
    var accTotalX: f32 = 0.0; var accTotalY: f32 = 0.0;

    // B-field accumulators
    var accBz: f32 = 0.0;
    var accBgz: f32 = 0.0;
    var accDBzdx: f32 = 0.0; var accDBzdy: f32 = 0.0;
    var accDBgzdx: f32 = 0.0; var accDBgzdy: f32 = 0.0;

    // Torque accumulators
    var accFrameDrag: f32 = 0.0;
    var accTidal: f32 = 0.0;

    // Number of tiles needed to cover all particles
    let numTiles = (N + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < numTiles; t++) {
        // Collaborative tile load: each thread loads one source particle
        let srcIdx = t * TILE_SIZE + localIdx;
        if (srcIdx < N && (flags[srcIdx] & FLAG_ALIVE) != 0u) {
            tile[localIdx].posX = posX[srcIdx];
            tile[localIdx].posY = posY[srcIdx];
            tile[localIdx].velX = velX[srcIdx];
            tile[localIdx].velY = velY[srcIdx];
            tile[localIdx].mass = mass[srcIdx];
            tile[localIdx].charge = charge[srcIdx];
            tile[localIdx].angVel = angVel[srcIdx];
            tile[localIdx].magMoment = magMoment[srcIdx];
            tile[localIdx].angMomentum = angMomentum[srcIdx];
            tile[localIdx].axMod = axMod[srcIdx];
            tile[localIdx].yukMod = yukMod[srcIdx];
            tile[localIdx].radiusSq = radiusSq[srcIdx];
        } else {
            // Mark as invalid (zero mass = no force contribution)
            tile[localIdx].mass = 0.0;
        }

        workgroupBarrier();

        // Each thread accumulates forces from all sources in this tile
        if (alive) {
            for (var j: u32 = 0u; j < TILE_SIZE; j++) {
                let sIdx = t * TILE_SIZE + j;
                if (sIdx >= N || sIdx == idx) { continue; }
                let s = tile[j];
                if (s.mass < EPSILON) { continue; }

                // Minimum image displacement (torus only for Phase 2)
                var rx: f32;
                var ry: f32;
                if (isPeriodic) {
                    let mi = torusMinImage(pPosX, pPosY, s.posX, s.posY);
                    rx = mi.x;
                    ry = mi.y;
                } else {
                    rx = s.posX - pPosX;
                    ry = s.posY - pPosY;
                }

                let rawRSq = rx * rx + ry * ry;
                let rSq = rawRSq + softeningSq;
                let invRSq = 1.0 / rSq;
                let invR = sqrt(invRSq);
                let invR3 = invR * invRSq;
                let invR5 = invR3 * invRSq;

                // (v_s x r)_z for Biot-Savart
                let crossSV = s.velX * ry - s.velY * rx;

                // Axion modulation (geometric mean)
                let axModPair = select(1.0, sqrt(pAxMod * s.axMod), needAxMod);

                // ── Gravity ──
                if (gravOn) {
                    let k = pMass * s.mass;
                    let fDir = k * invR3;
                    accGravX += rx * fDir;
                    accGravY += ry * fDir;
                    accTotalX += rx * fDir;
                    accTotalY += ry * fDir;

                    // Tidal locking torque: tau = -0.3 * coupling^2 * r_body^5 / r^6 * (w_spin - w_orbit)
                    let crossRV = rx * (s.velY - pVelY) - ry * (s.velX - pVelX);
                    let wOrbit = crossRV * invRSq;
                    let dw = pAngVel - wOrbit;
                    var coupling = s.mass;
                    if (coulOn && pMass > EPSILON) {
                        coupling += pCharge * s.charge / pMass;
                    }
                    let ri5 = pBodyRadiusSq * pBodyRadiusSq * pow(pMass, 1.0 / 3.0);
                    let invR6 = invRSq * invRSq * invRSq;
                    accTidal -= TIDAL_STRENGTH * coupling * coupling * ri5 * invR6 * dw;
                }

                // ── Coulomb ──
                if (coulOn) {
                    let k = -(pCharge * s.charge) * axModPair;
                    let fDir = k * invR3;
                    accCoulX += rx * fDir;
                    accCoulY += ry * fDir;
                    accTotalX += rx * fDir;
                    accTotalY += ry * fDir;
                }

                // ── 1PN EIH (gravity) ──
                if (onePNOn && gmOn) {
                    let r = 1.0 / invR;
                    let nx = rx * invR;
                    let ny = ry * invR;
                    let v1Sq = pVelX * pVelX + pVelY * pVelY;
                    let v2Sq = s.velX * s.velX + s.velY * s.velY;
                    let nDotV1 = nx * pVelX + ny * pVelY;
                    let nDotV2 = nx * s.velX + ny * s.velY;
                    let radial = -v1Sq - 2.0 * v2Sq
                        + 1.5 * nDotV2 * nDotV2
                        + 5.0 * pMass * invR + 4.0 * s.mass * invR;
                    let v1Coeff = 4.0 * nDotV1 - 3.0 * nDotV2;
                    let v2Coeff = 3.0 * nDotV2;
                    let base = s.mass * invR3;
                    let fx = base * (rx * radial + (pVelX * v1Coeff + s.velX * v2Coeff) * r);
                    let fy = base * (ry * radial + (pVelY * v1Coeff + s.velY * v2Coeff) * r);
                    acc1PNX += fx;
                    acc1PNY += fy;
                    accTotalX += fx;
                    accTotalY += fy;
                }

                // ── 1PN Darwin EM ──
                if (onePNOn && magOn) {
                    let nx = rx * invR;
                    let ny = ry * invR;
                    let v2DotN = s.velX * nx + s.velY * ny;
                    let v1DotN = pVelX * nx + pVelY * ny;
                    let coeff = 0.5 * pCharge * s.charge * invRSq;
                    let fx = coeff * (pVelX * v2DotN - 3.0 * nx * v1DotN * v2DotN);
                    let fy = coeff * (pVelY * v2DotN - 3.0 * ny * v1DotN * v2DotN);
                    acc1PNX += fx;
                    acc1PNY += fy;
                    accTotalX += fx;
                    accTotalY += fy;
                }

                // ── 1PN Bazanski (mixed gravity+EM) ──
                if (onePNOn && gmOn && magOn) {
                    let crossCoeff = pCharge * s.charge * (pMass + s.mass)
                        - (pCharge * pCharge * s.mass + s.charge * s.charge * pMass);
                    let fDir = crossCoeff * invRSq * invRSq;
                    acc1PNX += rx * fDir;
                    acc1PNY += ry * fDir;
                    accTotalX += rx * fDir;
                    accTotalY += ry * fDir;
                }

                // ── Magnetic dipole-dipole ──
                if (magOn) {
                    let fDir = 3.0 * (pMagMom * s.magMoment) * invR5 * axModPair;
                    accMagX += rx * fDir;
                    accMagY += ry * fDir;
                    accTotalX += rx * fDir;
                    accTotalY += ry * fDir;

                    // Bz from moving charge (Biot-Savart)
                    let BzMoving = s.charge * crossSV * invR3 * axModPair;
                    accBz += BzMoving;

                    // dBz gradients for spin-orbit
                    accDBzdx += 3.0 * BzMoving * rx * invRSq + s.charge * s.velY * invR3 * axModPair;
                    accDBzdy += 3.0 * BzMoving * ry * invRSq - s.charge * s.velX * invR3 * axModPair;

                    // Dipole-sourced Bz: -mu/r^3
                    accBz -= s.magMoment * invR3 * axModPair;
                    accDBzdx -= 3.0 * s.magMoment * rx * invR5 * axModPair;
                    accDBzdy -= 3.0 * s.magMoment * ry * invR5 * axModPair;
                }

                // ── Gravitomagnetic dipole-dipole ──
                if (gmOn) {
                    let fDir = 3.0 * (pAngMom * s.angMomentum) * invR5;
                    accGMX += rx * fDir;
                    accGMY += ry * fDir;
                    accTotalX += rx * fDir;
                    accTotalY += ry * fDir;

                    // Bgz from moving mass: -m_s(v_s x r_hat)_z / r^2
                    let BgzMoving = -s.mass * crossSV * invR3;
                    accBgz += BgzMoving;

                    // dBgz gradients for spin-orbit
                    accDBgzdx += 3.0 * BgzMoving * rx * invRSq - s.mass * s.velY * invR3;
                    accDBgzdy += 3.0 * BgzMoving * ry * invRSq + s.mass * s.velX * invR3;

                    // Spin-sourced Bgz: -2L/r^3
                    accBgz -= 2.0 * s.angMomentum * invR3;
                    accDBgzdx -= 6.0 * s.angMomentum * rx * invR5;
                    accDBgzdy -= 6.0 * s.angMomentum * ry * invR5;

                    // Frame-dragging torque: aligns spins toward co-rotation
                    accFrameDrag += 2.0 * s.angMomentum * (s.angVel - pAngVel) * invR3;
                }

                // ── Yukawa ──
                if (yukawaOn && rawRSq < yukCutoffSq) {
                    let r = 1.0 / invR;
                    let expMuR = exp(-yukMu * r);
                    let yukModPair = sqrt(pYukMod * s.yukMod);
                    let fDir = uniforms.yukawaCoupling * yukModPair * pMass * s.mass * expMuR * (invRSq + yukMu * invR) * invR;
                    accYukX += rx * fDir;
                    accYukY += ry * fDir;
                    accTotalX += rx * fDir;
                    accTotalY += ry * fDir;

                    // Scalar Breit 1PN correction
                    if (onePNOn) {
                        let nx = rx * invR;
                        let ny = ry * invR;
                        let nDotV1 = nx * pVelX + ny * pVelY;
                        let nDotV2 = nx * s.velX + ny * s.velY;
                        let v1DotV2 = pVelX * s.velX + pVelY * s.velY;
                        let alpha = 1.0 + yukMu * r;
                        let beta = 0.5 * uniforms.yukawaCoupling * yukModPair * pMass * s.mass * expMuR * invRSq;
                        let radial = -(alpha * v1DotV2 + (alpha * alpha + alpha + 1.0) * nDotV1 * nDotV2);
                        let fx = beta * (radial * nx + alpha * (nDotV2 * pVelX + nDotV1 * s.velX));
                        let fy = beta * (radial * ny + alpha * (nDotV2 * pVelY + nDotV1 * s.velY));
                        acc1PNX += fx;
                        acc1PNY += fy;
                        accTotalX += fx;
                        accTotalY += fy;
                    }
                }
            }
        }

        workgroupBarrier();
    }

    // Write accumulated forces to global memory
    if (alive) {
        forces0[idx] = vec4(accGravX, accGravY, accCoulX, accCoulY);
        forces1[idx] = vec4(accMagX, accMagY, accGMX, accGMY);
        forces2[idx] = vec4(acc1PNX, acc1PNY, 0.0, 0.0);  // spinCurv filled by spin-orbit pass
        forces3[idx] = vec4(0.0, 0.0, accYukX, accYukY);   // radiation.xy filled by Phase 4
        torques[idx] = vec4(0.0, accFrameDrag, accTidal, 0.0);
        bFields[idx] = vec4(accBz, accBgz, 0.0, 0.0);  // extBz added by external fields pass
        bFieldGrads[idx] = vec4(accDBzdx, accDBzdy, accDBgzdx, accDBgzdy);
        totalForceX[idx] = accTotalX;
        totalForceY[idx] = accTotalY;
    }
}
