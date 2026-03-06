// ─── Force Computation ───
// Pairwise and Barnes-Hut force accumulation. Separates E-like (position-dependent)
// from B-like (velocity-dependent) forces for the Boris integrator.

import { BH_THETA, INERTIA_K, MAG_MOMENT_K, TIDAL_STRENGTH, YUKAWA_G2, EPSILON } from './config.js';
import { getDelayedState } from './signal-delay.js';
import { TORUS, minImage } from './topology.js';

// Reused by minImage() to avoid per-call allocation
const _miOut = { x: 0, y: 0 };

/** Zero all per-particle force accumulators and field values before a new substep. */
export function resetForces(particles) {
    for (let i = 0, n = particles.length; i < n; i++) {
        const p = particles[i];
        p.force.x = p.force.y = 0;
        p.jerk.x = p.jerk.y = 0;
        p.forceGravity.x = p.forceGravity.y = 0;
        p.forceCoulomb.x = p.forceCoulomb.y = 0;
        p.forceMagnetic.x = p.forceMagnetic.y = 0;
        p.forceGravitomag.x = p.forceGravitomag.y = 0;
        p.force1PN.x = p.force1PN.y = 0;
        p.forceSpinCurv.x = p.forceSpinCurv.y = 0;
        p.forceRadiation.x = p.forceRadiation.y = 0;
        p.forceYukawa.x = p.forceYukawa.y = 0;
        p.forceExternal.x = p.forceExternal.y = 0;
        p.forceHiggs.x = p.forceHiggs.y = 0;
        p.forceAxion.x = p.forceAxion.y = 0;
        p.torqueSpinOrbit = 0;
        p.torqueFrameDrag = 0;
        p.torqueTidal = 0;
        p.Bz = 0;
        p.Bgz = 0;
        p.dBzdx = 0;
        p.dBzdy = 0;
        p.dBgzdx = 0;
        p.dBgzdy = 0;
        p._frameDragTorque = 0;
        p._tidalTorque = 0;
    }
}

/**
 * Top-level force dispatch: Barnes-Hut tree walk or exact pairwise O(N^2).
 * In pairwise mode with signal delay, source positions are evaluated on the
 * past light cone rather than at the current time.
 */
export function computeAllForces(particles, toggles, pool, root, barnesHutEnabled, relativityEnabled, simTime, periodic, domW, domH, topology = TORUS) {
    const halfDomW = domW * 0.5;
    const halfDomH = domH * 0.5;
    const n = particles.length;

    // Cache dipole moments once per particle (valid for all pairForce/pairPE calls this substep)
    for (let i = 0; i < n; i++) {
        const p = particles[i];
        const rSq = p.radiusSq;
        p.magMoment = MAG_MOMENT_K * p.charge * p.angVel * rSq;
        p.angMomentum = INERTIA_K * p.mass * p.angVel * rSq;
    }

    const useSignalDelay = relativityEnabled;

    if (barnesHutEnabled) {
        if (root < 0) return;
        for (let i = 0; i < n; i++) {
            calculateForce(particles[i], pool, root, BH_THETA, particles[i].force, toggles, periodic, domW, domH, halfDomW, halfDomH, topology, useSignalDelay, simTime);
        }
    } else {
        // When signal delay is off and 1PN is off, forces are symmetric — use j>i loop
        // (1PN uses per-particle velocity so forces are NOT symmetric; signal delay
        // makes source positions asymmetric)
        const canSymmetric = !useSignalDelay && !toggles.onePNEnabled;

        if (canSymmetric) {
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                for (let j = i + 1; j < n; j++) {
                    const o = particles[j];
                    pairForce(p, o.pos.x, o.pos.y, o.vel.x, o.vel.y,
                        o.mass, o.charge, o.angVel,
                        o.magMoment, o.angMomentum, p.force, toggles,
                        periodic, domW, domH, halfDomW, halfDomH, topology);
                    pairForce(o, p.pos.x, p.pos.y, p.vel.x, p.vel.y,
                        p.mass, p.charge, p.angVel,
                        p.magMoment, p.angMomentum, o.force, toggles,
                        periodic, domW, domH, halfDomW, halfDomH, topology);
                }
            }
        } else {
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                for (let j = 0; j < n; j++) {
                    if (i === j) continue;
                    const o = particles[j];

                    let sx, sy, svx, svy, sAngVel;
                    if (useSignalDelay && o.histCount >= 2) {
                        const ret = getDelayedState(o, p, simTime, periodic, domW, domH, halfDomW, halfDomH, topology);
                        if (ret) {
                            sx = ret.x; sy = ret.y; svx = ret.vx; svy = ret.vy;
                            sAngVel = o.angVel;
                        } else {
                            sx = o.pos.x; sy = o.pos.y; svx = o.vel.x; svy = o.vel.y;
                            sAngVel = o.angVel;
                        }
                    } else {
                        sx = o.pos.x; sy = o.pos.y; svx = o.vel.x; svy = o.vel.y;
                        sAngVel = o.angVel;
                    }

                    pairForce(p, sx, sy, svx, svy,
                        o.mass, o.charge, sAngVel,
                        o.magMoment, o.angMomentum, p.force, toggles,
                        periodic, domW, domH, halfDomW, halfDomH, topology);
                }
            }
        }
    }
}

/**
 * Pairwise force between test particle p and a source at (sx, sy).
 *
 * E-like (position-dependent) forces go into `out` + per-type display vectors.
 * B-like (velocity-dependent) forces are NOT applied here; instead Bz/Bgz
 * and their gradients are accumulated for the Boris rotation step.
 */
export function pairForce(p, sx, sy, svx, svy, sMass, sCharge, sAngVel, sMagMoment, sAngMomentum, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology = TORUS) {
    let rx, ry;
    if (periodic) {
        minImage(p.pos.x, p.pos.y, sx, sy, topology, domW, domH, halfDomW, halfDomH, _miOut);
        rx = _miOut.x; ry = _miOut.y;
    } else {
        rx = sx - p.pos.x; ry = sy - p.pos.y;
    }
    const rawRSq = rx * rx + ry * ry;
    const rSq = rawRSq + toggles.softeningSq;
    const invRSq = 1 / rSq;
    const invR = Math.sqrt(invRSq);       // 1/r via sqrt(1/r²) — one sqrt instead of sqrt + division
    const invR3 = invR * invRSq;          // 1 / r_eff³
    const invR5 = invR3 * invRSq;         // 1 / r_eff⁵

    // (v_s × r)_z — enters Biot-Savart-like field expressions
    const crossSV = svx * ry - svy * rx;

    // Test particle dipole moments (cached per-substep in computeAllForces)
    const pMagMoment = p.magMoment;
    const pAngMomentum = p.angMomentum;

    // Relative velocity (source - particle) for analytical jerk
    const vrx = svx - p.vel.x, vry = svy - p.vel.y;
    const rDotVr = rx * vrx + ry * vry;

    if (toggles.gravityEnabled) {
        const k = p.mass * sMass;
        const fDir = k * invR3;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceGravity.x += rx * fDir;
        p.forceGravity.y += ry * fDir;
        // Analytical jerk: k·[v_rel/r³ − 3·r·(r·v_rel)/r⁵]
        const jRadial = -3 * k * rDotVr * invR5;
        p.jerk.x += vrx * fDir + rx * jRadial;
        p.jerk.y += vry * fDir + ry * jRadial;
    }

    if (toggles.coulombEnabled) {
        const k = -(p.charge * sCharge) * p.axMod;
        const fDir = k * invR3;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceCoulomb.x += rx * fDir;
        p.forceCoulomb.y += ry * fDir;
        // Analytical jerk for Coulomb (same form, different coupling)
        const jRadial = -3 * k * rDotVr * invR5;
        p.jerk.x += vrx * fDir + rx * jRadial;
        p.jerk.y += vry * fDir + ry * jRadial;
    }

    if (toggles.onePNEnabled && (toggles.gravitomagEnabled || toggles.magneticEnabled)) {
        const r = 1 / invR;  // recover r from invR (cheaper than sqrt)
        const nx = rx * invR, ny = ry * invR;
        const pvx = p.vel.x, pvy = p.vel.y;

        if (toggles.gravitomagEnabled) {
            // 1PN EIH symmetric remainder: O(v²/c²) gravity after subtracting the
            // GM Lorentz piece (handled by Boris when GM is on, absent when GM is off).
            const v1Sq = pvx * pvx + pvy * pvy;
            const v2Sq = svx * svx + svy * svy;
            const nDotV1 = nx * pvx + ny * pvy;
            const nDotV2 = nx * svx + ny * svy;
            const radial = -v1Sq - 2 * v2Sq
                + 1.5 * nDotV2 * nDotV2
                + 5 * p.mass * invR + 4 * sMass * invR;
            const v1Coeff = 4 * nDotV1 - 3 * nDotV2;
            const v2Coeff = 3 * nDotV2;
            const base = sMass * invR3;
            const fx = base * (rx * radial + (pvx * v1Coeff + svx * v2Coeff) * r);
            const fy = base * (ry * radial + (pvy * v1Coeff + svy * v2Coeff) * r);
            out.x += fx;
            out.y += fy;
            p.force1PN.x += fx;
            p.force1PN.y += fy;
        }

        if (toggles.magneticEnabled) {
            // Darwin EM symmetric correction: O(v²/c²) from Darwin Lagrangian.
            const v2DotN = svx * nx + svy * ny;
            const v1DotN = pvx * nx + pvy * ny;
            const coeff = 0.5 * p.charge * sCharge * invRSq;
            const symX = coeff * (pvx * v2DotN - 3 * nx * v1DotN * v2DotN);
            const symY = coeff * (pvy * v2DotN - 3 * ny * v1DotN * v2DotN);
            out.x += symX;
            out.y += symY;
            p.force1PN.x += symX;
            p.force1PN.y += symY;
        }
    }

    if (toggles.onePNEnabled && toggles.gravitomagEnabled && toggles.magneticEnabled) {
        // Bazanski cross-term: gravity-EM 1PN mixed interaction (position-dependent only).
        // F = [q₁q₂(m₁+m₂) − (q₁²m₂ + q₂²m₁)] / r³ along r̂
        const crossCoeff = p.charge * sCharge * (p.mass + sMass)
            - (p.charge * p.charge * sMass + sCharge * sCharge * p.mass);
        const fDir = crossCoeff * invRSq * invRSq;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.force1PN.x += rx * fDir;
        p.force1PN.y += ry * fDir;
    }

    if (toggles.magneticEnabled) {
        // Axion modulation: all EM charge-dependent terms scale with local α_eff
        const axMod = p.axMod;
        // Dipole-dipole radial: F = −3μ₁μ₂/r⁴ (aligned ⊥-to-plane dipoles repel)
        const fDir = -3 * (pMagMoment * sMagMoment) * invR5 * axMod;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceMagnetic.x += rx * fDir;
        p.forceMagnetic.y += ry * fDir;

        // Bz from moving charge (Biot-Savart): B_z = q_s(v_s × r̂)_z / r²
        const BzMoving = sCharge * crossSV * invR3 * axMod;
        p.Bz += BzMoving;

        // ∇Bz for spin-orbit coupling (radial + angular terms)
        p.dBzdx += 3 * BzMoving * rx * invRSq + sCharge * svy * invR3 * axMod;
        p.dBzdy += 3 * BzMoving * ry * invRSq - sCharge * svx * invR3 * axMod;

        // Dipole-sourced Bz: equatorial field of z-aligned dipole, +μ/r³
        p.Bz += sMagMoment * invR3 * axMod;
        p.dBzdx += 3 * sMagMoment * rx * invR5 * axMod;
        p.dBzdy += 3 * sMagMoment * ry * invR5 * axMod;
    }

    if (toggles.gravitomagEnabled) {
        // GM dipole: F = +3L₁L₂/r⁴ (GEM sign flip: co-rotating masses attract)
        const fDir = 3 * (pAngMomentum * sAngMomentum) * invR5;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceGravitomag.x += rx * fDir;
        p.forceGravitomag.y += ry * fDir;

        // Bgz from moving mass: −m_s(v_s × r̂)_z / r² (negative: GEM convention)
        const BgzMoving = -sMass * crossSV * invR3;
        p.Bgz += BgzMoving;

        // ∇Bgz for spin-orbit coupling
        p.dBgzdx += 3 * BgzMoving * rx * invRSq - sMass * svy * invR3;
        p.dBgzdy += 3 * BgzMoving * ry * invRSq + sMass * svx * invR3;

        // Spin-sourced Bgz: −2L/r³ (GEM analog of magnetic dipole field)
        p.Bgz -= 2 * sAngMomentum * invR3;
        p.dBgzdx -= 6 * sAngMomentum * rx * invR5;
        p.dBgzdy -= 6 * sAngMomentum * ry * invR5;

        // Frame-dragging torque: aligns spins toward co-rotation
        const torque = 2 * sAngMomentum * (sAngVel - p.angVel) * invR3;
        p._frameDragTorque += torque;
    }

    if (toggles.yukawaEnabled) {
        const mu = toggles.yukawaMu;
        const r = 1 / invR;
        const expMuR = Math.exp(-mu * r);
        // F = g² · exp(-μr) · (1/r² + μ/r) · r̂  (attractive, like gravity)
        const fDir = YUKAWA_G2 * p.mass * sMass * expMuR * (invRSq + mu * invR) * invR;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceYukawa.x += rx * fDir;
        p.forceYukawa.y += ry * fDir;
        // Analytical jerk for radiation reaction
        const jBase = YUKAWA_G2 * p.mass * sMass * expMuR;
        const term1 = (invRSq + mu * invR) * invR;
        const jRadial = -(3 * invRSq + 2 * mu * invR + mu * mu) * rDotVr * jBase * invRSq * invR;
        p.jerk.x += vrx * jBase * term1 + rx * jRadial;
        p.jerk.y += vry * jBase * term1 + ry * jRadial;
    }

    if (toggles.tidalLockingEnabled) {
        // Tidal locking torque: drives spin toward synchronous rotation.
        // Coupling = (m_other + q₁q₂/m)² accounts for all cross-terms:
        // gravity-raises/gravity-torques, gravity/Coulomb, Coulomb/gravity, Coulomb/Coulomb.
        const crossRV = rx * (svy - p.vel.y) - ry * (svx - p.vel.x);
        const wOrbit = crossRV * invRSq;
        const dw = p.angVel - wOrbit;
        let coupling = 0;
        if (toggles.gravityEnabled) coupling += sMass;
        if (toggles.coulombEnabled && p.mass > EPSILON) coupling += p.charge * sCharge / p.mass;
        const ri3 = p.radiusSq * p.radius;
        const invR6 = invRSq * invRSq * invRSq;
        p._tidalTorque -= TIDAL_STRENGTH * coupling * coupling * ri3 * invR6 * dw;
    }
}

/**
 * Recompute 1PN forces pairwise O(N²) for velocity-Verlet correction.
 * Called after drift to get F_1PN(new) for the correction kick.
 */
export function compute1PNPairwise(particles, SOFTENING_SQ_VAL, periodic, domW, domH, halfDomW, halfDomH, topology = TORUS, gravitomagEnabled = true, magneticEnabled = false) {
    const n = particles.length;
    for (let i = 0; i < n; i++) {
        particles[i].force1PN.x = particles[i].force1PN.y = 0;
    }
    for (let i = 0; i < n; i++) {
        const p = particles[i];
        const px = p.pos.x, py = p.pos.y;
        const pvx = p.vel.x, pvy = p.vel.y;
        const pMass = p.mass, pCharge = p.charge;
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const o = particles[j];
            let rx, ry;
            if (periodic) {
                minImage(px, py, o.pos.x, o.pos.y, topology, domW, domH, halfDomW, halfDomH, _miOut);
                rx = _miOut.x; ry = _miOut.y;
            } else {
                rx = o.pos.x - px; ry = o.pos.y - py;
            }
            const rSq = rx * rx + ry * ry + SOFTENING_SQ_VAL;
            const invRSq = 1 / rSq;
            const invR = Math.sqrt(invRSq);
            const svx = o.vel.x, svy = o.vel.y;
            const nx = rx * invR, ny = ry * invR;

            // EIH gravity 1PN symmetric remainder
            if (gravitomagEnabled) {
                const v1Sq = pvx * pvx + pvy * pvy;
                const v2Sq = svx * svx + svy * svy;
                const nDotV1 = nx * pvx + ny * pvy;
                const nDotV2 = nx * svx + ny * svy;
                const radial = -v1Sq - 2 * v2Sq
                    + 1.5 * nDotV2 * nDotV2
                    + 5 * pMass * invR + 4 * o.mass * invR;
                const v1Coeff = 4 * nDotV1 - 3 * nDotV2;
                const v2Coeff = 3 * nDotV2;
                const r = 1 / invR;
                const base = o.mass * invRSq * invR;
                p.force1PN.x += base * (rx * radial + (pvx * v1Coeff + svx * v2Coeff) * r);
                p.force1PN.y += base * (ry * radial + (pvy * v1Coeff + svy * v2Coeff) * r);
            }

            // Darwin EM 1PN
            if (magneticEnabled) {
                const v2DotN = svx * nx + svy * ny;
                const v1DotN = pvx * nx + pvy * ny;
                const coeff = 0.5 * pCharge * o.charge * invRSq;
                const symX = coeff * (pvx * v2DotN - 3 * nx * v1DotN * v2DotN);
                const symY = coeff * (pvy * v2DotN - 3 * ny * v1DotN * v2DotN);
                p.force1PN.x += symX;
                p.force1PN.y += symY;
            }

            // Bazanski cross-term (position-dependent)
            if (gravitomagEnabled && magneticEnabled) {
                const crossCoeff = pCharge * o.charge * (pMass + o.mass)
                    - (pCharge * pCharge * o.mass + o.charge * o.charge * pMass);
                const fDir = crossCoeff * invRSq * invRSq;
                p.force1PN.x += rx * fDir;
                p.force1PN.y += ry * fDir;
            }
        }
    }
}

/**
 * Iterative Barnes-Hut tree walk. Uses aggregate multipole data for distant
 * nodes (size/d < theta), individual particles for nearby leaves.
 * Signal delay is applied at leaf level (individual particles); distant
 * nodes use current-time aggregates (retarded correction is negligible at
 * distances where the BH approximation kicks in).
 */
// Pre-allocated stack for iterative tree walk (avoids recursion overhead)
let _bhStack = new Int32Array(256);

export function calculateForce(particle, pool, rootIdx, theta, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology, useSignalDelay, simTime) {
    const thetaSq = theta * theta;
    const px = particle.pos.x, py = particle.pos.y;
    let stackTop = 0;
    if (_bhStack.length < pool.maxNodes) _bhStack = new Int32Array(pool.maxNodes);
    _bhStack[stackTop++] = rootIdx;

    while (stackTop > 0) {
        const nodeIdx = _bhStack[--stackTop];
        if (pool.totalMass[nodeIdx] === 0) continue;

        let dx, dy;
        if (periodic) {
            minImage(px, py, pool.comX[nodeIdx], pool.comY[nodeIdx], topology, domW, domH, halfDomW, halfDomH, _miOut);
            dx = _miOut.x; dy = _miOut.y;
        } else {
            dx = pool.comX[nodeIdx] - px;
            dy = pool.comY[nodeIdx] - py;
        }
        const dSq = dx * dx + dy * dy;
        const size = pool.bw[nodeIdx] * 2;

        if (!pool.divided[nodeIdx] && pool.pointCount[nodeIdx] > 0) {
            // Leaf node: iterate individual particles
            const base = nodeIdx * pool.nodeCapacity;
            for (let i = 0; i < pool.pointCount[nodeIdx]; i++) {
                const other = pool.points[base + i];
                if (other === particle) continue;
                if (other.isGhost && other.original === particle) continue;
                const real = other.isGhost ? other.original : other;
                let sx, sy, svx, svy;
                if (useSignalDelay && !other.isGhost && real.histCount >= 2) {
                    const ret = getDelayedState(real, particle, simTime, periodic, domW, domH, halfDomW, halfDomH, topology);
                    if (ret) { sx = ret.x; sy = ret.y; svx = ret.vx; svy = ret.vy; }
                    else { sx = other.pos.x; sy = other.pos.y; svx = other.vel.x; svy = other.vel.y; }
                } else {
                    sx = other.pos.x; sy = other.pos.y; svx = other.vel.x; svy = other.vel.y;
                }
                pairForce(particle, sx, sy, svx, svy, other.mass, other.charge, other.angVel, other.magMoment, other.angMomentum, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology);
            }
        } else if (pool.divided[nodeIdx] && (size * size < thetaSq * dSq)) {
            // Distant node: use aggregate (size/d < theta, computed as size²<theta²·d²)
            const nodeMass = pool.totalMass[nodeIdx];
            const avgVx = nodeMass > 0 ? pool.totalMomentumX[nodeIdx] / nodeMass : 0;
            const avgVy = nodeMass > 0 ? pool.totalMomentumY[nodeIdx] / nodeMass : 0;
            pairForce(particle, pool.comX[nodeIdx], pool.comY[nodeIdx], avgVx, avgVy, nodeMass, pool.totalCharge[nodeIdx], 0, pool.totalMagneticMoment[nodeIdx], pool.totalAngularMomentum[nodeIdx], out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology);
        } else if (pool.divided[nodeIdx]) {
            // Push children onto stack
            _bhStack[stackTop++] = pool.nw[nodeIdx];
            _bhStack[stackTop++] = pool.ne[nodeIdx];
            _bhStack[stackTop++] = pool.sw[nodeIdx];
            _bhStack[stackTop++] = pool.se[nodeIdx];
        }
    }
}
