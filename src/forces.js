// ─── Force Computation ───
// Pairwise and Barnes-Hut force accumulation. Separates E-like (position-dependent)
// from B-like (velocity-dependent) forces for the Boris integrator.

import { BH_THETA, SOFTENING_SQ, INERTIA_K, MAG_MOMENT_K, FRAME_DRAG_K, TIDAL_STRENGTH } from './config.js';
import { getDelayedState } from './signal-delay.js';
import { TORUS, minImage } from './topology.js';

// Reused by minImage() to avoid per-call allocation
const _miOut = { x: 0, y: 0 };

/** Zero all per-particle force accumulators and field values before a new substep. */
export function resetForces(particles) {
    for (const p of particles) {
        p.force.set(0, 0);
        p.jerk.set(0, 0);
        p.forceGravity.set(0, 0);
        p.forceCoulomb.set(0, 0);
        p.forceMagnetic.set(0, 0);
        p.forceGravitomag.set(0, 0);
        p.force1PN.set(0, 0);
        p.force1PNEM.set(0, 0);
        p.forceSpinCurv.set(0, 0);
        p.forceRadiation.set(0, 0);
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
export function computeAllForces(particles, toggles, pool, root, barnesHutEnabled, signalDelayEnabled, relativityEnabled, simTime, periodic, domW, domH, topology = TORUS) {
    const halfDomW = domW * 0.5;
    const halfDomH = domH * 0.5;
    if (barnesHutEnabled) {
        if (root < 0) return;
        for (const p of particles) {
            calculateForce(p, pool, root, BH_THETA, p.force, toggles, periodic, domW, domH, halfDomW, halfDomH, topology);
        }
    } else {
        const useSignalDelay = signalDelayEnabled && relativityEnabled;
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            for (let j = 0; j < particles.length; j++) {
                if (i === j) continue;
                const o = particles[j];

                let sx, sy, svx, svy, sAngVel;
                if (useSignalDelay && o.histCount >= 2) {
                    const ret = getDelayedState(o, p, simTime, periodic, domW, domH, halfDomW, halfDomH, topology);
                    if (ret) {
                        sx = ret.x; sy = ret.y; svx = ret.vx; svy = ret.vy;
                        sAngVel = o.angVel; // not history-tracked; use current value
                    } else {
                        sx = o.pos.x; sy = o.pos.y; svx = o.vel.x; svy = o.vel.y;
                        sAngVel = o.angVel;
                    }
                } else {
                    sx = o.pos.x; sy = o.pos.y; svx = o.vel.x; svy = o.vel.y;
                    sAngVel = o.angVel;
                }

                const oRSq = o.radius * o.radius;
                pairForce(p, sx, sy, svx, svy,
                    o.mass, o.charge, sAngVel,
                    MAG_MOMENT_K * o.charge * sAngVel * oRSq,
                    INERTIA_K * o.mass * sAngVel * oRSq, p.force, toggles,
                    periodic, domW, domH, halfDomW, halfDomH, topology);
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
    // Plummer softening: r²_eff = r² + ε² (consistent with PE in potential.js)
    const rSq = rawRSq + SOFTENING_SQ;
    const r = Math.sqrt(rSq);
    const invR = 1 / r;
    const invRSq = 1 / rSq;

    // (v_s × r)_z — enters Biot-Savart-like field expressions
    const crossSV = svx * ry - svy * rx;

    // Test particle dipole moments (uniform-density solid sphere)
    const pRSq = p.radius * p.radius;
    const pMagMoment = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
    const pAngMomentum = INERTIA_K * p.mass * p.angVel * pRSq;

    // Relative velocity (source - particle) for analytical jerk
    const vrx = svx - p.vel.x, vry = svy - p.vel.y;
    const rDotVr = rx * vrx + ry * vry;
    const invR5 = invRSq * invRSq * invR; // 1 / r_eff^5

    if (toggles.gravityEnabled) {
        const k = p.mass * sMass;
        const fDir = k * invRSq * invR;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceGravity.x += rx * fDir;
        p.forceGravity.y += ry * fDir;
        // Analytical jerk: k·[v_rel/r³ − 3·r·(r·v_rel)/r⁵]
        const jCoeff = k * invRSq * invR; // k / r_eff³
        const jRadial = -3 * k * rDotVr * invR5;
        p.jerk.x += vrx * jCoeff + rx * jRadial;
        p.jerk.y += vry * jCoeff + ry * jRadial;
    }

    if (toggles.coulombEnabled) {
        const k = -(p.charge * sCharge);
        const fDir = k * invRSq * invR;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceCoulomb.x += rx * fDir;
        p.forceCoulomb.y += ry * fDir;
        // Analytical jerk for Coulomb (same form, different coupling)
        const jCoeff = k * invRSq * invR;
        const jRadial = -3 * k * rDotVr * invR5;
        p.jerk.x += vrx * jCoeff + rx * jRadial;
        p.jerk.y += vry * jCoeff + ry * jRadial;
    }

    if (toggles.onePNEnabled && toggles.gravitomagEnabled) {
        // 1PN EIH symmetric remainder: O(v²/c²) gravity after subtracting the
        // GM Lorentz piece (handled by Boris when GM is on, absent when GM is off).
        // a = (m2/r²) * { n̂·R + v1·C1 + v2·C2 }
        const pvx = p.vel.x, pvy = p.vel.y;
        const v1Sq = pvx * pvx + pvy * pvy;
        const v2Sq = svx * svx + svy * svy;
        const nx = rx * invR, ny = ry * invR;
        const nDotV1 = nx * pvx + ny * pvy;
        const nDotV2 = nx * svx + ny * svy;
        const radial = -v1Sq - 2 * v2Sq
            + 1.5 * nDotV2 * nDotV2
            + 5 * p.mass * invR + 4 * sMass * invR;
        const v1Coeff = 4 * nDotV1 - 3 * nDotV2;
        const v2Coeff = 3 * nDotV2;
        const base = sMass * invRSq * invR;
        const fx = base * (rx * radial + (pvx * v1Coeff + svx * v2Coeff) * r);
        const fy = base * (ry * radial + (pvy * v1Coeff + svy * v2Coeff) * r);

        out.x += fx;
        out.y += fy;
        p.force1PN.x += fx;
        p.force1PN.y += fy;
    }

    if (toggles.onePNEnabled && toggles.magneticEnabled) {
        // Darwin EM symmetric correction: O(v²/c²) from Darwin Lagrangian.
        // F₁_sym = (q₁q₂)/(2r²) × { v₁(v₂·n̂) − 3n̂(v₁·n̂)(v₂·n̂) }
        // NOT Newton's 3rd law — each particle uses its own velocity.
        const nx = rx * invR, ny = ry * invR;
        const v2DotN = svx * nx + svy * ny;
        const v1DotN = p.vel.x * nx + p.vel.y * ny;
        const coeff = 0.5 * p.charge * sCharge * invRSq;
        const symX = coeff * (p.vel.x * v2DotN - 3 * nx * v1DotN * v2DotN);
        const symY = coeff * (p.vel.y * v2DotN - 3 * ny * v1DotN * v2DotN);
        out.x += symX;
        out.y += symY;
        p.force1PNEM.x += symX;
        p.force1PNEM.y += symY;
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
        // Dipole-dipole radial: F = −3μ₁μ₂/r⁴ (aligned ⊥-to-plane dipoles repel)
        const fDir = -3 * (pMagMoment * sMagMoment) * invRSq * invRSq * invR;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceMagnetic.x += rx * fDir;
        p.forceMagnetic.y += ry * fDir;

        // Bz from moving charge (Biot-Savart): B_z = q_s(v_s × r̂)_z / r²
        const BzMoving = sCharge * crossSV * invR * invRSq;
        p.Bz += BzMoving;

        // ∇Bz for spin-orbit coupling (radial + angular terms)
        p.dBzdx += 3 * BzMoving * rx * invRSq + sCharge * svy * invR * invRSq;
        p.dBzdy += 3 * BzMoving * ry * invRSq - sCharge * svx * invR * invRSq;

        // Dipole-sourced Bz: equatorial field of z-aligned dipole, +μ/r³
        p.Bz += sMagMoment * invR * invRSq;
        p.dBzdx += 3 * sMagMoment * rx * invRSq * invRSq * invR;
        p.dBzdy += 3 * sMagMoment * ry * invRSq * invRSq * invR;
    }

    if (toggles.gravitomagEnabled) {
        // GM dipole: F = +3L₁L₂/r⁴ (GEM sign flip: co-rotating masses attract)
        const fDir = 3 * (pAngMomentum * sAngMomentum) * invRSq * invRSq * invR;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceGravitomag.x += rx * fDir;
        p.forceGravitomag.y += ry * fDir;

        // Bgz from moving mass: −m_s(v_s × r̂)_z / r² (negative: GEM convention)
        const BgzMoving = -sMass * crossSV * invR * invRSq;
        p.Bgz += BgzMoving;

        // ∇Bgz for spin-orbit coupling
        p.dBgzdx += 3 * BgzMoving * rx * invRSq - sMass * svy * invR * invRSq;
        p.dBgzdy += 3 * BgzMoving * ry * invRSq + sMass * svx * invR * invRSq;

        // Spin-sourced Bgz: −2L/r³ (GEM analog of magnetic dipole field)
        p.Bgz -= 2 * sAngMomentum * invR * invRSq;
        p.dBgzdx -= 6 * sAngMomentum * rx * invRSq * invRSq * invR;
        p.dBgzdy -= 6 * sAngMomentum * ry * invRSq * invRSq * invR;

        // Frame-dragging torque: aligns spins toward co-rotation
        const torque = FRAME_DRAG_K * sMass * (sAngVel - p.angVel) * invR * invRSq;
        p._frameDragTorque += torque;
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
        if (toggles.coulombEnabled) coupling += p.charge * sCharge / p.mass;
        const ri3 = p.radius * p.radius * p.radius;
        const invR6 = invRSq * invRSq * invRSq;
        p._tidalTorque -= TIDAL_STRENGTH * coupling * coupling * ri3 * invR6 * dw;
    }
}

/**
 * Recompute 1PN forces pairwise O(N²) for velocity-Verlet correction.
 * Called after drift to get F_1PN(new) for the correction kick.
 */
export function compute1PNPairwise(particles, SOFTENING_SQ_VAL, periodic, domW, domH, halfDomW, halfDomH, topology = TORUS, gravitomagEnabled = true, magneticEnabled = false) {
    for (let i = 0; i < particles.length; i++) {
        particles[i].force1PN.set(0, 0);
        particles[i].force1PNEM.set(0, 0);
    }
    for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        for (let j = 0; j < particles.length; j++) {
            if (i === j) continue;
            const o = particles[j];
            let rx, ry;
            if (periodic) {
                minImage(p.pos.x, p.pos.y, o.pos.x, o.pos.y, topology, domW, domH, halfDomW, halfDomH, _miOut);
                rx = _miOut.x; ry = _miOut.y;
            } else {
                rx = o.pos.x - p.pos.x; ry = o.pos.y - p.pos.y;
            }
            const rSq = rx * rx + ry * ry + SOFTENING_SQ_VAL;
            const r = Math.sqrt(rSq);
            const invR = 1 / r;
            const invRSq = 1 / rSq;
            const pvx = p.vel.x, pvy = p.vel.y;
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
                    + 5 * p.mass * invR + 4 * o.mass * invR;
                const v1Coeff = 4 * nDotV1 - 3 * nDotV2;
                const v2Coeff = 3 * nDotV2;
                const base = o.mass * invRSq * invR;
                p.force1PN.x += base * (rx * radial + (pvx * v1Coeff + svx * v2Coeff) * r);
                p.force1PN.y += base * (ry * radial + (pvy * v1Coeff + svy * v2Coeff) * r);
            }

            // Darwin EM 1PN
            if (magneticEnabled) {
                const v2DotN = svx * nx + svy * ny;
                const v1DotN = pvx * nx + pvy * ny;
                const coeff = 0.5 * p.charge * o.charge * invRSq;
                const symX = coeff * (pvx * v2DotN - 3 * nx * v1DotN * v2DotN);
                const symY = coeff * (pvy * v2DotN - 3 * ny * v1DotN * v2DotN);
                p.force1PNEM.x += symX;
                p.force1PNEM.y += symY;
            }

            // Bazanski cross-term (position-dependent)
            if (gravitomagEnabled && magneticEnabled) {
                const crossCoeff = p.charge * o.charge * (p.mass + o.mass)
                    - (p.charge * p.charge * o.mass + o.charge * o.charge * p.mass);
                const fDir = crossCoeff * invRSq * invRSq;
                p.force1PN.x += rx * fDir;
                p.force1PN.y += ry * fDir;
            }
        }
    }
}

/**
 * Recursive Barnes-Hut tree walk. Uses aggregate multipole data for distant
 * nodes (size/d < theta), individual particles for nearby leaves.
 */
export function calculateForce(particle, pool, nodeIdx, theta, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology = TORUS) {
    if (pool.totalMass[nodeIdx] === 0) return;

    let dx, dy;
    if (periodic) {
        minImage(particle.pos.x, particle.pos.y, pool.comX[nodeIdx], pool.comY[nodeIdx], topology, domW, domH, halfDomW, halfDomH, _miOut);
        dx = _miOut.x; dy = _miOut.y;
    } else {
        dx = pool.comX[nodeIdx] - particle.pos.x;
        dy = pool.comY[nodeIdx] - particle.pos.y;
    }
    const dSq = dx * dx + dy * dy;
    const d = Math.sqrt(dSq);
    const size = pool.bw[nodeIdx] * 2;

    if ((!pool.divided[nodeIdx] && pool.pointCount[nodeIdx] > 0) || (pool.divided[nodeIdx] && (size / d < theta))) {
        if (!pool.divided[nodeIdx]) {
            const base = nodeIdx * pool.nodeCapacity;
            for (let i = 0; i < pool.pointCount[nodeIdx]; i++) {
                const other = pool.points[base + i];
                if (other === particle) continue;
                if (other.isGhost && other.original === particle) continue;
                const otherRSq = other.radius * other.radius;
                pairForce(particle, other.pos.x, other.pos.y, other.vel.x, other.vel.y, other.mass, other.charge, other.angVel, MAG_MOMENT_K * other.charge * other.angVel * otherRSq, INERTIA_K * other.mass * other.angVel * otherRSq, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology);
            }
        } else {
            const avgVx = pool.totalMass[nodeIdx] > 0 ? pool.totalMomentumX[nodeIdx] / pool.totalMass[nodeIdx] : 0;
            const avgVy = pool.totalMass[nodeIdx] > 0 ? pool.totalMomentumY[nodeIdx] / pool.totalMass[nodeIdx] : 0;
            pairForce(particle, pool.comX[nodeIdx], pool.comY[nodeIdx], avgVx, avgVy, pool.totalMass[nodeIdx], pool.totalCharge[nodeIdx], 0, pool.totalMagneticMoment[nodeIdx], pool.totalAngularMomentum[nodeIdx], out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology);
        }
    } else if (pool.divided[nodeIdx]) {
        calculateForce(particle, pool, pool.nw[nodeIdx], theta, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology);
        calculateForce(particle, pool, pool.ne[nodeIdx], theta, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology);
        calculateForce(particle, pool, pool.sw[nodeIdx], theta, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology);
        calculateForce(particle, pool, pool.se[nodeIdx], theta, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology);
    }
}
