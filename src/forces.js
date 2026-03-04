// ─── Force Computation ───
// Extracted from physics.js: force reset, pairwise force, tree traversal,
// and top-level force accumulation.

import { BH_THETA, SOFTENING_SQ, INERTIA_K, MAG_MOMENT_K, FRAME_DRAG_K } from './config.js';
import { getDelayedState } from './signal-delay.js';

/**
 * Reset all per-particle force accumulators and field values to zero.
 * @param {Array} particles
 */
export function resetForces(particles) {
    for (const p of particles) {
        p.force.set(0, 0);
        p.forceGravity.set(0, 0);
        p.forceCoulomb.set(0, 0);
        p.forceMagnetic.set(0, 0);
        p.forceGravitomag.set(0, 0);
        p.force1PN.set(0, 0);
        p.Bz = 0;
        p.Bgz = 0;
        p.dBzdx = 0;
        p.dBzdy = 0;
        p.dBgzdx = 0;
        p.dBgzdy = 0;
        p._frameDragTorque = 0;
    }
}

/**
 * Compute all forces on all particles, using either Barnes-Hut tree traversal
 * or exact pairwise summation.
 *
 * @param {Array} particles
 * @param {Object} toggles - { gravityEnabled, coulombEnabled, magneticEnabled, gravitomagEnabled }
 * @param {Object} pool - QuadTreePool instance
 * @param {number} root - Root node index in pool (-1 if no tree built)
 * @param {boolean} barnesHutEnabled
 * @param {boolean} signalDelayEnabled
 * @param {boolean} relativityEnabled
 * @param {number} simTime - Current simulation time (for signal delay)
 */
export function computeAllForces(particles, toggles, pool, root, barnesHutEnabled, signalDelayEnabled, relativityEnabled, simTime) {
    if (barnesHutEnabled) {
        if (root < 0) return; // No tree available
        for (const p of particles) {
            calculateForce(p, pool, root, BH_THETA, p.force, toggles);
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
                    const ret = getDelayedState(o, p, simTime);
                    if (ret) {
                        sx = ret.x; sy = ret.y; svx = ret.vx; svy = ret.vy;
                        sAngVel = o.angVel; // angular velocity not history-tracked
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
                    INERTIA_K * o.mass * sAngVel * oRSq, p.force, toggles);
            }
        }
    }
}

/**
 * Compute force from a source (particle or aggregate node) on a particle.
 *
 * Position-dependent (E-like) forces are accumulated into `out` and per-type vectors:
 * gravity, Coulomb, magnetic dipole, gravitomagnetic dipole.
 *
 * Velocity-dependent (B-like) forces (Lorentz, linear GM) are NOT computed here.
 * Instead, the B and Bg field z-components are accumulated on the particle for use
 * in the Boris rotation step, which handles these forces exactly.
 *
 * @param {Object} p - Test particle
 * @param {number} sx - Source x position
 * @param {number} sy - Source y position
 * @param {number} svx - Source x velocity
 * @param {number} svy - Source y velocity
 * @param {number} sMass - Source mass
 * @param {number} sCharge - Source charge
 * @param {number} sAngVel - Source angular velocity
 * @param {number} sMagMoment - Source magnetic moment
 * @param {number} sAngMomentum - Source angular momentum
 * @param {Object} out - Vec2 to accumulate force into
 * @param {Object} toggles - { gravityEnabled, coulombEnabled, magneticEnabled, gravitomagEnabled }
 */
export function pairForce(p, sx, sy, svx, svy, sMass, sCharge, sAngVel, sMagMoment, sAngMomentum, out, toggles) {
    const rx = sx - p.pos.x;
    const ry = sy - p.pos.y;
    const rawRSq = rx * rx + ry * ry;
    // Plummer softening: rSq_eff = r² + ε², keeps F = -dU/dr consistent
    const rSq = rawRSq + SOFTENING_SQ;
    const r = Math.sqrt(rSq);
    const invR = 1 / r;
    const invRSq = 1 / rSq;

    // Cross product of source velocity with separation: (v_s × r̂)_z component
    const crossSV = svx * ry - svy * rx;

    // Test particle dipole moments (uniform-density solid sphere)
    // Magnetic moment: μ = ⅕·q·ω·r²; GM moment (angular momentum): L = I·ω
    const pRSq = p.radius * p.radius;
    const pMagMoment = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
    const pAngMomentum = INERTIA_K * p.mass * p.angVel * pRSq;

    if (toggles.gravityEnabled) {
        const fDir = p.mass * sMass * invRSq * invR;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceGravity.x += rx * fDir;
        p.forceGravity.y += ry * fDir;
    }

    if (toggles.coulombEnabled) {
        const fDir = -(p.charge * sCharge) * invRSq * invR;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceCoulomb.x += rx * fDir;
        p.forceCoulomb.y += ry * fDir;
    }

    if (toggles.onePNEnabled) {
        // 1PN Einstein-Infeld-Hoffmann correction (natural units, G=c=1)
        // Uses coordinate velocities (not proper velocity) per EIH formulation
        const pvx = p.vel.x, pvy = p.vel.y;
        const v1Sq = pvx * pvx + pvy * pvy;
        const v2Sq = svx * svx + svy * svy;
        const v1DotV2 = pvx * svx + pvy * svy;
        const nx = rx * invR, ny = ry * invR;
        const nDotV1 = nx * pvx + ny * pvy;
        const nDotV2 = nx * svx + ny * svy;

        // Radial term coefficient
        const radial = -v1Sq - 2 * v2Sq + 4 * v1DotV2
            + 1.5 * nDotV2 * nDotV2
            + 5 * p.mass * invR + 4 * sMass * invR;

        // Tangential term coefficient (along v1 - v2)
        const tangential = 4 * nDotV1 - 3 * nDotV2;
        const dvx = pvx - svx, dvy = pvy - svy;

        // a_1PN = (m2/r^2) * { n * radial + (v1-v2) * tangential }
        const base = sMass * invRSq * invR;
        const fx = base * (rx * radial + dvx * tangential * r);
        const fy = base * (ry * radial + dvy * tangential * r);

        // Accumulate into total force (for kicks) and per-type display vector
        out.x += fx;
        out.y += fy;
        p.force1PN.x += fx;
        p.force1PN.y += fy;
    }

    if (toggles.magneticEnabled) {
        // Dipole radial component: F = 3μ₁μ₂/r⁴ (aligned ⊥-to-plane dipoles repel)
        const fDir = -3 * (pMagMoment * sMagMoment) * invRSq * invRSq * invR;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceMagnetic.x += rx * fDir;
        p.forceMagnetic.y += ry * fDir;

        // Accumulate EM magnetic field Bz for Boris rotation (Lorentz force)
        // B_z = q_s * (v_s × r̂)_z / r³
        p.Bz += sCharge * crossSV * invR * invRSq;

        // ∇Bz w.r.t. observer position (radial + angular terms)
        // ∂Bz/∂px = +3·Bz·rx/r² + q_s·vsy/r³
        // ∂Bz/∂py = +3·Bz·ry/r² - q_s·vsx/r³
        const Bz_contribution = sCharge * crossSV * invR * invRSq;
        p.dBzdx += 3 * Bz_contribution * rx * invRSq + sCharge * svy * invR * invRSq;
        p.dBzdy += 3 * Bz_contribution * ry * invRSq - sCharge * svx * invR * invRSq;

        // Dipole-sourced Bz: static magnetic field from spinning charged body
        // Bz_dipole = +mu_source / r^3 (equatorial field of z-aligned dipole)
        p.Bz += sMagMoment * invR * invRSq;
        // Gradient: d(mu/r^3)/dpx = +3*mu*rx/r^5
        p.dBzdx += 3 * sMagMoment * rx * invRSq * invRSq * invR;
        p.dBzdy += 3 * sMagMoment * ry * invRSq * invRSq * invR;
    }

    if (toggles.gravitomagEnabled) {
        // Dipole radial component: F = 3L₁L₂/r⁴, co-rotating masses attract (GEM flips EM sign)
        const fDir = 3 * (pAngMomentum * sAngMomentum) * invRSq * invRSq * invR;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceGravitomag.x += rx * fDir;
        p.forceGravitomag.y += ry * fDir;

        // Accumulate GM field Bgz for Boris rotation (linear gravitomagnetism)
        // Bg_z = -m_s * (v_s × r̂)_z / r³  (sign from r̂ = source−observer convention)
        p.Bgz -= sMass * crossSV * invR * invRSq;

        // ∇Bgz w.r.t. observer position (radial + angular terms)
        const Bgz_contribution = -sMass * crossSV * invR * invRSq;
        p.dBgzdx += 3 * Bgz_contribution * rx * invRSq - sMass * svy * invR * invRSq;
        p.dBgzdy += 3 * Bgz_contribution * ry * invRSq + sMass * svx * invR * invRSq;

        // Spin-sourced Bgz: gravitomagnetic field from spinning massive body
        // Bgz_spin = -2 * L_source / r^3 (GEM analog of dipole field)
        p.Bgz -= 2 * sAngMomentum * invR * invRSq;
        // Gradient: d(-2*L/r^3)/dpx = -6*L*rx/r^5
        p.dBgzdx -= 6 * sAngMomentum * rx * invRSq * invRSq * invR;
        p.dBgzdy -= 6 * sAngMomentum * ry * invRSq * invRSq * invR;

        // Frame-dragging torque: drives spins toward co-rotation
        const torque = FRAME_DRAG_K * sMass * (sAngVel - p.angVel) * invR * invRSq;
        p._frameDragTorque = (p._frameDragTorque || 0) + torque;
    }
}

/**
 * Recompute 1PN forces on all particles (pairwise, O(N^2)).
 * Used by the velocity-Verlet correction step — only needs 1PN, not all forces.
 * Resets force1PN before accumulating.
 */
export function compute1PNPairwise(particles, SOFTENING_SQ_VAL) {
    for (let i = 0; i < particles.length; i++) {
        particles[i].force1PN.set(0, 0);
    }
    for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        for (let j = 0; j < particles.length; j++) {
            if (i === j) continue;
            const o = particles[j];
            const rx = o.pos.x - p.pos.x;
            const ry = o.pos.y - p.pos.y;
            const rSq = rx * rx + ry * ry + SOFTENING_SQ_VAL;
            const r = Math.sqrt(rSq);
            const invR = 1 / r;
            const invRSq = 1 / rSq;
            const pvx = p.vel.x, pvy = p.vel.y;
            const svx = o.vel.x, svy = o.vel.y;
            const v1Sq = pvx * pvx + pvy * pvy;
            const v2Sq = svx * svx + svy * svy;
            const v1DotV2 = pvx * svx + pvy * svy;
            const nx = rx * invR, ny = ry * invR;
            const nDotV1 = nx * pvx + ny * pvy;
            const nDotV2 = nx * svx + ny * svy;
            const radial = -v1Sq - 2 * v2Sq + 4 * v1DotV2
                + 1.5 * nDotV2 * nDotV2
                + 5 * p.mass * invR + 4 * o.mass * invR;
            const tangential = 4 * nDotV1 - 3 * nDotV2;
            const dvx = pvx - svx, dvy = pvy - svy;
            const base = o.mass * invRSq * invR;
            p.force1PN.x += base * (rx * radial + dvx * tangential * r);
            p.force1PN.y += base * (ry * radial + dvy * tangential * r);
        }
    }
}

/**
 * Recursively compute force on a particle from a Barnes-Hut tree node.
 * @param {Object} particle - Test particle
 * @param {Object} pool - QuadTreePool
 * @param {number} nodeIdx - Current tree node index
 * @param {number} theta - Opening angle threshold
 * @param {Object} out - Vec2 to accumulate force into
 * @param {Object} toggles - { gravityEnabled, coulombEnabled, magneticEnabled, gravitomagEnabled }
 */
export function calculateForce(particle, pool, nodeIdx, theta, out, toggles) {
    if (pool.totalMass[nodeIdx] === 0) return;

    const dx = pool.comX[nodeIdx] - particle.pos.x;
    const dy = pool.comY[nodeIdx] - particle.pos.y;
    const dSq = dx * dx + dy * dy;
    const d = Math.sqrt(dSq);
    const size = pool.bw[nodeIdx] * 2;

    if ((!pool.divided[nodeIdx] && pool.pointCount[nodeIdx] > 0) || (pool.divided[nodeIdx] && (size / d < theta))) {
        if (!pool.divided[nodeIdx]) {
            const base = nodeIdx * pool.nodeCapacity;
            for (let i = 0; i < pool.pointCount[nodeIdx]; i++) {
                const other = pool.points[base + i];
                if (other === particle) continue;
                const otherRSq = other.radius * other.radius;
                pairForce(particle, other.pos.x, other.pos.y, other.vel.x, other.vel.y, other.mass, other.charge, other.angVel, MAG_MOMENT_K * other.charge * other.angVel * otherRSq, INERTIA_K * other.mass * other.angVel * otherRSq, out, toggles);
            }
        } else {
            const avgVx = pool.totalMass[nodeIdx] > 0 ? pool.totalMomentumX[nodeIdx] / pool.totalMass[nodeIdx] : 0;
            const avgVy = pool.totalMass[nodeIdx] > 0 ? pool.totalMomentumY[nodeIdx] / pool.totalMass[nodeIdx] : 0;
            pairForce(particle, pool.comX[nodeIdx], pool.comY[nodeIdx], avgVx, avgVy, pool.totalMass[nodeIdx], pool.totalCharge[nodeIdx], 0, pool.totalMagneticMoment[nodeIdx], pool.totalAngularMomentum[nodeIdx], out, toggles);
        }
    } else if (pool.divided[nodeIdx]) {
        calculateForce(particle, pool, pool.nw[nodeIdx], theta, out, toggles);
        calculateForce(particle, pool, pool.ne[nodeIdx], theta, out, toggles);
        calculateForce(particle, pool, pool.sw[nodeIdx], theta, out, toggles);
        calculateForce(particle, pool, pool.se[nodeIdx], theta, out, toggles);
    }
}
