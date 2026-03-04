// ─── Potential Energy Computation ───

import { BH_THETA, SOFTENING_SQ, INERTIA_K, MAG_MOMENT_K } from './config.js';

/**
 * Compute total potential energy using same tree/pairwise method as forces.
 * When BH is on: traverses tree per-particle with BH_THETA, divides by 2.
 * When BH is off: exact pairwise i<j (no double-counting).
 *
 * @param {Array} particles
 * @param {Object} toggles - { gravityEnabled, coulombEnabled, magneticEnabled, gravitomagEnabled }
 * @param {Object} pool - QuadTreePool
 * @param {number} root - Root node index
 * @param {boolean} barnesHutEnabled
 * @param {number} bhTheta - Barnes-Hut opening angle (typically BH_THETA)
 * @returns {number} Total potential energy
 */
export function computePE(particles, toggles, pool, root, barnesHutEnabled, bhTheta) {
    let pe = 0;

    if (barnesHutEnabled && root >= 0) {
        for (const p of particles) {
            pe += treePE(p, pool, root, bhTheta, toggles);
        }
        pe *= 0.5; // Each pair counted from both sides
    } else {
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            for (let j = i + 1; j < particles.length; j++) {
                const o = particles[j];
                const oRSq = o.radius * o.radius;
                pe += pairPE(p, o.pos.x, o.pos.y,
                    o.mass, o.charge, o.angVel,
                    MAG_MOMENT_K * o.charge * o.angVel * oRSq,
                    INERTIA_K * o.mass * o.angVel * oRSq, toggles);
            }
        }
    }

    return pe;
}

/**
 * Recursively compute PE contribution from a Barnes-Hut tree node.
 * @param {Object} particle - Test particle
 * @param {Object} pool - QuadTreePool
 * @param {number} nodeIdx - Current tree node index
 * @param {number} theta - Opening angle threshold
 * @param {Object} toggles - { gravityEnabled, coulombEnabled, magneticEnabled, gravitomagEnabled }
 * @returns {number} PE contribution
 */
export function treePE(particle, pool, nodeIdx, theta, toggles) {
    if (pool.totalMass[nodeIdx] === 0) return 0;

    const dx = pool.comX[nodeIdx] - particle.pos.x;
    const dy = pool.comY[nodeIdx] - particle.pos.y;
    const dSq = dx * dx + dy * dy;
    const d = Math.sqrt(dSq);
    const size = pool.bw[nodeIdx] * 2;

    if ((!pool.divided[nodeIdx] && pool.pointCount[nodeIdx] > 0) || (pool.divided[nodeIdx] && (size / d < theta))) {
        if (!pool.divided[nodeIdx]) {
            let pe = 0;
            const base = nodeIdx * pool.nodeCapacity;
            for (let i = 0; i < pool.pointCount[nodeIdx]; i++) {
                const other = pool.points[base + i];
                if (other === particle) continue;
                const oRSq = other.radius * other.radius;
                pe += pairPE(particle, other.pos.x, other.pos.y,
                    other.mass, other.charge, other.angVel,
                    MAG_MOMENT_K * other.charge * other.angVel * oRSq,
                    INERTIA_K * other.mass * other.angVel * oRSq, toggles);
            }
            return pe;
        } else {
            return pairPE(particle, pool.comX[nodeIdx], pool.comY[nodeIdx],
                pool.totalMass[nodeIdx], pool.totalCharge[nodeIdx], 0,
                pool.totalMagneticMoment[nodeIdx], pool.totalAngularMomentum[nodeIdx], toggles);
        }
    } else if (pool.divided[nodeIdx]) {
        return treePE(particle, pool, pool.nw[nodeIdx], theta, toggles)
            + treePE(particle, pool, pool.ne[nodeIdx], theta, toggles)
            + treePE(particle, pool, pool.sw[nodeIdx], theta, toggles)
            + treePE(particle, pool, pool.se[nodeIdx], theta, toggles);
    }
    return 0;
}

/**
 * Compute pairwise PE between a test particle and a source.
 * @param {Object} p - Test particle
 * @param {number} sx - Source x position
 * @param {number} sy - Source y position
 * @param {number} sMass - Source mass
 * @param {number} sCharge - Source charge
 * @param {number} sAngVel - Source angular velocity
 * @param {number} sMagMoment - Source magnetic moment
 * @param {number} sAngMomentum - Source angular momentum
 * @param {Object} toggles - { gravityEnabled, coulombEnabled, magneticEnabled, gravitomagEnabled }
 * @returns {number} PE contribution
 */
export function pairPE(p, sx, sy, sMass, sCharge, sAngVel, sMagMoment, sAngMomentum, toggles) {
    const rx = sx - p.pos.x;
    const ry = sy - p.pos.y;
    const rSq = rx * rx + ry * ry + SOFTENING_SQ;
    const r = Math.sqrt(rSq);
    const invR = 1 / r;
    const invRSq = 1 / rSq;
    const pRSq = p.radius * p.radius;
    const pMagMoment = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
    const pAngMomentum = INERTIA_K * p.mass * p.angVel * pRSq;

    let pe = 0;
    if (toggles.gravityEnabled)  pe -= p.mass * sMass * invR;
    if (toggles.coulombEnabled)  pe += p.charge * sCharge * invR;
    if (toggles.magneticEnabled) pe += (pMagMoment * sMagMoment) * invR * invRSq;
    if (toggles.gravitomagEnabled) pe -= (pAngMomentum * sAngMomentum) * invR * invRSq;
    return pe;
}
