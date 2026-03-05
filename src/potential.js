// ─── Potential Energy Computation ───
// Mirrors force calculation structure (BH tree or pairwise) for consistent PE.

import { BH_THETA, SOFTENING_SQ, INERTIA_K, MAG_MOMENT_K } from './config.js';
import { TORUS, minImage } from './topology.js';

const _miOut = { x: 0, y: 0 };

/** Total PE via BH tree traversal (halved to avoid double-counting) or exact pairwise. */
export function computePE(particles, toggles, pool, root, barnesHutEnabled, bhTheta, periodic, domW, domH, topology = TORUS) {
    let pe = 0;
    const halfDomW = domW * 0.5;
    const halfDomH = domH * 0.5;

    if (barnesHutEnabled && root >= 0) {
        for (let i = 0; i < particles.length; i++) {
            pe += treePE(particles[i], pool, root, bhTheta, toggles, periodic, domW, domH, halfDomW, halfDomH, topology);
        }
        pe *= 0.5; // tree counts each pair from both sides
    } else {
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            for (let j = i + 1; j < particles.length; j++) {
                const o = particles[j];
                const oRSq = o.radiusSq;
                pe += pairPE(p, o.pos.x, o.pos.y, o.vel.x, o.vel.y,
                    o.mass, o.charge, o.angVel,
                    MAG_MOMENT_K * o.charge * o.angVel * oRSq,
                    INERTIA_K * o.mass * o.angVel * oRSq, toggles,
                    periodic, domW, domH, halfDomW, halfDomH, topology);
            }
        }
    }

    return pe;
}

/** Recursive BH tree walk for PE; same theta criterion as force calculation. */
export function treePE(particle, pool, nodeIdx, theta, toggles, periodic, domW, domH, halfDomW, halfDomH, topology = TORUS) {
    if (pool.totalMass[nodeIdx] === 0) return 0;

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
            let pe = 0;
            const base = nodeIdx * pool.nodeCapacity;
            for (let i = 0; i < pool.pointCount[nodeIdx]; i++) {
                const other = pool.points[base + i];
                if (other === particle) continue;
                if (other.isGhost && other.original === particle) continue;
                const oRSq = other.radiusSq;
                pe += pairPE(particle, other.pos.x, other.pos.y, other.vel.x, other.vel.y,
                    other.mass, other.charge, other.angVel,
                    MAG_MOMENT_K * other.charge * other.angVel * oRSq,
                    INERTIA_K * other.mass * other.angVel * oRSq, toggles,
                    periodic, domW, domH, halfDomW, halfDomH, topology);
            }
            return pe;
        } else {
            const nodeMass = pool.totalMass[nodeIdx];
            const avgVx = nodeMass > 0 ? pool.totalMomentumX[nodeIdx] / nodeMass : 0;
            const avgVy = nodeMass > 0 ? pool.totalMomentumY[nodeIdx] / nodeMass : 0;
            return pairPE(particle, pool.comX[nodeIdx], pool.comY[nodeIdx], avgVx, avgVy,
                nodeMass, pool.totalCharge[nodeIdx], 0,
                pool.totalMagneticMoment[nodeIdx], pool.totalAngularMomentum[nodeIdx], toggles,
                periodic, domW, domH, halfDomW, halfDomH, topology);
        }
    } else if (pool.divided[nodeIdx]) {
        return treePE(particle, pool, pool.nw[nodeIdx], theta, toggles, periodic, domW, domH, halfDomW, halfDomH, topology)
            + treePE(particle, pool, pool.ne[nodeIdx], theta, toggles, periodic, domW, domH, halfDomW, halfDomH, topology)
            + treePE(particle, pool, pool.sw[nodeIdx], theta, toggles, periodic, domW, domH, halfDomW, halfDomH, topology)
            + treePE(particle, pool, pool.se[nodeIdx], theta, toggles, periodic, domW, domH, halfDomW, halfDomH, topology);
    }
    return 0;
}

/** Pairwise PE: gravity + Coulomb + magnetic dipole + GM dipole + 1PN correction. */
export function pairPE(p, sx, sy, svx, svy, sMass, sCharge, sAngVel, sMagMoment, sAngMomentum, toggles, periodic, domW, domH, halfDomW, halfDomH, topology = TORUS) {
    let rx, ry;
    if (periodic) {
        minImage(p.pos.x, p.pos.y, sx, sy, topology, domW, domH, halfDomW, halfDomH, _miOut);
        rx = _miOut.x; ry = _miOut.y;
    } else {
        rx = sx - p.pos.x; ry = sy - p.pos.y;
    }
    const rSq = rx * rx + ry * ry + SOFTENING_SQ;
    const invRSq = 1 / rSq;
    const invR = Math.sqrt(invRSq);
    const pRSq = p.radiusSq;
    const pMagMoment = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
    const pAngMomentum = INERTIA_K * p.mass * p.angVel * pRSq;

    let pe = 0;
    if (toggles.gravityEnabled)  pe -= p.mass * sMass * invR;
    if (toggles.coulombEnabled)  pe += p.charge * sCharge * invR;
    const invR3 = invR * invRSq;
    if (toggles.magneticEnabled) pe += (pMagMoment * sMagMoment) * invR3;
    if (toggles.gravitomagEnabled) pe -= (pAngMomentum * sAngMomentum) * invR3;
    if (toggles.onePNEnabled) {
        const pvx = p.vel.x, pvy = p.vel.y;
        const nx = rx * invR, ny = ry * invR;
        const v1DotN = pvx * nx + pvy * ny;
        const v2DotN = svx * nx + svy * ny;

        // EIH gravity 1PN PE
        if (toggles.gravitomagEnabled) {
            const v1Sq = pvx * pvx + pvy * pvy;
            const v2Sq = svx * svx + svy * svy;
            const v1DotV2 = pvx * svx + pvy * svy;
            pe -= p.mass * sMass * invR * (
                1.5 * (v1Sq + v2Sq) - 3.5 * v1DotV2 - 0.5 * v1DotN * v2DotN
                + p.mass * invR + sMass * invR
            );
        }

        // Darwin EM 1PN PE: −(q₁q₂)/(2r) × [(v₁·v₂) + (v₁·n̂)(v₂·n̂)]
        if (toggles.magneticEnabled) {
            const v1DotV2 = pvx * svx + pvy * svy;
            pe -= 0.5 * p.charge * sCharge * invR * (v1DotV2 + v1DotN * v2DotN);
        }

        // Bazanski cross-term PE: [q₁q₂(m₁+m₂) − (q₁²m₂ + q₂²m₁)] / (2r²)
        if (toggles.gravitomagEnabled && toggles.magneticEnabled) {
            const crossCoeff = p.charge * sCharge * (p.mass + sMass)
                - (p.charge * p.charge * sMass + sCharge * sCharge * p.mass);
            pe += 0.5 * crossCoeff * invRSq;
        }
    }
    return pe;
}
