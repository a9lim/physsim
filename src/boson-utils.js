// ─── Shared Boson Utilities ───
// Barnes-Hut tree walk for gravitational lensing of massless (photon) and
// massive (pion) bosons. Shared to avoid code duplication.

import { BH_THETA, BOSON_SOFTENING_SQ } from './config.js';

let _bStack = new Int32Array(256);

/**
 * Iterative BH tree walk for gravitational deflection of a boson.
 * Accumulates impulse into targetVec (boson.vel or pion.w).
 * @param {Vec2} pos        - boson position
 * @param {Vec2} targetVec  - velocity/proper-velocity vector to accumulate into
 * @param {number} grFactor - GR deflection factor (2 for photons, 1+v² for pions)
 * @param {number} dt       - timestep
 * @param {Object} pool     - quadtree pool
 * @param {number} rootIdx  - tree root index
 */
export function treeDeflectBoson(pos, targetVec, grFactor, dt, pool, rootIdx) {
    const thetaSq = BH_THETA * BH_THETA;
    const px = pos.x, py = pos.y;
    let stackTop = 0;
    if (_bStack.length < pool.maxNodes) _bStack = new Int32Array(pool.maxNodes);
    _bStack[stackTop++] = rootIdx;

    while (stackTop > 0) {
        const nodeIdx = _bStack[--stackTop];
        if (pool.totalMass[nodeIdx] === 0) continue;

        const dx = pool.comX[nodeIdx] - px;
        const dy = pool.comY[nodeIdx] - py;
        const dSq = dx * dx + dy * dy;
        const size = pool.bw[nodeIdx] * 2;

        if (!pool.divided[nodeIdx] && pool.pointCount[nodeIdx] > 0) {
            const base = nodeIdx * pool.nodeCapacity;
            for (let i = 0; i < pool.pointCount[nodeIdx]; i++) {
                const p = pool.points[base + i];
                const pdx = p.pos.x - px;
                const pdy = p.pos.y - py;
                const rSq = pdx * pdx + pdy * pdy + BOSON_SOFTENING_SQ;
                const invR3 = 1 / (rSq * Math.sqrt(rSq));
                targetVec.x += grFactor * p.mass * pdx * invR3 * dt;
                targetVec.y += grFactor * p.mass * pdy * invR3 * dt;
            }
        } else if (pool.divided[nodeIdx] && (size * size < thetaSq * dSq)) {
            const rSq = dSq + BOSON_SOFTENING_SQ;
            const invR3 = 1 / (rSq * Math.sqrt(rSq));
            targetVec.x += grFactor * pool.totalMass[nodeIdx] * dx * invR3 * dt;
            targetVec.y += grFactor * pool.totalMass[nodeIdx] * dy * invR3 * dt;
        } else if (pool.divided[nodeIdx]) {
            _bStack[stackTop++] = pool.nw[nodeIdx];
            _bStack[stackTop++] = pool.ne[nodeIdx];
            _bStack[stackTop++] = pool.sw[nodeIdx];
            _bStack[stackTop++] = pool.se[nodeIdx];
        }
    }
}
