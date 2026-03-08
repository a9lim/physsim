// ─── Shared Boson Utilities ───
// Barnes-Hut tree walk for gravitational lensing of massless (photon) and
// massive (pion) bosons. Shared to avoid code duplication.

import { BH_THETA_SQ, BOSON_SOFTENING_SQ, EPSILON } from './config.js';

let _bStack = new Int32Array(256);

/**
 * Iterative BH tree walk for gravitational deflection of a boson.
 * Accumulates impulse into targetVec (boson.vel or pion.w).
 * @param {Vec2} pos        - boson position
 * @param {Vec2} targetVec  - velocity/proper-velocity vector to accumulate into
 * @param {number} scale    - pre-multiplied factor: grFactor * dt
 * @param {Object} pool     - quadtree pool (particle tree)
 * @param {number} rootIdx  - tree root index
 */
export function treeDeflectBoson(pos, targetVec, scale, pool, rootIdx) {
    const px = pos.x, py = pos.y;
    let stackTop = 0;
    if (_bStack.length < pool.maxNodes) _bStack = new Int32Array(pool.maxNodes);
    _bStack[stackTop++] = rootIdx;

    while (stackTop > 0) {
        const nodeIdx = _bStack[--stackTop];
        if (pool.totalMass[nodeIdx] < EPSILON) continue;

        const dx = pool.comX[nodeIdx] - px;
        const dy = pool.comY[nodeIdx] - py;
        const dSq = dx * dx + dy * dy;
        const size = pool.bw[nodeIdx] * 2;

        const cnt = pool.pointCount[nodeIdx];
        if (!pool.divided[nodeIdx] && cnt > 0) {
            const base = nodeIdx * pool.nodeCapacity;
            for (let i = 0; i < cnt; i++) {
                const p = pool.points[base + i];
                const pdx = p.pos.x - px;
                const pdy = p.pos.y - py;
                const rSq = pdx * pdx + pdy * pdy + BOSON_SOFTENING_SQ;
                const invRSq = 1 / rSq;
                const f = scale * p.mass * Math.sqrt(invRSq) * invRSq;
                targetVec.x += pdx * f;
                targetVec.y += pdy * f;
            }
        } else if (pool.divided[nodeIdx] && (size * size < BH_THETA_SQ * dSq)) {
            const rSq = dSq + BOSON_SOFTENING_SQ;
            const invRSq = 1 / rSq;
            const f = scale * pool.totalMass[nodeIdx] * Math.sqrt(invRSq) * invRSq;
            targetVec.x += dx * f;
            targetVec.y += dy * f;
        } else if (pool.divided[nodeIdx]) {
            _bStack[stackTop++] = pool.nw[nodeIdx];
            _bStack[stackTop++] = pool.ne[nodeIdx];
            _bStack[stackTop++] = pool.sw[nodeIdx];
            _bStack[stackTop++] = pool.se[nodeIdx];
        }
    }
}
