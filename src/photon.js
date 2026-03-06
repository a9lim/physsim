import Vec2 from './vec2.js';
import { BH_THETA, PHOTON_SOFTENING_SQ, EPSILON } from './config.js';

// Pre-allocated stack for iterative tree walk (photon lensing)
let _phStack = new Int32Array(256);

export default class Photon {
    constructor(x, y, vx, vy, energy, emitterId = -1) {
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(vx, vy);
        this.energy = energy;
        this.lifetime = 0;
        this.alive = true;
        this.emitterId = emitterId;
        this.age = 0; // substeps since emission (self-absorption guard)
        this.type = 'em'; // 'em' for electromagnetic, 'grav' for graviton
    }

    update(dt, particles, pool, root) {
        // Gravitational deflection: GR gives 2× Newtonian (null geodesic)
        if (pool && root >= 0) {
            this._treeDeflect(dt, pool, root);
        } else if (particles) {
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                const dx = p.pos.x - this.pos.x;
                const dy = p.pos.y - this.pos.y;
                const rSq = dx * dx + dy * dy + PHOTON_SOFTENING_SQ;
                const invR3 = 1 / (rSq * Math.sqrt(rSq));
                this.vel.x += 2 * p.mass * dx * invR3 * dt;
                this.vel.y += 2 * p.mass * dy * invR3 * dt;
            }
        }

        // Renormalize to c = 1
        const v = Math.sqrt(this.vel.x * this.vel.x + this.vel.y * this.vel.y);
        if (v > EPSILON) {
            this.vel.x /= v;
            this.vel.y /= v;
        }

        this.pos.x += this.vel.x * dt;
        this.pos.y += this.vel.y * dt;
        this.lifetime += dt;
    }

    /** Barnes-Hut tree walk for gravitational lensing (mass only). */
    _treeDeflect(dt, pool, rootIdx) {
        const thetaSq = BH_THETA * BH_THETA;
        const px = this.pos.x, py = this.pos.y;
        let stackTop = 0;
        if (_phStack.length < pool.maxNodes) _phStack = new Int32Array(pool.maxNodes);
        _phStack[stackTop++] = rootIdx;

        while (stackTop > 0) {
            const nodeIdx = _phStack[--stackTop];
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
                    const rSq = pdx * pdx + pdy * pdy + PHOTON_SOFTENING_SQ;
                    const invR3 = 1 / (rSq * Math.sqrt(rSq));
                    this.vel.x += 2 * p.mass * pdx * invR3 * dt;
                    this.vel.y += 2 * p.mass * pdy * invR3 * dt;
                }
            } else if (pool.divided[nodeIdx] && (size * size < thetaSq * dSq)) {
                const rSq = dSq + PHOTON_SOFTENING_SQ;
                const invR3 = 1 / (rSq * Math.sqrt(rSq));
                this.vel.x += 2 * pool.totalMass[nodeIdx] * dx * invR3 * dt;
                this.vel.y += 2 * pool.totalMass[nodeIdx] * dy * invR3 * dt;
            } else if (pool.divided[nodeIdx]) {
                _phStack[stackTop++] = pool.nw[nodeIdx];
                _phStack[stackTop++] = pool.ne[nodeIdx];
                _phStack[stackTop++] = pool.sw[nodeIdx];
                _phStack[stackTop++] = pool.se[nodeIdx];
            }
        }
    }
}
