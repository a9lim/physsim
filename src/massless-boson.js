import Vec2 from './vec2.js';
import { BOSON_SOFTENING_SQ, EPSILON, MAX_PHOTONS } from './config.js';
import { treeDeflectBoson } from './boson-utils.js';

// ─── Object Pool ───
// Recycles dead MasslessBoson instances to eliminate GC pressure from
// frequent new/splice in the physics substep loop.
const _pool = [];
let _poolSize = 0;

export default class MasslessBoson {
    constructor(x, y, vx, vy, energy, emitterId = -1) {
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(vx, vy);
        this.energy = energy;
        this.gravMass = 2 * energy; // E·(1+v²) = E·2 since |v|=c=1
        this._srcMass = energy;     // source gravitational mass (no receiver GR factor)
        this.lifetime = 0;
        this.alive = true;
        this.emitterId = emitterId;
        this.age = 0; // substeps since emission (self-absorption guard)
        this.type = 'em'; // 'em' for electromagnetic, 'grav' for graviton
    }

    /** Reset all fields for pool reuse (avoids constructor allocation). */
    _reset(x, y, vx, vy, energy, emitterId) {
        this.pos.x = x; this.pos.y = y;
        this.vel.x = vx; this.vel.y = vy;
        this.energy = energy;
        this.gravMass = 2 * energy;
        this._srcMass = energy;
        this.lifetime = 0;
        this.alive = true;
        this.emitterId = emitterId;
        this.age = 0;
        this.type = 'em';
    }

    /** Acquire a MasslessBoson from the pool or create a new one. */
    static acquire(x, y, vx, vy, energy, emitterId = -1) {
        if (_poolSize > 0) {
            const b = _pool[--_poolSize];
            b._reset(x, y, vx, vy, energy, emitterId);
            return b;
        }
        return new MasslessBoson(x, y, vx, vy, energy, emitterId);
    }

    /** Return a dead boson to the pool for later reuse. M9: Cap to avoid unbounded GC tracing. */
    static release(b) {
        if (_poolSize < MAX_PHOTONS) _pool[_poolSize++] = b;
    }

    /** Drain all pooled instances (call on simulation reset). */
    static clearPool() {
        _poolSize = 0;
        _pool.length = 0;
    }

    update(dt, particles, pool, root) {
        // Gravitational deflection: GR gives 2× Newtonian (null geodesic)
        if (pool && root >= 0) {
            treeDeflectBoson(this.pos, this.vel, 2 * dt, pool, root);
        } else if (particles) {
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                const dx = p.pos.x - this.pos.x;
                const dy = p.pos.y - this.pos.y;
                const rSq = dx * dx + dy * dy + BOSON_SOFTENING_SQ;
                const invR3 = 1 / (rSq * Math.sqrt(rSq));
                this.vel.x += 2 * p.mass * dx * invR3 * dt;
                this.vel.y += 2 * p.mass * dy * invR3 * dt;
            }
        }

        // Renormalize to c = 1 (skip sqrt when speed drift is negligible)
        const vSq = this.vel.x * this.vel.x + this.vel.y * this.vel.y;
        if (Math.abs(vSq - 1) > 1e-6) {
            const v = Math.sqrt(vSq);
            if (v > EPSILON) {
                this.vel.x /= v;
                this.vel.y /= v;
            }
        }

        this.pos.x += this.vel.x * dt;
        this.pos.y += this.vel.y * dt;
        this.lifetime += dt;
        this.age++;
    }
}
