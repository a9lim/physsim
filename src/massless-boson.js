import Vec2 from './vec2.js';
import { BOSON_SOFTENING_SQ, EPSILON } from './config.js';
import { treeDeflectBoson } from './boson-utils.js';

export default class MasslessBoson {
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
            treeDeflectBoson(this.pos, this.vel, 2, dt, pool, root);
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

        // Renormalize to c = 1
        const v = Math.sqrt(this.vel.x * this.vel.x + this.vel.y * this.vel.y);
        if (v > EPSILON) {
            this.vel.x /= v;
            this.vel.y /= v;
        }

        this.pos.x += this.vel.x * dt;
        this.pos.y += this.vel.y * dt;
        this.lifetime += dt;
        this.age++;
    }
}
