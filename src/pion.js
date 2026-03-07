// ─── Pion ───
// Massive force carrier for the Yukawa potential.
// Unlike Photon (massless, |v|=c), pions travel at v<c with proper velocity w.

import Vec2 from './vec2.js';
import { BOSON_SOFTENING_SQ } from './config.js';
import { treeDeflectBoson } from './boson-utils.js';

export default class Pion {
    constructor(x, y, wx, wy, mass, charge, energy, emitterId = -1) {
        this.pos = new Vec2(x, y);
        this.w = new Vec2(wx, wy);
        this.vel = new Vec2(0, 0);
        this.mass = mass;
        this.charge = charge;   // +1, -1, or 0
        this.energy = energy;
        this.lifetime = 0;
        this.alive = true;
        this.emitterId = emitterId;
        this.age = 0;
        this._syncVel();
    }

    _syncVel() {
        const wSq = this.w.x * this.w.x + this.w.y * this.w.y;
        const invG = 1 / Math.sqrt(1 + wSq);
        this.vel.x = this.w.x * invG;
        this.vel.y = this.w.y * invG;
    }

    update(dt, particles, pool, root) {
        // Gravitational deflection: massive particle gets (1+v²) factor, not 2
        const vSq = this.vel.x * this.vel.x + this.vel.y * this.vel.y;
        const grFactor = 1 + vSq;
        if (pool && root >= 0) {
            treeDeflectBoson(this.pos, this.w, grFactor, dt, pool, root);
        } else if (particles) {
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                const dx = p.pos.x - this.pos.x;
                const dy = p.pos.y - this.pos.y;
                const rSq = dx * dx + dy * dy + BOSON_SOFTENING_SQ;
                const invR3 = 1 / (rSq * Math.sqrt(rSq));
                this.w.x += grFactor * p.mass * dx * invR3 * dt;
                this.w.y += grFactor * p.mass * dy * invR3 * dt;
            }
        }

        this._syncVel();
        this.pos.x += this.vel.x * dt;
        this.pos.y += this.vel.y * dt;
        this.lifetime += dt;
        this.age++;
    }

    /** Emit decay products into sim. pi0 -> 2 photons, pi+/- -> 1 photon. */
    decay(sim) {
        if (!sim || this.energy <= 0) return;
        const Photon = sim._PhotonClass;
        if (!Photon) return;
        // Remove pion's contribution before adding photons (avoids double-counting)
        sim.totalRadiated -= this.energy;
        sim.totalRadiatedPx -= this.energy * this.vel.x;
        sim.totalRadiatedPy -= this.energy * this.vel.y;
        const n = this.charge === 0 ? 2 : 1;
        const ePerPh = this.energy / n;
        for (let i = 0; i < n; i++) {
            const angle = this.charge === 0
                ? Math.atan2(this.vel.y, this.vel.x) + (i === 0 ? Math.PI / 2 : -Math.PI / 2)
                : Math.atan2(this.vel.y, this.vel.x);
            const cosA = Math.cos(angle), sinA = Math.sin(angle);
            const ph = new Photon(
                this.pos.x + cosA * 2, this.pos.y + sinA * 2,
                cosA, sinA, ePerPh, -1
            );
            sim.photons.push(ph);
            sim.totalRadiated += ePerPh;
            sim.totalRadiatedPx += ePerPh * cosA;
            sim.totalRadiatedPy += ePerPh * sinA;
        }
        this.alive = false;
    }
}
