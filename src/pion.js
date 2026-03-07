// ─── Pion ───
// Massive force carrier for the Yukawa potential.
// Unlike MasslessBoson (|v|=c), pions travel at v<c with proper velocity w.

import Vec2 from './vec2.js';
import { BOSON_SOFTENING_SQ, ELECTRON_MASS, spawnOffset } from './config.js';
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

    /** Emit decay products. pi0 -> 2 photons, pi+/- -> electron/positron + photon. */
    decay(sim) {
        if (!sim || this.energy <= 0) return;
        const Boson = sim._MasslessBosonClass;
        if (!Boson) return;
        // Remove pion's contribution before adding decay products (avoids double-counting)
        sim.totalRadiated -= this.energy;
        sim.totalRadiatedPx -= this.energy * this.vel.x;
        sim.totalRadiatedPy -= this.energy * this.vel.y;
        const offset = spawnOffset(Math.cbrt(this.mass));
        if (this.charge === 0) {
            // pi0 -> 2 photons: back-to-back in rest frame, Lorentz-boosted to lab.
            // Pick random rest-frame axis, emit E_rest = m/2 each, then boost.
            const restAngle = Math.random() * Math.PI * 2;
            const cosR = Math.cos(restAngle), sinR = Math.sin(restAngle);
            const vx = this.vel.x, vy = this.vel.y;
            const vSq = vx * vx + vy * vy;
            const gamma = 1 / Math.sqrt(1 - vSq + 1e-30);
            const eRest = this.mass * 0.5; // each photon in rest frame
            for (let i = 0; i < 2; i++) {
                const sign = i === 0 ? 1 : -1;
                // Rest-frame photon 4-momentum: (eRest, eRest*cos, eRest*sin)
                let pxR = sign * eRest * cosR;
                let pyR = sign * eRest * sinR;
                // Lorentz boost along pion velocity direction
                if (vSq > 1e-12) {
                    const v = Math.sqrt(vSq);
                    const nx = vx / v, ny = vy / v;
                    const pPar = pxR * nx + pyR * ny; // rest-frame parallel component
                    const pPerpX = pxR - pPar * nx;
                    const pPerpY = pyR - pPar * ny;
                    const eBoosted = gamma * (eRest + v * pPar);
                    const pParBoosted = gamma * (pPar + v * eRest);
                    pxR = pParBoosted * nx + pPerpX;
                    pyR = pParBoosted * ny + pPerpY;
                }
                const pMag = Math.sqrt(pxR * pxR + pyR * pyR);
                const eBoosted = pMag; // massless: E = |p|
                const cosA = pxR / pMag, sinA = pyR / pMag;
                const ph = new Boson(
                    this.pos.x + cosA * offset,
                    this.pos.y + sinA * offset,
                    cosA, sinA, eBoosted, this.emitterId
                );
                sim.photons.push(ph);
                sim.totalRadiated += eBoosted;
                sim.totalRadiatedPx += eBoosted * cosA;
                sim.totalRadiatedPy += eBoosted * sinA;
            }
        } else {
            // pi+/- -> electron/positron + photon (neutrino)
            // Two-body kinematics in pion rest frame, then Lorentz boost
            const mPi = this.mass;
            const mE = ELECTRON_MASS;
            if (mPi <= mE) {
                // Not enough rest energy for electron — emit photon only
                const angle = Math.atan2(this.vel.y, this.vel.x);
                const cosA = Math.cos(angle), sinA = Math.sin(angle);
                const ph = new Boson(
                    this.pos.x + cosA * offset,
                    this.pos.y + sinA * offset,
                    cosA, sinA, this.energy, this.emitterId
                );
                sim.photons.push(ph);
                sim.totalRadiated += this.energy;
                sim.totalRadiatedPx += this.energy * cosA;
                sim.totalRadiatedPy += this.energy * sinA;
                this.alive = false;
                return;
            }
            // Rest-frame energies (exact 2-body conservation)
            const ePhRest = (mPi * mPi - mE * mE) / (2 * mPi);
            const eElRest = mPi - ePhRest; // (mPi² + mE²) / (2 mPi)
            const pRest = ePhRest; // |p| = E_photon (massless), same for electron by momentum conservation

            // Random emission axis in rest frame
            const restAngle = Math.random() * Math.PI * 2;
            const cosR = Math.cos(restAngle), sinR = Math.sin(restAngle);

            // Rest-frame momenta: photon in +dir, electron in -dir
            let phPx = pRest * cosR, phPy = pRest * sinR;
            let elPx = -pRest * cosR, elPy = -pRest * sinR;
            let elELab = eElRest;

            // Lorentz boost along pion velocity
            const vx = this.vel.x, vy = this.vel.y;
            const vSq = vx * vx + vy * vy;
            if (vSq > 1e-12) {
                const v = Math.sqrt(vSq);
                const gamma = 1 / Math.sqrt(1 - vSq + 1e-30);
                const nx = vx / v, ny = vy / v;
                // Boost photon
                const phPar = phPx * nx + phPy * ny;
                const phPerpX = phPx - phPar * nx, phPerpY = phPy - phPar * ny;
                const phParB = gamma * (phPar + v * ePhRest);
                phPx = phParB * nx + phPerpX;
                phPy = phParB * ny + phPerpY;
                // Boost electron
                const elPar = elPx * nx + elPy * ny;
                const elPerpX = elPx - elPar * nx, elPerpY = elPy - elPar * ny;
                const elParB = gamma * (elPar + v * eElRest);
                elPx = elParB * nx + elPerpX;
                elPy = elParB * ny + elPerpY;
                elELab = gamma * (eElRest + v * elPar);
            }

            // Photon (neutrino)
            const phMag = Math.sqrt(phPx * phPx + phPy * phPy);
            const phCos = phPx / phMag, phSin = phPy / phMag;
            const ph = new Boson(
                this.pos.x + phCos * offset,
                this.pos.y + phSin * offset,
                phCos, phSin, phMag, this.emitterId
            );
            sim.photons.push(ph);
            sim.totalRadiated += phMag;
            sim.totalRadiatedPx += phMag * phCos;
            sim.totalRadiatedPy += phMag * phSin;

            // Electron (pi-) or positron (pi+)
            const elVx = elPx / elELab;
            const elVy = elPy / elELab;
            sim.addParticle(
                this.pos.x - phCos * offset,
                this.pos.y - phSin * offset,
                elVx, elVy,
                { mass: mE, charge: this.charge, antimatter: this.charge > 0, spin: 0, skipBaseline: true }
            );
        }
        this.alive = false;
    }
}
