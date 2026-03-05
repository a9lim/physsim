// ─── Barnes-Hut Quadtree ───
// SoA flat typed-array pool: pre-allocated, doubles on overflow, zero GC per frame.

import { INERTIA_K, MAG_MOMENT_K } from './config.js';

export class Rect {
    constructor(x, y, w, h) {
        this.x = x; this.y = y; this.w = w; this.h = h;
    }
}

const NONE = -1;

export default class QuadTreePool {
    constructor(capacity = 4, maxNodes = 512) {
        this.nodeCapacity = capacity;
        this.maxNodes = maxNodes;

        this.bx = new Float64Array(maxNodes);
        this.by = new Float64Array(maxNodes);
        this.bw = new Float64Array(maxNodes);
        this.bh = new Float64Array(maxNodes);

        this.totalMass = new Float64Array(maxNodes);
        this.totalCharge = new Float64Array(maxNodes);
        this.totalMagneticMoment = new Float64Array(maxNodes);
        this.totalAngularMomentum = new Float64Array(maxNodes);
        this.totalMomentumX = new Float64Array(maxNodes);
        this.totalMomentumY = new Float64Array(maxNodes);
        this.comX = new Float64Array(maxNodes);
        this.comY = new Float64Array(maxNodes);

        this.nw = new Int32Array(maxNodes).fill(NONE);
        this.ne = new Int32Array(maxNodes).fill(NONE);
        this.sw = new Int32Array(maxNodes).fill(NONE);
        this.se = new Int32Array(maxNodes).fill(NONE);

        this.points = new Array(maxNodes * capacity).fill(null);
        this.pointCount = new Uint8Array(maxNodes);

        this.divided = new Uint8Array(maxNodes);
        this.count = 0;

        /** Reusable buffer for queryReuse() — avoids per-call array allocation. */
        this._queryBuf = [];
    }

    reset() {
        this.count = 0;
    }

    alloc(bx, by, bw, bh) {
        if (this.count >= this.maxNodes) this._grow();
        const idx = this.count++;
        this.bx[idx] = bx;
        this.by[idx] = by;
        this.bw[idx] = bw;
        this.bh[idx] = bh;
        this.totalMass[idx] = 0;
        this.totalCharge[idx] = 0;
        this.totalMagneticMoment[idx] = 0;
        this.totalAngularMomentum[idx] = 0;
        this.totalMomentumX[idx] = 0;
        this.totalMomentumY[idx] = 0;
        this.comX[idx] = bx;
        this.comY[idx] = by;
        this.nw[idx] = NONE;
        this.ne[idx] = NONE;
        this.sw[idx] = NONE;
        this.se[idx] = NONE;
        this.pointCount[idx] = 0;
        this.divided[idx] = 0;
        const base = idx * this.nodeCapacity;
        for (let i = 0; i < this.nodeCapacity; i++) this.points[base + i] = null;
        return idx;
    }

    _grow() {
        const newMax = this.maxNodes * 2;
        const copyF64 = (old) => { const a = new Float64Array(newMax); a.set(old); return a; };
        const copyI32 = (old, fill) => { const a = new Int32Array(newMax); a.set(old); a.fill(fill, this.maxNodes); return a; };
        const copyU8 = (old) => { const a = new Uint8Array(newMax); a.set(old); return a; };

        this.bx = copyF64(this.bx); this.by = copyF64(this.by);
        this.bw = copyF64(this.bw); this.bh = copyF64(this.bh);
        this.totalMass = copyF64(this.totalMass); this.totalCharge = copyF64(this.totalCharge);
        this.totalMagneticMoment = copyF64(this.totalMagneticMoment);
        this.totalAngularMomentum = copyF64(this.totalAngularMomentum);
        this.totalMomentumX = copyF64(this.totalMomentumX);
        this.totalMomentumY = copyF64(this.totalMomentumY);
        this.comX = copyF64(this.comX); this.comY = copyF64(this.comY);
        this.nw = copyI32(this.nw, NONE); this.ne = copyI32(this.ne, NONE);
        this.sw = copyI32(this.sw, NONE); this.se = copyI32(this.se, NONE);
        this.pointCount = copyU8(this.pointCount);
        this.divided = copyU8(this.divided);

        const newPoints = new Array(newMax * this.nodeCapacity).fill(null);
        for (let i = 0; i < this.maxNodes * this.nodeCapacity; i++) newPoints[i] = this.points[i];
        this.points = newPoints;

        this.maxNodes = newMax;
    }

    _contains(idx, px, py) {
        return (px >= this.bx[idx] - this.bw[idx] &&
            px <= this.bx[idx] + this.bw[idx] &&
            py >= this.by[idx] - this.bh[idx] &&
            py <= this.by[idx] + this.bh[idx]);
    }

    _intersects(idx, rx, ry, rw, rh) {
        return !(rx - rw > this.bx[idx] + this.bw[idx] ||
            rx + rw < this.bx[idx] - this.bw[idx] ||
            ry - rh > this.by[idx] + this.bh[idx] ||
            ry + rh < this.by[idx] - this.bh[idx]);
    }

    _subdivide(idx) {
        const x = this.bx[idx], y = this.by[idx];
        const hw = this.bw[idx] / 2, hh = this.bh[idx] / 2;
        this.nw[idx] = this.alloc(x - hw, y - hh, hw, hh);
        this.ne[idx] = this.alloc(x + hw, y - hh, hw, hh);
        this.sw[idx] = this.alloc(x - hw, y + hh, hw, hh);
        this.se[idx] = this.alloc(x + hw, y + hh, hw, hh);
        this.divided[idx] = 1;
    }

    insert(idx, particle) {
        if (!this._contains(idx, particle.pos.x, particle.pos.y)) return false;

        const cap = this.nodeCapacity;
        if (this.pointCount[idx] < cap && !this.divided[idx]) {
            this.points[idx * cap + this.pointCount[idx]] = particle;
            this.pointCount[idx]++;
            return true;
        }

        if (!this.divided[idx]) {
            this._subdivide(idx);
            const base = idx * cap;
            for (let i = 0; i < this.pointCount[idx]; i++) {
                const p = this.points[base + i];
                this.insert(this.nw[idx], p) ||
                    this.insert(this.ne[idx], p) ||
                    this.insert(this.sw[idx], p) ||
                    this.insert(this.se[idx], p);
                this.points[base + i] = null;
            }
            this.pointCount[idx] = 0;
        }

        return this.insert(this.nw[idx], particle) ||
            this.insert(this.ne[idx], particle) ||
            this.insert(this.sw[idx], particle) ||
            this.insert(this.se[idx], particle);
    }

    calculateMassDistribution(idx) {
        if (!this.divided[idx]) {
            const cnt = this.pointCount[idx];
            if (cnt === 0) return;

            let mass = 0, charge = 0, magMom = 0, angMom = 0;
            let cx = 0, cy = 0, momX = 0, momY = 0;
            const base = idx * this.nodeCapacity;

            for (let i = 0; i < cnt; i++) {
                const p = this.points[base + i];
                const rSq = p.radiusSq;
                mass += p.mass;
                charge += p.charge;
                magMom += MAG_MOMENT_K * p.charge * p.angVel * rSq;
                angMom += INERTIA_K * p.mass * p.angVel * rSq;
                cx += p.pos.x * p.mass;
                cy += p.pos.y * p.mass;
                momX += p.mass * p.w.x;
                momY += p.mass * p.w.y;
            }

            this.totalMass[idx] = mass;
            this.totalCharge[idx] = charge;
            this.totalMagneticMoment[idx] = magMom;
            this.totalAngularMomentum[idx] = angMom;
            this.totalMomentumX[idx] = momX;
            this.totalMomentumY[idx] = momY;
            if (mass > 0) { this.comX[idx] = cx / mass; this.comY[idx] = cy / mass; }
        } else {
            const c0 = this.nw[idx], c1 = this.ne[idx], c2 = this.sw[idx], c3 = this.se[idx];
            this.calculateMassDistribution(c0);
            this.calculateMassDistribution(c1);
            this.calculateMassDistribution(c2);
            this.calculateMassDistribution(c3);

            const mass = this.totalMass[c0] + this.totalMass[c1] + this.totalMass[c2] + this.totalMass[c3];
            this.totalMass[idx] = mass;
            this.totalCharge[idx] = this.totalCharge[c0] + this.totalCharge[c1] + this.totalCharge[c2] + this.totalCharge[c3];
            this.totalMagneticMoment[idx] = this.totalMagneticMoment[c0] + this.totalMagneticMoment[c1] + this.totalMagneticMoment[c2] + this.totalMagneticMoment[c3];
            this.totalAngularMomentum[idx] = this.totalAngularMomentum[c0] + this.totalAngularMomentum[c1] + this.totalAngularMomentum[c2] + this.totalAngularMomentum[c3];
            this.totalMomentumX[idx] = this.totalMomentumX[c0] + this.totalMomentumX[c1] + this.totalMomentumX[c2] + this.totalMomentumX[c3];
            this.totalMomentumY[idx] = this.totalMomentumY[c0] + this.totalMomentumY[c1] + this.totalMomentumY[c2] + this.totalMomentumY[c3];
            if (mass > 0) {
                this.comX[idx] = (this.comX[c0] * this.totalMass[c0] + this.comX[c1] * this.totalMass[c1] + this.comX[c2] * this.totalMass[c2] + this.comX[c3] * this.totalMass[c3]) / mass;
                this.comY[idx] = (this.comY[c0] * this.totalMass[c0] + this.comY[c1] * this.totalMass[c1] + this.comY[c2] * this.totalMass[c2] + this.comY[c3] * this.totalMass[c3]) / mass;
            }
        }
    }

    query(idx, rx, ry, rw, rh, found) {
        if (!found) found = [];
        if (!this._intersects(idx, rx, ry, rw, rh)) return found;

        if (!this.divided[idx]) {
            const base = idx * this.nodeCapacity;
            for (let i = 0; i < this.pointCount[idx]; i++) {
                const p = this.points[base + i];
                if (p.pos.x >= rx - rw && p.pos.x <= rx + rw &&
                    p.pos.y >= ry - rh && p.pos.y <= ry + rh) {
                    found.push(p);
                }
            }
        } else {
            this.query(this.nw[idx], rx, ry, rw, rh, found);
            this.query(this.ne[idx], rx, ry, rw, rh, found);
            this.query(this.sw[idx], rx, ry, rw, rh, found);
            this.query(this.se[idx], rx, ry, rw, rh, found);
        }

        return found;
    }

    /** Query reusing a pooled results array — avoids allocation per call. */
    queryReuse(idx, rx, ry, rw, rh) {
        this._queryBuf.length = 0;
        return this.query(idx, rx, ry, rw, rh, this._queryBuf);
    }

    build(bx, by, bw, bh, particles) {
        this.reset();
        const root = this.alloc(bx, by, bw, bh);
        for (const p of particles) this.insert(root, p);
        this.calculateMassDistribution(root);
        return root;
    }
}
