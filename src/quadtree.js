// ─── Barnes-Hut Quadtree ───
// SoA flat typed-array pool: pre-allocated, doubles on overflow, zero GC per frame.

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
        // Skip null-fill: pointCount[idx]=0 means no reads past base
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
        // P8: Single capacity check before 4 allocations
        if (this.count + 4 > this.maxNodes) this._grow();
        const c = this.count;
        // Inline alloc without per-call capacity check
        this.count = c + 4;
        const ids = [c, c + 1, c + 2, c + 3];
        const xs = [x - hw, x + hw, x - hw, x + hw];
        const ys = [y - hh, y - hh, y + hh, y + hh];
        for (let k = 0; k < 4; k++) {
            const id = ids[k];
            this.bx[id] = xs[k]; this.by[id] = ys[k];
            this.bw[id] = hw; this.bh[id] = hh;
            this.totalMass[id] = 0; this.totalCharge[id] = 0;
            this.totalMagneticMoment[id] = 0; this.totalAngularMomentum[id] = 0;
            this.totalMomentumX[id] = 0; this.totalMomentumY[id] = 0;
            this.comX[id] = xs[k]; this.comY[id] = ys[k];
            this.nw[id] = NONE; this.ne[id] = NONE;
            this.sw[id] = NONE; this.se[id] = NONE;
            this.pointCount[id] = 0; this.divided[id] = 0;
        }
        this.nw[idx] = ids[0]; this.ne[idx] = ids[1];
        this.sw[idx] = ids[2]; this.se[idx] = ids[3];
        this.divided[idx] = 1;
    }

    // P1: Direct quadrant child selection — 2 comparisons instead of up to 4 _contains checks
    _childFor(idx, px, py) {
        return py <= this.by[idx]
            ? (px <= this.bx[idx] ? this.nw[idx] : this.ne[idx])
            : (px <= this.bx[idx] ? this.sw[idx] : this.se[idx]);
    }

    // P7: Iterative insert — eliminates recursive stack frames (up to depth 48)
    insert(idx, particle) {
        if (!this._contains(idx, particle.pos.x, particle.pos.y)) return false;
        const cap = this.nodeCapacity;
        const px = particle.pos.x, py = particle.pos.y;
        let depth = 0;

        while (depth <= 48) {
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
                    this.insert(this._childFor(idx, p.pos.x, p.pos.y), p);
                }
                this.pointCount[idx] = 0;
            }
            idx = this._childFor(idx, px, py);
            depth++;
        }
        return true; // depth guard: accept particle at max depth
    }

    calculateMassDistribution(rootIdx) {
        // Iterative post-order traversal — avoids recursion overhead for deep trees
        // Phase 1: push all nodes depth-first; Phase 2: process in reverse (children before parents)
        const stack = this._massStack || (this._massStack = new Int32Array(512));
        const order = this._massOrder || (this._massOrder = new Int32Array(512));
        let stackTop = 0, orderLen = 0;

        // Grow if needed
        if (stack.length < this.maxNodes) {
            this._massStack = new Int32Array(this.maxNodes);
            this._massOrder = new Int32Array(this.maxNodes);
        }
        const s = this._massStack, o = this._massOrder;

        s[stackTop++] = rootIdx;
        while (stackTop > 0) {
            const idx = s[--stackTop];
            o[orderLen++] = idx;
            if (this.divided[idx]) {
                s[stackTop++] = this.nw[idx];
                s[stackTop++] = this.ne[idx];
                s[stackTop++] = this.sw[idx];
                s[stackTop++] = this.se[idx];
            }
        }

        // Process in reverse order (leaves first, then parents)
        for (let k = orderLen - 1; k >= 0; k--) {
            const idx = o[k];
            if (!this.divided[idx]) {
                const cnt = this.pointCount[idx];
                if (cnt === 0) continue;

                let mass = 0, charge = 0, magMom = 0, angMom = 0;
                let cx = 0, cy = 0, momX = 0, momY = 0;
                const base = idx * this.nodeCapacity;

                for (let i = 0; i < cnt; i++) {
                    const p = this.points[base + i];
                    const rSq = p.radiusSq;
                    const pm = p.mass;
                    mass += pm;
                    charge += p.charge;
                    magMom += p.magMoment;
                    angMom += p.angMomentum;
                    cx += p.pos.x * pm;
                    cy += p.pos.y * pm;
                    momX += pm * p.w.x;
                    momY += pm * p.w.y;
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
                const m0 = this.totalMass[c0], m1 = this.totalMass[c1], m2 = this.totalMass[c2], m3 = this.totalMass[c3];
                const mass = m0 + m1 + m2 + m3;
                this.totalMass[idx] = mass;
                this.totalCharge[idx] = this.totalCharge[c0] + this.totalCharge[c1] + this.totalCharge[c2] + this.totalCharge[c3];
                this.totalMagneticMoment[idx] = this.totalMagneticMoment[c0] + this.totalMagneticMoment[c1] + this.totalMagneticMoment[c2] + this.totalMagneticMoment[c3];
                this.totalAngularMomentum[idx] = this.totalAngularMomentum[c0] + this.totalAngularMomentum[c1] + this.totalAngularMomentum[c2] + this.totalAngularMomentum[c3];
                this.totalMomentumX[idx] = this.totalMomentumX[c0] + this.totalMomentumX[c1] + this.totalMomentumX[c2] + this.totalMomentumX[c3];
                this.totalMomentumY[idx] = this.totalMomentumY[c0] + this.totalMomentumY[c1] + this.totalMomentumY[c2] + this.totalMomentumY[c3];
                if (mass > 0) {
                    this.comX[idx] = (this.comX[c0] * m0 + this.comX[c1] * m1 + this.comX[c2] * m2 + this.comX[c3] * m3) / mass;
                    this.comY[idx] = (this.comY[c0] * m0 + this.comY[c1] * m1 + this.comY[c2] * m2 + this.comY[c3] * m3) / mass;
                }
            }
        }
    }

    /** Lightweight mass distribution for boson trees — only totalMass + CoM.
     *  Reads p._srcMass (source gravitational mass) from stored points. */
    calculateBosonDistribution(rootIdx) {
        const stack = this._massStack || (this._massStack = new Int32Array(512));
        const order = this._massOrder || (this._massOrder = new Int32Array(512));
        let stackTop = 0, orderLen = 0;
        if (stack.length < this.maxNodes) {
            this._massStack = new Int32Array(this.maxNodes);
            this._massOrder = new Int32Array(this.maxNodes);
        }
        const s = this._massStack, o = this._massOrder;

        s[stackTop++] = rootIdx;
        while (stackTop > 0) {
            const idx = s[--stackTop];
            o[orderLen++] = idx;
            if (this.divided[idx]) {
                s[stackTop++] = this.nw[idx];
                s[stackTop++] = this.ne[idx];
                s[stackTop++] = this.sw[idx];
                s[stackTop++] = this.se[idx];
            }
        }

        for (let k = orderLen - 1; k >= 0; k--) {
            const idx = o[k];
            if (!this.divided[idx]) {
                const cnt = this.pointCount[idx];
                if (cnt === 0) continue;
                let mass = 0, cx = 0, cy = 0;
                const base = idx * this.nodeCapacity;
                for (let i = 0; i < cnt; i++) {
                    const p = this.points[base + i];
                    const gm = p._srcMass;
                    mass += gm;
                    cx += p.pos.x * gm;
                    cy += p.pos.y * gm;
                }
                this.totalMass[idx] = mass;
                if (mass > 0) { this.comX[idx] = cx / mass; this.comY[idx] = cy / mass; }
            } else {
                const c0 = this.nw[idx], c1 = this.ne[idx], c2 = this.sw[idx], c3 = this.se[idx];
                const m0 = this.totalMass[c0], m1 = this.totalMass[c1], m2 = this.totalMass[c2], m3 = this.totalMass[c3];
                const mass = m0 + m1 + m2 + m3;
                this.totalMass[idx] = mass;
                if (mass > 0) {
                    this.comX[idx] = (this.comX[c0] * m0 + this.comX[c1] * m1 + this.comX[c2] * m2 + this.comX[c3] * m3) / mass;
                    this.comY[idx] = (this.comY[c0] * m0 + this.comY[c1] * m1 + this.comY[c2] * m2 + this.comY[c3] * m3) / mass;
                }
            }
        }
    }

    query(idx, rx, ry, rw, rh, found) {
        if (!found) found = [];
        // Iterative query to avoid recursion overhead
        const qStack = this._qStack || (this._qStack = new Int32Array(256));
        if (qStack.length < this.maxNodes) this._qStack = new Int32Array(this.maxNodes);
        const qs = this._qStack;
        let top = 0;
        qs[top++] = idx;

        while (top > 0) {
            const ni = qs[--top];
            if (!this._intersects(ni, rx, ry, rw, rh)) continue;

            if (!this.divided[ni]) {
                const base = ni * this.nodeCapacity;
                const cnt = this.pointCount[ni];
                for (let i = 0; i < cnt; i++) {
                    const p = this.points[base + i];
                    if (p.pos.x >= rx - rw && p.pos.x <= rx + rw &&
                        p.pos.y >= ry - rh && p.pos.y <= ry + rh) {
                        found.push(p);
                    }
                }
            } else {
                qs[top++] = this.nw[ni];
                qs[top++] = this.ne[ni];
                qs[top++] = this.sw[ni];
                qs[top++] = this.se[ni];
            }
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
