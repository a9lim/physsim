// ─── Barnes-Hut Quadtree ───
// SoA flat typed-array pool: pre-allocated, doubles on overflow, zero GC per frame.

const NONE = -1;

// C10: Module-level typed insert stacks — V8 can't optimise a JS array that holds
// both integers (node indices) and object references.  Two parallel stacks avoid
// the polymorphic slot problem entirely.
let _workNodeStack = new Int32Array(256); // node indices
let _workPartStack = new Array(256);      // particle references
let _workTop = 0;

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
        this.totalCount = new Float64Array(maxNodes);
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
        this.totalCount[idx] = 0;
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
        this.totalCount = copyF64(this.totalCount);
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
        // C1: Unrolled — eliminates 3 temporary array allocations (ids/xs/ys) and loop.
        // NW = (x-hw, y-hh), NE = (x+hw, y-hh), SW = (x-hw, y+hh), SE = (x+hw, y+hh)
        this.count = c + 4;

        // NW (c)
        this.bx[c] = x - hw; this.by[c] = y - hh;
        this.bw[c] = hw; this.bh[c] = hh;
        this.totalMass[c] = 0; this.totalCharge[c] = 0; this.totalCount[c] = 0;
        this.totalMagneticMoment[c] = 0; this.totalAngularMomentum[c] = 0;
        this.totalMomentumX[c] = 0; this.totalMomentumY[c] = 0;
        this.comX[c] = x - hw; this.comY[c] = y - hh;
        this.nw[c] = NONE; this.ne[c] = NONE; this.sw[c] = NONE; this.se[c] = NONE;
        this.pointCount[c] = 0; this.divided[c] = 0;

        // NE (c+1)
        this.bx[c+1] = x + hw; this.by[c+1] = y - hh;
        this.bw[c+1] = hw; this.bh[c+1] = hh;
        this.totalMass[c+1] = 0; this.totalCharge[c+1] = 0; this.totalCount[c+1] = 0;
        this.totalMagneticMoment[c+1] = 0; this.totalAngularMomentum[c+1] = 0;
        this.totalMomentumX[c+1] = 0; this.totalMomentumY[c+1] = 0;
        this.comX[c+1] = x + hw; this.comY[c+1] = y - hh;
        this.nw[c+1] = NONE; this.ne[c+1] = NONE; this.sw[c+1] = NONE; this.se[c+1] = NONE;
        this.pointCount[c+1] = 0; this.divided[c+1] = 0;

        // SW (c+2)
        this.bx[c+2] = x - hw; this.by[c+2] = y + hh;
        this.bw[c+2] = hw; this.bh[c+2] = hh;
        this.totalMass[c+2] = 0; this.totalCharge[c+2] = 0; this.totalCount[c+2] = 0;
        this.totalMagneticMoment[c+2] = 0; this.totalAngularMomentum[c+2] = 0;
        this.totalMomentumX[c+2] = 0; this.totalMomentumY[c+2] = 0;
        this.comX[c+2] = x - hw; this.comY[c+2] = y + hh;
        this.nw[c+2] = NONE; this.ne[c+2] = NONE; this.sw[c+2] = NONE; this.se[c+2] = NONE;
        this.pointCount[c+2] = 0; this.divided[c+2] = 0;

        // SE (c+3)
        this.bx[c+3] = x + hw; this.by[c+3] = y + hh;
        this.bw[c+3] = hw; this.bh[c+3] = hh;
        this.totalMass[c+3] = 0; this.totalCharge[c+3] = 0; this.totalCount[c+3] = 0;
        this.totalMagneticMoment[c+3] = 0; this.totalAngularMomentum[c+3] = 0;
        this.totalMomentumX[c+3] = 0; this.totalMomentumY[c+3] = 0;
        this.comX[c+3] = x + hw; this.comY[c+3] = y + hh;
        this.nw[c+3] = NONE; this.ne[c+3] = NONE; this.sw[c+3] = NONE; this.se[c+3] = NONE;
        this.pointCount[c+3] = 0; this.divided[c+3] = 0;

        this.nw[idx] = c; this.ne[idx] = c + 1;
        this.sw[idx] = c + 2; this.se[idx] = c + 3;
        this.divided[idx] = 1;
    }

    // P1: Direct quadrant child selection — 2 comparisons instead of up to 4 _contains checks
    _childFor(idx, px, py) {
        return py <= this.by[idx]
            ? (px <= this.bx[idx] ? this.nw[idx] : this.ne[idx])
            : (px <= this.bx[idx] ? this.sw[idx] : this.se[idx]);
    }

    // P7: Iterative insert — eliminates recursive stack frames (up to depth 48)
    // C10: Uses module-level typed stacks (_workNodeStack / _workPartStack) so V8 sees
    //      a monomorphic Int32Array for indices and a stable object array for particles,
    //      rather than a single polymorphic JS array alternating both types.
    insert(idx, particle) {
        if (!this._contains(idx, particle.pos.x, particle.pos.y)) return false;
        const cap = this.nodeCapacity;

        // Grow module-level stacks if needed (rare — only on very large trees)
        if (_workNodeStack.length < this.maxNodes * cap + 8) {
            const newCap = (this.maxNodes * cap + 8) * 2;
            const nn = new Int32Array(newCap);
            nn.set(_workNodeStack);
            _workNodeStack = nn;
            const np = new Array(newCap);
            for (let i = 0; i < _workPartStack.length; i++) np[i] = _workPartStack[i];
            _workPartStack = np;
        }

        // Seed the work stack with the initial (node, particle) pair
        _workTop = 0;
        _workNodeStack[_workTop] = idx;
        _workPartStack[_workTop] = particle;
        _workTop++;

        while (_workTop > 0) {
            _workTop--;
            let nodeIdx = _workNodeStack[_workTop];
            let pt     = _workPartStack[_workTop];
            const px = pt.pos.x, py = pt.pos.y;
            let depth = 0;

            while (depth <= 48) {
                if (this.pointCount[nodeIdx] < cap && !this.divided[nodeIdx]) {
                    this.points[nodeIdx * cap + this.pointCount[nodeIdx]] = pt;
                    this.pointCount[nodeIdx]++;
                    break;
                }
                if (!this.divided[nodeIdx]) {
                    this._subdivide(nodeIdx);
                    const base = nodeIdx * cap;
                    for (let i = 0; i < this.pointCount[nodeIdx]; i++) {
                        const p = this.points[base + i];
                        _workNodeStack[_workTop] = this._childFor(nodeIdx, p.pos.x, p.pos.y);
                        _workPartStack[_workTop] = p;
                        _workTop++;
                    }
                    this.pointCount[nodeIdx] = 0;
                }
                nodeIdx = this._childFor(nodeIdx, px, py);
                depth++;
            }
            // depth guard: particle accepted at max depth (pointCount not incremented,
            // but this prevents infinite loops for co-located particles)
        }
        return true;
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

    /** Lightweight mass + charge distribution for boson trees.
     *  Reads p._srcMass (gravitational mass) and p._srcCharge from stored points. */
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
                let mass = 0, charge = 0, cx = 0, cy = 0;
                const base = idx * this.nodeCapacity;
                for (let i = 0; i < cnt; i++) {
                    const p = this.points[base + i];
                    const gm = p._srcMass;
                    mass += gm;
                    charge += p._srcCharge;
                    cx += p.pos.x * gm;
                    cy += p.pos.y * gm;
                }
                this.totalMass[idx] = mass;
                this.totalCharge[idx] = charge;
                this.totalCount[idx] = cnt;
                if (mass > 0) { this.comX[idx] = cx / mass; this.comY[idx] = cy / mass; }
            } else {
                const c0 = this.nw[idx], c1 = this.ne[idx], c2 = this.sw[idx], c3 = this.se[idx];
                const m0 = this.totalMass[c0], m1 = this.totalMass[c1], m2 = this.totalMass[c2], m3 = this.totalMass[c3];
                const mass = m0 + m1 + m2 + m3;
                this.totalMass[idx] = mass;
                this.totalCharge[idx] = this.totalCharge[c0] + this.totalCharge[c1] + this.totalCharge[c2] + this.totalCharge[c3];
                this.totalCount[idx] = this.totalCount[c0] + this.totalCount[c1] + this.totalCount[c2] + this.totalCount[c3];
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
        // M5: Indexed loop — avoids iterator protocol overhead of for...of
        for (let i = 0, n = particles.length; i < n; i++) this.insert(root, particles[i]);
        this.calculateMassDistribution(root);
        return root;
    }
}
