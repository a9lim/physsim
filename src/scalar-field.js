// ─── Scalar Field Base Class ───
// Shared grid infrastructure for dynamical scalar fields (Higgs, Axion).
// PQS (cubic B-spline, order 3) particle-grid coupling: 4×4 stencil,
// C² interpolation, C² continuous gradients (PQS-interpolated grid gradients).

import { EPSILON, FIELD_EXCITATION_SIGMA, MERGE_EXCITATION_SCALE } from './config.js';
import { TORUS, KLEIN, RP2 } from './topology.js';

// Boundary mode constants for inner-loop speed (avoid string comparison)
const BC_DESPAWN = 0;
const BC_BOUNCE = 1;
const BC_LOOP = 2;

export function bcFromString(mode) {
    return mode === 'loop' ? BC_LOOP : mode === 'bounce' ? BC_BOUNCE : BC_DESPAWN;
}

export default class ScalarField {
    constructor(gridSize, clampMax) {
        const gsq = gridSize * gridSize;
        this._grid = gridSize;
        this._gridSq = gsq;
        this._clampMax = clampMax;
        this._vacValue = 0; // Subclasses override: Higgs=1, Axion=0

        this.field = new Float64Array(gsq);
        this.fieldDot = new Float64Array(gsq);
        this._laplacian = new Float64Array(gsq);
        this._source = new Float64Array(gsq);

        // Pre-computed grid-point gradients (central differences, grid units)
        this._gradX = new Float64Array(gsq);
        this._gradY = new Float64Array(gsq);

        // PQS pre-allocated weight arrays (4 weights per axis)
        this._pqs = { ix: 0, iy: 0 };
        this._wx = new Float64Array(4);
        this._wy = new Float64Array(4);

        // Pre-allocated gradient and momentum outputs
        this._gradOut = { x: 0, y: 0 };
        this._momOut = { x: 0, y: 0 };

        // Offscreen canvas for rendering
        this.canvas = document.createElement('canvas');
        this.canvas.width = gridSize;
        this.canvas.height = gridSize;
        this._ctx = this.canvas.getContext('2d');
        this._imgData = this._ctx.createImageData(gridSize, gridSize);
    }

    reset(vacValue) {
        this.field.fill(vacValue);
        this.fieldDot.fill(0);
        this._gradX.fill(0);
        this._gradY.fill(0);
    }

    /**
     * Get grid index with boundary-aware wrapping.
     * Returns -1 for Dirichlet boundary (caller uses vacuum value).
     */
    _nb(nx, ny, bcMode, topoConst) {
        const GRID = this._grid;

        if (bcMode === BC_LOOP) {
            if (topoConst === TORUS) {
                if (nx < 0) nx += GRID; else if (nx >= GRID) nx -= GRID;
                if (ny < 0) ny += GRID; else if (ny >= GRID) ny -= GRID;
            } else if (topoConst === KLEIN) {
                if (nx < 0) nx += GRID; else if (nx >= GRID) nx -= GRID;
                if (ny < 0) { ny += GRID; nx = GRID - 1 - nx; }
                else if (ny >= GRID) { ny -= GRID; nx = GRID - 1 - nx; }
            } else { // RP2
                if (nx < 0) { nx += GRID; ny = GRID - 1 - ny; }
                else if (nx >= GRID) { nx -= GRID; ny = GRID - 1 - ny; }
                if (ny < 0) { ny += GRID; nx = GRID - 1 - nx; }
                else if (ny >= GRID) { ny -= GRID; nx = GRID - 1 - nx; }
            }
            return ny * GRID + nx;
        }

        if (bcMode === BC_BOUNCE) {
            // Neumann: clamp (zero-gradient)
            if (nx < 0) nx = 0; else if (nx >= GRID) nx = GRID - 1;
            if (ny < 0) ny = 0; else if (ny >= GRID) ny = GRID - 1;
            return ny * GRID + nx;
        }

        // Despawn: Dirichlet
        if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) return -1;
        return ny * GRID + nx;
    }

    /** Clamp-read field at grid coords. */
    _fieldAt(cx, cy) {
        const GRID = this._grid;
        if (cx < 0) cx = 0; else if (cx >= GRID) cx = GRID - 1;
        if (cy < 0) cy = 0; else if (cy >= GRID) cy = GRID - 1;
        return this.field[cy * GRID + cx];
    }

    /** Compute PQS (cubic B-spline) weights for position (x, y).
     *  Stores base index + fractional offset in _pqs, value weights in _wx/_wy.
     *  4×4 stencil: nodes [ix-1..ix+2] × [iy-1..iy+2].
     */
    _pqsCoords(x, y, invCellW, invCellH) {
        const gx = x * invCellW - 0.5;
        const gy = y * invCellH - 0.5;
        const p = this._pqs;
        p.ix = Math.floor(gx);
        p.iy = Math.floor(gy);

        const dx = gx - p.ix;
        const tx = 1 - dx;
        const dx2 = dx * dx;
        const dx3 = dx2 * dx;
        const wx = this._wx;
        wx[0] = tx * tx * tx / 6;
        wx[1] = (4 - 6 * dx2 + 3 * dx3) / 6;
        wx[2] = (1 + 3 * dx + 3 * dx2 - 3 * dx3) / 6;
        wx[3] = dx3 / 6;

        const dy = gy - p.iy;
        const ty = 1 - dy;
        const dy2 = dy * dy;
        const dy3 = dy2 * dy;
        const wy = this._wy;
        wy[0] = ty * ty * ty / 6;
        wy[1] = (4 - 6 * dy2 + 3 * dy3) / 6;
        wy[2] = (1 + 3 * dy + 3 * dy2 - 3 * dy3) / 6;
        wy[3] = dy3 / 6;
    }

    /** Deposit value at (x,y) into grid using PQS 4×4 stencil.
     *  Boundary-aware: wraps stencil nodes for periodic topologies.
     */
    _depositPQS(out, x, y, value, invCellW, invCellH, bcMode, topoConst) {
        this._pqsCoords(x, y, invCellW, invCellH);
        const { ix, iy } = this._pqs;
        const wx = this._wx;
        const wy = this._wy;
        for (let jy = 0; jy < 4; jy++) {
            const wyj = wy[jy];
            for (let jx = 0; jx < 4; jx++) {
                const idx = this._nb(ix + jx - 1, iy + jy - 1, bcMode, topoConst);
                if (idx < 0) continue; // Dirichlet boundary
                out[idx] += value * wx[jx] * wyj;
            }
        }
    }

    /** Compute discrete Laplacian with boundary conditions. */
    _computeLaplacian(bcMode, topoConst, invCellWSq, invCellHSq, vacValue) {
        const field = this.field;
        const lap = this._laplacian;
        const GRID = this._grid;
        const last = GRID - 1;

        // Interior cells: direct indexing (no _nb dispatch needed)
        for (let iy = 1; iy < last; iy++) {
            const row = iy * GRID;
            for (let ix = 1; ix < last; ix++) {
                const idx = row + ix;
                const fC = field[idx];
                lap[idx] = (field[idx - 1] + field[idx + 1] - 2 * fC) * invCellWSq
                         + (field[idx - GRID] + field[idx + GRID] - 2 * fC) * invCellHSq;
            }
        }

        // Border cells: use _nb for boundary-aware wrapping
        for (let iy = 0; iy < GRID; iy++) {
            for (let ix = 0; ix < GRID; ix++) {
                if (ix > 0 && ix < last && iy > 0 && iy < last) continue;
                const idx = iy * GRID + ix;
                const fC = field[idx];
                const iL = this._nb(ix - 1, iy, bcMode, topoConst);
                const iR = this._nb(ix + 1, iy, bcMode, topoConst);
                const iT = this._nb(ix, iy - 1, bcMode, topoConst);
                const iB = this._nb(ix, iy + 1, bcMode, topoConst);
                const fL = iL >= 0 ? field[iL] : vacValue;
                const fR = iR >= 0 ? field[iR] : vacValue;
                const fT = iT >= 0 ? field[iT] : vacValue;
                const fB = iB >= 0 ? field[iB] : vacValue;
                lap[idx] = (fL + fR - 2 * fC) * invCellWSq
                         + (fT + fB - 2 * fC) * invCellHSq;
            }
        }
    }

    /** Compute central-difference gradients at each grid point (grid units).
     *  Interior fast path + border path, same pattern as _computeLaplacian(). */
    _computeGridGradients(bcMode, topoConst, vacValue) {
        const field = this.field;
        const gx = this._gradX;
        const gy = this._gradY;
        const GRID = this._grid;
        const last = GRID - 1;

        // Interior cells: direct indexing (no _nb dispatch)
        for (let iy = 1; iy < last; iy++) {
            const row = iy * GRID;
            for (let ix = 1; ix < last; ix++) {
                const idx = row + ix;
                gx[idx] = (field[idx + 1] - field[idx - 1]) * 0.5;
                gy[idx] = (field[idx + GRID] - field[idx - GRID]) * 0.5;
            }
        }

        // Border cells: use _nb for boundary-aware wrapping
        for (let iy = 0; iy < GRID; iy++) {
            for (let ix = 0; ix < GRID; ix++) {
                if (ix > 0 && ix < last && iy > 0 && iy < last) continue;
                const idx = iy * GRID + ix;
                const iL = this._nb(ix - 1, iy, bcMode, topoConst);
                const iR = this._nb(ix + 1, iy, bcMode, topoConst);
                const iT = this._nb(ix, iy - 1, bcMode, topoConst);
                const iB = this._nb(ix, iy + 1, bcMode, topoConst);
                gx[idx] = ((iR >= 0 ? field[iR] : vacValue) - (iL >= 0 ? field[iL] : vacValue)) * 0.5;
                gy[idx] = ((iB >= 0 ? field[iB] : vacValue) - (iT >= 0 ? field[iT] : vacValue)) * 0.5;
            }
        }
    }

    /** PQS interpolation of field value at (x, y).
     *  Topology-aware via _nb() when bcMode/topoConst provided. */
    interpolate(x, y, invCellW, invCellH, bcMode, topoConst) {
        this._pqsCoords(x, y, invCellW, invCellH);
        const { ix, iy } = this._pqs;
        const wx = this._wx;
        const wy = this._wy;
        const vacVal = this._vacValue;
        let val = 0;
        for (let jy = 0; jy < 4; jy++) {
            const wyj = wy[jy];
            for (let jx = 0; jx < 4; jx++) {
                const idx = this._nb(ix + jx - 1, iy + jy - 1, bcMode, topoConst);
                val += (idx >= 0 ? this.field[idx] : vacVal) * wx[jx] * wyj;
            }
        }
        return val;
    }

    /** PQS-interpolated gradient at (x, y) from pre-computed grid gradients.
     *  Returns pre-allocated {x, y} or null on NaN.
     *  Topology-aware via _nb() when bcMode/topoConst provided.
     *  Uses standard PQS value weights on _gradX/_gradY arrays, giving C²
     *  continuous forces (vs C¹ from analytical B-spline derivatives). */
    gradient(x, y, invCellW, invCellH, bcMode, topoConst) {
        this._pqsCoords(x, y, invCellW, invCellH);
        const { ix, iy } = this._pqs;
        const wx = this._wx;
        const wy = this._wy;
        const gxArr = this._gradX;
        const gyArr = this._gradY;

        let gx = 0, gy = 0;
        for (let jy = 0; jy < 4; jy++) {
            const wyj = wy[jy];
            for (let jx = 0; jx < 4; jx++) {
                const idx = this._nb(ix + jx - 1, iy + jy - 1, bcMode, topoConst);
                const w = wx[jx] * wyj;
                gx += (idx >= 0 ? gxArr[idx] : 0) * w;
                gy += (idx >= 0 ? gyArr[idx] : 0) * w;
            }
        }

        const out = this._gradOut;
        out.x = gx * invCellW;
        out.y = gy * invCellH;
        if (out.x !== out.x || out.y !== out.y) return null;
        return out;
    }

    /** Deposit a Gaussian wave packet (field excitation / boson) at world (x,y).
     *  Energy goes into fieldDot so it propagates via the wave equation. */
    depositExcitation(x, y, energy, domainW, domainH) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;

        const gx = x / cellW;
        const gy = y / cellH;
        const sigma = FIELD_EXCITATION_SIGMA;
        const sigmaSq = sigma * sigma;
        const amplitude = MERGE_EXCITATION_SCALE * Math.sqrt(energy);
        const range = Math.ceil(3 * sigma);

        const ixMin = Math.max(0, Math.floor(gx) - range);
        const ixMax = Math.min(GRID - 1, Math.floor(gx) + range);
        const iyMin = Math.max(0, Math.floor(gy) - range);
        const iyMax = Math.min(GRID - 1, Math.floor(gy) + range);

        for (let iy = iyMin; iy <= iyMax; iy++) {
            const dy = iy - gy;
            for (let ix = ixMin; ix <= ixMax; ix++) {
                const dx = ix - gx;
                const rSq = dx * dx + dy * dy;
                if (rSq > 9 * sigmaSq) continue;
                this.fieldDot[iy * GRID + ix] += amplitude * Math.exp(-rSq / (2 * sigmaSq));
            }
        }
    }

    /** Shared field energy: KE + gradient + potential, integrated over grid.
     *  @param {number} domainW - world width
     *  @param {number} domainH - world height
     *  @param {function} potentialFn - V(fieldValue) for a single cell */
    _fieldEnergy(domainW, domainH, potentialFn) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return 0;
        const cellArea = cellW * cellH;
        const invCellWSq = 1 / (cellW * cellW);
        const invCellHSq = 1 / (cellH * cellH);
        const field = this.field;
        const fieldDot = this.fieldDot;
        let total = 0;

        for (let iy = 0; iy < GRID; iy++) {
            for (let ix = 0; ix < GRID; ix++) {
                const idx = iy * GRID + ix;
                const f = field[idx];
                const ke = 0.5 * fieldDot[idx] * fieldDot[idx];
                const fR = ix + 1 < GRID ? field[idx + 1] : f;
                const fB = iy + 1 < GRID ? field[idx + GRID] : f;
                const dfx = fR - f, dfy = fB - f;
                const gradE = 0.5 * (dfx * dfx * invCellWSq + dfy * dfy * invCellHSq);
                total += (ke + gradE + potentialFn(f)) * cellArea;
            }
        }

        return total === total ? total : 0;
    }

    /** Field momentum: P_i = -∫ φ̇ ∂_i φ dA (stress-energy T^{0i}). */
    momentum(domainW, domainH) {
        const out = this._momOut;
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) { out.x = 0; out.y = 0; return out; }
        const scaleX = cellH * 0.5;   // cellArea / (2·cellW)
        const scaleY = cellW * 0.5;   // cellArea / (2·cellH)
        const field = this.field;
        const fieldDot = this.fieldDot;
        let px = 0, py = 0;
        const last = GRID - 1;

        // Interior: centered differences, no bounds checks
        for (let iy = 1; iy < last; iy++) {
            const row = iy * GRID;
            for (let ix = 1; ix < last; ix++) {
                const idx = row + ix;
                const fd = fieldDot[idx];
                px -= fd * (field[idx + 1] - field[idx - 1]);
                py -= fd * (field[idx + GRID] - field[idx - GRID]);
            }
        }

        // Border: centered differences with clamping
        for (let iy = 0; iy < GRID; iy++) {
            for (let ix = 0; ix < GRID; ix++) {
                if (ix > 0 && ix < last && iy > 0 && iy < last) continue;
                const idx = iy * GRID + ix;
                const fd = fieldDot[idx];
                const f = field[idx];
                px -= fd * ((ix < last ? field[idx + 1] : f) - (ix > 0 ? field[idx - 1] : f));
                py -= fd * ((iy < last ? field[idx + GRID] : f) - (iy > 0 ? field[idx - GRID] : f));
            }
        }

        out.x = px * scaleX;
        out.y = py * scaleY;
        if (out.x !== out.x) { out.x = 0; out.y = 0; }
        return out;
    }

    /** Draw field overlay in world space. */
    draw(ctx, domainW, domainH) {
        ctx.save();
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.globalAlpha = 0.6;
        ctx.drawImage(this.canvas, 0, 0, domainW, domainH);
        ctx.globalAlpha = 1;
        ctx.restore();
    }
}
