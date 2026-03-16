// ─── Scalar Field Base Class ───
// Shared grid infrastructure for dynamical scalar fields (Higgs, Axion).
// PQS (cubic B-spline, order 3) particle-grid coupling: 4×4 stencil,
// C² interpolation, C² continuous gradients (PQS-interpolated grid gradients).

import { EPSILON, FIELD_EXCITATION_SIGMA, MERGE_EXCITATION_SCALE, EXCITATION_MAX_AMPLITUDE, BOUND_BOUNCE, BOUND_LOOP, TORUS, KLEIN, RP2, SELFGRAV_GRID } from './config.js';
import { minImage } from './topology.js';

// Zero-alloc output for minImage()
const _miOut = { x: 0, y: 0 };

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
        this._energyDensity = new Float64Array(gsq);

        // Pre-computed grid-point gradients (central differences, grid units)
        this._gradX = new Float64Array(gsq);
        this._gradY = new Float64Array(gsq);

        // Self-gravity: coarse grid (SG×SG) for O(SG⁴) potential, upsampled to full grid
        const sgGrid = SELFGRAV_GRID;
        this._sgGrid = sgGrid;
        this._sgGridSq = sgGrid * sgGrid;
        this._sgRatio = gridSize / sgGrid;
        this._sgRho = new Float64Array(sgGrid * sgGrid);
        this._sgPhi = new Float64Array(sgGrid * sgGrid);
        this._sgPhiFull = new Float64Array(gsq);
        this._sgGradX = new Float64Array(gsq);
        this._sgGradY = new Float64Array(gsq);

        // PQS pre-allocated weight arrays (4 weights per axis)
        this._pqs = { ix: 0, iy: 0 };
        this._wx = new Float64Array(4);
        this._wy = new Float64Array(4);

        // Pre-allocated gradient, momentum, and fused interpolate+gradient outputs
        this._gradOut = { x: 0, y: 0 };
        this._momOut = { x: 0, y: 0 };
        this._iwgResult = { value: 0, grad: null };

        // Cached 1/sqrt(r²+ε) table for coarse self-gravity potential (SG⁴ entries)
        this._sgInvR = new Float32Array(sgGrid * sgGrid * sgGrid * sgGrid);
        this._sgLastCellW = 0;
        this._sgLastCellH = 0;
        this._sgLastSoftSq = 0;
        this._sgLastPeriodic = false;
        this._sgLastTopology = 0;

        // Pre-computed upsample x-mapping tables (GRID entries)
        this._fineXcx0 = new Uint8Array(gridSize);
        this._fineXcx1 = new Uint8Array(gridSize);
        this._fineXwx = new Float32Array(gridSize);
        this._upsampleXBuilt = false;
        // Numerical viscosity buffer (ν·∇²(ȧ) per cell)
        this._viscBuf = new Float64Array(gsq);

        // M10: Cached flag for applyGravForces early exit
        this._hasEnergy = false;

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

        if (bcMode === BOUND_LOOP) {
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

        if (bcMode === BOUND_BOUNCE) {
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
        const GRID = this._grid;

        // Interior fast path: stencil [ix-1..ix+2]×[iy-1..iy+2] fully inside grid
        if (ix >= 1 && ix + 2 < GRID && iy >= 1 && iy + 2 < GRID) {
            for (let jy = 0; jy < 4; jy++) {
                const row = (iy + jy - 1) * GRID + (ix - 1);
                const vwy = value * wy[jy];
                out[row]     += vwy * wx[0];
                out[row + 1] += vwy * wx[1];
                out[row + 2] += vwy * wx[2];
                out[row + 3] += vwy * wx[3];
            }
            return;
        }

        // Border path: use _nb for boundary-aware wrapping
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

        // Border cells: explicit edge traversal (~248 cells vs 4096)
        const _lapBorder = (ix, iy) => {
            const idx = iy * GRID + ix;
            const fC = field[idx];
            const iL = this._nb(ix - 1, iy, bcMode, topoConst);
            const iR = this._nb(ix + 1, iy, bcMode, topoConst);
            const iT = this._nb(ix, iy - 1, bcMode, topoConst);
            const iB = this._nb(ix, iy + 1, bcMode, topoConst);
            lap[idx] = ((iL >= 0 ? field[iL] : vacValue) + (iR >= 0 ? field[iR] : vacValue) - 2 * fC) * invCellWSq
                     + ((iT >= 0 ? field[iT] : vacValue) + (iB >= 0 ? field[iB] : vacValue) - 2 * fC) * invCellHSq;
        };
        for (let ix = 0; ix < GRID; ix++) { _lapBorder(ix, 0); _lapBorder(ix, last); }
        for (let iy = 1; iy < last; iy++) { _lapBorder(0, iy); _lapBorder(last, iy); }
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

        // Border cells: explicit edge traversal (~248 cells vs 4096)
        const _gradBorder = (ix, iy) => {
            const idx = iy * GRID + ix;
            const iL = this._nb(ix - 1, iy, bcMode, topoConst);
            const iR = this._nb(ix + 1, iy, bcMode, topoConst);
            const iT = this._nb(ix, iy - 1, bcMode, topoConst);
            const iB = this._nb(ix, iy + 1, bcMode, topoConst);
            gx[idx] = ((iR >= 0 ? field[iR] : vacValue) - (iL >= 0 ? field[iL] : vacValue)) * 0.5;
            gy[idx] = ((iB >= 0 ? field[iB] : vacValue) - (iT >= 0 ? field[iT] : vacValue)) * 0.5;
        };
        for (let ix = 0; ix < GRID; ix++) { _gradBorder(ix, 0); _gradBorder(ix, last); }
        for (let iy = 1; iy < last; iy++) { _gradBorder(0, iy); _gradBorder(last, iy); }
    }

    /** Compute numerical viscosity: ν·∇²(ȧ) where ν = 1/(2√(1/dx²+1/dy²)).
     *  Gives Q=1 at Nyquist frequency, vanishes for physical (long-wavelength) modes.
     *  Clamp-to-edge at borders. Interior fast path + border path. */
    _computeViscosity(invCellWSq, invCellHSq) {
        const fd = this.fieldDot;
        const out = this._viscBuf;
        const GRID = this._grid;
        const last = GRID - 1;
        const visc = 0.5 / Math.sqrt(invCellWSq + invCellHSq);

        for (let iy = 1; iy < last; iy++) {
            const row = iy * GRID;
            for (let ix = 1; ix < last; ix++) {
                const i = row + ix;
                const fdC = fd[i];
                out[i] = visc * ((fd[i - 1] + fd[i + 1] - 2 * fdC) * invCellWSq
                               + (fd[i - GRID] + fd[i + GRID] - 2 * fdC) * invCellHSq);
            }
        }

        const _viscBorder = (ix, iy) => {
            const i = iy * GRID + ix;
            const fdC = fd[i];
            const fdL = ix > 0 ? fd[i - 1] : fdC;
            const fdR = ix < last ? fd[i + 1] : fdC;
            const fdT = iy > 0 ? fd[i - GRID] : fdC;
            const fdB = iy < last ? fd[i + GRID] : fdC;
            out[i] = visc * ((fdL + fdR - 2 * fdC) * invCellWSq
                           + (fdT + fdB - 2 * fdC) * invCellHSq);
        };
        for (let ix = 0; ix < GRID; ix++) { _viscBorder(ix, 0); _viscBorder(ix, last); }
        for (let iy = 1; iy < last; iy++) { _viscBorder(0, iy); _viscBorder(last, iy); }
    }

    /** PQS interpolation of field value at (x, y).
     *  Topology-aware via _nb() when bcMode/topoConst provided. */
    interpolate(x, y, invCellW, invCellH, bcMode, topoConst) {
        this._pqsCoords(x, y, invCellW, invCellH);
        const { ix, iy } = this._pqs;
        const wx = this._wx;
        const wy = this._wy;
        const GRID = this._grid;
        let val = 0;
        // M4: Interior fast path — direct indexing avoids 16 _nb() calls
        if (ix >= 1 && ix <= GRID - 3 && iy >= 1 && iy <= GRID - 3) {
            for (let jy = 0; jy < 4; jy++) {
                const wyj = wy[jy];
                const row = (iy + jy - 1) * GRID + ix - 1;
                for (let jx = 0; jx < 4; jx++) {
                    val += this.field[row + jx] * wx[jx] * wyj;
                }
            }
        } else {
            const vacVal = this._vacValue;
            for (let jy = 0; jy < 4; jy++) {
                const wyj = wy[jy];
                for (let jx = 0; jx < 4; jx++) {
                    const idx = this._nb(ix + jx - 1, iy + jy - 1, bcMode, topoConst);
                    val += (idx >= 0 ? this.field[idx] : vacVal) * wx[jx] * wyj;
                }
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
        const GRID = this._grid;

        let gx = 0, gy = 0;
        // M4: Interior fast path — direct indexing avoids 16 _nb() calls
        if (ix >= 1 && ix <= GRID - 3 && iy >= 1 && iy <= GRID - 3) {
            for (let jy = 0; jy < 4; jy++) {
                const wyj = wy[jy];
                const row = (iy + jy - 1) * GRID + ix - 1;
                for (let jx = 0; jx < 4; jx++) {
                    const idx = row + jx;
                    const w = wx[jx] * wyj;
                    gx += gxArr[idx] * w;
                    gy += gyArr[idx] * w;
                }
            }
        } else {
            for (let jy = 0; jy < 4; jy++) {
                const wyj = wy[jy];
                for (let jx = 0; jx < 4; jx++) {
                    const idx = this._nb(ix + jx - 1, iy + jy - 1, bcMode, topoConst);
                    const w = wx[jx] * wyj;
                    gx += (idx >= 0 ? gxArr[idx] : 0) * w;
                    gy += (idx >= 0 ? gyArr[idx] : 0) * w;
                }
            }
        }

        const out = this._gradOut;
        out.x = gx * invCellW;
        out.y = gy * invCellH;
        if (out.x !== out.x || out.y !== out.y) return null;
        return out;
    }

    /** Fused PQS interpolation of field value + gradient in one stencil walk.
     *  Avoids redundant _pqsCoords() and _nb() calls when both are needed.
     *  Returns { value, grad } where grad is pre-allocated {x, y} or null on NaN. */
    interpolateWithGradient(x, y, invCellW, invCellH, bcMode, topoConst) {
        this._pqsCoords(x, y, invCellW, invCellH);
        const { ix, iy } = this._pqs;
        const wx = this._wx;
        const wy = this._wy;
        const gxArr = this._gradX;
        const gyArr = this._gradY;
        const field = this.field;
        const GRID = this._grid;

        let val = 0, gx = 0, gy = 0;
        // M4: Interior fast path — direct indexing avoids 16 _nb() calls
        if (ix >= 1 && ix <= GRID - 3 && iy >= 1 && iy <= GRID - 3) {
            for (let jy = 0; jy < 4; jy++) {
                const wyj = wy[jy];
                const row = (iy + jy - 1) * GRID + ix - 1;
                for (let jx = 0; jx < 4; jx++) {
                    const idx = row + jx;
                    const w = wx[jx] * wyj;
                    val += field[idx] * w;
                    gx += gxArr[idx] * w;
                    gy += gyArr[idx] * w;
                }
            }
        } else {
            const vacVal = this._vacValue;
            for (let jy = 0; jy < 4; jy++) {
                const wyj = wy[jy];
                for (let jx = 0; jx < 4; jx++) {
                    const idx = this._nb(ix + jx - 1, iy + jy - 1, bcMode, topoConst);
                    const w = wx[jx] * wyj;
                    if (idx >= 0) {
                        val += field[idx] * w;
                        gx += gxArr[idx] * w;
                        gy += gyArr[idx] * w;
                    } else {
                        val += vacVal * w;
                    }
                }
            }
        }

        const out = this._gradOut;
        out.x = gx * invCellW;
        out.y = gy * invCellH;
        if (out.x !== out.x || out.y !== out.y) {
            this._iwgResult.value = val;
            this._iwgResult.grad = null;
            return this._iwgResult;
        }
        this._iwgResult.value = val;
        this._iwgResult.grad = out;
        return this._iwgResult;
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
        const amplitude = Math.min(MERGE_EXCITATION_SCALE * Math.sqrt(energy), EXCITATION_MAX_AMPLITUDE);
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

    /** Subclasses override to add V(field) per cell to _energyDensity. */
    _addPotentialEnergy() {}

    /** Compute energy density per grid cell: ρ = ½φ̇² + ½|∇φ|² + V(φ).
     *  Stores result in _energyDensity. Requires _gradX/_gradY to be current
     *  (called at end of subclass update()). */
    _computeEnergyDensity(domainW, domainH) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellWSq = 1 / (cellW * cellW);
        const invCellHSq = 1 / (cellH * cellH);
        const rho = this._energyDensity;
        const fieldDot = this.fieldDot;
        const gx = this._gradX;
        const gy = this._gradY;

        for (let i = 0; i < this._gridSq; i++) {
            const fd = fieldDot[i];
            const gxi = gx[i], gyi = gy[i];
            rho[i] = 0.5 * fd * fd
                   + 0.5 * (gxi * gxi * invCellWSq + gyi * gyi * invCellHSq);
        }
        this._addPotentialEnergy();
    }

    /** Downsample energy density from full grid to coarse self-gravity grid. */
    _downsampleRho() {
        const GRID = this._grid;
        const SG = this._sgGrid;
        const ratio = this._sgRatio;
        const invBlock = 1 / (ratio * ratio);
        const rho = this._energyDensity;
        const coarse = this._sgRho;

        for (let cy = 0; cy < SG; cy++) {
            const baseY = cy * ratio;
            for (let cx = 0; cx < SG; cx++) {
                const baseX = cx * ratio;
                let sum = 0;
                for (let dy = 0; dy < ratio; dy++) {
                    const row = (baseY + dy) * GRID + baseX;
                    for (let dx = 0; dx < ratio; dx++) {
                        sum += rho[row + dx];
                    }
                }
                coarse[cy * SG + cx] = sum * invBlock;
            }
        }
    }

    /** Rebuild the SG⁴ 1/sqrt table when geometry changes. */
    _buildSGInvRTable(domainW, domainH, softeningSq, periodic, topology) {
        const SG = this._sgGrid;
        const cellW = domainW / SG;
        const cellH = domainH / SG;
        if (cellW === this._sgLastCellW && cellH === this._sgLastCellH &&
            softeningSq === this._sgLastSoftSq && periodic === this._sgLastPeriodic &&
            topology === this._sgLastTopology) return;
        this._sgLastCellW = cellW;
        this._sgLastCellH = cellH;
        this._sgLastSoftSq = softeningSq;
        this._sgLastPeriodic = periodic;
        this._sgLastTopology = topology;
        const halfDomW = domainW * 0.5;
        const halfDomH = domainH * 0.5;
        const table = this._sgInvR;
        const SG2 = SG * SG;
        for (let iy = 0; iy < SG; iy++) {
            const cy = (iy + 0.5) * cellH;
            for (let ix = 0; ix < SG; ix++) {
                const cx = (ix + 0.5) * cellW;
                const rowBase = (iy * SG + ix) * SG2;
                for (let jy = 0; jy < SG; jy++) {
                    const sy = (jy + 0.5) * cellH;
                    for (let jx = 0; jx < SG; jx++) {
                        let dx, dy;
                        if (periodic) {
                            minImage(cx, cy, (jx + 0.5) * cellW, sy, topology, domainW, domainH, halfDomW, halfDomH, _miOut);
                            dx = _miOut.x; dy = _miOut.y;
                        } else {
                            dx = (jx + 0.5) * cellW - cx;
                            dy = sy - cy;
                        }
                        table[rowBase + jy * SG + jx] = 1 / Math.sqrt(dx * dx + dy * dy + softeningSq);
                    }
                }
            }
        }
    }

    /** Direct O(SG⁴) gravitational potential on coarse grid: Φ = -Σ ρ·dA/r.
     *  Uses cached 1/sqrt table — sqrt-free inner loop during normal play.
     *  M8: Pre-builds sparse source list to skip empty cells in inner loop. */
    _computeCoarsePotential(domainW, domainH, softeningSq, periodic, topology) {
        this._buildSGInvRTable(domainW, domainH, softeningSq, periodic, topology);
        const SG = this._sgGrid;
        const SG2 = SG * SG;
        const cellArea = (domainW / SG) * (domainH / SG);
        const rho = this._sgRho;
        const phi = this._sgPhi;
        const table = this._sgInvR;

        // M8: Build sparse source list (indices with nonzero rho)
        if (!this._sgSparseIdx) this._sgSparseIdx = new Int32Array(SG2);
        if (!this._sgSparseRho) this._sgSparseRho = new Float64Array(SG2);
        let nSources = 0;
        for (let j = 0; j < SG2; j++) {
            if (rho[j] >= EPSILON) {
                this._sgSparseIdx[nSources] = j;
                this._sgSparseRho[nSources] = rho[j] * cellArea;
                nSources++;
            }
        }
        const sparseIdx = this._sgSparseIdx;
        const sparseRho = this._sgSparseRho;

        for (let i = 0; i < SG2; i++) {
            let pot = 0;
            const rowBase = i * SG2;
            for (let s = 0; s < nSources; s++) {
                pot -= sparseRho[s] * table[rowBase + sparseIdx[s]];
            }
            phi[i] = pot;
        }
    }

    /** Build pre-computed x-axis upsample mapping (only when ratio changes). */
    _buildUpsampleXTable() {
        const SG = this._sgGrid;
        const GRID = this._grid;
        const invRatio = 1 / this._sgRatio;
        const sgLast = SG - 1;
        for (let ix = 0; ix < GRID; ix++) {
            const fx = (ix + 0.5) * invRatio - 0.5;
            const cx0 = Math.max(0, Math.min(sgLast - 1, Math.floor(fx)));
            this._fineXcx0[ix] = cx0;
            this._fineXcx1[ix] = Math.min(sgLast, cx0 + 1);
            this._fineXwx[ix] = Math.max(0, fx - cx0);
        }
        this._upsampleXBuilt = true;
    }

    /** Bilinear upsample gravitational potential from coarse to full grid. */
    _upsamplePhi() {
        if (!this._upsampleXBuilt) this._buildUpsampleXTable();
        const SG = this._sgGrid;
        const GRID = this._grid;
        const invRatio = 1 / this._sgRatio;
        const coarse = this._sgPhi;
        const full = this._sgPhiFull;
        const sgLast = SG - 1;
        const cx0arr = this._fineXcx0;
        const cx1arr = this._fineXcx1;
        const wxArr = this._fineXwx;

        for (let iy = 0; iy < GRID; iy++) {
            const fy = (iy + 0.5) * invRatio - 0.5;
            const cy0 = Math.max(0, Math.min(sgLast - 1, Math.floor(fy)));
            const cy1 = Math.min(sgLast, cy0 + 1);
            const wy = Math.max(0, fy - cy0);
            const wy0 = 1 - wy;
            const rowBase = iy * GRID;
            const row0 = cy0 * SG;
            const row1 = cy1 * SG;

            for (let ix = 0; ix < GRID; ix++) {
                const cx0 = cx0arr[ix];
                const cx1 = cx1arr[ix];
                const wx = wxArr[ix];
                full[rowBase + ix] = wy0 * ((1 - wx) * coarse[row0 + cx0] + wx * coarse[row0 + cx1])
                                   + wy  * ((1 - wx) * coarse[row1 + cx0] + wx * coarse[row1 + cx1]);
            }
        }
    }

    /** Central-difference gradient of upsampled gravitational potential (clamp-to-edge). */
    _computeSelfGravGradients() {
        const GRID = this._grid;
        const phi = this._sgPhiFull;
        const gx = this._sgGradX;
        const gy = this._sgGradY;
        const last = GRID - 1;

        for (let iy = 1; iy < last; iy++) {
            const row = iy * GRID;
            for (let ix = 1; ix < last; ix++) {
                const idx = row + ix;
                gx[idx] = (phi[idx + 1] - phi[idx - 1]) * 0.5;
                gy[idx] = (phi[idx + GRID] - phi[idx - GRID]) * 0.5;
            }
        }

        // Border cells only (explicit edge traversal, ~248 cells vs 4096)
        const _sgBorderCell = (ix, iy) => {
            const idx = iy * GRID + ix;
            gx[idx] = ((ix < last ? phi[idx + 1] : phi[idx]) - (ix > 0 ? phi[idx - 1] : phi[idx])) * 0.5;
            gy[idx] = ((iy < last ? phi[idx + GRID] : phi[idx]) - (iy > 0 ? phi[idx - GRID] : phi[idx])) * 0.5;
        };
        for (let ix = 0; ix < GRID; ix++) { _sgBorderCell(ix, 0); _sgBorderCell(ix, last); }
        for (let iy = 1; iy < last; iy++) { _sgBorderCell(0, iy); _sgBorderCell(last, iy); }
    }

    /** Compute self-gravity potential and gradient from field energy density.
     *  Coarse grid O(SG⁴) direct summation, bilinear upsampled to full grid. */
    computeSelfGravity(domainW, domainH, softeningSq, periodic, topology) {
        this._computeEnergyDensity(domainW, domainH);

        // M10: Scan full grid for nonzero energy density — used by applyGravForces() to skip O(N×GRID²)
        this._hasEnergy = false;
        for (let i = 0, len = this._gridSq; i < len; i++) {
            if (this._energyDensity[i] >= EPSILON) { this._hasEnergy = true; break; }
        }

        this._downsampleRho();
        // Early exit: if coarse energy density is negligible, skip O(SG⁴) potential
        const rho = this._sgRho;
        const SG2 = this._sgGridSq;
        let maxRho = 0;
        for (let i = 0; i < SG2; i++) {
            const v = rho[i];
            if (v > maxRho) maxRho = v;
        }
        if (maxRho < EPSILON) {
            this._sgPhiFull.fill(0);
            this._sgGradX.fill(0);
            this._sgGradY.fill(0);
            return;
        }
        this._computeCoarsePotential(domainW, domainH, softeningSq, periodic, topology);
        this._upsamplePhi();
        this._computeSelfGravGradients();
    }

    /** Apply gravitational force from field energy density onto particles.
     *  Direct summation: F = m · Σ ρ(x_j) · dA · (x_j - x_i) / |x_j - x_i|³.
     *  Uses cached _energyDensity from prior computeSelfGravity() or update() call. */
    applyGravForces(particles, domainW, domainH, softeningSq, periodic, topology) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const cellArea = cellW * cellH;
        const halfDomW = domainW * 0.5;
        const halfDomH = domainH * 0.5;

        // Use cached energy density — already computed in update() → computeSelfGravity()
        // or bootstrap. Avoids redundant O(GRID²) recomputation per substep.
        const rho = this._energyDensity;

        // M10: Use cached _hasEnergy flag (set by computeSelfGravity/bootstrap) to
        // skip O(N×GRID²) without an O(GRID²) scan per substep
        if (!this._hasEnergy) return;

        // Pre-compute cell centers (avoids (ix+0.5)*cellW per particle in inner loop)
        const ccx = this._cellCX;
        const ccy = this._cellCY;
        if (!ccx || ccx.length !== GRID) {
            this._cellCX = new Float64Array(GRID);
            this._cellCY = new Float64Array(GRID);
            for (let i = 0; i < GRID; i++) {
                this._cellCX[i] = (i + 0.5) * cellW;
                this._cellCY[i] = (i + 0.5) * cellH;
            }
        } else if (Math.abs(ccx[0] - 0.5 * cellW) > EPSILON) {
            // Recompute if domain changed
            for (let i = 0; i < GRID; i++) {
                this._cellCX[i] = (i + 0.5) * cellW;
                this._cellCY[i] = (i + 0.5) * cellH;
            }
        }
        const cx_arr = this._cellCX;
        const cy_arr = this._cellCY;

        for (let pi = 0; pi < particles.length; pi++) {
            const p = particles[pi];
            if (p.mass < EPSILON) continue;
            let fx = 0, fy = 0;
            for (let iy = 0; iy < GRID; iy++) {
                const cy = cy_arr[iy];
                for (let ix = 0; ix < GRID; ix++) {
                    const rhoVal = rho[iy * GRID + ix];
                    if (rhoVal < EPSILON) continue;
                    const cx = cx_arr[ix];
                    let dx, dy;
                    if (periodic) {
                        minImage(p.pos.x, p.pos.y, cx, cy, topology, domainW, domainH, halfDomW, halfDomH, _miOut);
                        dx = _miOut.x; dy = _miOut.y;
                    } else {
                        dx = cx - p.pos.x;
                        dy = cy - p.pos.y;
                    }
                    const rSq = dx * dx + dy * dy + softeningSq;
                    const invR = 1 / Math.sqrt(rSq);
                    const fMag = rhoVal * cellArea * invR * invR * invR;
                    fx += dx * fMag;
                    fy += dy * fMag;
                }
            }
            const gfx = p.mass * fx;
            const gfy = p.mass * fy;
            p.forceGravity.x += gfx;
            p.forceGravity.y += gfy;
            p.force.x += gfx;
            p.force.y += gfy;
        }
    }

    /** Gravitational PE between particles and field energy density.
     *  PE = Σ_particles Σ_cells -m · ρ · dA / r. */
    gravPE(particles, domainW, domainH, softeningSq, periodic, topology) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return 0;
        const cellArea = cellW * cellH;
        const halfDomW = domainW * 0.5;
        const halfDomH = domainH * 0.5;
        const rho = this._energyDensity; // uses cached from last applyGravForces()
        let pe = 0;

        for (let pi = 0; pi < particles.length; pi++) {
            const p = particles[pi];
            if (p.mass < EPSILON) continue;
            for (let iy = 0; iy < GRID; iy++) {
                const cy = (iy + 0.5) * cellH;
                for (let ix = 0; ix < GRID; ix++) {
                    const rhoVal = rho[iy * GRID + ix];
                    if (rhoVal < EPSILON) continue;
                    const cx = (ix + 0.5) * cellW;
                    let dx, dy;
                    if (periodic) {
                        minImage(p.pos.x, p.pos.y, cx, cy, topology, domainW, domainH, halfDomW, halfDomH, _miOut);
                        dx = _miOut.x; dy = _miOut.y;
                    } else {
                        dx = cx - p.pos.x;
                        dy = cy - p.pos.y;
                    }
                    const rSq = dx * dx + dy * dy + softeningSq;
                    pe -= p.mass * rhoVal * cellArea / Math.sqrt(rSq);
                }
            }
        }
        return pe;
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

        // Border: centered differences with clamping (explicit edge traversal)
        const _momBorder = (ix, iy) => {
            const idx = iy * GRID + ix;
            const fd = fieldDot[idx];
            const f = field[idx];
            px -= fd * ((ix < last ? field[idx + 1] : f) - (ix > 0 ? field[idx - 1] : f));
            py -= fd * ((iy < last ? field[idx + GRID] : f) - (iy > 0 ? field[idx - GRID] : f));
        };
        for (let ix = 0; ix < GRID; ix++) { _momBorder(ix, 0); _momBorder(ix, last); }
        for (let iy = 1; iy < last; iy++) { _momBorder(0, iy); _momBorder(last, iy); }

        out.x = px * scaleX;
        out.y = py * scaleY;
        if (out.x !== out.x) { out.x = 0; out.y = 0; }
        return out;
    }

    /** Draw field overlay in world space. */
    draw(ctx, domainW, domainH) {
        // R7: Explicit property set/restore instead of save()/restore() (~30 property snapshot)
        const prevSmooth = ctx.imageSmoothingEnabled;
        const prevQuality = ctx.imageSmoothingQuality;
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.globalAlpha = 0.6;
        ctx.drawImage(this.canvas, 0, 0, domainW, domainH);
        ctx.globalAlpha = 1;
        ctx.imageSmoothingEnabled = prevSmooth;
        ctx.imageSmoothingQuality = prevQuality;
    }
}
