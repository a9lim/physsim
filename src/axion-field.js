// ─── Axion Scalar Field ───
// Dynamical scalar field (ALP) on a 2D grid with quadratic potential.
// V(a) = 1/2 m_a^2 a^2  (no symmetry breaking — field oscillates around a=0)
// m_a is the free parameter (slider 0.01-0.25, default 0.05)
// Extends ScalarField for shared PQS infrastructure.
//
// Uses the scalar coupling L_int = -(1+g·a)F²/4, which makes α position-dependent:
//   α_eff(x) = α·(1 + a(x))
// The QCD axion's pseudoscalar aFF̃ ∝ E·B coupling vanishes in 2D (E in-plane, B⊥).
// The scalar aF² coupling is the simplest ALP interaction that works in 2D and
// correctly modifies all EM forces via a local coupling constant.
//
// Source: g·q² (regularized EM self-energy from aF² vertex, g = AXION_COUPLING).
// Gradient force: F = -g·q²·∇a. EM modulation: α_eff = α·(1 + g·a).

import { SCALAR_GRID, SCALAR_FIELD_MAX, DEFAULT_AXION_MASS, AXION_COUPLING, EPSILON } from './config.js';
import ScalarField, { bcFromString } from './scalar-field.js';

// Parse overlay colors from shared palette at module load (0-255 ints)
const _ph = window._parseHex; // hex -> [r,g,b] in 0–1
const _posRGB = _ph(window._PALETTE.extended.blue).map(v => (v * 255 + 0.5) | 0);
const _negRGB = _ph(window._PALETTE.extended.red).map(v => (v * 255 + 0.5) | 0);

export default class AxionField extends ScalarField {
    constructor() {
        super(SCALAR_GRID, SCALAR_FIELD_MAX);
        this.mass = DEFAULT_AXION_MASS;
        this.reset();
    }

    reset() {
        super.reset(0); // vacuum = 0
    }

    /** Evolve field one timestep using symplectic Euler (kick-drift). */
    update(dt, particles, boundaryMode, topoConst, domainW, domainH) {
        if (dt <= 0) return;
        const field = this.field;
        const fieldDot = this.fieldDot;
        const GRID = this._grid;
        const GRID_SQ = this._gridSq;

        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellWSq = 1 / (cellW * cellW);
        const invCellHSq = 1 / (cellH * cellH);
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;

        const bcMode = bcFromString(boundaryMode);

        // PQS source deposition: charged particles source the field
        const src = this._source;
        src.fill(0);
        this._depositSources(particles, invCellW, invCellH, bcMode, topoConst);
        const cellArea = cellW * cellH;
        const invCellArea = cellArea > EPSILON ? 1 / cellArea : 0;

        // Compute Laplacian (Dirichlet a=0)
        this._computeLaplacian(bcMode, topoConst, invCellWSq, invCellHSq, 0);

        // Kick fieldDot, then drift field (symplectic Euler)
        // Klein-Gordon: d²a/dt² = ∇²a - m_a²·a - g·m_a·ȧ + source  (ζ = g/2, Q = 1/g)
        const mA = this.mass;
        const mASq = mA * mA;
        const damp = AXION_COUPLING * mA; // Q = 1/g = 5, so g·Q = 1 (resonant buildup ≈ static response)
        const lap = this._laplacian;

        for (let i = 0; i < GRID_SQ; i++) {
            const ddA = lap[i]
                      - mASq * field[i]
                      - damp * fieldDot[i]
                      + src[i] * invCellArea;

            fieldDot[i] += ddA * dt;
            const newA = field[i] + fieldDot[i] * dt;

            // Clamp field to prevent numerical blowup
            if (newA !== newA) { // NaN guard
                field[i] = 0;
                fieldDot[i] = 0;
            } else if (newA > SCALAR_FIELD_MAX) {
                field[i] = SCALAR_FIELD_MAX;
                fieldDot[i] = 0;
            } else if (newA < -SCALAR_FIELD_MAX) {
                field[i] = -SCALAR_FIELD_MAX;
                fieldDot[i] = 0;
            } else {
                field[i] = newA;
            }
        }
    }

    /** PQS deposition of g·q² as scalar source (g = AXION_COUPLING). */
    _depositSources(particles, invCellW, invCellH, bcMode, topoConst) {
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            const qSq = p.charge * p.charge;
            if (qSq < EPSILON) continue;
            this._depositPQS(this._source, p.pos.x, p.pos.y, AXION_COUPLING * qSq, invCellW, invCellH, bcMode, topoConst);
        }
    }

    /** Interpolate local axion field value at each particle position.
     *  Sets p.axMod = 1 + g·a(x) where g = AXION_COUPLING.
     */
    interpolateAxMod(particles, domainW, domainH) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) {
            for (let i = 0; i < particles.length; i++) particles[i].axMod = 1;
            return;
        }
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            const aLocal = this.interpolate(p.pos.x, p.pos.y, invCellW, invCellH);
            // axMod = 1 + g·a, clamped >= 0 (can screen EM to zero but never reverse)
            const ga = AXION_COUPLING * aLocal;
            p.axMod = ga > -1 ? 1 + ga : 0;
        }
    }

    /** Apply gradient force: F = -g·q² * grad(a) where g = AXION_COUPLING.
     *  PQS gradient weights (derivative of cubic B-spline) give C¹ continuous forces.
     */
    applyForces(particles, domainW, domainH) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            const qSq = p.charge * p.charge;
            if (qSq < EPSILON) continue;

            const grad = this.gradient(p.pos.x, p.pos.y, invCellW, invCellH);
            if (!grad) continue;

            const forceX = -AXION_COUPLING * qSq * grad.x;
            const forceY = -AXION_COUPLING * qSq * grad.y;

            p.force.x += forceX;
            p.force.y += forceY;
            p.forceAxion.x += forceX;
            p.forceAxion.y += forceY;
        }
    }

    /** Total field energy: KE + gradient + potential, integrated over grid. */
    energy(domainW, domainH) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return 0;
        const cellArea = cellW * cellH;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        const field = this.field;
        const fieldDot = this.fieldDot;
        const mASq = this.mass * this.mass;
        let total = 0;

        for (let iy = 0; iy < GRID; iy++) {
            for (let ix = 0; ix < GRID; ix++) {
                const idx = iy * GRID + ix;
                const aVal = field[idx];

                // KE: 1/2 (da/dt)^2
                const ke = 0.5 * fieldDot[idx] * fieldDot[idx];

                // Gradient energy: 1/2 |grad a|^2
                const aR = ix + 1 < GRID ? field[idx + 1] : aVal;
                const aB = iy + 1 < GRID ? field[idx + GRID] : aVal;
                const dxA = (aR - aVal) * invCellW;
                const dyA = (aB - aVal) * invCellH;
                const gradE = 0.5 * (dxA * dxA + dyA * dyA);

                // V(a) = 1/2 m_a² a² (no offset needed — V(0)=0)
                const pot = 0.5 * mASq * aVal * aVal;

                total += (ke + gradE + pot) * cellArea;
            }
        }

        return total === total ? total : 0; // NaN guard
    }

    /** Render field to offscreen canvas. Blue = positive, red = negative. */
    render(isLight) {
        const field = this.field;
        const data = this._imgData.data;
        const GRID_SQ = this._gridSq;

        for (let i = 0; i < GRID_SQ; i++) {
            const aVal = field[i];
            const intensity = Math.min(Math.abs(aVal) * 4, 1.0);
            const alpha = intensity * (isLight ? 60 : 80);
            const idx = i * 4;
            const rgb = aVal > 0 ? _posRGB : _negRGB;

            data[idx] = rgb[0]; data[idx + 1] = rgb[1]; data[idx + 2] = rgb[2];
            data[idx + 3] = alpha;
        }
        this._ctx.putImageData(this._imgData, 0, 0);
    }
}
