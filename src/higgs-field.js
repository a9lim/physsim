// ─── Higgs Scalar Field ───
// Dynamical scalar field on a 2D grid with Mexican hat potential.
// V(phi) = -1/2 mu^2 phi^2 + 1/4 lambda phi^4  (VEV=1, lambda=mu^2=m_H^2/2)
// m_H is the free parameter (slider 0.25-1, default 0.5)
// Extends ScalarField for shared PQS infrastructure.

import { HIGGS_GRID, DEFAULT_HIGGS_MASS, HIGGS_PHI_MAX, EPSILON, kerrNewmanRadius } from './config.js';
import ScalarField, { bcFromString } from './scalar-field.js';

export default class HiggsField extends ScalarField {
    constructor() {
        super(HIGGS_GRID, HIGGS_PHI_MAX);
        this._thermal = new Float64Array(this._gridSq);
        this.mass = DEFAULT_HIGGS_MASS;
        this.reset();
    }

    reset() {
        super.reset(1); // VEV = 1
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

        // PQS source deposition
        const src = this._source;
        src.fill(0);
        this._depositSources(particles, invCellW, invCellH, bcMode, topoConst);
        const cellArea = cellW * cellH;
        const invCellArea = cellArea > EPSILON ? 1 / cellArea : 0;

        // Deposit thermal energy (drives phase transitions)
        const thermal = this._thermal;
        thermal.fill(0);
        this._depositThermal(particles, invCellW, invCellH, bcMode, topoConst);

        // Compute Laplacian (Dirichlet VEV=1)
        this._computeLaplacian(bcMode, topoConst, invCellWSq, invCellHSq, 1);

        // Kick fieldDot, then drift field (symplectic Euler)
        // VEV=1, λ = m_H²/2, μ² = m_H²/2, critical damping = 2*m_H
        const mH = this.mass;
        const muSq = 0.5 * mH * mH;
        const damp = 2 * mH;
        const lap = this._laplacian;

        for (let i = 0; i < GRID_SQ; i++) {
            const phiVal = field[i];

            // Thermal correction: μ²_eff = μ² - KE_local
            const muSqEff = muSq - thermal[i];

            // Klein-Gordon: d²φ/dt² = ∇²φ + μ²_eff·φ - λφ³ - damping·φ̇ + source
            const ddphi = lap[i]
                        + muSqEff * phiVal
                        - muSq * phiVal * phiVal * phiVal
                        - damp * fieldDot[i]
                        + src[i] * invCellArea;

            fieldDot[i] += ddphi * dt;
            const newPhi = field[i] + fieldDot[i] * dt;

            // Clamp field to prevent numerical blowup
            if (newPhi !== newPhi) { // NaN guard
                field[i] = 1;
                fieldDot[i] = 0;
            } else if (newPhi > HIGGS_PHI_MAX) {
                field[i] = HIGGS_PHI_MAX;
                fieldDot[i] = 0;
            } else if (newPhi < -HIGGS_PHI_MAX) {
                field[i] = -HIGGS_PHI_MAX;
                fieldDot[i] = 0;
            } else {
                field[i] = newPhi;
            }
        }
    }

    /** PQS deposition of particle baseMass as scalar source. */
    _depositSources(particles, invCellW, invCellH, bcMode, topoConst) {
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (p.baseMass < EPSILON) continue;
            this._depositPQS(this._source, p.pos.x, p.pos.y, p.baseMass, invCellW, invCellH, bcMode, topoConst);
        }
    }

    /** PQS deposition of local kinetic energy density. */
    _depositThermal(particles, invCellW, invCellH, bcMode, topoConst) {
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            const ke = 0.5 * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
            if (ke < EPSILON) continue;
            this._depositPQS(this._thermal, p.pos.x, p.pos.y, ke, invCellW, invCellH, bcMode, topoConst);
        }
    }

    /** Set particle effective masses: m = baseMass * |phi| (VEV=1).
     *  PQS interpolation is C² smooth — no self-force subtraction needed.
     */
    modulateMasses(particles, domainW, domainH, blackHoleEnabled) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (p.baseMass < EPSILON) continue;

            const phiLocal = this.interpolate(p.pos.x, p.pos.y, invCellW, invCellH);
            const newMass = Math.max(p.baseMass * Math.abs(phiLocal), EPSILON);
            if (newMass !== newMass) continue; // NaN guard
            p.mass = newMass;

            if (blackHoleEnabled) {
                p.radius = kerrNewmanRadius(p.mass, p.radiusSq, p.angVel, p.charge);
            } else {
                p.radius = Math.cbrt(p.mass);
            }
            p.radiusSq = p.radius * p.radius;
            p.invMass = 1 / p.mass;
        }
    }

    /** Apply gradient force: F = -baseMass * grad(phi).
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
            if (p.baseMass < EPSILON) continue;

            const grad = this.gradient(p.pos.x, p.pos.y, invCellW, invCellH);
            if (!grad) continue;

            const forceX = -p.baseMass * grad.x;
            const forceY = -p.baseMass * grad.y;

            p.force.x += forceX;
            p.force.y += forceY;
            p.forceHiggs.x += forceX;
            p.forceHiggs.y += forceY;
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
        const muSq = 0.5 * this.mass * this.mass;
        // V(VEV) = -½μ²·1² + ¼λ·1⁴ = -½μ² + ¼μ² = -¼μ², offset = +¼μ²
        const vacOffset = 0.25 * muSq;
        let total = 0;

        for (let iy = 0; iy < GRID; iy++) {
            for (let ix = 0; ix < GRID; ix++) {
                const idx = iy * GRID + ix;
                const p = field[idx];

                // KE: 1/2 (dphi/dt)^2
                const ke = 0.5 * fieldDot[idx] * fieldDot[idx];

                // Gradient energy: 1/2 |grad phi|^2
                const phiR = ix + 1 < GRID ? field[idx + 1] : p;
                const phiB = iy + 1 < GRID ? field[idx + GRID] : p;
                const dxPhi = (phiR - p) * invCellW;
                const dyPhi = (phiB - p) * invCellH;
                const gradE = 0.5 * (dxPhi * dxPhi + dyPhi * dyPhi);

                // V(φ) = -½μ²φ² + ¼λφ⁴ + vacOffset  (λ=μ², shifted so V(1)=0)
                const pot = -0.5 * muSq * p * p + 0.25 * muSq * p * p * p * p + vacOffset;

                total += (ke + gradE + pot) * cellArea;
            }
        }

        return total === total ? total : 0; // NaN guard
    }

    /** Render field deviation from VEV=1 to offscreen canvas. */
    render(isLight) {
        const field = this.field;
        const data = this._imgData.data;
        const GRID_SQ = this._gridSq;

        for (let i = 0; i < GRID_SQ; i++) {
            const deviation = field[i] - 1;
            const intensity = Math.min(Math.abs(deviation) * 2, 1.0);
            const alpha = intensity * (isLight ? 60 : 80);
            const idx = i * 4;

            if (deviation < 0) {
                // Depleted (below VEV) -> magenta #B4689C (180, 104, 156)
                data[idx] = 180; data[idx + 1] = 104; data[idx + 2] = 156;
            } else {
                // Enhanced (above VEV) -> cyan #4AACA0 (74, 172, 160)
                data[idx] = 74; data[idx + 1] = 172; data[idx + 2] = 160;
            }
            data[idx + 3] = alpha;
        }
        this._ctx.putImageData(this._imgData, 0, 0);
    }
}
