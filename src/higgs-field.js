// ─── Higgs Scalar Field ───
// Dynamical scalar field on a 2D grid with Mexican hat potential.
// V(phi) = -1/2 mu^2 phi^2 + 1/4 lambda phi^4  (VEV=1, lambda=mu^2=m_H^2/2)
// m_H is the free parameter (slider 0.25-1, default 0.5)
// Extends ScalarField for shared PQS infrastructure.

import { SCALAR_GRID, SCALAR_FIELD_MAX, DEFAULT_HIGGS_MASS, HIGGS_COUPLING, HIGGS_MASS_FLOOR, HIGGS_MASS_MAX_DELTA, HIGGS_AXION_COUPLING, SELFGRAV_PHI_MAX, EPSILON, BOUND_LOOP, kerrNewmanRadius } from './config.js';
import ScalarField from './scalar-field.js';

// Parse overlay colors from shared palette at module load (0-255 ints)
const _ph = window._parseHex; // hex -> [r,g,b] in 0–1
const _depletedRGB = _ph(window._PALETTE.extended.purple).map(v => (v * 255 + 0.5) | 0);
const _enhancedRGB = _ph(window._PALETTE.extended.lime).map(v => (v * 255 + 0.5) | 0);

export default class HiggsField extends ScalarField {
    constructor() {
        super(SCALAR_GRID, SCALAR_FIELD_MAX);
        this._vacValue = 1; // Higgs VEV = 1
        this._thermal = new Float64Array(this._gridSq);
        this.mass = DEFAULT_HIGGS_MASS;
        this.reset();
    }

    reset() {
        super.reset(1); // VEV = 1
    }

    /** C15: Fused energy density with Mexican hat potential V(φ) = μ²/4·(φ²−1)² in one loop. */
    _computeEnergyDensity(domainW, domainH) {
        const muSq = 0.5 * this.mass * this.mass;
        const vacOffset = 0.25 * muSq;
        super._computeEnergyDensity(domainW, domainH, phi => {
            const phiSq = phi * phi;
            return muSq * (-0.5 * phiSq + 0.25 * phiSq * phiSq) + vacOffset;
        });
    }

    /** Evolve field one timestep using Störmer-Verlet (kick-drift-kick, O(dt²)). */
    update(dt, particles, boundaryMode, topoConst, domainW, domainH, relativityEnabled, gravityEnabled = false, softeningSq = 64, otherField = null) {
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

        const bcMode = boundaryMode;

        // PQS source deposition
        const src = this._source;
        src.fill(0);
        this._depositSources(particles, invCellW, invCellH, bcMode, topoConst);
        const cellArea = cellW * cellH;
        const invCellArea = cellArea > EPSILON ? 1 / cellArea : 0;

        // Deposit thermal energy (drives phase transitions)
        const thermal = this._thermal;
        thermal.fill(0);
        this._depositThermal(particles, invCellW, invCellH, bcMode, topoConst, relativityEnabled);

        // Self-gravity: weak-field GR correction to Klein-Gordon equation
        // φ̈ = (1+4Φ)∇²φ + 2∇Φ·∇φ - (1+2Φ)V'(φ)
        // Correction: 4Φ·∇²φ + 2∇Φ·∇φ + 2Φ·(μ²_eff·φ - μ²·φ³)
        const sgOn = gravityEnabled;
        if (sgOn) this.computeSelfGravity(domainW, domainH, softeningSq, bcMode, topoConst);
        const sgFull = this._sgPhiFull;
        const sgGx = this._sgGradX;
        const sgGy = this._sgGradY;
        const fGx = this._gradX;
        const fGy = this._gradY;

        // Compute Laplacian (Dirichlet VEV=1)
        this._computeLaplacian(bcMode, topoConst, invCellWSq, invCellHSq, 1);

        // Störmer-Verlet: half-kick → drift → recompute Laplacian → second half-kick
        // VEV=1, λ = m_H²/2, μ² = m_H²/2, critical damping = 2*m_H
        const mH = this.mass;
        const muSq = 0.5 * mH * mH;
        const damp = 2 * mH;
        const lap = this._laplacian;
        const halfDt = dt * 0.5;

        // Portal coupling: V_portal = ½λφ²a², contributes -λa²φ to ddphi
        const portalArr = otherField ? otherField.field : null;

        // ── First half-kick ──
        this._computeViscosity(invCellWSq, invCellHSq);
        const visc = this._viscBuf;
        if (sgOn) {
            for (let i = 0; i < GRID_SQ; i++) {
                const phiVal = field[i];
                const muSqEff = muSq - thermal[i];
                const lapI = lap[i];
                const Phi = Math.max(-SELFGRAV_PHI_MAX, Math.min(SELFGRAV_PHI_MAX, sgFull[i]));
                const portalTerm = portalArr ? HIGGS_AXION_COUPLING * portalArr[i] * portalArr[i] : 0;
                const ddphi = lapI + muSqEff * phiVal - muSq * phiVal * phiVal * phiVal
                    - damp * fieldDot[i] + src[i] * invCellArea + visc[i]
                    + 4 * Phi * lapI
                    + 2 * (sgGx[i] * fGx[i] * invCellWSq + sgGy[i] * fGy[i] * invCellHSq)
                    + 2 * Phi * muSqEff * phiVal - 2 * Phi * muSq * phiVal * phiVal * phiVal
                    - portalTerm * phiVal - 2 * Phi * portalTerm * phiVal;
                fieldDot[i] += ddphi * halfDt;
                if (!isFinite(fieldDot[i])) { fieldDot[i] = 0; field[i] = 1; }
            }
        } else if (portalArr) {
            for (let i = 0; i < GRID_SQ; i++) {
                const phiVal = field[i];
                const ddphi = lap[i] + (muSq - thermal[i]) * phiVal
                    - muSq * phiVal * phiVal * phiVal
                    - damp * fieldDot[i] + src[i] * invCellArea + visc[i]
                    - HIGGS_AXION_COUPLING * portalArr[i] * portalArr[i] * phiVal;
                fieldDot[i] += ddphi * halfDt;
                if (!isFinite(fieldDot[i])) { fieldDot[i] = 0; field[i] = 1; }
            }
        } else {
            for (let i = 0; i < GRID_SQ; i++) {
                const phiVal = field[i];
                const ddphi = lap[i] + (muSq - thermal[i]) * phiVal
                    - muSq * phiVal * phiVal * phiVal
                    - damp * fieldDot[i] + src[i] * invCellArea + visc[i];
                fieldDot[i] += ddphi * halfDt;
                if (!isFinite(fieldDot[i])) { fieldDot[i] = 0; field[i] = 1; }
            }
        }

        // ── Full drift ──
        for (let i = 0; i < GRID_SQ; i++) {
            field[i] = Math.max(-SCALAR_FIELD_MAX, Math.min(SCALAR_FIELD_MAX, field[i] + fieldDot[i] * dt));
        }

        // ── Recompute Laplacian with updated field ──
        this._computeLaplacian(bcMode, topoConst, invCellWSq, invCellHSq, 1);

        // ── Refresh self-gravity at drifted field (restores O(dt²) for GR correction) ──
        if (sgOn) {
            this._computeGridGradients(bcMode, topoConst, 1);
            this.computeSelfGravity(domainW, domainH, softeningSq, bcMode, topoConst);
        }

        // ── Second half-kick (with updated field values) ──
        this._computeViscosity(invCellWSq, invCellHSq);
        if (sgOn) {
            for (let i = 0; i < GRID_SQ; i++) {
                const phiVal = field[i];
                const muSqEff = muSq - thermal[i];
                const lapI = lap[i];
                const Phi = Math.max(-SELFGRAV_PHI_MAX, Math.min(SELFGRAV_PHI_MAX, sgFull[i]));
                const portalTerm = portalArr ? HIGGS_AXION_COUPLING * portalArr[i] * portalArr[i] : 0;
                const ddphi = lapI + muSqEff * phiVal - muSq * phiVal * phiVal * phiVal
                    - damp * fieldDot[i] + src[i] * invCellArea + visc[i]
                    + 4 * Phi * lapI
                    + 2 * (sgGx[i] * fGx[i] * invCellWSq + sgGy[i] * fGy[i] * invCellHSq)
                    + 2 * Phi * muSqEff * phiVal - 2 * Phi * muSq * phiVal * phiVal * phiVal
                    - portalTerm * phiVal - 2 * Phi * portalTerm * phiVal;
                fieldDot[i] += ddphi * halfDt;
                if (!isFinite(fieldDot[i])) { fieldDot[i] = 0; field[i] = 1; }
            }
        } else if (portalArr) {
            for (let i = 0; i < GRID_SQ; i++) {
                const phiVal = field[i];
                const ddphi = lap[i] + (muSq - thermal[i]) * phiVal
                    - muSq * phiVal * phiVal * phiVal
                    - damp * fieldDot[i] + src[i] * invCellArea + visc[i]
                    - HIGGS_AXION_COUPLING * portalArr[i] * portalArr[i] * phiVal;
                fieldDot[i] += ddphi * halfDt;
                if (!isFinite(fieldDot[i])) { fieldDot[i] = 0; field[i] = 1; }
            }
        } else {
            for (let i = 0; i < GRID_SQ; i++) {
                const phiVal = field[i];
                const ddphi = lap[i] + (muSq - thermal[i]) * phiVal
                    - muSq * phiVal * phiVal * phiVal
                    - damp * fieldDot[i] + src[i] * invCellArea + visc[i];
                fieldDot[i] += ddphi * halfDt;
                if (!isFinite(fieldDot[i])) { fieldDot[i] = 0; field[i] = 1; }
            }
        }

        // Pre-compute grid gradients for C² smooth force interpolation
        this._computeGridGradients(bcMode, topoConst, 1);
    }

    /** PQS deposition of g·baseMass as scalar source (g = HIGGS_COUPLING). */
    _depositSources(particles, invCellW, invCellH, bcMode, topoConst) {
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (p.baseMass < EPSILON) continue;
            this._depositPQS(this._source, p.pos.x, p.pos.y, HIGGS_COUPLING * p.baseMass, invCellW, invCellH, bcMode, topoConst);
        }
    }

    /** PQS deposition of local kinetic energy density. */
    _depositThermal(particles, invCellW, invCellH, bcMode, topoConst, relativityEnabled) {
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            let ke;
            if (relativityEnabled) {
                const wSq = p.w.x * p.w.x + p.w.y * p.w.y;
                ke = wSq / (Math.sqrt(1 + wSq) + 1) * p.mass;
            } else {
                ke = 0.5 * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
            }
            if (ke < EPSILON) continue;
            this._depositPQS(this._thermal, p.pos.x, p.pos.y, ke, invCellW, invCellH, bcMode, topoConst);
        }
    }

    /** Set particle effective masses: m → baseMass * |phi| (VEV=1).
     *  PQS interpolation is C² smooth — no self-force subtraction needed.
     *  Clamped mass rate prevents resonant oscillation for massive particles.
     *  Conserves momentum: scales proper velocity w by old_mass/new_mass.
     */
    modulateMasses(particles, dt, domainW, domainH, blackHoleEnabled, boundaryMode, topoConst) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        const bcMode = boundaryMode;
        const maxDelta = HIGGS_MASS_MAX_DELTA * dt;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (p.baseMass < EPSILON) continue;

            const phiLocal = this.interpolate(p.pos.x, p.pos.y, invCellW, invCellH, bcMode, topoConst);
            p.higgsMod = Math.max(Math.abs(phiLocal), HIGGS_MASS_FLOOR);
            const targetMass = Math.max(p.baseMass * Math.abs(phiLocal), HIGGS_MASS_FLOOR * p.baseMass);
            if (targetMass !== targetMass) continue; // NaN guard
            const diff = targetMass - p.mass;
            const newMass = p.mass + (diff > maxDelta ? maxDelta : diff < -maxDelta ? -maxDelta : diff);

            // Conserve momentum by adjusting proper velocity
            const massRatio = p.mass / newMass;
            p.w.x *= massRatio;
            p.w.y *= massRatio;

            p.mass = newMass;

            const bodyR = Math.cbrt(p.mass);
            const bodyRSq = bodyR * bodyR;
            p.bodyRadiusSq = bodyRSq;
            if (blackHoleEnabled) {
                p.radius = kerrNewmanRadius(p.mass, bodyRSq, p.angVel, p.charge);
            } else {
                p.radius = bodyR;
            }
            p.radiusSq = p.radius * p.radius;
            p.invMass = 1 / p.mass;

            // Derive velocity from updated proper velocity
            const wSq = p.w.x * p.w.x + p.w.y * p.w.y;
            const gamma = Math.sqrt(1 + wSq);
            p.vel.x = p.w.x / gamma;
            p.vel.y = p.w.y / gamma;
        }
    }

    /** Modulate pion masses by local Higgs field value: m_pion = baseMass * |φ(x)|.
     *  Conserves momentum by scaling proper velocity w.
     */
    modulatePionMasses(pions, domainW, domainH, boundaryMode, topoConst) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        for (let i = 0; i < pions.length; i++) {
            const pi = pions[i];
            if (!pi.alive) continue;
            const phiLocal = this.interpolate(pi.pos.x, pi.pos.y, invCellW, invCellH, boundaryMode, topoConst);
            const newMass = Math.max(pi.baseMass * Math.abs(phiLocal), HIGGS_MASS_FLOOR * pi.baseMass);
            if (newMass !== newMass) continue; // NaN guard
            if (Math.abs(newMass - pi.mass) < EPSILON) continue;
            const ratio = pi.mass / newMass;
            pi.w.x *= ratio;
            pi.w.y *= ratio;
            pi.mass = newMass;
            pi._syncVel();
        }
    }

    /** Apply gradient force: F = +g·baseMass·sign(phi)·grad(phi) where g = HIGGS_COUPLING.
     *  sign(phi) ensures consistency with mass generation m = baseMass·|phi|.
     *  PQS-interpolated grid gradients give C² continuous forces.
     */
    applyForces(particles, domainW, domainH, boundaryMode, topoConst) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        const bcMode = boundaryMode;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (p.baseMass < EPSILON) continue;

            const result = this.interpolateWithGradient(p.pos.x, p.pos.y, invCellW, invCellH, bcMode, topoConst);
            const grad = result.grad;
            if (!grad) continue;

            const sign = result.value >= 0 ? 1 : -1;
            const forceX = HIGGS_COUPLING * p.baseMass * sign * grad.x;
            const forceY = HIGGS_COUPLING * p.baseMass * sign * grad.y;

            p.force.x += forceX;
            p.force.y += forceY;
            p.forceHiggs.x += forceX;
            p.forceHiggs.y += forceY;
        }
    }

    /** Particle-field interaction energy: Σ -baseMass·(|phi(x)| - 1).
     *  At VEV (phi=1), energy = 0. Depleted field (phi<1) costs energy.
     */
    particleFieldEnergy(particles, domainW, domainH, bcMode, topoConst) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return 0;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        let energy = 0;
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (p.baseMass < EPSILON) continue;
            const phiLocal = this.interpolate(p.pos.x, p.pos.y, invCellW, invCellH, bcMode, topoConst);
            energy -= p.baseMass * (Math.abs(phiLocal) - 1);
        }
        return energy;
    }

    /** Portal interaction energy: ½λ∫φ²a² dA. Only counted here (not in Axion) to avoid double-counting. */
    portalEnergy(axionField, domainW, domainH) {
        if (!axionField) return 0;
        const GRID = this._grid;
        const cellArea = (domainW / GRID) * (domainH / GRID);
        const hf = this.field;
        const af = axionField.field;
        let total = 0;
        for (let i = 0; i < this._gridSq; i++) {
            total += hf[i] * hf[i] * af[i] * af[i];
        }
        return 0.5 * HIGGS_AXION_COUPLING * total * cellArea;
    }

    /** Total field energy: KE + gradient + Mexican hat potential, integrated over grid. */
    energy(domainW, domainH) {
        const muSq = 0.5 * this.mass * this.mass;
        const vacOffset = 0.25 * muSq; // shift so V(VEV=1)=0
        return this._fieldEnergy(domainW, domainH,
            p => muSq * (-0.5 * p * p + 0.25 * p * p * p * p) + vacOffset);
    }

    /** Render field deviation from VEV=1 to offscreen canvas.
     *  Lime = enhanced (phi > 1), purple = depleted (phi < 1). */
    render(isLight) {
        const field = this.field;
        const data = this._imgData.data;
        const GRID_SQ = this._gridSq;

        for (let i = 0; i < GRID_SQ; i++) {
            const deviation = field[i] - 1;
            const intensity = Math.min(Math.abs(deviation) * (8 / HIGGS_COUPLING), 1.0);
            const alpha = intensity * (isLight ? 60 : 80);
            const idx = i * 4;

            const rgb = deviation < 0 ? _depletedRGB : _enhancedRGB;
            data[idx] = rgb[0]; data[idx + 1] = rgb[1]; data[idx + 2] = rgb[2];
            data[idx + 3] = alpha;
        }
        this._ctx.putImageData(this._imgData, 0, 0);
    }
}
