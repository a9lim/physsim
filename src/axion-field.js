// ─── Axion Scalar Field ───
// Dynamical scalar field (ALP) on a 2D grid with quadratic potential.
// V(a) = 1/2 m_a^2 a^2  (no symmetry breaking — field oscillates around a=0)
// m_a is the free parameter (slider 0.01-0.25, default 0.05)
// Extends ScalarField for shared PQS infrastructure.
//
// Two couplings:
//
// 1. Scalar EM coupling (aF²): L_int = -(1+g·a)F²/4
//    Makes α position-dependent: α_eff(x) = α·(1 + g·a(x))
//    Source: g·q². Gradient force: F = +g·q²·∇a. Same for matter and antimatter.
//    The QCD axion's pseudoscalar aFF̃ ∝ E·B vanishes in 2D (E in-plane, B⊥).
//
// 2. Pseudoscalar PQ coupling (aGG̃ analog, when Yukawa enabled):
//    Peccei-Quinn mechanism — flips sign under CP (matter vs antimatter).
//    Source: ±g·m (positive for matter, negative for antimatter).
//    Gradient force: F = ±g·m·∇a. Yukawa modulation: g²_eff = g²·yukMod.
//    yukMod = 1 + g·a for matter, 1 - g·a for antimatter.
//    At vacuum (a=0): yukMod = 1 for both → CP conserved (PQ solution).

import { SCALAR_GRID, SCALAR_FIELD_MAX, DEFAULT_AXION_MASS, AXION_COUPLING, HIGGS_AXION_COUPLING, SELFGRAV_PHI_MAX, EPSILON, SUPERRADIANCE_COEFF, INERTIA_K, MIN_MASS, kerrNewmanRadius } from './config.js';
import ScalarField from './scalar-field.js';

// Parse overlay colors from shared palette at module load (0-255 ints)
const _ph = window._parseHex; // hex -> [r,g,b] in 0–1
const _posRGB = _ph(window._PALETTE.extended.indigo).map(v => (v * 255 + 0.5) | 0);
const _negRGB = _ph(window._PALETTE.extended.yellow).map(v => (v * 255 + 0.5) | 0);

export default class AxionField extends ScalarField {
    constructor() {
        super(SCALAR_GRID, SCALAR_FIELD_MAX);
        this._vacValue = 0; // Axion vacuum = 0
        this.mass = DEFAULT_AXION_MASS;
        this.reset();
    }

    reset() {
        super.reset(0); // vacuum = 0
    }

    /** C15: Fused energy density with quadratic potential V(a) = ½m_a²a² in one loop. */
    _computeEnergyDensity(domainW, domainH) {
        const halfMaSq = 0.5 * this.mass * this.mass;
        super._computeEnergyDensity(domainW, domainH, a => halfMaSq * a * a);
    }

    /** Evolve field one timestep using Störmer-Verlet (kick-drift-kick, O(dt²)). */
    update(dt, particles, boundaryMode, topoConst, domainW, domainH, coulombEnabled = false, yukawaEnabled = false, gravityEnabled = false, softeningSq = 64, otherField = null, blackHoleEnabled = false) {
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

        // PQS source deposition: charged particles (EM) + massive particles (PQ)
        const src = this._source;
        src.fill(0);
        this._depositSources(particles, invCellW, invCellH, bcMode, topoConst, coulombEnabled, yukawaEnabled);

        if (blackHoleEnabled) {
            this._depositSuperradiance(particles, invCellW, invCellH, bcMode, topoConst, dt);
        }
        const cellArea = cellW * cellH;
        const invCellArea = cellArea > EPSILON ? 1 / cellArea : 0;

        // Self-gravity: weak-field GR correction to Klein-Gordon equation
        // ä = (1+4Φ)∇²a + 2∇Φ·∇a - (1+2Φ)V'(a)
        // Correction: 4Φ·∇²a + 2∇Φ·∇a - 2Φ·m_a²·a
        const sgOn = gravityEnabled;
        if (sgOn) this.computeSelfGravity(domainW, domainH, softeningSq, bcMode, topoConst);
        const sgFull = this._sgPhiFull;
        const sgGx = this._sgGradX;
        const sgGy = this._sgGradY;
        const fGx = this._gradX;
        const fGy = this._gradY;

        // Störmer-Verlet: half-kick → drift → recompute → second half-kick
        // Klein-Gordon: d²a/dt² = ∇²a - m_a²·a - g·m_a·ȧ + source  (ζ = g/2, Q = 1/g)
        const mA = this.mass;
        const mASq = mA * mA;
        const damp = AXION_COUPLING * mA; // Q = 1/g = 20, so g·Q = 1 (resonant buildup ≈ static response)
        const halfDt = dt * 0.5;

        // C16: Pre-compute viscosity coefficient
        const nu = 0.5 / Math.sqrt(invCellWSq + invCellHSq);
        const last = GRID - 1;

        // Portal coupling: V_portal = ½λφ²a², contributes -λφ²a to ddA
        const portalArr = otherField ? otherField.field : null;

        // ── First half-kick (C16: Laplacian + viscosity inlined) ──
        this._axionKick(halfDt, field, fieldDot, src, invCellArea,
            mASq, damp, nu, invCellWSq, invCellHSq, GRID, last, bcMode, topoConst,
            sgOn, sgFull, sgGx, sgGy, fGx, fGy, portalArr);

        // ── Full drift ──
        for (let i = 0; i < GRID_SQ; i++) {
            field[i] = Math.max(-SCALAR_FIELD_MAX, Math.min(SCALAR_FIELD_MAX, field[i] + fieldDot[i] * dt));
        }

        // ── Refresh self-gravity at drifted field (restores O(dt²) for GR correction) ──
        if (sgOn) {
            this._computeGridGradients(bcMode, topoConst, 0);
            this.computeSelfGravity(domainW, domainH, softeningSq, bcMode, topoConst);
        }

        // ── Second half-kick (C16: Laplacian + viscosity inlined) ──
        this._axionKick(halfDt, field, fieldDot, src, invCellArea,
            mASq, damp, nu, invCellWSq, invCellHSq, GRID, last, bcMode, topoConst,
            sgOn, sgFull, sgGx, sgGy, fGx, fGy, portalArr);

        // Pre-compute grid gradients for C² smooth force interpolation
        this._computeGridGradients(bcMode, topoConst, 0);
    }

    /** C16: Axion half-kick with inlined Laplacian + viscosity.
     *  Computes Laplacian of field and viscosity of fieldDot inline, avoiding separate grid passes.
     *  Interior cells use direct indexing; border cells use _nb() for topology-aware wrapping. */
    _axionKick(halfDt, field, fieldDot, src, invCellArea,
        mASq, damp, nu, invCellWSq, invCellHSq, GRID, last, bcMode, topoConst,
        sgOn, sgFull, sgGx, sgGy, fGx, fGy, portalArr) {
        const vacValue = 0; // Axion vacuum

        for (let iy = 0; iy < GRID; iy++) {
            for (let ix = 0; ix < GRID; ix++) {
                const i = iy * GRID + ix;
                let lapI, viscI;

                if (ix > 0 && ix < last && iy > 0 && iy < last) {
                    // Interior: direct indexing
                    const fC = field[i];
                    lapI = (field[i - 1] + field[i + 1] - 2 * fC) * invCellWSq
                         + (field[i - GRID] + field[i + GRID] - 2 * fC) * invCellHSq;
                    const fdC = fieldDot[i];
                    viscI = nu * ((fieldDot[i - 1] + fieldDot[i + 1] - 2 * fdC) * invCellWSq
                                + (fieldDot[i - GRID] + fieldDot[i + GRID] - 2 * fdC) * invCellHSq);
                } else {
                    // Border: topology-aware neighbor lookup
                    const fC = field[i];
                    const iL = this._nb(ix - 1, iy, bcMode, topoConst);
                    const iR = this._nb(ix + 1, iy, bcMode, topoConst);
                    const iT = this._nb(ix, iy - 1, bcMode, topoConst);
                    const iB = this._nb(ix, iy + 1, bcMode, topoConst);
                    lapI = ((iL >= 0 ? field[iL] : vacValue) + (iR >= 0 ? field[iR] : vacValue) - 2 * fC) * invCellWSq
                         + ((iT >= 0 ? field[iT] : vacValue) + (iB >= 0 ? field[iB] : vacValue) - 2 * fC) * invCellHSq;
                    // Viscosity: clamp-to-edge for border
                    const fdC = fieldDot[i];
                    const fdL = ix > 0 ? fieldDot[i - 1] : fdC;
                    const fdR = ix < last ? fieldDot[i + 1] : fdC;
                    const fdT = iy > 0 ? fieldDot[i - GRID] : fdC;
                    const fdB = iy < last ? fieldDot[i + GRID] : fdC;
                    viscI = nu * ((fdL + fdR - 2 * fdC) * invCellWSq
                                + (fdT + fdB - 2 * fdC) * invCellHSq);
                }

                const aVal = field[i];

                if (sgOn) {
                    const Phi = Math.max(-SELFGRAV_PHI_MAX, Math.min(SELFGRAV_PHI_MAX, sgFull[i]));
                    const portalTerm = portalArr ? HIGGS_AXION_COUPLING * portalArr[i] * portalArr[i] : 0;
                    const ddA = lapI - mASq * aVal - damp * fieldDot[i] + src[i] * invCellArea + viscI
                        + 4 * Phi * lapI
                        + 2 * (sgGx[i] * fGx[i] * invCellWSq + sgGy[i] * fGy[i] * invCellHSq)
                        - 2 * Phi * mASq * aVal
                        - portalTerm * aVal - 2 * Phi * portalTerm * aVal;
                    fieldDot[i] += ddA * halfDt;
                    if (!isFinite(fieldDot[i])) { fieldDot[i] = 0; field[i] = 0; }
                } else if (portalArr) {
                    const ddA = lapI - mASq * aVal - damp * fieldDot[i] + src[i] * invCellArea + viscI
                        - HIGGS_AXION_COUPLING * portalArr[i] * portalArr[i] * aVal;
                    fieldDot[i] += ddA * halfDt;
                    if (!isFinite(fieldDot[i])) { fieldDot[i] = 0; field[i] = 0; }
                } else {
                    const ddA = lapI - mASq * aVal - damp * fieldDot[i] + src[i] * invCellArea + viscI;
                    fieldDot[i] += ddA * halfDt;
                    if (!isFinite(fieldDot[i])) { fieldDot[i] = 0; field[i] = 0; }
                }
            }
        }
    }

    /** PQS source deposition.
     *  EM (scalar aF², when Coulomb on): g·q² — same sign for matter and antimatter.
     *  PQ (pseudoscalar aGG̃, when Yukawa on): ±g·m — flips sign for antimatter (CP violation).
     */
    _depositSources(particles, invCellW, invCellH, bcMode, topoConst, coulombEnabled = false, yukawaEnabled = false) {
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            let s = 0;
            if (coulombEnabled) {
                const qSq = p.charge * p.charge;
                if (qSq > EPSILON) s += AXION_COUPLING * qSq;
            }
            if (yukawaEnabled && p.mass > EPSILON) {
                s += AXION_COUPLING * p.mass * (p.antimatter ? -1 : 1);
            }
            if (s === 0) continue;
            this._depositPQS(this._source, p.pos.x, p.pos.y, s, invCellW, invCellH, bcMode, topoConst);
        }
    }

    /** Superradiant instability: spinning BH pumps axion field.
     *  Rate: Γ = C · (M·μ_a)² · max(Ω_H - μ_a, 0), deposits into _source via PQS.
     *  Back-reaction: BH angular momentum reduced by dJ = dE/Ω_H.
     */
    _depositSuperradiance(particles, invCellW, invCellH, bcMode, topoConst, dt) {
        const muA = this.mass;
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.torqueSuperradiance = 0;
            if (p.mass <= MIN_MASS) continue;

            const M = p.mass;
            const bodyRSq = p.bodyRadiusSq;
            const a = INERTIA_K * bodyRSq * Math.abs(p.angVel);
            const rPlus = kerrNewmanRadius(M, bodyRSq, p.angVel, p.charge);
            const rPlusSq = rPlus * rPlus;
            const sigma = rPlusSq + a * a;
            if (sigma < EPSILON) continue;
            const omegaH = a / sigma;

            // Superradiance condition: field frequency < horizon angular velocity
            if (omegaH <= muA) continue;

            const alphaG = M * muA;
            const rate = SUPERRADIANCE_COEFF * alphaG * alphaG * (omegaH - muA);
            const dE = rate * dt;
            if (dE < EPSILON) continue;

            // Deposit into source array (positive = excite field)
            this._depositPQS(this._source, p.pos.x, p.pos.y, dE, invCellW, invCellH, bcMode, topoConst);

            // Back-reaction: reduce BH angular momentum
            // dJ = dE / Ω_H, then Δangw = -sign(angw) · dJ / I
            const I = INERTIA_K * bodyRSq * M;
            if (I < EPSILON) continue;
            const dJ = dE / omegaH;
            p.angw -= Math.sign(p.angw) * dJ / I;
            // Recompute derived angular velocity
            const absAngw = Math.abs(p.angw);
            p.angVel = p.angw / Math.sqrt(1 + absAngw * absAngw * bodyRSq);

            // Record effective torque for display (same units as other torques: τ = dJ/dt)
            p.torqueSuperradiance = -Math.sign(p.angw) * rate / omegaH;
        }
    }

    /** Interpolate local axion field value at each particle position.
     *  Sets p.axMod = 1 + g·a(x) when Coulomb on (scalar EM coupling).
     *  Sets p.yukMod = 1 ± g·a(x) when Yukawa on (pseudoscalar PQ coupling).
     */
    interpolateAxMod(particles, domainW, domainH, coulombEnabled = false, yukawaEnabled = false, boundaryMode = 0, topoConst = 0) {
        // A2: Early return when neither coupling is active
        if (!coulombEnabled && !yukawaEnabled) {
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                if (!p.alive) continue;
                p.axMod = 1;
                p.yukMod = 1;
            }
            return;
        }
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) {
            for (let i = 0; i < particles.length; i++) {
                particles[i].axMod = 1;
                particles[i].yukMod = 1;
            }
            return;
        }
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        const bcMode = boundaryMode;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            const aLocal = this.interpolate(p.pos.x, p.pos.y, invCellW, invCellH, bcMode, topoConst);
            const ga = AXION_COUPLING * aLocal;
            // axMod: scalar EM coupling (only meaningful when Coulomb on)
            if (coulombEnabled) {
                p.axMod = ga > -1 ? 1 + ga : 0;
            } else {
                p.axMod = 1;
            }
            // yukMod: pseudoscalar PQ coupling, flips sign for antimatter
            if (yukawaEnabled) {
                const pq = (p.antimatter ? -1 : 1) * ga;
                p.yukMod = pq > -1 ? 1 + pq : 0;
            } else {
                p.yukMod = 1;
            }
        }
    }

    /** Apply gradient forces from axion field.
     *  EM (scalar aF², when Coulomb on): F = +g·q²·∇a — same for matter and antimatter.
     *  PQ (pseudoscalar aGG̃, when Yukawa on): F = ±g·m·∇a — flips for antimatter.
     *  PQS-interpolated grid gradients give C² continuous forces.
     */
    applyForces(particles, domainW, domainH, coulombEnabled = false, yukawaEnabled = false, boundaryMode = 0, topoConst = 0) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        const bcMode = boundaryMode;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            let coupling = 0;
            if (coulombEnabled) {
                const qSq = p.charge * p.charge;
                if (qSq > EPSILON) coupling += qSq;
            }
            if (yukawaEnabled && p.mass > EPSILON) {
                coupling += p.mass * (p.antimatter ? -1 : 1);
            }
            if (coupling === 0) continue;

            const grad = this.gradient(p.pos.x, p.pos.y, invCellW, invCellH, bcMode, topoConst);
            if (!grad) continue;

            const forceX = AXION_COUPLING * coupling * grad.x;
            const forceY = AXION_COUPLING * coupling * grad.y;

            p.force.x += forceX;
            p.force.y += forceY;
            p.forceAxion.x += forceX;
            p.forceAxion.y += forceY;
        }
    }

    /** Particle-field interaction energy from both coupling channels.
     *  EM (aF²): -g·q²·a(x) per particle.
     *  PQ (aGG̃): -g·m·(±1)·a(x) per particle (+ for matter, - for antimatter).
     */
    particleFieldEnergy(particles, domainW, domainH, coulombEnabled, yukawaEnabled, bcMode, topoConst) {
        const GRID = this._grid;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return 0;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        let energy = 0;
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            const aLocal = this.interpolate(p.pos.x, p.pos.y, invCellW, invCellH, bcMode, topoConst);
            if (coulombEnabled) {
                energy -= AXION_COUPLING * p.charge * p.charge * aLocal;
            }
            if (yukawaEnabled && p.mass > EPSILON) {
                const sign = p.antimatter ? -1 : 1;
                energy -= AXION_COUPLING * p.mass * sign * aLocal;
            }
        }
        return energy;
    }

    /** Total field energy: KE + gradient + quadratic potential, integrated over grid. */
    energy(domainW, domainH) {
        const mASq = this.mass * this.mass;
        return this._fieldEnergy(domainW, domainH, a => 0.5 * mASq * a * a);
    }

    /** Render field to offscreen canvas. Indigo = positive, yellow = negative. */
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
