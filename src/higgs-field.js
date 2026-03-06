// ─── Higgs Scalar Field ───
// Dynamical scalar field on a 2D grid with Mexican hat potential.
// V(phi) = -1/2 mu^2 phi^2 + 1/4 lambda phi^4
// VEV v = mu / sqrt(lambda), Higgs boson mass m_H = mu * sqrt(2)
// Symplectic Euler integration, bilinear interpolation, CIC source deposition.
// Self-force subtraction via analytical steady-state Green's function estimate.

import { HIGGS_GRID, HIGGS_LAMBDA, DEFAULT_HIGGS_VEV, DEFAULT_HIGGS_COUPLING, DEFAULT_HIGGS_THERMAL_K, HIGGS_DAMPING, EPSILON } from './config.js';
import { TORUS, KLEIN, RP2 } from './topology.js';

const GRID = HIGGS_GRID;
const GRID_SQ = GRID * GRID;

// Boundary mode constants for inner loop (avoid string comparison)
const BC_DESPAWN = 0;
const BC_BOUNCE = 1;
const BC_LOOP = 2;

// Field value clamp: prevent runaway in the wave equation
const PHI_MAX = 50;

export default class HiggsField {
    constructor() {
        this.phi = new Float64Array(GRID_SQ);
        this.phiDot = new Float64Array(GRID_SQ);
        this._laplacian = new Float64Array(GRID_SQ);
        this._source = new Float64Array(GRID_SQ);
        this._thermal = new Float64Array(GRID_SQ);

        this.lambda = HIGGS_LAMBDA;
        this.vev = DEFAULT_HIGGS_VEV;
        this.coupling = DEFAULT_HIGGS_COUPLING;
        this.thermalK = DEFAULT_HIGGS_THERMAL_K;
        this.damping = HIGGS_DAMPING;

        // Offscreen canvas for rendering
        this.canvas = document.createElement('canvas');
        this.canvas.width = GRID;
        this.canvas.height = GRID;
        this._ctx = this.canvas.getContext('2d');
        this._imgData = this._ctx.createImageData(GRID, GRID);
        this._cic = { ix: 0, iy: 0, fx: 0, fy: 0 };

        this.reset();
    }

    reset() {
        const v = this.vev;
        for (let i = 0; i < GRID_SQ; i++) {
            this.phi[i] = v;
            this.phiDot[i] = 0;
        }
    }

    /**
     * Get neighbor grid index with boundary-aware wrapping.
     * Returns -1 for Dirichlet boundary (use VEV value).
     * bcMode: BC_DESPAWN=0, BC_BOUNCE=1, BC_LOOP=2 (integer for inner-loop speed).
     */
    _nb(ix, iy, dx, dy, bcMode, topoConst) {
        let nx = ix + dx;
        let ny = iy + dy;

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

        // Despawn: Dirichlet (phi = VEV at boundary)
        if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) return -1;
        return ny * GRID + nx;
    }

    /** Clamp-read phi at grid coords, clamping to boundary. */
    _phiAt(cx, cy) {
        if (cx < 0) cx = 0; else if (cx >= GRID) cx = GRID - 1;
        if (cy < 0) cy = 0; else if (cy >= GRID) cy = GRID - 1;
        return this.phi[cy * GRID + cx];
    }

    /** Compute CIC fractional grid coords into pre-allocated this._cic. */
    _cicCoords(x, y, invCellW, invCellH) {
        const gx = x * invCellW - 0.5;
        const gy = y * invCellH - 0.5;
        const c = this._cic;
        c.ix = Math.floor(gx);
        c.iy = Math.floor(gy);
        c.fx = gx - c.ix;
        c.fy = gy - c.iy;
    }

    /** Evolve field one timestep using symplectic Euler (kick-drift). */
    update(dt, particles, boundaryMode, topoConst, domainW, domainH) {
        if (dt <= 0) return;
        const phi = this.phi;
        const phiDot = this.phiDot;
        const lap = this._laplacian;
        const src = this._source;
        const thermal = this._thermal;
        const v = this.vev;
        const lam = this.lambda;
        const muSq = v * v * lam; // mu^2 = v^2 * lambda

        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellWSq = 1 / (cellW * cellW);
        const invCellHSq = 1 / (cellH * cellH);
        const invCellArea = 1 / (cellW * cellH);

        // Convert string boundary mode to integer once
        const bcMode = boundaryMode === 'loop' ? BC_LOOP
                     : boundaryMode === 'bounce' ? BC_BOUNCE
                     : BC_DESPAWN;

        // Deposit particle sources and thermal energy via CIC
        src.fill(0);
        thermal.fill(0);
        this._depositSources(particles, domainW, domainH);
        if (this.thermalK > 0) this._depositThermal(particles, domainW, domainH);

        // Compute Laplacian with boundary conditions
        for (let iy = 0; iy < GRID; iy++) {
            for (let ix = 0; ix < GRID; ix++) {
                const idx = iy * GRID + ix;
                const phiC = phi[idx];

                const iL = this._nb(ix, iy, -1, 0, bcMode, topoConst);
                const iR = this._nb(ix, iy, 1, 0, bcMode, topoConst);
                const iT = this._nb(ix, iy, 0, -1, bcMode, topoConst);
                const iB = this._nb(ix, iy, 0, 1, bcMode, topoConst);

                const phiL = iL >= 0 ? phi[iL] : v;
                const phiR = iR >= 0 ? phi[iR] : v;
                const phiT = iT >= 0 ? phi[iT] : v;
                const phiB = iB >= 0 ? phi[iB] : v;

                lap[idx] = (phiL + phiR - 2 * phiC) * invCellWSq
                         + (phiT + phiB - 2 * phiC) * invCellHSq;
            }
        }

        // Kick phiDot, then drift phi (symplectic Euler)
        const thermK = this.thermalK;
        const damp = this.damping;
        const coup = this.coupling;
        for (let i = 0; i < GRID_SQ; i++) {
            const phiVal = phi[i];

            // Thermal correction: mu^2_eff = mu^2 - thermalK * T^2_local
            // thermal[i] is KE density ~ T^2, so linear dependence gives correct T^2
            let muSqEff = muSq;
            if (thermK > 0 && thermal[i] > 0) {
                muSqEff -= thermK * thermal[i];
            }

            // Klein-Gordon: d^2 phi/dt^2 = nabla^2 phi + mu^2_eff * phi - lambda * phi^3
            //               + coupling * source / cellArea - damping * dphi/dt
            const ddphi = lap[i]
                        + muSqEff * phiVal
                        - lam * phiVal * phiVal * phiVal
                        + coup * src[i] * invCellArea
                        - damp * phiDot[i];

            phiDot[i] += ddphi * dt;
            const newPhi = phi[i] + phiDot[i] * dt;

            // Clamp field to prevent numerical blowup
            if (newPhi !== newPhi) { // NaN guard
                phi[i] = v;
                phiDot[i] = 0;
            } else if (newPhi > PHI_MAX) {
                phi[i] = PHI_MAX;
                phiDot[i] = 0;
            } else if (newPhi < -PHI_MAX) {
                phi[i] = -PHI_MAX;
                phiDot[i] = 0;
            } else {
                phi[i] = newPhi;
            }
        }
    }

    /** CIC deposition of particle source strength (baseMass / VEV). */
    _depositSources(particles, domainW, domainH) {
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        const src = this._source;
        const invV = this.vev > EPSILON ? 1 / this.vev : 0;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (p.baseMass < EPSILON) continue;
            const strength = p.baseMass * invV;
            this._cicCoords(p.pos.x, p.pos.y, invCellW, invCellH);
            const { ix, iy, fx, fy } = this._cic;

            if (ix >= 0 && ix < GRID && iy >= 0 && iy < GRID)
                src[iy * GRID + ix] += strength * (1 - fx) * (1 - fy);
            if (ix + 1 < GRID && iy >= 0 && iy < GRID)
                src[iy * GRID + ix + 1] += strength * fx * (1 - fy);
            if (ix >= 0 && ix < GRID && iy + 1 < GRID)
                src[(iy + 1) * GRID + ix] += strength * (1 - fx) * fy;
            if (ix + 1 < GRID && iy + 1 < GRID)
                src[(iy + 1) * GRID + ix + 1] += strength * fx * fy;
        }
    }

    /** CIC deposition of local kinetic energy density. */
    _depositThermal(particles, domainW, domainH) {
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        const out = this._thermal;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            const ke = 0.5 * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
            if (ke < EPSILON) continue;
            this._cicCoords(p.pos.x, p.pos.y, invCellW, invCellH);
            const { ix, iy, fx, fy } = this._cic;

            if (ix >= 0 && ix < GRID && iy >= 0 && iy < GRID)
                out[iy * GRID + ix] += ke * (1 - fx) * (1 - fy);
            if (ix + 1 < GRID && iy >= 0 && iy < GRID)
                out[iy * GRID + ix + 1] += ke * fx * (1 - fy);
            if (ix >= 0 && ix < GRID && iy + 1 < GRID)
                out[(iy + 1) * GRID + ix] += ke * (1 - fx) * fy;
            if (ix + 1 < GRID && iy + 1 < GRID)
                out[(iy + 1) * GRID + ix + 1] += ke * fx * fy;
        }
    }

    /**
     * Set particle effective masses from local field value: m = baseMass * |phi| / VEV.
     * Subtracts the analytical self-perturbation before sampling to avoid self-interaction.
     *
     * Self-field estimate: in linearized steady state (-nabla^2 + m_H^2) dphi = source,
     * ignoring the Laplacian (local approx), dphi_cell ~ source_cell / m_H^2.
     * The self-contribution to the bilinear sample is sum(w_i^2) * selfBase.
     */
    modulateMasses(particles, domainW, domainH, blackHoleEnabled) {
        const v = this.vev;
        if (v < EPSILON) return;
        const invV = 1 / v;
        const lam = this.lambda;
        const mHsq = 2 * lam * v * v; // Higgs boson mass squared
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        const cellArea = cellW * cellH;

        // selfScale = coupling / (v * cellArea * mHsq), so selfBase = baseMass * selfScale
        const selfDenom = v * cellArea * mHsq;
        const selfScale = selfDenom > EPSILON ? this.coupling / selfDenom : 0;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (p.baseMass < EPSILON) continue;

            this._cicCoords(p.pos.x, p.pos.y, invCellW, invCellH);
            const ix = this._cic.ix, iy = this._cic.iy;
            const fx = this._cic.fx, fy = this._cic.fy;

            // CIC weights
            const w00 = (1 - fx) * (1 - fy);
            const w10 = fx * (1 - fy);
            const w01 = (1 - fx) * fy;
            const w11 = fx * fy;

            // Bilinear sample of phi
            const phiRaw = this._phiAt(ix, iy) * w00
                         + this._phiAt(ix + 1, iy) * w10
                         + this._phiAt(ix, iy + 1) * w01
                         + this._phiAt(ix + 1, iy + 1) * w11;

            // Subtract self-perturbation: dphi_self_sampled = selfBase * sum(w_i^2)
            const selfBase = p.baseMass * selfScale;
            const selfSample = selfBase * (w00 * w00 + w10 * w10 + w01 * w01 + w11 * w11);
            const phiLocal = phiRaw - selfSample;

            const newMass = Math.max(p.baseMass * Math.abs(phiLocal * invV), EPSILON);
            // NaN guard
            if (newMass !== newMass) continue;
            p.mass = newMass;

            // Update mass-derived quantities (color depends only on charge, skip getColor)
            if (blackHoleEnabled) {
                const M = p.mass;
                const rSq = p.radiusSq > EPSILON ? p.radiusSq : EPSILON;
                const I = 0.4 * M * rSq; // INERTIA_K = 0.4
                const omega = p.angVel || 0;
                const a = M > EPSILON ? I * Math.abs(omega) / M : 0;
                const Q = p.charge;
                const disc = M * M - a * a - Q * Q;
                p.radius = disc > EPSILON ? M + Math.sqrt(disc) : M * 0.5; // BH_NAKED_FLOOR
            } else {
                p.radius = Math.cbrt(p.mass);
            }
            p.radiusSq = p.radius * p.radius;
            p.invMass = 1 / p.mass;
        }
    }

    /**
     * Apply gradient force: F = -(baseMass/VEV) * coupling * grad(phi).
     * Subtracts the analytical self-force gradient to prevent self-interaction.
     *
     * Self-force gradient (derived from CIC kernel symmetry):
     *   selfGradX = selfBase * (2fx - 1) * ((1-fy)^2 + fy^2) / cellW
     *   selfGradY = selfBase * (2fy - 1) * ((1-fx)^2 + fx^2) / cellH
     */
    applyForces(particles, domainW, domainH) {
        const v = this.vev;
        if (v < EPSILON) return;
        const invV = 1 / v;
        const coup = this.coupling;
        const lam = this.lambda;
        const mHsq = 2 * lam * v * v;
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        const cellArea = cellW * cellH;

        const selfDenom = v * cellArea * mHsq;
        const selfScale = selfDenom > EPSILON ? coup / selfDenom : 0;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            if (p.baseMass < EPSILON) continue;

            this._cicCoords(p.pos.x, p.pos.y, invCellW, invCellH);
            const ix = this._cic.ix, iy = this._cic.iy;
            const fx = this._cic.fx, fy = this._cic.fy;

            // Sample 4 corner phi values
            const v00 = this._phiAt(ix, iy);
            const v10 = this._phiAt(ix + 1, iy);
            const v01 = this._phiAt(ix, iy + 1);
            const v11 = this._phiAt(ix + 1, iy + 1);

            // Bilinear gradient of total field
            let gradX = ((v10 - v00) * (1 - fy) + (v11 - v01) * fy) * invCellW;
            let gradY = ((v01 - v00) * (1 - fx) + (v11 - v10) * fx) * invCellH;

            // Subtract analytical self-force gradient
            const selfBase = p.baseMass * selfScale;
            const fy2 = fy * fy;
            const fx2 = fx * fx;
            gradX -= selfBase * (2 * fx - 1) * ((1 - fy) * (1 - fy) + fy2) * invCellW;
            gradY -= selfBase * (2 * fy - 1) * ((1 - fx) * (1 - fx) + fx2) * invCellH;

            const coeff = -p.baseMass * invV * coup;
            const forceX = coeff * gradX;
            const forceY = coeff * gradY;

            // NaN guard
            if (forceX !== forceX || forceY !== forceY) continue;

            p.force.x += forceX;
            p.force.y += forceY;
            p.forceHiggs.x = forceX;
            p.forceHiggs.y = forceY;
        }
    }

    /** Total field energy: KE + gradient + potential, integrated over grid. */
    energy(domainW, domainH) {
        const cellW = domainW / GRID;
        const cellH = domainH / GRID;
        if (cellW < EPSILON || cellH < EPSILON) return 0;
        const cellArea = cellW * cellH;
        const invCellW = 1 / cellW;
        const invCellH = 1 / cellH;
        const phi = this.phi;
        const phiDot = this.phiDot;
        const lam = this.lambda;
        const v = this.vev;
        const muSq = v * v * lam;

        // V(v) = -mu^4/(4*lambda), offset to make V(VEV)=0
        const vacOffset = lam > EPSILON ? 0.25 * muSq * muSq / lam : 0;
        let total = 0;
        for (let iy = 0; iy < GRID; iy++) {
            for (let ix = 0; ix < GRID; ix++) {
                const idx = iy * GRID + ix;
                const p = phi[idx];

                // KE: 1/2 (dphi/dt)^2
                const ke = 0.5 * phiDot[idx] * phiDot[idx];

                // Gradient energy: 1/2 |grad phi|^2
                const phiR = ix + 1 < GRID ? phi[idx + 1] : p;
                const phiB = iy + 1 < GRID ? phi[idx + GRID] : p;
                const dxPhi = (phiR - p) * invCellW;
                const dyPhi = (phiB - p) * invCellH;
                const gradE = 0.5 * (dxPhi * dxPhi + dyPhi * dyPhi);

                // Potential: V(phi) = -1/2 mu^2 phi^2 + 1/4 lambda phi^4 + vacOffset
                // Shifted so V(VEV) = 0 (vacuum has zero energy)
                const pot = -0.5 * muSq * p * p + 0.25 * lam * p * p * p * p + vacOffset;

                total += (ke + gradE + pot) * cellArea;
            }
        }

        // NaN guard
        return total === total ? total : 0;
    }

    /** Render field deviation from VEV to offscreen canvas. */
    render(isLight) {
        const phi = this.phi;
        const v = this.vev;
        const data = this._imgData.data;
        const halfV = Math.max(v * 0.5, EPSILON);
        const invHalfV = 1 / halfV;

        for (let i = 0; i < GRID_SQ; i++) {
            const deviation = phi[i] - v;
            const intensity = Math.min(Math.abs(deviation) * invHalfV, 1.0);
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
