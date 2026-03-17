// ─── Effective Potential Plot ───
// V_eff(r) = V(r) + L²/(2μr²) for selected particle vs most massive body.
// Draws curve + current-position marker on a sidebar canvas.

import { TWO_PI, SOFTENING_SQ, BH_SOFTENING_SQ, YUKAWA_COUPLING, TORUS } from './config.js';
import { minImage } from './topology.js';

const N_SAMPLES = 200;
const _miOut = { x: 0, y: 0 };
const MARGIN = 28;

export default class EffectivePotentialPlot {
    constructor() {
        this.enabled = true;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this._vBuf = new Float64Array(N_SAMPLES);
        this._rBuf = new Float64Array(N_SAMPLES);
        this.currentR = 0;
        this.currentV = 0;
        this._valid = false;
        // R11: Cached layout dimensions (avoids reflow from clientWidth reads)
        this._cachedWidth = 180;
        this._cachedDpr = devicePixelRatio || 1;
        // R12: Dirty flag — skip 200-sample recomputation when key inputs unchanged
        this._lastSelId = -1;
        this._lastRefId = -1;
        this._lastR10 = -1; // r rounded to 1 decimal
        this._lastToggleKey = '';
    }

    /** R11: Call on resize to refresh cached dimensions. */
    cacheSize() {
        this._cachedWidth = this.canvas.clientWidth || 180;
        this._cachedDpr = devicePixelRatio || 1;
    }

    update(particles, selectedParticle, physics) {
        if (!this.enabled || !selectedParticle) { this._valid = false; return; }

        const sel = selectedParticle;

        // Find most massive other particle as reference body
        let ref = null, maxM = 0;
        for (const p of particles) {
            if (p === sel) continue;
            if (p.mass > maxM) { maxM = p.mass; ref = p; }
        }
        if (!ref) { this._valid = false; return; }

        // R12: Build toggle key and check if curve needs recomputation
        const toggleKey = `${physics.gravityEnabled}${physics.coulombEnabled}${physics.magneticEnabled}${physics.gravitomagEnabled}${physics.yukawaEnabled}`;

        // Relative state (minimum-image for periodic boundaries)
        let dx, dy;
        if (physics.periodic) {
            minImage(ref.pos.x, ref.pos.y, sel.pos.x, sel.pos.y,
                     physics._topologyConst || TORUS, physics.domainW, physics.domainH,
                     physics.domainW * 0.5, physics.domainH * 0.5, _miOut);
            dx = _miOut.x; dy = _miOut.y;
        } else {
            dx = sel.pos.x - ref.pos.x;
            dy = sel.pos.y - ref.pos.y;
        }
        const r = Math.sqrt(dx * dx + dy * dy) || 1;

        // R12: Skip full curve recomputation when key inputs haven't changed
        const r10 = Math.round(r * 10);
        if (sel.id === this._lastSelId && ref.id === this._lastRefId &&
            r10 === this._lastR10 && toggleKey === this._lastToggleKey && this._valid) {
            // Only update marker position (cheap)
            this.currentR = r;
            return;
        }
        this._lastSelId = sel.id;
        this._lastRefId = ref.id;
        this._lastR10 = r10;
        this._lastToggleKey = toggleKey;

        const dvx = sel.vel.x - ref.vel.x;
        const dvy = sel.vel.y - ref.vel.y;

        // Angular momentum (scalar in 2D): L = μ * (r × v)
        const mu = (sel.mass * ref.mass) / (sel.mass + ref.mass); // reduced mass
        const Lz = mu * (dx * dvy - dy * dvx);

        // Toggle state
        const grav = physics.gravityEnabled;
        const coul = physics.coulombEnabled;
        const mag = physics.magneticEnabled;
        const gm = physics.gravitomagEnabled;
        const yuk = physics.yukawaEnabled;
        const softSq = physics.blackHoleEnabled ? BH_SOFTENING_SQ : SOFTENING_SQ;
        const axMod = sel.axMod;
        const yukMod = sel.yukMod;

        // Magnetic/GM moments (cached per-substep in computeAllForces)
        const selMu = sel.magMoment;
        const refMu = ref.magMoment;
        const selL = sel.angMomentum;
        const refL = ref.angMomentum;

        // Sample range: 0.5 to 4× current separation (clamped)
        const rMin = Math.max(Math.sqrt(softSq) * 0.5, r * 0.1);
        const rMax = Math.max(r * 4, rMin * 10);

        for (let i = 0; i < N_SAMPLES; i++) {
            const t = i / (N_SAMPLES - 1);
            const ri = rMin + t * (rMax - rMin);
            this._rBuf[i] = ri;
            this._vBuf[i] = this._vEff(ri, sel, ref, mu, Lz, grav, coul, mag, gm, yuk,
                softSq, axMod, yukMod, selMu, refMu, selL, refL, physics);
        }

        // Current position on curve
        this.currentR = r;
        this.currentV = this._vEff(r, sel, ref, mu, Lz, grav, coul, mag, gm, yuk,
            softSq, axMod, yukMod, selMu, refMu, selL, refL, physics);
        this._valid = true;
        this._rMin = rMin;
        this._rMax = rMax;
    }

    _vEff(r, sel, ref, mu, Lz, grav, coul, mag, gm, yuk, softSq, axMod, yukMod, selMu, refMu, selL, refL, physics) {
        const rEff = Math.sqrt(r * r + softSq);
        const invR = 1 / rEff;
        const invR3 = invR * invR * invR;

        let v = 0;

        // Centrifugal barrier: L²/(2μr²)
        if (mu > 0) v += (Lz * Lz) / (2 * mu * r * r);

        // Gravitational: -m₁m₂/r
        if (grav) v -= sel.mass * ref.mass * invR;

        // Coulomb: +q₁q₂/r (with axion modulation)
        if (coul) v += sel.charge * ref.charge * invR * axMod;

        // Magnetic dipole: +μ₁μ₂/r³
        if (mag) v += selMu * refMu * invR3 * axMod;

        // GM dipole: -L₁L₂/r³
        if (gm) v -= selL * refL * invR3;

        // Yukawa: -g²m₁m₂·exp(-μr)/r (with PQ modulation)
        if (yuk) {
            const muEff = physics.higgsEnabled ? physics.yukawaMu * Math.sqrt(sel.higgsMod * ref.higgsMod) : physics.yukawaMu;
            v -= YUKAWA_COUPLING * yukMod * sel.mass * ref.mass * Math.exp(-muEff * r) * invR;
        }

        return v;
    }

    draw(isLight) {
        if (!this.enabled || !this._valid) return;

        const dpr = this._cachedDpr;
        const ps = this._cachedWidth;
        const pxW = Math.round(ps * dpr);
        const pxH = Math.round(ps * dpr);
        if (this.canvas.width !== pxW || this.canvas.height !== pxH) {
            this.canvas.width = pxW;
            this.canvas.height = pxH;
        }

        const c = this.ctx;
        c.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Auto-scale V axis
        let vMin = Infinity, vMax = -Infinity;
        for (let i = 0; i < N_SAMPLES; i++) {
            const v = this._vBuf[i];
            if (isFinite(v)) {
                if (v < vMin) vMin = v;
                if (v > vMax) vMax = v;
            }
        }

        // Clip extreme values to keep the plot readable
        const vRange = vMax - vMin;
        if (vRange === 0 || !isFinite(vRange)) return;
        const vPad = vRange * 0.1;
        vMin -= vPad;
        vMax += vPad;
        const vSpan = vMax - vMin;

        const plotW = ps - MARGIN - 4;
        const plotH = ps - MARGIN - 4;

        c.clearRect(0, 0, ps, ps);
        c.fillStyle = isLight ? '#FCF7F244' : '#0C0B0988';
        c.fillRect(0, 0, ps, ps);

        // Axes
        c.strokeStyle = isLight ? '#1A161233' : '#E8DED433';
        c.lineWidth = 0.5;
        c.beginPath();
        c.moveTo(MARGIN, 0); c.lineTo(MARGIN, ps);
        c.moveTo(0, ps - MARGIN); c.lineTo(ps, ps - MARGIN);
        c.stroke();

        // Zero line (if visible)
        if (vMin < 0 && vMax > 0) {
            const y0 = (ps - MARGIN) - ((-vMin) / vSpan) * plotH;
            c.strokeStyle = isLight ? '#1A161218' : '#E8DED418';
            c.setLineDash([4, 4]);
            c.beginPath();
            c.moveTo(MARGIN, y0);
            c.lineTo(ps, y0);
            c.stroke();
            c.setLineDash([]);
        }

        // Labels
        c.fillStyle = isLight ? '#1A161288' : '#E8DED488';
        c.font = '9px Noto Sans Mono';
        c.fillText('r', ps - 12, ps - MARGIN + 12);
        c.fillText('V\u2091ff', MARGIN - 4, 12);

        // Draw curve
        c.beginPath();
        c.lineWidth = 1.5;
        let started = false;
        for (let i = 0; i < N_SAMPLES; i++) {
            const v = this._vBuf[i];
            if (!isFinite(v)) continue;
            const x = MARGIN + (i / (N_SAMPLES - 1)) * plotW;
            const y = (ps - MARGIN) - ((v - vMin) / vSpan) * plotH;
            if (!started) { c.moveTo(x, y); started = true; }
            else c.lineTo(x, y);
        }
        c.strokeStyle = '#5C92A8CC';
        c.stroke();

        // Current position marker
        const rNorm = (this.currentR - this._rMin) / (this._rMax - this._rMin);
        if (rNorm >= 0 && rNorm <= 1 && isFinite(this.currentV)) {
            const cx = MARGIN + rNorm * plotW;
            const cy = (ps - MARGIN) - ((this.currentV - vMin) / vSpan) * plotH;
            c.fillStyle = '#FE3B01';
            c.beginPath();
            c.arc(cx, cy, 4, 0, TWO_PI);
            c.fill();
        }
    }
}
