// ─── Phase Space Plot ───
// r vs v_r relative to the most massive body; 512-sample ring buffer.
import { TWO_PI, PHASE_BUFFER_LEN, TORUS } from './config.js';
import { minImage } from './topology.js';

const _miOut = { x: 0, y: 0 };

const BUFFER_LEN = PHASE_BUFFER_LEN;
const MARGIN = 24;

export default class PhasePlot {
    constructor() {
        this.enabled = true;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.rBuf = new Float32Array(BUFFER_LEN);
        this.vrBuf = new Float32Array(BUFFER_LEN);
        this.head = 0;
        this.count = 0;
        this.trackedId = -1;
        // R11: Cached layout dimensions (avoids reflow from clientWidth reads)
        this._cachedWidth = 180;
        this._cachedDpr = devicePixelRatio || 1;
    }

    /** R11: Call on resize to refresh cached dimensions. */
    cacheSize() {
        this._cachedWidth = this.canvas.clientWidth || 180;
        this._cachedDpr = devicePixelRatio || 1;
    }

    update(particles, selectedParticle, physics) {
        if (!this.enabled || !selectedParticle) return;

        const sel = selectedParticle;
        if (sel.id !== this.trackedId) {
            this.trackedId = sel.id;
            this.head = 0;
            this.count = 0;
        }

        // Radial quantities measured relative to most massive body
        let refX = 0, refY = 0, refVx = 0, refVy = 0, maxM = 0;
        for (const p of particles) {
            if (p === sel) continue;
            if (p.mass > maxM) {
                maxM = p.mass;
                refX = p.pos.x; refY = p.pos.y;
                refVx = p.vel.x; refVy = p.vel.y;
            }
        }

        let dx, dy;
        if (physics && physics.periodic) {
            minImage(refX, refY, sel.pos.x, sel.pos.y,
                     physics._topologyConst || TORUS, physics.domainW, physics.domainH,
                     physics.domainW * 0.5, physics.domainH * 0.5, _miOut);
            dx = _miOut.x; dy = _miOut.y;
        } else {
            dx = sel.pos.x - refX; dy = sel.pos.y - refY;
        }
        const r = Math.sqrt(dx * dx + dy * dy) || 1;
        const rx = dx / r, ry = dy / r;
        const dvx = sel.vel.x - refVx, dvy = sel.vel.y - refVy;
        const vr = dvx * rx + dvy * ry;

        this.rBuf[this.head] = r;
        this.vrBuf[this.head] = vr;
        this.head = (this.head + 1) % BUFFER_LEN;
        if (this.count < BUFFER_LEN) this.count++;
    }

    draw(isLight) {
        if (!this.enabled || this.count < 2) return;

        const dpr = this._cachedDpr;
        const ps = this._cachedWidth;
        const pxW = Math.round(ps * dpr);
        if (this.canvas.width !== pxW || this.canvas.height !== pxW) {
            this.canvas.width = pxW;
            this.canvas.height = pxW;
        }

        const c = this.ctx;
        c.setTransform(dpr, 0, 0, dpr, 0, 0);

        let rMin = Infinity, rMax = -Infinity, vrMin = Infinity, vrMax = -Infinity;
        for (let i = 0; i < this.count; i++) {
            const idx = (this.head - this.count + i + BUFFER_LEN) % BUFFER_LEN;
            const r = this.rBuf[idx], vr = this.vrBuf[idx];
            if (r < rMin) rMin = r; if (r > rMax) rMax = r;
            if (vr < vrMin) vrMin = vr; if (vr > vrMax) vrMax = vr;
        }
        const rRange = (rMax - rMin) || 1;
        const vrRange = (vrMax - vrMin) || 1;

        c.clearRect(0, 0, ps, ps);

        c.fillStyle = isLight ? '#FCF7F244' : '#0C0B0988';
        c.fillRect(0, 0, ps, ps);

        c.strokeStyle = isLight ? '#1A161233' : '#E8DED433';
        c.lineWidth = 0.5;
        c.beginPath();
        c.moveTo(MARGIN, 0); c.lineTo(MARGIN, ps);
        c.moveTo(0, ps - MARGIN); c.lineTo(ps, ps - MARGIN);
        c.stroke();

        c.fillStyle = isLight ? '#1A161288' : '#E8DED488';
        c.font = '9px Noto Sans Mono';
        c.fillText('r', ps - 12, ps - MARGIN + 12);
        c.fillText('v\u1D63', MARGIN - 18, 12);

        c.beginPath();
        c.lineWidth = 1.2;
        for (let i = 0; i < this.count; i++) {
            const idx = (this.head - this.count + i + BUFFER_LEN) % BUFFER_LEN;
            const x = MARGIN + ((this.rBuf[idx] - rMin) / rRange) * (ps - MARGIN - 4);
            const y = (ps - MARGIN) - ((this.vrBuf[idx] - vrMin) / vrRange) * (ps - MARGIN - 4);
            if (i === 0) {
                c.moveTo(x, y);
            } else {
                c.lineTo(x, y);
            }
        }
        c.strokeStyle = '#FE3B01CC';
        c.stroke();

        const lastIdx = (this.head - 1 + BUFFER_LEN) % BUFFER_LEN;
        const cx = MARGIN + ((this.rBuf[lastIdx] - rMin) / rRange) * (ps - MARGIN - 4);
        const cy = (ps - MARGIN) - ((this.vrBuf[lastIdx] - vrMin) / vrRange) * (ps - MARGIN - 4);
        c.fillStyle = '#FE3B01';
        c.beginPath();
        c.arc(cx, cy, 3, 0, TWO_PI);
        c.fill();
    }
}
