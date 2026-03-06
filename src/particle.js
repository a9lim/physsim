import Vec2 from './vec2.js';
import { HISTORY_SIZE, INERTIA_K } from './config.js';

const _PAL = window._PALETTE;
const _hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const _nRGB = _hex(_PAL.neutral);
const _posRGB = _hex(_PAL.extended.red);
const _negRGB = _hex(_PAL.extended.blue);

export default class Particle {
    static nextId = 0;

    constructor(x, y, mass = 10, charge = 0) {
        this.id = Particle.nextId++;
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(0, 0);
        this.w = new Vec2(0, 0);    // proper velocity (gamma*v)
        this.force = new Vec2(0, 0);
        this.jerk = new Vec2(0, 0);    // analytical dF/dt for radiation reaction
        this.forceGravity = new Vec2(0, 0);
        this.forceCoulomb = new Vec2(0, 0);
        this.forceMagnetic = new Vec2(0, 0);
        this.forceGravitomag = new Vec2(0, 0);
        this.force1PN = new Vec2(0, 0);
        this.forceSpinCurv = new Vec2(0, 0);
        this.forceRadiation = new Vec2(0, 0);
        this.forceYukawa = new Vec2(0, 0);
        this.torqueSpinOrbit = 0;
        this.torqueFrameDrag = 0;
        this.torqueTidal = 0;
        this._f1pnOld = { x: 0, y: 0 };

        this.mass = mass;
        this.charge = charge;
        this.angw = 0;      // angular celerity (unbounded)
        this.angVel = 0;    // angular velocity (derived)

        // Accumulated B fields (Boris rotation) and gradients (spin-orbit)
        this.Bz = 0;
        this.Bgz = 0;
        this.dBzdx = 0;
        this.dBzdy = 0;
        this.dBgzdx = 0;
        this.dBgzdy = 0;
        this._tidalTorque = 0;
        this._frameDragTorque = 0;

        // Radiation accumulators (used by integrator)
        this._radAccum = 0;
        this._hawkAccum = 0;
        this._radDisplayX = 0;
        this._radDisplayY = 0;
        // 3-point backward difference history for non-1/r² force jerk
        this._otherFx0 = 0; this._otherFy0 = 0; // two substeps ago
        this._otherFx1 = 0; this._otherFy1 = 0; // previous substep
        this._otherDt0 = 0; // dt between sample 0 and 1
        this._otherDt1 = 0; // dt between sample 1 and current
        this._otherCount = 0;
        // Per-frame residual force history for quadrupole jerk (backward diff)
        this._qResFx0 = 0; this._qResFy0 = 0;
        this._qResFx1 = 0; this._qResFy1 = 0;
        this._qResCount = 0;

        // Signal delay history (lazy-allocated)
        this.histX = null;
        this.histY = null;
        this.histVx = null;
        this.histVy = null;
        this.histTime = null;
        this.histHead = 0;
        this.histCount = 0;

        this.radius = Math.cbrt(this.mass);
        this.radiusSq = this.radius * this.radius;
        this.invMass = 1 / this.mass;
        this.color = this.getColor();
    }

    getColor() {
        const t = Math.max(-1, Math.min(1, this.charge / 5));
        const absT = Math.abs(t);
        if (absT < 1e-6) return _PAL.neutral;
        const tgt = t > 0 ? _posRGB : _negRGB;
        const r = Math.round(_nRGB[0] + (tgt[0] - _nRGB[0]) * absT);
        const g = Math.round(_nRGB[1] + (tgt[1] - _nRGB[1]) * absT);
        const b = Math.round(_nRGB[2] + (tgt[2] - _nRGB[2]) * absT);
        return `rgb(${r},${g},${b})`;
    }

    updateColor() {
        const bh = window.sim && window.sim.physics.blackHoleEnabled;
        if (bh) {
            const M = this.mass;
            const I = INERTIA_K * M * this.radiusSq;
            const omega = this.angVel || 0;
            const a = I * Math.abs(omega) / M;  // spin parameter J/M
            const Q = this.charge;
            const disc = M * M - a * a - Q * Q;
            this.radius = disc > 0 ? M + Math.sqrt(disc) : M * 0.5; // naked singularity floor
        } else {
            this.radius = Math.cbrt(this.mass);
        }
        this.radiusSq = this.radius * this.radius;
        this.invMass = 1 / this.mass;
        this.color = this.getColor();
    }

    _initHistory() {
        if (this.histX) return;
        this.histX = new Float64Array(HISTORY_SIZE);
        this.histY = new Float64Array(HISTORY_SIZE);
        this.histVx = new Float64Array(HISTORY_SIZE);
        this.histVy = new Float64Array(HISTORY_SIZE);
        this.histTime = new Float64Array(HISTORY_SIZE);
        this.histHead = 0;
        this.histCount = 0;
    }
}
