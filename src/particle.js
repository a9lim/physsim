import Vec2 from './vec2.js';
import { HISTORY_SIZE } from './config.js';

const _PAL = window._PALETTE;

export default class Particle {
    static nextId = 0;

    constructor(x, y, mass = 10, charge = 0) {
        this.id = Particle.nextId++;
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(0, 0);
        this.w = new Vec2(0, 0);    // proper velocity (gamma*v)
        this.force = new Vec2(0, 0);
        this.prevForce = new Vec2(0, 0);
        this.forceGravity = new Vec2(0, 0);
        this.forceCoulomb = new Vec2(0, 0);
        this.forceMagnetic = new Vec2(0, 0);
        this.forceGravitomag = new Vec2(0, 0);
        this.force1PN = new Vec2(0, 0);
        this.forceSpinCurv = new Vec2(0, 0);
        this.forceRadiation = new Vec2(0, 0);
        this.torqueSpinOrbit = 0;
        this.torqueFrameDrag = 0;
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

        // Signal delay history (lazy-allocated)
        this.histX = null;
        this.histY = null;
        this.histVx = null;
        this.histVy = null;
        this.histTime = null;
        this.histHead = 0;
        this.histCount = 0;

        this.radius = Math.cbrt(this.mass);
        this.color = this.getColor();
    }

    getColor() {
        if (this.charge === 0) return _PAL.neutral;

        const intensity = Math.min(Math.abs(this.charge) / 20, 1.0);
        const hue = this.charge > 0 ? _PAL.chargePos : _PAL.chargeNeg;
        const sat = 50 + 50 * intensity;
        const light = 60 - 20 * intensity;

        return `hsl(${hue}, ${sat}%, ${light}%)`;
    }

    updateColor() {
        this.radius = Math.cbrt(this.mass);
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
