import Vec2 from './vec2.js';

const _PAL = window._PALETTE;

export default class Particle {
    static nextId = 0;

    constructor(x, y, mass = 10, charge = 0) {
        this.id = Particle.nextId++;
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(0, 0);
        this.w = new Vec2(0, 0);    // proper velocity (γv, unbounded state variable)
        this.force = new Vec2(0, 0);
        this.forceGravity = new Vec2(0, 0);
        this.forceCoulomb = new Vec2(0, 0);
        this.forceMagnetic = new Vec2(0, 0);
        this.forceGravitomag = new Vec2(0, 0);

        this.mass = mass;
        this.charge = charge;
        this.spin = 0;      // proper angular velocity (unbounded state variable)
        this.angVel = 0;    // angular velocity (derived, like vel from w)
        this.torque = 0;
        this.torqueMagnetic = 0;
        this.torqueGravitomag = 0;

        // Accumulated magnetic field z-components (for Boris rotation)
        this.Bz = 0;            // EM magnetic field from moving charges
        this.Bgz = 0;           // Gravitomagnetic field from moving masses

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
}
