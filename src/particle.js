import Vec2 from './vec2.js';

export default class Particle {
    static nextId = 0;

    constructor(x, y, mass = 10, charge = 0) {
        this.id = Particle.nextId++;
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(0, 0);
        this.momentum = new Vec2(0, 0);

        this.mass = mass;
        this.charge = charge;
        this.spin = 0;

        this.radius = Math.sqrt(this.mass);
        this.color = this.getColor();
    }

    getColor() {
        if (this.charge === 0) return '#bdc3c7';

        const intensity = Math.min(Math.abs(this.charge) / 20, 1.0);
        const hue = this.charge > 0 ? 220 : 10;
        const sat = 50 + 50 * intensity;
        const light = 60 - 20 * intensity;

        return `hsl(${hue}, ${sat}%, ${light}%)`;
    }

    updateColor() {
        this.radius = Math.sqrt(this.mass);
        this.color = this.getColor();
    }
}
