import Vec2 from './vec2.js';

export default class Particle {
    constructor(x, y, mass = 10, charge = 0) {
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(0, 0); // Newtonian velocity (for display/approx)
        this.acc = new Vec2(0, 0);
        this.momentum = new Vec2(0, 0); // Relativistic momentum

        this.mass = mass; // Rest mass
        this.charge = charge;
        this.spin = 0; // Angular momentum

        this.radius = Math.sqrt(this.mass) * 3; // Visual size based on mass
        this.color = this.getColor();
    }

    getColor() {
        // Continuous color based on charge
        // Neutral (0) -> value near 200 (grey/white)
        // +Charge -> Blue (240 hue)
        // -Charge -> Red (0 hue)
        // Saturation/Lightness can vary with magnitude 

        if (this.charge === 0) return '#bdc3c7'; // Grey

        const maxChargeDisplay = 20; // Charge at which color is max saturated
        const intensity = Math.min(Math.abs(this.charge) / maxChargeDisplay, 1.0);

        // Use HSL
        // Blue: 220-240, Red: 0-20 or 340-360
        // Let's use simple interpolation: 
        // 0 intensity -> Grey (sat 0)
        // 1 intensity -> Max Sat

        const hue = this.charge > 0 ? 220 : 10; // Blue or Red
        const sat = 50 + 50 * intensity; // 50% to 100%
        const light = 60 - 20 * intensity; // 60% down to 40% (darker at high charge)

        return `hsl(${hue}, ${sat}%, ${light}%)`;
    }

    updateColor() {
        this.radius = Math.sqrt(this.mass) * 3; // Larger visual size
        this.color = this.getColor();
    }
}
