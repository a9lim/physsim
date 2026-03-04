import Vec2 from './vec2.js';

export default class Photon {
    constructor(x, y, vx, vy, energy) {
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(vx, vy); // direction, |v| = 1 (c)
        this.energy = energy;
        this.lifetime = 0;
        this.alive = true;
    }

    update(dt) {
        this.pos.x += this.vel.x * dt;
        this.pos.y += this.vel.y * dt;
        this.lifetime += dt;
    }
}
