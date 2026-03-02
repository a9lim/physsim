// ─── Relativistic Helpers (c = 1) ───

import { MAX_SPEED_RATIO } from './config.js';

/**
 * Derive angular velocity from proper angular velocity via rotational Lorentz factor.
 * ω = S / √(1 + S²r²), naturally caps surface velocity |ωr| < c.
 */
export function spinToAngVel(spin, radius) {
    return spin / Math.sqrt(1 + spin * spin * radius * radius);
}

/**
 * Set particle proper velocity from velocity components.
 * Clamps |v| < MAX_SPEED_RATIO, then sets p.vel and p.w = γv.
 */
export function setVelocity(p, vx, vy) {
    const speedSq = vx * vx + vy * vy;
    if (speedSq >= 1) {
        const s = MAX_SPEED_RATIO / Math.sqrt(speedSq);
        vx *= s;
        vy *= s;
    }
    const gamma = 1 / Math.sqrt(1 - vx * vx - vy * vy);
    p.vel.set(vx, vy);
    p.w.set(vx * gamma, vy * gamma);
}
