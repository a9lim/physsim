// ─── Relativistic Helpers (c = 1) ───

import { MAX_SPEED_RATIO } from './config.js';

/**
 * Derive angular velocity from angular celerity via rotational Lorentz factor.
 * ω = W / √(1 + W²r²), naturally caps surface velocity |ωr| < c.
 */
export function angwToAngVel(angw, radius) {
    return angw / Math.sqrt(1 + angw * angw * radius * radius);
}

/**
 * Derive angular celerity from angular velocity (inverse of angwToAngVel).
 * W = ω / √(1 - ω²r²), analogous to w = v / √(1 - v²).
 */
export function angVelToAngw(angVel, radius) {
    const sr = angVel * radius;
    const srSq = sr * sr;
    if (srSq >= 1) {
        const clampedSr = MAX_SPEED_RATIO;
        return Math.sign(angVel) * clampedSr / (radius * Math.sqrt(1 - clampedSr * clampedSr));
    }
    return angVel / Math.sqrt(1 - srSq);
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
