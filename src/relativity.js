// ─── Relativistic Helpers ───
// Conversions between celerity (unbounded state variable) and velocity (capped at c=1).

import { MAX_SPEED_RATIO, EPSILON } from './config.js';

/** angw -> angVel: omega = W / sqrt(1 + W^2 r^2). */
export function angwToAngVel(angw, radius) {
    return angw / Math.sqrt(1 + angw * angw * radius * radius);
}

/** angVel -> angw: W = omega / sqrt(1 - omega^2 r^2). Clamps at MAX_SPEED_RATIO. */
export function angVelToAngw(angVel, radius) {
    const sr = angVel * radius;
    const srSq = sr * sr;
    if (srSq >= 1) {
        const clampedSr = MAX_SPEED_RATIO;
        return Math.sign(angVel) * clampedSr / (radius * Math.sqrt(1 - clampedSr * clampedSr));
    }
    return angVel / Math.sqrt(1 - srSq);
}

/** Set p.vel and p.w from (vx,vy), clamping |v| < MAX_SPEED_RATIO. */
export function setVelocity(p, vx, vy) {
    const speedSq = vx * vx + vy * vy;
    if (speedSq >= 1) {
        const s = MAX_SPEED_RATIO / Math.sqrt(speedSq);
        vx *= s;
        vy *= s;
    }
    const gamma = 1 / Math.sqrt(Math.max(EPSILON, 1 - vx * vx - vy * vy));
    p.vel.set(vx, vy);
    p.w.set(vx * gamma, vy * gamma);
}
