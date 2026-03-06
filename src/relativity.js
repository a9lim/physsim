// ─── Relativistic Helpers ───
// Conversions between celerity (unbounded state variable) and velocity (capped at c=1).

import { MAX_SPEED_RATIO, EPSILON } from './config.js';

/** angw -> angVel: omega = W / sqrt(1 + W^2 r^2). */
export function angwToAngVel(angw, radius) {
    return angw / Math.sqrt(1 + angw * angw * radius * radius);
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
