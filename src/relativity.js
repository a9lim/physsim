// ─── Relativistic Helpers (c = 1) ───

import { MAX_SPEED_RATIO } from './config.js';

/** Lorentz factor from speed squared (v²/c²). */
export function gammaFromSpeed(speedSq) {
    return 1 / Math.sqrt(1 - speedSq);
}

/**
 * 1 / (mass * gamma) computed from momentum magnitude squared.
 * Multiply momentum components by this to get velocity: v = p * invMassGamma.
 */
export function invMassGamma(pMagSq, mass) {
    return 1 / (mass * Math.sqrt(1 + pMagSq / (mass * mass)));
}

/**
 * Set particle momentum from velocity components.
 * Clamps |v| < MAX_SPEED_RATIO, then sets p.vel and p.momentum.
 */
export function setMomentum(p, vx, vy) {
    const speedSq = vx * vx + vy * vy;
    if (speedSq >= 1) {
        const s = MAX_SPEED_RATIO / Math.sqrt(speedSq);
        vx *= s;
        vy *= s;
    }
    const gamma = gammaFromSpeed(vx * vx + vy * vy);
    p.vel.set(vx, vy);
    p.momentum.set(vx * gamma * p.mass, vy * gamma * p.mass);
}
