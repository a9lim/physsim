// ─── Collision Detection & Resolution ───

import { INERTIA_K } from './config.js';
import { setVelocity, angwToAngVel, angVelToAngw } from './relativity.js';

/**
 * Helper: set particle velocity from normal/tangential components.
 * Only used by resolveBounce (classical path).
 */
function setVelocityFromVel(p, vn, vt, nx, ny, tx, ty) {
    setVelocity(p, nx * vn + tx * vt, ny * vn + ty * vt);
}

/**
 * Detect and resolve collisions between all particle pairs.
 *
 * @param {Array} particles
 * @param {Object} pool - QuadTreePool
 * @param {number} root - Root node index
 * @param {string} mode - 'merge' or 'bounce'
 * @param {number} bounceFriction - Tangential friction coefficient
 * @param {boolean} relativityEnabled
 */
export function handleCollisions(particles, pool, root, mode, bounceFriction, relativityEnabled) {
    for (const p1 of particles) {
        if (p1.mass === 0) continue;

        const candidates = pool.query(root, p1.pos.x, p1.pos.y, p1.radius * 2, p1.radius * 2);

        for (const p2 of candidates) {
            if (p1 === p2 || p2.mass === 0 || p1.id >= p2.id) continue;

            const dist = p1.pos.dist(p2.pos);
            const minDist = p1.radius + p2.radius;

            if (dist < minDist) {
                if (mode === 'merge') {
                    resolveMerge(p1, p2, relativityEnabled);
                } else if (mode === 'bounce') {
                    resolveBounce(p1, p2, minDist, dist, bounceFriction, relativityEnabled);
                }
            }
        }
    }

    if (mode === 'merge') {
        let write = 0;
        for (let read = 0; read < particles.length; read++) {
            if (particles[read].mass !== 0) {
                particles[write++] = particles[read];
            }
        }
        particles.length = write;
    }
}

/**
 * Merge two overlapping particles, conserving mass, charge, momentum, and angular momentum.
 *
 * @param {Object} p1 - Surviving particle
 * @param {Object} p2 - Absorbed particle (mass set to 0)
 * @param {boolean} relativityEnabled
 */
export function resolveMerge(p1, p2, relativityEnabled) {
    const totalMass = p1.mass + p2.mass;
    // Conserve linear momentum: p = m*w, so w_new = (m1*w1 + m2*w2) / totalMass
    const newWx = (p1.mass * p1.w.x + p2.mass * p2.w.x) / totalMass;
    const newWy = (p1.mass * p1.w.y + p2.mass * p2.w.y) / totalMass;
    const newX = (p1.pos.x * p1.mass + p2.pos.x * p2.mass) / totalMass;
    const newY = (p1.pos.y * p1.mass + p2.pos.y * p2.mass) / totalMass;

    // Conserve angular momentum: orbital(about pair COM) + spin → merged spin
    // I = INERTIA_K * m * r² (uniform-density solid sphere)
    const dx1 = p1.pos.x - newX, dy1 = p1.pos.y - newY;
    const dx2 = p2.pos.x - newX, dy2 = p2.pos.y - newY;
    const Lorb = dx1 * (p1.mass * p1.w.y) - dy1 * (p1.mass * p1.w.x)
        + dx2 * (p2.mass * p2.w.y) - dy2 * (p2.mass * p2.w.x);
    const Lspin = INERTIA_K * p1.mass * p1.radius * p1.radius * p1.angw
        + INERTIA_K * p2.mass * p2.radius * p2.radius * p2.angw;

    p1.mass = totalMass;
    p1.charge = p1.charge + p2.charge;
    p1.w.set(newWx, newWy);
    p1.pos.set(newX, newY);
    p1.updateColor(); // updates radius = cbrt(totalMass)

    const newI = INERTIA_K * totalMass * p1.radius * p1.radius;
    p1.angw = (Lorb + Lspin) / newI;
    p1.angVel = relativityEnabled ? angwToAngVel(p1.angw, p1.radius) : p1.angw;

    // Re-derive velocity from proper velocity
    const invG = relativityEnabled ? 1 / Math.sqrt(1 + p1.w.magSq()) : 1;
    p1.vel.x = p1.w.x * invG;
    p1.vel.y = p1.w.y * invG;

    p2.mass = 0;
}

/**
 * Elastic bounce between two particles with spin friction.
 * Relativistic path uses Lorentz boost to COM frame; classical path uses standard elastic formulas.
 *
 * @param {Object} p1 - First particle
 * @param {Object} p2 - Second particle
 * @param {number} minDist - Sum of radii
 * @param {number} dist - Current distance
 * @param {number} bounceFriction - Tangential friction coefficient
 * @param {boolean} relativityEnabled
 */
export function resolveBounce(p1, p2, minDist, dist, bounceFriction, relativityEnabled) {
    const safeDist = dist === 0 ? 0.0001 : dist;

    let nx, ny;
    if (dist === 0) {
        nx = Math.random() - 0.5;
        ny = Math.random() - 0.5;
        const m = Math.sqrt(nx * nx + ny * ny);
        nx /= m; ny /= m;
    } else {
        nx = (p2.pos.x - p1.pos.x) / safeDist;
        ny = (p2.pos.y - p1.pos.y) / safeDist;
    }

    const tx = -ny, ty = nx;
    const m1 = p1.mass, m2 = p2.mass;
    const mSum = m1 + m2;

    if (relativityEnabled) {
        // ─── Relativistic elastic bounce ───
        // Conserves both relativistic momentum (m·w) and energy (m·γ).
        // Uses Lorentz boost to COM frame, reversal, and boost back.

        // Decompose proper velocities into normal/tangential
        const w1n = p1.w.x * nx + p1.w.y * ny;
        const w1t = p1.w.x * tx + p1.w.y * ty;
        const w2n = p2.w.x * nx + p2.w.y * ny;
        const w2t = p2.w.x * tx + p2.w.y * ty;

        // Approaching check using coordinate velocity
        const v1n = p1.vel.x * nx + p1.vel.y * ny;
        const v2n = p2.vel.x * nx + p2.vel.y * ny;
        if (v2n - v1n > 0) return;

        // Full Lorentz factors (including tangential components)
        const g1 = Math.sqrt(1 + w1n * w1n + w1t * w1t);
        const g2 = Math.sqrt(1 + w2n * w2n + w2t * w2t);

        // Total normal momentum and energy
        const Pn = m1 * w1n + m2 * w2n;
        const E = m1 * g1 + m2 * g2;

        // Invariant mass of the system
        const MSq = E * E - Pn * Pn;
        const M = Math.sqrt(MSq);

        // COM boost parameters (along normal direction)
        const Gc = E / M;   // COM Lorentz factor
        const Wc = Pn / M;  // COM proper velocity along normal

        // Boost each particle's normal component to COM frame
        const w1nc = Gc * w1n - Wc * g1;
        const g1c = Gc * g1 - Wc * w1n;
        const w2nc = Gc * w2n - Wc * g2;

        // Elastic collision in COM frame: reverse normal proper velocities
        // Then boost back to lab frame
        const w1nFinal = -Gc * w1nc + Wc * g1c;
        // g2c = Gc*g2 - Wc*w2n, but we can use momentum conservation instead
        const w2nFinal = (Pn - m1 * w1nFinal) / m2;

        // Tangential friction using coordinate velocities for surface velocity
        const v1t = p1.vel.x * tx + p1.vel.y * ty;
        const v2t = p2.vel.x * tx + p2.vel.y * ty;
        const surfaceV1 = v1t + p1.angVel * p1.radius;
        const surfaceV2 = v2t - p2.angVel * p2.radius;
        const effectiveMass = (m1 * m2) / mSum;
        const tangentialImpulse = bounceFriction * (surfaceV1 - surfaceV2) * effectiveMass;

        // Apply tangential impulse to proper velocity
        const w1tFinal = w1t - tangentialImpulse / m1;
        const w2tFinal = w2t + tangentialImpulse / m2;

        // Spin friction: compute new coordinate ω, then convert to angular celerity
        const I1 = INERTIA_K * m1 * p1.radius * p1.radius;
        const I2 = INERTIA_K * m2 * p2.radius * p2.radius;
        const omega1New = p1.angVel - tangentialImpulse / I1;
        const omega2New = p2.angVel - tangentialImpulse / I2;
        p1.angw = angVelToAngw(omega1New, p1.radius);
        p2.angw = angVelToAngw(omega2New, p2.radius);
        p1.angVel = angwToAngVel(p1.angw, p1.radius);
        p2.angVel = angwToAngVel(p2.angw, p2.radius);

        // Set proper velocity, derive coordinate velocity
        p1.w.set(nx * w1nFinal + tx * w1tFinal, ny * w1nFinal + ty * w1tFinal);
        p2.w.set(nx * w2nFinal + tx * w2tFinal, ny * w2nFinal + ty * w2tFinal);
        const invG1 = 1 / Math.sqrt(1 + p1.w.magSq());
        const invG2 = 1 / Math.sqrt(1 + p2.w.magSq());
        p1.vel.set(p1.w.x * invG1, p1.w.y * invG1);
        p2.vel.set(p2.w.x * invG2, p2.w.y * invG2);
    } else {
        // ─── Classical bounce: conserve m·v ───
        const v1n = p1.vel.x * nx + p1.vel.y * ny;
        const v1t = p1.vel.x * tx + p1.vel.y * ty;
        const v2n = p2.vel.x * nx + p2.vel.y * ny;
        const v2t = p2.vel.x * tx + p2.vel.y * ty;

        if (v2n - v1n > 0) return;

        const v1nFinal = (v1n * (m1 - m2) + 2 * m2 * v2n) / mSum;
        const v2nFinal = (v2n * (m2 - m1) + 2 * m1 * v1n) / mSum;

        const surfaceV1 = v1t + p1.angVel * p1.radius;
        const surfaceV2 = v2t - p2.angVel * p2.radius;
        const effectiveMass = (m1 * m2) / mSum;
        const tangentialImpulse = bounceFriction * (surfaceV1 - surfaceV2) * effectiveMass;

        const v1tFinal = v1t - tangentialImpulse / m1;
        const v2tFinal = v2t + tangentialImpulse / m2;

        const I1 = INERTIA_K * m1 * p1.radius * p1.radius;
        const I2 = INERTIA_K * m2 * p2.radius * p2.radius;
        p1.angw -= tangentialImpulse / I1;
        p2.angw -= tangentialImpulse / I2;
        p1.angVel = p1.angw;
        p2.angVel = p2.angw;

        setVelocityFromVel(p1, v1nFinal, v1tFinal, nx, ny, tx, ty);
        setVelocityFromVel(p2, v2nFinal, v2tFinal, nx, ny, tx, ty);
    }

    const overlap = (minDist - safeDist) / 2 + minDist * 0.01;
    p1.pos.x -= nx * overlap;
    p1.pos.y -= ny * overlap;
    p2.pos.x += nx * overlap;
    p2.pos.y += ny * overlap;
}
