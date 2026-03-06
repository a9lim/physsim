// ─── Collision Detection & Resolution ───
// Quadtree-accelerated overlap detection with merge or bounce resolution.

import { INERTIA_K, COLLISION_SAFE_DIST, OVERLAP_FACTOR, EPSILON } from './config.js';
import { setVelocity, angwToAngVel } from './relativity.js';
import { TORUS, minImage, wrapPosition } from './topology.js';

const _miOut = { x: 0, y: 0 };

/** Reconstruct velocity from (n,t) components. Used by classical bounce path. */
function setVelocityFromVel(p, vn, vt, nx, ny, tx, ty) {
    setVelocity(p, nx * vn + tx * vt, ny * vn + ty * vt);
}

/** Detect overlaps via quadtree query and resolve as merge or bounce. */
export function handleCollisions(particles, pool, root, mode, bounceFriction, relativityEnabled, periodic, domW, domH, topology = TORUS) {
    const halfDomW = domW * 0.5;
    const halfDomH = domH * 0.5;

    for (let ci = 0; ci < particles.length; ci++) {
        const p1 = particles[ci];
        if (p1.mass === 0) continue;

        const candidates = pool.queryReuse(root, p1.pos.x, p1.pos.y, p1.radius * 2, p1.radius * 2);

        for (let ck = 0; ck < candidates.length; ck++) {
            const p2 = candidates[ck];
            // Ghosts are periodic images; resolve against the real particle
            const real2 = p2.isGhost ? p2.original : p2;
            if (p1 === real2 || real2.mass === 0 || p1.id >= real2.id) continue;

            let dx, dy;
            if (periodic) {
                minImage(p1.pos.x, p1.pos.y, p2.pos.x, p2.pos.y, topology, domW, domH, halfDomW, halfDomH, _miOut);
                dx = _miOut.x; dy = _miOut.y;
            } else {
                dx = p2.pos.x - p1.pos.x; dy = p2.pos.y - p1.pos.y;
            }
            const dist = Math.sqrt(dx * dx + dy * dy);
            const minDist = p1.radius + real2.radius;

            if (dist < minDist) {
                if (mode === 'merge') {
                    resolveMerge(p1, real2, relativityEnabled, periodic, dx, dy);
                    if (periodic) {
                        wrapPosition(p1, topology, domW, domH);
                    }
                } else if (mode === 'bounce') {
                    resolveBounce(p1, real2, minDist, dist, bounceFriction, relativityEnabled, dx, dy);
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

/** Merge p2 into p1, conserving mass, charge, linear and angular momentum. */
export function resolveMerge(p1, p2, relativityEnabled, periodic, miDx, miDy) {
    const totalMass = p1.mass + p2.mass;
    const newWx = (p1.mass * p1.w.x + p2.mass * p2.w.x) / totalMass;
    const newWy = (p1.mass * p1.w.y + p2.mass * p2.w.y) / totalMass;
    // Use minimum-image offset so periodic p2 position is relative to p1
    const p2miX = p1.pos.x + miDx;
    const p2miY = p1.pos.y + miDy;
    const newX = (p1.pos.x * p1.mass + p2miX * p2.mass) / totalMass;
    const newY = (p1.pos.y * p1.mass + p2miY * p2.mass) / totalMass;

    // Orbital L about merged COM + spin L -> new spin
    const dx1 = p1.pos.x - newX, dy1 = p1.pos.y - newY;
    const dx2 = p2miX - newX, dy2 = p2miY - newY;
    const Lorb = dx1 * (p1.mass * p1.w.y) - dy1 * (p1.mass * p1.w.x)
        + dx2 * (p2.mass * p2.w.y) - dy2 * (p2.mass * p2.w.x);
    const Lspin = INERTIA_K * p1.mass * p1.radius * p1.radius * p1.angw
        + INERTIA_K * p2.mass * p2.radius * p2.radius * p2.angw;

    p1.mass = totalMass;
    p1.charge = p1.charge + p2.charge;
    p1.w.set(newWx, newWy);
    p1.pos.set(newX, newY);
    p1.updateColor();

    const newI = INERTIA_K * totalMass * p1.radius * p1.radius;
    p1.angw = (Lorb + Lspin) / newI;
    p1.angVel = relativityEnabled ? angwToAngVel(p1.angw, p1.radius) : p1.angw;

    const invG = relativityEnabled ? 1 / Math.sqrt(1 + p1.w.x * p1.w.x + p1.w.y * p1.w.y) : 1;
    p1.vel.x = p1.w.x * invG;
    p1.vel.y = p1.w.y * invG;

    p2.mass = 0;
}

/** Elastic bounce with spin friction. Relativistic: Lorentz boost to COM frame. */
export function resolveBounce(p1, p2, minDist, dist, bounceFriction, relativityEnabled, miDx, miDy) {
    const safeDist = dist === 0 ? COLLISION_SAFE_DIST : dist;

    let nx, ny;
    if (dist === 0) {
        nx = Math.random() - 0.5;
        ny = Math.random() - 0.5;
        const m = Math.sqrt(nx * nx + ny * ny);
        if (m > 0) { nx /= m; ny /= m; }
        else { nx = 1; ny = 0; }
    } else {
        nx = miDx / safeDist;
        ny = miDy / safeDist;
    }

    const tx = -ny, ty = nx;
    const m1 = p1.mass, m2 = p2.mass;
    const mSum = m1 + m2;

    if (relativityEnabled) {
        // ─── Relativistic elastic bounce ───
        // Boost to COM frame along collision normal, reverse, boost back.
        const w1n = p1.w.x * nx + p1.w.y * ny;
        const w1t = p1.w.x * tx + p1.w.y * ty;
        const w2n = p2.w.x * nx + p2.w.y * ny;
        const w2t = p2.w.x * tx + p2.w.y * ty;

        // Only resolve if approaching
        const v1n = p1.vel.x * nx + p1.vel.y * ny;
        const v2n = p2.vel.x * nx + p2.vel.y * ny;
        if (v2n - v1n > 0) return;

        const g1 = Math.sqrt(1 + w1n * w1n + w1t * w1t);
        const g2 = Math.sqrt(1 + w2n * w2n + w2t * w2t);

        const Pn = m1 * w1n + m2 * w2n;
        const E = m1 * g1 + m2 * g2;
        const M = Math.sqrt(Math.max(0, E * E - Pn * Pn)); // invariant mass; max(0,...) guards numerical underflow at near-c

        // Degenerate invariant mass (collinear ultra-relativistic): COM frame is undefined
        if (M < EPSILON) return;

        // COM frame boost parameters along normal
        const Gc = E / M;
        const Wc = Pn / M;

        // Boost -> reverse normal -> boost back
        const w1nc = Gc * w1n - Wc * g1;
        const g1c = Gc * g1 - Wc * w1n;
        const w2nc = Gc * w2n - Wc * g2;
        const w1nFinal = -Gc * w1nc + Wc * g1c;
        const w2nFinal = (Pn - m1 * w1nFinal) / m2; // momentum conservation

        // Tangential friction: surface velocity = v_tan + omega*r
        const v1t = p1.vel.x * tx + p1.vel.y * ty;
        const v2t = p2.vel.x * tx + p2.vel.y * ty;
        const surfaceV1 = v1t + p1.angVel * p1.radius;
        const surfaceV2 = v2t - p2.angVel * p2.radius;
        const effectiveMass = (m1 * m2) / mSum;
        const tangentialImpulse = bounceFriction * (surfaceV1 - surfaceV2) * effectiveMass;

        const w1tFinal = w1t - tangentialImpulse / m1;
        const w2tFinal = w2t + tangentialImpulse / m2;

        // Spin impulse: angular impulse = r × J at contact point
        // Update angw directly (L = I·angw is the conserved spin angular momentum)
        p1.angw -= tangentialImpulse / (INERTIA_K * m1 * p1.radius);
        p2.angw -= tangentialImpulse / (INERTIA_K * m2 * p2.radius);
        p1.angVel = angwToAngVel(p1.angw, p1.radius);
        p2.angVel = angwToAngVel(p2.angw, p2.radius);

        p1.w.set(nx * w1nFinal + tx * w1tFinal, ny * w1nFinal + ty * w1tFinal);
        p2.w.set(nx * w2nFinal + tx * w2tFinal, ny * w2nFinal + ty * w2tFinal);
        const invG1 = 1 / Math.sqrt(1 + p1.w.x * p1.w.x + p1.w.y * p1.w.y);
        const invG2 = 1 / Math.sqrt(1 + p2.w.x * p2.w.x + p2.w.y * p2.w.y);
        p1.vel.set(p1.w.x * invG1, p1.w.y * invG1);
        p2.vel.set(p2.w.x * invG2, p2.w.y * invG2);
    } else {
        // ─── Classical elastic bounce ───
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

        p1.angw -= tangentialImpulse / (INERTIA_K * m1 * p1.radius);
        p2.angw -= tangentialImpulse / (INERTIA_K * m2 * p2.radius);
        p1.angVel = p1.angw;
        p2.angVel = p2.angw;

        setVelocityFromVel(p1, v1nFinal, v1tFinal, nx, ny, tx, ty);
        setVelocityFromVel(p2, v2nFinal, v2tFinal, nx, ny, tx, ty);
    }

    const overlap = (minDist - safeDist) / 2 + minDist * OVERLAP_FACTOR;
    p1.pos.x -= nx * overlap;
    p1.pos.y -= ny * overlap;
    p2.pos.x += nx * overlap;
    p2.pos.y += ny * overlap;
}
