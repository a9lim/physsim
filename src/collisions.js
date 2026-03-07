// ─── Collision Detection & Resolution ───
// Quadtree-accelerated overlap detection with merge resolution.

import { INERTIA_K } from './config.js';
import { angwToAngVel } from './relativity.js';
import { TORUS, minImage, wrapPosition } from './topology.js';

const _miOut = { x: 0, y: 0 };

/** Relativistic KE: wSq / (gamma + 1) * m. Avoids catastrophic cancellation at low v. */
function _particleKE(p) {
    const wSq = p.w.x * p.w.x + p.w.y * p.w.y;
    return wSq / (Math.sqrt(1 + wSq) + 1) * p.mass;
}

/** Detect overlaps via quadtree query and resolve as merge.
 *  Returns array of annihilation events [{x, y, energy, px, py}] for photon emission. */
export function handleCollisions(particles, pool, root, mode, bounceFriction, relativityEnabled, periodic, domW, domH, topology = TORUS) {
    const halfDomW = domW * 0.5;
    const halfDomH = domH * 0.5;
    const annihilations = [];
    const merges = [];

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
                // Annihilation: matter + antimatter -> energy
                if (p1.antimatter !== real2.antimatter && mode === 'merge') {
                    const annihilated = Math.min(p1.mass, real2.mass);
                    const cx = (p1.pos.x + real2.pos.x) * 0.5;
                    const cy = (p1.pos.y + real2.pos.y) * 0.5;
                    // Total momentum of annihilating mass
                    const apx = (p1.w.x + real2.w.x) * annihilated;
                    const apy = (p1.w.y + real2.w.y) * annihilated;
                    const fraction1 = annihilated / p1.mass;
                    const fraction2 = annihilated / real2.mass;
                    const keAnnihilated = fraction1 * _particleKE(p1) + fraction2 * _particleKE(real2);
                    annihilations.push({ x: cx, y: cy, energy: 2 * annihilated + keAnnihilated, px: apx, py: apy });
                    const origM1 = p1.mass, origM2 = real2.mass;
                    // Save pre-annihilation mass for signal delay retirement
                    p1._deathMass = origM1;
                    real2._deathMass = origM2;
                    p1.mass -= annihilated;
                    real2.mass -= annihilated;
                    if (origM1 > 0) p1.baseMass *= p1.mass / origM1;
                    if (origM2 > 0) real2.baseMass *= real2.mass / origM2;
                    p1.updateColor();
                    real2.updateColor();
                } else if (mode === 'merge') {
                    // Compute KE before merge for field excitation energy (relativistic)
                    const keBefore = _particleKE(p1) + _particleKE(real2);
                    const mx = (p1.pos.x * p1.mass + real2.pos.x * real2.mass) / (p1.mass + real2.mass);
                    const my = (p1.pos.y * p1.mass + real2.pos.y * real2.mass) / (p1.mass + real2.mass);
                    // Save pre-merge mass for signal delay retirement
                    real2._deathMass = real2.mass;
                    resolveMerge(p1, real2, relativityEnabled, periodic, dx, dy);
                    const keAfter = _particleKE(p1);
                    const keLost = Math.max(0, keBefore - keAfter);
                    if (keLost > 0) merges.push({ x: mx, y: my, energy: keLost });
                    if (periodic) {
                        wrapPosition(p1, topology, domW, domH);
                    }
                }
            }
        }
    }

    const removed = [];
    if (mode === 'merge') {
        let write = 0;
        for (let read = 0; read < particles.length; read++) {
            if (particles[read].mass !== 0) {
                particles[write++] = particles[read];
            } else {
                removed.push(particles[read]);
            }
        }
        particles.length = write;
    }

    return { annihilations, merges, removed };
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
    p1.baseMass = p1.baseMass + p2.baseMass;
    p2.baseMass = 0;
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

