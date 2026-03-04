// ─── Energy & Momentum Computation ───
import { INERTIA_K, SOFTENING_SQ } from './config.js';

/**
 * Compute all energy, momentum, and angular momentum quantities.
 * Returns object with all values needed for display.
 */
export function computeEnergies(particles, physics, sim) {
    let linearKE = 0;
    let spinKE = 0;
    let totalMass = 0;
    let comX = 0, comY = 0;
    let px = 0, py = 0;
    const relOn = physics.relativityEnabled;

    // ─── Pass 1: KE, momentum, COM ───
    for (const p of particles) {
        const rSq = p.radius * p.radius;
        if (relOn) {
            // Relativistic linear KE: (γ - 1)mc², γ = √(1 + w²)
            const gamma = Math.sqrt(1 + p.w.magSq());
            linearKE += (gamma - 1) * p.mass;
            // Relativistic spin KE: m_rot·(γ_rot - 1) where m_rot = I/r² = INERTIA_K·m
            const srSq = p.angw * p.angw * rSq;
            spinKE += INERTIA_K * p.mass * (Math.sqrt(1 + srSq) - 1);
        } else {
            const speedSq = p.vel.x * p.vel.x + p.vel.y * p.vel.y;
            linearKE += 0.5 * p.mass * speedSq;
            // Classical spin KE: ½Iω²
            spinKE += 0.5 * INERTIA_K * p.mass * rSq * p.angVel * p.angVel;
        }

        // Momentum: p = m·w (relativistic or classical, since w = v when rel off)
        px += p.mass * p.w.x;
        py += p.mass * p.w.y;

        totalMass += p.mass;
        comX += p.mass * p.pos.x;
        comY += p.mass * p.pos.y;
    }

    // ─── Pass 2: Angular momentum about COM ───
    let orbitalAngMom = 0;
    let spinAngMom = 0;
    if (totalMass > 0) {
        comX /= totalMass;
        comY /= totalMass;

        for (const p of particles) {
            const dx = p.pos.x - comX;
            const dy = p.pos.y - comY;
            orbitalAngMom += dx * (p.mass * p.w.y) - dy * (p.mass * p.w.x);
            spinAngMom += INERTIA_K * p.mass * p.radius * p.radius * p.angw;
        }
    }

    // ─── Pass 3: Darwin field energy & momentum (O(v²/c²) correction) ───
    // EM: U = -(1/2) Σ_{i<j} (qi·qj/r) [(vi·vj) + (vi·r̂)(vj·r̂)]
    // Grav: U = +(1/2) Σ_{i<j} (mi·mj/r) [(vi·vj) + (vi·r̂)(vj·r̂)]  (opposite sign)
    let fieldEnergy = 0;
    let fieldPx = 0, fieldPy = 0;
    const n = particles.length;
    const hasCoulomb = physics.coulombEnabled;
    const hasGM = physics.gravitomagEnabled;

    if (hasCoulomb || hasGM) {
        for (let i = 0; i < n; i++) {
            const pi = particles[i];
            for (let j = i + 1; j < n; j++) {
                const pj = particles[j];
                const dx = pj.pos.x - pi.pos.x;
                const dy = pj.pos.y - pi.pos.y;
                const rSq = dx * dx + dy * dy + SOFTENING_SQ;
                const invR = 1 / Math.sqrt(rSq);
                const rx = dx * invR, ry = dy * invR;
                const viDotVj = pi.vel.x * pj.vel.x + pi.vel.y * pj.vel.y;
                const viDotR = pi.vel.x * rx + pi.vel.y * ry;
                const vjDotR = pj.vel.x * rx + pj.vel.y * ry;
                const velTerm = viDotVj + viDotR * vjDotR;

                // Sum velocity for field momentum
                const svx = pi.vel.x + pj.vel.x, svy = pi.vel.y + pj.vel.y;
                const svDotR = svx * rx + svy * ry;

                if (hasCoulomb) {
                    const qqInvR = pi.charge * pj.charge * invR;
                    fieldEnergy -= 0.5 * qqInvR * velTerm;
                    const coeff = qqInvR * 0.5;
                    fieldPx += coeff * (svx + rx * svDotR);
                    fieldPy += coeff * (svy + ry * svDotR);
                }

                if (hasGM) {
                    const mmInvR = pi.mass * pj.mass * invR;
                    fieldEnergy += 0.5 * mmInvR * velTerm;
                    const coeff = mmInvR * 0.5;
                    fieldPx += coeff * (svx + rx * svDotR);
                    fieldPy += coeff * (svy + ry * svDotR);
                }
            }
        }
    }

    return {
        linearKE, spinKE,
        pe: physics.potentialEnergy,
        fieldEnergy, fieldPx, fieldPy,
        px, py,
        orbitalAngMom, spinAngMom,
        comX, comY,
    };
}
