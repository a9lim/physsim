// ─── Energy & Momentum Computation ───
// Three-pass accumulator: KE + momentum, angular momentum about COM, Darwin field corrections.

import { INERTIA_K, SOFTENING_SQ } from './config.js';
import { TORUS, minImage } from './topology.js';

const _miOut = { x: 0, y: 0 };

/** Compute all conserved quantities for the stats display. */
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
            const gamma = Math.sqrt(1 + p.w.magSq());
            linearKE += (gamma - 1) * p.mass;
            // Spin KE via rotational Lorentz factor: I/r²·(γ_rot − 1)
            const srSq = p.angw * p.angw * rSq;
            spinKE += INERTIA_K * p.mass * (Math.sqrt(1 + srSq) - 1);
        } else {
            const speedSq = p.vel.x * p.vel.x + p.vel.y * p.vel.y;
            linearKE += 0.5 * p.mass * speedSq;
            spinKE += 0.5 * INERTIA_K * p.mass * rSq * p.angVel * p.angVel;
        }

        // w = v when rel off, so m·w works for both regimes
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
    // Accounts for energy and momentum stored in EM and gravitational fields.
    // EM and GM have opposite signs (GEM attractive convention).
    //
    // Gated on the velocity-dependent force toggles (magnetic / gravitomag),
    // not the base force toggles (Coulomb / gravity). Coulomb and gravity alone
    // conserve particle momentum exactly; the field corrections only matter when
    // the Lorentz / gravitomagnetic Lorentz-like forces (Boris rotation) are
    // active, since those are what move momentum into the field.
    //
    // Field ENERGY is further suppressed when 1PN is on for the corresponding
    // sector, because the 1PN PE (EIH / Darwin EM) already captures the same
    // correction. Field MOMENTUM is always computed — it's the canonical
    // momentum correction from the Darwin Lagrangian, needed regardless of
    // whether the Darwin force is applied.
    let fieldEnergy = 0;
    let fieldPx = 0, fieldPy = 0;
    const n = particles.length;
    const magneticOn = physics.magneticEnabled;
    const gmOn = physics.gravitomagEnabled;
    const emFieldEnergyOn = magneticOn && !physics.onePNEnabled;
    const gmFieldEnergyOn = gmOn && !(physics.onePNEnabled && physics.gravityEnabled);

    const periodic = physics.periodic;
    const domW = physics.domainW;
    const domH = physics.domainH;
    const halfDomW = domW * 0.5;
    const halfDomH = domH * 0.5;
    const topology = physics._topologyConst !== undefined ? physics._topologyConst : TORUS;

    if (magneticOn || gmOn) {
        for (let i = 0; i < n; i++) {
            const pi = particles[i];
            for (let j = i + 1; j < n; j++) {
                const pj = particles[j];
                let dx, dy;
                if (periodic) {
                    minImage(pi.pos.x, pi.pos.y, pj.pos.x, pj.pos.y, topology, domW, domH, halfDomW, halfDomH, _miOut);
                    dx = _miOut.x; dy = _miOut.y;
                } else {
                    dx = pj.pos.x - pi.pos.x; dy = pj.pos.y - pi.pos.y;
                }
                const rSq = dx * dx + dy * dy + SOFTENING_SQ;
                const invR = 1 / Math.sqrt(rSq);
                const rx = dx * invR, ry = dy * invR;
                const viDotVj = pi.vel.x * pj.vel.x + pi.vel.y * pj.vel.y;
                const viDotR = pi.vel.x * rx + pi.vel.y * ry;
                const vjDotR = pj.vel.x * rx + pj.vel.y * ry;
                const velTerm = viDotVj + viDotR * vjDotR;

                const svx = pi.vel.x + pj.vel.x, svy = pi.vel.y + pj.vel.y;
                const svDotR = svx * rx + svy * ry;

                if (magneticOn) {
                    const qqInvR = pi.charge * pj.charge * invR;
                    if (emFieldEnergyOn) fieldEnergy -= 0.5 * qqInvR * velTerm;
                    const coeff = qqInvR * 0.5;
                    fieldPx += coeff * (svx + rx * svDotR);
                    fieldPy += coeff * (svy + ry * svDotR);
                }

                if (gmOn) {
                    const mmInvR = pi.mass * pj.mass * invR;
                    if (gmFieldEnergyOn) fieldEnergy += 0.5 * mmInvR * velTerm;
                    const coeff = mmInvR * 0.5;
                    fieldPx -= coeff * (svx + rx * svDotR);
                    fieldPy -= coeff * (svy + ry * svDotR);
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
