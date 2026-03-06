// ─── Energy & Momentum Computation ───
// Three-pass accumulator: KE + momentum, angular momentum about COM, Darwin field corrections.

import { INERTIA_K, SOFTENING_SQ, BH_SOFTENING_SQ, AXION_G } from './config.js';
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
    const n = particles.length;

    // ─── Pass 1: KE, momentum, COM ───
    for (let i = 0; i < n; i++) {
        const p = particles[i];
        const pm = p.mass;
        const rSq = p.radiusSq;
        if (relOn) {
            // Use w²/(γ+1) instead of (γ−1) to avoid catastrophic cancellation when |w|≪1
            const wSq = p.w.x * p.w.x + p.w.y * p.w.y;
            const gamma = Math.sqrt(1 + wSq);
            linearKE += wSq / (gamma + 1) * pm;
            const srSq = p.angw * p.angw * rSq;
            const gammaRot = Math.sqrt(1 + srSq);
            spinKE += INERTIA_K * pm * srSq / (gammaRot + 1);
        } else {
            const speedSq = p.vel.x * p.vel.x + p.vel.y * p.vel.y;
            linearKE += 0.5 * pm * speedSq;
            spinKE += 0.5 * INERTIA_K * pm * rSq * p.angVel * p.angVel;
        }

        // w = v when rel off, so m·w works for both regimes
        px += pm * p.w.x;
        py += pm * p.w.y;

        totalMass += pm;
        comX += pm * p.pos.x;
        comY += pm * p.pos.y;
    }

    // ─── Pass 2: Angular momentum about COM ───
    let orbitalAngMom = 0;
    let spinAngMom = 0;
    if (totalMass > 0) {
        comX /= totalMass;
        comY /= totalMass;

        for (let i = 0; i < n; i++) {
            const p = particles[i];
            const dx = p.pos.x - comX;
            const dy = p.pos.y - comY;
            orbitalAngMom += dx * (p.mass * p.w.y) - dy * (p.mass * p.w.x);
            spinAngMom += INERTIA_K * p.mass * p.radiusSq * p.angw;
        }
    }

    // ─── Pass 3: Darwin field energy & momentum (O(v²/c²) correction) ───
    let fieldEnergy = 0;
    let fieldPx = 0, fieldPy = 0;
    const magneticOn = physics.magneticEnabled;
    const gmOn = physics.gravitomagEnabled;
    const emFieldEnergyOn = magneticOn && !physics.onePNEnabled;
    const gmFieldEnergyOn = gmOn && !physics.onePNEnabled;

    const periodic = physics.periodic;
    const domW = physics.domainW;
    const domH = physics.domainH;
    const halfDomW = domW * 0.5;
    const halfDomH = domH * 0.5;
    const topology = physics._topologyConst;
    const softeningSq = physics.blackHoleEnabled ? BH_SOFTENING_SQ : SOFTENING_SQ;
    const axMod = physics.axionEnabled
        ? 1 + AXION_G * Math.cos(physics.axionMass * physics.simTime)
        : 1.0;

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
                const rSq = dx * dx + dy * dy + softeningSq;
                const invR = 1 / Math.sqrt(rSq);
                const rx = dx * invR, ry = dy * invR;
                const viDotVj = pi.vel.x * pj.vel.x + pi.vel.y * pj.vel.y;
                const viDotR = pi.vel.x * rx + pi.vel.y * ry;
                const vjDotR = pj.vel.x * rx + pj.vel.y * ry;
                const velTerm = viDotVj + viDotR * vjDotR;

                const svx = pi.vel.x + pj.vel.x, svy = pi.vel.y + pj.vel.y;
                const svDotR = svx * rx + svy * ry;

                if (magneticOn) {
                    const qqInvR = pi.charge * pj.charge * invR * axMod;
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

                // Bazanski cross-term field energy (suppressed when 1PN is on)
                if (magneticOn && gmOn && !physics.onePNEnabled) {
                    const invRSq = invR * invR;
                    const crossCoeff = pi.charge * pj.charge * (pi.mass + pj.mass)
                        - (pi.charge * pi.charge * pj.mass + pj.charge * pj.charge * pi.mass);
                    fieldEnergy += 0.5 * crossCoeff * invRSq * axMod;
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
