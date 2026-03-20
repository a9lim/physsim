// ─── Force Computation ───
// Pairwise and Barnes-Hut force accumulation. Separates E-like (position-dependent)
// from B-like (velocity-dependent) forces for the Boris integrator.

import { BH_THETA_SQ, INERTIA_K, MAG_MOMENT_K, TIDAL_STRENGTH, YUKAWA_COUPLING, HIGGS_MASS_FLOOR, EPSILON, TORUS, BOSON_SOFTENING_SQ, BOSON_MIN_AGE } from './config.js';
import { getDelayedState } from './signal-delay.js';
import { minImage } from './topology.js';

// Reused by minImage() to avoid per-call allocation
const _miOut = { x: 0, y: 0 };

// Module-level PE accumulator: populated during force computation, read by integrator.
// Avoids a separate O(N log N) / O(N²) tree walk for PE.
let _peAccum = 0;
let _accumulatePE = true;

// P10: Precomputed per-frame flags (set in computeAllForces)
let _needAxMod = false;
// P2: Yukawa cutoff squared — pairs beyond this skip Math.exp
let _yukawaCutoffSq = Infinity;

/** Reset PE accumulator (call before computeAllForces). */
export function resetPEAccum() { _peAccum = 0; }

/** Get accumulated PE (halved for double-counting since each pair is visited from both sides). */
export function getPEAccum() { return _peAccum * 0.5; }

/**
 * Zero per-particle force accumulators and field values before a new substep.
 */
export function resetForces(particles) {
    for (let i = 0, n = particles.length; i < n; i++) {
        const p = particles[i];
        p.force.x = p.force.y = 0;
        p.jerk.x = p.jerk.y = 0;
        p.forceGravity.x = p.forceGravity.y = 0;
        p.forceCoulomb.x = p.forceCoulomb.y = 0;
        p.forceMagnetic.x = p.forceMagnetic.y = 0;
        p.forceGravitomag.x = p.forceGravitomag.y = 0;
        p.force1PN.x = p.force1PN.y = 0;
        p.forceSpinCurv.x = p.forceSpinCurv.y = 0;
        p.forceRadiation.x = p.forceRadiation.y = 0;
        p.forceYukawa.x = p.forceYukawa.y = 0;
        p.forceExternal.x = p.forceExternal.y = 0;
        p.forceHiggs.x = p.forceHiggs.y = 0;
        p.forceAxion.x = p.forceAxion.y = 0;
        p.Bz = 0; p.dBzdx = 0; p.dBzdy = 0;
        p.Bgz = 0; p.dBgzdx = 0; p.dBgzdy = 0;
        p.torqueSpinOrbit = 0;
        p._frameDragTorque = 0; p.torqueFrameDrag = 0;
        p._tidalTorque = 0; p.torqueTidal = 0;
        p.torqueContact = 0; p._contactTorque = 0;
    }
}

/**
 * Top-level force dispatch: Barnes-Hut tree walk or exact pairwise O(N^2).
 * In pairwise mode with signal delay, source positions are evaluated on the
 * past light cone rather than at the current time.
 */
export function computeAllForces(particles, toggles, pool, root, barnesHutEnabled, relativityEnabled, simTime, periodic, domW, domH, topology = TORUS, deadParticles = null) {
    const halfDomW = domW * 0.5;
    const halfDomH = domH * 0.5;
    const n = particles.length;

    // Reset PE accumulator for this force computation pass
    _peAccum = 0;
    _accumulatePE = true;

    // Cache dipole moments once per particle (valid for all pairForce/pairPE calls this substep)
    for (let i = 0; i < n; i++) {
        const p = particles[i];
        const rSq = p.bodyRadiusSq;
        p.magMoment = MAG_MOMENT_K * p.charge * p.angVel * rSq;
        p.angMomentum = INERTIA_K * p.mass * p.angVel * rSq;
    }

    // P10: Precompute axion modulation flag (constant per frame)
    _needAxMod = (toggles.coulombEnabled || toggles.magneticEnabled) && toggles.axionEnabled;
    // P2: Yukawa cutoff distance: exp(-mu*r) < 0.002 when mu*r > 6
    // When Higgs enabled, muEff can be as small as yukawaMu * HIGGS_MASS_FLOOR — widen cutoff
    const muMin = toggles.higgsEnabled ? toggles.yukawaMu * HIGGS_MASS_FLOOR : toggles.yukawaMu;
    _yukawaCutoffSq = toggles.yukawaEnabled ? (6 / muMin) ** 2 : Infinity;

    const useSignalDelay = relativityEnabled;

    if (barnesHutEnabled) {
        // Pre-size stack once before per-particle tree walks
        if (root >= 0 && _bhStack.length < pool.maxNodes) _bhStack = new Int32Array(pool.maxNodes * 2);
        if (root >= 0) for (let i = 0; i < n; i++) {
            calculateForce(particles[i], pool, root, particles[i].force, toggles, periodic, domW, domH, halfDomW, halfDomH, topology, useSignalDelay, simTime);
        }
    } else {
        for (let i = 0; i < n; i++) {
            const p = particles[i];
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                const o = particles[j];

                let sx, sy, svx, svy, sAngVel, sMagMoment, sAngMomentum;
                if (useSignalDelay) {
                    if (o.histCount < 2) continue; // no history — outside light cone
                    const ret = getDelayedState(o, p, simTime, periodic, domW, domH, halfDomW, halfDomH, topology);
                    if (!ret) continue; // signal-delayed time predates particle
                    sx = ret.x; sy = ret.y; svx = ret.vx; svy = ret.vy;
                    // Retarded angular velocity from history
                    const retAngwSq = ret.angw * ret.angw;
                    const retRadiusSq = o.bodyRadiusSq;
                    sAngVel = ret.angw / Math.sqrt(1 + retAngwSq * retRadiusSq);
                    sMagMoment = MAG_MOMENT_K * o.charge * sAngVel * retRadiusSq;
                    sAngMomentum = INERTIA_K * o.mass * sAngVel * retRadiusSq;
                } else {
                    sx = o.pos.x; sy = o.pos.y; svx = o.vel.x; svy = o.vel.y;
                    sAngVel = o.angVel;
                    sMagMoment = o.magMoment;
                    sAngMomentum = o.angMomentum;
                }

                pairForce(p, sx, sy, svx, svy,
                    o.mass, o.charge, sAngVel,
                    sMagMoment, sAngMomentum, p.force, toggles,
                    periodic, domW, domH, halfDomW, halfDomH, topology,
                    o.axMod, o.yukMod, useSignalDelay, o.higgsMod);
            }
        }
    }

    // Forces from dead particles (signal delay fade-out)
    // Dead particles don't contribute to PE (consistent with computePE behavior)
    _accumulatePE = false;
    const deadN = deadParticles ? deadParticles.length : 0;
    if (deadN > 0 && useSignalDelay) {
        for (let i = 0; i < n; i++) {
            const p = particles[i];
            for (let j = 0; j < deadN; j++) {
                const o = deadParticles[j];
                if (o.histCount < 2) continue;
                const ret = getDelayedState(o, p, simTime, periodic, domW, domH, halfDomW, halfDomH, topology);
                if (!ret) continue;
                // Retarded angular velocity from history (use _deathMass for radius)
                const retAngwSq = ret.angw * ret.angw;
                const retRadiusSq = o.bodyRadiusSq;
                const retAngVel = ret.angw / Math.sqrt(1 + retAngwSq * retRadiusSq);
                const retMagMoment = MAG_MOMENT_K * o.charge * retAngVel * retRadiusSq;
                const retAngMomentum = INERTIA_K * o._deathMass * retAngVel * retRadiusSq;
                pairForce(p, ret.x, ret.y, ret.vx, ret.vy,
                    o._deathMass, o.charge, retAngVel,
                    retMagMoment, retAngMomentum, p.force, toggles,
                    periodic, domW, domH, halfDomW, halfDomH, topology,
                    o.axMod || 1, o.yukMod || 1, true, o.higgsMod || 1);
            }
        }
    }
}

/**
 * Pairwise force between test particle p and a source at (sx, sy).
 *
 * E-like (position-dependent) forces go into `out` + per-type display vectors.
 * B-like (velocity-dependent) forces are NOT applied here; instead Bz/Bgz
 * and their gradients are accumulated for the Boris rotation step.
 */
export function pairForce(p, sx, sy, svx, svy, sMass, sCharge, sAngVel, sMagMoment, sAngMomentum, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology = TORUS, sAxMod = 1, sYukMod = 1, signalDelayed = false, sHiggsMod = 1) {
    let rx, ry;
    if (periodic) {
        minImage(p.pos.x, p.pos.y, sx, sy, topology, domW, domH, halfDomW, halfDomH, _miOut);
        rx = _miOut.x; ry = _miOut.y;
    } else {
        rx = sx - p.pos.x; ry = sy - p.pos.y;
    }
    const rawRSq = rx * rx + ry * ry;
    const rSq = rawRSq + toggles.softeningSq;
    const invRSq = 1 / rSq;
    const invR = Math.sqrt(invRSq);       // 1/r via sqrt(1/r²) — one sqrt instead of sqrt + division
    const invR3 = invR * invRSq;          // 1 / r_eff³
    const invR5 = invR3 * invRSq;          // 1 / r_eff⁵ (needed by gravity jerk + dipole forces)

    // Liénard-Wiechert aberration factor (1 - n̂·v_source)^{-3}
    // n̂ = unit vector from source to observer = (-rx, -ry) / r
    // nDotV = (-rx*svx - ry*svy) * invR
    // P4: Pre-multiply aberration into invR3/invR5 to avoid per-force-type * aberr
    let aberr = 1;
    if (signalDelayed) {
        const nDotV = -(rx * svx + ry * svy) * invR;
        const denom = Math.max(1 - nDotV, 0.01);
        aberr = Math.min(1 / (denom * denom * denom), 100);
    }
    const invR3a = signalDelayed ? invR3 * aberr : invR3;
    const invR5a = signalDelayed ? invR5 * aberr : invR5;

    // (v_s × r)_z — enters Biot-Savart-like field expressions
    const crossSV = svx * ry - svy * rx;

    // Test particle dipole moments (cached per-substep in computeAllForces)
    const pMagMoment = p.magMoment;
    const pAngMomentum = p.angMomentum;

    // P3: Relative velocity only needed for radiation jerk
    let vrx, vry, rDotVr;
    if (toggles.radiationEnabled) {
        vrx = svx - p.vel.x; vry = svy - p.vel.y;
        rDotVr = rx * vrx + ry * vry;
    }

    if (toggles.gravityEnabled) {
        const k = p.mass * sMass;
        const fDir = k * invR3a;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceGravity.x += rx * fDir;
        p.forceGravity.y += ry * fDir;
        // P3: Analytical jerk only when radiation enabled
        if (toggles.radiationEnabled) {
            const jRadial = -3 * k * rDotVr * invR5a;
            p.jerk.x += vrx * fDir + rx * jRadial;
            p.jerk.y += vry * fDir + ry * jRadial;
        }
        // PE: -m₁m₂/r
        if (_accumulatePE) _peAccum -= k * invR;
    }

    // P10: Use precomputed axion modulation flag (set in computeAllForces)
    const axModPair = _needAxMod ? Math.sqrt(p.axMod * sAxMod) : 1;

    if (toggles.coulombEnabled) {
        const k = -(p.charge * sCharge) * axModPair;
        const fDir = k * invR3a;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceCoulomb.x += rx * fDir;
        p.forceCoulomb.y += ry * fDir;
        // P3: Analytical jerk only when radiation enabled
        if (toggles.radiationEnabled) {
            const jRadial = -3 * k * rDotVr * invR5a;
            p.jerk.x += vrx * fDir + rx * jRadial;
            p.jerk.y += vry * fDir + ry * jRadial;
        }
        // PE: +q₁q₂/r * axMod
        if (_accumulatePE) _peAccum += p.charge * sCharge * invR * axModPair;
    }

    if (toggles.onePNEnabled && (toggles.gravitomagEnabled || toggles.magneticEnabled)) {
        const r = 1 / invR;  // recover r from invR (cheaper than sqrt)
        const nx = rx * invR, ny = ry * invR;
        const pvx = p.vel.x, pvy = p.vel.y;
        const v1DotV2 = pvx * svx + pvy * svy;

        if (toggles.gravitomagEnabled) {
            const v1Sq = pvx * pvx + pvy * pvy;
            const v2Sq = svx * svx + svy * svy;
            const nDotV1 = nx * pvx + ny * pvy;
            const nDotV2 = nx * svx + ny * svy;
            const radial = -v1Sq - 2 * v2Sq
                + 1.5 * nDotV2 * nDotV2
                + 5 * p.mass * invR + 4 * sMass * invR;
            const v1Coeff = 4 * nDotV1 - 3 * nDotV2;
            const v2Coeff = 3 * nDotV2;
            const base = p.mass * sMass * invR3;
            const fx = base * (rx * radial + (pvx * v1Coeff + svx * v2Coeff) * r);
            const fy = base * (ry * radial + (pvy * v1Coeff + svy * v2Coeff) * r);
            out.x += fx;
            out.y += fy;
            p.force1PN.x += fx;
            p.force1PN.y += fy;
            // Analytical jerk for position-only EIH term: F = m₂(5m₁+4m₂)·r/r⁴
            // Velocity-dependent EIH terms (v², v·n) omitted — O(1/c⁵) contribution
            if (toggles.radiationEnabled) {
                const kEIH = p.mass * sMass * (5 * p.mass + 4 * sMass);
                const fDirEIH = kEIH * invRSq * invRSq;
                const jRadialEIH = -4 * kEIH * rDotVr * invRSq * invRSq * invRSq;
                p.jerk.x += vrx * fDirEIH + rx * jRadialEIH;
                p.jerk.y += vry * fDirEIH + ry * jRadialEIH;
            }
            // 1PN EIH PE: -m₁m₂/r · [1.5(v₁²+v₂²) - 3.5v₁·v₂ - 0.5(v₁·n̂)(v₂·n̂) + m₁/r + m₂/r]
            if (_accumulatePE) {
                _peAccum -= p.mass * sMass * invR * (
                    1.5 * (v1Sq + v2Sq) - 3.5 * v1DotV2 - 0.5 * nDotV1 * nDotV2
                    + p.mass * invR + sMass * invR
                );
            }
        }

        if (toggles.magneticEnabled) {
            const v2DotN = svx * nx + svy * ny;
            const v1DotN = pvx * nx + pvy * ny;
            const coeff = 0.5 * p.charge * sCharge * invRSq;
            const symX = coeff * (pvx * v2DotN - 3 * nx * v1DotN * v2DotN);
            const symY = coeff * (pvy * v2DotN - 3 * ny * v1DotN * v2DotN);
            out.x += symX;
            out.y += symY;
            p.force1PN.x += symX;
            p.force1PN.y += symY;
            // Darwin PE: -0.5q₁q₂/r · [v₁·v₂ + (v₁·n̂)(v₂·n̂)]
            if (_accumulatePE) {
                _peAccum -= 0.5 * p.charge * sCharge * invR * (v1DotV2 + v1DotN * v2DotN);
            }
        }
    }

    if (toggles.onePNEnabled && toggles.gravitomagEnabled && toggles.magneticEnabled) {
        const crossCoeff = p.charge * sCharge * (p.mass + sMass)
            - (p.charge * p.charge * sMass + sCharge * sCharge * p.mass);
        const fDir = crossCoeff * invRSq * invRSq;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.force1PN.x += rx * fDir;
        p.force1PN.y += ry * fDir;
        // Analytical jerk: F = crossCoeff·r/r⁴, d/dt(1/r⁴) = -4·rDotVr/r⁶
        if (toggles.radiationEnabled) {
            const jRadial = -4 * crossCoeff * rDotVr * invRSq * invRSq * invRSq;
            p.jerk.x += vrx * fDir + rx * jRadial;
            p.jerk.y += vry * fDir + ry * jRadial;
        }
        // Bazanski PE: +0.5·crossCoeff/r²
        if (_accumulatePE) _peAccum += 0.5 * crossCoeff * invRSq;
    }

    if (toggles.magneticEnabled) {
        // Axion modulation: geometric mean of observer and source α_eff
        const axMod = axModPair;
        // Dipole-dipole radial: F = -3μ₁μ₂/r⁴ (aligned ⊥-to-plane dipoles repel)
        const fDir = -3 * (pMagMoment * sMagMoment) * invR5a * axMod;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceMagnetic.x += rx * fDir;
        p.forceMagnetic.y += ry * fDir;
        // Analytical jerk: F = -3μ₁μ₂·r/r⁵, d/dt(1/r⁵) = -5·rDotVr/r⁷
        if (toggles.radiationEnabled) {
            const invR7a = invR5a * invRSq;
            const jRadial = 5 * 3 * (pMagMoment * sMagMoment) * rDotVr * invR7a * axMod;
            p.jerk.x += vrx * fDir + rx * jRadial;
            p.jerk.y += vry * fDir + ry * jRadial;
        }
        // PE: +μ₁μ₂/r³ * axMod
        if (_accumulatePE) _peAccum += (pMagMoment * sMagMoment) * invR3 * axMod;

        // Bz from moving charge (Biot-Savart): B_z = q_s(v_s × r̂)_z / r²
        const BzMoving = sCharge * crossSV * invR3 * axMod;
        p.Bz += BzMoving;

        // ∇Bz for spin-orbit coupling (radial + angular terms)
        p.dBzdx += 3 * BzMoving * rx * invRSq + sCharge * svy * invR3 * axMod;
        p.dBzdy += 3 * BzMoving * ry * invRSq - sCharge * svx * invR3 * axMod;

        // Dipole-sourced Bz: equatorial field of z-aligned dipole, -μ/r³
        p.Bz -= sMagMoment * invR3 * axMod;
        p.dBzdx -= 3 * sMagMoment * rx * invR5 * axMod;
        p.dBzdy -= 3 * sMagMoment * ry * invR5 * axMod;
    }

    if (toggles.gravitomagEnabled) {
        // GM dipole: F = +3L₁L₂/r⁴ (GEM sign flip: co-rotating masses attract)
        const fDir = 3 * (pAngMomentum * sAngMomentum) * invR5a;
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceGravitomag.x += rx * fDir;
        p.forceGravitomag.y += ry * fDir;
        // Analytical jerk: F = 3L₁L₂·r/r⁵, d/dt(1/r⁵) = -5·rDotVr/r⁷
        if (toggles.radiationEnabled) {
            const invR7a = invR5a * invRSq;
            const jRadial = -5 * 3 * (pAngMomentum * sAngMomentum) * rDotVr * invR7a;
            p.jerk.x += vrx * fDir + rx * jRadial;
            p.jerk.y += vry * fDir + ry * jRadial;
        }
        // PE: -L₁L₂/r³
        if (_accumulatePE) _peAccum -= (pAngMomentum * sAngMomentum) * invR3;

        // Bgz from moving mass: −m_s(v_s × r̂)_z / r² (negative: GEM convention)
        const BgzMoving = -sMass * crossSV * invR3;
        p.Bgz += BgzMoving;

        // ∇Bgz for spin-orbit coupling
        p.dBgzdx += 3 * BgzMoving * rx * invRSq - sMass * svy * invR3;
        p.dBgzdy += 3 * BgzMoving * ry * invRSq + sMass * svx * invR3;

        // Spin-sourced Bgz: −2L/r³ (GEM analog of magnetic dipole field)
        p.Bgz -= 2 * sAngMomentum * invR3;
        p.dBgzdx -= 6 * sAngMomentum * rx * invR5;
        p.dBgzdy -= 6 * sAngMomentum * ry * invR5;

        // Frame-dragging torque: aligns spins toward co-rotation
        const torque = 2 * sAngMomentum * (sAngVel - p.angVel) * invR3;
        p._frameDragTorque += torque;
    }

    // P2: Yukawa cutoff — skip Math.exp for distant pairs where force is negligible
    if (toggles.yukawaEnabled && rawRSq < _yukawaCutoffSq) {
        const mu = toggles.higgsEnabled ? toggles.yukawaMu * Math.sqrt(p.higgsMod * sHiggsMod) : toggles.yukawaMu;
        const r = 1 / invR;
        const expMuR = Math.exp(-mu * r);
        const yukModPair = _needAxMod ? Math.sqrt(p.yukMod * sYukMod) : 1;
        const ym = yukModPair;
        const fDir = YUKAWA_COUPLING * ym * p.mass * sMass * expMuR * (invRSq + mu * invR) * (signalDelayed ? invR * aberr : invR);
        out.x += rx * fDir;
        out.y += ry * fDir;
        p.forceYukawa.x += rx * fDir;
        p.forceYukawa.y += ry * fDir;
        // P3: Analytical jerk only when radiation enabled
        if (toggles.radiationEnabled) {
            const jRadial = -(3 * invRSq + 3 * mu * invR + mu * mu) * rDotVr * YUKAWA_COUPLING * ym * p.mass * sMass * expMuR * invRSq * (signalDelayed ? invR * aberr : invR);
            p.jerk.x += vrx * fDir + rx * jRadial;
            p.jerk.y += vry * fDir + ry * jRadial;
        }
        // PE: -g²m₁m₂·exp(-μr)/r
        if (_accumulatePE) _peAccum -= YUKAWA_COUPLING * ym * p.mass * sMass * expMuR * invR;

        // Scalar Breit O(v²/c²) correction
        if (toggles.onePNEnabled) {
            const pvx = p.vel.x, pvy = p.vel.y;
            const nx = rx * invR, ny = ry * invR;
            const nDotV1 = nx * pvx + ny * pvy;
            const nDotV2 = nx * svx + ny * svy;
            const v1DotV2 = pvx * svx + pvy * svy;
            const alpha = 1 + mu * r;
            const beta = 0.5 * YUKAWA_COUPLING * ym * p.mass * sMass * expMuR * invRSq;
            const radial = -(alpha * v1DotV2 + (alpha * alpha + alpha + 1) * nDotV1 * nDotV2);
            const fx = beta * (radial * nx + alpha * (nDotV2 * pvx + nDotV1 * svx));
            const fy = beta * (radial * ny + alpha * (nDotV2 * pvy + nDotV1 * svy));
            out.x += fx;
            out.y += fy;
            p.force1PN.x += fx;
            p.force1PN.y += fy;
            // Scalar Breit PE: +g²m₁m₂e^{-μr}/(2r)·[v₁·v₂ + (n̂·v₁)(n̂·v₂)(1+μr)]
            if (_accumulatePE) _peAccum += 0.5 * YUKAWA_COUPLING * ym * p.mass * sMass * expMuR * invR * (v1DotV2 + nDotV1 * nDotV2 * alpha);
        }
    }

    if (toggles.gravityEnabled) {
        // Tidal locking torque: drives spin toward synchronous rotation.
        // Coupling = (m_other + q₁q₂/m)² accounts for all cross-terms:
        // gravity-raises/gravity-torques, gravity/Coulomb, Coulomb/gravity, Coulomb/Coulomb.
        const crossRV = rx * (svy - p.vel.y) - ry * (svx - p.vel.x);
        const wOrbit = crossRV * invRSq;
        const dw = p.angVel - wOrbit;
        let coupling = sMass;
        if (toggles.coulombEnabled && p.mass > EPSILON) coupling += p.charge * sCharge / p.mass;
        const bodyR = Math.sqrt(p.bodyRadiusSq);
        const ri5 = p.bodyRadiusSq * p.bodyRadiusSq * bodyR;
        const invR6 = invRSq * invRSq * invRSq;
        p._tidalTorque -= TIDAL_STRENGTH * coupling * coupling * ri5 * invR6 * dw;
    }
}

/** Accumulate 1PN force from one resolved source onto particle p. */
function _accum1PN(p, px, py, pvx, pvy, pMass, pCharge,
                   sx, sy, svx, svy, sMass, sCharge, sYukMod,
                   softeningSq, periodic, domW, domH, halfDomW, halfDomH, topology,
                   gravitomagEnabled, magneticEnabled, yukawaEnabled, yukawaMu, higgsEnabled = false, sHiggsMod = 1) {
    let rx, ry;
    if (periodic) {
        minImage(px, py, sx, sy, topology, domW, domH, halfDomW, halfDomH, _miOut);
        rx = _miOut.x; ry = _miOut.y;
    } else {
        rx = sx - px; ry = sy - py;
    }
    const rSq = rx * rx + ry * ry + softeningSq;
    const invRSq = 1 / rSq;
    const invR = Math.sqrt(invRSq);
    const r = 1 / invR;
    const nx = rx * invR, ny = ry * invR;

    if (gravitomagEnabled) {
        const v1Sq = pvx * pvx + pvy * pvy;
        const v2Sq = svx * svx + svy * svy;
        const nDotV1 = nx * pvx + ny * pvy;
        const nDotV2 = nx * svx + ny * svy;
        const radial = -v1Sq - 2 * v2Sq
            + 1.5 * nDotV2 * nDotV2
            + 5 * pMass * invR + 4 * sMass * invR;
        const v1Coeff = 4 * nDotV1 - 3 * nDotV2;
        const v2Coeff = 3 * nDotV2;
        const base = pMass * sMass * invRSq * invR;
        p.force1PN.x += base * (rx * radial + (pvx * v1Coeff + svx * v2Coeff) * r);
        p.force1PN.y += base * (ry * radial + (pvy * v1Coeff + svy * v2Coeff) * r);
    }

    if (magneticEnabled) {
        const v2DotN = svx * nx + svy * ny;
        const v1DotN = pvx * nx + pvy * ny;
        const coeff = 0.5 * pCharge * sCharge * invRSq;
        p.force1PN.x += coeff * (pvx * v2DotN - 3 * nx * v1DotN * v2DotN);
        p.force1PN.y += coeff * (pvy * v2DotN - 3 * ny * v1DotN * v2DotN);
    }

    if (gravitomagEnabled && magneticEnabled) {
        const crossCoeff = pCharge * sCharge * (pMass + sMass)
            - (pCharge * pCharge * sMass + sCharge * sCharge * pMass);
        const fDir = crossCoeff * invRSq * invRSq;
        p.force1PN.x += rx * fDir;
        p.force1PN.y += ry * fDir;
    }

    if (yukawaEnabled) {
        const mu = higgsEnabled ? yukawaMu * Math.sqrt(p.higgsMod * sHiggsMod) : yukawaMu;
        const expMuR = Math.exp(-mu * r);
        const nDotV1 = nx * pvx + ny * pvy;
        const nDotV2 = nx * svx + ny * svy;
        const v1DotV2 = pvx * svx + pvy * svy;
        const alpha = 1 + mu * r;
        const beta = 0.5 * YUKAWA_COUPLING * Math.sqrt(p.yukMod * sYukMod) * pMass * sMass * expMuR * invRSq;
        const radial = -(alpha * v1DotV2 + (alpha * alpha + alpha + 1) * nDotV1 * nDotV2);
        p.force1PN.x += beta * (radial * nx + alpha * (nDotV2 * pvx + nDotV1 * svx));
        p.force1PN.y += beta * (radial * ny + alpha * (nDotV2 * pvy + nDotV1 * svy));
    }
}

// Pre-allocated stack for 1PN tree walk
let _1pnStack = new Int32Array(256);

/** BH tree walk for 1PN velocity-Verlet correction. Signal delay at leaf level. */
function _compute1PNTreeWalk(particle, pool, rootIdx, softeningSq, periodic, domW, domH, halfDomW, halfDomH, topology, gravitomagEnabled, magneticEnabled, yukawaEnabled, yukawaMu, simTime, higgsEnabled = false) {
    const thetaSq = BH_THETA_SQ;
    const px = particle.pos.x, py = particle.pos.y;
    const pvx = particle.vel.x, pvy = particle.vel.y;
    const pMass = particle.mass, pCharge = particle.charge;
    let stackTop = 0;
    _1pnStack[stackTop++] = rootIdx;

    while (stackTop > 0) {
        const nodeIdx = _1pnStack[--stackTop];
        if (pool.totalMass[nodeIdx] < EPSILON) continue;

        let dx, dy;
        if (periodic) {
            minImage(px, py, pool.comX[nodeIdx], pool.comY[nodeIdx], topology, domW, domH, halfDomW, halfDomH, _miOut);
            dx = _miOut.x; dy = _miOut.y;
        } else {
            dx = pool.comX[nodeIdx] - px;
            dy = pool.comY[nodeIdx] - py;
        }
        const dSq = dx * dx + dy * dy;
        const size = pool.bw[nodeIdx] * 2;

        if (!pool.divided[nodeIdx] && pool.pointCount[nodeIdx] > 0) {
            const base = nodeIdx * pool.nodeCapacity;
            for (let i = 0; i < pool.pointCount[nodeIdx]; i++) {
                const other = pool.points[base + i];
                if (other === particle) continue;
                if (other.isGhost && other.original === particle) continue;
                const real = other.isGhost ? other.original : other;
                let sx, sy, svx, svy;
                if (!other.isGhost) {
                    if (real.histCount < 2) continue;
                    const ret = getDelayedState(real, particle, simTime, periodic, domW, domH, halfDomW, halfDomH, topology);
                    if (!ret) continue;
                    sx = ret.x; sy = ret.y; svx = ret.vx; svy = ret.vy;
                } else {
                    sx = other.pos.x; sy = other.pos.y; svx = other.vel.x; svy = other.vel.y;
                }
                _accum1PN(particle, px, py, pvx, pvy, pMass, pCharge,
                    sx, sy, svx, svy, other.mass, other.charge, real.yukMod,
                    softeningSq, periodic, domW, domH, halfDomW, halfDomH, topology,
                    gravitomagEnabled, magneticEnabled, yukawaEnabled, yukawaMu, higgsEnabled, real.higgsMod);
            }
        } else if (pool.divided[nodeIdx] && (size * size < thetaSq * dSq)) {
            const nodeMass = pool.totalMass[nodeIdx];
            const avgVx = pool.totalMomentumX[nodeIdx] / nodeMass;
            const avgVy = pool.totalMomentumY[nodeIdx] / nodeMass;
            _accum1PN(particle, px, py, pvx, pvy, pMass, pCharge,
                pool.comX[nodeIdx], pool.comY[nodeIdx], avgVx, avgVy,
                nodeMass, pool.totalCharge[nodeIdx], 1,
                softeningSq, periodic, domW, domH, halfDomW, halfDomH, topology,
                gravitomagEnabled, magneticEnabled, yukawaEnabled, yukawaMu, higgsEnabled, 1);
        } else if (pool.divided[nodeIdx]) {
            _1pnStack[stackTop++] = pool.nw[nodeIdx];
            _1pnStack[stackTop++] = pool.ne[nodeIdx];
            _1pnStack[stackTop++] = pool.sw[nodeIdx];
            _1pnStack[stackTop++] = pool.se[nodeIdx];
        }
    }
}

/**
 * Recompute 1PN forces for velocity-Verlet correction.
 * O(N log N) BH tree walk when enabled, O(N²) pairwise fallback.
 * Signal delay at leaf level (1PN requires relativity).
 */
export function compute1PN(particles, SOFTENING_SQ_VAL, periodic, domW, domH, halfDomW, halfDomH, topology = TORUS, gravitomagEnabled = true, magneticEnabled = false, yukawaEnabled = false, yukawaMu = 0.05, simTime = 0, pool = null, root = -1, barnesHutEnabled = false, higgsEnabled = false) {
    const n = particles.length;
    for (let i = 0; i < n; i++) {
        particles[i].force1PN.x = particles[i].force1PN.y = 0;
    }

    if (barnesHutEnabled && root >= 0) {
        if (_1pnStack.length < pool.maxNodes) _1pnStack = new Int32Array(pool.maxNodes * 2);
        for (let i = 0; i < n; i++) {
            _compute1PNTreeWalk(particles[i], pool, root,
                SOFTENING_SQ_VAL, periodic, domW, domH, halfDomW, halfDomH, topology,
                gravitomagEnabled, magneticEnabled, yukawaEnabled, yukawaMu, simTime, higgsEnabled);
        }
    } else {
        for (let i = 0; i < n; i++) {
            const p = particles[i];
            const px = p.pos.x, py = p.pos.y;
            const pvx = p.vel.x, pvy = p.vel.y;
            const pMass = p.mass, pCharge = p.charge;
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                const o = particles[j];
                if (o.histCount < 2) continue;
                const ret = getDelayedState(o, p, simTime, periodic, domW, domH, halfDomW, halfDomH, topology);
                if (!ret) continue;
                _accum1PN(p, px, py, pvx, pvy, pMass, pCharge,
                    ret.x, ret.y, ret.vx, ret.vy, o.mass, o.charge, o.yukMod,
                    SOFTENING_SQ_VAL, periodic, domW, domH, halfDomW, halfDomH, topology,
                    gravitomagEnabled, magneticEnabled, yukawaEnabled, yukawaMu, higgsEnabled, o.higgsMod);
            }
        }
    }
}

/**
 * Iterative Barnes-Hut tree walk. Uses aggregate multipole data for distant
 * nodes (size/d < theta), individual particles for nearby leaves.
 * Signal delay is applied at leaf level (individual particles); distant
 * nodes use current-time aggregates (retarded correction is negligible at
 * distances where the BH approximation kicks in).
 */
// Pre-allocated stack for iterative tree walk (avoids recursion overhead)
let _bhStack = new Int32Array(256);

export function calculateForce(particle, pool, rootIdx, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology, useSignalDelay, simTime) {
    const thetaSq = BH_THETA_SQ;
    const px = particle.pos.x, py = particle.pos.y;
    let stackTop = 0;
    _bhStack[stackTop++] = rootIdx;

    while (stackTop > 0) {
        const nodeIdx = _bhStack[--stackTop];
        if (pool.totalMass[nodeIdx] < EPSILON) continue;

        let dx, dy;
        if (periodic) {
            minImage(px, py, pool.comX[nodeIdx], pool.comY[nodeIdx], topology, domW, domH, halfDomW, halfDomH, _miOut);
            dx = _miOut.x; dy = _miOut.y;
        } else {
            dx = pool.comX[nodeIdx] - px;
            dy = pool.comY[nodeIdx] - py;
        }
        const dSq = dx * dx + dy * dy;
        const size = pool.bw[nodeIdx] * 2;

        if (!pool.divided[nodeIdx] && pool.pointCount[nodeIdx] > 0) {
            // Leaf node: iterate individual particles
            const base = nodeIdx * pool.nodeCapacity;
            for (let i = 0; i < pool.pointCount[nodeIdx]; i++) {
                const other = pool.points[base + i];
                if (other === particle) continue;
                if (other.isGhost && other.original === particle) continue;
                const real = other.isGhost ? other.original : other;
                let sx, sy, svx, svy, sAngVel, sMagMom, sAngMom, delayed;
                if (useSignalDelay) {
                    if (real.histCount < 2) continue; // no history — outside light cone
                    const ret = getDelayedState(real, particle, simTime, periodic, domW, domH, halfDomW, halfDomH, topology);
                    if (!ret) continue; // signal delay predates particle
                    if (other.isGhost) {
                        // Signal-delayed original position + periodic shift
                        const shiftX = other.pos.x - real.pos.x;
                        const shiftY = other.pos.y - real.pos.y;
                        sx = ret.x + shiftX;
                        sy = ret.y + shiftY;
                    } else {
                        sx = ret.x;
                        sy = ret.y;
                    }
                    svx = ret.vx; svy = ret.vy;
                    const retAngwSq = ret.angw * ret.angw;
                    const retRadiusSq = real.bodyRadiusSq;
                    sAngVel = ret.angw / Math.sqrt(1 + retAngwSq * retRadiusSq);
                    sMagMom = MAG_MOMENT_K * other.charge * sAngVel * retRadiusSq;
                    sAngMom = INERTIA_K * other.mass * sAngVel * retRadiusSq;
                    delayed = true;
                } else {
                    sx = other.pos.x; sy = other.pos.y; svx = other.vel.x; svy = other.vel.y;
                    sAngVel = other.angVel; sMagMom = other.magMoment; sAngMom = other.angMomentum;
                    delayed = false;
                }
                pairForce(particle, sx, sy, svx, svy, other.mass, other.charge, sAngVel, sMagMom, sAngMom, out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology, real.axMod, real.yukMod, delayed, real.higgsMod);
            }
        } else if (pool.divided[nodeIdx] && (size * size < thetaSq * dSq)) {
            // Distant node: use aggregate (size/d < theta, computed as size²<theta²·d²)
            const nodeMass = pool.totalMass[nodeIdx];
            const avgVx = pool.totalMomentumX[nodeIdx] / nodeMass;
            const avgVy = pool.totalMomentumY[nodeIdx] / nodeMass;
            pairForce(particle, pool.comX[nodeIdx], pool.comY[nodeIdx], avgVx, avgVy, nodeMass, pool.totalCharge[nodeIdx], 0, pool.totalMagneticMoment[nodeIdx], pool.totalAngularMomentum[nodeIdx], out, toggles, periodic, domW, domH, halfDomW, halfDomH, topology, 1, 1, false, 1);
        } else if (pool.divided[nodeIdx]) {
            // Push children onto stack
            _bhStack[stackTop++] = pool.nw[nodeIdx];
            _bhStack[stackTop++] = pool.ne[nodeIdx];
            _bhStack[stackTop++] = pool.sw[nodeIdx];
            _bhStack[stackTop++] = pool.se[nodeIdx];
        }
    }
}

// Separate stack for boson BH tree walks (avoids conflict with particle _bhStack)
let _bosonBHStack = new Int32Array(256);

// Module-level output for _walkBosonTree — avoids per-call allocation
const _bwOut = { x: 0, y: 0 };

/**
 * Core BH tree walk for boson interactions. Accumulates impulse into _bwOut.
 * @param {number}   px, py        - query position
 * @param {number}   scale         - pre-multiplied factor (e.g. grFactor*dt)
 * @param {number}   softeningSq   - gravitational softening squared
 * @param {Object}   pool          - boson QuadTreePool
 * @param {number}   root          - tree root index
 * @param {function} nodeVal       - (pool, nodeIdx) -> aggregate scalar (mass or charge)
 * @param {function} pointVal      - (b) -> per-point scalar (._srcMass or ._srcCharge)
 * @param {function} skipNode      - (pool, nodeIdx) -> bool: skip this node entirely
 * @param {function} skipPoint     - (b) -> bool: skip this leaf point
 */
function _walkBosonTreeCore(px, py, scale, softeningSq, pool, root, periodic, topology, domW, domH, halfDomW, halfDomH, nodeVal, pointVal, skipNode, skipPoint) {
    let kx = 0, ky = 0;
    let stackTop = 0;
    _bosonBHStack[stackTop++] = root;

    while (stackTop > 0) {
        const nodeIdx = _bosonBHStack[--stackTop];
        if (skipNode(pool, nodeIdx)) continue;

        let dx, dy;
        if (periodic) {
            minImage(px, py, pool.comX[nodeIdx], pool.comY[nodeIdx], topology, domW, domH, halfDomW, halfDomH, _miOut);
            dx = _miOut.x; dy = _miOut.y;
        } else {
            dx = pool.comX[nodeIdx] - px;
            dy = pool.comY[nodeIdx] - py;
        }
        const dSq = dx * dx + dy * dy;
        const size = pool.bw[nodeIdx] * 2;
        const cnt = pool.pointCount[nodeIdx];

        if (!pool.divided[nodeIdx] && cnt > 0) {
            const base = nodeIdx * pool.nodeCapacity;
            for (let k = 0; k < cnt; k++) {
                const b = pool.points[base + k];
                if (skipPoint(b)) continue;
                let bdx, bdy;
                if (periodic) {
                    minImage(px, py, b.pos.x, b.pos.y, topology, domW, domH, halfDomW, halfDomH, _miOut);
                    bdx = _miOut.x; bdy = _miOut.y;
                } else {
                    bdx = b.pos.x - px;
                    bdy = b.pos.y - py;
                }
                const rSq = bdx * bdx + bdy * bdy + softeningSq;
                const invRSq = 1 / rSq;
                const f = scale * pointVal(b) * Math.sqrt(invRSq) * invRSq;
                kx += bdx * f;
                ky += bdy * f;
            }
        } else if (pool.divided[nodeIdx] && size * size < BH_THETA_SQ * dSq) {
            const rSq = dSq + softeningSq;
            const invRSq = 1 / rSq;
            const f = scale * nodeVal(pool, nodeIdx) * Math.sqrt(invRSq) * invRSq;
            kx += dx * f;
            ky += dy * f;
        } else if (pool.divided[nodeIdx]) {
            _bosonBHStack[stackTop++] = pool.nw[nodeIdx];
            _bosonBHStack[stackTop++] = pool.ne[nodeIdx];
            _bosonBHStack[stackTop++] = pool.sw[nodeIdx];
            _bosonBHStack[stackTop++] = pool.se[nodeIdx];
        }
    }

    _bwOut.x = kx;
    _bwOut.y = ky;
}

// Callbacks for gravity walk (mass-based)
const _massNodeVal   = (pool, idx) => pool.totalMass[idx];
const _massPointVal  = (b) => b._srcMass;
const _massSkipNode  = (pool, idx) => pool.totalMass[idx] < EPSILON;
const _massSkipPoint = () => false;

/**
 * Shared BH tree walk for boson gravity. Accumulates gravitational impulse into _bwOut.
 * Reads _srcMass from leaf points, totalMass from aggregate nodes.
 * @param {number} px, py      - query position
 * @param {number} scale       - pre-multiplied factor (e.g. 1 for force, grFactor*dt for kick)
 * @param {number} softeningSq - gravitational softening squared
 * @param {Object} pool        - boson QuadTreePool
 * @param {number} root        - tree root index
 */
function _walkBosonTree(px, py, scale, softeningSq, pool, root, periodic, topology, domW, domH, halfDomW, halfDomH) {
    _walkBosonTreeCore(px, py, scale, softeningSq, pool, root, periodic, topology, domW, domH, halfDomW, halfDomH, _massNodeVal, _massPointVal, _massSkipNode, _massSkipPoint);
}

/**
 * Gravitational force from bosons onto particles via Barnes-Hut tree walk.
 * O(N_particles × log(N_bosons)).
 */
export function computeBosonGravity(particles, bosonPool, bosonRoot, softeningSq, periodic, topology, domW, domH) {
    const n = particles.length;
    if (n === 0 || bosonRoot < 0 || bosonPool.totalMass[bosonRoot] < EPSILON) return;
    if (_bosonBHStack.length < bosonPool.maxNodes * 2) _bosonBHStack = new Int32Array(bosonPool.maxNodes * 2);
    const halfDomW = domW * 0.5, halfDomH = domH * 0.5;

    for (let i = 0; i < n; i++) {
        const p = particles[i];
        // Pass p.mass as scale — avoids multiply-by-1 per node + 2 post-multiplies
        _walkBosonTree(p.pos.x, p.pos.y, p.mass, softeningSq, bosonPool, bosonRoot, periodic, topology, domW, domH, halfDomW, halfDomH);
        p.force.x += _bwOut.x;
        p.force.y += _bwOut.y;
        p.forceGravity.x += _bwOut.x;
        p.forceGravity.y += _bwOut.y;
    }
}

/**
 * Mutual gravitational interaction between bosons via Barnes-Hut tree walk.
 * GR receiver factors: 2 for photons (null geodesic), 1+v² for pions (massive).
 * O(N_bosons × log(N_bosons)).
 */
export function applyBosonBosonGravity(photons, pions, dt, bosonPool, bosonRoot, periodic, topology, domW, domH) {
    const nPh = photons ? photons.length : 0;
    const nPi = pions ? pions.length : 0;
    if (nPh + nPi < 2 || bosonRoot < 0 || bosonPool.totalMass[bosonRoot] < EPSILON) return;
    if (_bosonBHStack.length < bosonPool.maxNodes * 2) _bosonBHStack = new Int32Array(bosonPool.maxNodes * 2);
    const halfDomW = domW * 0.5, halfDomH = domH * 0.5;

    // Photons: receiver GR factor = 2, kick into vel, renormalize to c=1
    const twoDt = 2 * dt;
    for (let i = 0; i < nPh; i++) {
        const ph = photons[i];
        if (!ph.alive) continue;
        _walkBosonTree(ph.pos.x, ph.pos.y, twoDt, BOSON_SOFTENING_SQ, bosonPool, bosonRoot, periodic, topology, domW, domH, halfDomW, halfDomH);
        ph.vel.x += _bwOut.x;
        ph.vel.y += _bwOut.y;
        const vSq = ph.vel.x * ph.vel.x + ph.vel.y * ph.vel.y;
        if (Math.abs(vSq - 1) > 1e-6) {
            const v = Math.sqrt(vSq);
            if (v > EPSILON) { ph.vel.x /= v; ph.vel.y /= v; }
        }
    }

    // Pions: receiver GR factor = 1+v², kick into w, sync derived state
    for (let i = 0; i < nPi; i++) {
        const pn = pions[i];
        if (!pn.alive) continue;
        _walkBosonTree(pn.pos.x, pn.pos.y, (1 + pn.vSq) * dt, BOSON_SOFTENING_SQ, bosonPool, bosonRoot, periodic, topology, domW, domH, halfDomW, halfDomH);
        pn.w.x += _bwOut.x;
        pn.w.y += _bwOut.y;
        pn._syncVel();
    }
}

// Callbacks for Coulomb walk (charge-based)
const _chargeNodeVal   = (pool, idx) => pool.totalCharge[idx];
const _chargePointVal  = (b) => b._srcCharge;
const _chargeSkipNode  = (pool, idx) => pool.totalCharge[idx] === 0;
const _chargeSkipPoint = (b) => b._srcCharge === 0;

/**
 * Shared BH tree walk for boson Coulomb. Accumulates Coulomb impulse into _bwOut.
 * Reads _srcCharge from leaf points, totalCharge from aggregate nodes.
 */
function _walkBosonTreeCharge(px, py, scale, softeningSq, pool, root, periodic, topology, domW, domH, halfDomW, halfDomH) {
    _walkBosonTreeCore(px, py, scale, softeningSq, pool, root, periodic, topology, domW, domH, halfDomW, halfDomH, _chargeNodeVal, _chargePointVal, _chargeSkipNode, _chargeSkipPoint);
}

/**
 * Mutual Coulomb interaction between charged pions via Barnes-Hut tree walk.
 * F = -q_i * q_j / r² (like-charges repel). O(N_pions × log(N_bosons)).
 */
export function applyPionPionCoulomb(pions, dt, bosonPool, bosonRoot, periodic, topology, domW, domH) {
    const nPi = pions ? pions.length : 0;
    if (nPi < 2 || bosonRoot < 0) return;
    if (_bosonBHStack.length < bosonPool.maxNodes * 2) _bosonBHStack = new Int32Array(bosonPool.maxNodes * 2);
    const halfDomW = domW * 0.5, halfDomH = domH * 0.5;

    for (let i = 0; i < nPi; i++) {
        const pn = pions[i];
        if (!pn.alive || pn.charge === 0) continue;
        _walkBosonTreeCharge(pn.pos.x, pn.pos.y, -pn.charge * dt, BOSON_SOFTENING_SQ, bosonPool, bosonRoot, periodic, topology, domW, domH, halfDomW, halfDomH);
        pn.w.x += _bwOut.x;
        pn.w.y += _bwOut.y;
        pn._syncVel();
    }
}

/**
 * π⁺π⁻ annihilation: opposite-charge pions within softening distance → 2 photons.
 * Returns array of {pion1, pion2} pairs to annihilate.
 * Uses boson tree range query. O(N_charged × log(N_bosons)).
 */
export function findPionAnnihilations(pions, bosonPool, bosonRoot) {
    const nPi = pions ? pions.length : 0;
    if (nPi < 2 || bosonRoot < 0) return _annihPairs;
    _annihPairs.length = 0;

    const annihDistSq = BOSON_SOFTENING_SQ; // annihilate within softening distance
    const searchR = Math.sqrt(BOSON_SOFTENING_SQ);

    for (let i = 0; i < nPi; i++) {
        const pn = pions[i];
        if (!pn.alive || pn.charge === 0 || pn.age < BOSON_MIN_AGE) continue;

        // Range query of boson tree for nearby bosons
        const candidates = bosonPool.queryReuse(bosonRoot,
            pn.pos.x, pn.pos.y, searchR, searchR);
        for (let ci = 0; ci < candidates.length; ci++) {
            const other = candidates[ci];
            if (other === pn || !other.alive) continue;
            // Only pions have _srcCharge !== undefined and charge !== 0
            if (!other.charge) continue;
            if (other.charge === pn.charge) continue; // same sign: no annihilation
            if (other.age < BOSON_MIN_AGE) continue;
            const dx = pn.pos.x - other.pos.x;
            const dy = pn.pos.y - other.pos.y;
            if (dx * dx + dy * dy < annihDistSq) {
                pn.alive = false;
                other.alive = false;
                _annihPairs.push(pn, other);
                break; // each pion annihilates at most once
            }
        }
    }
    return _annihPairs;
}
const _annihPairs = [];
