// ─── Boris Integrator ───
// Adaptive-substep Boris integrator with spin-orbit, frame-drag, and radiation.
// Delegates force computation, collision resolution, PE, and signal delay
// to focused modules.

import QuadTreePool, { Rect } from './quadtree.js';
import { SOFTENING, DESPAWN_MARGIN, INERTIA_K, MAG_MOMENT_K, MAX_SUBSTEPS, LARMOR_K, RADIATION_THRESHOLD, MAX_PHOTONS, LL_FORCE_CLAMP, TIDAL_STRENGTH, MIN_FRAGMENT_MASS, FRAGMENT_COUNT, SOFTENING_SQ, QUADTREE_CAPACITY, BH_THETA, HISTORY_SIZE } from './config.js';
import Photon from './photon.js';
import { angwToAngVel } from './relativity.js';

import { resetForces, computeAllForces, compute1PNPairwise } from './forces.js';
import { handleCollisions } from './collisions.js';
import { computePE } from './potential.js';

export default class Physics {
    constructor() {
        this.boundary = new Rect(0, 0, 0, 0);
        this.pool = new QuadTreePool(QUADTREE_CAPACITY);

        // Force toggles (default all on)
        this.gravityEnabled = true;
        this.coulombEnabled = true;
        this.magneticEnabled = true;
        this.gravitomagEnabled = true;
        this.relativityEnabled = true;
        this.barnesHutEnabled = false;
        this.bounceFriction = 0.4;
        this.radiationEnabled = false;
        this.tidalEnabled = false;
        this.signalDelayEnabled = false;
        this.spinOrbitEnabled = false;
        this.onePNEnabled = false;

        this.sim = null; // set externally by Simulation
        this.simTime = 0; // accumulated simulation time for history

        // Accumulated potential energy (set during force calculation)
        this.potentialEnergy = 0;

        // Track whether forces have been initialized
        this._forcesInit = false;

        // Reusable toggles object passed to extracted force/PE functions (avoids per-frame allocation)
        this._toggles = {
            gravityEnabled: true,
            coulombEnabled: true,
            magneticEnabled: true,
            gravitomagEnabled: true,
            onePNEnabled: false,
        };
    }

    /** Sync cached toggles object with current toggle state. Call once per update(). */
    _syncToggles() {
        this._toggles.gravityEnabled = this.gravityEnabled;
        this._toggles.coulombEnabled = this.coulombEnabled;
        this._toggles.magneticEnabled = this.magneticEnabled;
        this._toggles.gravitomagEnabled = this.gravitomagEnabled;
        this._toggles.onePNEnabled = this.onePNEnabled;
    }

    update(particles, dt, collisionMode, boundaryMode, width, height, offX = 0, offY = 0) {
        this.boundary.x = offX + width / 2;
        this.boundary.y = offY + height / 2;
        this.boundary.w = width * 2;
        this.boundary.h = height * 2;

        const n = particles.length;
        const relOn = this.relativityEnabled;
        this._syncToggles();
        const toggles = this._toggles;

        // ─── Boris Integrator with Adaptive Substepping ───
        // Separates position-dependent (E-like) forces from velocity-dependent
        // (B-like) forces. The Boris rotation exactly preserves |v|, giving
        // superior long-term stability for magnetic/gravitomagnetic interactions.
        //
        // Steps per sub-step: half-kick(E) → Boris rotate(B) → half-kick(E) →
        //        drift → rebuild tree → collisions → new forces+fields
        //
        // Substep count is determined by max acceleration (dt_safe = √(ε / a_max))
        // and cyclotron frequency (≥8 steps per orbit). nSteps = ceil(dt / dt_safe),
        // capped at MAX_SUBSTEPS.

        // First frame: compute initial forces + B fields if not yet done
        if (!this._forcesInit && n > 0) {
            for (const p of particles) {
                p.angVel = relOn ? angwToAngVel(p.angw, p.radius) : p.angw;
            }
            resetForces(particles);
            // Build tree for BH mode (subsequent substeps rebuild in the loop)
            const initRoot = this.barnesHutEnabled
                ? this.pool.build(this.boundary.x, this.boundary.y, this.boundary.w, this.boundary.h, particles)
                : -1;
            computeAllForces(particles, toggles, this.pool, initRoot, this.barnesHutEnabled, this.signalDelayEnabled, this.relativityEnabled, this.simTime);
            this._forcesInit = true;
        }

        const hasMagnetic = this.magneticEnabled;
        const hasGM = this.gravitomagEnabled;

        // Preliminary force pass when velocity-dependent forces are active.
        // Ensures B fields reflect the current particle set (handles newly
        // added particles whose Bz/Bgz would otherwise be stale/zero) so
        // the adaptive substep count accounts for cyclotron frequencies.
        if ((hasMagnetic || hasGM) && this._forcesInit) {
            resetForces(particles);
            const prelimRoot = this.barnesHutEnabled
                ? this.pool.build(this.boundary.x, this.boundary.y, this.boundary.w, this.boundary.h, particles)
                : -1;
            computeAllForces(particles, toggles, this.pool, prelimRoot, this.barnesHutEnabled, this.signalDelayEnabled, this.relativityEnabled, this.simTime);
        }

        // ─── Adaptive substepping with per-step re-evaluation ───
        // Instead of fixing nSteps up front, we consume `dtRemain` in steps
        // whose size is re-evaluated after each force computation, so the
        // substep count adapts to changing B fields (not just the initial ones).
        let dtRemain = dt;
        let totalSteps = 0;
        let lastRoot = -1;
        while (dtRemain > 1e-15 && totalSteps < MAX_SUBSTEPS) {
            // Compute dtSafe from current forces and B fields
            let maxAccelSq = 0;
            let maxCyclotron = 0;
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const aSq = p.force.magSq() / (p.mass * p.mass);
                if (aSq > maxAccelSq) maxAccelSq = aSq;
                if (hasMagnetic && Math.abs(p.Bz) > 0) {
                    const wc = Math.abs(p.charge * p.Bz / p.mass);
                    if (wc > maxCyclotron) maxCyclotron = wc;
                }
                if (hasGM && Math.abs(p.Bgz) > 0) {
                    const wc = 4 * Math.abs(p.Bgz);
                    if (wc > maxCyclotron) maxCyclotron = wc;
                }
            }
            const aMax = Math.sqrt(maxAccelSq);
            let dtSafe = aMax > 0 ? Math.sqrt(SOFTENING / aMax) : dtRemain;
            if (maxCyclotron > 0) {
                const dtCyclotron = (2 * Math.PI / maxCyclotron) / 8;
                if (dtCyclotron < dtSafe) dtSafe = dtCyclotron;
            }
            // Use the smaller of dtSafe and remaining time; if we'd overshoot,
            // split remaining time evenly across the budget we have left.
            const budgetLeft = MAX_SUBSTEPS - totalSteps;
            const stepsNeeded = Math.min(Math.ceil(dtRemain / dtSafe), budgetLeft);
            const dtSub = dtRemain / stepsNeeded;

            totalSteps++;
            dtRemain -= dtSub;
            // Store 1PN forces for velocity-Verlet correction (recomputed after drift)
            const has1PN = toggles.onePNEnabled;
            if (has1PN) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (!p._f1pnOld) p._f1pnOld = { x: 0, y: 0 };
                    p._f1pnOld.x = p.force1PN.x;
                    p._f1pnOld.y = p.force1PN.y;
                }
            }

            // Step 1: Half-kick proper velocity with position-dependent (E-like) forces
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const halfDtOverM = dtSub * 0.5 / p.mass;
                p.w.x += p.force.x * halfDtOverM;
                p.w.y += p.force.y * halfDtOverM;
            }

            // Step 2: Boris rotation for velocity-dependent (B-like) forces
            // Handles EM Lorentz and linear gravitomagnetism exactly.
            // Combined rotation parameter: t = ((q/(2m))·Bz + 2·Bgz) · dtSub / γ⁻
            // s = 2t / (1 + t²)
            if (hasMagnetic || hasGM) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    const gamma = relOn ? Math.sqrt(1 + p.w.magSq()) : 1;

                    let t = 0;
                    if (hasMagnetic) t += (p.charge / (2 * p.mass)) * p.Bz;
                    if (hasGM) t += 2 * p.Bgz;
                    t *= dtSub / gamma;

                    if (t === 0) continue;

                    const s = 2 * t / (1 + t * t);
                    const wx = p.w.x, wy = p.w.y;

                    const wpx = wx + wy * t;
                    const wpy = wy - wx * t;

                    p.w.x = wx + wpy * s;
                    p.w.y = wy - wpx * s;
                }
            }

            // Step 3: Second half-kick with same E-like forces
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const halfDtOverM = dtSub * 0.5 / p.mass;
                p.w.x += p.force.x * halfDtOverM;
                p.w.y += p.force.y * halfDtOverM;
            }

            // Spin-orbit coupling: dE_spin/dt = -μ · (v · ∇B_z)
            if (hasMagnetic && relOn && this.spinOrbitEnabled) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (Math.abs(p.angVel) < 1e-10 || Math.abs(p.charge) < 1e-10) continue;
                    const pRSq = p.radius * p.radius;
                    const mu = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
                    const vDotGradB = p.vel.x * p.dBzdx + p.vel.y * p.dBzdy;
                    const dEspin = -mu * vDotGradB * dtSub;
                    const I = INERTIA_K * p.mass * pRSq;
                    if (Math.abs(I * p.angVel) > 1e-10) {
                        p.angw += dEspin / (I * p.angVel);
                        // Re-derive angVel from angw
                        const sr = p.angw * p.radius;
                        p.angVel = p.angw / Math.sqrt(1 + sr * sr);
                    }
                }
            }

            // GM Spin-orbit coupling: dE_spin/dt = -L · (v · ∇Bgz)
            if (hasGM && relOn && this.spinOrbitEnabled) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (Math.abs(p.angVel) < 1e-10) continue;
                    const pRSq = p.radius * p.radius;
                    const L = INERTIA_K * p.mass * p.angVel * pRSq;
                    const vDotGradBg = p.vel.x * p.dBgzdx + p.vel.y * p.dBgzdy;
                    const dEspin = -L * vDotGradBg * dtSub;
                    const I = INERTIA_K * p.mass * pRSq;
                    if (Math.abs(I * p.angVel) > 1e-10) {
                        p.angw += dEspin / (I * p.angVel);
                        const sr = p.angw * p.radius;
                        p.angVel = p.angw / Math.sqrt(1 + sr * sr);
                    }
                }
            }

            // Stern-Gerlach / Mathisson-Papapetrou force: center-of-mass kick from
            // spin-field gradient coupling. Uses the same gradients as spin-orbit
            // but applies a translational force instead of a spin torque.
            if (hasMagnetic && relOn && this.spinOrbitEnabled) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (Math.abs(p.angVel) < 1e-10 || Math.abs(p.charge) < 1e-10) continue;
                    const pRSq = p.radius * p.radius;
                    const mu = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
                    // F_SG = +mu * grad(Bz)
                    p.w.x += mu * p.dBzdx * dtSub / p.mass;
                    p.w.y += mu * p.dBzdy * dtSub / p.mass;
                }
            }
            if (hasGM && relOn && this.spinOrbitEnabled) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (Math.abs(p.angVel) < 1e-10) continue;
                    const pRSq = p.radius * p.radius;
                    const L = INERTIA_K * p.mass * p.angVel * pRSq;
                    // F_MP = -L * grad(Bgz)  (GEM sign flip)
                    p.w.x -= L * p.dBgzdx * dtSub / p.mass;
                    p.w.y -= L * p.dBgzdy * dtSub / p.mass;
                }
            }

            // Frame-dragging spin alignment torque
            if (hasGM) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (!p._frameDragTorque) continue;
                    const I = INERTIA_K * p.mass * p.radius * p.radius;
                    p.angw += p._frameDragTorque * dtSub / I;
                    const sr = p.angw * p.radius;
                    p.angVel = relOn ? p.angw / Math.sqrt(1 + sr * sr) : p.angw;
                }
            }

            // Abraham-Lorentz radiation reaction via Landau-Lifshitz approximation
            // Replaces direct KE drain with a proper force: jerk + Schott damping terms
            if (this.radiationEnabled && this.sim) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (Math.abs(p.charge) < 1e-10) continue;

                    const wMagSq = p.w.x * p.w.x + p.w.y * p.w.y;
                    if (wMagSq < 1e-20) {
                        p.prevForce.x = p.force.x;
                        p.prevForce.y = p.force.y;
                        continue;
                    }

                    const gamma = relOn ? Math.sqrt(1 + wMagSq) : 1;
                    const qSq = p.charge * p.charge;
                    const tau = 2 * LARMOR_K * qSq / p.mass;

                    // Jerk term: τ · dF/dt ≈ τ · (F - F_prev) / dt
                    const invDt = 1 / dtSub;
                    const jerkX = (p.force.x - p.prevForce.x) * invDt;
                    const jerkY = (p.force.y - p.prevForce.y) * invDt;

                    // Schott damping term: τ · |F|² · v / m
                    const Fsq = p.force.x * p.force.x + p.force.y * p.force.y;
                    const schottScale = Fsq / p.mass;
                    const schottX = schottScale * p.vel.x;
                    const schottY = schottScale * p.vel.y;

                    // Total LL radiation reaction force
                    let fRadX = tau * (jerkX - schottX);
                    let fRadY = tau * (jerkY - schottY);

                    // Relativistic correction: divide by γ³
                    if (relOn && gamma > 1) {
                        const invG3 = 1 / (gamma * gamma * gamma);
                        fRadX *= invG3;
                        fRadY *= invG3;
                    }

                    // Clamp to prevent instability: |F_rad · dt / m| ≤ LL_FORCE_CLAMP · |w|
                    const impulseX = fRadX * dtSub / p.mass;
                    const impulseY = fRadY * dtSub / p.mass;
                    const impulseMag = Math.sqrt(impulseX * impulseX + impulseY * impulseY);
                    const wMag = Math.sqrt(wMagSq);
                    const maxImpulse = LL_FORCE_CLAMP * wMag;

                    if (impulseMag > maxImpulse && impulseMag > 1e-20) {
                        const scale = maxImpulse / impulseMag;
                        fRadX *= scale;
                        fRadY *= scale;
                    }

                    // Measure KE before applying
                    const keBefore = relOn ? (gamma - 1) * p.mass : 0.5 * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);

                    // Apply as kick to proper velocity
                    p.w.x += fRadX * dtSub / p.mass;
                    p.w.y += fRadY * dtSub / p.mass;

                    // NaN guard
                    if (isNaN(p.w.x) || isNaN(p.w.y)) {
                        p.w.x = 0; p.w.y = 0;
                    }

                    // Measure KE after for energy tracking
                    const wMagSqAfter = p.w.x * p.w.x + p.w.y * p.w.y;
                    const gammaAfter = relOn ? Math.sqrt(1 + wMagSqAfter) : 1;
                    const keAfter = relOn ? (gammaAfter - 1) * p.mass : 0.5 * p.mass * wMagSqAfter / (gammaAfter * gammaAfter);
                    const dE = Math.max(0, keBefore - keAfter);
                    this.sim.totalRadiated += dE;

                    // Accumulate radiated momentum (deterministic anti-acceleration direction)
                    if (dE > 0) {
                        const ax = p.force.x / p.mass, ay = p.force.y / p.mass;
                        const radAngle = Math.atan2(ay, ax) + Math.PI;
                        this.sim.totalRadiatedPx += dE * Math.cos(radAngle);
                        this.sim.totalRadiatedPy += dE * Math.sin(radAngle);

                        // Spawn photon if energy exceeds threshold (visual with jitter)
                        if (dE > RADIATION_THRESHOLD && this.sim.photons.length < MAX_PHOTONS) {
                            const spawnAngle = radAngle + (Math.random() - 0.5) * 1.0;
                            this.sim.photons.push(new Photon(
                                p.pos.x, p.pos.y,
                                Math.cos(spawnAngle), Math.sin(spawnAngle),
                                dE
                            ));
                        }
                    }

                    // Store force for next step's jerk computation
                    p.prevForce.x = p.force.x;
                    p.prevForce.y = p.force.y;
                }
            }

            // Step 4: Derive velocity and angular velocity, drift positions
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const invG = relOn ? 1 / Math.sqrt(1 + p.w.magSq()) : 1;
                p.vel.x = p.w.x * invG;
                p.vel.y = p.w.y * invG;
                p.angVel = relOn ? angwToAngVel(p.angw, p.radius) : p.angw;
                p.pos.x += p.vel.x * dtSub;
                p.pos.y += p.vel.y * dtSub;

                // Record history for signal delay
                if (this.signalDelayEnabled) {
                    p._initHistory();
                    const h = p.histHead;
                    p.histX[h] = p.pos.x;
                    p.histY[h] = p.pos.y;
                    p.histVx[h] = p.vel.x;
                    p.histVy[h] = p.vel.y;
                    p.histTime[h] = this.simTime;
                    p.histHead = (h + 1) % HISTORY_SIZE;
                    if (p.histCount < HISTORY_SIZE) p.histCount++;
                }
            }
            this.simTime += dtSub;

            // 1PN velocity-Verlet correction: recompute 1PN at new positions/velocities
            // and apply half the difference as a correction kick for second-order accuracy
            if (has1PN) {
                // Derive coordinate velocities from updated proper velocities
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    const invG = relOn ? 1 / Math.sqrt(1 + p.w.magSq()) : 1;
                    p.vel.x = p.w.x * invG;
                    p.vel.y = p.w.y * invG;
                }
                // Recompute 1PN forces at new state (always pairwise for correction)
                compute1PNPairwise(particles, SOFTENING_SQ);
                // Apply correction kick: w += (F_1PN_new - F_1PN_old) * dt/2 / m
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    const halfDtOverM = dtSub * 0.5 / p.mass;
                    p.w.x += (p.force1PN.x - p._f1pnOld.x) * halfDtOverM;
                    p.w.y += (p.force1PN.y - p._f1pnOld.y) * halfDtOverM;
                }
            }

            // Step 5: Rebuild QuadTree with new positions
            const root = this.pool.build(this.boundary.x, this.boundary.y, this.boundary.w, this.boundary.h, particles);
            lastRoot = root;

            // Step 6: Handle collisions
            if (collisionMode !== 'pass') {
                handleCollisions(particles, this.pool, root, collisionMode, this.bounceFriction, this.relativityEnabled);
            }

            // Step 7: Calculate new forces and B fields
            resetForces(particles);
            computeAllForces(particles, toggles, this.pool, root, this.barnesHutEnabled, this.signalDelayEnabled, this.relativityEnabled, this.simTime);
        }

        // Compute PE (once per frame, using last substep's tree)
        this.potentialEnergy = computePE(particles, toggles, this.pool, lastRoot, this.barnesHutEnabled, BH_THETA);

        // Compute velocity-dependent forces for display only (after final substep).
        // These are applied via Boris rotation (not kicks), but we add them to the
        // per-type display vectors so force component arrows are accurate.
        if (hasMagnetic || hasGM) {
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                if (hasMagnetic) {
                    p.forceMagnetic.x += p.charge * p.vel.y * p.Bz;
                    p.forceMagnetic.y -= p.charge * p.vel.x * p.Bz;
                }
                if (hasGM) {
                    p.forceGravitomag.x += 4 * p.mass * p.vel.y * p.Bgz;
                    p.forceGravitomag.y -= 4 * p.mass * p.vel.x * p.Bgz;
                }
            }
        }

        // Step 8: Handle boundaries (once per frame, after all substeps)
        let writeIdx = 0;
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            const left = offX, top = offY;
            const right = offX + width, bottom = offY + height;

            if (boundaryMode === 'despawn') {
                if (p.pos.x < left - DESPAWN_MARGIN || p.pos.x > right + DESPAWN_MARGIN ||
                    p.pos.y < top - DESPAWN_MARGIN || p.pos.y > bottom + DESPAWN_MARGIN) {
                    continue; // skip — don't copy to output
                }
            } else if (boundaryMode === 'loop') {
                if (p.pos.x < left) p.pos.x += width;
                else if (p.pos.x > right) p.pos.x -= width;
                if (p.pos.y < top) p.pos.y += height;
                else if (p.pos.y > bottom) p.pos.y -= height;
            } else if (boundaryMode === 'bounce') {
                let bounced = false;
                if (p.pos.x < left + p.radius) { p.pos.x = left + p.radius; p.w.x *= -1; bounced = true; }
                else if (p.pos.x > right - p.radius) { p.pos.x = right - p.radius; p.w.x *= -1; bounced = true; }
                if (p.pos.y < top + p.radius) { p.pos.y = top + p.radius; p.w.y *= -1; bounced = true; }
                else if (p.pos.y > bottom - p.radius) { p.pos.y = bottom - p.radius; p.w.y *= -1; bounced = true; }

                if (bounced) {
                    const invG = relOn ? 1 / Math.sqrt(1 + p.w.magSq()) : 1;
                    p.vel.x = p.w.x * invG;
                    p.vel.y = p.w.y * invG;
                }
            }

            particles[writeIdx++] = p;
        }
        particles.length = writeIdx;
    }

    /**
     * Compute PE and store it on this.potentialEnergy.
     * Called externally by main.js (via computeEnergy → energy.js reads physics.potentialEnergy).
     * Also called at end of update() — but may be called independently (e.g. after preset load).
     */
    computePE(particles, root) {
        this._syncToggles();
        const toggles = this._toggles;
        this.potentialEnergy = computePE(particles, toggles, this.pool, root >= 0 ? root : -1, this.barnesHutEnabled, BH_THETA);
    }

    checkTidalBreakup(particles) {
        if (!this.tidalEnabled) return [];
        const fragments = [];

        for (const p of particles) {
            if (p.mass < MIN_FRAGMENT_MASS * FRAGMENT_COUNT) continue;

            const rSq = p.radius * p.radius;

            // Self-gravity binding force at surface: F_bind = m / r²
            const selfGravity = p.mass / rSq;

            // Per-particle self-disruption forces (no neighbor needed)
            // Centrifugal: F = ω² · r
            const centrifugal = p.angVel * p.angVel * p.radius;
            // Coulomb self-repulsion: uniform charge sphere surface field
            // F = q² / (4·r²) in natural units (k=1)
            const coulombSelf = (p.charge * p.charge) / (4 * rSq);

            if (centrifugal + coulombSelf > selfGravity) {
                fragments.push(p);
                continue;
            }

            // Tidal stretching from nearby bodies
            let maxTidal = 0;
            for (const other of particles) {
                if (other === p) continue;
                const dx = other.pos.x - p.pos.x, dy = other.pos.y - p.pos.y;
                const distSq = dx * dx + dy * dy + SOFTENING_SQ;
                const r = Math.sqrt(distSq);
                const tidalAccel = TIDAL_STRENGTH * other.mass * p.radius / (r * distSq);
                if (tidalAccel > maxTidal) maxTidal = tidalAccel;
            }

            // Combined: all outward forces vs binding
            if (maxTidal + centrifugal + coulombSelf > selfGravity) {
                fragments.push(p);
            }
        }

        return fragments;
    }
}
