import Vec2 from './vec2.js';
import QuadTree, { Rect } from './quadtree.js';
import { BH_THETA, QUADTREE_CAPACITY, SOFTENING_SQ, SOFTENING, DESPAWN_MARGIN, INERTIA_K, MAG_MOMENT_K, MAX_SUBSTEPS, LARMOR_K, RADIATION_THRESHOLD, MAX_PHOTONS, LL_FORCE_CLAMP, FRAME_DRAG_K, TIDAL_STRENGTH, MIN_FRAGMENT_MASS, FRAGMENT_COUNT, HISTORY_SIZE } from './config.js';
import Photon from './photon.js';
import { setVelocity, angwToAngVel, angVelToAngw } from './relativity.js';

function setVelocityFromVel(p, vn, vt, nx, ny, tx, ty) {
    setVelocity(p, nx * vn + tx * vt, ny * vn + ty * vt);
}

export default class Physics {
    constructor() {
        this.boundary = new Rect(0, 0, 0, 0);

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

        this.sim = null; // set externally by Simulation
        this.simTime = 0; // accumulated simulation time for history

        // Accumulated potential energy (set during force calculation)
        this.potentialEnergy = 0;

        // Track whether forces have been initialized
        this._forcesInit = false;
    }

    update(particles, dt, collisionMode, boundaryMode, width, height, offX = 0, offY = 0) {
        this.boundary.x = offX + width / 2;
        this.boundary.y = offY + height / 2;
        this.boundary.w = width * 2;
        this.boundary.h = height * 2;

        const n = particles.length;
        const relOn = this.relativityEnabled;

        // ─── Boris Integrator with Adaptive Substepping ───
        // Separates position-dependent (E-like) forces from velocity-dependent
        // (B-like) forces. The Boris rotation exactly preserves |v|, giving
        // superior long-term stability for magnetic/gravitomagnetic interactions.
        //
        // Steps per sub-step: half-kick(E) → Boris rotate(B) → half-kick(E) →
        //        drift → rebuild tree → collisions → new forces+fields
        //
        // Substep count is determined by max acceleration: dt_safe = √(ε / a_max),
        // nSteps = ceil(dt / dt_safe), capped at MAX_SUBSTEPS.

        // First frame: compute initial forces + B fields if not yet done
        if (!this._forcesInit && n > 0) {
            for (const p of particles) {
                p.angVel = relOn ? angwToAngVel(p.angw, p.radius) : p.angw;
            }
            this._resetForces(particles);
            this._computeAllForces(particles);
            this._forcesInit = true;
        }

        // ─── Determine substep count from max acceleration ───
        let maxAccelSq = 0;
        for (let i = 0; i < n; i++) {
            const p = particles[i];
            const aSq = p.force.magSq() / (p.mass * p.mass);
            if (aSq > maxAccelSq) maxAccelSq = aSq;
        }
        const aMax = Math.sqrt(maxAccelSq);
        const dtSafe = aMax > 0 ? Math.sqrt(SOFTENING / aMax) : dt;
        const nSteps = Math.min(Math.ceil(dt / dtSafe), MAX_SUBSTEPS);
        const dtSub = dt / nSteps;

        const hasMagnetic = this.magneticEnabled;
        const hasGM = this.gravitomagEnabled;

        let lastQt = null;
        for (let step = 0; step < nSteps; step++) {
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

            // Step 5: Rebuild QuadTree with new positions
            const qt = new QuadTree(this.boundary, QUADTREE_CAPACITY);
            for (const p of particles) qt.insert(p);
            qt.calculateMassDistribution();
            lastQt = qt;

            // Step 6: Handle collisions
            if (collisionMode !== 'pass') {
                this.handleCollisions(particles, qt, collisionMode);
            }

            // Step 7: Calculate new forces and B fields
            this._resetForces(particles);
            this._computeAllForces(particles, qt);
        }

        // Compute PE (once per frame, using last substep's tree)
        this.computePE(particles, lastQt);

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
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            const left = offX, top = offY;
            const right = offX + width, bottom = offY + height;

            if (boundaryMode === 'despawn') {
                if (p.pos.x < left - DESPAWN_MARGIN || p.pos.x > right + DESPAWN_MARGIN ||
                    p.pos.y < top - DESPAWN_MARGIN || p.pos.y > bottom + DESPAWN_MARGIN) {
                    particles.splice(i, 1);
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
        }
    }

    _resetForces(particles) {
        for (const p of particles) {
            p.force.set(0, 0);
            p.forceGravity.set(0, 0);
            p.forceCoulomb.set(0, 0);
            p.forceMagnetic.set(0, 0);
            p.forceGravitomag.set(0, 0);
            p.Bz = 0;
            p.Bgz = 0;
            p.dBzdx = 0;
            p.dBzdy = 0;
            p.dBgzdx = 0;
            p.dBgzdy = 0;
            p._frameDragTorque = 0;
        }
    }

    _computeAllForces(particles, qt) {
        if (this.barnesHutEnabled) {
            if (!qt) {
                qt = new QuadTree(this.boundary, QUADTREE_CAPACITY);
                for (const p of particles) qt.insert(p);
                qt.calculateMassDistribution();
            }
            for (const p of particles) {
                this.calculateForce(p, qt, BH_THETA, p.force);
            }
        } else {
            const useSignalDelay = this.signalDelayEnabled && this.relativityEnabled;
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                for (let j = 0; j < particles.length; j++) {
                    if (i === j) continue;
                    const o = particles[j];

                    let sx, sy, svx, svy, sAngVel;
                    if (useSignalDelay && o.histCount >= 2) {
                        const ret = this._getDelayedState(o, p);
                        if (ret) {
                            sx = ret.x; sy = ret.y; svx = ret.vx; svy = ret.vy;
                            sAngVel = o.angVel; // angular velocity not history-tracked
                        } else {
                            sx = o.pos.x; sy = o.pos.y; svx = o.vel.x; svy = o.vel.y;
                            sAngVel = o.angVel;
                        }
                    } else {
                        sx = o.pos.x; sy = o.pos.y; svx = o.vel.x; svy = o.vel.y;
                        sAngVel = o.angVel;
                    }

                    const oRSq = o.radius * o.radius;
                    this._pairForce(p, sx, sy, svx, svy,
                        o.mass, o.charge, sAngVel,
                        MAG_MOMENT_K * o.charge * sAngVel * oRSq,
                        INERTIA_K * o.mass * sAngVel * oRSq, p.force);
                }
            }
        }
    }

    handleCollisions(particles, qt, mode) {
        for (const p1 of particles) {
            if (p1.mass === 0) continue;

            const range = new Rect(p1.pos.x, p1.pos.y, p1.radius * 2, p1.radius * 2);
            const candidates = qt.query(range);

            for (const p2 of candidates) {
                if (p1 === p2 || p2.mass === 0 || p1.id >= p2.id) continue;

                const dist = p1.pos.dist(p2.pos);
                const minDist = p1.radius + p2.radius;

                if (dist < minDist) {
                    if (mode === 'merge') {
                        this.resolveMerge(p1, p2);
                    } else if (mode === 'bounce') {
                        this.resolveBounce(p1, p2, minDist, dist);
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

    resolveMerge(p1, p2) {
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
        p1.angVel = this.relativityEnabled ? angwToAngVel(p1.angw, p1.radius) : p1.angw;

        // Re-derive velocity from proper velocity
        const invG = this.relativityEnabled ? 1 / Math.sqrt(1 + p1.w.magSq()) : 1;
        p1.vel.x = p1.w.x * invG;
        p1.vel.y = p1.w.y * invG;

        p2.mass = 0;
    }

    resolveBounce(p1, p2, minDist, dist) {
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

        if (this.relativityEnabled) {
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
            const tangentialImpulse = this.bounceFriction * (surfaceV1 - surfaceV2) * effectiveMass;

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
            const tangentialImpulse = this.bounceFriction * (surfaceV1 - surfaceV2) * effectiveMass;

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

        const overlap = (minDist - safeDist) / 2 + 0.25;
        p1.pos.x -= nx * overlap;
        p1.pos.y -= ny * overlap;
        p2.pos.x += nx * overlap;
        p2.pos.y += ny * overlap;
    }

    calculateForce(particle, node, theta, out) {
        if (node.totalMass === 0) return;

        const dx = node.centerOfMass.x - particle.pos.x;
        const dy = node.centerOfMass.y - particle.pos.y;
        const dSq = dx * dx + dy * dy;
        const d = Math.sqrt(dSq);
        const size = node.boundary.w * 2;

        if ((!node.divided && node.points.length > 0) || (node.divided && (size / d < theta))) {
            if (!node.divided) {
                for (const other of node.points) {
                    if (other === particle) continue;
                    const otherRSq = other.radius * other.radius;
                    this._pairForce(particle, other.pos.x, other.pos.y, other.vel.x, other.vel.y, other.mass, other.charge, other.angVel, MAG_MOMENT_K * other.charge * other.angVel * otherRSq, INERTIA_K * other.mass * other.angVel * otherRSq, out);
                }
            } else {
                const avgVx = node.totalMass > 0 ? node.totalMomentumX / node.totalMass : 0;
                const avgVy = node.totalMass > 0 ? node.totalMomentumY / node.totalMass : 0;
                this._pairForce(particle, node.centerOfMass.x, node.centerOfMass.y, avgVx, avgVy, node.totalMass, node.totalCharge, 0, node.totalMagneticMoment, node.totalAngularMomentum, out);
            }
        } else if (node.divided) {
            this.calculateForce(particle, node.northwest, theta, out);
            this.calculateForce(particle, node.northeast, theta, out);
            this.calculateForce(particle, node.southwest, theta, out);
            this.calculateForce(particle, node.southeast, theta, out);
        }
    }

    /**
     * Compute force from a source (particle or aggregate node) on a particle.
     * Also accumulates potential energy (only half to avoid double-counting with aggregates).
     *
     * Position-dependent (E-like) forces are accumulated into `out` and per-type vectors:
     * gravity, Coulomb, magnetic dipole, gravitomagnetic dipole.
     *
     * Velocity-dependent (B-like) forces (Lorentz, linear GM) are NOT computed here.
     * Instead, the B and Bg field z-components are accumulated on the particle for use
     * in the Boris rotation step, which handles these forces exactly.
     */
    _pairForce(p, sx, sy, svx, svy, sMass, sCharge, sAngVel, sMagMoment, sAngMomentum, out) {
        const rx = sx - p.pos.x;
        const ry = sy - p.pos.y;
        const rawRSq = rx * rx + ry * ry;
        // Plummer softening: rSq_eff = r² + ε², keeps F = -dU/dr consistent
        const rSq = rawRSq + SOFTENING_SQ;
        const r = Math.sqrt(rSq);
        const invR = 1 / r;
        const invRSq = 1 / rSq;

        // Cross product of source velocity with separation: (v_s × r̂)_z component
        const crossSV = svx * ry - svy * rx;

        // Test particle dipole moments (uniform-density solid sphere)
        // Magnetic moment: μ = ⅕·q·ω·r²; GM moment (angular momentum): L = I·ω
        const pRSq = p.radius * p.radius;
        const pMagMoment = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
        const pAngMomentum = INERTIA_K * p.mass * p.angVel * pRSq;

        if (this.gravityEnabled) {
            const fDir = p.mass * sMass * invRSq * invR;
            out.x += rx * fDir;
            out.y += ry * fDir;
            p.forceGravity.x += rx * fDir;
            p.forceGravity.y += ry * fDir;
        }

        if (this.coulombEnabled) {
            const fDir = -(p.charge * sCharge) * invRSq * invR;
            out.x += rx * fDir;
            out.y += ry * fDir;
            p.forceCoulomb.x += rx * fDir;
            p.forceCoulomb.y += ry * fDir;
        }

        if (this.magneticEnabled) {
            // Dipole radial component: F = 3μ₁μ₂/r⁴ (aligned ⊥-to-plane dipoles repel)
            const fDir = -3 * (pMagMoment * sMagMoment) * invRSq * invRSq * invR;
            out.x += rx * fDir;
            out.y += ry * fDir;
            p.forceMagnetic.x += rx * fDir;
            p.forceMagnetic.y += ry * fDir;

            // Accumulate EM magnetic field Bz for Boris rotation (Lorentz force)
            // B_z = q_s * (v_s × r̂)_z / r³
            p.Bz += sCharge * crossSV * invR * invRSq;

            // ∇Bz w.r.t. observer position (radial + angular terms)
            // ∂Bz/∂px = +3·Bz·rx/r² + q_s·vsy/r³
            // ∂Bz/∂py = +3·Bz·ry/r² - q_s·vsx/r³
            const Bz_contribution = sCharge * crossSV * invR * invRSq;
            p.dBzdx += 3 * Bz_contribution * rx * invRSq + sCharge * svy * invR * invRSq;
            p.dBzdy += 3 * Bz_contribution * ry * invRSq - sCharge * svx * invR * invRSq;
        }

        if (this.gravitomagEnabled) {
            // Dipole radial component: F = 3L₁L₂/r⁴, co-rotating masses attract (GEM flips EM sign)
            const fDir = 3 * (pAngMomentum * sAngMomentum) * invRSq * invRSq * invR;
            out.x += rx * fDir;
            out.y += ry * fDir;
            p.forceGravitomag.x += rx * fDir;
            p.forceGravitomag.y += ry * fDir;

            // Accumulate GM field Bgz for Boris rotation (linear gravitomagnetism)
            // Bg_z = m_s * (v_s × r̂)_z / r³
            p.Bgz += sMass * crossSV * invR * invRSq;

            // ∇Bgz w.r.t. observer position (radial + angular terms)
            const Bgz_contribution = sMass * crossSV * invR * invRSq;
            p.dBgzdx += 3 * Bgz_contribution * rx * invRSq + sMass * svy * invR * invRSq;
            p.dBgzdy += 3 * Bgz_contribution * ry * invRSq - sMass * svx * invR * invRSq;

            // Frame-dragging torque: drives spins toward co-rotation
            const torque = FRAME_DRAG_K * sMass * (sAngVel - p.angVel) * invR * invRSq;
            p._frameDragTorque = (p._frameDragTorque || 0) + torque;
        }
    }

    // ─── Signal Delay ───
    // Solve for delayed time t_del such that |x_source(t_del) - x_observer(now)| = c·(now - t_del)
    // where c = 1 in natural units. Uses Newton-Raphson with 3 iterations.
    _getDelayedState(source, observer) {
        const now = this.simTime;
        const ox = observer.pos.x, oy = observer.pos.y;

        // Initial guess: t_del = now - |current separation|
        const dx0 = source.pos.x - ox, dy0 = source.pos.y - oy;
        let tDel = now - Math.sqrt(dx0 * dx0 + dy0 * dy0);

        for (let iter = 0; iter < 3; iter++) {
            const sp = this._interpolateHistory(source, tDel);
            if (!sp) return null;

            const dx = sp.x - ox, dy = sp.y - oy;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const residual = dist - (now - tDel);
            if (Math.abs(residual) < 0.01) break;

            // Newton step: d(residual)/d(tDel) ≈ -(v·r̂)/r - 1
            const denom = 1 + (sp.vx * dx + sp.vy * dy) / (dist * dist + 1);
            tDel += residual / denom;
        }

        return this._interpolateHistory(source, tDel);
    }

    // Interpolate position/velocity from circular history buffer at time t
    _interpolateHistory(p, t) {
        if (p.histCount < 2) return null;

        // Find bracketing entries via linear scan (buffer is chronological)
        const N = HISTORY_SIZE;
        const start = (p.histHead - p.histCount + N) % N;

        // Check bounds
        const oldest = p.histTime[start];
        const newest = p.histTime[(p.histHead - 1 + N) % N];
        if (t < oldest || t > newest) return null;

        // Linear scan from oldest to find bracket
        let lo = start;
        for (let k = 0; k < p.histCount - 1; k++) {
            const idx = (start + k) % N;
            const nextIdx = (start + k + 1) % N;
            if (p.histTime[idx] <= t && t <= p.histTime[nextIdx]) {
                lo = idx;
                break;
            }
        }
        const hi = (lo + 1) % N;
        const dt = p.histTime[hi] - p.histTime[lo];
        if (dt < 1e-12) return { x: p.histX[lo], y: p.histY[lo], vx: p.histVx[lo], vy: p.histVy[lo] };

        const frac = (t - p.histTime[lo]) / dt;
        return {
            x:  p.histX[lo]  + frac * (p.histX[hi]  - p.histX[lo]),
            y:  p.histY[lo]  + frac * (p.histY[hi]  - p.histY[lo]),
            vx: p.histVx[lo] + frac * (p.histVx[hi] - p.histVx[lo]),
            vy: p.histVy[lo] + frac * (p.histVy[hi] - p.histVy[lo]),
        };
    }

    /**
     * Compute potential energy using same tree/pairwise method as forces.
     * When BH is on: traverses tree per-particle with BH_THETA, divides by 2.
     * When BH is off: exact pairwise i<j (no double-counting).
     */
    computePE(particles, qt) {
        let pe = 0;

        if (this.barnesHutEnabled && qt) {
            for (const p of particles) {
                pe += this._treePE(p, qt, BH_THETA);
            }
            pe *= 0.5; // Each pair counted from both sides
        } else {
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                for (let j = i + 1; j < particles.length; j++) {
                    const o = particles[j];
                    const oRSq = o.radius * o.radius;
                    pe += this._pairPE(p, o.pos.x, o.pos.y,
                        o.mass, o.charge, o.angVel,
                        MAG_MOMENT_K * o.charge * o.angVel * oRSq,
                        INERTIA_K * o.mass * o.angVel * oRSq);
                }
            }
        }

        this.potentialEnergy = pe;
    }

    _treePE(particle, node, theta) {
        if (node.totalMass === 0) return 0;

        const dx = node.centerOfMass.x - particle.pos.x;
        const dy = node.centerOfMass.y - particle.pos.y;
        const dSq = dx * dx + dy * dy;
        const d = Math.sqrt(dSq);
        const size = node.boundary.w * 2;

        if ((!node.divided && node.points.length > 0) || (node.divided && (size / d < theta))) {
            if (!node.divided) {
                let pe = 0;
                for (const other of node.points) {
                    if (other === particle) continue;
                    const oRSq = other.radius * other.radius;
                    pe += this._pairPE(particle, other.pos.x, other.pos.y,
                        other.mass, other.charge, other.angVel,
                        MAG_MOMENT_K * other.charge * other.angVel * oRSq,
                        INERTIA_K * other.mass * other.angVel * oRSq);
                }
                return pe;
            } else {
                return this._pairPE(particle, node.centerOfMass.x, node.centerOfMass.y,
                    node.totalMass, node.totalCharge, 0,
                    node.totalMagneticMoment, node.totalAngularMomentum);
            }
        } else if (node.divided) {
            return this._treePE(particle, node.northwest, theta)
                + this._treePE(particle, node.northeast, theta)
                + this._treePE(particle, node.southwest, theta)
                + this._treePE(particle, node.southeast, theta);
        }
        return 0;
    }

    _pairPE(p, sx, sy, sMass, sCharge, sAngVel, sMagMoment, sAngMomentum) {
        const rx = sx - p.pos.x;
        const ry = sy - p.pos.y;
        const rSq = rx * rx + ry * ry + SOFTENING_SQ;
        const r = Math.sqrt(rSq);
        const invR = 1 / r;
        const invRSq = 1 / rSq;
        const pRSq = p.radius * p.radius;
        const pMagMoment = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
        const pAngMomentum = INERTIA_K * p.mass * p.angVel * pRSq;

        let pe = 0;
        if (this.gravityEnabled)  pe -= p.mass * sMass * invR;
        if (this.coulombEnabled)  pe += p.charge * sCharge * invR;
        if (this.magneticEnabled) pe += (pMagMoment * sMagMoment) * invR * invRSq;
        if (this.gravitomagEnabled) pe -= (pAngMomentum * sAngMomentum) * invR * invRSq;
        return pe;
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
                const distSq = dx * dx + dy * dy;
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
