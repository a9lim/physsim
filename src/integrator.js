// ─── Boris Integrator ───
// Adaptive-substep Boris integrator. Separates E-like (position-dependent) and
// B-like (velocity-dependent) forces for exact |v|-preserving rotation.

import QuadTreePool, { Rect } from './quadtree.js';
import { SOFTENING, DESPAWN_MARGIN, INERTIA_K, MAG_MOMENT_K, MAX_SUBSTEPS, LARMOR_K, RADIATION_THRESHOLD, MAX_PHOTONS, LL_FORCE_CLAMP, TIDAL_STRENGTH, MIN_FRAGMENT_MASS, FRAGMENT_COUNT, SOFTENING_SQ, QUADTREE_CAPACITY, BH_THETA, HISTORY_SIZE, HISTORY_STRIDE, HAWKING_K, MIN_BH_MASS } from './config.js';
import Photon from './photon.js';
import { angwToAngVel } from './relativity.js';

import { resetForces, computeAllForces, compute1PNPairwise } from './forces.js';
import { handleCollisions } from './collisions.js';
import { computePE } from './potential.js';
import { TORUS, KLEIN, RP2, minImage, wrapPosition } from './topology.js';

export default class Physics {
    constructor() {
        this.boundary = new Rect(0, 0, 0, 0);
        this.pool = new QuadTreePool(QUADTREE_CAPACITY);

        this.gravityEnabled = true;
        this.coulombEnabled = true;
        this.magneticEnabled = true;
        this.gravitomagEnabled = true;
        this.relativityEnabled = true;
        this.barnesHutEnabled = false;
        this.bounceFriction = 0.4;
        this.radiationEnabled = false;
        this.blackHoleEnabled = false;
        this.tidalEnabled = false;
        this.tidalLockingEnabled = false;
        this.signalDelayEnabled = true;
        this.spinOrbitEnabled = true;
        this.onePNEnabled = true;

        this.sim = null;
        this.simTime = 0;

        this.domainW = 0;
        this.domainH = 0;
        this.periodic = false;
        this._topologyConst = TORUS;
        this.potentialEnergy = 0;
        this._forcesInit = false;
        this._histStride = 0;

        // Reusable across frames to avoid per-call allocation
        this._toggles = {
            gravityEnabled: true,
            coulombEnabled: true,
            magneticEnabled: true,
            gravitomagEnabled: true,
            onePNEnabled: true,
            tidalLockingEnabled: false,
        };

        this._ghostPool = [];
        this._ghostCount = 0;
        this._treeParticles = [];
    }

    /** Copy current toggle booleans into reusable object (once per frame). */
    _syncToggles() {
        this._toggles.gravityEnabled = this.gravityEnabled;
        this._toggles.coulombEnabled = this.coulombEnabled;
        this._toggles.magneticEnabled = this.magneticEnabled;
        this._toggles.gravitomagEnabled = this.gravitomagEnabled;
        this._toggles.onePNEnabled = this.onePNEnabled;
        this._toggles.tidalLockingEnabled = this.tidalLockingEnabled;
    }

    /** Pool-allocate a ghost at (sx, sy) mirroring p. Flips for non-orientable topologies. */
    _addGhost(p, sx, sy, flipVx = false, flipVy = false) {
        let g;
        if (this._ghostCount < this._ghostPool.length) {
            g = this._ghostPool[this._ghostCount];
        } else {
            g = { pos: { x: 0, y: 0 }, vel: { x: 0, y: 0 }, w: { x: 0, y: 0, magSq() { return this.x * this.x + this.y * this.y; } }, mass: 0, charge: 0, angVel: 0, angw: 0, radius: 0, id: -1, isGhost: true, original: null };
            this._ghostPool.push(g);
        }
        g.pos.x = sx; g.pos.y = sy;
        g.vel.x = flipVx ? -p.vel.x : p.vel.x;
        g.vel.y = flipVy ? -p.vel.y : p.vel.y;
        g.w.x = flipVx ? -p.w.x : p.w.x;
        g.w.y = flipVy ? -p.w.y : p.w.y;
        g.mass = p.mass; g.charge = p.charge;
        g.angVel = (flipVx || flipVy) ? -p.angVel : p.angVel;
        g.angw = (flipVx || flipVy) ? -p.angw : p.angw;
        g.radius = p.radius; g.id = -1;
        g.original = p;
        this._ghostCount++;
        return g;
    }

    /** Generate periodic image ghosts for particles within BH_THETA margin of edges. */
    _generateGhosts(particles) {
        this._ghostCount = 0;
        const W = this.domainW, H = this.domainH;
        const margin = Math.max(W, H) * BH_THETA;
        const topo = this._topologyConst;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            const x = p.pos.x, y = p.pos.y;
            const nearL = x < margin, nearR = x > W - margin;
            const nearT = y < margin, nearB = y > H - margin;

            if (topo === TORUS) {
                if (nearL) this._addGhost(p, x + W, y);
                if (nearR) this._addGhost(p, x - W, y);
                if (nearT) this._addGhost(p, x, y + H);
                if (nearB) this._addGhost(p, x, y - H);
                if (nearL && nearT) this._addGhost(p, x + W, y + H);
                if (nearL && nearB) this._addGhost(p, x + W, y - H);
                if (nearR && nearT) this._addGhost(p, x - W, y + H);
                if (nearR && nearB) this._addGhost(p, x - W, y - H);
            } else if (topo === KLEIN) {
                // Klein: x wraps normally; y-wrap flips x and negates vx
                if (nearL) this._addGhost(p, x + W, y);
                if (nearR) this._addGhost(p, x - W, y);
                if (nearT) this._addGhost(p, W - x, y + H, true, false);
                if (nearB) this._addGhost(p, W - x, y - H, true, false);
                if (nearL && nearT) this._addGhost(p, W - x + W, y + H, true, false);
                if (nearL && nearB) this._addGhost(p, W - x + W, y - H, true, false);
                if (nearR && nearT) this._addGhost(p, W - x - W, y + H, true, false);
                if (nearR && nearB) this._addGhost(p, W - x - W, y - H, true, false);
            } else {
                // RP²: x-wrap flips y, y-wrap flips x
                if (nearL) this._addGhost(p, x + W, H - y, false, true);
                if (nearR) this._addGhost(p, x - W, H - y, false, true);
                if (nearT) this._addGhost(p, W - x, y + H, true, false);
                if (nearB) this._addGhost(p, W - x, y - H, true, false);
                if (nearL && nearT) this._addGhost(p, W - x + W, y + H, true, false);
                if (nearL && nearB) this._addGhost(p, W - x + W, y - H, true, false);
                if (nearR && nearT) this._addGhost(p, W - x - W, y + H, true, false);
                if (nearR && nearB) this._addGhost(p, W - x - W, y - H, true, false);
            }
        }
    }

    /** Build quadtree, including periodic ghosts if applicable. Returns root index. */
    _buildTree(particles) {
        if (this.periodic) {
            this._generateGhosts(particles);
            const tp = this._treeParticles;
            const nReal = particles.length;
            const nGhost = this._ghostCount;
            const total = nReal + nGhost;
            tp.length = total;
            for (let i = 0; i < nReal; i++) tp[i] = particles[i];
            for (let i = 0; i < nGhost; i++) tp[nReal + i] = this._ghostPool[i];
            return this.pool.build(this.boundary.x, this.boundary.y, this.boundary.w, this.boundary.h, tp);
        }
        return this.pool.build(this.boundary.x, this.boundary.y, this.boundary.w, this.boundary.h, particles);
    }

    update(particles, dt, collisionMode, boundaryMode, topology, width, height, offX = 0, offY = 0) {
        this.boundary.x = offX + width / 2;
        this.boundary.y = offY + height / 2;
        this.boundary.w = width * 2;
        this.boundary.h = height * 2;

        this.domainW = width;
        this.domainH = height;
        this.periodic = (boundaryMode === 'loop');
        this._topologyConst = topology === 'klein' ? KLEIN : topology === 'rp2' ? RP2 : TORUS;

        let n = particles.length;
        const relOn = this.relativityEnabled;
        this._syncToggles();
        const toggles = this._toggles;

        // ─── Boris Integrator with Adaptive Substepping ───
        // Per substep: half-kick(E) → Boris rotate(B) → half-kick(E) → drift
        // → rebuild tree → collisions → recompute forces.
        // dtSafe from max acceleration (√(ε/a_max)) and cyclotron period (T/8).

        // First frame: bootstrap initial forces + B fields
        if (!this._forcesInit && n > 0) {
            for (const p of particles) {
                p.angVel = relOn ? angwToAngVel(p.angw, p.radius) : p.angw;
            }
            resetForces(particles);
            const initRoot = this.barnesHutEnabled
                ? this._buildTree(particles)
                : -1;
            computeAllForces(particles, toggles, this.pool, initRoot, this.barnesHutEnabled, this.signalDelayEnabled, this.relativityEnabled, this.simTime, this.periodic, this.domainW, this.domainH, this._topologyConst);
            this._forcesInit = true;
        }

        const hasMagnetic = this.magneticEnabled;
        const hasGM = this.gravitomagEnabled;

        // Preliminary force pass: ensures Bz/Bgz are current for cyclotron
        // frequency estimation in the adaptive substep loop below.
        if ((hasMagnetic || hasGM) && this._forcesInit) {
            resetForces(particles);
            const prelimRoot = this.barnesHutEnabled
                ? this._buildTree(particles)
                : -1;
            computeAllForces(particles, toggles, this.pool, prelimRoot, this.barnesHutEnabled, this.signalDelayEnabled, this.relativityEnabled, this.simTime, this.periodic, this.domainW, this.domainH, this._topologyConst);
        }

        // ─── Adaptive substepping ───
        // Re-evaluates dtSafe each iteration so substep size tracks changing fields.
        let dtRemain = dt;
        let totalSteps = 0;
        let lastRoot = -1;
        while (dtRemain > 1e-15 && totalSteps < MAX_SUBSTEPS) {
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
                    // GM cyclotron: factor of 4 from F = 4m(v × Bg)
                    const wc = 4 * Math.abs(p.Bgz);
                    if (wc > maxCyclotron) maxCyclotron = wc;
                }
            }
            const aMax = Math.sqrt(maxAccelSq);
            let dtSafe = aMax > 0 ? Math.sqrt(SOFTENING / aMax) : dtRemain;
            if (maxCyclotron > 0) {
                // At least 8 substeps per cyclotron orbit
                const dtCyclotron = (2 * Math.PI / maxCyclotron) / 8;
                if (dtCyclotron < dtSafe) dtSafe = dtCyclotron;
            }
            const budgetLeft = MAX_SUBSTEPS - totalSteps;
            const stepsNeeded = Math.min(Math.ceil(dtRemain / dtSafe), budgetLeft);
            const dtSub = dtRemain / stepsNeeded;

            totalSteps++;
            dtRemain -= dtSub;

            // Save pre-drift 1PN forces for velocity-Verlet correction
            const has1PN = toggles.onePNEnabled;
            if (has1PN) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    p._f1pnOld.x = p.force1PN.x;
                    p._f1pnOld.y = p.force1PN.y;
                    p._f1pnEMOld.x = p.force1PNEM.x;
                    p._f1pnEMOld.y = p.force1PNEM.y;
                }
            }

            // Step 1: Half-kick w with E-like forces
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const halfDtOverM = dtSub * 0.5 / p.mass;
                p.w.x += p.force.x * halfDtOverM;
                p.w.y += p.force.y * halfDtOverM;
            }

            // Step 2: Boris rotation (Lorentz + linear GM)
            // t = ((q/(2m))·Bz + 2·Bgz)·dt/γ,  s = 2t/(1+t²)
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

            // Step 3: Second half-kick (same E-like forces)
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const halfDtOverM = dtSub * 0.5 / p.mass;
                p.w.x += p.force.x * halfDtOverM;
                p.w.y += p.force.y * halfDtOverM;
            }

            // EM spin-orbit: dE_spin = −μ·(v·∇Bz)·dt, transferred to/from angw
            if (hasMagnetic && this.spinOrbitEnabled) {
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
                        const sr = p.angw * p.radius;
                        p.angVel = p.angw / Math.sqrt(1 + sr * sr);
                    }
                }
            }

            // GM spin-orbit: dE_spin = −L·(v·∇Bgz)·dt
            if (hasGM && this.spinOrbitEnabled) {
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

            // Stern-Gerlach (EM) and Mathisson-Papapetrou (GM) translational kicks
            // from spin-field gradient coupling
            if (hasMagnetic && this.spinOrbitEnabled) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (Math.abs(p.angVel) < 1e-10 || Math.abs(p.charge) < 1e-10) continue;
                    const pRSq = p.radius * p.radius;
                    const mu = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
                    // F_SG = +μ·∇Bz
                    p.w.x += mu * p.dBzdx * dtSub / p.mass;
                    p.w.y += mu * p.dBzdy * dtSub / p.mass;
                }
            }
            if (hasGM && this.spinOrbitEnabled) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (Math.abs(p.angVel) < 1e-10) continue;
                    const pRSq = p.radius * p.radius;
                    const L = INERTIA_K * p.mass * p.angVel * pRSq;
                    // F_MP = −L·∇Bgz (GEM sign flip)
                    p.w.x -= L * p.dBgzdx * dtSub / p.mass;
                    p.w.y -= L * p.dBgzdy * dtSub / p.mass;
                }
            }

            // Frame-dragging torque: aligns spins toward co-rotation (requires Relativity)
            if (hasGM && relOn) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (!p._frameDragTorque) continue;
                    const I = INERTIA_K * p.mass * p.radius * p.radius;
                    p.angw += p._frameDragTorque * dtSub / I;
                    const sr = p.angw * p.radius;
                    p.angVel = relOn ? p.angw / Math.sqrt(1 + sr * sr) : p.angw;
                }
            }

            // Tidal locking torque: drives spin toward synchronous rotation
            if (this.tidalLockingEnabled) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (!p._tidalTorque) continue;
                    const I = INERTIA_K * p.mass * p.radius * p.radius;
                    p.angw += p._tidalTorque * dtSub / I;
                    const sr = p.angw * p.radius;
                    p.angVel = relOn ? p.angw / Math.sqrt(1 + sr * sr) : p.angw;
                }
            }

            // Landau-Lifshitz radiation reaction (jerk term only, no Schott damping)
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
                    const invDt = 1 / dtSub;
                    let fRadX = tau * (p.force.x - p.prevForce.x) * invDt;
                    let fRadY = tau * (p.force.y - p.prevForce.y) * invDt;

                    if (relOn && gamma > 1) {
                        const invG3 = 1 / (gamma * gamma * gamma);
                        fRadX *= invG3;
                        fRadY *= invG3;
                    }

                    // Clamp: |F_rad·dt/m| ≤ LL_FORCE_CLAMP·|w| to prevent runaway
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

                    const keBefore = relOn ? (gamma - 1) * p.mass : 0.5 * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);

                    p.w.x += fRadX * dtSub / p.mass;
                    p.w.y += fRadY * dtSub / p.mass;
                    p.forceRadiation.x += fRadX;
                    p.forceRadiation.y += fRadY;

                    if (isNaN(p.w.x) || isNaN(p.w.y)) {
                        p.w.x = 0; p.w.y = 0;
                    }

                    const wMagSqAfter = p.w.x * p.w.x + p.w.y * p.w.y;
                    const gammaAfter = relOn ? Math.sqrt(1 + wMagSqAfter) : 1;
                    const keAfter = relOn ? (gammaAfter - 1) * p.mass : 0.5 * p.mass * wMagSqAfter / (gammaAfter * gammaAfter);
                    const dE = Math.max(0, keBefore - keAfter);
                    this.sim.totalRadiated += dE;

                    if (dE > 0) {
                        const ax = p.force.x / p.mass, ay = p.force.y / p.mass;
                        const radAngle = Math.atan2(ay, ax) + Math.PI;
                        this.sim.totalRadiatedPx += dE * Math.cos(radAngle);
                        this.sim.totalRadiatedPy += dE * Math.sin(radAngle);

                        p._radAccum = (p._radAccum || 0) + dE;
                        if (p._radAccum >= RADIATION_THRESHOLD && this.sim.photons.length < MAX_PHOTONS) {
                            // sin²θ dipole pattern: peak emission ⊥ to acceleration
                            const accelAngle = Math.atan2(ay, ax);
                            let theta;
                            do { theta = Math.random() * 6.283185307; }
                            while (Math.random() > Math.sin(theta) * Math.sin(theta));
                            let emitAngle = accelAngle + theta;

                            // Relativistic aberration: beam forward at high γ
                            if (gamma > 1.01) {
                                const beta = Math.sqrt(1 - 1 / (gamma * gamma));
                                const velAngle = Math.atan2(p.vel.y, p.vel.x);
                                const delta = emitAngle - velAngle;
                                const sinD = Math.sin(delta), cosD = Math.cos(delta);
                                const denom = 1 + beta * cosD;
                                emitAngle = velAngle + Math.atan2(sinD / (gamma * denom), (cosD + beta) / denom);
                            }

                            const cosA = Math.cos(emitAngle), sinA = Math.sin(emitAngle);
                            this.sim.photons.push(new Photon(
                                p.pos.x + cosA * (p.radius + 1),
                                p.pos.y + sinA * (p.radius + 1),
                                cosA, sinA,
                                p._radAccum, p.id
                            ));
                            p._radAccum = 0;
                        }
                    }

                    p.prevForce.x = p.force.x;
                    p.prevForce.y = p.force.y;
                }
            }

            // Hawking radiation: P = HAWKING_K / M², isotropic emission, mass loss
            if (this.blackHoleEnabled && this.sim) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (p.mass <= MIN_BH_MASS) continue;
                    const power = HAWKING_K / (p.mass * p.mass);
                    const dE = power * dtSub;
                    p.mass -= dE;
                    p.radius = 2 * p.mass;
                    this.sim.totalRadiated += dE;

                    p._hawkAccum = (p._hawkAccum || 0) + dE;
                    if (p._hawkAccum >= RADIATION_THRESHOLD && this.sim.photons.length < MAX_PHOTONS) {
                        const emitAngle = Math.random() * 6.283185307;
                        const cosA = Math.cos(emitAngle), sinA = Math.sin(emitAngle);
                        this.sim.photons.push(new Photon(
                            p.pos.x + cosA * (p.radius + 1),
                            p.pos.y + sinA * (p.radius + 1),
                            cosA, sinA,
                            p._hawkAccum, p.id
                        ));
                        this.sim.totalRadiatedPx += p._hawkAccum * cosA;
                        this.sim.totalRadiatedPy += p._hawkAccum * sinA;
                        p._hawkAccum = 0;
                    }
                }
            }

            // Step 4: Derive coordinate velocity from w, then drift positions
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const invG = relOn ? 1 / Math.sqrt(1 + p.w.magSq()) : 1;
                p.vel.x = p.w.x * invG;
                p.vel.y = p.w.y * invG;
                p.angVel = relOn ? angwToAngVel(p.angw, p.radius) : p.angw;
                p.pos.x += p.vel.x * dtSub;
                p.pos.y += p.vel.y * dtSub;

            }
            this.simTime += dtSub;

            // 1PN velocity-Verlet correction: w += (F_new − F_old)·dt/(2m)
            if (has1PN) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    const invG = relOn ? 1 / Math.sqrt(1 + p.w.magSq()) : 1;
                    p.vel.x = p.w.x * invG;
                    p.vel.y = p.w.y * invG;
                }
                compute1PNPairwise(particles, SOFTENING_SQ, this.periodic, this.domainW, this.domainH, this.domainW * 0.5, this.domainH * 0.5, this._topologyConst, this.gravityEnabled, this.coulombEnabled);
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    const halfDtOverM = dtSub * 0.5 / p.mass;
                    p.w.x += (p.force1PN.x - p._f1pnOld.x + p.force1PNEM.x - p._f1pnEMOld.x) * halfDtOverM;
                    p.w.y += (p.force1PN.y - p._f1pnOld.y + p.force1PNEM.y - p._f1pnEMOld.y) * halfDtOverM;
                }
            }

            // Step 5: Rebuild quadtree at new positions
            const root = this._buildTree(particles);
            lastRoot = root;

            // Step 6: Collisions
            if (collisionMode !== 'pass') {
                handleCollisions(particles, this.pool, root, collisionMode, this.bounceFriction, this.relativityEnabled, this.periodic, this.domainW, this.domainH, this._topologyConst);
                n = particles.length;
            }

            // Photon absorption: p = E/c = E (natural units)
            if (this.radiationEnabled && this.sim && this.sim.photons.length > 0) {
                const photons = this.sim.photons;
                for (let pi = photons.length - 1; pi >= 0; pi--) {
                    const ph = photons[pi];
                    if (!ph.alive) continue;
                    ph.age++;
                    // Query quadtree for nearby particles
                    const candidates = this.pool.query(root,
                        ph.pos.x, ph.pos.y, SOFTENING, SOFTENING);
                    for (const target of candidates) {
                        if (target.isGhost) continue;
                        if (target.id === ph.emitterId && ph.age < 3) continue; // self-absorption guard
                        const dx = ph.pos.x - target.pos.x;
                        const dy = ph.pos.y - target.pos.y;
                        if (dx * dx + dy * dy < target.radius * target.radius) {
                            const impulse = ph.energy;
                            target.w.x += impulse * ph.vel.x / target.mass;
                            target.w.y += impulse * ph.vel.y / target.mass;
                            this.sim.totalRadiated -= ph.energy;
                            this.sim.totalRadiatedPx -= ph.energy * ph.vel.x;
                            this.sim.totalRadiatedPy -= ph.energy * ph.vel.y;
                            ph.alive = false;
                            break;
                        }
                    }
                }
            }

            // Snapshot radiation display force before resetForces() clears it
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                p._radDisplayX = p.forceRadiation.x;
                p._radDisplayY = p.forceRadiation.y;
            }

            // Step 7: Recompute forces and B fields for next substep
            resetForces(particles);
            computeAllForces(particles, toggles, this.pool, root, this.barnesHutEnabled, this.signalDelayEnabled, this.relativityEnabled, this.simTime, this.periodic, this.domainW, this.domainH, this._topologyConst);
        }

        // Record signal delay history (strided: ~60 snapshots/sec at 100× speed)
        if (this.signalDelayEnabled && n > 0 && ++this._histStride >= HISTORY_STRIDE) {
            this._histStride = 0;
            for (let i = 0; i < n; i++) {
                const p = particles[i];
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

        // PE once per frame, reusing last substep's tree
        this.potentialEnergy = computePE(particles, toggles, this.pool, lastRoot, this.barnesHutEnabled, BH_THETA, this.periodic, this.domainW, this.domainH, this._topologyConst);

        // Reconstruct B-like display forces from final-substep fields.
        // Boris rotation applied these implicitly; we reconstruct for arrow rendering.
        // Reconstruct velocity-dependent display forces from final-substep fields
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

        // Reconstruct spin-orbit display values from final-substep fields
        if (hasMagnetic && this.spinOrbitEnabled) {
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                if (Math.abs(p.angVel) < 1e-10 || Math.abs(p.charge) < 1e-10) continue;
                const pRSq = p.radius * p.radius;
                const mu = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
                p.torqueSpinOrbit += -mu * (p.vel.x * p.dBzdx + p.vel.y * p.dBzdy);
                p.forceSpinCurv.x += mu * p.dBzdx;
                p.forceSpinCurv.y += mu * p.dBzdy;
            }
        }
        if (hasGM && this.spinOrbitEnabled) {
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                if (Math.abs(p.angVel) < 1e-10) continue;
                const pRSq = p.radius * p.radius;
                const L = INERTIA_K * p.mass * p.angVel * pRSq;
                p.torqueSpinOrbit += -L * (p.vel.x * p.dBgzdx + p.vel.y * p.dBgzdy);
                p.forceSpinCurv.x -= L * p.dBgzdx;
                p.forceSpinCurv.y -= L * p.dBgzdy;
            }
        }
        // Frame-drag display: requires Relativity (matches substep gate)
        if (hasGM && relOn) {
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                if (p._frameDragTorque) p.torqueFrameDrag = p._frameDragTorque;
            }
        }
        // Tidal torque display
        if (this.tidalLockingEnabled) {
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                if (p._tidalTorque) p.torqueTidal = p._tidalTorque;
            }
        }

        // Restore radiation display force from snapshot
        if (this.radiationEnabled) {
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                p.forceRadiation.x = p._radDisplayX || 0;
                p.forceRadiation.y = p._radDisplayY || 0;
            }
        }

        // Step 8: Boundary handling (once per frame)
        let writeIdx = 0;
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            const left = offX, top = offY;
            const right = offX + width, bottom = offY + height;

            if (boundaryMode === 'despawn') {
                if (p.pos.x < left - DESPAWN_MARGIN || p.pos.x > right + DESPAWN_MARGIN ||
                    p.pos.y < top - DESPAWN_MARGIN || p.pos.y > bottom + DESPAWN_MARGIN) {
                    continue;
                }
            } else if (boundaryMode === 'loop') {
                wrapPosition(p, this._topologyConst, width, height);
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

    /** Recompute PE independently (e.g. after preset load). */
    computePE(particles, root) {
        this._syncToggles();
        const toggles = this._toggles;
        this.potentialEnergy = computePE(particles, toggles, this.pool, root >= 0 ? root : -1, this.barnesHutEnabled, BH_THETA, this.periodic, this.domainW, this.domainH, this._topologyConst);
    }

    checkTidalBreakup(particles) {
        if (!this.tidalEnabled) return [];
        const fragments = [];

        for (const p of particles) {
            if (p.mass < MIN_FRAGMENT_MASS * FRAGMENT_COUNT) continue;

            const rSq = p.radius * p.radius;

            const selfGravity = p.mass / rSq;      // binding: m/r²
            const centrifugal = p.angVel * p.angVel * p.radius;  // ω²r
            const coulombSelf = (p.charge * p.charge) / (4 * rSq); // q²/(4r²)

            if (centrifugal + coulombSelf > selfGravity) {
                fragments.push(p);
                continue;
            }

            // Tidal stretching from neighbors: TIDAL_STRENGTH·M·r/d³
            let maxTidal = 0;
            const _periodic = this.periodic;
            const _halfDomW = this.domainW * 0.5, _halfDomH = this.domainH * 0.5;
            const _domW = this.domainW, _domH = this.domainH;
            const _topo = this._topologyConst;
            const _miOut = { x: 0, y: 0 };
            for (const other of particles) {
                if (other === p) continue;
                let dx = other.pos.x - p.pos.x, dy = other.pos.y - p.pos.y;
                if (_periodic) {
                    minImage(p.pos.x, p.pos.y, other.pos.x, other.pos.y, _topo, _domW, _domH, _halfDomW, _halfDomH, _miOut);
                    dx = _miOut.x; dy = _miOut.y;
                }
                const distSq = dx * dx + dy * dy + SOFTENING_SQ;
                const r = Math.sqrt(distSq);
                const tidalAccel = TIDAL_STRENGTH * other.mass * p.radius / (r * distSq);
                if (tidalAccel > maxTidal) maxTidal = tidalAccel;
            }

            // Fragment if combined outward forces exceed binding
            if (maxTidal + centrifugal + coulombSelf > selfGravity) {
                fragments.push(p);
            }
        }

        return fragments;
    }
}
