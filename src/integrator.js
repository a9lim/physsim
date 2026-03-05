// ─── Boris Integrator ───
// Adaptive-substep Boris integrator. Separates E-like (position-dependent) and
// B-like (velocity-dependent) forces for exact |v|-preserving rotation.

import QuadTreePool, { Rect } from './quadtree.js';
import { SOFTENING, DESPAWN_MARGIN, INERTIA_K, MAG_MOMENT_K, MAX_SUBSTEPS, MIN_MASS, MAX_PHOTONS, LL_FORCE_CLAMP, TIDAL_STRENGTH, FRAGMENT_COUNT, SOFTENING_SQ, QUADTREE_CAPACITY, BH_THETA, HISTORY_SIZE, HISTORY_STRIDE, DEFAULT_YUKAWA_G2, DEFAULT_YUKAWA_MU, DEFAULT_AXION_G, DEFAULT_AXION_MASS, ROCHE_THRESHOLD, ROCHE_TRANSFER_RATE, ROCHE_MIN_PACKET, DEFAULT_HUBBLE } from './config.js';
import Photon from './photon.js';
import { angwToAngVel } from './relativity.js';

import { resetForces, computeAllForces, compute1PNPairwise } from './forces.js';
import { handleCollisions } from './collisions.js';
import { computePE } from './potential.js';
import { TORUS, KLEIN, RP2, minImage, wrapPosition } from './topology.js';

// Reused by tidal breakup to avoid per-call allocation
const _tidalMiOut = { x: 0, y: 0 };

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
        this.radiationEnabled = true;
        this.blackHoleEnabled = false;
        this.tidalEnabled = false;
        this.tidalLockingEnabled = false;
        this.signalDelayEnabled = true;
        this.spinOrbitEnabled = true;
        this.onePNEnabled = true;

        this.yukawaEnabled = false;
        this.yukawaG2 = DEFAULT_YUKAWA_G2;
        this.yukawaMu = DEFAULT_YUKAWA_MU;

        this.axionEnabled = false;
        this.axionG = DEFAULT_AXION_G;
        this.axionMass = DEFAULT_AXION_MASS;

        this.gwRadiationEnabled = false;
        this.expansionEnabled = false;
        this.hubbleParam = 0.001;

        this.sim = null;
        this.simTime = 0;
        this._quadHistory = []; // {Ixx, Ixy, Iyy, Qxx, Qxy, Qyy, t}
        this._gwAccum = 0;

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
            yukawaEnabled: false,
            yukawaG2: 1.0,
            yukawaMu: 0.2,
            axionEnabled: false,
            axionModulation: 1.0,
        };

        this._ghostPool = [];
        this._ghostCount = 0;
        this._treeParticles = [];
        this._lastRoot = -1;
    }

    /** Copy current toggle booleans into reusable object (once per frame). */
    _syncToggles() {
        this._toggles.gravityEnabled = this.gravityEnabled;
        this._toggles.coulombEnabled = this.coulombEnabled;
        this._toggles.magneticEnabled = this.magneticEnabled;
        this._toggles.gravitomagEnabled = this.gravitomagEnabled;
        this._toggles.onePNEnabled = this.onePNEnabled;
        this._toggles.tidalLockingEnabled = this.tidalLockingEnabled;
        this._toggles.yukawaEnabled = this.yukawaEnabled;
        this._toggles.yukawaG2 = this.yukawaG2;
        this._toggles.yukawaMu = this.yukawaMu;
        this._toggles.axionEnabled = this.axionEnabled;
        if (this.axionEnabled) {
            this._toggles.axionModulation = 1 + this.axionG * Math.cos(this.axionMass * this.simTime);
        } else {
            this._toggles.axionModulation = 1.0;
        }
    }

    /** Pool-allocate a ghost at (sx, sy) mirroring p. Flips for non-orientable topologies. */
    _addGhost(p, sx, sy, flipVx = false, flipVy = false) {
        let g;
        if (this._ghostCount < this._ghostPool.length) {
            g = this._ghostPool[this._ghostCount];
        } else {
            g = { pos: { x: 0, y: 0 }, vel: { x: 0, y: 0 }, w: { x: 0, y: 0, magSq() { return this.x * this.x + this.y * this.y; } }, mass: 0, charge: 0, angVel: 0, angw: 0, radius: 0, radiusSq: 0, invMass: 0, id: -1, isGhost: true, original: null };
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
        g.radius = p.radius; g.radiusSq = p.radiusSq; g.invMass = p.invMass; g.id = -1;
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

        // Bz/Bgz for cyclotron frequency estimation persist from the previous
        // frame's last substep (or from the bootstrap above). No preliminary
        // force pass needed.

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
                const aSq = p.force.magSq() * p.invMass * p.invMass;
                if (aSq > maxAccelSq) maxAccelSq = aSq;
                if (hasMagnetic && Math.abs(p.Bz) > 0) {
                    const wc = Math.abs(p.charge * p.Bz * p.invMass);
                    if (wc > maxCyclotron) maxCyclotron = wc;
                }
                if (hasGM && Math.abs(p.Bgz) > 0) {
                    // GM cyclotron: factor of 4 from F = 4m(v × Bg)
                    const wc = 4 * Math.abs(p.Bgz);
                    if (wc > maxCyclotron) maxCyclotron = wc;
                }
            }
            const aMax = Math.sqrt(maxAccelSq);
            let dtSafe = aMax > 1e-20 ? Math.sqrt(SOFTENING / aMax) : dtRemain;
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
                }
            }

            // Step 1: Half-kick w with E-like forces
            const halfDt = dtSub * 0.5;
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const halfDtOverM = halfDt * p.invMass;
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
                    if (hasMagnetic) t += (p.charge * 0.5 * p.invMass) * p.Bz;
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
                const halfDtOverM = halfDt * p.invMass;
                p.w.x += p.force.x * halfDtOverM;
                p.w.y += p.force.y * halfDtOverM;
            }

            // Spin-orbit energy transfer + Stern-Gerlach/Mathisson-Papapetrou kicks (fused)
            if (this.spinOrbitEnabled && (hasMagnetic || hasGM)) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (Math.abs(p.angVel) < 1e-10) continue;
                    const pRSq = p.radiusSq;
                    const IxOmega = INERTIA_K * p.mass * pRSq * p.angVel;
                    if (Math.abs(IxOmega) < 1e-10) continue;
                    const dtOverM = dtSub * p.invMass;

                    if (hasMagnetic && Math.abs(p.charge) > 1e-10) {
                        const mu = MAG_MOMENT_K * p.charge * p.angVel * pRSq;
                        // Energy transfer
                        p.angw -= mu * (p.vel.x * p.dBzdx + p.vel.y * p.dBzdy) * dtSub / IxOmega;
                        // Stern-Gerlach translational kick
                        p.w.x += mu * p.dBzdx * dtOverM;
                        p.w.y += mu * p.dBzdy * dtOverM;
                    }
                    if (hasGM) {
                        const L = INERTIA_K * p.mass * p.angVel * pRSq;
                        // Energy transfer
                        p.angw -= L * (p.vel.x * p.dBgzdx + p.vel.y * p.dBgzdy) * dtSub / IxOmega;
                        // Mathisson-Papapetrou translational kick
                        p.w.x -= L * p.dBgzdx * dtOverM;
                        p.w.y -= L * p.dBgzdy * dtOverM;
                    }
                    const sr = p.angw * p.radius;
                    p.angVel = p.angw / Math.sqrt(1 + sr * sr);
                }
            }

            // Frame-dragging torque + tidal locking (fused)
            if ((hasGM && relOn) || this.tidalLockingEnabled) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    let torque = 0;
                    if (hasGM && relOn) torque += p._frameDragTorque;
                    if (this.tidalLockingEnabled) torque += p._tidalTorque;
                    if (torque === 0) continue;
                    const I = INERTIA_K * p.mass * p.radiusSq;
                    p.angw += torque * dtSub / I;
                    const sr = p.angw * p.radius;
                    p.angVel = relOn ? p.angw / Math.sqrt(1 + sr * sr) : p.angw;
                }
            }

            // Landau-Lifshitz radiation reaction (full 1/c² terms)
            // F_rad = τ·[dF/dt / γ³ − v·F²/(m·γ²) + F·(v·F)/(m·γ⁴)]
            if (this.radiationEnabled && this.sim) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (Math.abs(p.charge) < 1e-10) continue;

                    const wMagSq = p.w.x * p.w.x + p.w.y * p.w.y;
                    if (wMagSq < 1e-20) continue;

                    const gamma = relOn ? Math.sqrt(1 + wMagSq) : 1;
                    const qSq = p.charge * p.charge;
                    const tau = 2 / 3 * qSq * p.invMass;

                    // Term 1: jerk — τ·dF/dt / γ³
                    // Analytical jerk for gravity + Coulomb (from pairForce)
                    let jerkX = p.jerk.x, jerkY = p.jerk.y;

                    // 3-point backward difference for residual (non-1/r²) forces
                    const otherFx = p.force.x - p.forceGravity.x - p.forceCoulomb.x;
                    const otherFy = p.force.y - p.forceGravity.y - p.forceCoulomb.y;
                    if (p._otherCount >= 2) {
                        // O(dt²) 3-point backward: variable-step Lagrange derivative at t₂
                        const h1 = p._otherDt0, h2 = p._otherDt1;
                        const h1h2 = h1 * h2, hSum = h1 + h2;
                        const c0 = h2 / (h1 * hSum);
                        const c1 = -hSum / h1h2;
                        const c2 = (h1 + 2 * h2) / (h2 * hSum);
                        jerkX += c0 * p._otherFx0 + c1 * p._otherFx1 + c2 * otherFx;
                        jerkY += c0 * p._otherFy0 + c1 * p._otherFy1 + c2 * otherFy;
                    } else if (p._otherCount >= 1 && p._otherDt1 > 1e-20) {
                        // O(dt) 2-point backward fallback
                        const invDt = 1 / p._otherDt1;
                        jerkX += (otherFx - p._otherFx1) * invDt;
                        jerkY += (otherFy - p._otherFy1) * invDt;
                    }
                    // Shift history
                    p._otherFx0 = p._otherFx1; p._otherFy0 = p._otherFy1;
                    p._otherDt0 = p._otherDt1;
                    p._otherFx1 = otherFx; p._otherFy1 = otherFy;
                    p._otherDt1 = dtSub;
                    if (p._otherCount < 2) p._otherCount++;

                    let fRadX = tau * jerkX;
                    let fRadY = tau * jerkY;

                    if (relOn && gamma > 1) {
                        const invG3 = 1 / (gamma * gamma * gamma);
                        fRadX *= invG3;
                        fRadY *= invG3;

                        // Derive coordinate velocity from current proper velocity
                        const invGamma = 1 / gamma;
                        const vx = p.w.x * invGamma, vy = p.w.y * invGamma;
                        const fx = p.force.x, fy = p.force.y;
                        const fSq = fx * fx + fy * fy;
                        const vDotF = vx * fx + vy * fy;
                        const invM = p.invMass;
                        const g2 = gamma * gamma;

                        // Term 2: −τ·v·F²/(m·γ²)
                        const t2 = -tau * fSq * invM / g2;
                        fRadX += t2 * vx;
                        fRadY += t2 * vy;

                        // Term 3: +τ·F·(v·F)/(m·γ⁴)
                        const t3 = tau * vDotF * invM / (g2 * g2);
                        fRadX += t3 * fx;
                        fRadY += t3 * fy;
                    }

                    // Clamp 1: LL validity — |F_rad| ≤ LL_FORCE_CLAMP · |F_ext|
                    // The LL approximation requires radiation force << external force
                    const fRadMag = Math.sqrt(fRadX * fRadX + fRadY * fRadY);
                    const fExtMag = Math.sqrt(p.force.x * p.force.x + p.force.y * p.force.y);
                    const maxFRad = LL_FORCE_CLAMP * fExtMag;
                    if (fRadMag > maxFRad && fRadMag > 1e-20) {
                        const scale = maxFRad / fRadMag;
                        fRadX *= scale;
                        fRadY *= scale;
                    }

                    const keBefore = relOn ? (gamma - 1) * p.mass : 0.5 * p.mass * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);

                    p.w.x += fRadX * dtSub * p.invMass;
                    p.w.y += fRadY * dtSub * p.invMass;
                    p.forceRadiation.x = fRadX;
                    p.forceRadiation.y = fRadY;

                    if (isNaN(p.w.x) || isNaN(p.w.y)) {
                        p.w.x = 0; p.w.y = 0;
                    }

                    const wMagSqAfter = p.w.x * p.w.x + p.w.y * p.w.y;
                    const gammaAfter = relOn ? Math.sqrt(1 + wMagSqAfter) : 1;
                    const keAfter = relOn ? (gammaAfter - 1) * p.mass : 0.5 * p.mass * wMagSqAfter / (gammaAfter * gammaAfter);
                    const dE = Math.max(0, keBefore - keAfter);
                    this.sim.totalRadiated += dE;

                    if (dE > 0) {
                        const ax = p.force.x * p.invMass, ay = p.force.y * p.invMass;
                        const accelAngle = Math.atan2(ay, ax);
                        const radAngle = accelAngle + Math.PI;
                        this.sim.totalRadiatedPx += dE * Math.cos(radAngle);
                        this.sim.totalRadiatedPy += dE * Math.sin(radAngle);

                        p._radAccum += dE;
                        if (p._radAccum >= MIN_MASS && this.sim.photons.length < MAX_PHOTONS) {
                            // sin²θ dipole pattern: peak emission ⊥ to acceleration
                            let theta, tries = 0;
                            do { theta = Math.random() * 6.283185307; }
                            while (Math.random() > Math.sin(theta) * Math.sin(theta) && ++tries < 20);
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

                }
            }

            // Hawking radiation: Kerr-Newman temperature
            if (this.blackHoleEnabled && this.sim) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (p.mass <= MIN_MASS) continue;
                    const M = p.mass;
                    const I = INERTIA_K * M * p.radiusSq;
                    const a = I * Math.abs(p.angVel) / M;
                    const Q = p.charge;
                    const disc = M * M - a * a - Q * Q;
                    let power;
                    if (disc > 1e-10) {
                        const rPlus = M + Math.sqrt(disc);
                        const kappa = Math.sqrt(disc) / (2 * M * rPlus);
                        const T = kappa / (2 * Math.PI);
                        const A = 4 * Math.PI * (rPlus * rPlus + a * a);
                        const sigma = Math.PI / 240;
                        power = sigma * T * T * T * T * A;
                    } else {
                        power = 0; // extremal: no radiation
                    }
                    const dE = power * dtSub;
                    if (dE <= 0) continue;
                    p.mass -= dE;
                    p.invMass = 1 / p.mass;
                    // Update Kerr-Newman radius
                    const newDisc = p.mass * p.mass - a * a - Q * Q;
                    p.radius = newDisc > 0 ? p.mass + Math.sqrt(newDisc) : p.mass * 0.5;
                    p.radiusSq = p.radius * p.radius;
                    this.sim.totalRadiated += dE;

                    p._hawkAccum += dE;
                    if (p._hawkAccum >= MIN_MASS && this.sim.photons.length < MAX_PHOTONS) {
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

            // Cosmological expansion: Hubble flow + drag
            if (this.expansionEnabled) {
                const H = this.hubbleParam;
                const cx = this.domainW * 0.5, cy = this.domainH * 0.5;
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    p.pos.x += H * (p.pos.x - cx) * dtSub;
                    p.pos.y += H * (p.pos.y - cy) * dtSub;
                    const decay = 1 - H * dtSub;
                    p.w.x *= decay;
                    p.w.y *= decay;
                }
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
                compute1PNPairwise(particles, SOFTENING_SQ, this.periodic, this.domainW, this.domainH, this.domainW * 0.5, this.domainH * 0.5, this._topologyConst, this.gravitomagEnabled, this.magneticEnabled);
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    const halfDtOverM = halfDt * p.invMass;
                    p.w.x += (p.force1PN.x - p._f1pnOld.x) * halfDtOverM;
                    p.w.y += (p.force1PN.y - p._f1pnOld.y) * halfDtOverM;
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
                    // Query quadtree for nearby particles (reuses pooled array)
                    const candidates = this.pool.queryReuse(root,
                        ph.pos.x, ph.pos.y, SOFTENING, SOFTENING);
                    for (let ci = 0; ci < candidates.length; ci++) {
                        const target = candidates[ci];
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

        // Gravitational wave radiation (quadrupole formula)
        if (this.gwRadiationEnabled && n >= 2 && this.sim) {
            let Ixx = 0, Ixy = 0, Iyy = 0;
            let Qxx = 0, Qxy = 0, Qyy = 0;
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                Ixx += p.mass * p.pos.x * p.pos.x;
                Ixy += p.mass * p.pos.x * p.pos.y;
                Iyy += p.mass * p.pos.y * p.pos.y;
                Qxx += p.charge * p.pos.x * p.pos.x;
                Qxy += p.charge * p.pos.x * p.pos.y;
                Qyy += p.charge * p.pos.y * p.pos.y;
            }
            this._quadHistory.push({ Ixx, Ixy, Iyy, Qxx, Qxy, Qyy, t: this.simTime });
            if (this._quadHistory.length > 5) this._quadHistory.shift();

            if (this._quadHistory.length >= 4) {
                const h = this._quadHistory;
                const n3 = h.length - 1;
                const dt1 = h[n3].t - h[n3 - 1].t;
                const dt2 = h[n3 - 1].t - h[n3 - 2].t;
                const dt3 = h[n3 - 2].t - h[n3 - 3].t;
                if (dt1 > 1e-15 && dt2 > 1e-15 && dt3 > 1e-15) {
                    const halfDtSum = (dt1 + dt2) * 0.5;
                    // Mass quadrupole 3rd derivatives
                    const d2Ixx_a = (h[n3].Ixx - 2 * h[n3 - 1].Ixx + h[n3 - 2].Ixx) / (dt1 * dt2);
                    const d2Ixx_b = (h[n3 - 1].Ixx - 2 * h[n3 - 2].Ixx + h[n3 - 3].Ixx) / (dt2 * dt3);
                    const d3Ixx = (d2Ixx_a - d2Ixx_b) / halfDtSum;
                    const d2Ixy_a = (h[n3].Ixy - 2 * h[n3 - 1].Ixy + h[n3 - 2].Ixy) / (dt1 * dt2);
                    const d2Ixy_b = (h[n3 - 1].Ixy - 2 * h[n3 - 2].Ixy + h[n3 - 3].Ixy) / (dt2 * dt3);
                    const d3Ixy = (d2Ixy_a - d2Ixy_b) / halfDtSum;
                    const d2Iyy_a = (h[n3].Iyy - 2 * h[n3 - 1].Iyy + h[n3 - 2].Iyy) / (dt1 * dt2);
                    const d2Iyy_b = (h[n3 - 1].Iyy - 2 * h[n3 - 2].Iyy + h[n3 - 3].Iyy) / (dt2 * dt3);
                    const d3Iyy = (d2Iyy_a - d2Iyy_b) / halfDtSum;

                    // GW power: P = (1/5)·(d³I_ij/dt³)²
                    let gwPower = 0.2 * (d3Ixx * d3Ixx + 2 * d3Ixy * d3Ixy + d3Iyy * d3Iyy);

                    // EM quadrupole: P_em = (1/180)·(d³Q_ij/dt³)²
                    if (this.radiationEnabled && this.coulombEnabled) {
                        const d2Qxx_a = (h[n3].Qxx - 2 * h[n3 - 1].Qxx + h[n3 - 2].Qxx) / (dt1 * dt2);
                        const d2Qxx_b = (h[n3 - 1].Qxx - 2 * h[n3 - 2].Qxx + h[n3 - 3].Qxx) / (dt2 * dt3);
                        const d3Qxx = (d2Qxx_a - d2Qxx_b) / halfDtSum;
                        const d2Qxy_a = (h[n3].Qxy - 2 * h[n3 - 1].Qxy + h[n3 - 2].Qxy) / (dt1 * dt2);
                        const d2Qxy_b = (h[n3 - 1].Qxy - 2 * h[n3 - 2].Qxy + h[n3 - 3].Qxy) / (dt2 * dt3);
                        const d3Qxy = (d2Qxy_a - d2Qxy_b) / halfDtSum;
                        const d2Qyy_a = (h[n3].Qyy - 2 * h[n3 - 1].Qyy + h[n3 - 2].Qyy) / (dt1 * dt2);
                        const d2Qyy_b = (h[n3 - 1].Qyy - 2 * h[n3 - 2].Qyy + h[n3 - 3].Qyy) / (dt2 * dt3);
                        const d3Qyy = (d2Qyy_a - d2Qyy_b) / halfDtSum;
                        const emQuadPower = (1 / 180) * (d3Qxx * d3Qxx + 2 * d3Qxy * d3Qxy + d3Qyy * d3Qyy);
                        gwPower += emQuadPower;
                    }

                    if (gwPower > 0) {
                        const dE = gwPower * dt;
                        this.sim.totalRadiated += dE;
                        this._gwAccum += dE;

                        // Apply orbital decay: radial kick toward COM
                        if (dE > 1e-10) {
                            let comX = 0, comY = 0, totalM = 0;
                            for (let i = 0; i < n; i++) {
                                comX += particles[i].mass * particles[i].pos.x;
                                comY += particles[i].mass * particles[i].pos.y;
                                totalM += particles[i].mass;
                            }
                            comX /= totalM; comY /= totalM;
                            for (let i = 0; i < n; i++) {
                                const p = particles[i];
                                const dx = comX - p.pos.x, dy = comY - p.pos.y;
                                const r = Math.sqrt(dx * dx + dy * dy);
                                if (r > 1e-10) {
                                    const kick = dE * p.mass / (totalM * r) * dt;
                                    p.w.x += kick * dx / r;
                                    p.w.y += kick * dy / r;
                                }
                            }
                        }

                        // Emit graviton when accumulated energy exceeds threshold
                        if (this._gwAccum >= MIN_MASS && this.sim.photons.length < MAX_PHOTONS) {
                            const angle = Math.random() * 6.283185307;
                            const cosA = Math.cos(angle), sinA = Math.sin(angle);
                            let gComX = 0, gComY = 0, gTotalM = 0;
                            for (let i = 0; i < n; i++) {
                                gComX += particles[i].mass * particles[i].pos.x;
                                gComY += particles[i].mass * particles[i].pos.y;
                                gTotalM += particles[i].mass;
                            }
                            gComX /= gTotalM; gComY /= gTotalM;
                            const gph = new Photon(gComX + cosA * 3, gComY + sinA * 3, cosA, sinA, this._gwAccum, -1);
                            gph.type = 'gw';
                            this.sim.photons.push(gph);
                            this.sim.totalRadiatedPx += this._gwAccum * cosA;
                            this.sim.totalRadiatedPy += this._gwAccum * sinA;
                            this._gwAccum = 0;
                        }
                    }
                }
            }
        }

        // PE once per frame, reusing last substep's tree
        this._lastRoot = lastRoot;
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
                const mu = MAG_MOMENT_K * p.charge * p.angVel * p.radiusSq;
                p.torqueSpinOrbit += -mu * (p.vel.x * p.dBzdx + p.vel.y * p.dBzdy);
                p.forceSpinCurv.x += mu * p.dBzdx;
                p.forceSpinCurv.y += mu * p.dBzdy;
            }
        }
        if (hasGM && this.spinOrbitEnabled) {
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                if (Math.abs(p.angVel) < 1e-10) continue;
                const L = INERTIA_K * p.mass * p.angVel * p.radiusSq;
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
                p.forceRadiation.x = p._radDisplayX;
                p.forceRadiation.y = p._radDisplayY;
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

    checkTidalBreakup(particles, lastRoot) {
        if (!this.tidalEnabled) return { fragments: [], transfers: [] };
        const fragments = [];
        const transfers = [];
        const _periodic = this.periodic;
        const _halfDomW = this.domainW * 0.5, _halfDomH = this.domainH * 0.5;
        const _domW = this.domainW, _domH = this.domainH;
        const _topo = this._topologyConst;
        const useTree = this.barnesHutEnabled && lastRoot >= 0;
        const tidalSearchR = Math.max(_domW, _domH) * 0.5;

        for (let pi = 0; pi < particles.length; pi++) {
            const p = particles[pi];
            if (p.mass < MIN_MASS * FRAGMENT_COUNT) continue;

            const rSq = p.radiusSq;
            const selfGravity = p.mass / rSq;
            const centrifugal = p.angVel * p.angVel * p.radius;
            const coulombSelf = (p.charge * p.charge) / (4 * rSq);

            if (centrifugal + coulombSelf > selfGravity) {
                fragments.push(p);
                continue;
            }

            let maxTidal = 0;
            let strongestOther = null;
            let strongestDx = 0, strongestDy = 0, strongestDist = 0;

            const _checkNeighbor = (other, dx, dy) => {
                const distSq = dx * dx + dy * dy + SOFTENING_SQ;
                const invDistSq = 1 / distSq;
                const tidalAccel = TIDAL_STRENGTH * other.mass * p.radius * Math.sqrt(invDistSq) * invDistSq;
                if (tidalAccel > maxTidal) {
                    maxTidal = tidalAccel;
                    strongestOther = other;
                    strongestDx = dx; strongestDy = dy;
                    strongestDist = Math.sqrt(distSq - SOFTENING_SQ);
                }
            };

            if (useTree) {
                const candidates = this.pool.queryReuse(lastRoot,
                    p.pos.x, p.pos.y, tidalSearchR, tidalSearchR);
                for (let ci = 0; ci < candidates.length; ci++) {
                    const other = candidates[ci];
                    if (other === p || (other.isGhost && other.original === p)) continue;
                    let dx = other.pos.x - p.pos.x, dy = other.pos.y - p.pos.y;
                    if (_periodic) {
                        minImage(p.pos.x, p.pos.y, other.pos.x, other.pos.y, _topo, _domW, _domH, _halfDomW, _halfDomH, _tidalMiOut);
                        dx = _tidalMiOut.x; dy = _tidalMiOut.y;
                    }
                    _checkNeighbor(other, dx, dy);
                }
            } else {
                for (let oi = 0; oi < particles.length; oi++) {
                    const other = particles[oi];
                    if (other === p) continue;
                    let dx = other.pos.x - p.pos.x, dy = other.pos.y - p.pos.y;
                    if (_periodic) {
                        minImage(p.pos.x, p.pos.y, other.pos.x, other.pos.y, _topo, _domW, _domH, _halfDomW, _halfDomH, _tidalMiOut);
                        dx = _tidalMiOut.x; dy = _tidalMiOut.y;
                    }
                    _checkNeighbor(other, dx, dy);
                }
            }

            if (maxTidal + centrifugal + coulombSelf > selfGravity) {
                fragments.push(p);
            } else if (strongestOther && strongestDist > 1e-10 && p.mass > ROCHE_MIN_PACKET * 4) {
                // Roche lobe overflow: Eggleton formula r_Roche ≈ 0.462·d·(m/(m+M))^(1/3)
                const d = strongestDist;
                const q = p.mass / (p.mass + strongestOther.mass);
                const rRoche = 0.462 * d * Math.cbrt(q);
                if (p.radius > rRoche * ROCHE_THRESHOLD) {
                    const l1Mag = Math.sqrt(strongestDx * strongestDx + strongestDy * strongestDy);
                    if (l1Mag > 1e-10) {
                        const l1x = strongestDx / l1Mag, l1y = strongestDy / l1Mag;
                        const overflow = p.radius / rRoche - ROCHE_THRESHOLD;
                        const dM = Math.min(overflow * ROCHE_TRANSFER_RATE * p.mass, p.mass * 0.1);
                        if (dM >= ROCHE_MIN_PACKET) {
                            transfers.push({
                                source: p,
                                mass: dM,
                                charge: dM * p.charge / p.mass,
                                spawnX: p.pos.x + l1x * p.radius * 1.2,
                                spawnY: p.pos.y + l1y * p.radius * 1.2,
                                vx: p.vel.x + (-l1y) * Math.sqrt(strongestOther.mass / d) * 0.5,
                                vy: p.vel.y + l1x * Math.sqrt(strongestOther.mass / d) * 0.5,
                            });
                        }
                    }
                }
            }
        }

        return { fragments, transfers };
    }
}
