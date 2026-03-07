// ─── Boris Integrator ───
// Adaptive-substep Boris integrator. Separates E-like (position-dependent) and
// B-like (velocity-dependent) forces for exact |v|-preserving rotation.

import QuadTreePool from './quadtree.js';
import { PI, TWO_PI, SOFTENING, BH_SOFTENING, DESPAWN_MARGIN, INERTIA_K, MAG_MOMENT_K, MAX_SUBSTEPS, MIN_MASS, MAX_PHOTONS, LL_FORCE_CLAMP, TIDAL_STRENGTH, SPAWN_COUNT, SOFTENING_SQ, BH_SOFTENING_SQ, QUADTREE_CAPACITY, BH_THETA, HISTORY_SIZE, HISTORY_STRIDE, DEFAULT_YUKAWA_MU, DEFAULT_AXION_MASS, ROCHE_THRESHOLD, ROCHE_TRANSFER_RATE, DEFAULT_HUBBLE, EPSILON, EPSILON_SQ, MAX_REJECTION_SAMPLES, QUADRUPOLE_POWER_CLAMP, ABERRATION_THRESHOLD, spawnOffset, kerrNewmanRadius, PION_LIFETIME, MAX_PIONS, YUKAWA_G2, BOSON_ABSORB_FRACTION, BOSON_MIN_AGE } from './config.js';
import Photon from './photon.js';
import Pion from './pion.js';
import { angwToAngVel } from './relativity.js';

import { resetForces, computeAllForces, compute1PNPairwise } from './forces.js';
import { handleCollisions } from './collisions.js';
import { computePE } from './potential.js';
import { TORUS, KLEIN, RP2, minImage, wrapPosition } from './topology.js';

// Reused by disintegration to avoid per-call allocation
const _disintMiOut = { x: 0, y: 0 };

/**
 * Rejection-sample a quadrupole emission angle.
 * Power ∝ (Axx·cos2φ + Axy·sin2φ)² where Axx, Axy are the relevant
 * d³ tensor components. Peak amplitude = sqrt(Axx²+Axy²).
 */
function _quadSample(Axx, Axy) {
    const peak2 = Axx * Axx + Axy * Axy;
    if (peak2 < EPSILON_SQ) return Math.random() * TWO_PI;
    for (let tries = 0; tries < MAX_REJECTION_SAMPLES; tries++) {
        const phi = Math.random() * TWO_PI;
        const c2 = Math.cos(2 * phi), s2 = Math.sin(2 * phi);
        const h = Axx * c2 + Axy * s2;
        if (Math.random() * peak2 <= h * h) return phi;
    }
    return Math.random() * TWO_PI;
}

export default class Physics {
    constructor() {
        this.boundary = { x: 0, y: 0, w: 0, h: 0 };
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
        this.disintegrationEnabled = false;
        this.spinOrbitEnabled = true;
        this.onePNEnabled = true;

        this.yukawaEnabled = false;
        this.yukawaMu = DEFAULT_YUKAWA_MU;

        this.axionEnabled = false;
        this.axionMass = DEFAULT_AXION_MASS;
        this.expansionEnabled = false;
        this.hubbleParam = DEFAULT_HUBBLE;

        this.higgsEnabled = false;

        // External background fields (uniform, independent of force toggles)
        this.extGravity = 0;          // field strength |g|
        this.extGravityAngle = PI * 0.5;  // direction in radians (default: down)
        this.extElectric = 0;         // field strength |E|
        this.extElectricAngle = 0;    // direction in radians (default: right)
        this.extBz = 0;              // uniform magnetic field (z-component)

        // Cached direction vectors (updated once per frame in _cacheExternalFields)
        this._extGx = 0; this._extGy = 0;
        this._extEx = 0; this._extEy = 0;

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
            yukawaEnabled: false,
            yukawaMu: DEFAULT_YUKAWA_MU,
            axionEnabled: false,
            softeningSq: SOFTENING_SQ,
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
        this._toggles.yukawaEnabled = this.yukawaEnabled;
        this._toggles.yukawaMu = this.yukawaMu;
        this._toggles.axionEnabled = this.axionEnabled;
        this._toggles.softeningSq = this.blackHoleEnabled ? BH_SOFTENING_SQ : SOFTENING_SQ;
        this._toggles.higgsEnabled = this.higgsEnabled;
    }

    /** Move a removed particle to deadParticles for signal delay fade-out. */
    _retireParticle(p) {
        if (!this.relativityEnabled || !this.sim) return;
        if (p.mass > 0) p._deathMass = p.mass;
        p._deathAngVel = p.angVel;
        p.deathTime = this.simTime;
        // Record final history snapshot so buffer extends to death time
        p._initHistory();
        const h = p.histHead;
        p.histX[h] = p.pos.x;
        p.histY[h] = p.pos.y;
        p.histVx[h] = p.vel.x;
        p.histVy[h] = p.vel.y;
        p.histTime[h] = this.simTime;
        p.histHead = (h + 1) % HISTORY_SIZE;
        if (p.histCount < HISTORY_SIZE) p.histCount++;
        // Cache dipole moments at death (won't be recached in computeAllForces)
        p.magMoment = MAG_MOMENT_K * p.charge * p.angVel * p.radiusSq;
        p.angMomentum = INERTIA_K * p._deathMass * p.angVel * p.radiusSq;
        this.sim.deadParticles.push(p);
    }

    /** Pool-allocate a ghost at (sx, sy) mirroring p. Flips for non-orientable topologies. */
    _addGhost(p, sx, sy, flipVx = false, flipVy = false) {
        let g;
        if (this._ghostCount < this._ghostPool.length) {
            g = this._ghostPool[this._ghostCount];
        } else {
            g = { pos: { x: 0, y: 0 }, vel: { x: 0, y: 0 }, w: { x: 0, y: 0, magSq() { return this.x * this.x + this.y * this.y; } }, mass: 0, charge: 0, angVel: 0, angw: 0, magMoment: 0, angMomentum: 0, radius: 0, radiusSq: 0, invMass: 0, id: -1, isGhost: true, original: null };
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
        g.magMoment = MAG_MOMENT_K * g.charge * g.angVel * g.radiusSq;
        g.angMomentum = INERTIA_K * g.mass * g.angVel * g.radiusSq;
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

    /** Cache external field direction vectors (call once per frame, not per substep). */
    _cacheExternalFields() {
        const g = this.extGravity;
        const E = this.extElectric;
        this._extGx = g * Math.cos(this.extGravityAngle);
        this._extGy = g * Math.sin(this.extGravityAngle);
        this._extEx = E * Math.cos(this.extElectricAngle);
        this._extEy = E * Math.sin(this.extElectricAngle);
    }

    /** Apply uniform external fields (g, E, B) to all particles. */
    _applyExternalFields(particles) {
        const g = this.extGravity;
        const E = this.extElectric;
        const Bext = this.extBz;
        if (g === 0 && E === 0 && Bext === 0) return;
        const gx = this._extGx;
        const gy = this._extGy;
        const ex = this._extEx;
        const ey = this._extEy;
        for (let i = 0, n = particles.length; i < n; i++) {
            const p = particles[i];
            if (g !== 0) {
                const fx = p.mass * gx, fy = p.mass * gy;
                p.force.x += fx;
                p.force.y += fy;
                p.forceExternal.x += fx;
                p.forceExternal.y += fy;
            }
            if (E !== 0) {
                const fx = p.charge * ex, fy = p.charge * ey;
                p.force.x += fx;
                p.force.y += fy;
                p.forceExternal.x += fx;
                p.forceExternal.y += fy;
            }
            if (Bext !== 0) {
                p.Bz += Bext;
            }
        }
    }

    /** Sync per-particle axMod: interpolate from axion field or default to 1. */
    _syncAxionField(particles, width, height) {
        if (this.axionEnabled && this.sim && this.sim.axionField) {
            this.sim.axionField.interpolateAxMod(particles, width, height);
        } else {
            for (let i = 0, n = particles.length; i < n; i++) particles[i].axMod = 1;
        }
    }

    /** Apply Hertz wall repulsion for boundary mode 'bounce'. */
    _applyBoundaryForces(particles, width, height, offX, offY) {
        const friction = this.bounceFriction;
        const left = offX, top = offY;
        const right = offX + width, bottom = offY + height;
        for (let i = 0, n = particles.length; i < n; i++) {
            const p = particles[i];
            const r = p.radius;
            // Left wall
            let delta = r - (p.pos.x - left);
            if (delta > 0) {
                const Fn = delta * Math.sqrt(delta);
                p.force.x += Fn;
                p.forceExternal.x += Fn;
                if (friction > 0) {
                    const vt = p.vel.y + p.angVel * r;
                    const Ft = -friction * Fn * Math.max(-1, Math.min(1, vt * 10));
                    p.force.y += Ft;
                    p._tidalTorque += r * Ft;
                }
            }
            // Right wall
            delta = r - (right - p.pos.x);
            if (delta > 0) {
                const Fn = delta * Math.sqrt(delta);
                p.force.x -= Fn;
                p.forceExternal.x -= Fn;
                if (friction > 0) {
                    const vt = -p.vel.y - p.angVel * r;
                    const Ft = -friction * Fn * Math.max(-1, Math.min(1, vt * 10));
                    p.force.y -= Ft;
                    p._tidalTorque -= r * Ft;
                }
            }
            // Top wall
            delta = r - (p.pos.y - top);
            if (delta > 0) {
                const Fn = delta * Math.sqrt(delta);
                p.force.y += Fn;
                p.forceExternal.y += Fn;
                if (friction > 0) {
                    const vt = -p.vel.x + p.angVel * r;
                    const Ft = -friction * Fn * Math.max(-1, Math.min(1, vt * 10));
                    p.force.x -= Ft;
                    p._tidalTorque += r * Ft;
                }
            }
            // Bottom wall
            delta = r - (bottom - p.pos.y);
            if (delta > 0) {
                const Fn = delta * Math.sqrt(delta);
                p.force.y -= Fn;
                p.forceExternal.y -= Fn;
                if (friction > 0) {
                    const vt = p.vel.x - p.angVel * r;
                    const Ft = -friction * Fn * Math.max(-1, Math.min(1, vt * 10));
                    p.force.x += Ft;
                    p._tidalTorque -= r * Ft;
                }
            }
        }
    }

    /** Apply short-range repulsive contact force (Hertz model) when collisionMode is 'bounce'. */
    _applyRepulsion(particles, pool, root) {
        const friction = this.bounceFriction;
        const n = particles.length;
        const useTree = root >= 0;
        for (let i = 0; i < n; i++) {
            const p1 = particles[i];
            if (useTree) {
                const searchR = p1.radius * 2;
                const candidates = pool.queryReuse(root, p1.pos.x, p1.pos.y, searchR, searchR);
                for (let ci = 0; ci < candidates.length; ci++) {
                    const p2raw = candidates[ci];
                    const p2 = p2raw.isGhost ? p2raw.original : p2raw;
                    if (p1 === p2 || p1.id >= p2.id) continue;
                    this._repelPair(p1, p2raw.pos.x, p2raw.pos.y, p2, friction);
                }
            } else {
                for (let j = i + 1; j < n; j++) {
                    const p2 = particles[j];
                    this._repelPair(p1, p2.pos.x, p2.pos.y, p2, friction);
                }
            }
        }
    }

    /** Apply Hertz contact + friction between one pair. */
    _repelPair(p1, p2x, p2y, p2, friction) {
        const dx = p2x - p1.pos.x;
        const dy = p2y - p1.pos.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const minDist = p1.radius + p2.radius;
        const delta = minDist - dist;
        if (delta <= 0) return;
        // Hertz contact: F = delta^1.5
        const Fn = delta * Math.sqrt(delta);
        const safeDist = dist > EPSILON ? dist : EPSILON;
        const nx = dx / safeDist, ny = dy / safeDist;
        // Normal repulsion (Newton's 3rd law)
        p1.force.x -= nx * Fn;
        p1.force.y -= ny * Fn;
        p1.forceExternal.x -= nx * Fn;
        p1.forceExternal.y -= ny * Fn;
        p2.force.x += nx * Fn;
        p2.force.y += ny * Fn;
        p2.forceExternal.x += nx * Fn;
        p2.forceExternal.y += ny * Fn;
        // Tangential friction for torque transfer (accumulates into tidal torque slot)
        if (friction > 0) {
            const tx = -ny, ty = nx;
            const v1t = p1.vel.x * tx + p1.vel.y * ty + p1.angVel * p1.radius;
            const v2t = p2.vel.x * tx + p2.vel.y * ty - p2.angVel * p2.radius;
            const vRel = v1t - v2t;
            const Ft = -friction * Fn * Math.max(-1, Math.min(1, vRel * 10));
            p1.force.x += tx * Ft;
            p1.force.y += ty * Ft;
            p2.force.x -= tx * Ft;
            p2.force.y -= ty * Ft;
            // Torque: r × F_t (accumulated for the torque step)
            p1._tidalTorque += p1.radius * Ft;
            p2._tidalTorque -= p2.radius * Ft;
        }
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
        this._cacheExternalFields();
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
            this._syncAxionField(particles, width, height);
            const initRoot = this.barnesHutEnabled
                ? this._buildTree(particles)
                : -1;
            computeAllForces(particles, toggles, this.pool, initRoot, this.barnesHutEnabled, relOn, this.simTime, this.periodic, this.domainW, this.domainH, this._topologyConst, this.sim && this.sim.deadParticles);
            this._applyExternalFields(particles);
            if (this.higgsEnabled && this.sim && this.sim.higgsField) {
                this.sim.higgsField.applyForces(particles, width, height);
            }
            if (this.axionEnabled && this.sim && this.sim.axionField) {
                this.sim.axionField.applyForces(particles, width, height);
            }
            if (collisionMode === 'bounce') this._applyRepulsion(particles, this.pool, initRoot);
            if (boundaryMode === 'bounce') this._applyBoundaryForces(particles, width, height, offX, offY);
            this._forcesInit = true;
        }

        const hasGrav = this.gravityEnabled;
        const hasMagnetic = this.magneticEnabled;
        const hasGM = this.gravitomagEnabled;
        const hasExtBz = this.extBz !== 0;

        // Bz/Bgz for cyclotron frequency estimation persist from the previous
        // frame's last substep (or from the bootstrap above). No preliminary
        // force pass needed.

        // Clear reconstructed radiation display from previous frame so it
        // doesn't leak into the first substep's snapshot (forceRadiation is
        // only set by Larmor for charged particles — neutral ones would carry
        // stale values from reconstruction, accumulating frame over frame).
        for (let i = 0; i < n; i++) {
            particles[i].forceRadiation.x = 0;
            particles[i].forceRadiation.y = 0;
        }

        // ─── Adaptive substepping ───
        // Re-evaluates dtSafe each iteration so substep size tracks changing fields.
        let dtRemain = dt;
        let totalSteps = 0;
        let lastRoot = -1;
        while (dtRemain > EPSILON && totalSteps < MAX_SUBSTEPS) {
            let maxAccelSq = 0;
            let maxCyclotron = 0;
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const fx = p.force.x, fy = p.force.y;
                const aSq = (fx * fx + fy * fy) * p.invMass * p.invMass;
                if (aSq > maxAccelSq) maxAccelSq = aSq;
                if (hasMagnetic || hasExtBz) {
                    const absBz = p.Bz > 0 ? p.Bz : -p.Bz;
                    if (absBz > 0) {
                        const absQ = p.charge > 0 ? p.charge : -p.charge;
                        const wc = absQ * absBz * p.invMass;
                        if (wc > maxCyclotron) maxCyclotron = wc;
                    }
                }
                if (hasGM) {
                    const absBgz = p.Bgz > 0 ? p.Bgz : -p.Bgz;
                    if (absBgz > 0) {
                        const wc = 4 * absBgz;
                        if (wc > maxCyclotron) maxCyclotron = wc;
                    }
                }
            }
            const aMax = Math.sqrt(maxAccelSq);
            let dtSafe = aMax > EPSILON_SQ ? Math.sqrt((this.blackHoleEnabled ? BH_SOFTENING : SOFTENING) / aMax) : dtRemain;
            if (maxCyclotron > 0) {
                const dtCyclotron = 0.7853981633974483 / maxCyclotron; // (2π/8) = π/4
                if (dtCyclotron < dtSafe) dtSafe = dtCyclotron;
            }
            const budgetLeft = MAX_SUBSTEPS - totalSteps;
            const stepsNeeded = Math.min(Math.ceil(dtRemain / dtSafe), budgetLeft);
            const dtSub = dtRemain / stepsNeeded;

            totalSteps++;
            dtRemain -= dtSub;

            const has1PN = toggles.onePNEnabled;
            const halfDt = dtSub * 0.5;
            const needBoris = hasMagnetic || hasGM || hasExtBz;

            // Fused loop: save 1PN old, half-kick 1, Boris rotation, half-kick 2
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const invM = p.invMass;
                const halfDtOverM = halfDt * invM;

                // Save pre-drift 1PN forces
                if (has1PN) {
                    p._f1pnOld.x = p.force1PN.x;
                    p._f1pnOld.y = p.force1PN.y;
                }

                // Half-kick 1
                const fx = p.force.x, fy = p.force.y;
                p.w.x += fx * halfDtOverM;
                p.w.y += fy * halfDtOverM;

                // Boris rotation
                if (needBoris) {
                    let t = 0;
                    if (hasMagnetic || hasExtBz) t += (p.charge * 0.5 * invM) * p.Bz;
                    if (hasGM) t += 2 * p.Bgz;
                    if (t !== 0) {
                        const gamma = relOn ? Math.sqrt(1 + p.w.x * p.w.x + p.w.y * p.w.y) : 1;
                        t *= dtSub / gamma;
                        const s = 2 * t / (1 + t * t);
                        const wx = p.w.x, wy = p.w.y;
                        const wpx = wx + wy * t;
                        const wpy = wy - wx * t;
                        p.w.x = wx + wpy * s;
                        p.w.y = wy - wpx * s;
                    }
                }

                // Half-kick 2
                p.w.x += fx * halfDtOverM;
                p.w.y += fy * halfDtOverM;

                // NaN guard: catch bad state before it propagates to all particles
                if (p.w.x !== p.w.x || p.w.y !== p.w.y) { p.w.x = 0; p.w.y = 0; }
            }

            // Spin-orbit energy transfer + Stern-Gerlach/Mathisson-Papapetrou kicks (fused)
            if (this.spinOrbitEnabled && (hasMagnetic || hasGM)) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (Math.abs(p.angVel) < EPSILON) continue;
                    const IxOmega = p.angMomentum;
                    if (Math.abs(IxOmega) < EPSILON) continue;
                    const dtOverM = dtSub * p.invMass;

                    if (hasMagnetic && Math.abs(p.charge) > EPSILON) {
                        const mu = p.magMoment;
                        // Energy transfer
                        p.angw -= mu * (p.vel.x * p.dBzdx + p.vel.y * p.dBzdy) * dtSub / IxOmega;
                        // Stern-Gerlach translational kick
                        p.w.x += mu * p.dBzdx * dtOverM;
                        p.w.y += mu * p.dBzdy * dtOverM;
                    }
                    if (hasGM) {
                        const L = p.angMomentum;
                        // Energy transfer
                        p.angw -= L * (p.vel.x * p.dBgzdx + p.vel.y * p.dBgzdy) * dtSub / IxOmega;
                        // Mathisson-Papapetrou translational kick
                        p.w.x -= L * p.dBgzdx * dtOverM;
                        p.w.y -= L * p.dBgzdy * dtOverM;
                    }
                    if (p.angw !== p.angw) p.angw = 0; // NaN guard
                    const sr = p.angw * p.radius;
                    p.angVel = relOn ? p.angw / Math.sqrt(1 + sr * sr) : p.angw;
                }
            }

            // Frame-dragging torque + tidal locking + bounce contact torque (fused)
            if ((hasGM && relOn) || hasGrav || collisionMode === 'bounce') {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    let torque = 0;
                    if (hasGM && relOn) torque += p._frameDragTorque;
                    if (hasGrav) torque += p._tidalTorque;
                    if (torque === 0) continue;
                    const I = INERTIA_K * p.mass * p.radiusSq;
                    p.angw += torque * dtSub / I;
                    if (p.angw !== p.angw) p.angw = 0; // NaN guard
                    const sr = p.angw * p.radius;
                    p.angVel = relOn ? p.angw / Math.sqrt(1 + sr * sr) : p.angw;
                }
            }

            // Landau-Lifshitz radiation reaction (full 1/c² terms)
            // F_rad = τ·[dF/dt / γ³ − v·F²/(m·γ²) + F·(v·F)/(m·γ⁴)]
            if (this.radiationEnabled && this.coulombEnabled && this.sim) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (Math.abs(p.charge) < EPSILON) continue;

                    const wMagSq = p.w.x * p.w.x + p.w.y * p.w.y;
                    if (wMagSq < EPSILON_SQ) continue;

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
                    } else if (p._otherCount >= 1 && p._otherDt1 > EPSILON_SQ) {
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
                    if (fRadMag > maxFRad && fRadMag > EPSILON_SQ) {
                        const scale = maxFRad / fRadMag;
                        fRadX *= scale;
                        fRadY *= scale;
                    }

                    const wSqBefore = p.w.x * p.w.x + p.w.y * p.w.y;
                    const keBefore = wSqBefore / (gamma + 1) * p.mass;

                    p.w.x += fRadX * dtSub * p.invMass;
                    p.w.y += fRadY * dtSub * p.invMass;
                    p.forceRadiation.x = fRadX;
                    p.forceRadiation.y = fRadY;

                    if (isNaN(p.w.x) || isNaN(p.w.y)) {
                        p.w.x = 0; p.w.y = 0;
                    }

                    const wMagSqAfter = p.w.x * p.w.x + p.w.y * p.w.y;
                    const gammaAfter = Math.sqrt(1 + wMagSqAfter);
                    const keAfter = wMagSqAfter / (gammaAfter + 1) * p.mass;
                    const dE = Math.max(0, keBefore - keAfter);
                    this.sim.totalRadiated += dE;

                    if (dE > 0) {
                        const ax = p.force.x * p.invMass, ay = p.force.y * p.invMass;
                        const accelAngle = Math.atan2(ay, ax);
                        const radAngle = accelAngle + PI;
                        this.sim.totalRadiatedPx += dE * Math.cos(radAngle);
                        this.sim.totalRadiatedPy += dE * Math.sin(radAngle);

                        p._radAccum += dE;
                        if (p._radAccum >= MIN_MASS && this.sim.photons.length < MAX_PHOTONS) {
                            // sin²θ dipole pattern: peak emission ⊥ to acceleration
                            let theta, tries = 0;
                            do { theta = Math.random() * TWO_PI; }
                            while (Math.random() > Math.sin(theta) * Math.sin(theta) && ++tries < MAX_REJECTION_SAMPLES);
                            let emitAngle = accelAngle + theta;

                            // Relativistic aberration: beam forward at high γ
                            if (gamma > ABERRATION_THRESHOLD) {
                                const beta = Math.sqrt(1 - 1 / (gamma * gamma));
                                const velAngle = Math.atan2(p.vel.y, p.vel.x);
                                const delta = emitAngle - velAngle;
                                const sinD = Math.sin(delta), cosD = Math.cos(delta);
                                const denom = 1 + beta * cosD;
                                emitAngle = velAngle + Math.atan2(sinD / (gamma * denom), (cosD + beta) / denom);
                            }

                            const cosA = Math.cos(emitAngle), sinA = Math.sin(emitAngle);
                            const pOff = spawnOffset(p.radius);
                            this.sim.photons.push(new Photon(
                                p.pos.x + cosA * pOff,
                                p.pos.y + sinA * pOff,
                                cosA, sinA,
                                p._radAccum, p.id
                            ));
                            p._radAccum = 0;
                        }
                    }

                }
            }

            // Hawking radiation: Kerr-Newman temperature
            if (this.blackHoleEnabled && this.radiationEnabled && this.sim) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    if (p.mass <= MIN_MASS) continue;
                    const M = p.mass;
                    const a = INERTIA_K * Math.cbrt(M) ** 2 * Math.abs(p.angVel);
                    const Q = p.charge;
                    const disc = M * M - a * a - Q * Q;
                    let power;
                    if (disc > EPSILON) {
                        const rPlus = M + Math.sqrt(disc);
                        const kappa = Math.sqrt(disc) / (rPlus * rPlus + a * a);
                        const T = kappa / TWO_PI;
                        const A = 4 * PI * (rPlus * rPlus + a * a);
                        const sigma = PI * PI / 60;
                        power = sigma * T * T * T * T * A;
                    } else {
                        power = 0; // extremal: no radiation
                    }
                    const dE = Math.min(power * dtSub, p.mass);
                    if (dE <= 0) continue;
                    p.mass -= dE;
                    p.invMass = 1 / p.mass;
                    // Update Kerr-Newman radius (use body r² = cbrt(mass)², not horizon radiusSq)
                    const bodyRSq = Math.cbrt(p.mass) ** 2;
                    p.radius = kerrNewmanRadius(p.mass, bodyRSq, p.angVel, p.charge);
                    p.radiusSq = p.radius * p.radius;
                    this.sim.totalRadiated += dE;

                    p.baseMass *= p.mass / (p.mass + dE);
                    p._hawkAccum += dE;
                    if (p._hawkAccum >= MIN_MASS && this.sim.photons.length < MAX_PHOTONS) {
                        const emitAngle = Math.random() * TWO_PI;
                        const cosA = Math.cos(emitAngle), sinA = Math.sin(emitAngle);
                        const hOff = spawnOffset(p.radius);
                        this.sim.photons.push(new Photon(
                            p.pos.x + cosA * hOff,
                            p.pos.y + sinA * hOff,
                            cosA, sinA,
                            p._hawkAccum, p.id
                        ));
                        this.sim.totalRadiatedPx += p._hawkAccum * cosA;
                        this.sim.totalRadiatedPy += p._hawkAccum * sinA;
                        p._hawkAccum = 0;
                    }
                }
            }

            // Pion emission from Yukawa interactions (scalar Larmor radiation)
            if (this.yukawaEnabled && this.radiationEnabled && this.sim) {
                const pions = this.sim.pions;
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    const fYukSq = p.forceYukawa.x * p.forceYukawa.x + p.forceYukawa.y * p.forceYukawa.y;
                    if (fYukSq < EPSILON_SQ) continue;
                    // Scalar Larmor: P = g²m²a²/3 = g²F²/3
                    // 1/3 angular factor for spin-0 (cf. 2/3 for spin-1 EM Larmor)
                    // Scalar charge Q = g·m (Yukawa couples ∝ m), so Q²a² = g²m²(F/m)² = g²F²
                    const dE = YUKAWA_G2 / 3 * fYukSq * dtSub;
                    p._yukawaRadAccum += dE;
                    if (p._yukawaRadAccum >= MIN_MASS && pions.length < MAX_PIONS) {
                        const pionMass = this.yukawaMu;
                        const ke = p._yukawaRadAccum - pionMass;
                        if (ke > 0) {
                            const angle = Math.atan2(p.forceYukawa.y, p.forceYukawa.x);
                            const speed = Math.sqrt(ke * (ke + 2 * pionMass)) / (ke + pionMass);
                            const gamma = 1 / Math.sqrt(1 - speed * speed);
                            const wx = gamma * speed * Math.cos(angle);
                            const wy = gamma * speed * Math.sin(angle);
                            const charge = Math.abs(p.charge) < EPSILON ? 0 : (Math.random() < 0.5 ? 0 : (Math.random() < 0.5 ? 1 : -1));
                            p.charge -= charge;
                            if (charge !== 0) p.updateColor();
                            const offset = spawnOffset(p.radius);
                            pions.push(new Pion(
                                p.pos.x + Math.cos(angle) * offset,
                                p.pos.y + Math.sin(angle) * offset,
                                wx, wy, pionMass, charge, p._yukawaRadAccum, p.id
                            ));
                            this.sim.totalRadiated += p._yukawaRadAccum;
                            this.sim.totalRadiatedPx += pionMass * wx;
                            this.sim.totalRadiatedPy += pionMass * wy;
                            // Radiation reaction: subtract emitted energy from particle KE
                            const wSq = p.w.x * p.w.x + p.w.y * p.w.y;
                            if (wSq > EPSILON_SQ) {
                                const pKE = relOn
                                    ? (Math.sqrt(1 + wSq) - 1) * p.mass
                                    : 0.5 * p.mass * wSq;
                                if (pKE > p._yukawaRadAccum) {
                                    const scale = Math.sqrt(1 - p._yukawaRadAccum / pKE);
                                    p.w.x *= scale;
                                    p.w.y *= scale;
                                }
                            }
                            p._yukawaRadAccum = 0;
                        }
                    }
                }
            }

            // Step 4: Derive coordinate velocity from w, then drift positions
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const invG = relOn ? 1 / Math.sqrt(1 + p.w.x * p.w.x + p.w.y * p.w.y) : 1;
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
                // Resync vel from w only if expansion modified w after drift
                if (relOn && this.expansionEnabled) {
                    for (let i = 0; i < n; i++) {
                        const p = particles[i];
                        const invG = 1 / Math.sqrt(1 + p.w.x * p.w.x + p.w.y * p.w.y);
                        p.vel.x = p.w.x * invG;
                        p.vel.y = p.w.y * invG;
                    }
                }
                compute1PNPairwise(particles, toggles.softeningSq, this.periodic, this.domainW, this.domainH, this.domainW * 0.5, this.domainH * 0.5, this._topologyConst, this.gravitomagEnabled, this.magneticEnabled, this.yukawaEnabled, this.yukawaMu);
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    const halfDtOverM = halfDt * p.invMass;
                    p.w.x += (p.force1PN.x - p._f1pnOld.x) * halfDtOverM;
                    p.w.y += (p.force1PN.y - p._f1pnOld.y) * halfDtOverM;
                    if (p.w.x !== p.w.x || p.w.y !== p.w.y) { p.w.x = 0; p.w.y = 0; }
                }
            }

            // Higgs field evolution + mass modulation
            if (this.higgsEnabled && this.sim && this.sim.higgsField) {
                this.sim.higgsField.update(dtSub, particles, boundaryMode, this._topologyConst, width, height);
                this.sim.higgsField.modulateMasses(particles, width, height, this.blackHoleEnabled);
            }

            // Axion field evolution (axMod interpolation deferred to step 7)
            if (this.axionEnabled && this.sim && this.sim.axionField) {
                this.sim.axionField.update(dtSub, particles, boundaryMode, this._topologyConst, width, height);
            }

            // Step 5: Rebuild quadtree at new positions
            const root = this._buildTree(particles);
            lastRoot = root;

            // Step 6: Collisions (bounce uses force-based Hertz repulsion; only merge goes here)
            if (collisionMode === 'merge') {
                const { annihilations, merges, removed } = handleCollisions(particles, this.pool, root, collisionMode, this.bounceFriction, this.relativityEnabled, this.periodic, this.domainW, this.domainH, this._topologyConst);
                n = particles.length;
                // Retire removed particles for signal delay fade-out
                for (let ri = 0; ri < removed.length; ri++) this._retireParticle(removed[ri]);
                // Annihilation: emit photon burst from matter-antimatter collisions
                if (annihilations.length > 0 && this.sim) {
                    for (const ann of annihilations) {
                        this.sim.emitPhotonBurst(ann.x, ann.y, ann.energy, 0, -1);
                    }
                }
                // Field excitations from merge collisions (Higgs/Axion boson emission)
                if (merges.length > 0 && this.sim) {
                    const hasHiggs = this.higgsEnabled && this.sim.higgsField;
                    const hasAxion = this.axionEnabled && this.sim.axionField;
                    const share = (hasHiggs && hasAxion) ? 0.5 : 1;
                    for (const m of merges) {
                        const e = m.energy * share;
                        if (hasHiggs) this.sim.higgsField.depositExcitation(m.x, m.y, e, width, height);
                        if (hasAxion) this.sim.axionField.depositExcitation(m.x, m.y, e, width, height);
                    }
                }
            }

            // Photon absorption: p = E/c = E (natural units)
            if (this.radiationEnabled && this.sim && this.sim.photons.length > 0) {
                const softening = this.blackHoleEnabled ? BH_SOFTENING : SOFTENING;
                const photons = this.sim.photons;
                for (let pi = photons.length - 1; pi >= 0; pi--) {
                    const ph = photons[pi];
                    if (!ph.alive) continue;
                    // Query quadtree for nearby particles (reuses pooled array)
                    const candidates = this.pool.queryReuse(root,
                        ph.pos.x, ph.pos.y, softening, softening);
                    for (let ci = 0; ci < candidates.length; ci++) {
                        const target = candidates[ci];
                        if (target.isGhost) continue;
                        if (ph.age < BOSON_MIN_AGE) continue;
                        if (target.id === ph.emitterId && ph.age < BOSON_MIN_AGE * 2) continue;
                        const dx = ph.pos.x - target.pos.x;
                        const dy = ph.pos.y - target.pos.y;
                        const absR = target.radius * BOSON_ABSORB_FRACTION;
                        if (dx * dx + dy * dy < absR * absR) {
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

            // Pion absorption: transfer momentum + charge to absorbing particle
            if (this.yukawaEnabled && this.sim && this.sim.pions.length > 0) {
                const softening = this.blackHoleEnabled ? BH_SOFTENING : SOFTENING;
                const pions = this.sim.pions;
                for (let pi = pions.length - 1; pi >= 0; pi--) {
                    const pn = pions[pi];
                    if (!pn.alive) continue;
                    const candidates = this.pool.queryReuse(root,
                        pn.pos.x, pn.pos.y, softening, softening);
                    for (let ci = 0; ci < candidates.length; ci++) {
                        const target = candidates[ci];
                        if (target.isGhost) continue;
                        if (pn.age < BOSON_MIN_AGE) continue;
                        if (target.id === pn.emitterId && pn.age < BOSON_MIN_AGE * 2) continue;
                        const dx = pn.pos.x - target.pos.x;
                        const dy = pn.pos.y - target.pos.y;
                        const absR = target.radius * BOSON_ABSORB_FRACTION;
                        if (dx * dx + dy * dy < absR * absR) {
                            target.w.x += pn.energy * pn.vel.x / target.mass;
                            target.w.y += pn.energy * pn.vel.y / target.mass;
                            target.charge += pn.charge;
                            if (pn.charge !== 0) target.updateColor();
                            this.sim.totalRadiated -= pn.energy;
                            this.sim.totalRadiatedPx -= pn.energy * pn.vel.x;
                            this.sim.totalRadiatedPy -= pn.energy * pn.vel.y;
                            pn.alive = false;
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
            this._syncAxionField(particles, width, height);
            computeAllForces(particles, toggles, this.pool, root, this.barnesHutEnabled, relOn, this.simTime, this.periodic, this.domainW, this.domainH, this._topologyConst, this.sim && this.sim.deadParticles);
            this._applyExternalFields(particles);
            if (this.higgsEnabled && this.sim && this.sim.higgsField) {
                this.sim.higgsField.applyForces(particles, width, height);
            }
            if (this.axionEnabled && this.sim && this.sim.axionField) {
                this.sim.axionField.applyForces(particles, width, height);
            }
            if (collisionMode === 'bounce') this._applyRepulsion(particles, this.pool, root);
            if (boundaryMode === 'bounce') this._applyBoundaryForces(particles, width, height, offX, offY);
        }

        // Record signal delay history (strided: ~60 snapshots/sec at 100× speed)
        if (relOn && n > 0 && ++this._histStride >= HISTORY_STRIDE) {
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

        // Quadrupole radiation (analytical d³I/dt³ with hybrid jerk)
        // d³I_xx = Σ 2·(3·vx·Fx + x·Jx), d³I_yy analogous
        // d³I_xy = Σ (Jx·y + 3·Fx·vy + 3·vx·Fy + x·Jy)
        // where Jx = dFx/dt = analytical jerk (grav+Coulomb+Yukawa) + backward-diff residual
        if (this.radiationEnabled && n >= 2 && this.sim && (this.gravityEnabled || this.coulombEnabled)) {
            const gwQuad = this.gravityEnabled;
            const emQuad = this.coulombEnabled;
            let d3Ixx = 0, d3Ixy = 0, d3Iyy = 0;
            let d3Qxx = 0, d3Qxy = 0, d3Qyy = 0;
            let jerkReady = true;

            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const x = p.pos.x, y = p.pos.y;
                const vx = p.vel.x, vy = p.vel.y;
                const Fx = p.force.x, Fy = p.force.y;

                // Analytical jerk for grav + Coulomb + Yukawa (from pairForce)
                let Jx = p.jerk.x, Jy = p.jerk.y;

                // Backward difference for residual forces (magnetic, GM, 1PN, spin-curv)
                const resFx = Fx - p.forceGravity.x - p.forceCoulomb.x - p.forceYukawa.x;
                const resFy = Fy - p.forceGravity.y - p.forceCoulomb.y - p.forceYukawa.y;
                if (p._qResCount >= 2 && dt > EPSILON) {
                    // O(dt²) 3-point backward derivative (uniform dt = PHYSICS_DT)
                    const invDt = 1 / dt;
                    const c0 = 0.5 * invDt;
                    const c1 = -2 * invDt;
                    const c2 = 1.5 * invDt;
                    Jx += c0 * p._qResFx0 + c1 * p._qResFx1 + c2 * resFx;
                    Jy += c0 * p._qResFy0 + c1 * p._qResFy1 + c2 * resFy;
                } else if (p._qResCount >= 1 && dt > EPSILON) {
                    // O(dt) 2-point backward fallback
                    Jx += (resFx - p._qResFx1) / dt;
                    Jy += (resFy - p._qResFy1) / dt;
                } else {
                    jerkReady = false;
                }
                // Shift history
                p._qResFx0 = p._qResFx1; p._qResFy0 = p._qResFy1;
                p._qResFx1 = resFx; p._qResFy1 = resFy;
                if (p._qResCount < 2) p._qResCount++;

                // Mass quadrupole d³I_ij/dt³
                if (gwQuad) {
                    d3Ixx += 6 * vx * Fx + 2 * x * Jx;
                    d3Ixy += Jx * y + 3 * Fx * vy + 3 * vx * Fy + x * Jy;
                    d3Iyy += 6 * vy * Fy + 2 * y * Jy;
                }

                // EM quadrupole d³Q_ij/dt³ (same structure, weighted by q/m)
                if (emQuad) {
                    const qm = p.charge * p.invMass;
                    d3Qxx += qm * (6 * vx * Fx + 2 * x * Jx);
                    d3Qxy += qm * (Jx * y + 3 * Fx * vy + 3 * vx * Fy + x * Jy);
                    d3Qyy += qm * (6 * vy * Fy + 2 * y * Jy);
                }
            }

            if (jerkReady) {
                // P_GW = (1/5)|d³I_ij/dt³|²
                const gwPower = 0.2 * (d3Ixx * d3Ixx + 2 * d3Ixy * d3Ixy + d3Iyy * d3Iyy);
                // P_EM = (1/180)|d³Q_ij/dt³|²
                const emPower = emQuad ? (1 / 180) * (d3Qxx * d3Qxx + 2 * d3Qxy * d3Qxy + d3Qyy * d3Qyy) : 0;
                const quadPower = gwPower + emPower;

                if (quadPower > 0) {
                    // Clamp to 1% of system KE to prevent instability
                    let totalKE = 0;
                    for (let i = 0; i < n; i++) {
                        const p = particles[i];
                        const wSq = p.w.x * p.w.x + p.w.y * p.w.y;
                        totalKE += wSq > EPSILON_SQ ? p.mass * wSq / (Math.sqrt(1 + wSq) + 1) : 0;
                    }
                    let dE = quadPower * dt;
                    if (totalKE > EPSILON_SQ) dE = Math.min(dE, QUADRUPOLE_POWER_CLAMP * totalKE);

                    // Split dE proportionally between GW and EM channels
                    const gwFrac = gwPower / quadPower;
                    const gwDE = dE * gwFrac;
                    const emDE = dE - gwDE;

                    this.sim.totalRadiated += dE;

                    // Tangential drag + per-particle accumulation (both ∝ KE)
                    if (dE > EPSILON && totalKE > EPSILON_SQ) {
                        const f = Math.min(0.5 * dE / totalKE, 1);
                        const scale = 1 - f;
                        const fOverDt = f / dt;
                        const invKE = 1 / totalKE;
                        for (let i = 0; i < n; i++) {
                            const p = particles[i];
                            p._radDisplayX -= p.mass * p.w.x * fOverDt;
                            p._radDisplayY -= p.mass * p.w.y * fOverDt;
                            p.w.x *= scale;
                            p.w.y *= scale;
                            // Distribute energy to each particle proportional to KE
                            const wSq = p.w.x * p.w.x + p.w.y * p.w.y;
                            const ke = wSq > EPSILON_SQ ? p.mass * wSq / (Math.sqrt(1 + wSq) + 1) : 0;
                            const frac = ke * invKE;
                            p._quadAccum += gwDE * frac;
                            p._emQuadAccum += emDE * frac;
                        }
                    }

                    // Per-particle emission with quadrupole angular pattern
                    const photons = this.sim.photons;
                    for (let i = 0; i < n && photons.length < MAX_PHOTONS; i++) {
                        const p = particles[i];
                        // GW graviton
                        if (p._quadAccum >= MIN_MASS) {
                            const angle = _quadSample(d3Ixx, d3Ixy);
                            const cosA = Math.cos(angle), sinA = Math.sin(angle);
                            const qOff = spawnOffset(p.radius);
                            const gph = new Photon(
                                p.pos.x + cosA * qOff,
                                p.pos.y + sinA * qOff,
                                cosA, sinA, p._quadAccum, p.id);
                            gph.type = 'grav';
                            photons.push(gph);
                            this.sim.totalRadiatedPx += p._quadAccum * cosA;
                            this.sim.totalRadiatedPy += p._quadAccum * sinA;
                            p._quadAccum = 0;
                        }
                        // EM quadrupole photon
                        if (p._emQuadAccum >= MIN_MASS && photons.length < MAX_PHOTONS) {
                            const angle = _quadSample(d3Qxx, d3Qxy);
                            const cosA = Math.cos(angle), sinA = Math.sin(angle);
                            const eOff = spawnOffset(p.radius);
                            photons.push(new Photon(
                                p.pos.x + cosA * eOff,
                                p.pos.y + sinA * eOff,
                                cosA, sinA, p._emQuadAccum, p.id));
                            this.sim.totalRadiatedPx += p._emQuadAccum * cosA;
                            this.sim.totalRadiatedPy += p._emQuadAccum * sinA;
                            p._emQuadAccum = 0;
                        }
                    }
                }
            }
        }

        // PE once per frame, reusing last substep's tree
        this._lastRoot = lastRoot;
        this.potentialEnergy = computePE(particles, toggles, this.pool, lastRoot, this.barnesHutEnabled, BH_THETA, this.periodic, this.domainW, this.domainH, this._topologyConst);

        // Reconstruct all display forces in a single fused loop
        {
            const soEnabled = this.spinOrbitEnabled;
            const fdEnabled = hasGM && relOn;
            const tidEnabled = hasGrav;
            const radEnabled = this.radiationEnabled;
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const vx = p.vel.x, vy = p.vel.y;
                if (hasMagnetic || hasExtBz) {
                    p.forceMagnetic.x += p.charge * vy * p.Bz;
                    p.forceMagnetic.y -= p.charge * vx * p.Bz;
                }
                if (hasGM) {
                    p.forceGravitomag.x += 4 * p.mass * vy * p.Bgz;
                    p.forceGravitomag.y -= 4 * p.mass * vx * p.Bgz;
                }
                if (soEnabled) {
                    const absAngVel = p.angVel > 0 ? p.angVel : -p.angVel;
                    if (absAngVel >= EPSILON) {
                        if (hasMagnetic) {
                            const absQ = p.charge > 0 ? p.charge : -p.charge;
                            if (absQ >= EPSILON) {
                                const mu = p.magMoment;
                                p.torqueSpinOrbit += -mu * (vx * p.dBzdx + vy * p.dBzdy);
                                p.forceSpinCurv.x += mu * p.dBzdx;
                                p.forceSpinCurv.y += mu * p.dBzdy;
                            }
                        }
                        if (hasGM) {
                            const L = p.angMomentum;
                            p.torqueSpinOrbit += -L * (vx * p.dBgzdx + vy * p.dBgzdy);
                            p.forceSpinCurv.x -= L * p.dBgzdx;
                            p.forceSpinCurv.y -= L * p.dBgzdy;
                        }
                    }
                }
                if (fdEnabled && p._frameDragTorque) p.torqueFrameDrag = p._frameDragTorque;
                if (tidEnabled && p._tidalTorque) p.torqueTidal = p._tidalTorque;
                if (radEnabled) {
                    p.forceRadiation.x = p._radDisplayX;
                    p.forceRadiation.y = p._radDisplayY;
                }
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
                    this._retireParticle(p);
                    continue;
                }
            } else if (boundaryMode === 'loop') {
                wrapPosition(p, this._topologyConst, width, height);
            } else if (boundaryMode === 'bounce') {
                // Safety clamp: Hertz wall forces handle repulsion during substeps,
                // but clamp position to prevent deep penetration at extreme speeds
                if (p.pos.x < left) p.pos.x = left;
                else if (p.pos.x > right) p.pos.x = right;
                if (p.pos.y < top) p.pos.y = top;
                else if (p.pos.y > bottom) p.pos.y = bottom;
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

    checkDisintegration(particles, lastRoot) {
        if (!this.disintegrationEnabled) return { fragments: [], transfers: [] };
        const fragments = [];
        const transfers = [];
        const _periodic = this.periodic;
        const _halfDomW = this.domainW * 0.5, _halfDomH = this.domainH * 0.5;
        const _domW = this.domainW, _domH = this.domainH;
        const _topo = this._topologyConst;
        const useTree = this.barnesHutEnabled && lastRoot >= 0;
        const disintSearchR = Math.max(_domW, _domH) * 0.5;
        const softeningSq = this.blackHoleEnabled ? BH_SOFTENING_SQ : SOFTENING_SQ;

        for (let pi = 0; pi < particles.length; pi++) {
            const p = particles[pi];
            if (p.mass < MIN_MASS * SPAWN_COUNT) continue;

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
                const distSq = dx * dx + dy * dy + softeningSq;
                const invDistSq = 1 / distSq;
                const tidalAccel = TIDAL_STRENGTH * other.mass * p.radius * Math.sqrt(invDistSq) * invDistSq;
                if (tidalAccel > maxTidal) {
                    maxTidal = tidalAccel;
                    strongestOther = other;
                    strongestDx = dx; strongestDy = dy;
                    strongestDist = Math.sqrt(distSq - softeningSq);
                }
            };

            if (useTree) {
                const candidates = this.pool.queryReuse(lastRoot,
                    p.pos.x, p.pos.y, disintSearchR, disintSearchR);
                for (let ci = 0; ci < candidates.length; ci++) {
                    const other = candidates[ci];
                    if (other === p || (other.isGhost && other.original === p)) continue;
                    let dx = other.pos.x - p.pos.x, dy = other.pos.y - p.pos.y;
                    if (_periodic) {
                        minImage(p.pos.x, p.pos.y, other.pos.x, other.pos.y, _topo, _domW, _domH, _halfDomW, _halfDomH, _disintMiOut);
                        dx = _disintMiOut.x; dy = _disintMiOut.y;
                    }
                    _checkNeighbor(other, dx, dy);
                }
            } else {
                for (let oi = 0; oi < particles.length; oi++) {
                    const other = particles[oi];
                    if (other === p) continue;
                    let dx = other.pos.x - p.pos.x, dy = other.pos.y - p.pos.y;
                    if (_periodic) {
                        minImage(p.pos.x, p.pos.y, other.pos.x, other.pos.y, _topo, _domW, _domH, _halfDomW, _halfDomH, _disintMiOut);
                        dx = _disintMiOut.x; dy = _disintMiOut.y;
                    }
                    _checkNeighbor(other, dx, dy);
                }
            }

            if (maxTidal + centrifugal + coulombSelf > selfGravity) {
                fragments.push(p);
            } else if (strongestOther && strongestDist > EPSILON && p.mass > MIN_MASS * 4) {
                // Roche lobe overflow: Eggleton formula r_Roche ≈ 0.462·d·(m/(m+M))^(1/3)
                const d = strongestDist;
                const q = p.mass / (p.mass + strongestOther.mass);
                const rRoche = 0.462 * d * Math.cbrt(q);
                if (p.radius > rRoche * ROCHE_THRESHOLD) {
                    const l1Mag = Math.sqrt(strongestDx * strongestDx + strongestDy * strongestDy);
                    if (l1Mag > EPSILON) {
                        const l1x = strongestDx / l1Mag, l1y = strongestDy / l1Mag;
                        const overflow = p.radius / rRoche - ROCHE_THRESHOLD;
                        const dM = Math.min(overflow * ROCHE_TRANSFER_RATE * p.mass, p.mass * 0.1);
                        if (dM >= MIN_MASS) {
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
