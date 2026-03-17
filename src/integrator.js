// ─── Boris Integrator ───
// Adaptive-substep Boris integrator. Separates E-like (position-dependent) and
// B-like (velocity-dependent) forces for exact |v|-preserving rotation.

import QuadTreePool from './quadtree.js';
import { PI, TWO_PI, SOFTENING, BH_SOFTENING, DESPAWN_MARGIN, INERTIA_K, MAG_MOMENT_K, MAX_SUBSTEPS, MIN_MASS, MAX_PHOTONS, MAX_SPEED_RATIO, TIDAL_STRENGTH, SPAWN_COUNT, SOFTENING_SQ, BH_SOFTENING_SQ, QUADTREE_CAPACITY, BH_THETA, HISTORY_SIZE, HISTORY_MASK, HISTORY_STRIDE, DEFAULT_PION_MASS, DEFAULT_AXION_MASS, ROCHE_THRESHOLD, ROCHE_TRANSFER_RATE, DEFAULT_HUBBLE, EPSILON, EPSILON_SQ, MAX_REJECTION_SAMPLES, ABERRATION_THRESHOLD, spawnOffset, kerrNewmanRadius, MAX_PIONS, YUKAWA_COUPLING, BOSON_MIN_AGE, HIGGS_COUPLING, AXION_COUPLING, DEFAULT_HIGGS_MASS, COL_BOUNCE, COL_MERGE, BOUND_LOOP, BOUND_BOUNCE, BOUND_DESPAWN, TORUS, KLEIN, RP2 } from './config.js';
import MasslessBoson from './massless-boson.js';
import Pion from './pion.js';
import { angwToAngVel } from './relativity.js';

import { resetForces, computeAllForces, compute1PN, computeBosonGravity, applyBosonBosonGravity, applyPionPionCoulomb, findPionAnnihilations, getPEAccum } from './forces.js';
import { handleCollisions } from './collisions.js';
import { computePE } from './potential.js'; // kept for preset-load recomputation
import { minImage, wrapPosition } from './topology.js';

// Reused by disintegration to avoid per-call allocation
const _disintMiOut = { x: 0, y: 0 };

// Per-particle quadrupole contribution arrays (Fix A7)
let _d3IContrib = new Float64Array(256);
let _d3QContrib = new Float64Array(256);

/**
 * Rejection-sample a scalar dipole emission angle.
 * Power ∝ cos²(φ − accelAngle): two lobes along ±acceleration axis.
 */
function _scalarDipoleSample(accelAngle) {
    for (let tries = 0; tries < MAX_REJECTION_SAMPLES; tries++) {
        const phi = Math.random() * TWO_PI;
        const cosTheta = Math.cos(phi - accelAngle);
        if (Math.random() <= cosTheta * cosTheta) return phi;
    }
    return accelAngle + (Math.random() < 0.5 ? PI : 0);
}

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
        this._bosonPool = new QuadTreePool(QUADTREE_CAPACITY, 128);
        this._collisionCount = 0;

        this.gravityEnabled = true;
        this.bosonInterEnabled = false;
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
        this.yukawaMu = DEFAULT_PION_MASS;

        this.axionEnabled = false;
        this.axionMass = DEFAULT_AXION_MASS;
        this.expansionEnabled = false;
        this.hubbleParam = DEFAULT_HUBBLE;

        this.higgsEnabled = false;
        this.higgsMass = DEFAULT_HIGGS_MASS;

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
            yukawaMu: DEFAULT_PION_MASS,
            axionEnabled: false,
            softeningSq: SOFTENING_SQ,
        };

        this._ghostPool = [];
        this._ghostCount = 0;
        this._treeParticles = [];
        this._lastRoot = -1;

        // Pre-allocated return arrays for checkDisintegration (avoids GC)
        this._disintFragments = [];
        this._disintTransfers = [];
        this._disintResult = { fragments: this._disintFragments, transfers: this._disintTransfers };
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
        p.histAngW[h] = p.angw;
        p.histTime[h] = this.simTime;
        p.histHead = (h + 1) & HISTORY_MASK;
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

    /** Build Barnes-Hut tree for alive bosons (photons + pions). Returns root index or -1. */
    _buildBosonTree() {
        const photons = this.sim.photons;
        const pions = this.sim.pions;
        const nPh = photons.length, nPi = pions.length;
        if (nPh === 0 && nPi === 0) return -1;
        const bp = this._bosonPool;
        bp.reset();
        const root = bp.alloc(this.boundary.x, this.boundary.y, this.boundary.w, this.boundary.h);
        for (let i = 0; i < nPh; i++) if (photons[i].alive) bp.insert(root, photons[i]);
        for (let i = 0; i < nPi; i++) if (pions[i].alive) bp.insert(root, pions[i]);
        bp.calculateBosonDistribution(root);
        return root;
    }

    /** Annihilate a π⁺π⁻ pair into 2 photons. COM-frame kinematics + Lorentz boost. */
    _annihilatePions(p1, p2) {
        if (!this.sim) return;
        const MBoson = this.sim._MasslessBosonClass;
        if (!MBoson) return;

        // Combined 4-momentum
        const g1 = Math.sqrt(1 + p1.w.x * p1.w.x + p1.w.y * p1.w.y);
        const g2 = Math.sqrt(1 + p2.w.x * p2.w.x + p2.w.y * p2.w.y);
        const E = p1.mass * g1 + p2.mass * g2;
        const px = p1.w.x * p1.mass + p2.w.x * p2.mass; // total proper momentum
        const py = p1.w.y * p1.mass + p2.w.y * p2.mass;
        if (E < EPSILON) return;

        // COM velocity
        const vComX = px / E, vComY = py / E;
        const vComSq = vComX * vComX + vComY * vComY;
        const gammaCom = vComSq < 1e-12 ? 1 : 1 / Math.sqrt(1 - Math.min(vComSq, MAX_SPEED_RATIO * MAX_SPEED_RATIO));

        // COM energy (invariant mass)
        const sCom = E * E - px * px - py * py;
        const mInv = sCom > 0 ? Math.sqrt(sCom) : E;
        const ePhRest = mInv * 0.5; // each photon in COM frame

        // Random rest-frame angle
        const angle = Math.random() * TWO_PI;
        const cosA = Math.cos(angle), sinA = Math.sin(angle);

        const midX = (p1.pos.x + p2.pos.x) * 0.5;
        const midY = (p1.pos.y + p2.pos.y) * 0.5;
        const offset = spawnOffset(Math.cbrt(p1.mass));

        // Bookkeeping: remove pion energy from radiated tally
        this.sim.totalRadiated -= p1.energy + p2.energy;
        this.sim.totalRadiatedPx -= p1.energy * p1.vel.x + p2.energy * p2.vel.x;
        this.sim.totalRadiatedPy -= p1.energy * p1.vel.y + p2.energy * p2.vel.y;

        for (let s = 0; s < 2; s++) {
            const sign = s === 0 ? 1 : -1;
            let phPx = sign * ePhRest * cosA;
            let phPy = sign * ePhRest * sinA;

            // Lorentz boost from COM to lab
            if (vComSq > 1e-12) {
                const vCom = Math.sqrt(vComSq);
                const nx = vComX / vCom, ny = vComY / vCom;
                const pPar = phPx * nx + phPy * ny;
                const pPerpX = phPx - pPar * nx;
                const pPerpY = phPy - pPar * ny;
                const pParB = gammaCom * (pPar + vCom * ePhRest);
                phPx = pParB * nx + pPerpX;
                phPy = pParB * ny + pPerpY;
            }

            const pMag = Math.sqrt(phPx * phPx + phPy * phPy);
            if (pMag < EPSILON) continue;
            const dirX = phPx / pMag, dirY = phPy / pMag;
            const ph = MBoson.acquire(
                midX + dirX * offset, midY + dirY * offset,
                dirX, dirY, pMag, -1 // no emitter
            );
            this.sim.photons.push(ph);
            this.sim.totalRadiated += pMag;
            this.sim.totalRadiatedPx += pMag * dirX;
            this.sim.totalRadiatedPy += pMag * dirY;
        }
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

    /** Sync per-particle axMod/yukMod: interpolate from axion field or default to 1. */
    _syncAxionField(particles, width, height, boundaryMode) {
        if (this.axionEnabled && this.sim && this.sim.axionField) {
            this.sim.axionField.interpolateAxMod(particles, width, height, this.coulombEnabled, this.yukawaEnabled, boundaryMode, this._topologyConst);
        } else {
            for (let i = 0, n = particles.length; i < n; i++) {
                particles[i].axMod = 1;
                particles[i].yukMod = 1;
            }
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
                    p._contactTorque += r * Ft;
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
                    p._contactTorque -= r * Ft;
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
                    p._contactTorque += r * Ft;
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
                    p._contactTorque -= r * Ft;
                }
            }
        }
    }

    /** Apply short-range repulsive contact force (Hertz model) when collisionMode is 'bounce'. */
    _applyRepulsion(particles, pool, root) {
        const friction = this.bounceFriction;
        const n = particles.length;
        for (let i = 0; i < n; i++) {
            const p1 = particles[i];
            const searchR = p1.radius * 2;
            const candidates = pool.queryReuse(root, p1.pos.x, p1.pos.y, searchR, searchR);
            for (let ci = 0; ci < candidates.length; ci++) {
                const p2raw = candidates[ci];
                const p2 = p2raw.isGhost ? p2raw.original : p2raw;
                if (p1 === p2 || p1.id >= p2.id) continue;
                this._repelPair(p1, p2raw.pos.x, p2raw.pos.y, p2, friction);
            }
        }
    }

    /** Apply Hertz contact + friction between one pair. */
    _repelPair(p1, p2x, p2y, p2, friction) {
        const dx = p2x - p1.pos.x;
        const dy = p2y - p1.pos.y;
        const distSq = dx * dx + dy * dy;
        const minDist = p1.radius + p2.radius;
        if (distSq >= minDist * minDist) return; // squared comparison avoids sqrt
        const dist = Math.sqrt(distSq);
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
            p1._contactTorque += p1.radius * Ft;
            p2._contactTorque -= p2.radius * Ft;
        }
    }

    update(particles, dt, collisionMode, boundaryMode, topology, width, height, offX = 0, offY = 0) {
        this.boundary.x = offX + width / 2;
        this.boundary.y = offY + height / 2;
        this.boundary.w = width * 2;
        this.boundary.h = height * 2;

        this.domainW = width;
        this.domainH = height;
        this.periodic = (boundaryMode === BOUND_LOOP);
        this._topologyConst = topology;

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
            this._syncAxionField(particles, width, height, boundaryMode);
            const initRoot = this._buildTree(particles);
            computeAllForces(particles, toggles, this.pool, initRoot, this.barnesHutEnabled, relOn, this.simTime, this.periodic, this.domainW, this.domainH, this._topologyConst, this.sim && this.sim.deadParticles);
            this._applyExternalFields(particles);
            if (this.higgsEnabled && this.sim && this.sim.higgsField) {
                this.sim.higgsField.applyForces(particles, width, height, boundaryMode, this._topologyConst);
            }
            if (this.axionEnabled && this.sim && this.sim.axionField) {
                this.sim.axionField.applyForces(particles, width, height, this.coulombEnabled, this.yukawaEnabled, boundaryMode, this._topologyConst);
            }
            if (this.gravityEnabled && this.sim) {
                if (this.higgsEnabled && this.sim.higgsField) {
                    this.sim.higgsField.applyGravForces(particles, width, height);
                }
                if (this.axionEnabled && this.sim.axionField) {
                    this.sim.axionField.applyGravForces(particles, width, height);
                }
            }
            if (this.bosonInterEnabled && this.sim) {
                const bRoot = this._buildBosonTree();
                if (bRoot >= 0 && this.gravityEnabled) {
                    computeBosonGravity(particles, this._bosonPool, bRoot, toggles.softeningSq);
                }
            }
            if (collisionMode === COL_BOUNCE) this._applyRepulsion(particles, this.pool, initRoot);
            if (boundaryMode === BOUND_BOUNCE) this._applyBoundaryForces(particles, width, height, offX, offY);
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
            let aMax = Math.sqrt(maxAccelSq);
            if (aMax !== aMax) aMax = 0; // NaN guard: prevents NaN from propagating to dtSafe
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
                if (p.w.x !== p.w.x || p.w.y !== p.w.y) { p.w.x = 0; p.w.y = 0; p.vel.x = 0; p.vel.y = 0; }
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

            // Frame-dragging torque + tidal locking + contact friction torque (fused)
            if ((hasGM && relOn) || hasGrav || collisionMode === COL_BOUNCE) {
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    let torque = 0;
                    if (hasGM && relOn) torque += p._frameDragTorque;
                    if (hasGrav) torque += p._tidalTorque;
                    torque += p._contactTorque;
                    if (torque === 0) continue;
                    const I = INERTIA_K * p.mass * p.radiusSq;
                    if (I < EPSILON) continue; // avoid division by near-zero moment of inertia
                    p.angw += torque * dtSub / I;
                    if (p.angw !== p.angw) p.angw = 0; // NaN guard
                    const sr = p.angw * p.radius;
                    p.angVel = relOn ? p.angw / Math.sqrt(1 + sr * sr) : p.angw;
                }
            }

            // Landau-Lifshitz radiation reaction (full 1/c² terms)
            // F_rad = τ·[dF/dt / γ³ − γ·v·(F² − (v·F)²)/m]
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
                    // Analytical jerk from pairForce: gravity, Coulomb, Yukawa,
                    // magnetic dipole, GM dipole, Bazanski, EIH position-only
                    let jerkX = p.jerk.x, jerkY = p.jerk.y;

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

                        // Terms 2+3: −τ·γ·v·(F² − (v·F)²)/(m)
                        // Standard LL form: both power-dissipation terms along v
                        const t23 = -tau * gamma * (fSq - vDotF * vDotF) * invM;
                        fRadX += t23 * vx;
                        fRadY += t23 * vy;
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
                        // Radiated momentum direction = -acceleration (no atan2/cos/sin needed)
                        const aMagSq = ax * ax + ay * ay;
                        if (aMagSq > EPSILON_SQ) {
                            const invAMag = 1 / Math.sqrt(aMagSq);
                            this.sim.totalRadiatedPx -= dE * ax * invAMag;
                            this.sim.totalRadiatedPy -= dE * ay * invAMag;
                        }

                        p._radAccum += dE;
                        if (p._radAccum >= MIN_MASS && this.sim.photons.length < MAX_PHOTONS) {
                            // sin²θ dipole pattern: peak emission ⊥ to acceleration
                            const accelAngle = Math.atan2(ay, ax);
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
                            this.sim.photons.push(MasslessBoson.acquire(
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
                    if (p.mass <= 0) continue;
                    const M = p.mass;
                    const a = INERTIA_K * Math.cbrt(M) ** 2 * Math.abs(p.angVel);
                    const Q = p.charge;
                    const disc = M * M - a * a - Q * Q;
                    let power;
                    if (disc > EPSILON) {
                        const rPlus = M + Math.sqrt(disc);
                        const kappa = Math.sqrt(disc) / (2 * M * rPlus);
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
                    this.sim.totalRadiated += dE;
                    p.baseMass *= 1 - dE / (p.mass + dE);
                    p._hawkAccum += dE;

                    // Full evaporation — mark for removal (main.js cleanup emits final burst)
                    if (p.mass <= MIN_MASS) continue;

                    p.invMass = 1 / p.mass;
                    // Update Kerr-Newman radius (use body r² = cbrt(mass)², not horizon radiusSq)
                    const bodyRSq = Math.cbrt(p.mass) ** 2;
                    p.bodyRadiusSq = bodyRSq;
                    p.radius = kerrNewmanRadius(p.mass, bodyRSq, p.angVel, p.charge);
                    p.radiusSq = p.radius * p.radius;

                    if (p._hawkAccum >= MIN_MASS && this.sim.photons.length < MAX_PHOTONS) {
                        const emitAngle = Math.random() * TWO_PI;
                        const cosA = Math.cos(emitAngle), sinA = Math.sin(emitAngle);
                        const hOff = spawnOffset(p.radius);
                        this.sim.photons.push(MasslessBoson.acquire(
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
                    const dE = YUKAWA_COUPLING / 3 * fYukSq * dtSub;
                    p._yukawaRadAccum += dE;
                    const pionMass = this.yukawaMu;
                    if (p._yukawaRadAccum >= pionMass + MIN_MASS && pions.length < MAX_PIONS) {
                        const ke = p._yukawaRadAccum - pionMass;
                        if (ke > 0) {
                            // Scalar dipole: cos²θ — rejection-sample from two lobes along ±acceleration
                            let angle = _scalarDipoleSample(Math.atan2(p.forceYukawa.y, p.forceYukawa.x));

                            // Relativistic aberration: boost from particle rest frame to lab frame
                            const betaP = Math.min(Math.sqrt(p.vel.x * p.vel.x + p.vel.y * p.vel.y), MAX_SPEED_RATIO);
                            if (betaP > EPSILON) {
                                const gammaP = 1 / Math.sqrt(1 - betaP * betaP);
                                const boostAngle = Math.atan2(p.vel.y, p.vel.x);
                                const phiRel = angle - boostAngle;
                                const labRel = Math.atan2(Math.sin(phiRel), gammaP * (Math.cos(phiRel) + betaP));
                                angle = labRel + boostAngle;
                            }
                            const speed = Math.min(Math.sqrt(ke * (ke + 2 * pionMass)) / (ke + pionMass), MAX_SPEED_RATIO);
                            const gamma = 1 / Math.sqrt(1 - speed * speed);
                            const wx = gamma * speed * Math.cos(angle);
                            const wy = gamma * speed * Math.sin(angle);
                            const charge = Math.abs(p.charge) < EPSILON ? 0 : (Math.random() < 0.5 ? 0 : (Math.random() < 0.5 ? 1 : -1));
                            p.charge -= charge;
                            if (charge !== 0) p.updateColor();
                            const offset = spawnOffset(p.radius);
                            pions.push(Pion.acquire(
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
                                const gamma = Math.sqrt(1 + wSq);
                                const pKE = relOn
                                    ? (gamma - 1) * p.mass
                                    : 0.5 * p.mass * wSq;
                                if (pKE > p._yukawaRadAccum) {
                                    const keNew = pKE - p._yukawaRadAccum;
                                    let wSqNew;
                                    if (relOn) {
                                        const gammaNew = 1 + keNew / p.mass;
                                        wSqNew = gammaNew * gammaNew - 1;
                                    } else {
                                        wSqNew = 2 * keNew / p.mass;
                                    }
                                    if (wSqNew > EPSILON_SQ) {
                                        const scale = Math.sqrt(wSqNew / wSq);
                                        p.w.x *= scale;
                                        p.w.y *= scale;
                                    }
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
            let vvRoot = -1;
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
                const bhOn = this.barnesHutEnabled;
                vvRoot = bhOn ? this._buildTree(particles) : -1;
                compute1PN(particles, toggles.softeningSq, this.periodic, this.domainW, this.domainH, this.domainW * 0.5, this.domainH * 0.5, this._topologyConst, this.gravitomagEnabled, this.magneticEnabled, this.yukawaEnabled, this.yukawaMu, this.simTime, bhOn ? this.pool : null, vvRoot, bhOn);
                for (let i = 0; i < n; i++) {
                    const p = particles[i];
                    const halfDtOverM = halfDt * p.invMass;
                    p.w.x += (p.force1PN.x - p._f1pnOld.x) * halfDtOverM;
                    p.w.y += (p.force1PN.y - p._f1pnOld.y) * halfDtOverM;
                    if (p.w.x !== p.w.x || p.w.y !== p.w.y) { p.w.x = 0; p.w.y = 0; p.vel.x = 0; p.vel.y = 0; }
                }
            }

            // Higgs field evolution + mass modulation
            if (this.higgsEnabled && this.sim && this.sim.higgsField) {
                this.sim.higgsField.update(dtSub, particles, boundaryMode, this._topologyConst, width, height, this.relativityEnabled, this.gravityEnabled, toggles.softeningSq);
                this.sim.higgsField.modulateMasses(particles, dtSub, width, height, this.blackHoleEnabled, boundaryMode, this._topologyConst);
            }

            // Axion field evolution (axMod/yukMod interpolation deferred to step 7)
            if (this.axionEnabled && this.sim && this.sim.axionField) {
                this.sim.axionField.update(dtSub, particles, boundaryMode, this._topologyConst, width, height, this.coulombEnabled, this.yukawaEnabled, this.gravityEnabled, toggles.softeningSq);
            }

            // Step 5: Rebuild quadtree at new positions
            // When 1PN+BH are both on, the VV correction already built the tree at these
            // exact positions. Reuse it for collisions/absorption (spatial queries only).
            // Mass distribution must be recalculated if Higgs changed masses.
            let root;
            if (has1PN && this.barnesHutEnabled && vvRoot >= 0) {
                root = vvRoot;
                // Higgs may have changed particle masses → stale aggregate masses
                if (this.higgsEnabled) this.pool.calculateMassDistribution(root);
            } else {
                root = this._buildTree(particles);
            }
            lastRoot = root;

            // Step 6: Collisions (bounce uses force-based Hertz repulsion; only merge goes here)
            if (collisionMode === COL_MERGE) {
                const { annihilations, merges, removed, spawns } = handleCollisions(particles, this.pool, root, collisionMode, this.bounceFriction, this.relativityEnabled, this.periodic, this.domainW, this.domainH, this._topologyConst);
                this._collisionCount += annihilations.length + merges.length;
                // Retire removed particles for signal delay fade-out
                for (let ri = 0; ri < removed.length; ri++) this._retireParticle(removed[ri]);
                // Deselect removed particles
                if (this.sim && this.sim.selectedParticle) {
                    for (let ri = 0; ri < removed.length; ri++) {
                        if (this.sim.selectedParticle === removed[ri]) {
                            this.sim.selectedParticle = null;
                            break;
                        }
                    }
                }
                // Spawn new particles from merges
                if (spawns.length > 0 && this.sim) {
                    for (let si = 0; si < spawns.length; si++) {
                        const s = spawns[si];
                        this.sim.addParticle(s.x, s.y, 0, 0, {
                            mass: s.mass, baseMass: s.baseMass,
                            charge: s.charge, antimatter: s.antimatter,
                            skipBaseline: true,
                        });
                        // addParticle sets velocity via setVelocity (from vx,vy),
                        // but we need to set proper velocity (w) directly
                        const p = particles[particles.length - 1];
                        p.w.set(s.wx, s.wy);
                        p.angw = s.angw;
                        p.angVel = this.relativityEnabled ? angwToAngVel(p.angw, p.radius) : p.angw;
                        const wSq = s.wx * s.wx + s.wy * s.wy;
                        const invG = this.relativityEnabled ? 1 / Math.sqrt(1 + wSq) : 1;
                        p.vel.x = s.wx * invG;
                        p.vel.y = s.wy * invG;
                        if (this.periodic) wrapPosition(p, this._topologyConst, this.domainW, this.domainH);
                    }
                }
                n = particles.length;
                // Annihilation: emit photon burst from matter-antimatter collisions
                if (annihilations.length > 0 && this.sim) {
                    for (let ai = 0; ai < annihilations.length; ai++) {
                        const ann = annihilations[ai];
                        this.sim.emitPhotonBurst(ann.x, ann.y, ann.energy, 0, -1);
                    }
                }
                // Field excitations from merges and annihilations (Higgs/Axion boson emission)
                const excitations = merges.length > 0 || annihilations.length > 0;
                if (excitations && this.sim) {
                    const hasHiggs = this.higgsEnabled && this.sim.higgsField;
                    const hasAxion = this.axionEnabled && this.sim.axionField;
                    if (hasHiggs || hasAxion) {
                        const g2H = HIGGS_COUPLING * HIGGS_COUPLING;
                        const g2A = AXION_COUPLING * AXION_COUPLING;
                        const fracH = (hasHiggs && hasAxion) ? g2H / (g2H + g2A) : 1;
                        const fracA = (hasHiggs && hasAxion) ? g2A / (g2H + g2A) : 1;
                        for (let ei = 0; ei < merges.length; ei++) {
                            const ev = merges[ei];
                            if (hasHiggs) this.sim.higgsField.depositExcitation(ev.x, ev.y, ev.energy * fracH, width, height);
                            if (hasAxion) this.sim.axionField.depositExcitation(ev.x, ev.y, ev.energy * fracA, width, height);
                        }
                        for (let ei = 0; ei < annihilations.length; ei++) {
                            const ev = annihilations[ei];
                            if (hasHiggs) this.sim.higgsField.depositExcitation(ev.x, ev.y, ev.energy * fracH, width, height);
                            if (hasAxion) this.sim.axionField.depositExcitation(ev.x, ev.y, ev.energy * fracA, width, height);
                        }
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
                        if (target.id === ph.emitterId) continue;
                        const dx = ph.pos.x - target.pos.x;
                        const dy = ph.pos.y - target.pos.y;
                        if (dx * dx + dy * dy < target.radiusSq) {
                            const impulse = ph.energy;
                            const invTM = target.mass > EPSILON ? 1 / target.mass : 0;
                            target.w.x += impulse * ph.vel.x * invTM;
                            target.w.y += impulse * ph.vel.y * invTM;
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
                        if (target.id === pn.emitterId) continue;
                        const dx = pn.pos.x - target.pos.x;
                        const dy = pn.pos.y - target.pos.y;
                        if (dx * dx + dy * dy < target.radiusSq) {
                            const invTM = target.mass > EPSILON ? 1 / target.mass : 0;
                            target.w.x += pn.energy * pn.vel.x * invTM;
                            target.w.y += pn.energy * pn.vel.y * invTM;
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
            this._syncAxionField(particles, width, height, boundaryMode);
            computeAllForces(particles, toggles, this.pool, root, this.barnesHutEnabled, relOn, this.simTime, this.periodic, this.domainW, this.domainH, this._topologyConst, this.sim && this.sim.deadParticles);
            this._applyExternalFields(particles);
            if (this.higgsEnabled && this.sim && this.sim.higgsField) {
                this.sim.higgsField.applyForces(particles, width, height, boundaryMode, this._topologyConst);
            }
            if (this.axionEnabled && this.sim && this.sim.axionField) {
                this.sim.axionField.applyForces(particles, width, height, this.coulombEnabled, this.yukawaEnabled, boundaryMode, this._topologyConst);
            }
            if (this.gravityEnabled && this.sim) {
                if (this.higgsEnabled && this.sim.higgsField) {
                    this.sim.higgsField.applyGravForces(particles, width, height);
                }
                if (this.axionEnabled && this.sim.axionField) {
                    this.sim.axionField.applyGravForces(particles, width, height);
                }
            }
            if (this.bosonInterEnabled && this.sim) {
                const bRoot = this._buildBosonTree();
                if (bRoot >= 0) {
                    if (this.gravityEnabled) {
                        computeBosonGravity(particles, this._bosonPool, bRoot, toggles.softeningSq);
                        applyBosonBosonGravity(this.sim.photons, this.sim.pions, dtSub, this._bosonPool, bRoot);
                    }
                    if (this.coulombEnabled) {
                        applyPionPionCoulomb(this.sim.pions, dtSub, this._bosonPool, bRoot);
                    }
                    // π⁺π⁻ annihilation: opposite-charge pions → 2 photons
                    if (this.coulombEnabled || this.yukawaEnabled) {
                        const pairs = findPionAnnihilations(this.sim.pions, this._bosonPool, bRoot);
                        for (let ai = 0; ai < pairs.length; ai += 2) {
                            this._annihilatePions(pairs[ai], pairs[ai + 1]);
                        }
                    }
                }
            }
            if (collisionMode === COL_BOUNCE) this._applyRepulsion(particles, this.pool, root);
            if (boundaryMode === BOUND_BOUNCE) this._applyBoundaryForces(particles, width, height, offX, offY);
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
                p.histAngW[h] = p.angw;
                p.histTime[h] = this.simTime;
                p.histHead = (h + 1) & HISTORY_MASK;
                if (p.histCount < HISTORY_SIZE) p.histCount++;
            }
        }

        // Quadrupole radiation (analytical d³I/dt³ with hybrid jerk)
        // d³I_xx = Σ 2·(3·vx·Fx + x·Jx), d³I_yy analogous
        // d³I_xy = Σ (Jx·y + 3·Fx·vy + 3·vx·Fy + x·Jy)
        // where Jx = dFx/dt = analytical jerk (grav+Coulomb+Yukawa) + backward-diff residual
        // Coordinates are COM-relative (A6); GW uses trace-free tensor (U1);
        // energy extraction weighted by per-particle contribution (A7).
        if (this.radiationEnabled && n >= 2 && this.sim && (this.gravityEnabled || this.coulombEnabled)) {
            const gwQuad = this.gravityEnabled;
            const emQuad = this.coulombEnabled;
            let d3Ixx = 0, d3Ixy = 0, d3Iyy = 0;
            let d3Qxx = 0, d3Qxy = 0, d3Qyy = 0;
            // Grow per-particle contribution arrays if needed (A7)
            if (n > _d3IContrib.length) {
                _d3IContrib = new Float64Array(n);
                _d3QContrib = new Float64Array(n);
            }

            // Compute center of mass for COM-relative coordinates (A6)
            let comX = 0, comY = 0, totalMassQ = 0;
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                comX += p.pos.x * p.mass;
                comY += p.pos.y * p.mass;
                totalMassQ += p.mass;
            }
            if (totalMassQ > EPSILON) {
                comX /= totalMassQ;
                comY /= totalMassQ;
            }

            // totalKE computed inline to avoid separate O(N) pass
            let totalKE = 0;
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const x = p.pos.x - comX, y = p.pos.y - comY;
                const vx = p.vel.x, vy = p.vel.y;
                const Fx = p.force.x, Fy = p.force.y;

                // Accumulate KE for quadrupole power clamping
                const _wSq = p.w.x * p.w.x + p.w.y * p.w.y;
                if (_wSq > EPSILON_SQ) totalKE += p.mass * _wSq / (Math.sqrt(1 + _wSq) + 1);

                // Analytical jerk from pairForce: gravity, Coulomb, Yukawa,
                // magnetic dipole, GM dipole, Bazanski, EIH position-only
                let Jx = p.jerk.x, Jy = p.jerk.y;

                // Per-particle mass quadrupole contribution d³I_ij/dt³ (A7)
                let d3I_xx_i = 0, d3I_xy_i = 0, d3I_yy_i = 0;
                if (gwQuad) {
                    d3I_xx_i = 6 * vx * Fx + 2 * x * Jx;
                    d3I_xy_i = Jx * y + 3 * Fx * vy + 3 * vx * Fy + x * Jy;
                    d3I_yy_i = 6 * vy * Fy + 2 * y * Jy;
                    d3Ixx += d3I_xx_i;
                    d3Ixy += d3I_xy_i;
                    d3Iyy += d3I_yy_i;
                    _d3IContrib[i] = d3I_xx_i * d3I_xx_i + 2 * d3I_xy_i * d3I_xy_i + d3I_yy_i * d3I_yy_i;
                } else {
                    _d3IContrib[i] = 0;
                }

                // Per-particle EM quadrupole contribution d³Q_ij/dt³ (A7)
                if (emQuad) {
                    const qm = p.charge * p.invMass;
                    const d3Q_xx_i = qm * (6 * vx * Fx + 2 * x * Jx);
                    const d3Q_xy_i = qm * (Jx * y + 3 * Fx * vy + 3 * vx * Fy + x * Jy);
                    const d3Q_yy_i = qm * (6 * vy * Fy + 2 * y * Jy);
                    d3Qxx += d3Q_xx_i;
                    d3Qxy += d3Q_xy_i;
                    d3Qyy += d3Q_yy_i;
                    _d3QContrib[i] = d3Q_xx_i * d3Q_xx_i + 2 * d3Q_xy_i * d3Q_xy_i + d3Q_yy_i * d3Q_yy_i;
                } else {
                    _d3QContrib[i] = 0;
                }
            }

            {
                // P_GW = (1/5)|d³I^TF_ij/dt³|² using trace-free reduced quadrupole (U1)
                // I^TF_ij = I_ij - (1/3)δ_ij·I_kk; for 2D motion in 3D (I_zz=0): trace = I_xx+I_yy
                const trI = d3Ixx + d3Iyy;
                const d3Ixx_tf = d3Ixx - trI / 3;
                const d3Iyy_tf = d3Iyy - trI / 3;
                // d3Ixy unchanged (off-diagonal)
                const gwPower = 0.2 * (d3Ixx_tf * d3Ixx_tf + 2 * d3Ixy * d3Ixy + d3Iyy_tf * d3Iyy_tf);
                // P_EM = (1/180)|d³Q_ij/dt³|² (EM quadrupole is NOT trace-free)
                const emPower = emQuad ? (1 / 180) * (d3Qxx * d3Qxx + 2 * d3Qxy * d3Qxy + d3Qyy * d3Qyy) : 0;
                const quadPower = gwPower + emPower;

                if (quadPower > 0) {
                    const dE = quadPower * dt;

                    // Split dE proportionally between GW and EM channels
                    const gwFrac = gwPower / quadPower;
                    const gwDE = dE * gwFrac;
                    const emDE = dE - gwDE;

                    this.sim.totalRadiated += dE;

                    // Per-particle weighted drag + accumulation (∝ contribution, A7)
                    // Exact relativistic rescaling: dKE_i removes the correct energy
                    // fraction from each particle proportional to its quadrupole contribution.
                    if (dE > EPSILON && totalKE > EPSILON_SQ) {
                        // Sum per-particle contributions for weighting (A7)
                        let totalD3I = 0, totalD3Q = 0;
                        for (let i = 0; i < n; i++) { totalD3I += _d3IContrib[i]; totalD3Q += _d3QContrib[i]; }
                        const invD3I = totalD3I > EPSILON_SQ ? 1 / totalD3I : 0;
                        const invD3Q = totalD3Q > EPSILON_SQ ? 1 / totalD3Q : 0;

                        for (let i = 0; i < n; i++) {
                            const p = particles[i];
                            // Per-particle energy to remove (weighted by contribution)
                            const dKE_i = (invD3I > 0 ? gwDE * _d3IContrib[i] * invD3I : 0)
                                        + (invD3Q > 0 ? emDE * _d3QContrib[i] * invD3Q : 0);
                            // Distribute energy to accumulators for photon emission
                            if (invD3I > 0) p._quadAccum += gwDE * _d3IContrib[i] * invD3I;
                            if (invD3Q > 0) p._emQuadAccum += emDE * _d3QContrib[i] * invD3Q;

                            if (dKE_i <= 0) continue;
                            const wSq = p.w.x * p.w.x + p.w.y * p.w.y;
                            if (wSq < EPSILON_SQ) continue;
                            const gamma = Math.sqrt(1 + wSq);
                            const KE_i = wSq / (gamma + 1) * p.mass;
                            if (dKE_i >= KE_i) {
                                p._radDisplayX -= p.mass * p.w.x / dt;
                                p._radDisplayY -= p.mass * p.w.y / dt;
                                p.w.x = 0; p.w.y = 0;
                            } else {
                                const gammaNew = 1 + (KE_i - dKE_i) / p.mass;
                                const wSqNew = gammaNew * gammaNew - 1;
                                const sc = Math.sqrt(wSqNew / wSq);
                                p._radDisplayX -= p.mass * p.w.x * (1 - sc) / dt;
                                p._radDisplayY -= p.mass * p.w.y * (1 - sc) / dt;
                                p.w.x *= sc;
                                p.w.y *= sc;
                            }
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
                            const gph = MasslessBoson.acquire(
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
                            photons.push(MasslessBoson.acquire(
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

        // PE cached from the last substep's force computation (eliminates separate tree walk)
        this._lastRoot = lastRoot;
        this.potentialEnergy = getPEAccum();
        if (this.gravityEnabled && this.sim) {
            if (this.higgsEnabled && this.sim.higgsField)
                this.potentialEnergy += this.sim.higgsField.gravPE(particles, width, height);
            if (this.axionEnabled && this.sim.axionField)
                this.potentialEnergy += this.sim.axionField.gravPE(particles, width, height);
        }

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
                if (p._contactTorque) p.torqueContact = p._contactTorque;
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

            if (boundaryMode === BOUND_DESPAWN) {
                if (p.pos.x < left - DESPAWN_MARGIN || p.pos.x > right + DESPAWN_MARGIN ||
                    p.pos.y < top - DESPAWN_MARGIN || p.pos.y > bottom + DESPAWN_MARGIN) {
                    this._retireParticle(p);
                    continue;
                }
            } else if (boundaryMode === BOUND_LOOP) {
                wrapPosition(p, this._topologyConst, width, height);
            } else if (boundaryMode === BOUND_BOUNCE) {
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
        this.potentialEnergy = computePE(particles, toggles, this.pool, root >= 0 ? root : -1, this.barnesHutEnabled, BH_THETA, this.periodic, this.domainW, this.domainH, this._topologyConst, this.relativityEnabled, this.simTime);
        if (this.gravityEnabled && this.sim) {
            const dw = this.domainW, dh = this.domainH;
            if (this.higgsEnabled && this.sim.higgsField)
                this.potentialEnergy += this.sim.higgsField.gravPE(particles, dw, dh);
            if (this.axionEnabled && this.sim.axionField)
                this.potentialEnergy += this.sim.axionField.gravPE(particles, dw, dh);
        }
    }

    checkDisintegration(particles, lastRoot) {
        if (!this.disintegrationEnabled) return this._disintResult;
        this._disintFragments.length = 0;
        this._disintTransfers.length = 0;
        const fragments = this._disintFragments;
        const transfers = this._disintTransfers;
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
            const pRadius = p.radius;

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
                    const distSq = dx * dx + dy * dy + softeningSq;
                    const invDistSq = 1 / distSq;
                    const tidalAccel = TIDAL_STRENGTH * other.mass * pRadius * Math.sqrt(invDistSq) * invDistSq;
                    if (tidalAccel > maxTidal) {
                        maxTidal = tidalAccel;
                        strongestOther = other;
                        strongestDx = dx; strongestDy = dy;
                        strongestDist = Math.sqrt(distSq - softeningSq);
                    }
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
                    const distSq = dx * dx + dy * dy + softeningSq;
                    const invDistSq = 1 / distSq;
                    const tidalAccel = TIDAL_STRENGTH * other.mass * pRadius * Math.sqrt(invDistSq) * invDistSq;
                    if (tidalAccel > maxTidal) {
                        maxTidal = tidalAccel;
                        strongestOther = other;
                        strongestDx = dx; strongestDy = dy;
                        strongestDist = Math.sqrt(distSq - softeningSq);
                    }
                }
            }

            if (maxTidal + centrifugal + coulombSelf > selfGravity) {
                fragments.push(p);
            } else if (strongestOther && strongestDist > EPSILON && p.mass > MIN_MASS * 4) {
                // Roche lobe overflow: full Eggleton (1983) formula r_L/a = 0.49q^(2/3) / [0.6q^(2/3) + ln(1+q^(1/3))]
                const d = strongestDist;
                const q = p.mass / (p.mass + strongestOther.mass);
                const q13 = Math.cbrt(q);
                const q23 = q13 * q13;
                const rRoche = d * 0.49 * q23 / (0.6 * q23 + Math.log(1 + q13));
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

        return this._disintResult;
    }
}
