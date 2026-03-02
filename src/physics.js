import Vec2 from './vec2.js';
import QuadTree, { Rect } from './quadtree.js';
import { BH_THETA, QUADTREE_CAPACITY, MIN_DIST_SQ, BOUNCE_FRICTION, DESPAWN_MARGIN } from './config.js';
import { setVelocity, spinToAngVel } from './relativity.js';

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
        this.spinOrbitEnabled = true;

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

        // ─── Velocity Verlet (kick-drift-kick) ───

        // First frame: compute initial forces if not yet done
        if (!this._forcesInit && n > 0) {
            for (const p of particles) {
                p.angVel = this.relativityEnabled ? spinToAngVel(p.spin, p.radius) : p.spin;
            }
            const qt0 = new QuadTree(this.boundary, QUADTREE_CAPACITY);
            for (const p of particles) qt0.insert(p);
            qt0.calculateMassDistribution();
            this.potentialEnergy = 0;
            for (const p of particles) {
                p.force.set(0, 0);
                p.forceGravity.set(0, 0);
                p.forceCoulomb.set(0, 0);
                p.forceMagnetic.set(0, 0);
                p.forceGravitomag.set(0, 0);
                p.torque = 0;
                p.torqueMagnetic = 0;
                p.torqueGravitomag = 0;
                this.calculateForce(p, qt0, BH_THETA, p.force);
            }
            this._forcesInit = true;
        }

        // Step 1: Half-kick proper velocity with old forces
        for (let i = 0; i < n; i++) {
            const p = particles[i];
            const halfDtOverM = dt * 0.5 / p.mass;
            p.w.x += p.force.x * halfDtOverM;
            p.w.y += p.force.y * halfDtOverM;
            if (this.spinOrbitEnabled) p.spin += p.torque * dt * 0.5;
        }

        // Step 2: Derive velocity and angular velocity, drift positions
        const relOn = this.relativityEnabled;
        for (let i = 0; i < n; i++) {
            const p = particles[i];
            const invG = relOn ? 1 / Math.sqrt(1 + p.w.magSq()) : 1;

            p.vel.x = p.w.x * invG;
            p.vel.y = p.w.y * invG;
            p.angVel = relOn ? spinToAngVel(p.spin, p.radius) : p.spin;
            p.pos.x += p.vel.x * dt;
            p.pos.y += p.vel.y * dt;
        }

        // Step 3: Rebuild QuadTree with new positions
        const qt = new QuadTree(this.boundary, QUADTREE_CAPACITY);
        for (const p of particles) qt.insert(p);
        qt.calculateMassDistribution();

        // Step 4: Handle collisions
        if (collisionMode !== 'pass') {
            this.handleCollisions(particles, qt, collisionMode);
        }

        // Step 5: Calculate new forces and accumulate PE
        this.potentialEnergy = 0;
        for (const p of particles) {
            p.force.set(0, 0);
            p.forceGravity.set(0, 0);
            p.forceCoulomb.set(0, 0);
            p.forceMagnetic.set(0, 0);
            p.forceGravitomag.set(0, 0);
            p.torque = 0;
            p.torqueMagnetic = 0;
            p.torqueGravitomag = 0;
            this.calculateForce(p, qt, BH_THETA, p.force);
        }

        // Step 6: Half-kick with new forces
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            const halfDtOverM = dt * 0.5 / p.mass;
            p.w.x += p.force.x * halfDtOverM;
            p.w.y += p.force.y * halfDtOverM;
            if (this.spinOrbitEnabled) p.spin += p.torque * dt * 0.5;

            // Re-derive velocity and angular velocity after second kick
            const invG = relOn ? 1 / Math.sqrt(1 + p.w.magSq()) : 1;
            p.vel.x = p.w.x * invG;
            p.vel.y = p.w.y * invG;
            p.angVel = relOn ? spinToAngVel(p.spin, p.radius) : p.spin;

            // Step 7: Handle boundaries
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
        // Conserve momentum: p = m*w, so w_new = (m1*w1 + m2*w2) / totalMass
        const newWx = (p1.mass * p1.w.x + p2.mass * p2.w.x) / totalMass;
        const newWy = (p1.mass * p1.w.y + p2.mass * p2.w.y) / totalMass;
        const newX = (p1.pos.x * p1.mass + p2.pos.x * p2.mass) / totalMass;
        const newY = (p1.pos.y * p1.mass + p2.pos.y * p2.mass) / totalMass;

        p1.mass = totalMass;
        p1.charge = p1.charge + p2.charge;
        p1.spin = p1.spin + p2.spin;
        p1.w.set(newWx, newWy);
        p1.pos.set(newX, newY);
        p1.updateColor();
        p1.angVel = this.relativityEnabled ? spinToAngVel(p1.spin, p1.radius) : p1.spin;

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

        const v1n = p1.vel.x * nx + p1.vel.y * ny;
        const v1t = p1.vel.x * tx + p1.vel.y * ty;
        const v2n = p2.vel.x * nx + p2.vel.y * ny;
        const v2t = p2.vel.x * tx + p2.vel.y * ty;

        if (v2n - v1n > 0) return;

        const m1 = p1.mass, m2 = p2.mass;
        const mSum = m1 + m2;

        const v1nFinal = (v1n * (m1 - m2) + 2 * m2 * v2n) / mSum;
        const v2nFinal = (v2n * (m2 - m1) + 2 * m1 * v1n) / mSum;

        const surfaceV1 = v1t + p1.angVel * p1.radius;
        const surfaceV2 = v2t - p2.angVel * p2.radius;
        const effectiveMass = (m1 * m2) / mSum;
        const tangentialImpulse = BOUNCE_FRICTION * (surfaceV1 - surfaceV2) * effectiveMass;

        const v1tFinal = v1t - tangentialImpulse / m1;
        const v2tFinal = v2t + tangentialImpulse / m2;

        p1.spin -= tangentialImpulse / (m1 * p1.radius);
        p2.spin -= tangentialImpulse / (m2 * p2.radius);
        p1.angVel = this.relativityEnabled ? spinToAngVel(p1.spin, p1.radius) : p1.spin;
        p2.angVel = this.relativityEnabled ? spinToAngVel(p2.spin, p2.radius) : p2.spin;

        setVelocityFromVel(p1, v1nFinal, v1tFinal, nx, ny, tx, ty);
        setVelocityFromVel(p2, v2nFinal, v2tFinal, nx, ny, tx, ty);

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
                    this._pairForce(particle, other.pos.x, other.pos.y, other.vel.x, other.vel.y, other.mass, other.charge, other.angVel, other.charge * other.angVel, other.mass * other.angVel, out);
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
     * Includes radial forces (gravity, Coulomb, dipole) and velocity-dependent forces
     * (Lorentz, linear gravitomagnetism, spin-orbit torque).
     */
    _pairForce(p, sx, sy, svx, svy, sMass, sCharge, sSpin, sMagMoment, sAngMomentum, out) {
        const rx = sx - p.pos.x;
        const ry = sy - p.pos.y;
        let rSq = rx * rx + ry * ry;
        rSq = rSq < MIN_DIST_SQ ? MIN_DIST_SQ : rSq;
        const r = Math.sqrt(rSq);
        const invR = 1 / r;
        const invRSq = 1 / rSq;

        // Cross product of source velocity with separation: (v_s × r̂)_z component
        const crossSV = svx * ry - svy * rx;

        if (this.gravityEnabled) {
            const fDir = p.mass * sMass * invRSq * invR;
            out.x += rx * fDir;
            out.y += ry * fDir;
            p.forceGravity.x += rx * fDir;
            p.forceGravity.y += ry * fDir;
            // Gravitational PE: -G*m1*m2/r (G=1)
            this.potentialEnergy -= p.mass * sMass * invR * 0.5;
        }

        if (this.coulombEnabled) {
            const fDir = -(p.charge * sCharge) * invRSq * invR;
            out.x += rx * fDir;
            out.y += ry * fDir;
            p.forceCoulomb.x += rx * fDir;
            p.forceCoulomb.y += ry * fDir;
            // Coulomb PE: +k*q1*q2/r (k=1)
            this.potentialEnergy += p.charge * sCharge * invR * 0.5;
        }

        if (this.magneticEnabled) {
            // Dipole radial component
            const fDir = (p.charge * p.angVel * sMagMoment) * invRSq * invRSq * invR;
            out.x += rx * fDir;
            out.y += ry * fDir;
            p.forceMagnetic.x += rx * fDir;
            p.forceMagnetic.y += ry * fDir;

            // Lorentz force: F = q_test × v_test × B, where B_z = q_s * crossSV / r³
            const Bz = sCharge * crossSV * invR * invRSq;
            const lorentzX = p.charge * p.vel.y * Bz;
            const lorentzY = -(p.charge * p.vel.x * Bz);
            out.x += lorentzX;
            out.y += lorentzY;
            p.forceMagnetic.x += lorentzX;
            p.forceMagnetic.y += lorentzY;

            // Spin-orbit torque (EM): d(spin)/dt += (q/m) * B_z
            if (this.spinOrbitEnabled && p.mass > 0) {
                const emTorque = (p.charge / p.mass) * Bz;
                p.torque += emTorque;
                p.torqueMagnetic += emTorque;
            }
        }

        if (this.gravitomagEnabled) {
            // Dipole radial component (angular velocity bounded by relativistic derivation)
            const fDir = -(p.mass * p.angVel * sAngMomentum) * invRSq * invRSq * invR;
            out.x += rx * fDir;
            out.y += ry * fDir;
            p.forceGravitomag.x += rx * fDir;
            p.forceGravitomag.y += ry * fDir;

            // Linear gravitomagnetism: opposite sign (co-moving repels)
            const Bgz = sMass * crossSV * invR * invRSq;
            const gmX = -(p.mass * p.vel.y * Bgz);
            const gmY = p.mass * p.vel.x * Bgz;
            out.x += gmX;
            out.y += gmY;
            p.forceGravitomag.x += gmX;
            p.forceGravitomag.y += gmY;

            // Spin-orbit torque (GM): d(spin)/dt -= Bg_z
            if (this.spinOrbitEnabled) {
                p.torque -= Bgz;
                p.torqueGravitomag -= Bgz;
            }
        }
    }
}
