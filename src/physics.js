import Vec2 from './vec2.js';
import QuadTree, { Rect } from './quadtree.js';
import { BH_THETA, QUADTREE_CAPACITY, SOFTENING_SQ, DESPAWN_MARGIN, INERTIA_K, MAG_MOMENT_K } from './config.js';
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
        this.barnesHutEnabled = true;
        this.bounceFriction = 0.4;

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

        // ─── Boris Integrator ───
        // Separates position-dependent (E-like) forces from velocity-dependent
        // (B-like) forces. The Boris rotation exactly preserves |v|, giving
        // superior long-term stability for magnetic/gravitomagnetic interactions.
        //
        // Steps: compute forces+fields → half-kick(E) → Boris rotate(B) →
        //        half-kick(E) → drift → rebuild tree → collisions → new forces

        // First frame: compute initial forces + B fields if not yet done
        if (!this._forcesInit && n > 0) {
            for (const p of particles) {
                p.angVel = relOn ? spinToAngVel(p.spin, p.radius) : p.spin;
            }
            this.potentialEnergy = 0;
            this._resetForces(particles);
            this._computeAllForces(particles);
            this._forcesInit = true;
        }

        // Step 1: Half-kick proper velocity with position-dependent (E-like) forces
        for (let i = 0; i < n; i++) {
            const p = particles[i];
            const halfDtOverM = dt * 0.5 / p.mass;
            p.w.x += p.force.x * halfDtOverM;
            p.w.y += p.force.y * halfDtOverM;
            if (this.spinOrbitEnabled) p.spin += p.torque * dt * 0.5;
        }

        // Step 2: Boris rotation for velocity-dependent (B-like) forces
        // Handles EM Lorentz and linear gravitomagnetism exactly.
        // In 2D with fields along z, the rotation is in the xy-plane.
        //
        // EM Lorentz: F = q·Bz·(vy, -vx) → dw/dt = (q/m)·Bz/γ · (wy, -wx)
        // Linear GM:  F = 4m·Bgz·(vy, -vx) → dw/dt = 4·Bgz/γ · (wy, -wx)
        //
        // Combined rotation parameter: t = ((q/(2m))·Bz + 2·Bgz) · dt / γ⁻
        // where γ⁻ = √(1 + |w⁻|²) after the first half-kick.
        // s = 2t / (1 + t²)
        // w' = w⁻ + w⁻ × t̂  →  w'x = wx + wy·t,  w'y = wy − wx·t
        // w⁺ = w⁻ + w' × ŝ  →  w⁺x = wx + w'y·s,  w⁺y = wy − w'x·s
        const hasMagnetic = this.magneticEnabled;
        const hasGM = this.gravitomagEnabled;
        if (hasMagnetic || hasGM) {
            for (let i = 0; i < n; i++) {
                const p = particles[i];
                const gamma = relOn ? Math.sqrt(1 + p.w.magSq()) : 1;

                // Combined rotation parameter from EM and GM fields
                let t = 0;
                if (hasMagnetic) t += (p.charge / (2 * p.mass)) * p.Bz;
                if (hasGM) t += 2 * p.Bgz;
                t *= dt / gamma;

                if (t === 0) continue;

                const s = 2 * t / (1 + t * t);
                const wx = p.w.x, wy = p.w.y;

                // w' = w⁻ + w⁻ × t̂  (2D: rotation in xy-plane)
                const wpx = wx + wy * t;
                const wpy = wy - wx * t;

                // w⁺ = w⁻ + w' × ŝ
                p.w.x = wx + wpy * s;
                p.w.y = wy - wpx * s;
            }
        }

        // Step 3: Second half-kick with same E-like forces
        for (let i = 0; i < n; i++) {
            const p = particles[i];
            const halfDtOverM = dt * 0.5 / p.mass;
            p.w.x += p.force.x * halfDtOverM;
            p.w.y += p.force.y * halfDtOverM;
            if (this.spinOrbitEnabled) p.spin += p.torque * dt * 0.5;
        }

        // Step 4: Derive velocity and angular velocity, drift positions
        for (let i = 0; i < n; i++) {
            const p = particles[i];
            const invG = relOn ? 1 / Math.sqrt(1 + p.w.magSq()) : 1;
            p.vel.x = p.w.x * invG;
            p.vel.y = p.w.y * invG;
            p.angVel = relOn ? spinToAngVel(p.spin, p.radius) : p.spin;
            p.pos.x += p.vel.x * dt;
            p.pos.y += p.vel.y * dt;
        }

        // Step 5: Rebuild QuadTree with new positions
        const qt = new QuadTree(this.boundary, QUADTREE_CAPACITY);
        for (const p of particles) qt.insert(p);
        qt.calculateMassDistribution();

        // Step 6: Handle collisions
        if (collisionMode !== 'pass') {
            this.handleCollisions(particles, qt, collisionMode);
        }

        // Step 7: Calculate new forces, B fields, and accumulate PE
        this.potentialEnergy = 0;
        this._resetForces(particles);
        this._computeAllForces(particles, qt);

        // Step 7b: Compute velocity-dependent forces for display only.
        // These are applied via Boris rotation (not kicks), but we add them to the
        // per-type display vectors so force component arrows are accurate.
        // p.force is NOT modified — it contains only position-dependent (E-like)
        // forces used by the half-kicks. The net force display in the renderer
        // and selected particle stats should use the sum of component vectors.
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

        // Step 8: Handle boundaries
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
            p.torque = 0;
            p.torqueMagnetic = 0;
            p.torqueGravitomag = 0;
            p.Bz = 0;
            p.Bgz = 0;
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
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                for (let j = 0; j < particles.length; j++) {
                    if (i === j) continue;
                    const o = particles[j];
                    const oRSq = o.radius * o.radius;
                    this._pairForce(p, o.pos.x, o.pos.y, o.vel.x, o.vel.y,
                        o.mass, o.charge, o.angVel,
                        MAG_MOMENT_K * o.charge * o.angVel * oRSq,
                        INERTIA_K * o.mass * o.angVel * oRSq, p.force);
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
        const Lspin = INERTIA_K * p1.mass * p1.radius * p1.radius * p1.spin
            + INERTIA_K * p2.mass * p2.radius * p2.radius * p2.spin;

        p1.mass = totalMass;
        p1.charge = p1.charge + p2.charge;
        p1.w.set(newWx, newWy);
        p1.pos.set(newX, newY);
        p1.updateColor(); // updates radius = cbrt(totalMass)

        const newI = INERTIA_K * totalMass * p1.radius * p1.radius;
        p1.spin = (Lorb + Lspin) / newI;
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

            // Spin friction: Δspin = J·r / I = J / (INERTIA_K·m·r)
            // Same sign for both — torque arms on opposite sides
            p1.spin -= tangentialImpulse / (INERTIA_K * m1 * p1.radius);
            p2.spin -= tangentialImpulse / (INERTIA_K * m2 * p2.radius);
            p1.angVel = spinToAngVel(p1.spin, p1.radius);
            p2.angVel = spinToAngVel(p2.spin, p2.radius);

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

            p1.spin -= tangentialImpulse / (INERTIA_K * m1 * p1.radius);
            p2.spin -= tangentialImpulse / (INERTIA_K * m2 * p2.radius);
            p1.angVel = p1.spin;
            p2.angVel = p2.spin;

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
     * gravity, Coulomb, magnetic dipole, gravitomagnetic dipole, spin-orbit torques.
     *
     * Velocity-dependent (B-like) forces (Lorentz, linear GM) are NOT computed here.
     * Instead, the B and Bg field z-components are accumulated on the particle for use
     * in the Boris rotation step, which handles these forces exactly.
     */
    _pairForce(p, sx, sy, svx, svy, sMass, sCharge, sSpin, sMagMoment, sAngMomentum, out) {
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
        // Moment of inertia (for torque → angular acceleration conversion)
        const pI = INERTIA_K * p.mass * pRSq;

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
            // Dipole radial component: F = 3μ₁μ₂/r⁴ (aligned ⊥-to-plane dipoles repel)
            const fDir = -3 * (pMagMoment * sMagMoment) * invRSq * invRSq * invR;
            out.x += rx * fDir;
            out.y += ry * fDir;
            p.forceMagnetic.x += rx * fDir;
            p.forceMagnetic.y += ry * fDir;
            // Magnetic dipole PE: +(μ₁μ₂)/r³, aligned repels (F = -dU/dr = 3μ₁μ₂/r⁴)
            this.potentialEnergy += (pMagMoment * sMagMoment) * invR * invRSq * 0.5;

            // Accumulate EM magnetic field Bz for Boris rotation (Lorentz force)
            // B_z = q_s * (v_s × r̂)_z / r³
            const Bz = sCharge * crossSV * invR * invRSq;
            p.Bz += Bz;

            // Spin-orbit torque (EM): τ = μ·B, d(spin)/dt = τ/I = (⅕·q·ω·r²·B) / I
            if (this.spinOrbitEnabled && pI > 0) {
                const emTorque = pMagMoment * Bz / pI;
                p.torque += emTorque;
                p.torqueMagnetic += emTorque;
            }
        }

        if (this.gravitomagEnabled) {
            // Dipole radial component: F = 3L₁L₂/r⁴, co-rotating masses attract (GEM flips EM sign)
            const fDir = 3 * (pAngMomentum * sAngMomentum) * invRSq * invRSq * invR;
            out.x += rx * fDir;
            out.y += ry * fDir;
            p.forceGravitomag.x += rx * fDir;
            p.forceGravitomag.y += ry * fDir;
            // Gravitomagnetic dipole PE: -(L₁·L₂)/r³, co-rotating attracts (F = -dU/dr = 3L₁L₂/r⁴)
            this.potentialEnergy -= (pAngMomentum * sAngMomentum) * invR * invRSq * 0.5;

            // Accumulate GM field Bgz for Boris rotation (linear gravitomagnetism)
            // Bg_z = m_s * (v_s × r̂)_z / r³
            const Bgz = sMass * crossSV * invR * invRSq;
            p.Bgz += Bgz;

            // Spin-orbit torque (GM): τ = L·Bg_phys, Bg_phys = -2·Bgz_stored
            // Factor of 2 from GEM; sign convention: co-rotating attracts → use +2
            if (this.spinOrbitEnabled && pI > 0) {
                const gmTorque = 2 * pAngMomentum * Bgz / pI;
                p.torque += gmTorque;
                p.torqueGravitomag += gmTorque;
            }
        }
    }
}
