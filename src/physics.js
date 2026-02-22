import Vec2 from './vec2.js';
import QuadTree, { Rect } from './quadtree.js';

export default class Physics {
    constructor() {
        this.boundary = new Rect(window.innerWidth / 2, window.innerHeight / 2, window.innerWidth * 2, window.innerHeight * 2);
    }

    update(particles, dt, collisionMode, boundaryMode) {
        const width = window.innerWidth;
        const height = window.innerHeight;

        // Rebuild QuadTree
        this.boundary = new Rect(width / 2, height / 2, width * 2, height * 2);
        const qt = new QuadTree(this.boundary, 4);

        for (let p of particles) {
            qt.insert(p);
        }

        qt.calculateMassDistribution();

        if (collisionMode !== 'pass') {
            this.handleCollisions(particles, qt, collisionMode);
        }

        const theta = 0.5;
        const forces = particles.map(p => this.calculateForce(p, qt, theta));

        const despawnLimit = 100;

        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            const F = forces[i];

            p.momentum.add(F.clone().scale(dt));

            const pMagSq = p.momentum.magSq();
            const mSq = p.mass * p.mass;
            const gamma = Math.sqrt(1 + pMagSq / mSq);
            const vRel = p.momentum.clone().scale(1 / (p.mass * gamma));

            p.vel = vRel;
            p.pos.add(vRel.clone().scale(dt));

            if (boundaryMode === 'despawn') {
                if (p.pos.x < -despawnLimit || p.pos.x > width + despawnLimit ||
                    p.pos.y < -despawnLimit || p.pos.y > height + despawnLimit) {
                    particles.splice(i, 1);
                }
            } else if (boundaryMode === 'loop') {
                if (p.pos.x < 0) p.pos.x += width;
                if (p.pos.x > width) p.pos.x -= width;
                if (p.pos.y < 0) p.pos.y += height;
                if (p.pos.y > height) p.pos.y -= height;
            } else if (boundaryMode === 'bounce') {
                let bounced = false;
                if (p.pos.x < p.radius) { p.pos.x = p.radius; p.momentum.x *= -1; bounced = true; }
                if (p.pos.x > width - p.radius) { p.pos.x = width - p.radius; p.momentum.x *= -1; bounced = true; }
                if (p.pos.y < p.radius) { p.pos.y = p.radius; p.momentum.y *= -1; bounced = true; }
                if (p.pos.y > height - p.radius) { p.pos.y = height - p.radius; p.momentum.y *= -1; bounced = true; }

                if (bounced) {
                    const mSqLocal = p.mass * p.mass;
                    const pMagSqLocal = p.momentum.magSq();
                    const gammaLocal = Math.sqrt(1 + pMagSqLocal / mSqLocal);
                    p.vel = p.momentum.clone().scale(1 / (p.mass * gammaLocal));
                }
            }
        }
    }

    handleCollisions(particles, qt, mode) {
        for (let p1 of particles) {
            if (p1.mass === 0) continue;

            const range = new Rect(p1.pos.x, p1.pos.y, p1.radius * 2, p1.radius * 2);
            const candidates = qt.query(range);

            for (let p2 of candidates) {
                if (p1 === p2 || p2.mass === 0) continue;
                if (p1.id >= p2.id) continue;

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
            for (let i = particles.length - 1; i >= 0; i--) {
                if (particles[i].mass === 0) {
                    particles.splice(i, 1);
                }
            }
        }
    }

    resolveMerge(p1, p2) {
        const totalMass = p1.mass + p2.mass;
        const totalCharge = p1.charge + p2.charge;
        const totalSpin = p1.spin + p2.spin;
        const newMomentum = Vec2.add(p1.momentum, p2.momentum);
        const newPos = Vec2.add(p1.pos.clone().scale(p1.mass), p2.pos.clone().scale(p2.mass)).scale(1 / totalMass);

        p1.mass = totalMass;
        p1.charge = totalCharge;
        p1.spin = totalSpin;
        p1.momentum = newMomentum;
        p1.pos = newPos;
        p1.updateColor();

        p2.mass = 0;
    }

    resolveBounce(p1, p2, minDist, dist) {
        const safeDist = dist === 0 ? 0.0001 : dist;
        const offset = dist === 0 ? new Vec2(Math.random() - 0.5, Math.random() - 0.5) : Vec2.sub(p2.pos, p1.pos);

        const n = offset.clone().normalize();
        const t = new Vec2(-n.y, n.x); // tangent direction

        const v1 = p1.vel;
        const v2 = p2.vel;

        const v1n = v1.dot(n);
        const v1t = v1.dot(t);
        const v2n = v2.dot(n);
        const v2t = v2.dot(t);

        if (v2n - v1n > 0) return;

        const m1 = p1.mass;
        const m2 = p2.mass;

        // Normal elastic collision
        const v1nFinal = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2);
        const v2nFinal = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2);

        // Spin-induced tangential shearing
        // Surface velocity at contact: v_surface = spin * radius (2D: spin is angular velocity)
        // Relative tangential surface velocity = (v1t + s1*r1) - (v2t - s2*r2)
        // Friction impulse transfers angular momentum → linear momentum
        const surfaceV1 = v1t + p1.spin * p1.radius;
        const surfaceV2 = v2t - p2.spin * p2.radius;
        const relSurfaceV = surfaceV1 - surfaceV2;

        // Friction coefficient (how much tangential impulse is transferred)
        const friction = 0.4;
        // Effective mass for tangential impulse (like reduced mass)
        const effectiveMass = (m1 * m2) / (m1 + m2);
        const tangentialImpulse = friction * relSurfaceV * effectiveMass;

        // Apply tangential impulse to linear velocities
        const v1tFinal = v1t - tangentialImpulse / m1;
        const v2tFinal = v2t + tangentialImpulse / m2;

        // Spin deceleration: impulse × radius removes angular momentum
        // ΔL = impulse * r, spin change = ΔL / (moment of inertia)
        // For a disk: I = 0.5 * m * r², for simplicity use m * r²
        const I1 = m1 * p1.radius * p1.radius;
        const I2 = m2 * p2.radius * p2.radius;
        p1.spin -= (tangentialImpulse * p1.radius) / I1;
        p2.spin -= (tangentialImpulse * p2.radius) / I2;

        const v1Final = Vec2.add(n.clone().scale(v1nFinal), t.clone().scale(v1tFinal));
        const v2Final = Vec2.add(n.clone().scale(v2nFinal), t.clone().scale(v2tFinal));

        const setMomentumFromVelocity = (p, v) => {
            const speedSq = v.magSq();
            let validV = v;
            if (speedSq >= 1) {
                validV = v.clone().normalize().scale(0.99);
            }
            const gamma = 1 / Math.sqrt(1 - validV.magSq());
            p.momentum = validV.clone().scale(gamma * p.mass);
            p.vel = validV;
        };

        setMomentumFromVelocity(p1, v1Final);
        setMomentumFromVelocity(p2, v2Final);

        const overlap = (minDist - safeDist) + 0.5;
        const correction = n.clone().scale(overlap / 2);
        p1.pos.sub(correction);
        p2.pos.add(correction);
    }

    calculateForce(particle, node, theta) {
        let force = new Vec2(0, 0);
        if (node.totalMass === 0) return force;

        const dVec = Vec2.sub(node.centerOfMass, particle.pos);
        const dSq = dVec.magSq();
        const d = Math.sqrt(dSq);
        const size = node.boundary.w * 2;

        if ((!node.divided && node.points.length > 0) || (node.divided && (size / d < theta))) {
            if (!node.divided) {
                for (let other of node.points) {
                    if (other === particle) continue;

                    const rVec = Vec2.sub(other.pos, particle.pos);
                    let rSq = rVec.magSq();
                    rSq = Math.max(rSq, 25);
                    const r = Math.sqrt(rSq);

                    const fGravity = (particle.mass * other.mass) / rSq;
                    const fCoulomb = -(particle.charge * other.charge) / rSq;
                    const fMagnetic = (particle.charge * particle.spin * other.charge * other.spin) / (rSq * r);
                    const fGravitomag = (particle.mass * particle.spin * other.mass * other.spin) / (rSq * r);

                    const fTotal = fGravity + fCoulomb + fMagnetic + fGravitomag;
                    force.add(rVec.clone().normalize().scale(fTotal));
                }
            } else {
                let rSq = dSq;
                rSq = Math.max(rSq, 25);
                const r = Math.sqrt(rSq);

                const fGravity = (particle.mass * node.totalMass) / rSq;
                const fCoulomb = -(particle.charge * node.totalCharge) / rSq;
                const fMagnetic = (particle.charge * particle.spin * node.totalMagneticMoment) / (rSq * r);
                const fGravitomag = (particle.mass * particle.spin * node.totalAngularMomentum) / (rSq * r);

                const fTotal = fGravity + fCoulomb + fMagnetic + fGravitomag;
                force.add(dVec.clone().normalize().scale(fTotal));
            }
        } else if (node.divided) {
            force.add(this.calculateForce(particle, node.northwest, theta));
            force.add(this.calculateForce(particle, node.northeast, theta));
            force.add(this.calculateForce(particle, node.southwest, theta));
            force.add(this.calculateForce(particle, node.southeast, theta));
        }

        return force;
    }
}
