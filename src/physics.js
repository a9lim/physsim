import Vec2 from './vec2.js';
import QuadTree, { Rect } from './quadtree.js';

// Reusable scratch vectors to avoid allocation in hot paths
const _tmpForce = new Vec2();
const _tmpR = new Vec2();

export default class Physics {
    constructor() {
        this.boundary = new Rect(0, 0, 0, 0);
    }

    update(particles, dt, collisionMode, boundaryMode, width, height) {
        this.boundary.x = width / 2;
        this.boundary.y = height / 2;
        this.boundary.w = width * 2;
        this.boundary.h = height * 2;

        const qt = new QuadTree(this.boundary, 4);
        for (const p of particles) {
            qt.insert(p);
        }
        qt.calculateMassDistribution();

        if (collisionMode !== 'pass') {
            this.handleCollisions(particles, qt, collisionMode);
        }

        const theta = 0.5;
        const n = particles.length;

        // Calculate all forces before integrating (avoids order-dependent updates)
        // Reuse array across frames to reduce GC
        if (!this._forces || this._forces.length !== n) {
            this._forces = new Array(n);
            for (let i = 0; i < n; i++) this._forces[i] = new Vec2();
        }
        for (let i = 0; i < n; i++) {
            this._forces[i].set(0, 0);
            this.calculateForce(particles[i], qt, theta, this._forces[i]);
        }

        const despawnLimit = 100;

        for (let i = n - 1; i >= 0; i--) {
            const p = particles[i];
            const F = this._forces[i];

            p.momentum.x += F.x * dt;
            p.momentum.y += F.y * dt;

            const pMagSq = p.momentum.magSq();
            const mSq = p.mass * p.mass;
            const invMGamma = 1 / (p.mass * Math.sqrt(1 + pMagSq / mSq));

            p.vel.x = p.momentum.x * invMGamma;
            p.vel.y = p.momentum.y * invMGamma;
            p.pos.x += p.vel.x * dt;
            p.pos.y += p.vel.y * dt;

            if (boundaryMode === 'despawn') {
                if (p.pos.x < -despawnLimit || p.pos.x > width + despawnLimit ||
                    p.pos.y < -despawnLimit || p.pos.y > height + despawnLimit) {
                    particles.splice(i, 1);
                }
            } else if (boundaryMode === 'loop') {
                if (p.pos.x < 0) p.pos.x += width;
                else if (p.pos.x > width) p.pos.x -= width;
                if (p.pos.y < 0) p.pos.y += height;
                else if (p.pos.y > height) p.pos.y -= height;
            } else if (boundaryMode === 'bounce') {
                let bounced = false;
                if (p.pos.x < p.radius) { p.pos.x = p.radius; p.momentum.x *= -1; bounced = true; }
                else if (p.pos.x > width - p.radius) { p.pos.x = width - p.radius; p.momentum.x *= -1; bounced = true; }
                if (p.pos.y < p.radius) { p.pos.y = p.radius; p.momentum.y *= -1; bounced = true; }
                else if (p.pos.y > height - p.radius) { p.pos.y = height - p.radius; p.momentum.y *= -1; bounced = true; }

                if (bounced) {
                    const pMagSq2 = p.momentum.magSq();
                    const invMG = 1 / (p.mass * Math.sqrt(1 + pMagSq2 / (p.mass * p.mass)));
                    p.vel.x = p.momentum.x * invMG;
                    p.vel.y = p.momentum.y * invMG;
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
            // Filter dead particles in one pass instead of repeated splice
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
        const newPx = p1.momentum.x + p2.momentum.x;
        const newPy = p1.momentum.y + p2.momentum.y;
        const newX = (p1.pos.x * p1.mass + p2.pos.x * p2.mass) / totalMass;
        const newY = (p1.pos.y * p1.mass + p2.pos.y * p2.mass) / totalMass;

        p1.mass = totalMass;
        p1.charge = p1.charge + p2.charge;
        p1.spin = p1.spin + p2.spin;
        p1.momentum.set(newPx, newPy);
        p1.pos.set(newX, newY);
        p1.updateColor();

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

        const surfaceV1 = v1t + p1.spin * p1.radius;
        const surfaceV2 = v2t - p2.spin * p2.radius;
        const friction = 0.4;
        const effectiveMass = (m1 * m2) / mSum;
        const tangentialImpulse = friction * (surfaceV1 - surfaceV2) * effectiveMass;

        const v1tFinal = v1t - tangentialImpulse / m1;
        const v2tFinal = v2t + tangentialImpulse / m2;

        p1.spin -= tangentialImpulse / (m1 * p1.radius);
        p2.spin -= tangentialImpulse / (m2 * p2.radius);

        const setMomentumFromVel = (p, vn, vt) => {
            let vx = nx * vn + tx * vt;
            let vy = ny * vn + ty * vt;
            const speedSq = vx * vx + vy * vy;
            if (speedSq >= 1) {
                const s = 0.99 / Math.sqrt(speedSq);
                vx *= s; vy *= s;
            }
            const gamma = 1 / Math.sqrt(1 - (vx * vx + vy * vy));
            p.momentum.set(vx * gamma * p.mass, vy * gamma * p.mass);
            p.vel.set(vx, vy);
        };

        setMomentumFromVel(p1, v1nFinal, v1tFinal);
        setMomentumFromVel(p2, v2nFinal, v2tFinal);

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

                    const rx = other.pos.x - particle.pos.x;
                    const ry = other.pos.y - particle.pos.y;
                    let rSq = rx * rx + ry * ry;
                    rSq = rSq < 25 ? 25 : rSq;
                    const r = Math.sqrt(rSq);
                    const invR = 1 / r;
                    const invRSq = 1 / rSq;

                    const fGravity = particle.mass * other.mass * invRSq;
                    const fCoulomb = -(particle.charge * other.charge) * invRSq;
                    const fMagnetic = (particle.charge * particle.spin * other.charge * other.spin) * invRSq * invR;
                    const fGravitomag = (particle.mass * particle.spin * other.mass * other.spin) * invRSq * invR;

                    const fTotal = (fGravity + fCoulomb + fMagnetic + fGravitomag) * invR;
                    out.x += rx * fTotal;
                    out.y += ry * fTotal;
                }
            } else {
                let rSq = dSq < 25 ? 25 : dSq;
                const r = Math.sqrt(rSq);
                const invR = 1 / r;
                const invRSq = 1 / rSq;

                const fGravity = particle.mass * node.totalMass * invRSq;
                const fCoulomb = -(particle.charge * node.totalCharge) * invRSq;
                const fMagnetic = (particle.charge * particle.spin * node.totalMagneticMoment) * invRSq * invR;
                const fGravitomag = (particle.mass * particle.spin * node.totalAngularMomentum) * invRSq * invR;

                const fTotal = (fGravity + fCoulomb + fMagnetic + fGravitomag) * invR;
                out.x += dx * fTotal;
                out.y += dy * fTotal;
            }
        } else if (node.divided) {
            this.calculateForce(particle, node.northwest, theta, out);
            this.calculateForce(particle, node.northeast, theta, out);
            this.calculateForce(particle, node.southwest, theta, out);
            this.calculateForce(particle, node.southeast, theta, out);
        }
    }
}
