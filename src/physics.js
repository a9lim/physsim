import Vec2 from './vec2.js';
import QuadTree, { Rect } from './quadtree.js';

export default class Physics {
    constructor(config) {
        this.G = config.G;
        this.k = config.k;
        this.c = config.c;
        this.dt = config.dt;

        // Simulation bounds for QuadTree (arbitrarily large to cover likely play area)
        // Center at 0,0? No, canvas coords are positive. Center at window center approx.
        // Let's make it dynamic or just huge.
        this.boundary = new Rect(window.innerWidth / 2, window.innerHeight / 2, window.innerWidth * 2, window.innerHeight * 2);
    }

    update(particles, dt, collisionMode, boundaryMode) {
        const c = this.c;
        const cSq = c * c;

        // 0. Update Quantities
        // ... QuadTree rebuild ...
        // For wrapping/bouncing, boundary depends on screen size.
        // We need to know screen dimensions. 
        // Let's pass bounds rect to update? Or store it in Physics.
        // Using `window.innerWidth` directly inside Physics is okay for this simple app, 
        // but better to use `this.boundary` which we updated in constructor/resize?
        // Constructor set it to huge.
        // Let's use `window` dims for now for the "Edge". 
        // NOTE: Despawn mode uses "huge limit", Loop/Bounce use "Screen Edge".

        const width = window.innerWidth;
        const height = window.innerHeight;

        // 0. Update QuadTree Boundary
        this.boundary = new Rect(width / 2, height / 2, width * 2, height * 2);
        const qt = new QuadTree(this.boundary, 4);

        // Insert particles into QuadTree
        for (let p of particles) {
            qt.insert(p);
        }

        // Calculate Mass Distribution for Barnes-Hut
        qt.calculateMassDistribution();

        // 1. Handle Collisions
        if (collisionMode !== 'pass') {
            this.handleCollisions(particles, qt, collisionMode);
        }

        // 2. Calculate Forces using Barnes-Hut
        const theta = 0.5; // Threshold
        const forces = particles.map(p => this.calculateForce(p, qt, theta));

        // 3. Integration & Boundary
        const despanLimit = 100; // Despawn distance from edge

        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            const F = forces[i];

            p.momentum.add(F.scale(dt));

            const pMagSq = p.momentum.magSq();
            const mSq = p.mass * p.mass;
            const gamma = Math.sqrt(1 + pMagSq / (mSq * cSq));
            const vRel = p.momentum.clone().scale(1 / (p.mass * gamma));

            p.vel = vRel;
            p.pos.add(vRel.clone().scale(dt));

            // Boundary Handling
            if (boundaryMode === 'despawn') {
                if (p.pos.x < -despanLimit || p.pos.x > width + despanLimit ||
                    p.pos.y < -despanLimit || p.pos.y > height + despanLimit) {
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
                    // Re-sync velocity to momentum
                    // v = p / (m * gamma). Since only direction changed, gamma is constant.
                    // Just updating momentum is enough as v is derived next frame or we update v now.
                    // Let's update v for rendering consistency immediate
                    const pMagSq = p.momentum.magSq();
                    const gamma = Math.sqrt(1 + pMagSq / (mSq * cSq));
                    p.vel = p.momentum.clone().scale(1 / (p.mass * gamma));
                }
            }
        }
    }

    handleCollisions(particles, qt, mode) {
        // Use QuadTree to query candidates
        for (let p1 of particles) {
            if (p1.mass === 0) continue;

            const range = new Rect(p1.pos.x, p1.pos.y, p1.radius * 2, p1.radius * 2);
            const candidates = qt.query(range);

            for (let p2 of candidates) {
                if (p1 === p2 || p2.mass === 0) continue;

                // Fix: Add unique IDs to particles or use a Set of processed pairs? 
                // Let's rely on the unique ID generated in particle.js.
                if (p1.id >= p2.id) continue; // Reliable ordering to process pair once

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

        // Remove merged (mass=0)
        if (mode === 'merge') {
            // Filter in place?
            // The main loop above iterates standard array, handled in splice there? 
            // No, we modify `particles` array in `update` loop for despawn.
            // Merging sets mass to 0. We should clean them up.
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
        const newMomentum = Vec2.add(p1.momentum, p2.momentum);

        // Weighted position
        const newPos = Vec2.add(p1.pos.clone().scale(p1.mass), p2.pos.clone().scale(p2.mass)).scale(1 / totalMass);

        p1.mass = totalMass;
        p1.charge = totalCharge;
        p1.momentum = newMomentum;
        p1.pos = newPos;
        p1.updateColor();

        p2.mass = 0; // Mark for removal
    }

    resolveBounce(p1, p2, minDist, dist) {
        // Prevent div by 0 and vector NaN corruptions if overlapping exactly
        const safeDist = dist === 0 ? 0.0001 : dist;
        const offset = dist === 0 ? new Vec2(Math.random() - 0.5, Math.random() - 0.5) : Vec2.sub(p2.pos, p1.pos);

        // 1. Calculate Normal and Tangent
        const n = offset.normalize();
        const t = new Vec2(-n.y, n.x);

        // 2. Decompose Momentum (using Momentum instead of Velocity for Relativistic correctness approx)
        // We treat it as an elastic collision of momentum vectors.
        // For equal mass, they swap normal momentum.
        // For different mass, we use 1D elastic formula on the normal component of momentum / mass?
        // Actually, 1D elastic collision formula on velocity is standard.
        // v1n' = (v1n(m1-m2) + 2m2v2n) / (m1+m2)

        // Let's get velocities
        const v1 = p1.vel;
        const v2 = p2.vel;

        const v1n = v1.dot(n);
        const v1t = v1.dot(t);
        const v2n = v2.dot(n);
        const v2t = v2.dot(t);

        // Check if moving apart
        // n is p2-p1. If v2n - v1n > 0, they are separating.
        if (v2n - v1n > 0) return;

        // Elastic Collision Formula
        const m1 = p1.mass;
        const m2 = p2.mass;

        const v1nFinal = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2);
        const v2nFinal = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2);

        // New Velocity Vectors (ideal)
        // v1' = v1n' * n + v1t * t
        // v2' = v2n' * n + v2t * t

        const v1Final = Vec2.add(n.clone().scale(v1nFinal), t.clone().scale(v1t));
        const v2Final = Vec2.add(n.clone().scale(v2nFinal), t.clone().scale(v2t));

        // Update Momentum to match these new velocities
        // P = gamma * m * v
        // We simply re-calculate momentum from these target velocities.
        // This is "Newtonian Bounce" applied to relativistic particles. 
        // It preserves logical consistency (angle of incidence etc) but might violate relativistic energy if we aren't careful, 
        // but since we derive P from v, it's safe for the simulation stability (v < c is enforced by P->v calc, here we might exceed c if we just set v).

        // Wait, if v1nFinal > c (impossible if inputs < c), we are fine.
        // But let's set P directly.

        const setMomentumFromVelocity = (p, v) => {
            const speedSq = v.magSq();
            const cSq = this.c * this.c;
            let validV = v;

            if (speedSq >= cSq) {
                validV = v.normalize().scale(this.c * 0.99);
            }

            const gamma = 1 / Math.sqrt(1 - validV.magSq() / cSq);
            p.momentum = validV.clone().scale(gamma * p.mass);
            p.vel = validV; // update immediate for next check
        };

        setMomentumFromVelocity(p1, v1Final);
        setMomentumFromVelocity(p2, v2Final);

        // Position Correction (Anti-stick)
        const overlap = (minDist - safeDist) + 0.5; // +epsilon
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

        // If node is a single body (leaf) or sufficiently far away
        if ((!node.divided && node.points.length > 0) || (node.divided && (size / d < theta))) {
            if (!node.divided) {
                // Leaf node: iterate over particles
                for (let other of node.points) {
                    if (other === particle) continue;

                    const rVec = Vec2.sub(other.pos, particle.pos);
                    let rSq = rVec.magSq();
                    rSq = Math.max(rSq, 25); // Softening
                    const fGravityMag = (this.G * particle.mass * other.mass) / rSq;
                    const fTotalMag = fGravityMag - (this.k * particle.charge * other.charge) / rSq;
                    const f = rVec.normalize().scale(fTotalMag);
                    force.add(f);
                }
            } else {
                // Internal node treated as body
                let rSq = dSq;
                rSq = Math.max(rSq, 25);

                const fGravityMag = (this.G * particle.mass * node.totalMass) / rSq;
                const fTotalMag = fGravityMag - (this.k * particle.charge * node.totalCharge) / rSq;

                const f = dVec.normalize().scale(fTotalMag);
                force.add(f);
            }
        } else if (node.divided) {
            // Recurse
            force.add(this.calculateForce(particle, node.northwest, theta));
            force.add(this.calculateForce(particle, node.northeast, theta));
            force.add(this.calculateForce(particle, node.southwest, theta));
            force.add(this.calculateForce(particle, node.southeast, theta));
        }

        return force;
    }
}
