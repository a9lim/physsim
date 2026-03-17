// Boundary wrap/bounce/despawn shader.
// Supports all three topologies: Torus, Klein bottle, RP² (real projective plane).
// Klein/RP² glide reflections flip velocities and angular velocity on axis crossing.
// Bounce uses Hertz contact repulsion (F = delta^1.5) with tangential friction,
// matching CPU _applyBoundaryForces(). Forces written to allForces.f4.xy for display.
// Despawn writes death metadata (deathTime, deathMass, deathAngVel) for signal delay fade-out.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particleState: array<ParticleState>;
@group(0) @binding(2) var<storage, read_write> particleAux: array<ParticleAux>;
@group(0) @binding(3) var<storage, read_write> allForces: array<AllForces>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    var ps = particleState[idx];
    if ((ps.flags & FLAG_ALIVE) == 0u) { return; }

    var x = ps.posX;
    var y = ps.posY;
    let w = uniforms.domainW;
    let h = uniforms.domainH;

    if (uniforms.boundaryMode == BOUND_LOOP) {
        let topo = uniforms.topologyMode;

        if (topo == TOPO_TORUS) {
            // Torus: simple periodic wrap on both axes
            if (x < 0.0) { x += w; }
            else if (x >= w) { x -= w; }
            if (y < 0.0) { y += h; }
            else if (y >= h) { y -= h; }
            ps.posX = x;
            ps.posY = y;

        } else if (topo == TOPO_KLEIN) {
            // Klein bottle: x is periodic, y-wrap is a glide reflection
            // y crossing mirrors x-position and negates x-velocity + angular velocity
            if (x < 0.0) { x += w; }
            else if (x >= w) { x -= w; }
            if (y < 0.0) {
                y += h;
                x = w - x;
                ps.velWX = -ps.velWX;
                ps.angW = -ps.angW;
            } else if (y >= h) {
                y -= h;
                x = w - x;
                ps.velWX = -ps.velWX;
                ps.angW = -ps.angW;
            }
            ps.posX = x;
            ps.posY = y;

        } else {
            // RP² (real projective plane): both axes carry glide reflections
            // x crossing flips y-position and negates y-velocity + angular velocity
            // y crossing flips x-position and negates x-velocity + angular velocity
            var vx = ps.velWX;
            var vy = ps.velWY;
            var aw = ps.angW;

            if (x < 0.0) {
                x += w;
                y = h - y;
                vy = -vy;
                aw = -aw;
            } else if (x >= w) {
                x -= w;
                y = h - y;
                vy = -vy;
                aw = -aw;
            }
            if (y < 0.0) {
                y += h;
                x = w - x;
                vx = -vx;
                aw = -aw;
            } else if (y >= h) {
                y -= h;
                x = w - x;
                vx = -vx;
                aw = -aw;
            }

            ps.posX = x;
            ps.posY = y;
            ps.velWX = vx;
            ps.velWY = vy;
            ps.angW = aw;
        }

    } else if (uniforms.boundaryMode == BOUND_BOUNCE) {
        // Hertz wall repulsion matching CPU _applyBoundaryForces().
        // Applied as impulse (Δw = F·dt/m) since boundary runs after Boris drift.
        let r = particleAux[idx].radius;
        let friction = uniforms.bounceFriction;
        let dt = uniforms.dt;
        let m = ps.mass;
        let relOn = (uniforms.toggles0 & RELATIVITY_BIT) != 0u;

        // Coordinate velocity from proper velocity
        let wSq = ps.velWX * ps.velWX + ps.velWY * ps.velWY;
        let invGamma = select(1.0, 1.0 / sqrt(1.0 + wSq), relOn);
        let vx = ps.velWX * invGamma;
        let vy = ps.velWY * invGamma;

        // Angular velocity from angular proper velocity
        let sr = ps.angW * r;
        let av = select(ps.angW, ps.angW / sqrt(1.0 + sr * sr), relOn);

        let invM = select(0.0, 1.0 / m, m > EPSILON);
        let I = INERTIA_K * m * r * r;

        var extFx: f32 = 0.0;
        var extFy: f32 = 0.0;
        var contactTorque: f32 = 0.0;

        // Left wall (x = 0)
        var delta = r - x;
        if (delta > 0.0) {
            let Fn = delta * sqrt(delta);
            extFx += Fn;
            if (friction > 0.0) {
                let vt = vy + av * r;
                let Ft = -friction * Fn * clamp(vt * 10.0, -1.0, 1.0);
                extFy += Ft;
                contactTorque += r * Ft;
            }
        }

        // Right wall (x = w)
        delta = r - (w - x);
        if (delta > 0.0) {
            let Fn = delta * sqrt(delta);
            extFx -= Fn;
            if (friction > 0.0) {
                let vt = -vy - av * r;
                let Ft = -friction * Fn * clamp(vt * 10.0, -1.0, 1.0);
                extFy -= Ft;
                contactTorque -= r * Ft;
            }
        }

        // Top wall (y = 0)
        delta = r - y;
        if (delta > 0.0) {
            let Fn = delta * sqrt(delta);
            extFy += Fn;
            if (friction > 0.0) {
                let vt = -vx + av * r;
                let Ft = -friction * Fn * clamp(vt * 10.0, -1.0, 1.0);
                extFx -= Ft;
                contactTorque += r * Ft;
            }
        }

        // Bottom wall (y = h)
        delta = r - (h - y);
        if (delta > 0.0) {
            let Fn = delta * sqrt(delta);
            extFy -= Fn;
            if (friction > 0.0) {
                let vt = vx - av * r;
                let Ft = -friction * Fn * clamp(vt * 10.0, -1.0, 1.0);
                extFx += Ft;
                contactTorque -= r * Ft;
            }
        }

        // Apply impulse to proper velocity: Δw = F·dt/m
        ps.velWX += extFx * dt * invM;
        ps.velWY += extFy * dt * invM;

        // Apply angular impulse: ΔangW = τ·dt/I
        if (friction > 0.0 && I > EPSILON) {
            ps.angW += contactTorque * dt / I;
        }

        // Safety clamp: prevent deep wall penetration at extreme speeds
        ps.posX = clamp(x, 0.0, w);
        ps.posY = clamp(y, 0.0, h);

        // Record force + torque in allForces for display (external force arrow)
        if (extFx != 0.0 || extFy != 0.0 || contactTorque != 0.0) {
            var af = allForces[idx];
            af.f4.x += extFx;
            af.f4.y += extFy;
            af.totalForce.x += extFx;
            af.totalForce.y += extFy;
            af.torques.w += contactTorque;
            allForces[idx] = af;
        }

    } else {
        // Despawn: mark particles outside domain + margin as dead
        let margin: f32 = DESPAWN_MARGIN;
        if (x < -margin || x >= w + margin || y < -margin || y >= h + margin) {
            // Write death metadata for signal delay fade-out
            var aux = particleAux[idx];
            aux.deathTime = uniforms.simTime;
            aux.deathMass = ps.mass;
            let r = aux.radius;
            let sr = ps.angW * r;
            let relOn = (uniforms.toggles0 & RELATIVITY_BIT) != 0u;
            aux.deathAngVel = select(ps.angW, ps.angW / sqrt(1.0 + sr * sr), relOn);
            particleAux[idx] = aux;

            // Mark dead + retired so dead GC can reclaim the slot
            ps.flags = (ps.flags & ~FLAG_ALIVE) | FLAG_RETIRED;
        }
    }


    particleState[idx] = ps;
}
