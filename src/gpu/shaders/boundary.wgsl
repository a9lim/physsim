// Boundary wrap/bounce/despawn shader.
// Supports all three topologies: Torus, Klein bottle, RP² (real projective plane).
// Klein/RP² glide reflections flip velocities and angular velocity on axis crossing.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particleState: array<ParticleState>;

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
        var vx = ps.velWX;
        var vy = ps.velWY;
        if (x < 0.0) { x = -x; vx = abs(vx); }
        else if (x >= w) { x = 2.0 * w - x; vx = -abs(vx); }
        if (y < 0.0) { y = -y; vy = abs(vy); }
        else if (y >= h) { y = 2.0 * h - y; vy = -abs(vy); }
        ps.posX = x;
        ps.posY = y;
        ps.velWX = vx;
        ps.velWY = vy;

    } else {
        // Despawn: mark particles outside domain + margin as dead
        // DESPAWN_MARGIN = 64 world units (matches CPU config.js)
        let margin: f32 = 64.0;
        if (x < -margin || x >= w + margin || y < -margin || y >= h + margin) {
            // Mark dead + retired so dead GC can reclaim the slot
            ps.flags = (ps.flags & ~FLAG_ALIVE) | FLAG_RETIRED;
        }
    }

    particleState[idx] = ps;
}
