// Boundary wrap/bounce/despawn shader.
// Phase 1: torus wrap only. Klein/RP2 added in Phase 3.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> posX: array<f32>;
@group(0) @binding(2) var<storage, read_write> posY: array<f32>;
@group(0) @binding(3) var<storage, read_write> velWX: array<f32>;
@group(0) @binding(4) var<storage, read_write> velWY: array<f32>;
@group(0) @binding(5) var<storage, read_write> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    let flag = flags[idx];
    if ((flag & FLAG_ALIVE) == 0u) { return; }

    var x = posX[idx];
    var y = posY[idx];
    let w = uniforms.domainW;
    let h = uniforms.domainH;

    if (uniforms.boundaryMode == BOUND_LOOP) {
        // Torus wrap (Phase 1 — periodic only)
        if (x < 0.0) { x += w; }
        else if (x >= w) { x -= w; }
        if (y < 0.0) { y += h; }
        else if (y >= h) { y -= h; }
        posX[idx] = x;
        posY[idx] = y;

    } else if (uniforms.boundaryMode == BOUND_BOUNCE) {
        var vx = velWX[idx];
        var vy = velWY[idx];
        if (x < 0.0) { x = -x; vx = abs(vx); }
        else if (x >= w) { x = 2.0 * w - x; vx = -abs(vx); }
        if (y < 0.0) { y = -y; vy = abs(vy); }
        else if (y >= h) { y = 2.0 * h - y; vy = -abs(vy); }
        posX[idx] = x;
        posY[idx] = y;
        velWX[idx] = vx;
        velWY[idx] = vy;

    } else {
        // Despawn: mark particles outside domain as dead
        if (x < 0.0 || x >= w || y < 0.0 || y >= h) {
            flags[idx] = flag & ~FLAG_ALIVE;
        }
    }
}
