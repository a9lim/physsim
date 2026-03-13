// Minimal drift shader: pos += vel * dt
// Used in Phase 1 before Boris integrator is ported.
// Velocity is NOT updated (no forces yet) — particles drift in straight lines.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> posX: array<f32>;
@group(0) @binding(2) var<storage, read_write> posY: array<f32>;
@group(0) @binding(3) var<storage, read> velWX: array<f32>;
@group(0) @binding(4) var<storage, read> velWY: array<f32>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    // Skip dead particles
    let flag = flags[idx];
    if ((flag & FLAG_ALIVE) == 0u) { return; }

    // For Phase 1: vel = w (no relativistic correction yet)
    let vx = velWX[idx];
    let vy = velWY[idx];

    posX[idx] = posX[idx] + vx * uniforms.dt;
    posY[idx] = posY[idx] + vy * uniforms.dt;
}
