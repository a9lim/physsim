// Boris half-kick: w += F/m * dt/2
// First or second half of the Boris split. Used for both half-kick passes.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> velWX: array<f32>;
@group(0) @binding(2) var<storage, read_write> velWY: array<f32>;
@group(0) @binding(3) var<storage, read> mass: array<f32>;
@group(0) @binding(4) var<storage, read> totalForce: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }
    if ((flags[idx] & FLAG_ALIVE) == 0u) { return; }

    let m = mass[idx];
    if (m < EPSILON) { return; }
    let invM = 1.0 / m;
    let halfDtOverM = uniforms.dt * 0.5 * invM;

    let tf = totalForce[idx];
    let fx = tf.x;
    let fy = tf.y;

    var wx = velWX[idx];
    var wy = velWY[idx];

    wx += fx * halfDtOverM;
    wy += fy * halfDtOverM;

    // NaN guard
    if (wx != wx || wy != wy) { wx = 0.0; wy = 0.0; }

    velWX[idx] = wx;
    velWY[idx] = wy;
}
