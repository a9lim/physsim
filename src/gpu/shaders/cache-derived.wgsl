// Compute derived particle properties: radius, gamma.
// Phase 1: radius = cbrt(mass), gamma = sqrt(1 + wSq).

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read> mass: array<f32>;
@group(0) @binding(2) var<storage, read> velWX: array<f32>;
@group(0) @binding(3) var<storage, read> velWY: array<f32>;
@group(0) @binding(4) var<storage, read_write> radius: array<f32>;
@group(0) @binding(5) var<storage, read_write> gamma: array<f32>;
@group(0) @binding(6) var<storage, read> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    let flag = flags[idx];
    if ((flag & FLAG_ALIVE) == 0u) { return; }

    let m = mass[idx];
    radius[idx] = pow(m, 1.0 / 3.0);  // cbrt

    let wx = velWX[idx];
    let wy = velWY[idx];
    let wSq = wx * wx + wy * wy;
    gamma[idx] = sqrt(1.0 + wSq);
}
