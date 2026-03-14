// Zero all force/torque/bField accumulators at start of each substep.
// Uses packed AllForces struct — 1 storage buffer instead of 10.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> allForces: array<AllForces>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    let zero4 = vec4(0.0, 0.0, 0.0, 0.0);
    var af: AllForces;
    af.f0 = zero4;
    af.f1 = zero4;
    af.f2 = zero4;
    af.f3 = zero4;
    af.f4 = zero4;
    af.f5 = zero4;
    af.torques = zero4;
    af.bFields = zero4;
    af.bFieldGrads = zero4;
    af.totalForce = vec2(0.0, 0.0);
    af._pad = vec2(0.0, 0.0);
    allForces[idx] = af;
}
