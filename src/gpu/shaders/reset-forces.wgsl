// Zero all force/torque/bField accumulators.
// Phase 1 stub — no forces computed yet, but maintains pipeline structure.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
// Force buffers would be bound here in Phase 2+.
// For Phase 1, this is a no-op placeholder.

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // No-op in Phase 1. Forces will be added in Phase 2.
}
