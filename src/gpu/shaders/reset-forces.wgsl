// Zero all force/torque/bField accumulators at start of each substep.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> forces0: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> forces1: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> forces2: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> forces3: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> forces4: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> forces5: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> torques: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> bFields: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> bFieldGrads: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> totalForceX: array<f32>;
@group(0) @binding(11) var<storage, read_write> totalForceY: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    let zero4 = vec4(0.0, 0.0, 0.0, 0.0);
    forces0[idx] = zero4;
    forces1[idx] = zero4;
    forces2[idx] = zero4;
    forces3[idx] = zero4;
    forces4[idx] = zero4;
    forces5[idx] = zero4;
    torques[idx] = zero4;
    bFields[idx] = zero4;
    bFieldGrads[idx] = zero4;
    totalForceX[idx] = 0.0;
    totalForceY[idx] = 0.0;
}
