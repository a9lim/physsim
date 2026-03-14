// Boris half-kick: w += F/m * dt/2
// First or second half of the Boris split. Used for both half-kick passes.
// Reads totalForce from packed AllForces struct.
// Reads/writes proper velocity from packed ParticleState struct.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read> allForces: array<AllForces>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }
    if ((particles[idx].flags & FLAG_ALIVE) == 0u) { return; }

    let m = particles[idx].mass;
    if (m < EPSILON) { return; }
    let invM = 1.0 / m;
    let halfDtOverM = uniforms.dt * 0.5 * invM;

    let tf = allForces[idx].totalForce;
    let fx = tf.x;
    let fy = tf.y;

    var wx = particles[idx].velWX;
    var wy = particles[idx].velWY;

    wx += fx * halfDtOverM;
    wy += fy * halfDtOverM;

    // NaN guard
    if (wx != wx || wy != wy) { wx = 0.0; wy = 0.0; }

    particles[idx].velWX = wx;
    particles[idx].velWY = wy;
}
