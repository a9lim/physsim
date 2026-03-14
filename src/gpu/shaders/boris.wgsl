// Boris drift: derive coordinate velocity from proper velocity, then update position.
// vel = w / sqrt(1 + w^2) when relativity on, vel = w when off.
// Reads/writes position and proper velocity from packed ParticleState struct.
// Writes coordinate velocity to packed ParticleDerived struct.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read_write> derived: array<ParticleDerived>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }
    if ((particles[idx].flags & FLAG_ALIVE) == 0u) { return; }

    let wx = particles[idx].velWX;
    let wy = particles[idx].velWY;
    let relOn = hasToggle0(RELATIVITY_BIT);
    let invG = select(1.0, 1.0 / sqrt(1.0 + wx * wx + wy * wy), relOn);
    let vx = wx * invG;
    let vy = wy * invG;

    // Store coordinate velocity in derived struct for next substep's force computation
    var d = derived[idx];
    d.velX = vx;
    d.velY = vy;
    derived[idx] = d;

    // Drift position
    particles[idx].posX = particles[idx].posX + vx * uniforms.dt;
    particles[idx].posY = particles[idx].posY + vy * uniforms.dt;
}
