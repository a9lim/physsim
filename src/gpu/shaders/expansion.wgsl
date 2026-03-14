// ─── Cosmological Expansion ───
// Hubble flow: pos += H*(pos - center)*dt
// Momentum drag: w *= (1 - H*dt)

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct ExpansionUniforms {
    hubbleParam: f32,
    dt: f32,
    centerX: f32,  // domainW * 0.5
    centerY: f32,  // domainH * 0.5
    particleCount: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(1) var<uniform> eu: ExpansionUniforms;

@compute @workgroup_size(256)
fn applyExpansion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= eu.particleCount) { return; }
    var p = particles[pid];
    let flag = p.flags;
    if ((flag & 1u) == 0u) { return; }

    let H = eu.hubbleParam;
    let dt = eu.dt;

    // Hubble flow
    p.posX += H * (p.posX - eu.centerX) * dt;
    p.posY += H * (p.posY - eu.centerY) * dt;

    // Momentum drag
    let decay = 1.0 - H * dt;
    p.velWX *= decay;
    p.velWY *= decay;

    particles[pid] = p;
}
