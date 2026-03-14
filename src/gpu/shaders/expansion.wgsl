// ─── Cosmological Expansion ───
// Hubble flow: pos += H*(pos - center)*dt
// Momentum drag: w *= (1 - H*dt)

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

@group(0) @binding(0) var<storage, read_write> posX: array<f32>;
@group(0) @binding(1) var<storage, read_write> posY: array<f32>;
@group(0) @binding(2) var<storage, read_write> velWX: array<f32>;
@group(0) @binding(3) var<storage, read_write> velWY: array<f32>;
@group(0) @binding(4) var<storage, read> flags: array<u32>;
@group(0) @binding(5) var<uniform> eu: ExpansionUniforms;

@compute @workgroup_size(256)
fn applyExpansion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= eu.particleCount) { return; }
    let flag = flags[pid];
    if ((flag & 1u) == 0u) { return; }

    let H = eu.hubbleParam;
    let dt = eu.dt;

    // Hubble flow
    posX[pid] += H * (posX[pid] - eu.centerX) * dt;
    posY[pid] += H * (posY[pid] - eu.centerY) * dt;

    // Momentum drag
    let decay = 1.0 - H * dt;
    velWX[pid] *= decay;
    velWY[pid] *= decay;
}
