// Boris drift: derive coordinate velocity from proper velocity, then update position.
// vel = w / sqrt(1 + w^2) when relativity on, vel = w when off.
// Reads/writes position and proper velocity from packed ParticleState struct.
// Writes coordinate velocity to packed ParticleDerived struct.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read_write> derived: array<ParticleDerived>;
@group(0) @binding(3) var<storage, read_write> allForces: array<AllForces>;

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
    // NaN guard — freeze velocity rather than propagate corruption to Boris rotation / spin-orbit
    var d = derived[idx];
    d.velX = select(vx, 0.0, vx != vx);
    d.velY = select(vy, 0.0, vy != vy);
    derived[idx] = d;

    // Drift position
    var newPosX = particles[idx].posX + vx * uniforms.dt;
    var newPosY = particles[idx].posY + vy * uniforms.dt;

    // NaN guard — freeze particle rather than corrupt simulation
    if (newPosX != newPosX) { newPosX = particles[idx].posX; }
    if (newPosY != newPosY) { newPosY = particles[idx].posY; }

    particles[idx].posX = newPosX;
    particles[idx].posY = newPosY;

    // Reconstruct velocity-dependent display forces (B-like forces handled by Boris rotation).
    // These are not real forces — they're the equivalent Lorentz-like force for display only.
    // Matches CPU integrator.js post-substep reconstruction.
    var af = allForces[idx];
    let q = particles[idx].charge;
    let m = particles[idx].mass;
    let Bz = af.bFields.x;
    let Bgz = af.bFields.y;

    // Magnetic Lorentz: F_display = q * v × B  (into f1.xy)
    let hasMag = hasToggle0(MAGNETIC_BIT) || uniforms.extBz != 0.0;
    if (hasMag) {
        af.f1.x += q * vy * Bz;
        af.f1.y -= q * vx * Bz;
    }

    // Gravitomagnetic Lorentz analog: F_display = 4m * v × Bgz  (into f1.zw)
    if (hasToggle0(GRAVITOMAG_BIT)) {
        af.f1.z += 4.0 * m * vy * Bgz;
        af.f1.w -= 4.0 * m * vx * Bgz;
    }

    allForces[idx] = af;
}
