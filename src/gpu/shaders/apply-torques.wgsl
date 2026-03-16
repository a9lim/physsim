// Apply accumulated torques to angular proper velocity.
// Reads torques from packed AllForces struct.
// Updates angVel in packed ParticleDerived struct.
// Reads angW and mass from packed ParticleState struct.
// Torque = I * dw/dt => dw = torque * dt / I
// I = INERTIA_K * m * r^2
//
// NOTE: angW is written directly back to ParticleState.angW.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read_write> allForces: array<AllForces>;
@group(0) @binding(3) var<storage, read_write> derived: array<ParticleDerived>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }
    if ((particles[idx].flags & FLAG_ALIVE) == 0u) { return; }

    let hasGM = hasToggle0(GRAVITOMAG_BIT);
    let hasGrav = hasToggle0(GRAVITY_BIT);
    let relOn = hasToggle0(RELATIVITY_BIT);

    let t = allForces[idx].torques; // spinOrbit, frameDrag, tidal, contact
    var torque: f32 = 0.0;
    if (hasGM && relOn) { torque += t.y; }  // frame drag
    if (hasGrav) { torque += t.z; }         // tidal
    torque += t.w;                          // contact (from bounce collisions -- Phase 3)

    if (torque == 0.0) { return; }

    let m = particles[idx].mass;
    let bodyRSq = derived[idx].bodyRSq;  // pre-cached pow(mass, 2/3) from cache-derived
    let I = INERTIA_K * m * bodyRSq;
    if (I < EPSILON) { return; }

    var aw = particles[idx].angW;
    aw += torque * uniforms.dt / I;

    // NaN guard
    if (aw != aw) { aw = 0.0; }

    let newAngVel = select(aw, aw / sqrt(1.0 + aw * aw * bodyRSq), relOn);

    // Write back angW to ParticleState
    particles[idx].angW = aw;

    // Update derived struct with new angVel
    var d = derived[idx];
    d.angVel = newAngVel;
    derived[idx] = d;
}
