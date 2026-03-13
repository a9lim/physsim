// Apply accumulated torques to angular proper velocity.
// torques buffer: [spinOrbit, frameDrag, tidal, contact]
// Torque = I * dw/dt => dw = torque * dt / I
// I = INERTIA_K * m * r^2

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> angW: array<f32>;
@group(0) @binding(2) var<storage, read> mass: array<f32>;
@group(0) @binding(3) var<storage, read> radius: array<f32>;
@group(0) @binding(4) var<storage, read> torques: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> angVelBuf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }
    if ((flags[idx] & FLAG_ALIVE) == 0u) { return; }

    let hasGM = hasToggle0(GRAVITOMAG_BIT);
    let hasGrav = hasToggle0(GRAVITY_BIT);
    let relOn = hasToggle0(RELATIVITY_BIT);

    let t = torques[idx]; // spinOrbit, frameDrag, tidal, contact
    var torque: f32 = 0.0;
    if (hasGM && relOn) { torque += t.y; }  // frame drag
    if (hasGrav) { torque += t.z; }         // tidal
    torque += t.w;                          // contact (from bounce collisions — Phase 3)

    if (torque == 0.0) { return; }

    let m = mass[idx];
    let r = radius[idx];
    let I = INERTIA_K * m * r * r;
    if (I < EPSILON) { return; }

    var aw = angW[idx];
    aw += torque * uniforms.dt / I;

    // NaN guard
    if (aw != aw) { aw = 0.0; }

    let sr = aw * r;
    let newAngVel = select(aw, aw / sqrt(1.0 + sr * sr), relOn);

    angW[idx] = aw;
    angVelBuf[idx] = newAngVel;
}
