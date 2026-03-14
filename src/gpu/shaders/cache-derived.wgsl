// Compute all derived particle properties at start of each substep.
// Outputs packed ParticleDerived struct: magMoment, angMomentum, invMass, radiusSq,
// velX, velY, angVel.
// Also writes radius buffer (used by other shaders independently).
// 8 storage buffers (was 12).

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read> massBuf: array<f32>;
@group(0) @binding(2) var<storage, read> velWX: array<f32>;
@group(0) @binding(3) var<storage, read> velWY: array<f32>;
@group(0) @binding(4) var<storage, read> angWBuf: array<f32>;
@group(0) @binding(5) var<storage, read> chargeBuf: array<f32>;
@group(0) @binding(6) var<storage, read_write> radiusBuf: array<f32>;
@group(0) @binding(7) var<storage, read_write> derived: array<ParticleDerived>;
@group(0) @binding(8) var<storage, read> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    let flag = flags[idx];
    if ((flag & FLAG_ALIVE) == 0u) { return; }

    let m = massBuf[idx];
    let r = pow(m, 1.0 / 3.0);  // cbrt
    let rSq = r * r;
    let invM = select(0.0, 1.0 / m, m > EPSILON);

    let wx = velWX[idx];
    let wy = velWY[idx];
    let wSq = wx * wx + wy * wy;
    let g = sqrt(1.0 + wSq);
    let relOn = hasToggle0(RELATIVITY_BIT);
    let invG = select(1.0, 1.0 / g, relOn);

    // Coordinate velocity
    let vx = wx * invG;
    let vy = wy * invG;

    // Angular velocity from angular proper velocity
    let aw = angWBuf[idx];
    let sr = aw * r;
    let angVel = select(aw, aw / sqrt(1.0 + sr * sr), relOn);

    // Dipole moments
    let q = chargeBuf[idx];
    let magMom = MAG_MOMENT_K * q * angVel * rSq;
    let angMom = INERTIA_K * m * angVel * rSq;

    // Write radius (separate buffer, used by many shaders)
    radiusBuf[idx] = r;

    // Write packed derived struct
    var d: ParticleDerived;
    d.magMoment = magMom;
    d.angMomentum = angMom;
    d.invMass = invM;
    d.radiusSq = rSq;
    d.velX = vx;
    d.velY = vy;
    d.angVel = angVel;
    d._pad = 0.0;
    derived[idx] = d;
}
