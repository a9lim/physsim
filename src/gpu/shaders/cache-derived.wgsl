// Compute all derived particle properties at start of each substep.
// radius = cbrt(mass)
// gamma = sqrt(1 + wSq)
// invMass = 1 / mass
// radiusSq = radius^2
// angVel = angw / sqrt(1 + angw^2 * r^2) (when relativity on, else = angw)
// magMoment = MAG_MOMENT_K * charge * angVel * radiusSq
// angMomentum = INERTIA_K * mass * angVel * radiusSq
// velX, velY = coordinate velocity from proper velocity

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read> massBuf: array<f32>;
@group(0) @binding(2) var<storage, read> velWX: array<f32>;
@group(0) @binding(3) var<storage, read> velWY: array<f32>;
@group(0) @binding(4) var<storage, read> angWBuf: array<f32>;
@group(0) @binding(5) var<storage, read> chargeBuf: array<f32>;
@group(0) @binding(6) var<storage, read_write> radiusBuf: array<f32>;
@group(0) @binding(7) var<storage, read_write> gammaBuf: array<f32>;
@group(0) @binding(8) var<storage, read_write> magMomentBuf: array<f32>;
@group(0) @binding(9) var<storage, read_write> angMomentumBuf: array<f32>;
@group(0) @binding(10) var<storage, read_write> velXBuf: array<f32>;
@group(0) @binding(11) var<storage, read_write> velYBuf: array<f32>;
@group(0) @binding(12) var<storage, read_write> angVelBuf: array<f32>;
@group(0) @binding(13) var<storage, read_write> invMassBuf: array<f32>;
@group(0) @binding(14) var<storage, read_write> radiusSqBuf: array<f32>;
@group(0) @binding(15) var<storage, read> flags: array<u32>;

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

    // Write all derived quantities
    radiusBuf[idx] = r;
    gammaBuf[idx] = g;
    invMassBuf[idx] = invM;
    radiusSqBuf[idx] = rSq;
    velXBuf[idx] = vx;
    velYBuf[idx] = vy;
    angVelBuf[idx] = angVel;
    magMomentBuf[idx] = magMom;
    angMomentumBuf[idx] = angMom;
}
