// Compute all derived particle properties at start of each substep.
// Outputs packed ParticleDerived struct: magMoment, angMomentum, invMass, radiusSq,
// velX, velY, angVel.
// Also writes radius to ParticleAux (used by other shaders independently).

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read_write> derived: array<ParticleDerived>;
@group(0) @binding(3) var<storage, read_write> particleAux: array<ParticleAux>;
@group(0) @binding(4) var<storage, read_write> axYukMod: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    let flag = particles[idx].flags;
    if ((flag & FLAG_ALIVE) == 0u) { return; }

    let m = particles[idx].mass;
    let bodyR = pow(m, 1.0 / 3.0);  // cbrt(mass)
    let bodyRSq = bodyR * bodyR;
    let invM = select(0.0, 1.0 / m, m > EPSILON);

    let wx = particles[idx].velWX;
    let wy = particles[idx].velWY;
    let wSq = wx * wx + wy * wy;
    let relOn = hasToggle0(RELATIVITY_BIT);
    var invG: f32 = 1.0;
    if (relOn) { invG = 1.0 / sqrt(1.0 + wSq); }

    // Coordinate velocity
    let vx = wx * invG;
    let vy = wy * invG;

    // Angular velocity from angular proper velocity
    let aw = particles[idx].angW;
    var angVel = aw;
    if (relOn) { let sr = aw * bodyR; angVel = aw / sqrt(1.0 + sr * sr); }

    // Dipole moments (always use body radius for I and mu)
    let q = particles[idx].charge;
    let magMom = MAG_MOMENT_K * q * angVel * bodyRSq;
    let angMom = INERTIA_K * m * angVel * bodyRSq;

    // Compute effective radius: body radius normally, Kerr-Newman horizon in BH mode
    let bhOn = hasToggle0(BLACK_HOLE_BIT);
    var activeR = bodyR;
    var activeRSq = bodyRSq;
    if (bhOn) {
        // kerrNewmanRadius: r+ = M + sqrt(M² - a² - Q²)
        let a = INERTIA_K * bodyRSq * abs(angVel);
        let disc = m * m - a * a - q * q;
        activeR = select(m * BH_NAKED_FLOOR, m + sqrt(max(0.0, disc)), disc >= 0.0);
        activeRSq = activeR * activeR;
    }

    // Write radius to particleAux
    var aux = particleAux[idx];
    aux.radius = activeR;
    particleAux[idx] = aux;

    // Write packed derived struct
    var d: ParticleDerived;
    d.magMoment = magMom;
    d.angMomentum = angMom;
    d.invMass = invM;
    d.radiusSq = activeRSq;
    d.velX = vx;
    d.velY = vy;
    d.angVel = angVel;
    d.bodyRSq = bodyRSq;
    derived[idx] = d;

    // Reset axYukMod to default (1,1,1,0) — recomputed by field-forces when fields enabled.
    // Ensures newly spawned particles (e.g. from pion decay) get safe defaults.
    axYukMod[idx] = vec4<f32>(1.0, 1.0, 1.0, 0.0);
}
