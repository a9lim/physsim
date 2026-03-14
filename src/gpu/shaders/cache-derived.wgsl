// Compute all derived particle properties at start of each substep.
// Outputs packed ParticleDerived struct: magMoment, angMomentum, invMass, radiusSq,
// velX, velY, angVel.
// Also writes radius to ParticleAux (used by other shaders independently).

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read_write> derived: array<ParticleDerived>;
@group(0) @binding(3) var<storage, read_write> particleAux: array<ParticleAux>;
@group(0) @binding(4) var<storage, read_write> axYukMod: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }

    let flag = particles[idx].flags;
    if ((flag & FLAG_ALIVE) == 0u) { return; }

    let m = particles[idx].mass;
    let r = pow(m, 1.0 / 3.0);  // cbrt
    let rSq = r * r;
    let invM = select(0.0, 1.0 / m, m > EPSILON);

    let wx = particles[idx].velWX;
    let wy = particles[idx].velWY;
    let wSq = wx * wx + wy * wy;
    let g = sqrt(1.0 + wSq);
    let relOn = hasToggle0(RELATIVITY_BIT);
    let invG = select(1.0, 1.0 / g, relOn);

    // Coordinate velocity
    let vx = wx * invG;
    let vy = wy * invG;

    // Angular velocity from angular proper velocity
    let aw = particles[idx].angW;
    let sr = aw * r;
    let angVel = select(aw, aw / sqrt(1.0 + sr * sr), relOn);

    // Dipole moments
    let q = particles[idx].charge;
    let magMom = MAG_MOMENT_K * q * angVel * rSq;
    let angMom = INERTIA_K * m * angVel * rSq;

    // Write radius to particleAux
    var aux = particleAux[idx];
    aux.radius = r;
    particleAux[idx] = aux;

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

    // Reset axYukMod to default (1,1) — recomputed by applyAxionForces when axion enabled.
    // Ensures newly spawned particles (e.g. from pion decay) get safe defaults.
    axYukMod[idx] = vec2<f32>(1.0, 1.0);
}
