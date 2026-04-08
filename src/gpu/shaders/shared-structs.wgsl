// ── Shared struct definitions for all compute/render shaders ──
// Prepended to shaders that need particle/force/boson struct types.
// Do NOT add bindings, uniforms, or functions here — structs only.

// Core particle state: position, proper velocity, mass, charge, angW, baseMass, flags.
// 36 bytes = 9 × f32/u32.
struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

// Auxiliary per-particle metadata: radius, particleId, death state.
// 20 bytes = 5 × f32/u32.
struct ParticleAux {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
};

// Derived per-particle state computed by cache-derived (32 bytes per particle).
struct ParticleDerived {
    magMoment: f32,
    angMomentum: f32,
    invMass: f32,
    radiusSq: f32,
    velX: f32,
    velY: f32,
    angVel: f32,
    bodyRSq: f32,   // pow(mass, 2/3) — true body radius² (differs from radiusSq in BH mode)
};

// All force/torque/bField accumulators in one struct (160 bytes per particle).
struct AllForces {
    f0: vec4<f32>,          // gravity.xy, coulomb.xy
    f1: vec4<f32>,          // magnetic.xy, gravitomag.xy
    f2: vec4<f32>,          // f1pn.xy, spinCurv.xy
    f3: vec4<f32>,          // radiation.xy, yukawa.xy
    f4: vec4<f32>,          // external.xy, higgs.xy
    f5: vec4<f32>,          // axion.xy, torqueSuperradiance, pad
    torques: vec4<f32>,     // spinOrbit, frameDrag, tidal, contact
    bFields: vec4<f32>,     // Bz, Bgz, extBz, pad
    bFieldGrads: vec4<f32>, // dBzdx, dBzdy, dBgzdx, dBgzdy
    totalForce: vec2<f32>,
    jerk: vec2<f32>,
};

// Radiation accumulator state per particle.
// 48 bytes = 12 × f32.
struct RadiationState {
    radAccum: f32,
    hawkAccum: f32,
    yukawaRadAccum: f32,
    radDisplayX: f32, radDisplayY: f32,
    quadAccum: f32,
    emQuadAccum: f32,
    d3IContrib: f32,
    d3QContrib: f32,
    schwingerAccum: f32,
    _pad1: f32,
    _pad2: f32,
};

// Packed photon pool entry (32 bytes = 8 × f32/u32).
struct Photon {
    posX: f32, posY: f32,
    velX: f32, velY: f32,
    energy: f32,
    emitterId: u32, lifetime: f32, flags: u32,
};

// Packed pion pool entry (48 bytes = 12 × f32/u32).
struct Pion {
    posX: f32, posY: f32,
    wX: f32, wY: f32,
    mass: f32, charge: f32, energy: f32,
    emitterId: u32, age: u32, flags: u32,
    kind: u32, _pad1: u32,   // kind: 0=pion, 1=lepton
};

// Sim uniforms: the full uniform buffer layout shared by most shaders.
// 144 bytes = 36 × f32/u32.
struct SimUniforms {
    dt: f32,
    simTime: f32,
    domainW: f32,
    domainH: f32,
    speedScale: f32,
    softening: f32,
    softeningSq: f32,
    toggles0: u32,
    toggles1: u32,
    yukawaCoupling: f32,
    yukawaMu: f32,
    higgsMass: f32,
    axionMass: f32,
    boundaryMode: u32,
    topologyMode: u32,
    collisionMode: u32,
    maxParticles: u32,
    aliveCount: u32,
    extGravity: f32,
    extGravityAngle: f32,
    extElectric: f32,
    extElectricAngle: f32,
    extBz: f32,
    bounceFriction: f32,
    extGx: f32,
    extGy: f32,
    extEx: f32,
    extEy: f32,
    axionCoupling: f32,
    higgsCoupling: f32,
    particleCount: u32,
    bhTheta: f32,
    frameCount: u32,
    _pad4: u32,
};
