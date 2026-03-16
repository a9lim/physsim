// Common structs and constants shared across all compute/render shaders.
// This file is prepended to other shaders before compilation.

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
    // Phase 2 additions
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

// Toggle query helpers
fn hasToggle0(bit: u32) -> bool {
    return (uniforms.toggles0 & bit) != 0u;
}

fn hasToggle1(bit: u32) -> bool {
    return (uniforms.toggles1 & bit) != 0u;
}

// ── Packed buffer structs (reduces storage buffer count per shader stage) ──

// All force/torque/bField accumulators in one struct (160 bytes per particle).
// Replaces: forces0-5, torques, bFields, bFieldGrads, totalForce (10 buffers → 1).
struct AllForces {
    f0: vec4<f32>,          // gravity.xy, coulomb.xy
    f1: vec4<f32>,          // magnetic.xy, gravitomag.xy
    f2: vec4<f32>,          // f1pn.xy, spinCurv.xy
    f3: vec4<f32>,          // radiation.xy, yukawa.xy
    f4: vec4<f32>,          // external.xy, higgs.xy
    f5: vec4<f32>,          // axion.xy, pad, pad
    torques: vec4<f32>,     // spinOrbit, frameDrag, tidal, contact
    bFields: vec4<f32>,     // Bz, Bgz, extBz, pad
    bFieldGrads: vec4<f32>, // dBzdx, dBzdy, dBgzdx, dBgzdy
    totalForce: vec2<f32>,
    jerk: vec2<f32>,  // analytical jerk for Larmor radiation (was _pad)
};

// Derived per-particle state computed by cache-derived (32 bytes per particle).
// Replaces: magAngMom, invMassRadSq, vel, angVel (4 buffers → 1).
struct ParticleDerived {
    magMoment: f32,
    angMomentum: f32,
    invMass: f32,
    radiusSq: f32,
    velX: f32,
    velY: f32,
    angVel: f32,
    _pad: f32,
};

// ── Packed particle state structs (reduces storage buffer count per shader stage) ──

// Core particle state: position, proper velocity, mass, charge, angW, baseMass, flags.
// 36 bytes = 9 × f32/u32. Replaces 9 individual SoA buffers → 1.
struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

// Auxiliary per-particle metadata: radius, particleId, death state.
// 20 bytes = 5 × f32/u32. Replaces 5 individual SoA buffers → 1.
struct ParticleAux {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
};

// Radiation accumulator state per particle.
// 48 bytes = 12 × f32. Accumulators + display + quadrupole scratch.
struct RadiationState {
    radAccum: f32,           // Larmor energy accumulator
    hawkAccum: f32,          // Hawking energy accumulator
    yukawaRadAccum: f32,     // pion emission energy accumulator
    radDisplayX: f32, radDisplayY: f32,  // display force vector
    quadAccum: f32,          // GW quadrupole energy accumulator
    emQuadAccum: f32,        // EM quadrupole energy accumulator
    d3IContrib: f32,         // scratch: per-particle GW contribution norm
    d3QContrib: f32,         // scratch: per-particle EM contribution norm
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

// Packed photon pool entry.
// 32 bytes = 8 × f32/u32. Replaces 8 individual SoA buffers → 1.
struct Photon {
    posX: f32, posY: f32,
    velX: f32, velY: f32,
    energy: f32,
    emitterId: u32, lifetime: f32, flags: u32,
};

// Packed pion pool entry.
// 48 bytes = 12 × f32/u32 (with padding for alignment). Replaces 10 individual SoA buffers → 1.
struct Pion {
    posX: f32, posY: f32,
    wX: f32, wY: f32,
    mass: f32, charge: i32, energy: f32,
    emitterId: u32, age: u32, flags: u32,
    _pad0: u32, _pad1: u32,
};

// Minimum image displacement for torus topology (most common case).
// Returns displacement vector from observer at (ox, oy) to source at (sx, sy).
fn torusMinImage(ox: f32, oy: f32, sx: f32, sy: f32) -> vec2<f32> {
    let w = uniforms.domainW;
    let h = uniforms.domainH;
    let halfW = w * 0.5;
    let halfH = h * 0.5;
    var rx = sx - ox;
    if (rx > halfW) { rx -= w; } else if (rx < -halfW) { rx += w; }
    var ry = sy - oy;
    if (ry > halfH) { ry -= h; } else if (ry < -halfH) { ry += h; }
    return vec2(rx, ry);
}

// Full topology-aware minimum image displacement (Torus, Klein, RP²).
// For Klein/RP², evaluates multiple glide-reflection candidates and returns the closest.
fn fullMinImage(ox: f32, oy: f32, sx: f32, sy: f32) -> vec2<f32> {
    let w = uniforms.domainW;
    let h = uniforms.domainH;
    let halfW = w * 0.5;
    let halfH = h * 0.5;
    let topo = uniforms.topologyMode;

    if (topo == TOPO_TORUS) {
        return torusMinImage(ox, oy, sx, sy);
    }

    // Candidate 0: only torus-wrap axes with translational (not glide) periodicity.
    // Klein: x periodic (period W), y glide (period 2H) — only wrap x.
    // RP²: both glide — no wrapping.
    var dx0 = sx - ox;
    var dy0 = sy - oy;
    if (topo == TOPO_KLEIN) {
        if (dx0 > halfW) { dx0 -= w; } else if (dx0 < -halfW) { dx0 += w; }
    }
    var bestSq = dx0 * dx0 + dy0 * dy0;
    var bestDx = dx0;
    var bestDy = dy0;

    if (topo == TOPO_KLEIN) {
        // Klein: y-wrap is glide reflection (x,y) ~ (W-x, y+H)
        let gx = w - sx;
        var dx1 = gx - ox;
        if (dx1 > halfW) { dx1 -= w; } else if (dx1 < -halfW) { dx1 += w; }
        var dy1 = (sy + h) - oy;
        if (dy1 > h) { dy1 -= 2.0 * h; } else if (dy1 < -h) { dy1 += 2.0 * h; }
        let dSq1 = dx1 * dx1 + dy1 * dy1;
        if (dSq1 < bestSq) { bestDx = dx1; bestDy = dy1; bestSq = dSq1; }

        var dy1b = (sy - h) - oy;
        if (dy1b > h) { dy1b -= 2.0 * h; } else if (dy1b < -h) { dy1b += 2.0 * h; }
        let dSq1b = dx1 * dx1 + dy1b * dy1b;
        if (dSq1b < bestSq) { bestDx = dx1; bestDy = dy1b; }
    } else {
        // RP²: both axes carry glide reflections (translational periods 2W, 2H)

        // Candidate 1: y-glide  (x,y) ~ (W-x, y+H) — x not wrapped
        let gx = w - sx;
        let dxG = gx - ox;
        var dyG = (sy + h) - oy;
        if (dyG > h) { dyG -= 2.0 * h; } else if (dyG < -h) { dyG += 2.0 * h; }
        let dSqG = dxG * dxG + dyG * dyG;
        if (dSqG < bestSq) { bestDx = dxG; bestDy = dyG; bestSq = dSqG; }

        // Candidate 2: x-glide  (x,y) ~ (x+W, H-y) — y not wrapped
        let gy = h - sy;
        var dxH = (sx + w) - ox;
        if (dxH > w) { dxH -= 2.0 * w; } else if (dxH < -w) { dxH += 2.0 * w; }
        let dyH = gy - oy;
        let dSqH = dxH * dxH + dyH * dyH;
        if (dSqH < bestSq) { bestDx = dxH; bestDy = dyH; bestSq = dSqH; }

        // Candidate 3: both glides  (x,y) ~ (2W-x, 2H-y)
        var dxC = (2.0 * w - sx) - ox;
        if (dxC > w) { dxC -= 2.0 * w; } else if (dxC < -w) { dxC += 2.0 * w; }
        var dyC = (2.0 * h - sy) - oy;
        if (dyC > h) { dyC -= 2.0 * h; } else if (dyC < -h) { dyC += 2.0 * h; }
        let dSqC = dxC * dxC + dyC * dyC;
        if (dSqC < bestSq) { bestDx = dxC; bestDy = dyC; }
    }

    return vec2(bestDx, bestDy);
}
