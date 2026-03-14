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

// Toggle bit constants (toggles0)
const GRAVITY_BIT: u32       = 1u;
const COULOMB_BIT: u32       = 2u;
const MAGNETIC_BIT: u32      = 4u;
const GRAVITOMAG_BIT: u32    = 8u;
const ONE_PN_BIT: u32        = 16u;
const RELATIVITY_BIT: u32    = 32u;
const SPIN_ORBIT_BIT: u32    = 64u;
const RADIATION_BIT: u32     = 128u;
const BLACK_HOLE_BIT: u32    = 256u;
const DISINTEGRATION_BIT: u32 = 512u;
const EXPANSION_BIT: u32     = 1024u;
const YUKAWA_BIT: u32        = 2048u;
const HIGGS_BIT: u32         = 4096u;
const AXION_BIT: u32         = 8192u;
const BARNES_HUT_BIT: u32    = 16384u;
const BOSON_GRAV_BIT: u32    = 32768u;

// Toggle bit constants (toggles1)
const FIELD_GRAV_BIT: u32    = 1u;
const HERTZ_BOUNCE_BIT: u32  = 2u;

// Particle flag bits
const FLAG_ALIVE: u32    = 1u;
const FLAG_RETIRED: u32  = 2u;
const FLAG_ANTIMATTER: u32 = 4u;
const FLAG_BH: u32       = 8u;
const FLAG_GHOST: u32    = 16u;

// Boundary modes
const BOUND_DESPAWN: u32 = 0u;
const BOUND_BOUNCE: u32  = 1u;
const BOUND_LOOP: u32    = 2u;

// Topology modes
const TOPO_TORUS: u32 = 0u;
const TOPO_KLEIN: u32 = 1u;
const TOPO_RP2: u32   = 2u;

// Physics constants (from config.js)
const SOFTENING: f32 = 8.0;
const SOFTENING_SQ: f32 = 64.0;
const BH_SOFTENING: f32 = 4.0;
const BH_SOFTENING_SQ: f32 = 16.0;
const INERTIA_K: f32 = 0.4;
const MAG_MOMENT_K: f32 = 0.2;
const TIDAL_STRENGTH: f32 = 0.3;
const YUKAWA_COUPLING_DEFAULT: f32 = 14.0;
const EPSILON: f32 = 1e-9;
const EPSILON_SQ: f32 = 1e-18;
const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717959;

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
    _pad: vec2<f32>,
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
// 32 bytes = 8 × f32. Replaces 8 individual SoA buffers → 1.
struct RadiationState {
    jerkX: f32, jerkY: f32,
    radAccum: f32, hawkAccum: f32, yukawaRadAccum: f32,
    radDisplayX: f32, radDisplayY: f32,
    _pad: f32,
};

// Packed photon pool entry.
// 32 bytes = 8 × f32/u32. Replaces 8 individual SoA buffers → 1.
struct Photon {
    posX: f32, posY: f32,
    velX: f32, velY: f32,
    energy: f32,
    emitterId: u32, age: u32, flags: u32,
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

    // Candidate 0: direct displacement (with torus wrap)
    var dx0 = sx - ox;
    if (dx0 > halfW) { dx0 -= w; } else if (dx0 < -halfW) { dx0 += w; }
    var dy0 = sy - oy;
    if (dy0 > halfH) { dy0 -= h; } else if (dy0 < -halfH) { dy0 += h; }
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
        // RP²: both axes carry glide reflections
        let gx = w - sx;
        var dxG = gx - ox;
        if (dxG > halfW) { dxG -= w; } else if (dxG < -halfW) { dxG += w; }
        var dyG = (sy + h) - oy;
        if (dyG > h) { dyG -= 2.0 * h; } else if (dyG < -h) { dyG += 2.0 * h; }
        let dSqG = dxG * dxG + dyG * dyG;
        if (dSqG < bestSq) { bestDx = dxG; bestDy = dyG; bestSq = dSqG; }

        let gy = h - sy;
        var dxH = (sx + w) - ox;
        if (dxH > w) { dxH -= 2.0 * w; } else if (dxH < -w) { dxH += 2.0 * w; }
        var dyH = gy - oy;
        if (dyH > halfH) { dyH -= h; } else if (dyH < -halfH) { dyH += h; }
        let dSqH = dxH * dxH + dyH * dyH;
        if (dSqH < bestSq) { bestDx = dxH; bestDy = dyH; bestSq = dSqH; }

        var dxC = (w - sx + w) - ox;
        if (dxC > w) { dxC -= 2.0 * w; } else if (dxC < -w) { dxC += 2.0 * w; }
        var dyC = (h - sy + h) - oy;
        if (dyC > h) { dyC -= 2.0 * h; } else if (dyC < -h) { dyC += 2.0 * h; }
        let dSqC = dxC * dxC + dyC * dyC;
        if (dSqC < bestSq) { bestDx = dxC; bestDy = dyC; }
    }

    return vec2(bestDx, bestDy);
}
