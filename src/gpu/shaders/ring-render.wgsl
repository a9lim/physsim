// ring-render.wgsl — Instanced dashed ring overlays for ergosphere and antimatter indicators.
//
// Dispatched twice per frame (once for ergosphere, once for antimatter) with different
// RingParams uniforms. Triangle-strip full-circle ribbon with fragment-shader dashing.
//
// Constants from generated wgslConstants block:
//   FLAG_ALIVE, FLAG_ANTIMATTER, INERTIA_K, TWO_PI

struct CameraUniforms {
    viewMatrix: mat4x4<f32>,
    invViewMatrix: mat4x4<f32>,
    zoom: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    isDarkMode: f32,
};

struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct ParticleAux {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
};

struct ParticleDerived {
    magMoment: f32,
    angMomentum: f32,
    invMass: f32,
    radiusSq: f32,
    velX: f32, velY: f32,
    angVel: f32,
    _pad: f32,
};

struct RingParams {
    color: vec4f,       // ring color (non-premultiplied — premultiply in fragment)
    dashLen: f32,       // dash length in world units
    gapLen: f32,        // gap length in world units
    halfWidth: f32,     // ring half-width in world units
    ringType: u32,      // 0 = ergosphere, 1 = antimatter
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read> particleAux: array<ParticleAux>;
@group(0) @binding(3) var<storage, read> derived: array<ParticleDerived>;
@group(0) @binding(4) var<uniform> ring: RingParams;

const RING_SEGMENTS: u32 = 64u;

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) ringColor: vec4f,
    @location(1) arcAngle: f32,
    @location(2) ringRadius: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32,
) -> VertexOut {
    var out: VertexOut;
    let offScreen = vec4f(0.0, 0.0, -2.0, 1.0);

    let p = particles[instIdx];
    if ((p.flags & FLAG_ALIVE) == 0u) {
        out.position = offScreen;
        return out;
    }

    let aux = particleAux[instIdx];
    let der = derived[instIdx];

    // Compute ring radius based on type
    var ringR: f32;
    if (ring.ringType == 0u) {
        // Ergosphere: r_ergo = M + sqrt(M² - a²)
        if (p.mass <= 0.0) {
            out.position = offScreen;
            return out;
        }
        let M = p.mass;
        let bodyR = pow(M, 1.0 / 3.0);
        let bodyRSq = bodyR * bodyR;
        let a = INERTIA_K * bodyRSq * abs(der.angVel);
        let disc = max(0.0, M * M - a * a);
        ringR = M + sqrt(disc);
        // Skip if ergosphere too close to horizon
        if (ringR <= aux.radius + 0.3) {
            out.position = offScreen;
            return out;
        }
    } else {
        // Antimatter: ring at particle radius + 0.4
        if ((p.flags & FLAG_ANTIMATTER) == 0u) {
            out.position = offScreen;
            return out;
        }
        ringR = aux.radius + 0.4;
    }

    // Generate full-circle triangle strip (RING_SEGMENTS + 1 pairs to close)
    let pointIdx = vertIdx >> 1u;
    let side = vertIdx & 1u;

    if (pointIdx > RING_SEGMENTS) {
        out.position = offScreen;
        return out;
    }

    let t = f32(pointIdx) / f32(RING_SEGMENTS);
    let angle = t * TWO_PI;

    let hw = ring.halfWidth;
    let edgeR = select(ringR - hw, ringR + hw, side == 1u);

    let wx = p.posX + cos(angle) * edgeR;
    let wy = p.posY + sin(angle) * edgeR;

    let worldPos = camera.viewMatrix * vec4f(wx, wy, 0.0, 1.0);
    out.position = worldPos;
    out.ringColor = ring.color;
    out.arcAngle = angle;
    out.ringRadius = ringR;

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    // Dash pattern: snap period to circumference so dashes tile seamlessly at the seam.
    let circumference = TWO_PI * in.ringRadius;
    let nominalPeriod = ring.dashLen + ring.gapLen;
    let nDashes = max(round(circumference / nominalPeriod), 1.0);
    let period = circumference / nDashes;
    let dashFrac = ring.dashLen / nominalPeriod; // preserve dash/gap ratio

    let arcLen = in.arcAngle * in.ringRadius;
    let phase = arcLen - floor(arcLen / period) * period;
    if (phase > period * dashFrac) {
        discard;
    }

    // Premultiplied alpha output
    let c = in.ringColor;
    return vec4f(c.rgb * c.a, c.a);
}
