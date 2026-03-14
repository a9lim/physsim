// spin-render.wgsl — Instanced spin indicator arcs around particles.
//
// Each particle with non-zero angular velocity gets an arc (partial circle)
// indicating rotation direction and speed. Arc length proportional to |angVel|,
// direction indicates sign (CW = positive in y-down canvas).
//
// Rendered as a line strip forming an arc. 32 vertices per instance.

struct CameraUniforms {
    viewMatrix: mat4x4<f32>,
    invViewMatrix: mat4x4<f32>,
    zoom: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    _pad: f32,
};

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

// Packed auxiliary struct (matches common.wgsl ParticleAux)
struct ParticleAux {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
};

// Cached derived quantities (matches common.wgsl ParticleDerived)
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

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read> particleAux: array<ParticleAux>;
@group(0) @binding(3) var<storage, read> derived: array<ParticleDerived>;

const ALIVE_BIT: u32 = 1u;
const ARC_SEGMENTS: u32 = 32u;
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const HALF_PI: f32 = 1.5707963268;
const MIN_ANGVEL: f32 = 0.01;  // Skip drawing for very slow rotation

// Cyan (#4AACA0) for positive angVel (CW), orange (#CC8E4E) for negative (CCW)
const COLOR_CW: vec4f  = vec4f(0.29, 0.67, 0.63, 0.7);
const COLOR_CCW: vec4f = vec4f(0.80, 0.56, 0.31, 0.7);

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let p = particles[instIdx];
    if ((p.flags & ALIVE_BIT) == 0u || vertIdx >= ARC_SEGMENTS) {
        out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
        out.color = vec4f(0.0);
        return out;
    }

    let angVel = derived[instIdx].angVel;
    if (abs(angVel) < MIN_ANGVEL) {
        out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
        out.color = vec4f(0.0);
        return out;
    }

    let r = particleAux[instIdx].radius;
    let cx = p.posX;
    let cy = p.posY;

    // Arc length: min(|angVel| * r, 1.0) * TWO_PI — matches CPU renderer
    let arcFrac = clamp(abs(angVel) * r, 0.0, 1.0);
    let totalAngle = arcFrac * TWO_PI;

    // Always start from top (-PI/2). Positive angVel = CW (y-down), sweep clockwise (negative direction).
    let startAngle = -HALF_PI;
    let dir = -sign(angVel);   // negate: CW on y-down canvas means decreasing angle
    let t = f32(vertIdx) / f32(ARC_SEGMENTS - 1u);
    let angle = startAngle + dir * t * totalAngle;

    // Arc sits just outside the particle radius
    let arcR = r + 0.5;
    let wx = cx + cos(angle) * arcR;
    let wy = cy + sin(angle) * arcR;

    let worldPos = vec4f(wx, wy, 0.0, 1.0);
    out.pos = camera.viewMatrix * worldPos;

    // Cyan for positive angVel (CW), orange for negative (CCW)
    out.color = select(COLOR_CCW, COLOR_CW, angVel > 0.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}
