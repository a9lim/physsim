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

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read> particleAux: array<ParticleAux>;
@group(0) @binding(3) var<storage, read> color: array<u32>;

const ALIVE_BIT: u32 = 1u;
const ARC_SEGMENTS: u32 = 32u;
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const MIN_ANGVEL: f32 = 0.01;  // Skip drawing for very slow rotation

fn unpackRGBA(packed: u32) -> vec4f {
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    return vec4f(r, g, b, 1.0);
}

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

    let aw = p.angW;
    if (abs(aw) < MIN_ANGVEL) {
        out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
        out.color = vec4f(0.0);
        return out;
    }

    let r = particleAux[instIdx].radius;
    let cx = p.posX;
    let cy = p.posY;

    // Arc length proportional to |angVel|, clamped to full circle
    let arcFrac = clamp(abs(aw) * 2.0, 0.1, 1.0);
    let totalAngle = arcFrac * TWO_PI;

    // Direction: positive angVel (CW in y-down) starts from right
    let startAngle = select(0.0, PI, aw < 0.0);
    let t = f32(vertIdx) / f32(ARC_SEGMENTS - 1u);
    let angle = startAngle + t * totalAngle * sign(aw);

    // Arc sits just outside the particle radius
    let arcR = r * 1.3;
    let wx = cx + cos(angle) * arcR;
    let wy = cy + sin(angle) * arcR;

    let worldPos = vec4f(wx, wy, 0.0, 1.0);
    out.pos = camera.viewMatrix * worldPos;

    // Use particle color with slight transparency
    out.color = unpackRGBA(color[instIdx]) * vec4f(1.0, 1.0, 1.0, 0.6);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}
