// trail-render.wgsl — Instanced line-strip rendering for particle trails.
//
// Each particle's trail is drawn as a series of line segments.
// Vertex shader reads from the ring buffer, applies camera transform,
// fades alpha by age. Fragment shader outputs the faded color.
//
// Draw call: drawIndirect with vertexCount = TRAIL_LEN, instanceCount = aliveCount.
// Each instance = one particle's trail.

struct CameraUniforms {
    viewMatrix: mat4x4<f32>,
    invViewMatrix: mat4x4<f32>,
    zoom: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    _pad: f32,
};

struct TrailUniforms {
    trailLen: u32,
    domainW: f32,
    domainH: f32,
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

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> trailParams: TrailUniforms;
@group(0) @binding(2) var<storage, read> trailX: array<f32>;
@group(0) @binding(3) var<storage, read> trailY: array<f32>;
@group(0) @binding(4) var<storage, read> trailWriteIdx: array<u32>;
@group(0) @binding(5) var<storage, read> trailCount: array<u32>;
@group(0) @binding(6) var<storage, read> color: array<u32>;
@group(0) @binding(7) var<storage, read> particles: array<ParticleState>;

const ALIVE_BIT: u32 = 1u;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) alpha: f32,
    @location(1) color: vec4f,
};

fn unpackRGBA(packed: u32) -> vec4f {
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec4f(r, g, b, a);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Skip dead particles
    let p = particles[instIdx];
    if ((p.flags & ALIVE_BIT) == 0u) {
        out.pos = vec4f(0.0, 0.0, -2.0, 1.0);  // clip away
        out.alpha = 0.0;
        out.color = vec4f(0.0);
        return out;
    }

    let count = trailCount[instIdx];
    if (vertIdx >= count) {
        out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
        out.alpha = 0.0;
        out.color = vec4f(0.0);
        return out;
    }

    let writeIdx = trailWriteIdx[instIdx];
    let trailLen = trailParams.trailLen;
    let base = instIdx * trailLen;

    // Read from ring buffer: oldest first
    let readIdx = (writeIdx + trailLen - count + vertIdx) % trailLen;
    let wx = trailX[base + readIdx];
    let wy = trailY[base + readIdx];

    // Wrap-break detection: if gap to previous point > half domain, hide this vertex
    if (vertIdx > 0u) {
        let prevReadIdx = (writeIdx + trailLen - count + vertIdx - 1u) % trailLen;
        let prevX = trailX[base + prevReadIdx];
        let prevY = trailY[base + prevReadIdx];
        let dx = abs(wx - prevX);
        let dy = abs(wy - prevY);
        if (dx > trailParams.domainW * 0.5 || dy > trailParams.domainH * 0.5) {
            out.pos = vec4f(0.0, 0.0, -2.0, 1.0);  // clip to break strip
            out.alpha = 0.0;
            out.color = vec4f(0.0);
            return out;
        }
    }

    // Camera transform (world → clip)
    let worldPos = vec4f(wx, wy, 0.0, 1.0);
    out.pos = camera.viewMatrix * worldPos;

    // Alpha fades from 0 (oldest) to 0.6 (newest)
    let age = f32(vertIdx) / f32(max(count - 1u, 1u));
    out.alpha = age * 0.6;

    out.color = unpackRGBA(color[instIdx]);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(in.color.rgb, in.alpha);
}
