// trail-render.wgsl — Instanced triangle-strip rendering for particle trails.
//
// Each particle's trail is drawn as a ribbon (triangle strip) with width
// proportional to particle radius (0.5 * radius), matching CPU renderer.
//
// Draw call: drawIndirect with vertexCount = TRAIL_LEN * 2, instanceCount = aliveCount.
// Each instance = one particle's trail. Each trail point → 2 vertices (left/right).

struct CameraUniforms {
    viewMatrix: mat4x4<f32>,
    invViewMatrix: mat4x4<f32>,
    zoom: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    isDarkMode: f32,
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

// Packed auxiliary struct (matches common.wgsl ParticleAux)
struct ParticleAux {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> trailParams: TrailUniforms;
@group(0) @binding(2) var<storage, read> trailX: array<f32>;
@group(0) @binding(3) var<storage, read> trailY: array<f32>;
@group(0) @binding(4) var<storage, read> trailWriteIdx: array<u32>;
@group(0) @binding(5) var<storage, read> trailCount: array<u32>;
@group(0) @binding(6) var<storage, read> color: array<u32>;
@group(0) @binding(7) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(8) var<storage, read> particleAux: array<ParticleAux>;

// FLAG_ALIVE provided by generated wgslConstants block.

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

fn clipVert() -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
    out.alpha = 0.0;
    out.color = vec4f(0.0);
    return out;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32,
) -> VertexOutput {
    // Skip dead particles
    let p = particles[instIdx];
    if ((p.flags & FLAG_ALIVE) == 0u) { return clipVert(); }

    let pointIdx = vertIdx >> 1u;  // which trail point
    let side = vertIdx & 1u;       // 0 = left, 1 = right

    let count = trailCount[instIdx];
    if (pointIdx >= count || count < 2u) { return clipVert(); }

    let writeIdx = trailWriteIdx[instIdx];
    let trailLen = trailParams.trailLen;
    let base = instIdx * trailLen;

    // Read current point from ring buffer (oldest first)
    let readIdx = (writeIdx + trailLen - count + pointIdx) % trailLen;
    let wx = trailX[base + readIdx];
    let wy = trailY[base + readIdx];

    // Read prev/next points for direction + wrap detection
    let hasPrev = pointIdx > 0u;
    let hasNext = pointIdx < count - 1u;
    let halfW = trailParams.domainW * 0.5;
    let halfH = trailParams.domainH * 0.5;

    var prevX = wx; var prevY = wy;
    var nextX = wx; var nextY = wy;
    var wrapPrev = false;
    var wrapNext = false;

    if (hasPrev) {
        let prevRI = (writeIdx + trailLen - count + pointIdx - 1u) % trailLen;
        prevX = trailX[base + prevRI];
        prevY = trailY[base + prevRI];
        wrapPrev = abs(wx - prevX) > halfW || abs(wy - prevY) > halfH;
    }
    if (hasNext) {
        let nextRI = (writeIdx + trailLen - count + pointIdx + 1u) % trailLen;
        nextX = trailX[base + nextRI];
        nextY = trailY[base + nextRI];
        wrapNext = abs(nextX - wx) > halfW || abs(nextY - wy) > halfH;
    }

    // Clip if any adjacent segment wraps (prevents cross-screen triangles)
    if (wrapPrev || wrapNext) { return clipVert(); }

    // Compute tangent direction (average of prev→curr and curr→next for smooth joins)
    var dx: f32; var dy: f32;
    if (hasPrev && hasNext) {
        dx = nextX - prevX;
        dy = nextY - prevY;
    } else if (hasNext) {
        dx = nextX - wx;
        dy = nextY - wy;
    } else {
        dx = wx - prevX;
        dy = wy - prevY;
    }

    // Perpendicular offset: halfWidth = 0.25 * radius (total width = 0.5 * radius)
    let halfWidth = 0.25 * particleAux[instIdx].radius;
    let len = sqrt(dx * dx + dy * dy);
    var perpX: f32; var perpY: f32;
    if (len > 0.0001) {
        let invLen = halfWidth / len;
        perpX = -dy * invLen;
        perpY = dx * invLen;
    } else {
        perpX = halfWidth;
        perpY = 0.0;
    }

    // Offset to left (-1) or right (+1) side
    let s = select(-1.0, 1.0, side == 1u);
    let finalX = wx + perpX * s;
    let finalY = wy + perpY * s;

    var out: VertexOutput;
    out.pos = camera.viewMatrix * vec4f(finalX, finalY, 0.0, 1.0);

    // Alpha fades from 0 (oldest) to alphaMax (newest): 0.7 light / 0.9 dark
    let age = f32(pointIdx) / f32(max(count - 1u, 1u));
    let alphaMax = select(0.7, 0.9, camera.isDarkMode > 0.5);
    out.alpha = age * alphaMax;

    out.color = unpackRGBA(color[instIdx]);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // Premultiplied alpha output (required by alphaMode: 'premultiplied')
    return vec4f(in.color.rgb * in.alpha, in.alpha);
}
