// ─── Boson Rendering ───
// Instanced point sprites for photons and pions.
// Photons: yellow (EM) / red (grav), alpha fades over lifetime.
// Pions: green, constant alpha.

struct CameraUniforms {
    viewMatrix: mat4x4f,
    invViewMatrix: mat4x4f,
    zoom: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    _pad: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Photon pool (read-only for rendering)
@group(1) @binding(0) var<storage, read> phPosX: array<f32>;
@group(1) @binding(1) var<storage, read> phPosY: array<f32>;
@group(1) @binding(2) var<storage, read> phAge: array<u32>;
@group(1) @binding(3) var<storage, read> phFlags: array<u32>;
@group(1) @binding(4) var<storage, read> phCount: u32;

// Pion pool (read-only for rendering)
@group(2) @binding(0) var<storage, read> piPosX: array<f32>;
@group(2) @binding(1) var<storage, read> piPosY: array<f32>;
@group(2) @binding(2) var<storage, read> piAge: array<u32>;
@group(2) @binding(3) var<storage, read> piFlags: array<u32>;
@group(2) @binding(4) var<storage, read> piCount: u32;

// Quad vertex positions (triangle strip: 4 verts -> 2 triangles via index buffer)
fn quadOffset(vertexIdx: u32) -> vec2f {
    // 0: (-1,-1), 1: (1,-1), 2: (-1,1), 3: (1,1)
    let x = select(-1.0, 1.0, (vertexIdx & 1u) != 0u);
    let y = select(-1.0, 1.0, (vertexIdx & 2u) != 0u);
    return vec2f(x, y);
}

@vertex
fn vertexPhoton(
    @builtin(instance_index) instanceIdx: u32,
    @builtin(vertex_index) vertexIdx: u32,
) -> VertexOutput {
    var out: VertexOutput;

    if (instanceIdx >= phCount) {
        out.position = vec4f(0.0, 0.0, -2.0, 1.0); // cull
        return out;
    }
    if ((phFlags[instanceIdx] & 1u) == 0u) {
        out.position = vec4f(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let px = phPosX[instanceIdx];
    let py = phPosY[instanceIdx];
    let age = f32(phAge[instanceIdx]);
    let maxAge = 256.0 * 128.0; // PHOTON_LIFETIME * PHYSICS_DT_INV approx
    let alpha = max(0.0, 1.0 - age / maxAge);

    // Point sprite quad
    let qOff = quadOffset(vertexIdx);
    let spriteSize = 3.0; // pixels
    let worldPos = vec2f(px, py);
    let clipPos = camera.viewMatrix * vec4f(worldPos, 0.0, 1.0);

    out.position = clipPos + vec4f(qOff * spriteSize / vec2f(camera.canvasWidth, camera.canvasHeight), 0.0, 0.0);

    // Color: yellow for EM, red for grav
    let isGrav = (phFlags[instanceIdx] & 2u) != 0u;
    if (isGrav) {
        out.color = vec4f(1.0, 0.2, 0.1, alpha);
    } else {
        out.color = vec4f(1.0, 0.9, 0.2, alpha);
    }
    out.uv = qOff * 0.5 + 0.5;
    return out;
}

@vertex
fn vertexPion(
    @builtin(instance_index) instanceIdx: u32,
    @builtin(vertex_index) vertexIdx: u32,
) -> VertexOutput {
    var out: VertexOutput;

    if (instanceIdx >= piCount) {
        out.position = vec4f(0.0, 0.0, -2.0, 1.0);
        return out;
    }
    if ((piFlags[instanceIdx] & 1u) == 0u) {
        out.position = vec4f(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let px = piPosX[instanceIdx];
    let py = piPosY[instanceIdx];

    let qOff = quadOffset(vertexIdx);
    let spriteSize = 4.0; // slightly larger than photons
    let worldPos = vec2f(px, py);
    let clipPos = camera.viewMatrix * vec4f(worldPos, 0.0, 1.0);

    out.position = clipPos + vec4f(qOff * spriteSize / vec2f(camera.canvasWidth, camera.canvasHeight), 0.0, 0.0);

    // Pion: green, constant alpha
    out.color = vec4f(0.31, 0.6, 0.47, 0.9); // ~_PALETTE.extended.green
    out.uv = qOff * 0.5 + 0.5;
    return out;
}

@fragment
fn fragmentBoson(@location(0) color: vec4f, @location(1) uv: vec2f) -> @location(0) vec4f {
    let dist = length(uv - 0.5) * 2.0;
    if (dist > 1.0) { discard; }
    let falloff = 1.0 - dist * dist;
    return vec4f(color.rgb, color.a * falloff);
}
