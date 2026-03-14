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
    isDarkMode: f32,
};

// Packed photon struct (matches common.wgsl Photon)
struct Photon {
    posX: f32, posY: f32,
    velX: f32, velY: f32,
    energy: f32,
    emitterId: u32, lifetime: f32, flags: u32,
};

// Packed pion struct (matches common.wgsl Pion)
struct Pion {
    posX: f32, posY: f32,
    wX: f32, wY: f32,
    mass: f32, charge: i32, energy: f32,
    emitterId: u32, age: u32, flags: u32,
    _pad0: u32, _pad1: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Photon pool (packed, read-only for rendering)
@group(1) @binding(0) var<storage, read> photonPool: array<Photon>;
@group(1) @binding(1) var<storage, read> phCount: array<u32>;

// Pion pool (packed, read-only for rendering)
@group(2) @binding(0) var<storage, read> pionPool: array<Pion>;
@group(2) @binding(1) var<storage, read> piCount: array<u32>;

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

    if (instanceIdx >= phCount[0]) {
        out.position = vec4f(0.0, 0.0, -2.0, 1.0); // cull
        return out;
    }

    let ph = photonPool[instanceIdx];
    if ((ph.flags & 1u) == 0u) {
        out.position = vec4f(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let px = ph.posX;
    let py = ph.posY;
    let rawAlpha = max(0.0, 1.0 - ph.lifetime / 256.0); // PHOTON_LIFETIME = 256 time units
    // Alpha scale: 0.6 for light, 0.8 for dark (matches CPU renderer bucket alpha)
    let alphaScale = select(0.6, 0.8, camera.isDarkMode > 0.5);
    let alpha = rawAlpha * alphaScale;

    // Point sprite quad — energy-proportional size in world space (matches CPU renderer)
    let qOff = quadOffset(vertexIdx);
    let worldRadius = clamp(0.25 + 2.0 * ph.energy, 0.25, 5.0);
    let pixelRadius = worldRadius * camera.zoom;
    let worldPos = vec2f(px, py);
    let clipPos = camera.viewMatrix * vec4f(worldPos, 0.0, 1.0);

    out.position = clipPos + vec4f(qOff * pixelRadius * 2.0 / vec2f(camera.canvasWidth, camera.canvasHeight), 0.0, 0.0);

    // Color: yellow for EM (#CCA84C), red for grav (#C05048)
    let isGrav = (ph.flags & 2u) != 0u;
    if (isGrav) {
        out.color = vec4f(0.753, 0.314, 0.282, alpha);
    } else {
        out.color = vec4f(0.800, 0.659, 0.298, alpha);
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

    if (instanceIdx >= piCount[0]) {
        out.position = vec4f(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let pi = pionPool[instanceIdx];
    if ((pi.flags & 1u) == 0u) {
        out.position = vec4f(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let px = pi.posX;
    let py = pi.posY;

    // Energy-proportional size in world space (matches CPU renderer)
    let qOff = quadOffset(vertexIdx);
    let worldRadius = clamp(0.25 + 2.0 * pi.energy, 0.25, 5.0);
    let pixelRadius = worldRadius * camera.zoom;
    let worldPos = vec2f(px, py);
    let clipPos = camera.viewMatrix * vec4f(worldPos, 0.0, 1.0);

    out.position = clipPos + vec4f(qOff * pixelRadius * 2.0 / vec2f(camera.canvasWidth, camera.canvasHeight), 0.0, 0.0);

    // Pion: green, theme-dependent alpha (0.7 light / 0.9 dark)
    let pionAlpha = select(0.7, 0.9, camera.isDarkMode > 0.5);
    out.color = vec4f(0.31, 0.6, 0.47, pionAlpha); // ~_PALETTE.extended.green
    out.uv = qOff * 0.5 + 0.5;
    return out;
}

@fragment
fn fragmentBoson(@location(0) color: vec4f, @location(1) uv: vec2f) -> @location(0) vec4f {
    let dist = length(uv - 0.5) * 2.0;
    if (dist > 1.0) { discard; }
    let falloff = 1.0 - dist * dist;
    let finalAlpha = color.a * falloff;
    // Premultiplied alpha output (matches particle.wgsl and canvas alphaMode: 'premultiplied')
    return vec4f(color.rgb * finalAlpha, finalAlpha);
}
