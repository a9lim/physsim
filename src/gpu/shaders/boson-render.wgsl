// ─── Boson Rendering ───
// Instanced point sprites for photons and pions.
// Photons: yellow (EM) / red (grav), alpha fades over lifetime.
// Pions: green, constant alpha.
// Dark mode: exponential glow halo (matches CPU shadowBlur = 12).

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
    @location(2) isDark: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Photon pool (packed, read-only for rendering)
@group(1) @binding(0) var<storage, read> photonPool: array<Photon>;
@group(1) @binding(1) var<storage, read> phCount: array<u32>;

// Pion pool (packed, read-only for rendering)
@group(2) @binding(0) var<storage, read> pionPool: array<Pion>;
@group(2) @binding(1) var<storage, read> piCount: array<u32>;

// In dark mode, extend the quad for glow halo (matches particle.wgsl)
const DARK_QUAD_SCALE: f32 = 1.8;
// Glow decay rate for bosons (CPU shadowBlur = 12; roughly maps to medium decay)
const BOSON_GLOW_INTENSITY: f32 = 0.5;
const BOSON_GLOW_DECAY: f32 = 4.0;  // 6.0 - 4.0 * 0.5

// Quad vertex positions (triangle strip: 4 verts -> 2 triangles)
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

    let rawAlpha = max(0.0, 1.0 - ph.lifetime / PHOTON_LIFETIME);
    // Alpha scale: 0.6 for light, 0.8 for dark (matches CPU renderer)
    let isDark = camera.isDarkMode;
    let alphaScale = select(0.6, 0.8, isDark > 0.5);
    let alpha = rawAlpha * alphaScale;

    // Expand quad in dark mode for glow halo
    let quadScale = 1.0 + isDark * (DARK_QUAD_SCALE - 1.0);

    // Point sprite quad — energy-proportional size in world space (matches CPU renderer)
    let qOff = quadOffset(vertexIdx);
    let worldRadius = clamp(0.25 + 2.0 * ph.energy, 0.25, 5.0);
    let pixelRadius = max(worldRadius * camera.zoom, 2.0);
    let clipPos = camera.viewMatrix * vec4f(ph.posX, ph.posY, 0.0, 1.0);

    out.position = clipPos + vec4f(qOff * pixelRadius * quadScale * 2.0 / vec2f(camera.canvasWidth, camera.canvasHeight), 0.0, 0.0);

    // Color: yellow for EM, red for grav (from palette)
    let isGrav = (ph.flags & 2u) != 0u;
    out.color = select(vec4f(COLOR_YELLOW, alpha), vec4f(COLOR_RED, alpha), isGrav);
    // UV in [-1,1] scaled by quadScale so dist==1 is the circle edge
    out.uv = qOff * quadScale;
    out.isDark = isDark;
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

    let isDark = camera.isDarkMode;
    let quadScale = 1.0 + isDark * (DARK_QUAD_SCALE - 1.0);

    // Energy-proportional size in world space (matches CPU renderer)
    let qOff = quadOffset(vertexIdx);
    let worldRadius = clamp(0.25 + 2.0 * pi.energy, 0.25, 5.0);
    let pixelRadius = max(worldRadius * camera.zoom, 2.0);
    let clipPos = camera.viewMatrix * vec4f(pi.posX, pi.posY, 0.0, 1.0);

    out.position = clipPos + vec4f(qOff * pixelRadius * quadScale * 2.0 / vec2f(camera.canvasWidth, camera.canvasHeight), 0.0, 0.0);

    // Pion: green (from palette), theme-dependent alpha (0.7 light / 0.9 dark)
    let pionAlpha = select(0.7, 0.9, isDark > 0.5);
    out.color = vec4f(COLOR_GREEN, pionAlpha);
    out.uv = qOff * quadScale;
    out.isDark = isDark;
    return out;
}

@fragment
fn fragmentBoson(
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
    @location(2) isDark: f32,
) -> @location(0) vec4f {
    let dist = length(uv);

    // Discard beyond glow range
    if (dist > DARK_QUAD_SCALE + 0.05) { discard; }

    // Sharp circle edge (matches CPU ctx.arc fill)
    let circleAlpha = select(0.0, 1.0, dist <= 1.0) * color.a;

    // Dark mode glow halo: exponential falloff beyond circle edge (matches CPU shadowBlur=12)
    let glowRange = DARK_QUAD_SCALE - 1.0;
    let glowDist = clamp((dist - 1.0) / glowRange, 0.0, 1.0);
    let glowAlpha = exp(-glowDist * BOSON_GLOW_DECAY) * (1.0 - glowDist) * color.a * isDark * BOSON_GLOW_INTENSITY;

    let totalAlpha = clamp(circleAlpha + glowAlpha * 0.55, 0.0, 1.0);

    // Premultiplied alpha output (matches particle.wgsl and canvas alphaMode: 'premultiplied')
    return vec4f(color.rgb * totalAlpha, totalAlpha);
}
