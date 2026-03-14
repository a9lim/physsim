// Instanced particle rendering — vertex + fragment.
// Each instance = one particle. Renders as a screen-aligned quad, fragment discards outside circle.

struct CameraUniforms {
    viewMatrix: mat4x4<f32>,
    invViewMatrix: mat4x4<f32>,
    zoom: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    isDarkMode: f32,   // 0.0 = light, 1.0 = dark (was _pad)
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

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,       // -1..+1 within quad (extended for glow)
    @location(1) particleColor: vec4<f32>,
    @location(2) softness: f32,       // for edge falloff (= 1/pixelRadius)
    @location(3) isDark: f32,         // 0 or 1, passed through from camera uniform
    @location(4) glowIntensity: f32,  // charge-dependent glow: 0.1 (neutral) to 1.0 (high charge)
};

// Quad vertices: 2 triangles forming a [-1,1] square
const QUAD_POS = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
);

// In dark mode, we extend the quad by this factor so the glow halo fits inside.
// 1.0 = no extension (light mode), 1.8 = 80% extra radius for glow halo (dark mode).
const DARK_QUAD_SCALE: f32 = 1.8;

@vertex
fn vs_main(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32,
) -> VertexOut {
    var out: VertexOut;

    let p = particles[instanceIndex];
    let flag = p.flags;
    // Skip non-alive particles by pushing off-screen
    if ((flag & 1u) == 0u) {
        out.position = vec4(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let px = p.posX;
    let py = p.posY;
    let r = particleAux[instanceIndex].radius;

    // Transform world position to clip space via camera
    let worldPos = camera.viewMatrix * vec4(px, py, 0.0, 1.0);

    // In dark mode, expand the quad to accommodate the glow halo
    let isDark = camera.isDarkMode;
    let quadScale = 1.0 + isDark * (DARK_QUAD_SCALE - 1.0);

    // Quad corner offset in pixels, then to clip space
    let quadCorner = QUAD_POS[vertexIndex];
    let pixelRadius = max(r * camera.zoom, 2.0); // minimum 2px
    let offsetPx = quadCorner * pixelRadius * quadScale;

    let clipX = worldPos.x + offsetPx.x * 2.0 / camera.canvasWidth;
    let clipY = worldPos.y + offsetPx.y * 2.0 / camera.canvasHeight;

    out.position = vec4(clipX, clipY, 0.0, 1.0);
    // UV stays in [-1,1] relative to the *un-extended* quad so dist==1 is the circle edge.
    // We scale UV back: in dark mode quadCorner maps to uv = quadCorner * quadScale.
    out.uv = quadCorner * quadScale;

    // Unpack RGBA from u32 (ABGR packed)
    let packed = color[instanceIndex];
    out.particleColor = vec4(
        f32(packed & 0xFFu) / 255.0,
        f32((packed >> 8u) & 0xFFu) / 255.0,
        f32((packed >> 16u) & 0xFFu) / 255.0,
        f32((packed >> 24u) & 0xFFu) / 255.0,
    );

    out.softness = 1.0 / max(pixelRadius, 1.0);
    out.isDark = isDark;
    // Charge-dependent glow: neutral=0.1, scales up to 1.0 at |charge|=5
    out.glowIntensity = clamp(abs(p.charge) / 5.0, 0.1, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dist = length(in.uv);

    // Discard fully beyond glow range (slightly past DARK_QUAD_SCALE)
    if (dist > DARK_QUAD_SCALE + 0.05) { discard; }

    // Solid circle with 1-pixel anti-aliased edge
    let circleAlpha = smoothstep(1.0, 1.0 - in.softness * 2.0, dist) * in.particleColor.a;

    // Dark mode glow halo: exponential falloff beyond the circle edge.
    // The halo only contributes outside the solid circle radius.
    // glowDist = how far past the edge we are (0 at edge, 1 at DARK_QUAD_SCALE).
    let glowRange = DARK_QUAD_SCALE - 1.0;
    let glowDist = clamp((dist - 1.0) / glowRange, 0.0, 1.0);
    // Charge-dependent glow: higher charge = slower decay = wider glow (matches CPU shadowBlur buckets)
    let decayRate = 6.0 - 4.0 * in.glowIntensity;  // neutral: 5.6, high charge: 2.0
    let glowAlpha = exp(-glowDist * decayRate) * (1.0 - glowDist) * in.particleColor.a * in.isDark * in.glowIntensity;

    let totalAlpha = clamp(circleAlpha + glowAlpha * 0.55, 0.0, 1.0);

    // Premultiplied alpha output (required by alphaMode: 'premultiplied').
    // In additive dark mode the blend is: src.rgb * 1 + dst.rgb * 1
    // so we output premultiplied color and the pipeline blend handles the rest.
    return vec4(in.particleColor.rgb * totalAlpha, totalAlpha);
}
