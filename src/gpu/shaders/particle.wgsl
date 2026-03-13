// Instanced particle rendering — vertex + fragment.
// Each instance = one particle. Renders as a screen-aligned quad, fragment discards outside circle.

struct CameraUniforms {
    viewMatrix: mat4x4<f32>,
    invViewMatrix: mat4x4<f32>,
    zoom: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> posX: array<f32>;
@group(0) @binding(2) var<storage, read> posY: array<f32>;
@group(0) @binding(3) var<storage, read> radius: array<f32>;
@group(0) @binding(4) var<storage, read> color: array<u32>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,       // -1..+1 within quad
    @location(1) particleColor: vec4<f32>,
    @location(2) softness: f32,       // for edge falloff
};

// Quad vertices: 2 triangles forming a [-1,1] square
const QUAD_POS = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32,
) -> VertexOut {
    var out: VertexOut;

    let flag = flags[instanceIndex];
    // Skip non-alive particles by pushing off-screen
    if ((flag & 1u) == 0u) {
        out.position = vec4(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let px = posX[instanceIndex];
    let py = posY[instanceIndex];
    let r = radius[instanceIndex];

    // Transform world position to clip space via camera
    let worldPos = camera.viewMatrix * vec4(px, py, 0.0, 1.0);

    // Quad corner offset in pixels, then to clip space
    let quadCorner = QUAD_POS[vertexIndex];
    let pixelRadius = max(r * camera.zoom, 2.0); // minimum 2px
    let offsetPx = quadCorner * pixelRadius;

    let clipX = worldPos.x + offsetPx.x * 2.0 / camera.canvasWidth;
    let clipY = worldPos.y + offsetPx.y * 2.0 / camera.canvasHeight;

    out.position = vec4(clipX, clipY, 0.0, 1.0);
    out.uv = quadCorner;

    // Unpack RGBA from u32 (ABGR packed)
    let packed = color[instanceIndex];
    out.particleColor = vec4(
        f32(packed & 0xFFu) / 255.0,
        f32((packed >> 8u) & 0xFFu) / 255.0,
        f32((packed >> 16u) & 0xFFu) / 255.0,
        f32((packed >> 24u) & 0xFFu) / 255.0,
    );

    out.softness = 1.0 / max(pixelRadius, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Circle with soft edge
    let dist = length(in.uv);
    if (dist > 1.0) { discard; }

    // Smooth falloff at edge (replaces Canvas2D shadowBlur)
    let alpha = smoothstep(1.0, 0.7, dist) * in.particleColor.a;

    // Premultiplied alpha output (required by alphaMode: 'premultiplied')
    return vec4(in.particleColor.rgb * alpha, alpha);
}
