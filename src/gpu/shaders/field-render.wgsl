// ─── Scalar Field Overlay Rendering ───
// Fullscreen triangle, fragment shader samples field grid, bilinear upscale,
// color-maps deviation from vacuum. Rendered with alpha blending, depth off.

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle (3 vertices, no vertex buffer)
@vertex
fn vsFullscreen(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    // Fullscreen triangle: vertices at (-1,-1), (3,-1), (-1,3)
    let x = f32(i32(vid & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vid >> 1u)) * 4.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

struct FieldRenderUniforms {
    // Camera transform to map UV to world coords
    cameraX: f32,
    cameraY: f32,
    cameraZoom: f32,
    canvasW: f32,
    canvasH: f32,
    domainW: f32,
    domainH: f32,
    isLight: u32,
    fieldType: u32,  // 0=higgs, 1=axion
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    // Colors (packed as vec4 RGBA floats)
    color0: vec4<f32>,  // Higgs depleted / Axion positive
    color1: vec4<f32>,  // Higgs enhanced / Axion negative
};

@group(0) @binding(0) var<storage, read> field: array<f32>;
@group(0) @binding(1) var<uniform> renderUniforms: FieldRenderUniforms;

@fragment
fn fsFieldOverlay(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map screen UV to world position
    let halfW = renderUniforms.canvasW / (2.0 * renderUniforms.cameraZoom);
    let halfH = renderUniforms.canvasH / (2.0 * renderUniforms.cameraZoom);
    let worldX = renderUniforms.cameraX + (in.uv.x - 0.5) * 2.0 * halfW;
    let worldY = renderUniforms.cameraY + (in.uv.y - 0.5) * 2.0 * halfH;

    // Map world to grid coords
    let gx = worldX / renderUniforms.domainW * f32(GRID);
    let gy = worldY / renderUniforms.domainH * f32(GRID);

    // Bounds check
    if (gx < 0.0 || gx >= f32(GRID) || gy < 0.0 || gy >= f32(GRID)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Bilinear interpolation
    let ix0 = u32(floor(gx));
    let iy0 = u32(floor(gy));
    let ix1 = min(ix0 + 1u, GRID - 1u);
    let iy1 = min(iy0 + 1u, GRID - 1u);
    let fx = gx - f32(ix0);
    let fy = gy - f32(iy0);

    let f00 = field[iy0 * GRID + ix0];
    let f10 = field[iy0 * GRID + ix1];
    let f01 = field[iy1 * GRID + ix0];
    let f11 = field[iy1 * GRID + ix1];
    let val = mix(mix(f00, f10, fx), mix(f01, f11, fx), fy);

    // Color mapping
    var deviation: f32;
    var intensity: f32;
    var color: vec3<f32>;

    if (renderUniforms.fieldType == 0u) {
        // Higgs: deviation from VEV=1
        deviation = val - 1.0;
        intensity = min(abs(deviation) * 8.0, 1.0);
        color = select(renderUniforms.color1.rgb, renderUniforms.color0.rgb, deviation < 0.0);
    } else {
        // Axion: deviation from vacuum=0
        deviation = val;
        intensity = min(abs(deviation) * 4.0, 1.0);
        color = select(renderUniforms.color1.rgb, renderUniforms.color0.rgb, deviation > 0.0);
    }

    let maxAlpha = select(80.0 / 255.0, 60.0 / 255.0, renderUniforms.isLight != 0u);
    let alpha = intensity * maxAlpha * 0.6;  // * 0.6 matches CPU globalAlpha

    return vec4<f32>(color * alpha, alpha);  // premultiplied alpha
}
