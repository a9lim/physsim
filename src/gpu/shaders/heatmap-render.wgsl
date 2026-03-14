// ─── Heatmap Overlay Rendering ───
// Fullscreen triangle, blends gravity (slate), electric (blue/red), and Yukawa (green)
// channels using fastTanh color mapping. Same approach as field-render.wgsl.

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle (3 vertices, no vertex buffer)
@vertex
fn vsFullscreen(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vid & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vid >> 1u)) * 4.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

struct HeatmapRenderUniforms {
    cameraX: f32,
    cameraY: f32,
    cameraZoom: f32,
    canvasW: f32,
    canvasH: f32,
    viewLeft: f32,
    viewTop: f32,
    cellW: f32,
    cellH: f32,
    sensitivity: f32,    // HEATMAP_SENSITIVITY = 2
    maxAlpha: f32,       // 100/255
    isLight: u32,
    doGravity: u32,
    doCoulomb: u32,
    doYukawa: u32,
    _pad: f32,
};

// HGRID provided by generated wgslConstants block.

@group(0) @binding(0) var<storage, read> gravPotential: array<f32>;
@group(0) @binding(1) var<storage, read> elecPotential: array<f32>;
@group(0) @binding(2) var<storage, read> yukawaPotential: array<f32>;
@group(0) @binding(3) var<uniform> ru: HeatmapRenderUniforms;

// Fast tanh approximation: x/(1+|x|)
fn fastTanh(x: f32) -> f32 {
    return x / (1.0 + abs(x));
}

@fragment
fn fsHeatmapOverlay(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map screen UV to world position
    let halfW = ru.canvasW / (2.0 * ru.cameraZoom);
    let halfH = ru.canvasH / (2.0 * ru.cameraZoom);
    let worldX = ru.cameraX + (in.uv.x - 0.5) * 2.0 * halfW;
    let worldY = ru.cameraY + (in.uv.y - 0.5) * 2.0 * halfH;

    // Map world to heatmap grid coords
    let gx = (worldX - ru.viewLeft) / ru.cellW;
    let gy = (worldY - ru.viewTop) / ru.cellH;

    // Bounds check
    if (gx < 0.0 || gx >= f32(HGRID) || gy < 0.0 || gy >= f32(HGRID)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Bilinear interpolation
    let ix0 = u32(floor(gx));
    let iy0 = u32(floor(gy));
    let ix1 = min(ix0 + 1u, HGRID - 1u);
    let iy1 = min(iy0 + 1u, HGRID - 1u);
    let fx = gx - f32(ix0);
    let fy = gy - f32(iy0);

    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;

    // Gravity channel (slate from palette)
    if (ru.doGravity != 0u) {
        let g00 = gravPotential[iy0 * HGRID + ix0];
        let g10 = gravPotential[iy0 * HGRID + ix1];
        let g01 = gravPotential[iy1 * HGRID + ix0];
        let g11 = gravPotential[iy1 * HGRID + ix1];
        let gVal = mix(mix(g00, g10, fx), mix(g01, g11, fx), fy);
        let mapped = fastTanh(gVal * ru.sensitivity);
        let intensity = abs(mapped);
        r += intensity * COLOR_SLATE.r;
        g += intensity * COLOR_SLATE.g;
        b += intensity * COLOR_SLATE.b;
    }

    // Electric channel (blue for negative, red for positive — from palette)
    if (ru.doCoulomb != 0u) {
        let e00 = elecPotential[iy0 * HGRID + ix0];
        let e10 = elecPotential[iy0 * HGRID + ix1];
        let e01 = elecPotential[iy1 * HGRID + ix0];
        let e11 = elecPotential[iy1 * HGRID + ix1];
        let eVal = mix(mix(e00, e10, fx), mix(e01, e11, fx), fy);
        let mapped = fastTanh(eVal * ru.sensitivity);
        if (mapped > 0.0) {
            r += mapped * COLOR_RED.r;
        } else {
            b += abs(mapped) * COLOR_BLUE.b;
        }
    }

    // Yukawa channel (green from palette)
    if (ru.doYukawa != 0u) {
        let y00 = yukawaPotential[iy0 * HGRID + ix0];
        let y10 = yukawaPotential[iy0 * HGRID + ix1];
        let y01 = yukawaPotential[iy1 * HGRID + ix0];
        let y11 = yukawaPotential[iy1 * HGRID + ix1];
        let yVal = mix(mix(y00, y10, fx), mix(y01, y11, fx), fy);
        let mapped = fastTanh(yVal * ru.sensitivity);
        let intensity = abs(mapped);
        r += intensity * COLOR_GREEN.r;
        g += intensity * COLOR_GREEN.g;
        b += intensity * COLOR_GREEN.b;
    }

    let maxC = max(r, max(g, b));
    if (maxC < 0.001) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let alpha = min(maxC, ru.maxAlpha);
    return vec4<f32>(r * alpha, g * alpha, b * alpha, alpha);  // premultiplied
}
