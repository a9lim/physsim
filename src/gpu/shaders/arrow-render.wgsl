// arrow-render.wgsl — Instanced force arrow rendering.
//
// Each arrow is drawn as a triangle strip (4 vertices for shaft + 3 for head = 7 total).
// One draw call per force type (11 types), each with a different color uniform.
// Instance count = alive particle count. Vertex shader reads force vector from the
// appropriate force accumulator buffer.

struct CameraUniforms {
    viewMatrix: mat4x4<f32>,
    invViewMatrix: mat4x4<f32>,
    zoom: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    _pad: f32,
};

struct ArrowUniforms {
    // Which force component to read (0=gravity, 1=coulomb, ..., 10=axion)
    forceType: u32,
    // Arrow color (RGBA)
    colorR: f32,
    colorG: f32,
    colorB: f32,
    // Scale factor for arrow length
    arrowScale: f32,
    // Minimum force magnitude to draw (skip tiny arrows)
    minMag: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> arrowParams: ArrowUniforms;
@group(0) @binding(2) var<storage, read> posX: array<f32>;
@group(0) @binding(3) var<storage, read> posY: array<f32>;
@group(0) @binding(4) var<storage, read> radius: array<f32>;
@group(0) @binding(5) var<storage, read> mass: array<f32>;
@group(0) @binding(6) var<storage, read> flags: array<u32>;
// Force buffers: packed vec4 per particle
@group(0) @binding(7) var<storage, read> forces0: array<vec4f>;  // gravity.xy, coulomb.xy
@group(0) @binding(8) var<storage, read> forces1: array<vec4f>;  // magnetic.xy, gravitomag.xy
@group(0) @binding(9) var<storage, read> forces2: array<vec4f>;  // f1pn.xy, spinCurv.xy
@group(0) @binding(10) var<storage, read> forces3: array<vec4f>; // radiation.xy, yukawa.xy
@group(0) @binding(11) var<storage, read> forces4: array<vec4f>; // external.xy, higgs.xy
@group(0) @binding(12) var<storage, read> forces5: array<vec4f>; // axion.xy, pad, pad

const ALIVE_BIT: u32 = 1u;

// Arrow geometry: shaft width, head width, head length (world units)
const SHAFT_HALF_W: f32 = 0.06;
const HEAD_HALF_W: f32 = 0.15;
const HEAD_LEN: f32 = 0.25;

fn getForceVector(idx: u32, forceType: u32) -> vec2f {
    switch (forceType) {
        case 0u:  { return forces0[idx].xy; }           // gravity
        case 1u:  { return forces0[idx].zw; }           // coulomb
        case 2u:  { return forces1[idx].xy; }           // magnetic
        case 3u:  { return forces1[idx].zw; }           // gravitomag
        case 4u:  { return forces2[idx].xy; }           // 1pn
        case 5u:  { return forces2[idx].zw; }           // spinCurv
        case 6u:  { return forces3[idx].xy; }           // radiation
        case 7u:  { return forces3[idx].zw; }           // yukawa
        case 8u:  { return forces4[idx].xy; }           // external
        case 9u:  { return forces4[idx].zw; }           // higgs
        case 10u: { return forces5[idx].xy; }           // axion
        default:  { return vec2f(0.0); }
    }
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) alpha: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.alpha = 0.8;

    if ((flags[instIdx] & ALIVE_BIT) == 0u) {
        out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let f = getForceVector(instIdx, arrowParams.forceType);
    // Force is F/m (acceleration) — scale for visibility
    let mag = length(f);
    if (mag < arrowParams.minMag) {
        out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
        return out;
    }

    let dir = f / mag;
    let perp = vec2f(-dir.y, dir.x);
    let scaledLen = mag * arrowParams.arrowScale;
    let r = radius[instIdx];

    // Arrow starts at particle edge, extends outward
    let base = vec2f(posX[instIdx], posY[instIdx]) + dir * r;
    let tip = base + dir * scaledLen;
    let headBase = tip - dir * HEAD_LEN;

    // 7 vertices: shaft (0-3 as triangle strip), then head (4-6 as triangle)
    var localPos: vec2f;
    switch (vertIdx) {
        // Shaft quad (triangle strip: 0,1,2,3)
        case 0u: { localPos = base - perp * SHAFT_HALF_W; }
        case 1u: { localPos = base + perp * SHAFT_HALF_W; }
        case 2u: { localPos = headBase - perp * SHAFT_HALF_W; }
        case 3u: { localPos = headBase + perp * SHAFT_HALF_W; }
        // Head triangle (4,5,6)
        case 4u: { localPos = headBase - perp * HEAD_HALF_W; }
        case 5u: { localPos = headBase + perp * HEAD_HALF_W; }
        case 6u: { localPos = tip; }
        default: { localPos = vec2f(0.0); }
    }

    let worldPos = vec4f(localPos, 0.0, 1.0);
    out.pos = camera.viewMatrix * worldPos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(arrowParams.colorR, arrowParams.colorG, arrowParams.colorB, in.alpha);
}
