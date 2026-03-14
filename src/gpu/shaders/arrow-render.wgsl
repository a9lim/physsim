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

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState_AR {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

// Packed auxiliary struct (matches common.wgsl ParticleAux)
struct ParticleAux_AR {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
};

// Packed force struct (mirrors common.wgsl AllForces)
struct AllForces_AR {
    f0: vec4<f32>,
    f1: vec4<f32>,
    f2: vec4<f32>,
    f3: vec4<f32>,
    f4: vec4<f32>,
    f5: vec4<f32>,
    torques: vec4<f32>,
    bFields: vec4<f32>,
    bFieldGrads: vec4<f32>,
    totalForce: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> arrowParams: ArrowUniforms;
@group(0) @binding(2) var<storage, read> particles: array<ParticleState_AR>;
@group(0) @binding(3) var<storage, read> particleAux: array<ParticleAux_AR>;
@group(0) @binding(4) var<storage, read> allForces: array<AllForces_AR>;

const ALIVE_BIT: u32 = 1u;

// Arrow geometry: shaft width, head width, head length (world units)
const SHAFT_HALF_W: f32 = 0.06;
const HEAD_HALF_W: f32 = 0.15;
const HEAD_LEN: f32 = 0.25;

fn getForceVector(idx: u32, forceType: u32) -> vec2f {
    let af = allForces[idx];
    switch (forceType) {
        case 0u:  { return af.f0.xy; }           // gravity
        case 1u:  { return af.f0.zw; }           // coulomb
        case 2u:  { return af.f1.xy; }           // magnetic
        case 3u:  { return af.f1.zw; }           // gravitomag
        case 4u:  { return af.f2.xy; }           // 1pn
        case 5u:  { return af.f2.zw; }           // spinCurv
        case 6u:  { return af.f3.xy; }           // radiation
        case 7u:  { return af.f3.zw; }           // yukawa
        case 8u:  { return af.f4.xy; }           // external
        case 9u:  { return af.f4.zw; }           // higgs
        case 10u: { return af.f5.xy; }           // axion
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

    let p = particles[instIdx];
    if ((p.flags & ALIVE_BIT) == 0u) {
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
    let r = particleAux[instIdx].radius;

    // Arrow starts at particle edge, extends outward
    let base = vec2f(p.posX, p.posY) + dir * r;
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
