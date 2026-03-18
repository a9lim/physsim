// torque-render.wgsl — Instanced torque arc rendering around particles.
//
// Each particle with non-zero torque gets an arc (partial circle) at a radius
// offset from the particle, with an arrowhead indicating direction.
// Modeled on spin-render.wgsl, but reads torque data from AllForces buffer.
//
// Four torque types (drawn in separate passes via uniform):
//   0: spinOrbit  (torques.x)  offset 2.5
//   1: frameDrag  (torques.y)  offset 2.0
//   2: tidal      (torques.z)  offset 1.5
//   3: contact    (torques.w)  offset 1.0
//  11: total      (sum of all) offset 3.0
//
// Two sets of entry points:
//   vs_main/fs_main — triangle-strip arc ribbon (ARC_SEGMENTS * 2 vertices)
//   vs_arrow/fs_arrow — triangle-list arrowhead (3 vertices)

struct CameraUniforms {
    viewMatrix: mat4x4<f32>,
    invViewMatrix: mat4x4<f32>,
    zoom: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    isDarkMode: f32,
};

struct TorqueUniforms {
    torqueType: u32,       // 0-3 for components, 11 for total
    colorR: f32,
    colorG: f32,
    colorB: f32,
    torqueScale: f32,      // FORCE_VECTOR_SCALE / INERTIA_K = 640
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

// Struct definitions (ParticleState, ParticleAux, AllForces) provided by shared-structs.wgsl.

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> torqueParams: TorqueUniforms;
@group(0) @binding(2) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(3) var<storage, read> particleAux: array<ParticleAux>;
@group(0) @binding(4) var<storage, read> allForces: array<AllForces>;

// Constants (FLAG_ALIVE, INERTIA_K, PI, TWO_PI, HALF_PI) from wgslConstants block.
const ARC_SEGMENTS: u32 = 32u;
const ARC_HALF_WIDTH: f32 = 0.125;  // matches CPU lineWidth 0.25 / 2
const MIN_TORQUE: f32 = 1e-8;

fn getTorqueValue(idx: u32, torqueType: u32) -> f32 {
    let t = allForces[idx].torques;
    switch (torqueType) {
        case 0u:  { return t.x; }  // spinOrbit
        case 1u:  { return t.y; }  // frameDrag
        case 2u:  { return t.z; }  // tidal
        case 3u:  { return t.w; }  // contact
        case 11u: { return t.x + t.y + t.z + t.w; }  // total
        default:  { return 0.0; }
    }
}

fn getTorqueOffset(torqueType: u32) -> f32 {
    switch (torqueType) {
        case 0u:  { return 2.5; }  // spinOrbit
        case 1u:  { return 2.0; }  // frameDrag
        case 2u:  { return 1.5; }  // tidal
        case 3u:  { return 1.0; }  // contact
        case 11u: { return 3.0; }  // total
        default:  { return 1.0; }
    }
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
};

fn clipTorqueVert() -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
    out.color = vec4f(0.0);
    return out;
}

// ── Arc (triangle-strip ribbon, ARC_SEGMENTS * 2 vertices per instance) ──

@vertex
fn vs_main(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32,
) -> VertexOutput {
    let p = particles[instIdx];
    if ((p.flags & FLAG_ALIVE) == 0u || vertIdx >= ARC_SEGMENTS * 2u) { return clipTorqueVert(); }

    var val = getTorqueValue(instIdx, torqueParams.torqueType);
    if (abs(val) < MIN_TORQUE) { return clipTorqueVert(); }

    let r = particleAux[instIdx].radius;
    let cx = p.posX;
    let cy = p.posY;

    // Convert torque to angular acceleration: val / (I) = val / (INERTIA_K * m * r²)
    val = val / (INERTIA_K * p.mass * r * r);

    let offset = getTorqueOffset(torqueParams.torqueType);
    let ringRadius = r + offset;
    let sweep = clamp(torqueParams.torqueScale * abs(val), 0.0, TWO_PI);

    // Direction: positive torque → CW arc (negative sweep in y-down coords)
    let dir = select(1.0, -1.0, val > 0.0);
    let startAngle = -HALF_PI;

    let pointIdx = vertIdx >> 1u;
    let side = vertIdx & 1u;
    let t = f32(pointIdx) / f32(ARC_SEGMENTS - 1u);
    let angle = startAngle - dir * t * sweep;

    let edgeR = select(ringRadius - ARC_HALF_WIDTH, ringRadius + ARC_HALF_WIDTH, side == 1u);
    let wx = cx + cos(angle) * edgeR;
    let wy = cy + sin(angle) * edgeR;

    var out: VertexOutput;
    out.pos = camera.viewMatrix * vec4f(wx, wy, 0.0, 1.0);
    out.color = vec4f(torqueParams.colorR, torqueParams.colorG, torqueParams.colorB, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(in.color.rgb * in.color.a, in.color.a);
}

// ── Arrowhead (triangle-list, 3 vertices per instance) ──

@vertex
fn vs_arrow(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32,
) -> VertexOutput {
    let p = particles[instIdx];
    if ((p.flags & FLAG_ALIVE) == 0u || vertIdx >= 3u) { return clipTorqueVert(); }

    var val = getTorqueValue(instIdx, torqueParams.torqueType);
    if (abs(val) < MIN_TORQUE) { return clipTorqueVert(); }

    let r = particleAux[instIdx].radius;
    let cx = p.posX;
    let cy = p.posY;

    val = val / (INERTIA_K * p.mass * r * r);

    let offset = getTorqueOffset(torqueParams.torqueType);
    let ringRadius = r + offset;
    let sweep = clamp(torqueParams.torqueScale * abs(val), 0.0, TWO_PI);

    // Skip arrowhead if arc is too short
    if (sweep * ringRadius < 0.5) { return clipTorqueVert(); }

    let dir = select(1.0, -1.0, val > 0.0);
    let endAngle = -HALF_PI - dir * sweep;

    let ax = cx + cos(endAngle) * ringRadius;
    let ay = cy + sin(endAngle) * ringRadius;

    // Arrowhead tip points tangent to arc in sweep direction
    let sweepDir = endAngle - dir * HALF_PI;
    let h: f32 = 0.5;
    let spread: f32 = h * 0.4;

    var vx: f32; var vy: f32;
    if (vertIdx == 0u) {
        // Tip
        vx = ax + cos(sweepDir) * h;
        vy = ay + sin(sweepDir) * h;
    } else if (vertIdx == 1u) {
        // Base left (radial offset)
        vx = ax + cos(endAngle) * spread;
        vy = ay + sin(endAngle) * spread;
    } else {
        // Base right
        vx = ax - cos(endAngle) * spread;
        vy = ay - sin(endAngle) * spread;
    }

    var out: VertexOutput;
    out.pos = camera.viewMatrix * vec4f(vx, vy, 0.0, 1.0);
    out.color = vec4f(torqueParams.colorR, torqueParams.colorG, torqueParams.colorB, 1.0);
    return out;
}

@fragment
fn fs_arrow(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(in.color.rgb * in.color.a, in.color.a);
}
