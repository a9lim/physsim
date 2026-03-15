// spin-render.wgsl — Instanced spin indicator arcs + arrowheads around particles.
//
// Each particle with non-zero angular velocity gets an arc (partial circle)
// indicating rotation direction and speed. Arc length proportional to |angVel|,
// direction indicates sign (CW = positive in y-down canvas).
//
// Two sets of entry points:
//   vs_main/fs_main — triangle-strip arc ribbon (ARC_SEGMENTS * 2 vertices per instance)
//   vs_arrow/fs_arrow — triangle-list arrowhead (3 vertices per instance)

struct CameraUniforms {
    viewMatrix: mat4x4<f32>,
    invViewMatrix: mat4x4<f32>,
    zoom: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    isDarkMode: f32,
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

// Cached derived quantities (matches common.wgsl ParticleDerived)
struct ParticleDerived {
    magMoment: f32,
    angMomentum: f32,
    invMass: f32,
    radiusSq: f32,
    velX: f32,
    velY: f32,
    angVel: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read> particleAux: array<ParticleAux>;
@group(0) @binding(3) var<storage, read> derived: array<ParticleDerived>;

// Constants (FLAG_ALIVE, PI, TWO_PI, HALF_PI, COLOR_SPIN_CW, COLOR_SPIN_CCW)
// provided by generated wgslConstants block.
// Shader-specific constants:
const ARC_SEGMENTS: u32 = 32u;
const ARC_HALF_WIDTH: f32 = 0.1;  // Total width 0.2, matching CPU ctx.lineWidth
const MIN_ANGVEL: f32 = 0.01;     // Skip drawing for very slow rotation

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
};

fn clipSpinVert() -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(0.0, 0.0, -2.0, 1.0);
    out.color = vec4f(0.0);
    return out;
}

fn spinColor(angVel: f32) -> vec4f {
    let spinAlpha = select(0.8, 0.9, camera.isDarkMode > 0.5);
    let rgb = select(COLOR_SPIN_CCW, COLOR_SPIN_CW, angVel > 0.0);
    return vec4f(rgb, spinAlpha);
}

// ── Arc (triangle-strip ribbon, ARC_SEGMENTS * 2 vertices per instance) ──

@vertex
fn vs_main(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32,
) -> VertexOutput {
    let p = particles[instIdx];
    if ((p.flags & FLAG_ALIVE) == 0u || vertIdx >= ARC_SEGMENTS * 2u) { return clipSpinVert(); }

    let angVel = derived[instIdx].angVel;
    if (abs(angVel) < MIN_ANGVEL) { return clipSpinVert(); }

    let r = particleAux[instIdx].radius;
    let cx = p.posX;
    let cy = p.posY;

    // Arc length: min(|angVel| * r, 1.0) * TWO_PI — matches CPU renderer
    let arcFrac = clamp(abs(angVel) * r, 0.0, 1.0);
    let totalAngle = arcFrac * TWO_PI;

    // Always start from top (-PI/2). Positive angVel = CW (y-down) = increasing angle direction.
    let startAngle = -HALF_PI;
    let dir = sign(angVel);
    let pointIdx = vertIdx >> 1u;    // which arc sample point
    let side = vertIdx & 1u;         // 0 = inner edge, 1 = outer edge
    let t = f32(pointIdx) / f32(ARC_SEGMENTS - 1u);
    let angle = startAngle + dir * t * totalAngle;

    // Arc center radius sits just outside the particle; offset radially for ribbon width
    let arcR = r + 0.5;
    let edgeR = select(arcR - ARC_HALF_WIDTH, arcR + ARC_HALF_WIDTH, side == 1u);
    let wx = cx + cos(angle) * edgeR;
    let wy = cy + sin(angle) * edgeR;

    var out: VertexOutput;
    out.pos = camera.viewMatrix * vec4f(wx, wy, 0.0, 1.0);
    out.color = spinColor(angVel);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // Premultiplied alpha output
    return vec4f(in.color.rgb * in.color.a, in.color.a);
}

// ── Arrowhead (triangle-list, 3 vertices per instance) ──

@vertex
fn vs_arrow(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32,
) -> VertexOutput {
    let p = particles[instIdx];
    if ((p.flags & FLAG_ALIVE) == 0u || vertIdx >= 3u) { return clipSpinVert(); }

    let angVel = derived[instIdx].angVel;
    if (abs(angVel) < MIN_ANGVEL) { return clipSpinVert(); }

    let r = particleAux[instIdx].radius;
    let cx = p.posX;
    let cy = p.posY;

    let arcFrac = clamp(abs(angVel) * r, 0.0, 1.0);
    let totalAngle = arcFrac * TWO_PI;
    let dir = sign(angVel);
    let endAngle = -HALF_PI + dir * totalAngle;

    let arcR = r + 0.5;
    let ax = cx + cos(endAngle) * arcR;
    let ay = cy + sin(endAngle) * arcR;

    // Arrowhead tip points tangent to arc in sweep direction
    let sweepDir = endAngle + dir * HALF_PI;
    let spread = 0.4;

    var vx: f32; var vy: f32;
    if (vertIdx == 0u) {
        // Tip (1 unit along tangent)
        vx = ax + cos(sweepDir);
        vy = ay + sin(sweepDir);
    } else if (vertIdx == 1u) {
        // Base left (radial offset)
        vx = ax + cos(endAngle) * spread;
        vy = ay + sin(endAngle) * spread;
    } else {
        // Base right (radial offset, opposite)
        vx = ax - cos(endAngle) * spread;
        vy = ay - sin(endAngle) * spread;
    }

    var out: VertexOutput;
    out.pos = camera.viewMatrix * vec4f(vx, vy, 0.0, 1.0);
    out.color = spinColor(angVel);
    return out;
}

@fragment
fn fs_arrow(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(in.color.rgb * in.color.a, in.color.a);
}
