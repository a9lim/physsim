// ─── Particle-Field Gravity ───
// F = -m · ∇Φ(x), where Φ is the gravitational potential from field energy density.
// PQS interpolation of pre-computed grid gradients: O(N × 16).
// Dispatched once per active field (Higgs, Axion) when fieldGravEnabled.

struct ParticleState_FG {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct AllForces_FG {
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
    jerk: vec2<f32>,
};

struct FGUniforms {
    domainW: f32,
    domainH: f32,
    aliveCount: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> u: FGUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState_FG>;
@group(0) @binding(2) var<storage, read_write> allForces: array<AllForces_FG>;

@group(1) @binding(0) var<storage, read> sgGradX: array<f32>;
@group(1) @binding(1) var<storage, read> sgGradY: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= u.aliveCount) { return; }
    let flag = particles[idx].flags;
    if ((flag & 1u) == 0u) { return; } // FLAG_ALIVE

    let m = particles[idx].mass;
    if (m < EPSILON) { return; }

    let px = particles[idx].posX;
    let py = particles[idx].posY;
    let cellW = u.domainW / f32(GRID);
    let cellH = u.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }
    let invCellW = 1.0 / cellW;
    let invCellH = 1.0 / cellH;

    // PQS weights at particle position
    let gx = px * invCellW - 0.5;
    let gy = py * invCellH - 0.5;
    let ix = i32(floor(gx));
    let iy = i32(floor(gy));

    let dx = gx - f32(ix);
    let tx = 1.0 - dx;
    let dx2 = dx * dx;
    let dx3 = dx2 * dx;
    var wx: array<f32, 4>;
    wx[0] = tx * tx * tx / 6.0;
    wx[1] = (4.0 - 6.0 * dx2 + 3.0 * dx3) / 6.0;
    wx[2] = (1.0 + 3.0 * dx + 3.0 * dx2 - 3.0 * dx3) / 6.0;
    wx[3] = dx3 / 6.0;

    let dy = gy - f32(iy);
    let ty = 1.0 - dy;
    let dy2 = dy * dy;
    let dy3 = dy2 * dy;
    var wy: array<f32, 4>;
    wy[0] = ty * ty * ty / 6.0;
    wy[1] = (4.0 - 6.0 * dy2 + 3.0 * dy3) / 6.0;
    wy[2] = (1.0 + 3.0 * dy + 3.0 * dy2 - 3.0 * dy3) / 6.0;
    wy[3] = dy3 / 6.0;

    var gradX: f32 = 0.0;
    var gradY: f32 = 0.0;
    let G = i32(GRID);

    // Interior fast path
    if (ix >= 1 && ix + 2 < G && iy >= 1 && iy + 2 < G) {
        for (var jy = 0u; jy < 4u; jy++) {
            let wyj = wy[jy];
            let row = u32(iy + i32(jy) - 1) * GRID + u32(ix - 1);
            for (var jx = 0u; jx < 4u; jx++) {
                let w = wx[jx] * wyj;
                gradX += sgGradX[row + jx] * w;
                gradY += sgGradY[row + jx] * w;
            }
        }
    } else {
        // Border: clamp-to-edge
        for (var jy = 0u; jy < 4u; jy++) {
            let wyj = wy[jy];
            let ny = clamp(iy + i32(jy) - 1, 0, G - 1);
            for (var jx = 0u; jx < 4u; jx++) {
                let nx = clamp(ix + i32(jx) - 1, 0, G - 1);
                let cellIdx = u32(ny) * GRID + u32(nx);
                let w = wx[jx] * wyj;
                gradX += sgGradX[cellIdx] * w;
                gradY += sgGradY[cellIdx] * w;
            }
        }
    }

    // F = -m · ∇Φ (gradients in grid units, convert to world)
    let gfx = -m * gradX * invCellW;
    let gfy = -m * gradY * invCellH;

    // Accumulate into gravity force slot (f0.xy) and totalForce
    var af = allForces[idx];
    af.f0.x += gfx;
    af.f0.y += gfy;
    af.totalForce.x += gfx;
    af.totalForce.y += gfy;
    allForces[idx] = af;
}
