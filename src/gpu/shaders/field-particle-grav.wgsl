// ─── Particle-Field Gravity ───
// F = -m · ∇Φ(x), where Φ is the gravitational potential from field energy density.
// PQS interpolation of pre-computed grid gradients: O(N × 16).
// Dispatched once per active field (Higgs, Axion) when gravity is enabled.

// Struct definitions (ParticleState, AllForces) provided by shared-structs.wgsl.

struct FGUniforms {
    domainW: f32,
    domainH: f32,
    aliveCount: u32,
    boundaryMode: u32,
    topologyMode: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<uniform> u: FGUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read_write> allForces: array<AllForces>;

@group(1) @binding(0) var<storage, read> sgGradX: array<f32>;
@group(1) @binding(1) var<storage, read> sgGradY: array<f32>;

fn nbIndex(nx: i32, ny: i32, bcMode: u32, topoMode: u32) -> i32 {
    var cx = nx;
    var cy = ny;
    let G = i32(GRID);
    if (bcMode == BOUND_LOOP) {
        if (topoMode == TORUS) {
            if (cx < 0) { cx += G; } else if (cx >= G) { cx -= G; }
            if (cy < 0) { cy += G; } else if (cy >= G) { cy -= G; }
        } else if (topoMode == KLEIN) {
            if (cx < 0) { cx += G; } else if (cx >= G) { cx -= G; }
            if (cy < 0) { cy += G; cx = G - 1 - cx; }
            else if (cy >= G) { cy -= G; cx = G - 1 - cx; }
        } else {
            if (cx < 0) { cx += G; cy = G - 1 - cy; }
            else if (cx >= G) { cx -= G; cy = G - 1 - cy; }
            if (cy < 0) { cy += G; cx = G - 1 - cx; }
            else if (cy >= G) { cy -= G; cx = G - 1 - cx; }
        }
        return cy * G + cx;
    }
    if (bcMode == BOUND_BOUNCE) {
        cx = clamp(cx, 0, G - 1);
        cy = clamp(cy, 0, G - 1);
        return cy * G + cx;
    }
    if (cx < 0 || cx >= G || cy < 0 || cy >= G) { return -1; }
    return cy * G + cx;
}

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
        // Border: topology-aware wrapping
        for (var jy = 0u; jy < 4u; jy++) {
            let wyj = wy[jy];
            for (var jx = 0u; jx < 4u; jx++) {
                let cellIdx = nbIndex(ix + i32(jx) - 1, iy + i32(jy) - 1,
                                      u.boundaryMode, u.topologyMode);
                if (cellIdx < 0) { continue; }
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
