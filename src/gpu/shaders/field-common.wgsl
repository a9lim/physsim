// ─── Scalar Field Common Utilities ───
// Shared by field-deposit, field-evolve, field-forces, field-selfgrav, field-excitation
// Grid and physics constants are provided by the generated wgslConstants block.

struct FieldUniforms {
    dt: f32,
    domainW: f32,
    domainH: f32,
    boundaryMode: u32,
    topologyMode: u32,
    // Higgs params
    higgsMass: f32,
    higgsCoupling: f32,
    higgsMassFloor: f32,
    higgsMassMaxDelta: f32,
    // Axion params
    axionMass: f32,
    axionCoupling: f32,
    // Toggle bits
    higgsEnabled: u32,
    axionEnabled: u32,
    coulombEnabled: u32,
    yukawaEnabled: u32,
    gravityEnabled: u32,      // gravity (enables field self-gravity + particle-field gravity)
    relativityEnabled: u32,
    blackHoleEnabled: u32,
    particleCount: u32,
    softeningSq: f32,
    currentFieldType: u32,  // 0=higgs, 1=axion (set before each field's dispatch)
};

// ─── PQS (Cubic B-Spline, Order 3) Weight Computation ───
// 4x4 stencil centered at floor(gx), floor(gy)
// Returns base grid index (ix, iy) and weights wx[4], wy[4]

struct PQSResult {
    ix: i32,
    iy: i32,
    wx: array<f32, 4>,
    wy: array<f32, 4>,
};

fn pqsWeights(x: f32, y: f32, invCellW: f32, invCellH: f32) -> PQSResult {
    var result: PQSResult;

    let gx = x * invCellW - 0.5;
    let gy = y * invCellH - 0.5;
    result.ix = i32(floor(gx));
    result.iy = i32(floor(gy));

    let dx = gx - f32(result.ix);
    let tx = 1.0 - dx;
    let dx2 = dx * dx;
    let dx3 = dx2 * dx;
    result.wx[0] = tx * tx * tx / 6.0;
    result.wx[1] = (4.0 - 6.0 * dx2 + 3.0 * dx3) / 6.0;
    result.wx[2] = (1.0 + 3.0 * dx + 3.0 * dx2 - 3.0 * dx3) / 6.0;
    result.wx[3] = dx3 / 6.0;

    let dy = gy - f32(result.iy);
    let ty = 1.0 - dy;
    let dy2 = dy * dy;
    let dy3 = dy2 * dy;
    result.wy[0] = ty * ty * ty / 6.0;
    result.wy[1] = (4.0 - 6.0 * dy2 + 3.0 * dy3) / 6.0;
    result.wy[2] = (1.0 + 3.0 * dy + 3.0 * dy2 - 3.0 * dy3) / 6.0;
    result.wy[3] = dy3 / 6.0;

    return result;
}

// ─── Boundary-Aware Grid Neighbor ───
// Returns grid index for (nx, ny) with wrapping. Returns -1 for Dirichlet boundary.
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
        } else { // RP2
            if (cx < 0) { cx += G; cy = G - 1 - cy; }
            else if (cx >= G) { cx -= G; cy = G - 1 - cy; }
            if (cy < 0) { cy += G; cx = G - 1 - cx; }
            else if (cy >= G) { cy -= G; cx = G - 1 - cx; }
        }
        return cy * G + cx;
    }

    if (bcMode == BOUND_BOUNCE) {
        // Neumann: clamp
        cx = clamp(cx, 0, G - 1);
        cy = clamp(cy, 0, G - 1);
        return cy * G + cx;
    }

    // Despawn: Dirichlet
    if (cx < 0 || cx >= G || cy < 0 || cy >= G) { return -1; }
    return cy * G + cx;
}

// Check if stencil [ix-1..ix+2] x [iy-1..iy+2] is fully inside grid
fn isInterior(ix: i32, iy: i32) -> bool {
    return ix >= 1 && ix + 2 < i32(GRID) && iy >= 1 && iy + 2 < i32(GRID);
}
