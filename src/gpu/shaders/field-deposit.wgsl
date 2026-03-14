// ─── PQS Two-Pass Scatter/Gather Deposition ───
// Pass 1 (scatter): One thread per particle. Computes PQS weights, writes
//   16 weighted values + base grid index to scratch buffer.
// Pass 2 (gather): One thread per grid cell. Scans all particles, checks
//   if stencil overlaps this cell, accumulates contributions.

// Bindings (set by JS for each field and deposition target)
@group(0) @binding(0) var<storage, read> posX: array<f32>;
@group(0) @binding(1) var<storage, read> posY: array<f32>;
@group(0) @binding(2) var<storage, read> mass: array<f32>;
@group(0) @binding(3) var<storage, read> baseMass: array<f32>;
@group(0) @binding(4) var<storage, read> charge: array<f32>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;
@group(0) @binding(6) var<storage, read> velWX: array<f32>;
@group(0) @binding(7) var<storage, read> velWY: array<f32>;

@group(1) @binding(0) var<storage, read_write> scratchWeights: array<f32>;  // maxParticles * 16
@group(1) @binding(1) var<storage, read_write> scratchIndices: array<i32>;  // maxParticles * 2 (ix, iy)
@group(1) @binding(2) var<storage, read_write> targetGrid: array<f32>;      // GRID_SQ
@group(1) @binding(3) var<uniform> uniforms: FieldUniforms;

// ─── Deposit Mode Constants ───
// Set via push constant or uniform to select what to deposit:
// 0 = Higgs source (g * baseMass)
// 1 = Axion source (g * q^2 for EM + g * m * sign for PQ)
// 2 = Higgs thermal (particle KE)
const DEPOSIT_HIGGS_SOURCE: u32 = 0u;
const DEPOSIT_AXION_SOURCE: u32 = 1u;
const DEPOSIT_HIGGS_THERMAL: u32 = 2u;

// Include field-common.wgsl utilities inline (WGSL has no #include; JS concatenates)
// PQS weights function, nbIndex, constants — assumed prepended by JS

// ─── Pass 1: Scatter ───
// One thread per particle. Writes 16 weighted values to scratch.
@compute @workgroup_size(256)
fn scatterDeposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= uniforms.particleCount) { return; }

    // Check alive flag
    let flag = flags[pid];
    if ((flag & 1u) == 0u) { return; }  // not alive

    let px = posX[pid];
    let py = posY[pid];

    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }
    let invCellW = 1.0 / cellW;
    let invCellH = 1.0 / cellH;

    // Compute PQS weights
    let pqs = pqsWeights(px, py, invCellW, invCellH);

    // Compute deposit value based on mode
    // Mode is encoded in uniforms — we use a combined shader with branching
    // The JS side dispatches with the appropriate mode uniform
    var value: f32 = 0.0;

    // Higgs source: g * baseMass
    let bm = baseMass[pid];
    if (bm < EPSILON) { return; }

    // For Higgs source mode (default path — JS selects target grid)
    value = uniforms.higgsCoupling * bm;

    // Write base indices
    let base = pid * 2u;
    scratchIndices[base] = pqs.ix;
    scratchIndices[base + 1u] = pqs.iy;

    // Write 16 stencil weights (4x4, row-major)
    let wBase = pid * 16u;
    for (var jy = 0u; jy < 4u; jy++) {
        let vwy = value * pqs.wy[jy];
        for (var jx = 0u; jx < 4u; jx++) {
            scratchWeights[wBase + jy * 4u + jx] = vwy * pqs.wx[jx];
        }
    }
}

// Axion source scatter variant
@compute @workgroup_size(256)
fn scatterDepositAxion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= uniforms.particleCount) { return; }

    let flag = flags[pid];
    if ((flag & 1u) == 0u) { return; }

    let px = posX[pid];
    let py = posY[pid];

    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }
    let invCellW = 1.0 / cellW;
    let invCellH = 1.0 / cellH;

    let pqs = pqsWeights(px, py, invCellW, invCellH);

    // Axion source: g*q^2 (EM, when Coulomb on) + g*m*sign (PQ, when Yukawa on)
    var value: f32 = 0.0;
    let g = uniforms.axionCoupling;
    if (uniforms.coulombEnabled != 0u) {
        let q = charge[pid];
        let qSq = q * q;
        if (qSq > EPSILON) { value += g * qSq; }
    }
    if (uniforms.yukawaEnabled != 0u) {
        let m = mass[pid];
        if (m > EPSILON) {
            let isAntimatter = (flag & 4u) != 0u;  // antimatter bit
            let sign = select(1.0, -1.0, isAntimatter);
            value += g * m * sign;
        }
    }
    if (abs(value) < EPSILON) { return; }

    let base = pid * 2u;
    scratchIndices[base] = pqs.ix;
    scratchIndices[base + 1u] = pqs.iy;

    let wBase = pid * 16u;
    for (var jy = 0u; jy < 4u; jy++) {
        let vwy = value * pqs.wy[jy];
        for (var jx = 0u; jx < 4u; jx++) {
            scratchWeights[wBase + jy * 4u + jx] = vwy * pqs.wx[jx];
        }
    }
}

// Thermal KE scatter (Higgs phase transitions)
@compute @workgroup_size(256)
fn scatterDepositThermal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= uniforms.particleCount) { return; }

    let flag = flags[pid];
    if ((flag & 1u) == 0u) { return; }

    let px = posX[pid];
    let py = posY[pid];

    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }
    let invCellW = 1.0 / cellW;
    let invCellH = 1.0 / cellH;

    let pqs = pqsWeights(px, py, invCellW, invCellH);

    // Compute KE
    let wx = velWX[pid];
    let wy = velWY[pid];
    let wSq = wx * wx + wy * wy;
    var ke: f32;
    if (uniforms.relativityEnabled != 0u) {
        ke = wSq / (sqrt(1.0 + wSq) + 1.0) * mass[pid];
    } else {
        // vel = w when relativity off
        ke = 0.5 * mass[pid] * wSq;
    }
    if (ke < EPSILON) { return; }

    let base = pid * 2u;
    scratchIndices[base] = pqs.ix;
    scratchIndices[base + 1u] = pqs.iy;

    let wBase = pid * 16u;
    for (var jy = 0u; jy < 4u; jy++) {
        let vwy = ke * pqs.wy[jy];
        for (var jx = 0u; jx < 4u; jx++) {
            scratchWeights[wBase + jy * 4u + jx] = vwy * pqs.wx[jx];
        }
    }
}

// ─── Pass 2: Gather ───
// One thread per grid cell. Sums contributions from all particles whose
// 4x4 stencil overlaps this cell.
@compute @workgroup_size(8, 8)
fn gatherDeposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if (gx >= GRID || gy >= GRID) { return; }

    let cellIdx = gy * GRID + gx;
    var total: f32 = 0.0;

    let bcMode = uniforms.boundaryMode;
    let topoMode = uniforms.topologyMode;

    // For each particle, check if this cell falls within its 4x4 stencil
    for (var pid = 0u; pid < uniforms.particleCount; pid++) {
        let flag = flags[pid];
        if ((flag & 1u) == 0u) { continue; }

        let base = pid * 2u;
        let ix = scratchIndices[base];
        let iy = scratchIndices[base + 1u];

        // Stencil covers [ix-1..ix+2] x [iy-1..iy+2]
        // For interior: direct check
        if (isInterior(ix, iy)) {
            let sx = i32(gx) - (ix - 1);
            let sy = i32(gy) - (iy - 1);
            if (sx >= 0 && sx < 4 && sy >= 0 && sy < 4) {
                total += scratchWeights[pid * 16u + u32(sy) * 4u + u32(sx)];
            }
        } else {
            // Border path: check each stencil node against this cell via nbIndex
            let wBase = pid * 16u;
            for (var jy = 0u; jy < 4u; jy++) {
                for (var jx = 0u; jx < 4u; jx++) {
                    let nIdx = nbIndex(ix + i32(jx) - 1, iy + i32(jy) - 1, bcMode, topoMode);
                    if (nIdx == i32(cellIdx)) {
                        total += scratchWeights[wBase + jy * 4u + jx];
                    }
                }
            }
        }
    }

    targetGrid[cellIdx] = total;
}

// ─── Clear Grid ───
@compute @workgroup_size(8, 8)
fn clearGrid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * GRID + gid.x;
    if (gid.x >= GRID || gid.y >= GRID) { return; }
    targetGrid[idx] = 0.0;
}
