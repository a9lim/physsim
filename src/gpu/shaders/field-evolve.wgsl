// ─── Scalar Field Evolution ───
// Störmer-Verlet KDK: half-kick → drift → recompute Laplacian → second half-kick
// Separate entry points for Higgs (Mexican hat) and Axion (quadratic)

@group(0) @binding(0) var<storage, read_write> field: array<f32>;
@group(0) @binding(1) var<storage, read_write> fieldDot: array<f32>;
@group(0) @binding(2) var<storage, read_write> laplacian: array<f32>;
@group(0) @binding(3) var<storage, read> source: array<f32>;
@group(0) @binding(4) var<storage, read> thermal: array<f32>;  // Higgs only
@group(0) @binding(5) var<storage, read> sgPhiFull: array<f32>;  // self-gravity potential
@group(0) @binding(6) var<storage, read> sgGradX: array<f32>;
@group(0) @binding(7) var<storage, read> sgGradY: array<f32>;
@group(0) @binding(8) var<storage, read_write> fieldGradX: array<f32>;  // field gradients (rw for computeGridGradients)
@group(0) @binding(9) var<storage, read_write> fieldGradY: array<f32>;
@group(0) @binding(10) var<uniform> uniforms: FieldUniforms;

// ─── Laplacian (5-point stencil) ───
@compute @workgroup_size(8, 8)
fn computeLaplacian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= GRID || iy >= GRID) { return; }

    let idx = iy * GRID + ix;
    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    let invCWsq = 1.0 / (cellW * cellW);
    let invCHsq = 1.0 / (cellH * cellH);
    let fC = field[idx];
    let bcMode = uniforms.boundaryMode;
    let topoMode = uniforms.topologyMode;

    // Determine vacuum value based on which field this is
    // Higgs: vacValue=1, Axion: vacValue=0
    // Encoded in uniforms (higgsEnabled selects vac=1, else vac=0)
    let vacValue = select(0.0, 1.0, uniforms.higgsEnabled != 0u);

    // Interior fast path
    if (ix > 0u && ix < GRID_LAST && iy > 0u && iy < GRID_LAST) {
        laplacian[idx] = (field[idx - 1u] + field[idx + 1u] - 2.0 * fC) * invCWsq
                       + (field[idx - GRID] + field[idx + GRID] - 2.0 * fC) * invCHsq;
        return;
    }

    // Border path
    let iL = nbIndex(i32(ix) - 1, i32(iy), bcMode, topoMode);
    let iR = nbIndex(i32(ix) + 1, i32(iy), bcMode, topoMode);
    let iT = nbIndex(i32(ix), i32(iy) - 1, bcMode, topoMode);
    let iB = nbIndex(i32(ix), i32(iy) + 1, bcMode, topoMode);
    let fL = select(vacValue, field[iL], iL >= 0);
    let fR = select(vacValue, field[iR], iR >= 0);
    let fT = select(vacValue, field[iT], iT >= 0);
    let fB = select(vacValue, field[iB], iB >= 0);
    laplacian[idx] = (fL + fR - 2.0 * fC) * invCWsq + (fT + fB - 2.0 * fC) * invCHsq;
}

// ─── Higgs KDK Half-Kick ───
// V(φ) = -½μ²φ² + ¼λφ⁴, VEV=1, λ=μ²=m_H²/2
// ddphi = ∇²φ + μ²_eff·φ - μ²·φ³ - 2m_H·φ̇ + source/cellArea
//   + self-grav: 4Φ·∇²φ + 2(∇Φ·∇φ)/cellArea² + 2Φ·(μ²_eff·φ - μ²·φ³)
@compute @workgroup_size(8, 8)
fn higgsHalfKick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= GRID || iy >= GRID) { return; }

    let idx = iy * GRID + ix;
    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    let cellArea = cellW * cellH;
    let invCellArea = select(0.0, 1.0 / cellArea, cellArea > EPSILON);
    let invCWsq = 1.0 / (cellW * cellW);
    let invCHsq = 1.0 / (cellH * cellH);

    let mH = uniforms.higgsMass;
    let muSq = 0.5 * mH * mH;
    let damp = 2.0 * mH;
    let halfDt = uniforms.dt * 0.5;

    let phi = field[idx];
    let lapI = laplacian[idx];
    let muSqEff = muSq - thermal[idx];  // Phase transition: KE reduces effective μ²
    let srcTerm = source[idx] * invCellArea;

    var ddphi = lapI + muSqEff * phi - muSq * phi * phi * phi
              - damp * fieldDot[idx] + srcTerm;

    // Self-gravity correction (weak-field GR)
    if (uniforms.gravityEnabled != 0u) {
        let Phi = sgPhiFull[idx];
        ddphi += 4.0 * Phi * lapI
               + 2.0 * (sgGradX[idx] * fieldGradX[idx] * invCWsq
                       + sgGradY[idx] * fieldGradY[idx] * invCHsq)
               + 2.0 * Phi * (muSqEff * phi - muSq * phi * phi * phi);
    }

    fieldDot[idx] += ddphi * halfDt;
}

// ─── Axion KDK Half-Kick ───
// V(a) = ½m_a²a², vacuum at a=0
// ddA = ∇²a - m_a²·a - g·m_a·ȧ + source/cellArea
//   + self-grav: 4Φ·∇²a + 2(∇Φ·∇a)/cellArea² - 2Φ·m_a²·a
@compute @workgroup_size(8, 8)
fn axionHalfKick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= GRID || iy >= GRID) { return; }

    let idx = iy * GRID + ix;
    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    let cellArea = cellW * cellH;
    let invCellArea = select(0.0, 1.0 / cellArea, cellArea > EPSILON);
    let invCWsq = 1.0 / (cellW * cellW);
    let invCHsq = 1.0 / (cellH * cellH);

    let mA = uniforms.axionMass;
    let mASq = mA * mA;
    let damp = uniforms.axionCoupling * mA;  // ζ=g/2, Q=1/g → damp = g*m_a

    let aVal = field[idx];
    let lapI = laplacian[idx];
    let srcTerm = source[idx] * invCellArea;

    var ddA = lapI - mASq * aVal - damp * fieldDot[idx] + srcTerm;

    if (uniforms.gravityEnabled != 0u) {
        let Phi = sgPhiFull[idx];
        ddA += 4.0 * Phi * lapI
             + 2.0 * (sgGradX[idx] * fieldGradX[idx] * invCWsq
                     + sgGradY[idx] * fieldGradY[idx] * invCHsq)
             - 2.0 * Phi * mASq * aVal;
    }

    fieldDot[idx] += ddA * (uniforms.dt * 0.5);
}

// ─── Field Drift (shared Higgs/Axion) ───
// field += fieldDot * dt, clamped to [-SCALAR_FIELD_MAX, SCALAR_FIELD_MAX]
@compute @workgroup_size(8, 8)
fn fieldDrift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * GRID + gid.x;
    if (gid.x >= GRID || gid.y >= GRID) { return; }
    field[idx] = clamp(field[idx] + fieldDot[idx] * uniforms.dt,
                       -SCALAR_FIELD_MAX, SCALAR_FIELD_MAX);
}

// ─── NaN Fixup (post second half-kick) ───
// Higgs: reset to VEV=1, Axion: reset to 0
@compute @workgroup_size(8, 8)
fn nanFixupHiggs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * GRID + gid.x;
    if (gid.x >= GRID || gid.y >= GRID) { return; }
    let phi = field[idx];
    if (phi != phi) {  // NaN check
        field[idx] = 1.0;
        fieldDot[idx] = 0.0;
    }
}

@compute @workgroup_size(8, 8)
fn nanFixupAxion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * GRID + gid.x;
    if (gid.x >= GRID || gid.y >= GRID) { return; }
    let a = field[idx];
    if (a != a) {
        field[idx] = 0.0;
        fieldDot[idx] = 0.0;
    }
}

// ─── Grid Gradients (central differences) ───
@compute @workgroup_size(8, 8)
fn computeGridGradients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= GRID || iy >= GRID) { return; }

    let idx = iy * GRID + ix;
    let bcMode = uniforms.boundaryMode;
    let topoMode = uniforms.topologyMode;
    let vacValue = select(0.0, 1.0, uniforms.higgsEnabled != 0u);

    // Note: gradX/gradY are bound as read_write via separate bind group for output
    // Interior fast path
    if (ix > 0u && ix < GRID_LAST && iy > 0u && iy < GRID_LAST) {
        fieldGradX[idx] = (field[idx + 1u] - field[idx - 1u]) * 0.5;
        fieldGradY[idx] = (field[idx + GRID] - field[idx - GRID]) * 0.5;
        return;
    }

    // Border path
    let iL = nbIndex(i32(ix) - 1, i32(iy), bcMode, topoMode);
    let iR = nbIndex(i32(ix) + 1, i32(iy), bcMode, topoMode);
    let iT = nbIndex(i32(ix), i32(iy) - 1, bcMode, topoMode);
    let iB = nbIndex(i32(ix), i32(iy) + 1, bcMode, topoMode);
    let fL = select(vacValue, field[iL], iL >= 0);
    let fR = select(vacValue, field[iR], iR >= 0);
    let fT = select(vacValue, field[iT], iT >= 0);
    let fB = select(vacValue, field[iB], iB >= 0);
    fieldGradX[idx] = (fR - fL) * 0.5;
    fieldGradY[idx] = (fB - fT) * 0.5;
}
