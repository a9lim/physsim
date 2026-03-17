// ─── Scalar Field Evolution (Fused) ───
// KDK half-kicks compute Laplacian inline (no separate Laplacian pass).
// NaN/Inf fixup integrated into drift + kick (no separate fixup pass).

@group(0) @binding(0) var<storage, read_write> field: array<f32>;
@group(0) @binding(1) var<storage, read_write> fieldDot: array<f32>;
@group(0) @binding(2) var<storage, read_write> laplacian: array<f32>;  // unused (kept for layout compat)
@group(0) @binding(3) var<storage, read_write> source: array<f32>;     // rw for encoder compat
@group(0) @binding(4) var<storage, read_write> thermal: array<f32>;    // rw for encoder compat
@group(0) @binding(5) var<storage, read_write> sgPhiFull: array<f32>;  // rw for encoder compat
@group(0) @binding(6) var<storage, read_write> sgGradX: array<f32>;    // rw for encoder compat
@group(0) @binding(7) var<storage, read_write> sgGradY: array<f32>;    // rw for encoder compat
@group(0) @binding(8) var<storage, read_write> fieldGradX: array<f32>;  // field gradients (rw for computeGridGradients)
@group(0) @binding(9) var<storage, read_write> fieldGradY: array<f32>;
@group(0) @binding(10) var<uniform> uniforms: FieldUniforms;

// ─── Inline Laplacian helper ───
// 5-point stencil with topology-aware border handling.
fn inlineLaplacian(ix: u32, iy: u32, idx: u32, invCWsq: f32, invCHsq: f32) -> f32 {
    let fC = field[idx];

    // Interior fast path (covers ~99.6% of cells)
    if (ix > 0u && ix < GRID_LAST && iy > 0u && iy < GRID_LAST) {
        return (field[idx - 1u] + field[idx + 1u] - 2.0 * fC) * invCWsq
             + (field[idx - GRID] + field[idx + GRID] - 2.0 * fC) * invCHsq;
    }

    // Border: topology-aware via nbIndex
    let bcMode = uniforms.boundaryMode;
    let topoMode = uniforms.topologyMode;
    let vacValue = select(0.0, 1.0, uniforms.currentFieldType == 0u);
    let iL = nbIndex(i32(ix) - 1, i32(iy), bcMode, topoMode);
    let iR = nbIndex(i32(ix) + 1, i32(iy), bcMode, topoMode);
    let iT = nbIndex(i32(ix), i32(iy) - 1, bcMode, topoMode);
    let iB = nbIndex(i32(ix), i32(iy) + 1, bcMode, topoMode);
    let fL = select(vacValue, field[iL], iL >= 0);
    let fR = select(vacValue, field[iR], iR >= 0);
    let fT = select(vacValue, field[iT], iT >= 0);
    let fB = select(vacValue, field[iB], iB >= 0);
    return (fL + fR - 2.0 * fC) * invCWsq + (fT + fB - 2.0 * fC) * invCHsq;
}

// ─── Higgs KDK Half-Kick (Fused: inline Laplacian + NaN guard) ───
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
    let lapI = inlineLaplacian(ix, iy, idx, invCWsq, invCHsq);
    let muSqEff = muSq - thermal[idx];  // Phase transition: KE reduces effective μ²
    let srcTerm = source[idx] * invCellArea;
    let fdC = fieldDot[idx];

    // Numerical viscosity: ν·∇²(ȧ) with ν = 1/(2√(1/dx²+1/dy²)) → Q=1 at Nyquist, vanishes for physical modes
    let fdL = select(fdC, fieldDot[idx - 1u], ix > 0u);
    let fdR = select(fdC, fieldDot[idx + 1u], ix < GRID_LAST);
    let fdT = select(fdC, fieldDot[idx - GRID], iy > 0u);
    let fdB = select(fdC, fieldDot[idx + GRID], iy < GRID_LAST);
    let fdLap = (fdL + fdR - 2.0 * fdC) * invCWsq + (fdT + fdB - 2.0 * fdC) * invCHsq;
    let viscosity = 0.5 * inverseSqrt(invCWsq + invCHsq) * fdLap;

    var ddphi = lapI + muSqEff * phi - muSq * phi * phi * phi
              - damp * fdC + srcTerm + viscosity;

    // Self-gravity correction (weak-field GR, clamp Φ for stability)
    if (uniforms.gravityEnabled != 0u) {
        let Phi = clamp(sgPhiFull[idx], -SELFGRAV_PHI_MAX, SELFGRAV_PHI_MAX);
        ddphi += 4.0 * Phi * lapI
               + 2.0 * (sgGradX[idx] * fieldGradX[idx] * invCWsq
                       + sgGradY[idx] * fieldGradY[idx] * invCHsq)
               + 2.0 * Phi * (muSqEff * phi - muSq * phi * phi * phi);
    }

    var newFdot = fdC + ddphi * halfDt;
    // NaN/Inf guard on fieldDot
    if (newFdot != newFdot || abs(newFdot) > 1e6) { newFdot = 0.0; }
    fieldDot[idx] = newFdot;
}

// ─── Axion KDK Half-Kick (Fused: inline Laplacian + NaN guard) ───
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
    let lapI = inlineLaplacian(ix, iy, idx, invCWsq, invCHsq);
    let srcTerm = source[idx] * invCellArea;
    let fdC = fieldDot[idx];

    // Numerical viscosity: ν·∇²(ȧ) with ν = 1/(2√(1/dx²+1/dy²)) → Q=1 at Nyquist, vanishes for physical modes
    let fdL = select(fdC, fieldDot[idx - 1u], ix > 0u);
    let fdR = select(fdC, fieldDot[idx + 1u], ix < GRID_LAST);
    let fdT = select(fdC, fieldDot[idx - GRID], iy > 0u);
    let fdB = select(fdC, fieldDot[idx + GRID], iy < GRID_LAST);
    let fdLap = (fdL + fdR - 2.0 * fdC) * invCWsq + (fdT + fdB - 2.0 * fdC) * invCHsq;
    let viscosity = 0.5 * inverseSqrt(invCWsq + invCHsq) * fdLap;

    var ddA = lapI - mASq * aVal - damp * fdC + srcTerm + viscosity;

    if (uniforms.gravityEnabled != 0u) {
        let Phi = clamp(sgPhiFull[idx], -SELFGRAV_PHI_MAX, SELFGRAV_PHI_MAX);
        ddA += 4.0 * Phi * lapI
             + 2.0 * (sgGradX[idx] * fieldGradX[idx] * invCWsq
                     + sgGradY[idx] * fieldGradY[idx] * invCHsq)
             - 2.0 * Phi * mASq * aVal;
    }

    var newFdot = fdC + ddA * (uniforms.dt * 0.5);
    // NaN/Inf guard on fieldDot
    if (newFdot != newFdot || abs(newFdot) > 1e6) { newFdot = 0.0; }
    fieldDot[idx] = newFdot;
}

// ─── Field Drift (with NaN/Inf fixup) ───
// field += fieldDot * dt, clamped to [-SCALAR_FIELD_MAX, SCALAR_FIELD_MAX]
@compute @workgroup_size(8, 8)
fn fieldDrift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * GRID + gid.x;
    if (gid.x >= GRID || gid.y >= GRID) { return; }

    var newField = field[idx] + fieldDot[idx] * uniforms.dt;

    // NaN/Inf fixup: reset to vacuum value
    if (newField != newField || abs(newField) > 1e6) {
        let vacValue = select(0.0, 1.0, uniforms.currentFieldType == 0u);
        newField = vacValue;
        fieldDot[idx] = 0.0;
    }

    field[idx] = clamp(newField, -SCALAR_FIELD_MAX, SCALAR_FIELD_MAX);
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
    let vacValue = select(0.0, 1.0, uniforms.currentFieldType == 0u);

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
