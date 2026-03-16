// ─── Field Self-Gravity ───
// Energy density computation + SG gradient computation.
// Potential Φ is computed via FFT convolution (field-fft.wgsl), not here.

// Group 0: core field arrays + uniform
@group(0) @binding(0) var<storage, read_write> field: array<f32>;
@group(0) @binding(1) var<storage, read_write> fieldDot: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradX: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradY: array<f32>;
@group(0) @binding(4) var<storage, read_write> energyDensity: array<f32>;
@group(0) @binding(5) var<uniform> uniforms: FieldUniforms;

// Group 1: self-gravity output arrays
@group(1) @binding(0) var<storage, read_write> sgPhiFull: array<f32>;
@group(1) @binding(1) var<storage, read_write> sgGradX: array<f32>;
@group(1) @binding(2) var<storage, read_write> sgGradY: array<f32>;

// ─── Energy Density: ρ = ½φ̇² + ½|∇φ|² + V(φ) ───
@compute @workgroup_size(8, 8)
fn computeEnergyDensityHiggs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= GRID || iy >= GRID) { return; }

    let idx = iy * GRID + ix;
    let cellW = max(uniforms.domainW / f32(GRID), EPSILON);
    let cellH = max(uniforms.domainH / f32(GRID), EPSILON);
    let invCWsq = 1.0 / (cellW * cellW);
    let invCHsq = 1.0 / (cellH * cellH);

    let fd = fieldDot[idx];
    let gxi = gradX[idx];
    let gyi = gradY[idx];
    var rho = 0.5 * fd * fd + 0.5 * (gxi * gxi * invCWsq + gyi * gyi * invCHsq);

    // Mexican hat potential: V(φ) = μ²/4·(φ²-1)² (shifted so V(1)=0)
    let phi = field[idx];
    let muSq = 0.5 * uniforms.higgsMass * uniforms.higgsMass;
    let vacOffset = 0.25 * muSq;
    rho += muSq * (-0.5 * phi * phi + 0.25 * phi * phi * phi * phi) + vacOffset;

    energyDensity[idx] = rho;
}

@compute @workgroup_size(8, 8)
fn computeEnergyDensityAxion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= GRID || iy >= GRID) { return; }

    let idx = iy * GRID + ix;
    let cellW = max(uniforms.domainW / f32(GRID), EPSILON);
    let cellH = max(uniforms.domainH / f32(GRID), EPSILON);
    let invCWsq = 1.0 / (cellW * cellW);
    let invCHsq = 1.0 / (cellH * cellH);

    let fd = fieldDot[idx];
    let gxi = gradX[idx];
    let gyi = gradY[idx];
    var rho = 0.5 * fd * fd + 0.5 * (gxi * gxi * invCWsq + gyi * gyi * invCHsq);

    // Quadratic potential: V(a) = ½m_a²a²
    let a = field[idx];
    let halfMaSq = 0.5 * uniforms.axionMass * uniforms.axionMass;
    rho += halfMaSq * a * a;

    energyDensity[idx] = rho;
}

// ─── Self-Gravity Gradients (clamp-to-edge) ───
// sgPhiFull is written by FFT convolution pipeline (field-fft.wgsl).
@compute @workgroup_size(8, 8)
fn computeSelfGravGradients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= GRID || iy >= GRID) { return; }

    let idx = iy * GRID + ix;
    let phi = sgPhiFull[idx];

    // Interior fast path
    if (ix > 0u && ix < GRID_LAST && iy > 0u && iy < GRID_LAST) {
        sgGradX[idx] = (sgPhiFull[idx + 1u] - sgPhiFull[idx - 1u]) * 0.5;
        sgGradY[idx] = (sgPhiFull[idx + GRID] - sgPhiFull[idx - GRID]) * 0.5;
        return;
    }

    // Border: clamp-to-edge
    let fR = select(phi, sgPhiFull[idx + 1u], ix < GRID_LAST);
    let fL = select(phi, sgPhiFull[idx - 1u], ix > 0u);
    let fB = select(phi, sgPhiFull[idx + GRID], iy < GRID_LAST);
    let fT = select(phi, sgPhiFull[idx - GRID], iy > 0u);
    sgGradX[idx] = (fR - fL) * 0.5;
    sgGradY[idx] = (fB - fT) * 0.5;
}
