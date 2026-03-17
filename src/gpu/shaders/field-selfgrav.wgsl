// ─── Field Self-Gravity (Fused) ───
// Energy density → FFT complex pack (fused), and
// FFT complex unpack → Φ extraction → SG gradient computation (fused).
// Potential Φ is computed via FFT convolution (field-fft.wgsl) between these two stages.

// Group 0: core field arrays + uniform
@group(0) @binding(0) var<storage, read_write> field: array<f32>;
@group(0) @binding(1) var<storage, read_write> fieldDot: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradX: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradY: array<f32>;
@group(0) @binding(4) var<storage, read_write> energyDensity: array<f32>;  // unused (kept for layout compat)
@group(0) @binding(5) var<uniform> uniforms: FieldUniforms;

// Group 1: self-gravity output arrays
@group(1) @binding(0) var<storage, read_write> sgPhiFull: array<f32>;
@group(1) @binding(1) var<storage, read_write> sgGradX: array<f32>;
@group(1) @binding(2) var<storage, read_write> sgGradY: array<f32>;

// Group 2: FFT complex buffer (interleaved re/im pairs, size GRID*GRID*2)
@group(2) @binding(0) var<storage, read_write> fftComplex: array<f32>;

// ─── Fused: Energy Density + Pack (Higgs) ───
// Computes ρ = ½φ̇² + ½|∇φ|² + V(φ), then writes ρ·cellArea directly
// to interleaved complex FFT buffer (real part only, imaginary = 0).
@compute @workgroup_size(8, 8)
fn energyDensityHiggsAndPack(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // Pack directly into complex FFT buffer: re = ρ·dA, im = 0
    let cellArea = cellW * cellH;
    fftComplex[2u * idx] = rho * cellArea;
    fftComplex[2u * idx + 1u] = 0.0;
}

// ─── Fused: Energy Density + Pack (Axion) ───
@compute @workgroup_size(8, 8)
fn energyDensityAxionAndPack(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // Pack directly into complex FFT buffer: re = ρ·dA, im = 0
    let cellArea = cellW * cellH;
    fftComplex[2u * idx] = rho * cellArea;
    fftComplex[2u * idx + 1u] = 0.0;
}

// ─── Fused: Unpack IFFT Result + SG Gradients ───
// Reads Φ from interleaved complex IFFT output (real part at stride 2),
// stores to sgPhiFull, and computes topology-aware central-difference gradients.
@compute @workgroup_size(8, 8)
fn unpackAndSGGradients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= GRID || iy >= GRID) { return; }

    let idx = iy * GRID + ix;

    // Extract real part of IFFT result → Φ
    let phi = fftComplex[2u * idx];
    sgPhiFull[idx] = phi;

    // Interior fast path: direct indexing of complex buffer at stride 2
    if (ix > 0u && ix < GRID_LAST && iy > 0u && iy < GRID_LAST) {
        sgGradX[idx] = (fftComplex[2u * (idx + 1u)] - fftComplex[2u * (idx - 1u)]) * 0.5;
        sgGradY[idx] = (fftComplex[2u * (idx + GRID)] - fftComplex[2u * (idx - GRID)]) * 0.5;
        return;
    }

    // Border: topology-aware via nbIndex()
    let bcMode = uniforms.boundaryMode;
    let topoMode = uniforms.topologyMode;
    let iR = nbIndex(i32(ix) + 1, i32(iy), bcMode, topoMode);
    let iL = nbIndex(i32(ix) - 1, i32(iy), bcMode, topoMode);
    let iB = nbIndex(i32(ix), i32(iy) + 1, bcMode, topoMode);
    let iT = nbIndex(i32(ix), i32(iy) - 1, bcMode, topoMode);
    let fR = select(0.0, fftComplex[2u * u32(iR)], iR >= 0);
    let fL = select(0.0, fftComplex[2u * u32(iL)], iL >= 0);
    let fB = select(0.0, fftComplex[2u * u32(iB)], iB >= 0);
    let fT = select(0.0, fftComplex[2u * u32(iT)], iT >= 0);
    sgGradX[idx] = (fR - fL) * 0.5;
    sgGradY[idx] = (fB - fT) * 0.5;
}
