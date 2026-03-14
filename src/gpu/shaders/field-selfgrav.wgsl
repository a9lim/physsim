// ─── Field Self-Gravity ───
// Coarse 8x8 grid O(SG^4=4096) direct potential, bilinear upsampled to 64x64

@group(0) @binding(0) var<storage, read> field: array<f32>;
@group(0) @binding(1) var<storage, read> fieldDot: array<f32>;
@group(0) @binding(2) var<storage, read> gradX: array<f32>;
@group(0) @binding(3) var<storage, read> gradY: array<f32>;
@group(0) @binding(4) var<storage, read_write> energyDensity: array<f32>;
@group(0) @binding(5) var<storage, read_write> coarseRho: array<f32>;
@group(0) @binding(6) var<storage, read_write> coarsePhi: array<f32>;
@group(0) @binding(7) var<storage, read_write> sgPhiFull: array<f32>;
@group(0) @binding(8) var<storage, read_write> sgGradX: array<f32>;
@group(0) @binding(9) var<storage, read_write> sgGradY: array<f32>;
@group(0) @binding(10) var<storage, read> sgInvR: array<f32>;
@group(0) @binding(11) var<uniform> uniforms: FieldUniforms;

// ─── Energy Density: ρ = ½φ̇² + ½|∇φ|² + V(φ) ───
// V(φ) added by field-specific variant
@compute @workgroup_size(8, 8)
fn computeEnergyDensityHiggs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= GRID || iy >= GRID) { return; }

    let idx = iy * GRID + ix;
    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
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
    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
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

// ─── Downsample to Coarse Grid ───
// One thread per coarse cell. Averages over ratio×ratio fine cells.
@compute @workgroup_size(8, 8)
fn downsampleRho(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = gid.x;
    let cy = gid.y;
    if (cx >= COARSE || cy >= COARSE) { return; }

    let ratio = SG_RATIO;
    let invBlock = 1.0 / f32(ratio * ratio);
    let baseX = cx * ratio;
    let baseY = cy * ratio;
    var sum: f32 = 0.0;

    for (var dy = 0u; dy < ratio; dy++) {
        let row = (baseY + dy) * GRID + baseX;
        for (var dx = 0u; dx < ratio; dx++) {
            sum += energyDensity[row + dx];
        }
    }

    coarseRho[cy * COARSE + cx] = sum * invBlock;
}

// ─── Coarse Potential: Φ = -Σ ρ·dA/r ───
// One thread per coarse cell. O(SG^2) sum per cell = O(SG^4) total.
// Uses pre-computed 1/sqrt table.
@compute @workgroup_size(8, 8)
fn computeCoarsePotential(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= COARSE || iy >= COARSE) { return; }

    let i = iy * COARSE + ix;
    let cellArea = (uniforms.domainW / f32(COARSE)) * (uniforms.domainH / f32(COARSE));
    var pot: f32 = 0.0;
    let rowBase = i * COARSE_SQ;

    for (var j = 0u; j < COARSE_SQ; j++) {
        let rhoJ = coarseRho[j];
        if (rhoJ >= EPSILON) {
            pot -= rhoJ * cellArea * sgInvR[rowBase + j];
        }
    }

    coarsePhi[i] = pot;
}

// ─── Bilinear Upsample: Coarse → Full Grid ───
@compute @workgroup_size(8, 8)
fn upsamplePhi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= GRID || iy >= GRID) { return; }

    let invRatio = 1.0 / f32(SG_RATIO);
    let sgLast = i32(COARSE) - 1;

    let fx = (f32(ix) + 0.5) * invRatio - 0.5;
    let fy = (f32(iy) + 0.5) * invRatio - 0.5;

    let cx0 = clamp(i32(floor(fx)), 0, sgLast - 1);
    let cx1 = min(cx0 + 1, sgLast);
    let cy0 = clamp(i32(floor(fy)), 0, sgLast - 1);
    let cy1 = min(cy0 + 1, sgLast);

    let wx = max(0.0, fx - f32(cx0));
    let wy = max(0.0, fy - f32(cy0));

    let v00 = coarsePhi[cy0 * i32(COARSE) + cx0];
    let v10 = coarsePhi[cy0 * i32(COARSE) + cx1];
    let v01 = coarsePhi[cy1 * i32(COARSE) + cx0];
    let v11 = coarsePhi[cy1 * i32(COARSE) + cx1];

    sgPhiFull[iy * GRID + ix] = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v10)
                               + wy * ((1.0 - wx) * v01 + wx * v11);
}

// ─── Self-Gravity Gradients (clamp-to-edge) ───
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
