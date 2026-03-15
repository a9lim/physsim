// ─── Field Self-Gravity ───
// Coarse 8x8 grid O(SG^4=4096) direct potential, bilinear upsampled to 64x64

// Group 0: core field arrays + uniform (6 storage + 1 uniform = 7 bindings)
@group(0) @binding(0) var<storage, read_write> field: array<f32>;
@group(0) @binding(1) var<storage, read_write> fieldDot: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradX: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradY: array<f32>;
@group(0) @binding(4) var<storage, read_write> energyDensity: array<f32>;
@group(0) @binding(5) var<storage, read_write> coarseRho: array<f32>;
@group(0) @binding(6) var<uniform> uniforms: FieldUniforms;

// Group 1: self-gravity arrays (4 storage — sgInvR removed, computed inline)
@group(1) @binding(0) var<storage, read_write> coarsePhi: array<f32>;
@group(1) @binding(1) var<storage, read_write> sgPhiFull: array<f32>;
@group(1) @binding(2) var<storage, read_write> sgGradX: array<f32>;
@group(1) @binding(3) var<storage, read_write> sgGradY: array<f32>;

// ─── Energy Density: ρ = ½φ̇² + ½|∇φ|² + V(φ) ───
// V(φ) added by field-specific variant
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
// Computes 1/r inline (avoids needing a pre-computed sgInvR lookup table buffer).
@compute @workgroup_size(8, 8)
fn computeCoarsePotential(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= COARSE || iy >= COARSE) { return; }

    let cellW = uniforms.domainW / f32(COARSE);
    let cellH = uniforms.domainH / f32(COARSE);
    let cellArea = cellW * cellH;
    var pot: f32 = 0.0;

    let periodic = uniforms.boundaryMode == BOUND_LOOP;
    let topo = uniforms.topologyMode;
    let halfDomW = uniforms.domainW * 0.5;
    let halfDomH = uniforms.domainH * 0.5;
    let domW = uniforms.domainW;
    let domH = uniforms.domainH;

    for (var jy = 0u; jy < COARSE; jy++) {
        for (var jx = 0u; jx < COARSE; jx++) {
            let j = jy * COARSE + jx;
            let rhoJ = coarseRho[j];
            if (rhoJ < EPSILON) { continue; }
            // Source position in world coords (cell center)
            let sx = (f32(jx) + 0.5) * cellW;
            let sy = (f32(jy) + 0.5) * cellH;
            // Observer position
            let ox = (f32(ix) + 0.5) * cellW;
            let oy = (f32(iy) + 0.5) * cellH;
            var dx = sx - ox;
            var dy = sy - oy;
            // Topology-aware minimum-image wrapping
            if (periodic) {
                if (topo == TOPO_TORUS) {
                    if (dx > halfDomW) { dx -= domW; } else if (dx < -halfDomW) { dx += domW; }
                    if (dy > halfDomH) { dy -= domH; } else if (dy < -halfDomH) { dy += domH; }
                } else if (topo == TOPO_KLEIN) {
                    // Direct
                    var dx0 = dx;
                    if (dx0 > halfDomW) { dx0 -= domW; } else if (dx0 < -halfDomW) { dx0 += domW; }
                    var dy0 = dy;
                    if (dy0 > halfDomH) { dy0 -= domH; } else if (dy0 < -halfDomH) { dy0 += domH; }
                    var bestSq = dx0 * dx0 + dy0 * dy0;
                    dx = dx0; dy = dy0;
                    // Klein glide reflection
                    let gx = domW - sx;
                    var dx1 = gx - ox;
                    if (dx1 > halfDomW) { dx1 -= domW; } else if (dx1 < -halfDomW) { dx1 += domW; }
                    var dy1 = (sy + domH) - oy;
                    if (dy1 > domH) { dy1 -= 2.0 * domH; } else if (dy1 < -domH) { dy1 += 2.0 * domH; }
                    if (dx1 * dx1 + dy1 * dy1 < bestSq) { dx = dx1; dy = dy1; bestSq = dx1 * dx1 + dy1 * dy1; }
                    var dy1b = (sy - domH) - oy;
                    if (dy1b > domH) { dy1b -= 2.0 * domH; } else if (dy1b < -domH) { dy1b += 2.0 * domH; }
                    if (dx1 * dx1 + dy1b * dy1b < bestSq) { dx = dx1; dy = dy1b; }
                } else {
                    // RP²: both axes glide reflections (use torus as approximation — matching
                    // accuracy is acceptable for coarse 8×8 self-gravity potential)
                    if (dx > halfDomW) { dx -= domW; } else if (dx < -halfDomW) { dx += domW; }
                    if (dy > halfDomH) { dy -= domH; } else if (dy < -halfDomH) { dy += domH; }
                }
            }
            // Use softening for self-cell (matches CPU: 1/sqrt(dx²+dy²+softeningSq))
            let rSq = dx * dx + dy * dy + uniforms.softeningSq;
            pot -= rhoJ * cellArea * inverseSqrt(rSq);
        }
    }

    coarsePhi[iy * COARSE + ix] = pot;
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
