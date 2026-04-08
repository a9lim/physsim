// ─── Scalar Field → Particle Forces ───
// One thread per particle. PQS-interpolated gradient forces + Higgs mass modulation.

// Struct definitions (ParticleState, ParticleAux, ParticleDerived, AllForces) provided by shared-structs.wgsl.

// Group 0: particleState (rw) + derived (rw)
// NOTE: particleAux.radius NOT updated here (would exceed 10 storage buffer limit).
// cacheDerived at start of next substep recomputes radius from updated mass.
@group(0) @binding(0) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(1) var<storage, read_write> derived: array<ParticleDerived>;

// Higgs field arrays (read_write for encoder compat)
@group(1) @binding(0) var<storage, read_write> higgsField: array<f32>;
@group(1) @binding(1) var<storage, read_write> higgsGradX: array<f32>;
@group(1) @binding(2) var<storage, read_write> higgsGradY: array<f32>;
// Axion field arrays (read_write for encoder compat)
@group(1) @binding(3) var<storage, read_write> axionField: array<f32>;
@group(1) @binding(4) var<storage, read_write> axionGradX: array<f32>;
@group(1) @binding(5) var<storage, read_write> axionGradY: array<f32>;

// Packed force accumulators + axYukMod output
@group(2) @binding(0) var<storage, read_write> allForces: array<AllForces>;
@group(2) @binding(1) var<storage, read_write> axYukMod: array<vec4<f32>>; // packed: axMod, yukMod, higgsMod, pad

@group(3) @binding(0) var<uniform> uniforms: FieldUniforms;

// PQS interpolation: returns field value at particle position
fn pqsInterpolate(fieldArr: ptr<storage, array<f32>, read_write>,
                  px: f32, py: f32, invCellW: f32, invCellH: f32,
                  bcMode: u32, topoMode: u32, vacValue: f32) -> f32 {
    let pqs = pqsWeights(px, py, invCellW, invCellH);
    var val: f32 = 0.0;

    if (isInterior(pqs.ix, pqs.iy)) {
        for (var jy = 0u; jy < 4u; jy++) {
            let wyj = pqs.wy[jy];
            let row = u32(pqs.iy + i32(jy) - 1) * GRID + u32(pqs.ix - 1);
            for (var jx = 0u; jx < 4u; jx++) {
                val += (*fieldArr)[row + jx] * pqs.wx[jx] * wyj;
            }
        }
    } else {
        for (var jy = 0u; jy < 4u; jy++) {
            let wyj = pqs.wy[jy];
            for (var jx = 0u; jx < 4u; jx++) {
                let idx = nbIndex(pqs.ix + i32(jx) - 1, pqs.iy + i32(jy) - 1, bcMode, topoMode);
                let fv = select(vacValue, (*fieldArr)[idx], idx >= 0);
                val += fv * pqs.wx[jx] * wyj;
            }
        }
    }
    return val;
}

// PQS gradient interpolation: returns (gx, gy) in world units
fn pqsGradient(gradXArr: ptr<storage, array<f32>, read_write>,
               gradYArr: ptr<storage, array<f32>, read_write>,
               px: f32, py: f32, invCellW: f32, invCellH: f32,
               bcMode: u32, topoMode: u32) -> vec2<f32> {
    let pqs = pqsWeights(px, py, invCellW, invCellH);
    var gx: f32 = 0.0;
    var gy: f32 = 0.0;

    if (isInterior(pqs.ix, pqs.iy)) {
        for (var jy = 0u; jy < 4u; jy++) {
            let wyj = pqs.wy[jy];
            let row = u32(pqs.iy + i32(jy) - 1) * GRID + u32(pqs.ix - 1);
            for (var jx = 0u; jx < 4u; jx++) {
                let w = pqs.wx[jx] * wyj;
                gx += (*gradXArr)[row + jx] * w;
                gy += (*gradYArr)[row + jx] * w;
            }
        }
    } else {
        for (var jy = 0u; jy < 4u; jy++) {
            let wyj = pqs.wy[jy];
            for (var jx = 0u; jx < 4u; jx++) {
                let idx = nbIndex(pqs.ix + i32(jx) - 1, pqs.iy + i32(jy) - 1, bcMode, topoMode);
                if (idx >= 0) {
                    let w = pqs.wx[jx] * wyj;
                    gx += (*gradXArr)[idx] * w;
                    gy += (*gradYArr)[idx] * w;
                }
            }
        }
    }
    return vec2<f32>(gx * invCellW, gy * invCellH);
}

// ─── Apply Higgs Forces + Mass Modulation ───
@compute @workgroup_size(256)
fn applyHiggsForces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= uniforms.particleCount) { return; }
    var p = particles[pid];
    let flag = p.flags;
    if ((flag & 1u) == 0u) { return; }

    let bm = p.baseMass;
    if (bm < EPSILON) { return; }

    let px = p.posX;
    let py = p.posY;
    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }
    let invCellW = 1.0 / cellW;
    let invCellH = 1.0 / cellH;
    let bcMode = uniforms.boundaryMode;
    let topoMode = uniforms.topologyMode;

    // Interpolate field value
    let phiLocal = pqsInterpolate(&higgsField, px, py, invCellW, invCellH, bcMode, topoMode, 1.0);

    // Cache higgsMod = max(|φ(x)|, floor) for Yukawa μ modulation
    let hmFloor = uniforms.higgsMassFloor;
    var aym = axYukMod[pid];
    aym.z = max(abs(phiLocal), hmFloor);
    axYukMod[pid] = aym;

    // ── Mass modulation ──
    let floor = hmFloor;
    let targetMass = max(bm * abs(phiLocal), floor * bm);
    let maxDelta = uniforms.higgsMassMaxDelta * uniforms.dt;
    let currentMass = p.mass;
    let diff = targetMass - currentMass;
    let clampedDiff = clamp(diff, -maxDelta, maxDelta);
    // Guard: never modulate below MIN_MASS to prevent div-by-zero
    let newMass = max(currentMass + clampedDiff, MIN_MASS);

    // Conserve momentum: scale proper velocity (newMass guaranteed > 0)
    let massRatio = currentMass / newMass;
    p.velWX *= massRatio;
    p.velWY *= massRatio;

    p.mass = newMass;
    particles[pid] = p;

    // Update derived quantities after mass change
    let bodyR = pow(newMass, 1.0 / 3.0);  // cbrt
    let bodyRSq = bodyR * bodyR;
    var d = derived[pid];
    d.invMass = select(0.0, 1.0 / newMass, newMass > EPSILON);
    d.radiusSq = bodyRSq;

    // Recompute coordinate velocity from scaled proper velocity
    let wSq = p.velWX * p.velWX + p.velWY * p.velWY;
    if (uniforms.relativityEnabled != 0u) {
        let gamma = sqrt(1.0 + wSq);
        d.velX = p.velWX / gamma;
        d.velY = p.velWY / gamma;
    } else {
        d.velX = p.velWX;
        d.velY = p.velWY;
    }

    // Recompute angular velocity from angW
    let sr = p.angW * bodyR;
    if (uniforms.relativityEnabled != 0u) {
        d.angVel = p.angW / sqrt(1.0 + sr * sr);
    } else {
        d.angVel = p.angW;
    }

    // Recompute cached dipole moments
    d.magMoment = MAG_MOMENT_K * p.charge * d.angVel * bodyRSq;
    d.angMomentum = INERTIA_K * newMass * bodyRSq * d.angVel;
    derived[pid] = d;

    // ── Gradient force: F = +g * baseMass * sign(phi) * grad(phi) ──
    let grad = pqsGradient(&higgsGradX, &higgsGradY, px, py, invCellW, invCellH, bcMode, topoMode);
    let signPhi = select(-1.0, 1.0, phiLocal >= 0.0);
    let g = uniforms.higgsCoupling;
    let fx = g * bm * signPhi * grad.x;
    let fy = g * bm * signPhi * grad.y;

    // Accumulate into higgs force slot (allForces.f4.zw) and totalForce
    var af = allForces[pid];
    af.f4.z += fx;
    af.f4.w += fy;
    af.totalForce.x += fx;
    af.totalForce.y += fy;
    allForces[pid] = af;
}

// ─── Apply Axion Forces + Modulation ───
@compute @workgroup_size(256)
fn applyAxionForces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= uniforms.particleCount) { return; }
    let p = particles[pid];
    let flag = p.flags;
    if ((flag & 1u) == 0u) { return; }

    let px = p.posX;
    let py = p.posY;
    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }
    let invCellW = 1.0 / cellW;
    let invCellH = 1.0 / cellH;
    let bcMode = uniforms.boundaryMode;
    let topoMode = uniforms.topologyMode;
    let g = uniforms.axionCoupling;

    // Interpolate axion field for axMod/yukMod
    let aLocal = pqsInterpolate(&axionField, px, py, invCellW, invCellH, bcMode, topoMode, 0.0);
    let ga = g * aLocal;

    // Read existing axYukMod (preserves higgsMod written by applyHiggsForces)
    var aymVal = axYukMod[pid];
    // axMod: scalar EM coupling
    if (uniforms.coulombEnabled != 0u) {
        aymVal.x = select(0.0, 1.0 + ga, ga > -1.0);
    } else {
        aymVal.x = 1.0;
    }

    // yukMod: pseudoscalar PQ coupling (flips for antimatter)
    if (uniforms.yukawaEnabled != 0u) {
        let isAnti = (flag & 4u) != 0u;
        let sign = select(1.0, -1.0, isAnti);
        let pq = sign * ga;
        aymVal.y = select(0.0, 1.0 + pq, pq > -1.0);
    } else {
        aymVal.y = 1.0;
    }
    // Gradient force
    var coupling: f32 = 0.0;
    if (uniforms.coulombEnabled != 0u) {
        let q = p.charge;
        let qSq = q * q;
        if (qSq > EPSILON) { coupling += qSq; }
    }
    if (uniforms.yukawaEnabled != 0u) {
        let m = p.mass;
        if (m > EPSILON) {
            let isAnti = (flag & 4u) != 0u;
            let sign = select(1.0, -1.0, isAnti);
            coupling += m * sign;
        }
    }
    // Superradiance torque for display (τ = rate / Ω_H, spin-down)
    // Computed before the coupling early-return: superradiance needs BH + axion only,
    // not Coulomb or Yukawa.
    // Also stores φ² in axYukMod.w for the deposit shader's stimulated amplification.
    var af = allForces[pid];
    if (uniforms.blackHoleEnabled != 0u) {
        let M = p.mass;
        if (M > MIN_MASS) {
            // Store local field amplitude squared for deposit shader
            aymVal.w = aLocal * aLocal;

            let bodyRSq = pow(M, 2.0 / 3.0);
            let angw = p.angW;
            let absAngw = abs(angw);
            let angvel = angw / sqrt(1.0 + absAngw * absAngw * bodyRSq);
            let a_sr = INERTIA_K * bodyRSq * abs(angvel);
            let disc = M * M - a_sr * a_sr - p.charge * p.charge;
            let rPlus = select(M, M + sqrt(max(0.0, disc)), disc >= 0.0);
            let rPlusSq = rPlus * rPlus;
            let sigma = rPlusSq + a_sr * a_sr;
            if (sigma >= EPSILON) {
                let omegaH = a_sr / sigma;
                let muA = uniforms.axionMass;
                if (omegaH > muA) {
                    let alphaG = M * muA;
                    let rate = SUPERRADIANCE_COEFF * alphaG * alphaG * (omegaH - muA) * (1.0 + aLocal * aLocal);
                    let signW = select(-1.0, 1.0, angw > 0.0);
                    af.f5.z = -signW * rate / omegaH;
                }
            }
        }
    }
    axYukMod[pid] = aymVal;

    if (abs(coupling) < EPSILON) {
        allForces[pid] = af;
        return;
    }

    let grad = pqsGradient(&axionGradX, &axionGradY, px, py, invCellW, invCellH, bcMode, topoMode);
    let fx = g * coupling * grad.x;
    let fy = g * coupling * grad.y;

    af.f5.x += fx;
    af.f5.y += fy;
    af.totalForce.x += fx;
    af.totalForce.y += fy;
    allForces[pid] = af;
}
