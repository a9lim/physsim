// ─── Scalar Field → Particle Forces ───
// One thread per particle. PQS-interpolated gradient forces + Higgs mass modulation.

@group(0) @binding(0) var<storage, read> posX: array<f32>;
@group(0) @binding(1) var<storage, read> posY: array<f32>;
@group(0) @binding(2) var<storage, read_write> mass: array<f32>;
@group(0) @binding(3) var<storage, read> baseMass: array<f32>;
@group(0) @binding(4) var<storage, read> charge: array<f32>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> velWX: array<f32>;
@group(0) @binding(7) var<storage, read_write> velWY: array<f32>;
@group(0) @binding(8) var<storage, read_write> angW: array<f32>;
@group(0) @binding(9) var<storage, read_write> radius: array<f32>;
@group(0) @binding(10) var<storage, read_write> invMassRadSq: array<vec2<f32>>; // packed: invMass, radiusSq

// Higgs field arrays
@group(1) @binding(0) var<storage, read> higgsField: array<f32>;
@group(1) @binding(1) var<storage, read> higgsGradX: array<f32>;
@group(1) @binding(2) var<storage, read> higgsGradY: array<f32>;
// Axion field arrays
@group(1) @binding(3) var<storage, read> axionField: array<f32>;
@group(1) @binding(4) var<storage, read> axionGradX: array<f32>;
@group(1) @binding(5) var<storage, read> axionGradY: array<f32>;

// Force accumulators (forces4 = external.xy, higgs.xy; forces5 = axion.xy, pad, pad)
@group(2) @binding(0) var<storage, read_write> forces4: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> forces5: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read_write> totalForce: array<vec2<f32>>;
// axMod/yukMod output
@group(2) @binding(3) var<storage, read_write> axMod: array<f32>;
@group(2) @binding(4) var<storage, read_write> yukMod: array<f32>;

@group(3) @binding(0) var<uniform> uniforms: FieldUniforms;

// PQS interpolation: returns field value at particle position
fn pqsInterpolate(fieldArr: ptr<storage, array<f32>, read>,
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
fn pqsGradient(gradXArr: ptr<storage, array<f32>, read>,
               gradYArr: ptr<storage, array<f32>, read>,
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
    let flag = flags[pid];
    if ((flag & 1u) == 0u) { return; }

    let bm = baseMass[pid];
    if (bm < EPSILON) { return; }

    let px = posX[pid];
    let py = posY[pid];
    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }
    let invCellW = 1.0 / cellW;
    let invCellH = 1.0 / cellH;
    let bcMode = uniforms.boundaryMode;
    let topoMode = uniforms.topologyMode;

    // Interpolate field value
    let phiLocal = pqsInterpolate(&higgsField, px, py, invCellW, invCellH, bcMode, topoMode, 1.0);

    // ── Mass modulation ──
    let floor = uniforms.higgsMassFloor;
    let targetMass = max(bm * abs(phiLocal), floor * bm);
    let maxDelta = uniforms.higgsMassMaxDelta * uniforms.dt;
    let currentMass = mass[pid];
    let diff = targetMass - currentMass;
    let clampedDiff = clamp(diff, -maxDelta, maxDelta);
    let newMass = currentMass + clampedDiff;

    // Conserve momentum: scale proper velocity
    let massRatio = currentMass / newMass;
    velWX[pid] *= massRatio;
    velWY[pid] *= massRatio;

    mass[pid] = newMass;
    let bodyR = pow(newMass, 1.0 / 3.0);  // cbrt
    radius[pid] = bodyR;
    var imrs = invMassRadSq[pid];
    imrs.x = 1.0 / newMass;
    invMassRadSq[pid] = imrs;

    // ── Gradient force: F = +g * baseMass * sign(phi) * grad(phi) ──
    let grad = pqsGradient(&higgsGradX, &higgsGradY, px, py, invCellW, invCellH, bcMode, topoMode);
    let signPhi = select(-1.0, 1.0, phiLocal >= 0.0);
    let g = uniforms.higgsCoupling;
    let fx = g * bm * signPhi * grad.x;
    let fy = g * bm * signPhi * grad.y;

    // Accumulate into higgs force slot (forces4.zw)
    var f4 = forces4[pid];
    f4.z += fx;
    f4.w += fy;
    forces4[pid] = f4;

    var tf = totalForce[pid];
    tf.x += fx;
    tf.y += fy;
    totalForce[pid] = tf;
}

// ─── Apply Axion Forces + Modulation ───
@compute @workgroup_size(256)
fn applyAxionForces(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    let bcMode = uniforms.boundaryMode;
    let topoMode = uniforms.topologyMode;
    let g = uniforms.axionCoupling;

    // Interpolate axion field for axMod/yukMod
    let aLocal = pqsInterpolate(&axionField, px, py, invCellW, invCellH, bcMode, topoMode, 0.0);
    let ga = g * aLocal;

    // axMod: scalar EM coupling
    if (uniforms.coulombEnabled != 0u) {
        axMod[pid] = select(0.0, 1.0 + ga, ga > -1.0);
    } else {
        axMod[pid] = 1.0;
    }

    // yukMod: pseudoscalar PQ coupling (flips for antimatter)
    if (uniforms.yukawaEnabled != 0u) {
        let isAnti = (flag & 4u) != 0u;
        let sign = select(1.0, -1.0, isAnti);
        let pq = sign * ga;
        yukMod[pid] = select(0.0, 1.0 + pq, pq > -1.0);
    } else {
        yukMod[pid] = 1.0;
    }

    // Gradient force
    var coupling: f32 = 0.0;
    if (uniforms.coulombEnabled != 0u) {
        let q = charge[pid];
        let qSq = q * q;
        if (qSq > EPSILON) { coupling += qSq; }
    }
    if (uniforms.yukawaEnabled != 0u) {
        let m = mass[pid];
        if (m > EPSILON) {
            let isAnti = (flag & 4u) != 0u;
            let sign = select(1.0, -1.0, isAnti);
            coupling += m * sign;
        }
    }
    if (abs(coupling) < EPSILON) { return; }

    let grad = pqsGradient(&axionGradX, &axionGradY, px, py, invCellW, invCellH, bcMode, topoMode);
    let fx = g * coupling * grad.x;
    let fy = g * coupling * grad.y;

    // Accumulate into axion force slot (forces5.xy)
    var f5 = forces5[pid];
    f5.x += fx;
    f5.y += fy;
    forces5[pid] = f5;

    var tf = totalForce[pid];
    tf.x += fx;
    tf.y += fy;
    totalForce[pid] = tf;
}
