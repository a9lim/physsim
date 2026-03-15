// ─── Scalar Field → Particle Forces ───
// One thread per particle. PQS-interpolated gradient forces + Higgs mass modulation.

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState_FF {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

// Packed auxiliary struct (matches common.wgsl ParticleAux)
struct ParticleAux_FF {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
};

// Packed struct (mirrors common.wgsl ParticleDerived)
struct ParticleDerived_FF {
    magMoment: f32,
    angMomentum: f32,
    invMass: f32,
    radiusSq: f32,
    velX: f32,
    velY: f32,
    angVel: f32,
    _pad: f32,
};

// Packed struct (mirrors common.wgsl AllForces)
struct AllForces_FF {
    f0: vec4<f32>,
    f1: vec4<f32>,
    f2: vec4<f32>,
    f3: vec4<f32>,
    f4: vec4<f32>,
    f5: vec4<f32>,
    torques: vec4<f32>,
    bFields: vec4<f32>,
    bFieldGrads: vec4<f32>,
    totalForce: vec2<f32>,
    _pad: vec2<f32>,
};

// Group 0: particleState (rw) + derived (rw)
// NOTE: particleAux.radius NOT updated here (would exceed 10 storage buffer limit).
// cacheDerived at start of next substep recomputes radius from updated mass.
@group(0) @binding(0) var<storage, read_write> particles: array<ParticleState_FF>;
@group(0) @binding(1) var<storage, read_write> derived: array<ParticleDerived_FF>;

// Higgs field arrays (read_write for encoder compat)
@group(1) @binding(0) var<storage, read_write> higgsField: array<f32>;
@group(1) @binding(1) var<storage, read_write> higgsGradX: array<f32>;
@group(1) @binding(2) var<storage, read_write> higgsGradY: array<f32>;
// Axion field arrays (read_write for encoder compat)
@group(1) @binding(3) var<storage, read_write> axionField: array<f32>;
@group(1) @binding(4) var<storage, read_write> axionGradX: array<f32>;
@group(1) @binding(5) var<storage, read_write> axionGradY: array<f32>;

// Packed force accumulators + axYukMod output
@group(2) @binding(0) var<storage, read_write> allForces: array<AllForces_FF>;
@group(2) @binding(1) var<storage, read_write> axYukMod: array<vec2<f32>>; // packed: axMod, yukMod

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

    // ── Mass modulation ──
    let floor = uniforms.higgsMassFloor;
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
    d.magMoment = 0.2 * p.charge * d.angVel * bodyRSq;
    d.angMomentum = 0.4 * newMass * bodyRSq * d.angVel;
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

    // axMod: scalar EM coupling
    var aymVal = vec2<f32>(1.0, 1.0);
    if (uniforms.coulombEnabled != 0u) {
        aymVal.x = select(0.0, 1.0 + ga, ga > -1.0);
    }

    // yukMod: pseudoscalar PQ coupling (flips for antimatter)
    if (uniforms.yukawaEnabled != 0u) {
        let isAnti = (flag & 4u) != 0u;
        let sign = select(1.0, -1.0, isAnti);
        let pq = sign * ga;
        aymVal.y = select(0.0, 1.0 + pq, pq > -1.0);
    }
    axYukMod[pid] = aymVal;

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
    if (abs(coupling) < EPSILON) { return; }

    let grad = pqsGradient(&axionGradX, &axionGradY, px, py, invCellW, invCellH, bcMode, topoMode);
    let fx = g * coupling * grad.x;
    let fy = g * coupling * grad.y;

    // Accumulate into axion force slot (allForces.f5.xy) and totalForce
    var af = allForces[pid];
    af.f5.x += fx;
    af.f5.y += fy;
    af.totalForce.x += fx;
    af.totalForce.y += fy;
    allForces[pid] = af;
}
