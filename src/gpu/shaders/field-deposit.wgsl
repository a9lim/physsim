// ─── PQS Single-Pass Atomic Deposition ───
// One thread per particle. Computes PQS weights and atomically deposits
// into the target grid using fixed-point i32 encoding (FP_SCALE = 2^20).
// Axion sources can be negative (PQ coupling), so we use atomic<i32>.
// A separate finalize pass converts atomic i32 → f32.

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<ParticleState>;

@group(1) @binding(0) var<storage, read_write> atomicGrid: array<atomic<i32>>;  // GRID_SQ
@group(1) @binding(1) var<storage, read_write> targetGrid: array<f32>;          // GRID_SQ (f32 output)
@group(1) @binding(2) var<uniform> uniforms: FieldUniforms;

// Fixed-point scale: 2^20 ≈ 1M. Gives ~6 decimal digits of precision.
// Max representable value: 2^31 / 2^20 ≈ 2048 — well above any deposit sum.
const FP_SCALE: f32 = 1048576.0;  // 2^20
const INV_FP_SCALE: f32 = 1.0 / 1048576.0;

// Include field-common.wgsl utilities inline (WGSL has no #include; JS concatenates)
// PQS weights function, nbIndex, constants — assumed prepended by JS

// ─── Atomic PQS deposit helper ───
fn atomicDeposit(pqs: PQSResult, value: f32, bcMode: u32, topoMode: u32) {
    let ix = pqs.ix;
    let iy = pqs.iy;

    // Interior fast path: stencil [ix-1..ix+2]×[iy-1..iy+2] fully inside grid
    if (isInterior(ix, iy)) {
        for (var jy = 0u; jy < 4u; jy++) {
            let vwy = value * pqs.wy[jy];
            let row = u32(iy + i32(jy) - 1) * GRID + u32(ix - 1);
            for (var jx = 0u; jx < 4u; jx++) {
                let w = vwy * pqs.wx[jx];
                let fixed = i32(w * FP_SCALE);
                if (fixed != 0) {
                    atomicAdd(&atomicGrid[row + jx], fixed);
                }
            }
        }
        return;
    }

    // Border path: use nbIndex for boundary-aware wrapping
    for (var jy = 0u; jy < 4u; jy++) {
        let vwy = value * pqs.wy[jy];
        for (var jx = 0u; jx < 4u; jx++) {
            let idx = nbIndex(ix + i32(jx) - 1, iy + i32(jy) - 1, bcMode, topoMode);
            if (idx >= 0) {
                let w = vwy * pqs.wx[jx];
                let fixed = i32(w * FP_SCALE);
                if (fixed != 0) {
                    atomicAdd(&atomicGrid[idx], fixed);
                }
            }
        }
    }
}

// ─── Higgs Source Deposit ───
// Deposits g * baseMass per particle.
@compute @workgroup_size(256)
fn depositHiggsSource(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= uniforms.particleCount) { return; }

    let p = particles[pid];
    if ((p.flags & 1u) == 0u) { return; }

    let bm = p.baseMass;
    if (bm < EPSILON) { return; }

    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }

    let pqs = pqsWeights(p.posX, p.posY, 1.0 / cellW, 1.0 / cellH);
    let value = uniforms.higgsCoupling * bm;
    atomicDeposit(pqs, value, uniforms.boundaryMode, uniforms.topologyMode);
}

// ─── Axion Source Deposit ───
// Deposits g*q² (EM, when Coulomb on) + g*m*sign (PQ, when Yukawa on).
// Can be negative (PQ coupling flips for antimatter).
@compute @workgroup_size(256)
fn depositAxionSource(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= uniforms.particleCount) { return; }

    let p = particles[pid];
    let flag = p.flags;
    if ((flag & 1u) == 0u) { return; }

    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }

    let g = uniforms.axionCoupling;
    var value: f32 = 0.0;
    if (uniforms.coulombEnabled != 0u) {
        let qSq = p.charge * p.charge;
        if (qSq > EPSILON) { value += g * qSq; }
    }
    if (uniforms.yukawaEnabled != 0u) {
        let m = p.mass;
        if (m > EPSILON) {
            let isAntimatter = (flag & 4u) != 0u;
            let sign = select(1.0, -1.0, isAntimatter);
            value += g * m * sign;
        }
    }
    if (abs(value) < EPSILON) { return; }

    let pqs = pqsWeights(p.posX, p.posY, 1.0 / cellW, 1.0 / cellH);
    atomicDeposit(pqs, value, uniforms.boundaryMode, uniforms.topologyMode);
}

// ─── Thermal KE Deposit (Higgs phase transitions) ───
@compute @workgroup_size(256)
fn depositThermal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= uniforms.particleCount) { return; }

    let p = particles[pid];
    if ((p.flags & 1u) == 0u) { return; }

    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }

    let wSq = p.velWX * p.velWX + p.velWY * p.velWY;
    var ke: f32;
    if (uniforms.relativityEnabled != 0u) {
        ke = wSq / (sqrt(1.0 + wSq) + 1.0) * p.mass;
    } else {
        ke = 0.5 * p.mass * wSq;
    }
    if (ke < EPSILON) { return; }

    let pqs = pqsWeights(p.posX, p.posY, 1.0 / cellW, 1.0 / cellH);
    atomicDeposit(pqs, ke, uniforms.boundaryMode, uniforms.topologyMode);
}

// ─── Finalize: atomic i32 → f32, then clear atomic grid ───
@compute @workgroup_size(8, 8)
fn finalizeDeposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * GRID + gid.x;
    if (gid.x >= GRID || gid.y >= GRID) { return; }
    targetGrid[idx] = f32(atomicExchange(&atomicGrid[idx], 0)) * INV_FP_SCALE;
}
