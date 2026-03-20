// compute-stats.wgsl — Parallel stats reduction: KE, PE, momentum, angular momentum,
// Darwin field energy/momentum, scalar field energy/momentum, selected particle readback.
//
// Four entry points dispatched sequentially:
//   statsKEMom:  (1,1,1) @workgroup_size(64) — Passes 1+2: KE, momentum, COM, angular momentum
//   statsPE:     (1,1,1) @workgroup_size(1)  — Pass 3: PE + Darwin field energy (O(N²))
//   statsField:  (1,1,1) @workgroup_size(64) — Pass 4: Scalar field energy + momentum
//   statsPFISel: (1,1,1) @workgroup_size(1)  — Passes 5+6: PFI + selected particle
//
// Constants from wgslConstants: INERTIA_K, SOFTENING_SQ, BH_SOFTENING_SQ, YUKAWA_COUPLING,
//   FLAG_ALIVE, FLAG_ANTIMATTER, ONE_PN_BIT, GRAVITY_BIT, COULOMB_BIT, MAGNETIC_BIT,
//   GRAVITOMAG_BIT, YUKAWA_BIT, HIGGS_BIT, AXION_BIT, BLACK_HOLE_BIT, AXION_COUPLING

struct StatsUniforms {
    aliveCount: u32,
    selectedIdx: i32,
    toggles0: u32,       // physics toggle bitfield
    domainW: f32,        // 16
    domainH: f32,
    yukawaMu: f32,
    higgsMass: f32,
    axionMass: f32,      // 32
    fieldGridRes: u32,
    boundaryMode: u32,   // BOUND_DESPAWN/BOUND_BOUNCE/BOUND_LOOP
    topologyMode: u32,   // TOPO_TORUS/TOPO_KLEIN/TOPO_RP2
    _pad3: u32,          // 48
};

// Struct definitions (ParticleState, ParticleDerived, AllForces) provided by shared-structs.wgsl.

struct AxYukMod {
    axMod: f32,
    yukMod: f32,
    higgsMod: f32,
    _pad: f32,
};

// Group 0: particle data
@group(0) @binding(0) var<uniform> params: StatsUniforms;
@group(0) @binding(1) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read> derived: array<ParticleDerived>;
@group(0) @binding(3) var<storage, read> forces: array<AllForces>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;
@group(0) @binding(5) var<storage, read> axYukMod: array<AxYukMod>;

// Group 1: scalar field grids (optional — bound to dummy 4-byte buffers when field inactive)
@group(1) @binding(0) var<storage, read> higgsField: array<f32>;
@group(1) @binding(1) var<storage, read> higgsFieldDot: array<f32>;
@group(1) @binding(2) var<storage, read> axionField: array<f32>;
@group(1) @binding(3) var<storage, read> axionFieldDot: array<f32>;

// Output layout (128 f32 = 512 bytes):
// [0]  linearKE      [1]  spinKE       [2]  px           [3]  py
// [4]  orbAngMom     [5]  spinAngMom   [6]  comX         [7]  comY
// [8]  totalMass     [9]  aliveCount   [10] pe           [11] darwinFieldE
// [12] darwinFieldPx [13] darwinFieldPy [14] higgsFieldE [15] axionFieldE
// [16] higgsPfiE     [17] axionPfiE    [18] fieldMomX    [19] fieldMomY
// Selected particle [32-85]:
//   [32] posX .. [85] axion force (same layout as before, shifted from 16→32)

fn toggle(bit: u32) -> bool { return (params.toggles0 & bit) != 0u; }

// ═══════════════════════════════════════════════════════════════════════════
// Entry 1: statsKEMom — Parallel KE, momentum, COM, angular momentum
// Dispatch (1,1,1) with workgroup_size(64). Each thread processes N/64 particles.
// Two-phase: first accumulate KE+momentum+COM, barrier, then angular momentum about COM.
// ═══════════════════════════════════════════════════════════════════════════

const KE_WG: u32 = 64u;

// Shared memory for tree reduction (9 channels for pass 1, 2 for pass 2)
var<workgroup> sh_linearKE: array<f32, KE_WG>;
var<workgroup> sh_spinKE: array<f32, KE_WG>;
var<workgroup> sh_px: array<f32, KE_WG>;
var<workgroup> sh_py: array<f32, KE_WG>;
var<workgroup> sh_totalMass: array<f32, KE_WG>;
var<workgroup> sh_comX: array<f32, KE_WG>;
var<workgroup> sh_comY: array<f32, KE_WG>;
var<workgroup> sh_orbAngMom: array<f32, KE_WG>;
var<workgroup> sh_spinAngMom: array<f32, KE_WG>;

// Broadcast COM from thread 0 to all threads
var<workgroup> sh_comXFinal: f32;
var<workgroup> sh_comYFinal: f32;

@compute @workgroup_size(KE_WG)
fn statsKEMom(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let n = params.aliveCount;
    let relOn = toggle(RELATIVITY_BIT);

    // ─── Pass 1: Each thread accumulates KE, momentum, COM for its partition ───
    var localLinearKE: f32 = 0.0;
    var localSpinKE: f32 = 0.0;
    var localPx: f32 = 0.0;
    var localPy: f32 = 0.0;
    var localTotalMass: f32 = 0.0;
    var localComX: f32 = 0.0;
    var localComY: f32 = 0.0;

    // Strided loop: thread tid processes particles tid, tid+64, tid+128, ...
    var i = tid;
    while (i < n) {
        let p = particles[i];
        if ((p.flags & FLAG_ALIVE) != 0u) {
            let d = derived[i];
            let m = p.mass;
            if (relOn) {
                let wSq = p.velWX * p.velWX + p.velWY * p.velWY;
                let gamma = sqrt(1.0 + wSq);
                localLinearKE += wSq / (gamma + 1.0) * m;
                let srSq = p.angW * p.angW * d.radiusSq;
                let gammaRot = sqrt(1.0 + srSq);
                localSpinKE += INERTIA_K * m * srSq / (gammaRot + 1.0);
            } else {
                let vSq = d.velX * d.velX + d.velY * d.velY;
                localLinearKE += 0.5 * m * vSq;
                localSpinKE += 0.5 * INERTIA_K * m * d.radiusSq * d.angVel * d.angVel;
            }
            localPx += m * p.velWX;
            localPy += m * p.velWY;
            localTotalMass += m;
            localComX += m * p.posX;
            localComY += m * p.posY;
        }
        i += KE_WG;
    }

    // Store to shared memory
    sh_linearKE[tid] = localLinearKE;
    sh_spinKE[tid] = localSpinKE;
    sh_px[tid] = localPx;
    sh_py[tid] = localPy;
    sh_totalMass[tid] = localTotalMass;
    sh_comX[tid] = localComX;
    sh_comY[tid] = localComY;
    workgroupBarrier();

    // Tree reduction (log2(64) = 6 steps)
    for (var stride = KE_WG >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            sh_linearKE[tid] += sh_linearKE[tid + stride];
            sh_spinKE[tid] += sh_spinKE[tid + stride];
            sh_px[tid] += sh_px[tid + stride];
            sh_py[tid] += sh_py[tid + stride];
            sh_totalMass[tid] += sh_totalMass[tid + stride];
            sh_comX[tid] += sh_comX[tid + stride];
            sh_comY[tid] += sh_comY[tid + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 computes final COM and broadcasts
    if (tid == 0u) {
        let tm = sh_totalMass[0];
        var cx = sh_comX[0];
        var cy = sh_comY[0];
        if (tm > 0.0) { cx /= tm; cy /= tm; }
        sh_comXFinal = cx;
        sh_comYFinal = cy;

        // Write pass 1 results
        stats[0] = sh_linearKE[0];
        stats[1] = sh_spinKE[0];
        stats[2] = sh_px[0];
        stats[3] = sh_py[0];
        stats[8] = tm;
        stats[9] = f32(n);
        stats[6] = cx;
        stats[7] = cy;
    }
    workgroupBarrier();

    // ─── Pass 2: Angular momentum about COM ───
    let comXF = sh_comXFinal;
    let comYF = sh_comYFinal;

    var localOrbAngMom: f32 = 0.0;
    var localSpinAngMom: f32 = 0.0;
    i = tid;
    while (i < n) {
        let p = particles[i];
        if ((p.flags & FLAG_ALIVE) != 0u) {
            let d = derived[i];
            let m = p.mass;
            localOrbAngMom += (p.posX - comXF) * (m * p.velWY) - (p.posY - comYF) * (m * p.velWX);
            localSpinAngMom += INERTIA_K * m * d.radiusSq * p.angW;
        }
        i += KE_WG;
    }

    sh_orbAngMom[tid] = localOrbAngMom;
    sh_spinAngMom[tid] = localSpinAngMom;
    workgroupBarrier();

    for (var stride = KE_WG >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            sh_orbAngMom[tid] += sh_orbAngMom[tid + stride];
            sh_spinAngMom[tid] += sh_spinAngMom[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        stats[4] = sh_orbAngMom[0];
        stats[5] = sh_spinAngMom[0];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry 2: statsPE — O(N²) pairwise PE + Darwin field energy
// Dispatch (1,1,1) with workgroup_size(1). Unchanged from original.
// ═══════════════════════════════════════════════════════════════════════════

@compute @workgroup_size(1)
fn statsPE() {
    let n = params.aliveCount;
    let gravOn = toggle(GRAVITY_BIT);
    let coulOn = toggle(COULOMB_BIT);
    let magOn = toggle(MAGNETIC_BIT);
    let gmOn = toggle(GRAVITOMAG_BIT);
    let onePNOn = toggle(ONE_PN_BIT);
    let yukOn = toggle(YUKAWA_BIT);
    let higgsOn = toggle(HIGGS_BIT);
    let bhOn = toggle(BLACK_HOLE_BIT);
    let softeningSq = select(SOFTENING_SQ, BH_SOFTENING_SQ, bhOn);
    let yukMu = params.yukawaMu;

    var pe: f32 = 0.0;
    var darwinE: f32 = 0.0;
    var darwinPx: f32 = 0.0;
    var darwinPy: f32 = 0.0;

    for (var i = 0u; i < n; i++) {
        let pi = particles[i];
        if ((pi.flags & FLAG_ALIVE) == 0u) { continue; }
        let di = derived[i];
        let modi = axYukMod[i];
        for (var j = i + 1u; j < n; j++) {
            let pj = particles[j];
            if ((pj.flags & FLAG_ALIVE) == 0u) { continue; }
            let dj = derived[j];
            let modj = axYukMod[j];

            var dx = pj.posX - pi.posX;
            var dy = pj.posY - pi.posY;
            if (params.boundaryMode == BOUND_LOOP) {
                let mi = fullMinImageP(pi.posX, pi.posY, pj.posX, pj.posY,
                                       params.domainW, params.domainH, params.topologyMode);
                dx = mi.x; dy = mi.y;
            }
            let rSq = dx * dx + dy * dy + softeningSq;
            let invR = 1.0 / sqrt(rSq);
            let r = rSq * invR; // = sqrt(rSq)
            let invR3 = invR * invR * invR;
            let rx = dx * invR;
            let ry = dy * invR;

            let axModPair = sqrt(max(modi.axMod * modj.axMod, 0.0));
            let yukModPair = sqrt(max(modi.yukMod * modj.yukMod, 0.0));

            // Gravity PE
            if (gravOn) {
                pe -= pi.mass * pj.mass * invR;
            }

            // Coulomb PE
            if (coulOn) {
                pe += pi.charge * pj.charge * invR * axModPair;
            }

            // Magnetic dipole PE
            if (magOn) {
                pe += di.magMoment * dj.magMoment * invR3 * axModPair;
            }

            // GM dipole PE
            if (gmOn) {
                pe -= di.angMomentum * dj.angMomentum * invR3;
            }

            // Velocity-dependent terms
            let viDotVj = di.velX * dj.velX + di.velY * dj.velY;
            let viDotR = di.velX * rx + di.velY * ry;
            let vjDotR = dj.velX * rx + dj.velY * ry;
            let velTerm = viDotVj + viDotR * vjDotR;
            let sumVx = di.velX + dj.velX;
            let sumVy = di.velY + dj.velY;
            let svDotR = sumVx * rx + sumVy * ry;

            // 1PN terms (replace Darwin field energy when active)
            if (onePNOn) {
                // EIH
                if (gmOn) {
                    let viSq = di.velX * di.velX + di.velY * di.velY;
                    let vjSq = dj.velX * dj.velX + dj.velY * dj.velY;
                    pe -= pi.mass * pj.mass * invR * (
                        1.5 * (viSq + vjSq) - 3.5 * viDotVj - 0.5 * viDotR * vjDotR
                        + (pi.mass + pj.mass) * invR
                    );
                }
                // Darwin EM
                if (magOn) {
                    pe -= 0.5 * pi.charge * pj.charge * invR * velTerm;
                }
                // Bazanski
                if (magOn && gmOn) {
                    let crossCoeff = pi.charge * pj.charge * (pi.mass + pj.mass)
                        - (pi.charge * pi.charge * pj.mass + pj.charge * pj.charge * pi.mass);
                    pe += 0.5 * crossCoeff * invR * invR;
                }
            } else {
                // Darwin field energy (when 1PN off)
                if (magOn) {
                    let qqInvR = pi.charge * pj.charge * invR * axModPair;
                    darwinE -= 0.5 * qqInvR * velTerm;
                    let coeff = qqInvR * 0.5;
                    darwinPx += coeff * (sumVx + rx * svDotR);
                    darwinPy += coeff * (sumVy + ry * svDotR);
                }
                if (gmOn) {
                    let mmInvR = pi.mass * pj.mass * invR;
                    darwinE += 0.5 * mmInvR * velTerm;
                    let coeff = mmInvR * 0.5;
                    darwinPx -= coeff * (sumVx + rx * svDotR);
                    darwinPy -= coeff * (sumVy + ry * svDotR);
                }
                // Bazanski cross-term
                if (magOn && gmOn) {
                    let crossCoeff = pi.charge * pj.charge * (pi.mass + pj.mass)
                        - (pi.charge * pi.charge * pj.mass + pj.charge * pj.charge * pi.mass);
                    darwinE += 0.5 * crossCoeff * invR * invR;
                }
            }

            // Yukawa PE (Higgs-modulated μ when both enabled)
            if (yukOn) {
                let muEff = select(yukMu, yukMu * sqrt(modi.higgsMod * modj.higgsMod), higgsOn);
                let mur = muEff * r;
                if (mur < 6.0) {
                    let yukPE = -YUKAWA_COUPLING * yukModPair * pi.mass * pj.mass * exp(-mur) * invR;
                    pe += yukPE;
                    // Scalar Breit (Yukawa 1PN)
                    if (onePNOn) {
                        pe += 0.5 * YUKAWA_COUPLING * yukModPair * pi.mass * pj.mass * exp(-mur) * invR
                            * (viDotVj + viDotR * vjDotR * (1.0 + mur));
                    }
                }
            }
        }
    }

    stats[10] = pe;
    stats[11] = darwinE;
    stats[12] = darwinPx;
    stats[13] = darwinPy;
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry 3: statsField — Parallel scalar field energy + momentum
// Dispatch (1,1,1) with workgroup_size(64). Each thread processes GRID_SQ/64 cells.
// ═══════════════════════════════════════════════════════════════════════════

const FLD_WG: u32 = 64u;

var<workgroup> sh_higgsE: array<f32, FLD_WG>;
var<workgroup> sh_axionE: array<f32, FLD_WG>;
var<workgroup> sh_fmx: array<f32, FLD_WG>;
var<workgroup> sh_fmy: array<f32, FLD_WG>;

@compute @workgroup_size(FLD_WG)
fn statsField(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let higgsOn = toggle(HIGGS_BIT);
    let axionOn = toggle(AXION_BIT);
    let FGRID = params.fieldGridRes;
    let FGRID_SQ = FGRID * FGRID;

    var localHiggsE: f32 = 0.0;
    var localAxionE: f32 = 0.0;
    var localFmx: f32 = 0.0;
    var localFmy: f32 = 0.0;

    if ((higgsOn || axionOn) && FGRID > 0u) {
        let cellW = params.domainW / f32(FGRID);
        let cellH = params.domainH / f32(FGRID);
        let cellArea = cellW * cellH;
        let invCellWSq = 1.0 / (cellW * cellW);
        let invCellHSq = 1.0 / (cellH * cellH);
        let mH = params.higgsMass;
        let muSqH = 0.5 * mH * mH;
        let vacOffsetH = 0.25 * muSqH;
        let mA = params.axionMass;
        let mASq = mA * mA;
        let scaleX = cellH * 0.5;
        let scaleY = cellW * 0.5;

        // Strided loop: thread tid processes cells tid, tid+64, tid+128, ...
        var idx = tid;
        while (idx < FGRID_SQ) {
            let ix = idx % FGRID;
            let iy = idx / FGRID;
            let ixp = min(ix + 1u, FGRID - 1u);
            let iyp = min(iy + 1u, FGRID - 1u);
            let ixm = select(ix - 1u, 0u, ix == 0u);
            let iym = select(iy - 1u, 0u, iy == 0u);

            // Higgs field energy
            if (higgsOn) {
                let phi = higgsField[idx];
                let phiDot = higgsFieldDot[idx];
                let dfx = higgsField[iyp * FGRID + ix] - higgsField[iym * FGRID + ix];
                let dfy = higgsField[iy * FGRID + ixp] - higgsField[iy * FGRID + ixm];
                let ke = 0.5 * phiDot * phiDot;
                let grad = 0.5 * (dfx * dfx * invCellHSq + dfy * dfy * invCellWSq) * 0.25;
                let pot = muSqH * (-0.5 * phi * phi + 0.25 * phi * phi * phi * phi) + vacOffsetH;
                localHiggsE += (ke + grad + pot) * cellArea;
                localFmx -= phiDot * dfy * scaleX;
                localFmy -= phiDot * dfx * scaleY;
            }

            // Axion field energy
            if (axionOn) {
                let a = axionField[idx];
                let aDot = axionFieldDot[idx];
                let dfx = axionField[iyp * FGRID + ix] - axionField[iym * FGRID + ix];
                let dfy = axionField[iy * FGRID + ixp] - axionField[iy * FGRID + ixm];
                let ke = 0.5 * aDot * aDot;
                let grad = 0.5 * (dfx * dfx * invCellHSq + dfy * dfy * invCellWSq) * 0.25;
                let pot = 0.5 * mASq * a * a;
                localAxionE += (ke + grad + pot) * cellArea;
                localFmx -= aDot * dfy * scaleX;
                localFmy -= aDot * dfx * scaleY;
            }

            // Portal coupling energy: ½λφ²a² (counted in Higgs to match CPU)
            if (higgsOn && axionOn) {
                let phi = higgsField[idx];
                let a = axionField[idx];
                localHiggsE += 0.5 * HIGGS_AXION_COUPLING * phi * phi * a * a * cellArea;
            }

            idx += FLD_WG;
        }
    }

    // Store to shared memory and reduce
    sh_higgsE[tid] = localHiggsE;
    sh_axionE[tid] = localAxionE;
    sh_fmx[tid] = localFmx;
    sh_fmy[tid] = localFmy;
    workgroupBarrier();

    for (var stride = FLD_WG >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            sh_higgsE[tid] += sh_higgsE[tid + stride];
            sh_axionE[tid] += sh_axionE[tid + stride];
            sh_fmx[tid] += sh_fmx[tid + stride];
            sh_fmy[tid] += sh_fmy[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        stats[14] = sh_higgsE[0];
        stats[15] = sh_axionE[0];
        stats[18] = sh_fmx[0];
        stats[19] = sh_fmy[0];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry 4: statsPFISel — PFI (O(N×16) PQS) + selected particle copy
// Dispatch (1,1,1) with workgroup_size(1). Unchanged from original.
// ═══════════════════════════════════════════════════════════════════════════

@compute @workgroup_size(1)
fn statsPFISel() {
    let n = params.aliveCount;
    let coulOn = toggle(COULOMB_BIT);
    let yukOn = toggle(YUKAWA_BIT);
    let higgsOn = toggle(HIGGS_BIT);
    let axionOn = toggle(AXION_BIT);
    let FGRID = params.fieldGridRes;

    // ─── Pass 5: Particle-field interaction energy (PQS interpolation) ───
    var higgsPfiE: f32 = 0.0;
    var axionPfiE: f32 = 0.0;

    if ((higgsOn || axionOn) && FGRID > 0u) {
        let cellW = params.domainW / f32(FGRID);
        let cellH = params.domainH / f32(FGRID);
        let invCellW = 1.0 / cellW;
        let invCellH = 1.0 / cellH;

        for (var i = 0u; i < n; i++) {
            let p = particles[i];
            if ((p.flags & FLAG_ALIVE) == 0u) { continue; }
            // PQS grid coords
            let gx = p.posX * invCellW - 0.5;
            let gy = p.posY * invCellH - 0.5;
            let ix0 = i32(floor(gx));
            let iy0 = i32(floor(gy));
            let fx = gx - f32(ix0);
            let fy = gy - f32(iy0);

            // PQS (cubic B-spline) weights
            var wx: array<f32, 4>;
            var wy: array<f32, 4>;
            let fx2 = fx * fx; let fx3 = fx2 * fx;
            wx[0] = (1.0 - 3.0*fx + 3.0*fx2 - fx3) / 6.0;
            wx[1] = (4.0 - 6.0*fx2 + 3.0*fx3) / 6.0;
            wx[2] = (1.0 + 3.0*fx + 3.0*fx2 - 3.0*fx3) / 6.0;
            wx[3] = fx3 / 6.0;
            let fy2 = fy * fy; let fy3 = fy2 * fy;
            wy[0] = (1.0 - 3.0*fy + 3.0*fy2 - fy3) / 6.0;
            wy[1] = (4.0 - 6.0*fy2 + 3.0*fy3) / 6.0;
            wy[2] = (1.0 + 3.0*fy + 3.0*fy2 - 3.0*fy3) / 6.0;
            wy[3] = fy3 / 6.0;

            // Interpolate field values
            var higgsVal: f32 = 0.0;
            var axionVal: f32 = 0.0;
            for (var dy = 0; dy < 4; dy++) {
                let ny = clamp(iy0 + dy - 1, 0, i32(FGRID) - 1);
                let wwy = wy[dy];
                for (var dx = 0; dx < 4; dx++) {
                    let nx = clamp(ix0 + dx - 1, 0, i32(FGRID) - 1);
                    let w = wx[dx] * wwy;
                    let ci = u32(ny) * FGRID + u32(nx);
                    if (higgsOn) { higgsVal += higgsField[ci] * w; }
                    if (axionOn) { axionVal += axionField[ci] * w; }
                }
            }

            // Higgs PFI: -baseMass * (|phi| - 1)
            if (higgsOn) {
                higgsPfiE -= p.baseMass * (abs(higgsVal) - 1.0);
            }
            // Axion PFI: -g*q²*a (EM channel) - g*m*sign*a (PQ channel)
            if (axionOn) {
                let isAnti = (p.flags & FLAG_ANTIMATTER) != 0u;
                if (coulOn) {
                    axionPfiE -= AXION_COUPLING * p.charge * p.charge * axionVal;
                }
                if (yukOn) {
                    let sign = select(1.0, -1.0, isAnti);
                    axionPfiE -= AXION_COUPLING * p.mass * sign * axionVal;
                }
            }
        }
    }

    stats[16] = higgsPfiE;
    stats[17] = axionPfiE;

    // ─── Pass 6: Selected particle data (offset 32) ───
    let selIdx = params.selectedIdx;
    if (selIdx >= 0 && u32(selIdx) < n) {
        let si = u32(selIdx);
        let p = particles[si];
        let d = derived[si];
        let af = forces[si];
        stats[32] = p.posX;    stats[33] = p.posY;
        stats[34] = p.velWX;   stats[35] = p.velWY;
        stats[36] = p.mass;    stats[37] = p.charge;
        stats[38] = p.angW;    stats[39] = p.baseMass;
        stats[40] = bitcast<f32>(p.flags);
        stats[41] = sqrt(d.radiusSq);
        stats[42] = d.velX;    stats[43] = d.velY;
        stats[44] = d.angVel;
        stats[45] = d.magMoment;
        stats[46] = d.angMomentum;
        stats[47] = select(0.0, 1.0, (p.flags & FLAG_ANTIMATTER) != 0u);
        stats[48] = af.f0.x; stats[49] = af.f0.y;
        stats[50] = af.f0.z; stats[51] = af.f0.w;
        stats[52] = af.f1.x; stats[53] = af.f1.y;
        stats[54] = af.f1.z; stats[55] = af.f1.w;
        stats[56] = af.f2.x; stats[57] = af.f2.y;
        stats[58] = af.f2.z; stats[59] = af.f2.w;
        stats[60] = af.f3.x; stats[61] = af.f3.y;
        stats[62] = af.f3.z; stats[63] = af.f3.w;
        stats[64] = af.f4.x; stats[65] = af.f4.y;
        stats[66] = af.f4.z; stats[67] = af.f4.w;
        stats[68] = af.f5.x; stats[69] = af.f5.y;
    } else {
        stats[36] = -1.0; // mass = -1 signals no selected particle
    }
}
