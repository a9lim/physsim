// compute-stats.wgsl — Single-thread aggregate stats reduction + selected particle readback.
//
// Dispatched as (1,1,1). Two-pass: first computes KE/momentum/COM/mass,
// second computes angular momentum about COM.
// Also copies selected particle's state + forces to the output buffer.
//
// Constants from generated wgslConstants block: INERTIA_K, FLAG_ALIVE, FLAG_ANTIMATTER

struct StatsUniforms {
    aliveCount: u32,
    selectedIdx: i32,    // GPU buffer index of selected particle, -1 if none
    relativityEnabled: u32,
    _pad: u32,
};

struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct ParticleDerived {
    magMoment: f32,
    angMomentum: f32,
    invMass: f32,
    radiusSq: f32,
    velX: f32, velY: f32,
    angVel: f32,
    _pad: f32,
};

struct AllForces {
    f0: vec4<f32>,          // gravity.xy, coulomb.xy
    f1: vec4<f32>,          // magnetic.xy, gravitomag.xy
    f2: vec4<f32>,          // f1pn.xy, spinCurv.xy
    f3: vec4<f32>,          // radiation.xy, yukawa.xy
    f4: vec4<f32>,          // external.xy, higgs.xy
    f5: vec4<f32>,          // axion.xy, pad, pad
    torques: vec4<f32>,     // spinOrbit, frameDrag, tidal, contact
    bFields: vec4<f32>,     // Bz, Bgz, extBz, pad
    bFieldGrads: vec4<f32>, // dBzdx, dBzdy, dBgzdx, dBgzdy
    totalForce: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: StatsUniforms;
@group(0) @binding(1) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read> derived: array<ParticleDerived>;
@group(0) @binding(3) var<storage, read> forces: array<AllForces>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;

// Output layout (64 f32 = 256 bytes):
// Aggregates [0-15]:
//   [0] linearKE, [1] spinKE, [2] px, [3] py,
//   [4] orbAngMom, [5] spinAngMom, [6] comX, [7] comY,
//   [8] totalMass, [9] aliveCount
// Selected particle [16-55]:
//   [16] posX, [17] posY, [18] velWX, [19] velWY,
//   [20] mass, [21] charge, [22] angW, [23] baseMass,
//   [24] flags (bitcast), [25] radius, [26] velX, [27] velY,
//   [28] angVel, [29] magMoment, [30] angMomentum, [31] antimatter (0/1),
//   Forces (11 × 2 = 22 f32):
//   [32-33] gravity, [34-35] coulomb, [36-37] magnetic, [38-39] gravitomag,
//   [40-41] 1pn, [42-43] spinCurv, [44-45] radiation, [46-47] yukawa,
//   [48-49] external, [50-51] higgs, [52-53] axion

@compute @workgroup_size(1)
fn main() {
    let n = params.aliveCount;
    let relOn = params.relativityEnabled != 0u;

    // ─── Pass 1: KE, momentum, COM ───
    var linearKE: f32 = 0.0;
    var spinKE: f32 = 0.0;
    var px: f32 = 0.0;
    var py: f32 = 0.0;
    var totalMass: f32 = 0.0;
    var comX: f32 = 0.0;
    var comY: f32 = 0.0;

    for (var i = 0u; i < n; i++) {
        let p = particles[i];
        if ((p.flags & FLAG_ALIVE) == 0u) { continue; }
        let d = derived[i];
        let m = p.mass;

        if (relOn) {
            let wSq = p.velWX * p.velWX + p.velWY * p.velWY;
            let gamma = sqrt(1.0 + wSq);
            linearKE += wSq / (gamma + 1.0) * m;
            let srSq = p.angW * p.angW * d.radiusSq;
            let gammaRot = sqrt(1.0 + srSq);
            spinKE += INERTIA_K * m * srSq / (gammaRot + 1.0);
        } else {
            let vSq = d.velX * d.velX + d.velY * d.velY;
            linearKE += 0.5 * m * vSq;
            spinKE += 0.5 * INERTIA_K * m * d.radiusSq * d.angVel * d.angVel;
        }

        px += m * p.velWX;
        py += m * p.velWY;
        totalMass += m;
        comX += m * p.posX;
        comY += m * p.posY;
    }

    if (totalMass > 0.0) {
        comX /= totalMass;
        comY /= totalMass;
    }

    // ─── Pass 2: Angular momentum about COM ───
    var orbAngMom: f32 = 0.0;
    var spinAngMom: f32 = 0.0;

    for (var i = 0u; i < n; i++) {
        let p = particles[i];
        if ((p.flags & FLAG_ALIVE) == 0u) { continue; }
        let d = derived[i];
        let m = p.mass;
        let dx = p.posX - comX;
        let dy = p.posY - comY;
        orbAngMom += dx * (m * p.velWY) - dy * (m * p.velWX);
        spinAngMom += INERTIA_K * m * d.radiusSq * p.angW;
    }

    // ─── Write aggregates ───
    stats[0] = linearKE;
    stats[1] = spinKE;
    stats[2] = px;
    stats[3] = py;
    stats[4] = orbAngMom;
    stats[5] = spinAngMom;
    stats[6] = comX;
    stats[7] = comY;
    stats[8] = totalMass;
    stats[9] = f32(n);

    // ─── Selected particle data ───
    let selIdx = params.selectedIdx;
    if (selIdx >= 0 && u32(selIdx) < n) {
        let si = u32(selIdx);
        let p = particles[si];
        let d = derived[si];
        let af = forces[si];

        stats[16] = p.posX;    stats[17] = p.posY;
        stats[18] = p.velWX;   stats[19] = p.velWY;
        stats[20] = p.mass;    stats[21] = p.charge;
        stats[22] = p.angW;    stats[23] = p.baseMass;
        stats[24] = bitcast<f32>(p.flags);
        stats[25] = sqrt(d.radiusSq); // radius
        stats[26] = d.velX;    stats[27] = d.velY;
        stats[28] = d.angVel;
        stats[29] = d.magMoment;
        stats[30] = d.angMomentum;
        stats[31] = select(0.0, 1.0, (p.flags & FLAG_ANTIMATTER) != 0u);

        // Forces: 11 force vec2s packed into f0-f5
        stats[32] = af.f0.x; stats[33] = af.f0.y; // gravity
        stats[34] = af.f0.z; stats[35] = af.f0.w; // coulomb
        stats[36] = af.f1.x; stats[37] = af.f1.y; // magnetic
        stats[38] = af.f1.z; stats[39] = af.f1.w; // gravitomag
        stats[40] = af.f2.x; stats[41] = af.f2.y; // 1pn
        stats[42] = af.f2.z; stats[43] = af.f2.w; // spinCurv
        stats[44] = af.f3.x; stats[45] = af.f3.y; // radiation
        stats[46] = af.f3.z; stats[47] = af.f3.w; // yukawa
        stats[48] = af.f4.x; stats[49] = af.f4.y; // external
        stats[50] = af.f4.z; stats[51] = af.f4.w; // higgs
        stats[52] = af.f5.x; stats[53] = af.f5.y; // axion
    } else {
        // Mark no selection
        stats[20] = -1.0; // mass = -1 signals no selected particle
    }
}
