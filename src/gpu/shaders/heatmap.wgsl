// ─── Potential Field Heatmap ───
// 64x64 grid potential from particles. Gravity, Coulomb, Yukawa contributions.
// Signal-delay-aware when relativity enabled.
// Direct O(N*GRID^2) pairwise — tree acceleration deferred to future optimization.

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState_HM {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct HeatmapUniforms {
    // Camera/viewport for world-space grid
    viewLeft: f32,
    viewTop: f32,
    cellW: f32,
    cellH: f32,
    // Physics params
    softeningSq: f32,
    yukawaCoupling: f32,
    yukawaMu: f32,
    simTime: f32,
    // Domain
    domainW: f32,
    domainH: f32,
    // Toggle bits
    doGravity: u32,
    doCoulomb: u32,
    doYukawa: u32,
    useDelay: u32,
    periodic: u32,
    topologyMode: u32,
    particleCount: u32,
    deadCount: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

// Group 0: particleState (ro)
@group(0) @binding(0) var<storage, read> particles: array<ParticleState_HM>;

@group(1) @binding(0) var<storage, read_write> gravPotential: array<f32>;
@group(1) @binding(1) var<storage, read_write> elecPotential: array<f32>;
@group(1) @binding(2) var<storage, read_write> yukawaPotential: array<f32>;
@group(1) @binding(3) var<uniform> hu: HeatmapUniforms;

const HGRID: u32 = 64u;
const HGRID_SQ: u32 = 4096u;

// Yukawa cutoff: exp(-mu*r) < 0.002 when mu*r > 6
fn yukawaCutoffSq(mu: f32) -> f32 {
    let cutoff = 6.0 / mu;
    return cutoff * cutoff;
}

@compute @workgroup_size(8, 8)
fn computeHeatmap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if (gx >= HGRID || gy >= HGRID) { return; }

    let wx = hu.viewLeft + (f32(gx) + 0.5) * hu.cellW;
    let wy = hu.viewTop + (f32(gy) + 0.5) * hu.cellH;

    var gPhi: f32 = 0.0;
    var ePhi: f32 = 0.0;
    var yPhi: f32 = 0.0;

    let doG = hu.doGravity != 0u;
    let doC = hu.doCoulomb != 0u;
    let doY = hu.doYukawa != 0u;
    let yCutSq = select(1e30, yukawaCutoffSq(hu.yukawaMu), doY);

    // Direct pairwise (signal delay not supported on GPU heatmap — uses current positions)
    for (var i = 0u; i < hu.particleCount; i++) {
        let p = particles[i];
        let flag = p.flags;
        if ((flag & 1u) == 0u) { continue; }

        let dx = p.posX - wx;
        let dy = p.posY - wy;
        let rSq = dx * dx + dy * dy + hu.softeningSq;
        let invR = 1.0 / sqrt(rSq);

        if (doG) { gPhi -= p.mass * invR; }
        if (doC) { ePhi += p.charge * invR; }
        if (doY && rSq < yCutSq) {
            let r = 1.0 / invR;
            yPhi -= hu.yukawaCoupling * p.mass * exp(-hu.yukawaMu * r) * invR;
        }
    }

    let idx = gy * HGRID + gx;
    gravPotential[idx] = gPhi;
    elecPotential[idx] = ePhi;
    yukawaPotential[idx] = yPhi;
}

// ─── 3x3 Separable Box Blur ───
@group(0) @binding(0) var<storage, read_write> arr: array<f32>;
@group(0) @binding(1) var<storage, read_write> blurTemp: array<f32>;

@compute @workgroup_size(8, 8)
fn blurHorizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= HGRID || y >= HGRID) { return; }

    let row = y * HGRID;
    let l = select(arr[row], arr[row + x - 1u], x > 0u);
    let c = arr[row + x];
    let r = select(arr[row + HGRID - 1u], arr[row + x + 1u], x < HGRID - 1u);
    blurTemp[row + x] = (l + c + r) * (1.0 / 3.0);
}

@compute @workgroup_size(8, 8)
fn blurVertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= HGRID || y >= HGRID) { return; }

    let t = select(blurTemp[x], blurTemp[(y - 1u) * HGRID + x], y > 0u);
    let c = blurTemp[y * HGRID + x];
    let b = select(blurTemp[(HGRID - 1u) * HGRID + x], blurTemp[(y + 1u) * HGRID + x], y < HGRID - 1u);
    arr[y * HGRID + x] = (t + c + b) * (1.0 / 3.0);
}
