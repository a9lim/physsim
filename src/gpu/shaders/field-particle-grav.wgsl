// ─── Particle-Field Gravity ───
// Gravitational force from scalar field energy density onto particles.
// F_i = m_i * Σ_cells ρ(x_j) * dA * (x_j - x_i) / |x_j - x_i|³
// Direct O(N × GRID²) summation (matches CPU applyGravForces).
// Dispatched once per active field (Higgs, Axion) when fieldGravEnabled.

struct ParticleState_FG {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct AllForces_FG {
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

struct FGUniforms {
    domainW: f32,
    domainH: f32,
    softeningSq: f32,
    aliveCount: u32,
    boundaryMode: u32,
    topologyMode: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> u: FGUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState_FG>;
@group(0) @binding(2) var<storage, read_write> allForces: array<AllForces_FG>;

@group(1) @binding(0) var<storage, read> energyDensity: array<f32>;

// Workgroup shared: load entire 64×64 energy density grid (16KB @ f32)
var<workgroup> sharedRho: array<f32, 4096>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u) {
    let idx = gid.x;
    let localIdx = lid.x;

    // Collaborative grid load: 64 threads load 4096/64 = 64 cells each
    for (var chunk = 0u; chunk < 64u; chunk++) {
        let cellIdx = localIdx * 64u + chunk;
        if (cellIdx < GRID * GRID) {
            sharedRho[cellIdx] = energyDensity[cellIdx];
        }
    }
    workgroupBarrier();

    if (idx >= u.aliveCount) { return; }
    let flag = particles[idx].flags;
    if ((flag & 1u) == 0u) { return; } // FLAG_ALIVE

    let m = particles[idx].mass;
    if (m < EPSILON) { return; }

    let px = particles[idx].posX;
    let py = particles[idx].posY;
    let softeningSq = u.softeningSq;
    let cellW = u.domainW / f32(GRID);
    let cellH = u.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }
    let cellArea = cellW * cellH;
    let periodic = u.boundaryMode == BOUND_LOOP;
    let halfDomW = u.domainW * 0.5;
    let halfDomH = u.domainH * 0.5;
    let topo = u.topologyMode;

    var fx: f32 = 0.0;
    var fy: f32 = 0.0;

    for (var iy = 0u; iy < GRID; iy++) {
        let cy = (f32(iy) + 0.5) * cellH;
        for (var ix = 0u; ix < GRID; ix++) {
            let rhoVal = sharedRho[iy * GRID + ix];
            if (rhoVal < EPSILON) { continue; }
            let cx = (f32(ix) + 0.5) * cellW;

            var dx = cx - px;
            var dy = cy - py;
            if (periodic) {
                // Torus minimum image (fast path)
                if (topo == TOPO_TORUS) {
                    if (dx > halfDomW) { dx -= u.domainW; } else if (dx < -halfDomW) { dx += u.domainW; }
                    if (dy > halfDomH) { dy -= u.domainH; } else if (dy < -halfDomH) { dy += u.domainH; }
                } else {
                    // Full topology minImage for Klein/RP² (use torus fast path as approximation
                    // for field gravity — exact minImage would need full 3/4-candidate check but
                    // the dominant contribution is always from nearby cells)
                    if (dx > halfDomW) { dx -= u.domainW; } else if (dx < -halfDomW) { dx += u.domainW; }
                    if (dy > halfDomH) { dy -= u.domainH; } else if (dy < -halfDomH) { dy += u.domainH; }
                }
            }

            let rSq = dx * dx + dy * dy + softeningSq;
            let invR = 1.0 / sqrt(rSq);
            let fMag = rhoVal * cellArea * invR * invR * invR;
            fx += dx * fMag;
            fy += dy * fMag;
        }
    }

    // Force = m * gravitational field
    let gfx = m * fx;
    let gfy = m * fy;

    // Accumulate into gravity force slot (f0.xy) and totalForce
    var af = allForces[idx];
    af.f0.x += gfx;
    af.f0.y += gfy;
    af.totalForce.x += gfx;
    af.totalForce.y += gfy;
    allForces[idx] = af;
}
