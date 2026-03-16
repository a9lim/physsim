// ─── Potential Field Heatmap ───
// 64x64 grid potential from particles. Gravity, Coulomb, Yukawa contributions.
// Signal delay supported: when relativity enabled, uses retarded positions via
// shared getDelayedStateGPU() from signal-delay-common.wgsl.
// Dead/retired particles contribute via signal delay history (deathMass for gravity/Yukawa).
// Direct O(N*GRID^2) pairwise — tree acceleration deferred to future optimization.

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState_HM {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct ParticleAux_HM {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
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
    _padDead: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

// Group 0: particleState + particleAux (read_write for encoder compat)
@group(0) @binding(0) var<storage, read_write> particles: array<ParticleState_HM>;
@group(0) @binding(1) var<storage, read_write> particleAux: array<ParticleAux_HM>;

@group(1) @binding(0) var<storage, read_write> gravPotential: array<f32>;
@group(1) @binding(1) var<storage, read_write> elecPotential: array<f32>;
@group(1) @binding(2) var<storage, read_write> yukawaPotential: array<f32>;
@group(1) @binding(3) var<uniform> hu: HeatmapUniforms;

// Group 2: signal delay history (interleaved format, only used when useDelay != 0)
@group(2) @binding(0) var<storage, read_write> histData: array<f32>;
@group(2) @binding(1) var<storage, read_write> histMeta: array<u32>;

// Constants provided by generated wgslConstants block.

// ─── Topology-aware minimum image displacement ───
fn hmMinImage(ox: f32, oy: f32, sx: f32, sy: f32) -> vec2<f32> {
    let w = hu.domainW;
    let h = hu.domainH;
    let halfW = w * 0.5;
    let halfH = h * 0.5;
    let topo = hu.topologyMode;

    // Torus early return
    if (topo == TOPO_TORUS) {
        var dx = sx - ox;
        if (dx > halfW) { dx -= w; } else if (dx < -halfW) { dx += w; }
        var dy = sy - oy;
        if (dy > halfH) { dy -= h; } else if (dy < -halfH) { dy += h; }
        return vec2(dx, dy);
    }

    // Candidate 0: only torus-wrap axes with translational (not glide) periodicity.
    // Klein: x periodic (period W), y glide (period 2H) — only wrap x.
    // RP²: both glide — no wrapping.
    var dx0 = sx - ox;
    var dy0 = sy - oy;
    if (topo == TOPO_KLEIN) {
        if (dx0 > halfW) { dx0 -= w; } else if (dx0 < -halfW) { dx0 += w; }
    }
    var bestSq = dx0 * dx0 + dy0 * dy0;
    var bestDx = dx0;
    var bestDy = dy0;

    if (topo == TOPO_KLEIN) {
        let gx = w - sx;
        var dx1 = gx - ox;
        if (dx1 > halfW) { dx1 -= w; } else if (dx1 < -halfW) { dx1 += w; }
        var dy1 = (sy + h) - oy;
        if (dy1 > h) { dy1 -= 2.0 * h; } else if (dy1 < -h) { dy1 += 2.0 * h; }
        let dSq1 = dx1 * dx1 + dy1 * dy1;
        if (dSq1 < bestSq) { bestDx = dx1; bestDy = dy1; bestSq = dSq1; }
        var dy1b = (sy - h) - oy;
        if (dy1b > h) { dy1b -= 2.0 * h; } else if (dy1b < -h) { dy1b += 2.0 * h; }
        let dSq1b = dx1 * dx1 + dy1b * dy1b;
        if (dSq1b < bestSq) { bestDx = dx1; bestDy = dy1b; }
    } else {
        // RP²: both axes glide reflections (translational periods 2W, 2H)

        // Candidate 1: y-glide  (x,y) ~ (W-x, y+H) — x not wrapped
        let gx = w - sx;
        let dxG = gx - ox;
        var dyG = (sy + h) - oy;
        if (dyG > h) { dyG -= 2.0 * h; } else if (dyG < -h) { dyG += 2.0 * h; }
        let dSqG = dxG * dxG + dyG * dyG;
        if (dSqG < bestSq) { bestDx = dxG; bestDy = dyG; bestSq = dSqG; }

        // Candidate 2: x-glide  (x,y) ~ (x+W, H-y) — y not wrapped
        let gy = h - sy;
        var dxH = (sx + w) - ox;
        if (dxH > w) { dxH -= 2.0 * w; } else if (dxH < -w) { dxH += 2.0 * w; }
        let dyH = gy - oy;
        let dSqH = dxH * dxH + dyH * dyH;
        if (dSqH < bestSq) { bestDx = dxH; bestDy = dyH; bestSq = dSqH; }

        // Candidate 3: both glides  (x,y) ~ (2W-x, 2H-y)
        var dxC = (2.0 * w - sx) - ox;
        if (dxC > w) { dxC -= 2.0 * w; } else if (dxC < -w) { dxC += 2.0 * w; }
        var dyC = (2.0 * h - sy) - oy;
        if (dyC > h) { dyC -= 2.0 * h; } else if (dyC < -h) { dyC += 2.0 * h; }
        let dSqC = dxC * dxC + dyC * dyC;
        if (dSqC < bestSq) { bestDx = dxC; bestDy = dyC; }
    }

    return vec2(bestDx, bestDy);
}

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
    let useDelay = hu.useDelay != 0u;
    let yCutSq = select(1e30, yukawaCutoffSq(hu.yukawaMu), doY);

    // Alive particles
    for (var i = 0u; i < hu.particleCount; i++) {
        let p = particles[i];
        let flag = p.flags;
        if ((flag & FLAG_ALIVE) == 0u) { continue; }

        var srcX = p.posX;
        var srcY = p.posY;

        // Signal delay: solve for retarded position
        if (useDelay) {
            let ret = getDelayedStateGPU(i, wx, wy, hu.simTime,
                hu.periodic != 0u, hu.domainW, hu.domainH,
                hu.topologyMode, false);
            if (!ret.valid) { continue; } // outside past light cone — particle not yet visible
            srcX = ret.x;
            srcY = ret.y;
        }

        var dx: f32; var dy: f32;
        if (hu.periodic != 0u) {
            let d = hmMinImage(wx, wy, srcX, srcY);
            dx = d.x; dy = d.y;
        } else {
            dx = srcX - wx; dy = srcY - wy;
        }

        let rSq = dx * dx + dy * dy + hu.softeningSq;
        let invR = 1.0 / sqrt(rSq);

        if (doG) { gPhi -= p.mass * invR; }
        if (doC) { ePhi += p.charge * invR; }
        if (doY && rSq < yCutSq) {
            let r = 1.0 / invR;
            yPhi -= hu.yukawaCoupling * p.mass * exp(-hu.yukawaMu * r) * invR;
        }
    }

    // Dead/retired particles: signal delay fade-out
    if (useDelay) {
        for (var di = 0u; di < hu.particleCount; di++) {
            let dp = particles[di];
            if ((dp.flags & FLAG_RETIRED) == 0u) { continue; }
            if ((dp.flags & FLAG_ALIVE) != 0u) { continue; }

            let ret = getDelayedStateGPU(di, wx, wy, hu.simTime,
                hu.periodic != 0u, hu.domainW, hu.domainH,
                hu.topologyMode, true);
            if (!ret.valid) { continue; }

            let dAux = particleAux[di];
            var dx: f32; var dy: f32;
            if (hu.periodic != 0u) {
                let d = hmMinImage(wx, wy, ret.x, ret.y);
                dx = d.x; dy = d.y;
            } else {
                dx = ret.x - wx; dy = ret.y - wy;
            }
            let rSq = dx * dx + dy * dy + hu.softeningSq;
            let invR = 1.0 / sqrt(rSq);

            if (doG) { gPhi -= dAux.deathMass * invR; }
            if (doC) { ePhi += dp.charge * invR; }
            if (doY && rSq < yCutSq) {
                let r = 1.0 / invR;
                yPhi -= hu.yukawaCoupling * dAux.deathMass * exp(-hu.yukawaMu * r) * invR;
            }
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
