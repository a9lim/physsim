// ─── Potential Field Heatmap ───
// 64x64 grid potential from particles. Gravity, Coulomb, Yukawa contributions.
// Signal delay supported: when relativity enabled, uses retarded positions via
// Newton-Raphson light-cone solver (same algorithm as history.wgsl).
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

// Group 0: particleState (read_write for encoder compat)
@group(0) @binding(0) var<storage, read_write> particles: array<ParticleState_HM>;

@group(1) @binding(0) var<storage, read_write> gravPotential: array<f32>;
@group(1) @binding(1) var<storage, read_write> elecPotential: array<f32>;
@group(1) @binding(2) var<storage, read_write> yukawaPotential: array<f32>;
@group(1) @binding(3) var<uniform> hu: HeatmapUniforms;

// Group 2: signal delay history ring buffers (only used when useDelay != 0)
@group(2) @binding(0) var<storage, read_write> histPosX: array<f32>;
@group(2) @binding(1) var<storage, read_write> histPosY: array<f32>;
@group(2) @binding(2) var<storage, read_write> histTime: array<f32>;
@group(2) @binding(3) var<storage, read_write> histMeta: array<u32>;

const HGRID: u32 = 64u;
const HGRID_SQ: u32 = 4096u;

// Signal delay constants (must match history.wgsl)
const HISTORY_LEN: u32 = 256u;
const HISTORY_MASK: u32 = 255u;
const NR_MAX_ITER: u32 = 8u;
const NR_TOLERANCE: f32 = 1e-5;
const EPSILON: f32 = 1e-9;

// Topology constants
const TOPO_TORUS: u32 = 0u;
const TOPO_KLEIN: u32 = 1u;
const TOPO_RP2: u32 = 2u;

// ─── Topology-aware minimum image displacement ───
fn hmMinImage(ox: f32, oy: f32, sx: f32, sy: f32) -> vec2<f32> {
    let w = hu.domainW;
    let h = hu.domainH;
    let halfW = w * 0.5;
    let halfH = h * 0.5;
    let topo = hu.topologyMode;

    // Candidate 0: direct torus wrap
    var dx0 = sx - ox;
    if (dx0 > halfW) { dx0 -= w; } else if (dx0 < -halfW) { dx0 += w; }
    var dy0 = sy - oy;
    if (dy0 > halfH) { dy0 -= h; } else if (dy0 < -halfH) { dy0 += h; }

    if (topo == TOPO_TORUS) { return vec2(dx0, dy0); }

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
        // RP²
        let gx = w - sx;
        var dxG = gx - ox;
        if (dxG > halfW) { dxG -= w; } else if (dxG < -halfW) { dxG += w; }
        var dyG = (sy + h) - oy;
        if (dyG > h) { dyG -= 2.0 * h; } else if (dyG < -h) { dyG += 2.0 * h; }
        let dSqG = dxG * dxG + dyG * dyG;
        if (dSqG < bestSq) { bestDx = dxG; bestDy = dyG; bestSq = dSqG; }

        let gy = h - sy;
        var dxH = (sx + w) - ox;
        if (dxH > w) { dxH -= 2.0 * w; } else if (dxH < -w) { dxH += 2.0 * w; }
        var dyH = gy - oy;
        if (dyH > halfH) { dyH -= h; } else if (dyH < -halfH) { dyH += h; }
        let dSqH = dxH * dxH + dyH * dyH;
        if (dSqH < bestSq) { bestDx = dxH; bestDy = dyH; bestSq = dSqH; }

        var dxC = (w - sx + w) - ox;
        if (dxC > w) { dxC -= 2.0 * w; } else if (dxC < -w) { dxC += 2.0 * w; }
        var dyC = (h - sy + h) - oy;
        if (dyC > h) { dyC -= 2.0 * h; } else if (dyC < -h) { dyC += 2.0 * h; }
        let dSqC = dxC * dxC + dyC * dyC;
        if (dSqC < bestSq) { bestDx = dxC; bestDy = dyC; }
    }

    return vec2(bestDx, bestDy);
}

// ─── Signal delay: retarded position via Newton-Raphson light-cone solver ───
// Returns (posX, posY, valid) — simplified from history.wgsl (we only need position)
struct RetardedPos {
    x: f32, y: f32,
    valid: bool,
};

fn getRetardedPosition(
    srcIdx: u32,
    obsX: f32, obsY: f32,
    simTime: f32,
) -> RetardedPos {
    var result: RetardedPos;
    result.valid = false;

    let metaBase = srcIdx * 2u;
    let writeIdx = histMeta[metaBase];
    let count = histMeta[metaBase + 1u];
    if (count < 2u) { return result; }

    let start = (writeIdx - count + HISTORY_LEN) & HISTORY_MASK;
    let newest = (writeIdx - 1u + HISTORY_LEN) & HISTORY_MASK;
    let base = srcIdx * HISTORY_LEN;

    let tOldest = histTime[base + start];
    let tNewest = histTime[base + newest];
    let timeSpan = simTime - tOldest;
    if (timeSpan < NR_TOLERANCE) { return result; }

    // Current distance to newest sample
    let nxPos = histPosX[base + newest];
    let nyPos = histPosY[base + newest];
    var cdx: f32; var cdy: f32;
    if (hu.periodic != 0u) {
        let d = hmMinImage(obsX, obsY, nxPos, nyPos);
        cdx = d.x; cdy = d.y;
    } else {
        cdx = nxPos - obsX; cdy = nyPos - obsY;
    }
    let distSq = cdx * cdx + cdy * cdy;

    // ─── Phase 1: Newton-Raphson segment search ───
    if (distSq <= 4.0 * timeSpan * timeSpan) {
        var t = simTime - sqrt(distSq);
        t = clamp(t, tOldest, tNewest);

        let histSpan = tNewest - tOldest;
        var segK: i32;
        if (histSpan > NR_TOLERANCE) {
            segK = i32(floor((t - tOldest) / histSpan * f32(count - 1u)));
        } else {
            segK = 0;
        }
        segK = clamp(segK, 0, i32(count) - 2);

        // Walk to correct segment
        for (var w = 0; w < 256; w++) {
            if (segK >= i32(count) - 2) { break; }
            let nextIdx = base + ((start + u32(segK + 1)) & HISTORY_MASK);
            if (histTime[nextIdx] > t) { break; }
            segK++;
        }
        for (var w = 0; w < 256; w++) {
            if (segK <= 0) { break; }
            let curIdx = base + ((start + u32(segK)) & HISTORY_MASK);
            if (histTime[curIdx] <= t) { break; }
            segK--;
        }

        var prevSegK: i32 = -1;
        for (var iter = 0u; iter < NR_MAX_ITER; iter++) {
            if (segK == prevSegK) { break; }
            prevSegK = segK;

            let loIdx = base + ((start + u32(segK)) & HISTORY_MASK);
            let hiIdx = base + (((start + u32(segK)) + 1u) & HISTORY_MASK);
            let tLo = histTime[loIdx];
            let segDt = histTime[hiIdx] - tLo;
            if (segDt < NR_TOLERANCE) {
                if (segK < i32(count) - 2) { segK++; prevSegK = -1; continue; }
                break;
            }

            let xLo = histPosX[loIdx]; let yLo = histPosY[loIdx];
            var vxEff: f32; var vyEff: f32;
            if (hu.periodic != 0u) {
                let d = hmMinImage(xLo, yLo, histPosX[hiIdx], histPosY[hiIdx]);
                vxEff = d.x / segDt; vyEff = d.y / segDt;
            } else {
                vxEff = (histPosX[hiIdx] - xLo) / segDt;
                vyEff = (histPosY[hiIdx] - yLo) / segDt;
            }

            let s = t - tLo;
            let sx_interp = xLo + vxEff * s;
            let sy_interp = yLo + vyEff * s;

            var dx: f32; var dy: f32;
            if (hu.periodic != 0u) {
                let d = hmMinImage(obsX, obsY, sx_interp, sy_interp);
                dx = d.x; dy = d.y;
            } else {
                dx = sx_interp - obsX; dy = sy_interp - obsY;
            }

            let dSq = dx * dx + dy * dy;
            if (dSq < NR_TOLERANCE * NR_TOLERANCE) { break; }
            let dist = sqrt(dSq);

            let g = dist - (simTime - t);
            let gp = (dx * vxEff + dy * vyEff) / dist + 1.0;
            if (abs(gp) < NR_TOLERANCE) { break; }

            t -= g / gp;
            t = clamp(t, tOldest, tNewest);

            // Re-locate segment
            for (var w2 = 0; w2 < 64; w2++) {
                if (segK >= i32(count) - 2) { break; }
                let ni = base + ((start + u32(segK + 1)) & HISTORY_MASK);
                if (histTime[ni] > t) { break; }
                segK++;
            }
            for (var w2 = 0; w2 < 64; w2++) {
                if (segK <= 0) { break; }
                let ci = base + ((start + u32(segK)) & HISTORY_MASK);
                if (histTime[ci] <= t) { break; }
                segK--;
            }
        }

        // ─── Phase 2: Exact quadratic on converged segment (+/- 1 neighbor) ───
        let center = segK;
        for (var offset = 0; offset <= 1; offset++) {
            for (var dir = select(-1, 1, offset == 0); dir <= 1; dir += 2) {
                let k = center + offset * dir;
                if (k < 0 || k > i32(count) - 2) { continue; }

                let loIdx2 = base + ((start + u32(k)) & HISTORY_MASK);
                let hiIdx2 = base + (((start + u32(k)) + 1u) & HISTORY_MASK);
                let tLo2 = histTime[loIdx2];
                let segDt2 = histTime[hiIdx2] - tLo2;
                if (segDt2 < NR_TOLERANCE) { continue; }

                let xLo2 = histPosX[loIdx2]; let yLo2 = histPosY[loIdx2];
                let xHi2 = histPosX[hiIdx2]; let yHi2 = histPosY[hiIdx2];

                var dx2: f32; var dy2: f32;
                var vx2: f32; var vy2: f32;
                if (hu.periodic != 0u) {
                    let d0 = hmMinImage(obsX, obsY, xLo2, yLo2);
                    dx2 = d0.x; dy2 = d0.y;
                    let d1 = hmMinImage(xLo2, yLo2, xHi2, yHi2);
                    vx2 = d1.x / segDt2; vy2 = d1.y / segDt2;
                } else {
                    dx2 = xLo2 - obsX; dy2 = yLo2 - obsY;
                    vx2 = (xHi2 - xLo2) / segDt2; vy2 = (yHi2 - yLo2) / segDt2;
                }

                let rSq2 = dx2 * dx2 + dy2 * dy2;
                let vSq2 = vx2 * vx2 + vy2 * vy2;
                let dDotV = dx2 * vx2 + dy2 * vy2;
                let T = simTime - tLo2;

                // (v^2 - 1)s^2 + 2(d.v + T)s + (r^2 - T^2) = 0
                let a = vSq2 - 1.0;
                let h = dDotV + T;
                let c = rSq2 - T * T;
                let disc = h * h - a * c;
                if (disc < 0.0) { continue; }

                let sqrtDisc = sqrt(disc);
                var s_sol: f32;
                if (abs(a) < NR_TOLERANCE) {
                    if (abs(h) < NR_TOLERANCE) { continue; }
                    s_sol = -c / (2.0 * h);
                } else {
                    let s1 = (-h + sqrtDisc) / a;
                    let s2 = (-h - sqrtDisc) / a;
                    let ok1 = s1 >= -EPSILON && s1 <= segDt2 + EPSILON;
                    let ok2 = s2 >= -EPSILON && s2 <= segDt2 + EPSILON;
                    if (ok1 && ok2) { s_sol = max(s1, s2); }
                    else if (ok1) { s_sol = s1; }
                    else if (ok2) { s_sol = s2; }
                    else { continue; }
                }

                s_sol = clamp(s_sol, 0.0, segDt2);
                let frac = s_sol / segDt2;

                result.x = xLo2 + frac * (xHi2 - xLo2);
                result.y = yLo2 + frac * (yHi2 - yLo2);
                result.valid = true;
                return result;
            }
        }
    }

    // ─── Phase 3: Extrapolation from oldest sample ───
    {
        let xStart = histPosX[base + start];
        let yStart = histPosY[base + start];
        var dxE: f32; var dyE: f32;
        if (hu.periodic != 0u) {
            let d = hmMinImage(obsX, obsY, xStart, yStart);
            dxE = d.x; dyE = d.y;
        } else {
            dxE = xStart - obsX; dyE = yStart - obsY;
        }

        // Use average velocity from oldest two samples for extrapolation direction
        let next = (start + 1u) & HISTORY_MASK;
        let dt_ext = histTime[base + next] - tOldest;
        var vxE: f32 = 0.0; var vyE: f32 = 0.0;
        if (dt_ext > NR_TOLERANCE) {
            if (hu.periodic != 0u) {
                let dv = hmMinImage(xStart, yStart, histPosX[base + next], histPosY[base + next]);
                vxE = dv.x / dt_ext; vyE = dv.y / dt_ext;
            } else {
                vxE = (histPosX[base + next] - xStart) / dt_ext;
                vyE = (histPosY[base + next] - yStart) / dt_ext;
            }
        }

        let rSqE = dxE * dxE + dyE * dyE;
        let vSqE = vxE * vxE + vyE * vyE;
        let dDotVE = dxE * vxE + dyE * vyE;
        let T_E = timeSpan;

        let aE = vSqE - 1.0;
        let hE = dDotVE + T_E;
        let cE = rSqE - T_E * T_E;
        let discE = hE * hE - aE * cE;
        if (discE < 0.0) { return result; }

        let sqrtDiscE = sqrt(discE);
        var s_solE: f32;
        if (abs(aE) < NR_TOLERANCE) {
            if (abs(hE) < NR_TOLERANCE) { return result; }
            s_solE = -cE / (2.0 * hE);
        } else {
            let s1E = (-hE + sqrtDiscE) / aE;
            let s2E = (-hE - sqrtDiscE) / aE;
            let ok1E = s1E <= EPSILON;
            let ok2E = s2E <= EPSILON;
            if (ok1E && ok2E) { s_solE = max(s1E, s2E); }
            else if (ok1E) { s_solE = s1E; }
            else if (ok2E) { s_solE = s2E; }
            else { return result; }
        }
        if (s_solE > 0.0) { s_solE = 0.0; }

        result.x = xStart + vxE * s_solE;
        result.y = yStart + vyE * s_solE;
        result.valid = true;
        return result;
    }
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

    for (var i = 0u; i < hu.particleCount; i++) {
        let p = particles[i];
        let flag = p.flags;
        if ((flag & 1u) == 0u) { continue; }

        var srcX = p.posX;
        var srcY = p.posY;

        // Signal delay: solve for retarded position
        if (useDelay) {
            let ret = getRetardedPosition(i, wx, wy, hu.simTime);
            if (ret.valid) {
                srcX = ret.x;
                srcY = ret.y;
            }
            // If invalid (not enough history), fall back to current position
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
