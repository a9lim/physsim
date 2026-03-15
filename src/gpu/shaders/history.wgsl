// ─── Signal Delay History ───
// Ring buffer recording + Newton-Raphson light-cone retrieval.
// f32 precision with relative-time encoding for GPU.
//
// Standalone shader — defines own structs (NOT prepended with common.wgsl).

// Constants provided by generated wgslConstants block.

// ── Packed struct definitions ──

struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

// Must match SimUniforms byte layout in common.wgsl / writeUniforms() exactly.
// Only a few fields used; preceding fields kept as padding for alignment.
struct SimUniforms {
    _dt: f32,               // [0] dt
    simTime: f32,           // [1] simTime
    domainW: f32,           // [2] domainW
    domainH: f32,           // [3] domainH
    _pad0: f32,             // [4] speedScale
    _pad1: f32,             // [5] softening
    _pad2: f32,             // [6] softeningSq
    _pad3: u32,             // [7] toggles0
    _pad4: u32,             // [8] toggles1
    _pad5: f32,             // [9] yukawaCoupling
    _pad6: f32,             // [10] yukawaMu
    _pad7: f32,             // [11] higgsMass
    _pad8: f32,             // [12] axionMass
    boundaryMode: u32,      // [13] boundaryMode
    topologyMode: u32,      // [14] topologyMode
};

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState>;

// History ring buffers
@group(1) @binding(0) var<storage, read_write> histPosX: array<f32>;
@group(1) @binding(1) var<storage, read_write> histPosY: array<f32>;
@group(1) @binding(2) var<storage, read_write> histVelWX: array<f32>;
@group(1) @binding(3) var<storage, read_write> histVelWY: array<f32>;
@group(1) @binding(4) var<storage, read_write> histAngW: array<f32>;
@group(1) @binding(5) var<storage, read_write> histTime: array<f32>;
// histMeta: [writeIdx, count] per particle (u32 pairs)
@group(1) @binding(6) var<storage, read_write> histMeta: array<u32>;

@compute @workgroup_size(64)
fn recordHistory(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= arrayLength(&particles)) { return; }
    if ((particles[i].flags & FLAG_ALIVE) == 0u) { return; }

    let metaBase = i * 2u;
    var writeIdx = histMeta[metaBase];
    var count = histMeta[metaBase + 1u];

    let slot = i * HISTORY_LEN + (writeIdx & HISTORY_MASK);

    histPosX[slot] = particles[i].posX;
    histPosY[slot] = particles[i].posY;

    // Store coordinate velocity (vel = w / sqrt(1 + w²))
    let wx = particles[i].velWX;
    let wy = particles[i].velWY;
    let gamma = sqrt(1.0 + wx * wx + wy * wy);
    let invG = 1.0 / gamma;
    histVelWX[slot] = wx * invG;
    histVelWY[slot] = wy * invG;

    histAngW[slot] = particles[i].angW;

    // Relative time encoding: store (simTime - 0) = simTime for now,
    // but retrieval uses (simTime_at_query - histTime) for f32 precision
    histTime[slot] = u.simTime;

    writeIdx = (writeIdx + 1u) & HISTORY_MASK;
    count = min(count + 1u, HISTORY_LEN);
    histMeta[metaBase] = writeIdx;
    histMeta[metaBase + 1u] = count;
}

// Return struct for signal delay lookup
struct DelayedState {
    x: f32, y: f32,
    vx: f32, vy: f32,
    angw: f32,
    valid: bool,
};

// Full topology-aware minimum image displacement (Torus/Klein/RP²)
fn minImageDisp(ox: f32, oy: f32, sx: f32, sy: f32,
                domW: f32, domH: f32, topo: u32) -> vec2f {
    let halfW = domW * 0.5;
    let halfH = domH * 0.5;

    // Candidate 0: direct wrap (torus)
    var dx0 = sx - ox;
    if (dx0 > halfW) { dx0 -= domW; } else if (dx0 < -halfW) { dx0 += domW; }
    var dy0 = sy - oy;
    if (dy0 > halfH) { dy0 -= domH; } else if (dy0 < -halfH) { dy0 += domH; }

    if (topo == TOPO_TORUS) { return vec2f(dx0, dy0); }

    var bestSq = dx0 * dx0 + dy0 * dy0;
    var bestDx = dx0;
    var bestDy = dy0;

    if (topo == TOPO_KLEIN) {
        // Klein: y-wrap is glide reflection (x,y) ~ (W-x, y+H)
        let gx = domW - sx;
        var dx1 = gx - ox;
        if (dx1 > halfW) { dx1 -= domW; } else if (dx1 < -halfW) { dx1 += domW; }
        var dy1 = (sy + domH) - oy;
        if (dy1 > domH) { dy1 -= 2.0 * domH; } else if (dy1 < -domH) { dy1 += 2.0 * domH; }
        let dSq1 = dx1 * dx1 + dy1 * dy1;
        if (dSq1 < bestSq) { bestDx = dx1; bestDy = dy1; bestSq = dSq1; }
        var dy1b = (sy - domH) - oy;
        if (dy1b > domH) { dy1b -= 2.0 * domH; } else if (dy1b < -domH) { dy1b += 2.0 * domH; }
        let dSq1b = dx1 * dx1 + dy1b * dy1b;
        if (dSq1b < bestSq) { bestDx = dx1; bestDy = dy1b; }
    } else {
        // RP²: both axes carry glide reflections
        let gx = domW - sx;
        var dxG = gx - ox;
        if (dxG > halfW) { dxG -= domW; } else if (dxG < -halfW) { dxG += domW; }
        var dyG = (sy + domH) - oy;
        if (dyG > domH) { dyG -= 2.0 * domH; } else if (dyG < -domH) { dyG += 2.0 * domH; }
        let dSqG = dxG * dxG + dyG * dyG;
        if (dSqG < bestSq) { bestDx = dxG; bestDy = dyG; bestSq = dSqG; }

        let gy = domH - sy;
        var dxH = (sx + domW) - ox;
        if (dxH > domW) { dxH -= 2.0 * domW; } else if (dxH < -domW) { dxH += 2.0 * domW; }
        var dyH = gy - oy;
        if (dyH > halfH) { dyH -= domH; } else if (dyH < -halfH) { dyH += domH; }
        let dSqH = dxH * dxH + dyH * dyH;
        if (dSqH < bestSq) { bestDx = dxH; bestDy = dyH; bestSq = dSqH; }

        var dxC = (domW - sx + domW) - ox;
        if (dxC > domW) { dxC -= 2.0 * domW; } else if (dxC < -domW) { dxC += 2.0 * domW; }
        var dyC = (domH - sy + domH) - oy;
        if (dyC > domH) { dyC -= 2.0 * domH; } else if (dyC < -domH) { dyC += 2.0 * domH; }
        let dSqC = dxC * dxC + dyC * dyC;
        if (dSqC < bestSq) { bestDx = dxC; bestDy = dyC; }
    }

    return vec2f(bestDx, bestDy);
}

fn getDelayedStateGPU(
    srcIdx: u32,
    obsX: f32, obsY: f32,
    simTime: f32,
    periodic: bool,
    domW: f32, domH: f32,
    topoMode: u32,
    isDead: bool,
) -> DelayedState {
    var result: DelayedState;
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
    if (periodic) {
        let d = minImageDisp(obsX, obsY, nxPos, nyPos, domW, domH, topoMode);
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
            if (periodic) {
                let d = minImageDisp(xLo, yLo, histPosX[hiIdx], histPosY[hiIdx], domW, domH, topoMode);
                vxEff = d.x / segDt; vyEff = d.y / segDt;
            } else {
                vxEff = (histPosX[hiIdx] - xLo) / segDt;
                vyEff = (histPosY[hiIdx] - yLo) / segDt;
            }

            let s = t - tLo;
            let sx_interp = xLo + vxEff * s;
            let sy_interp = yLo + vyEff * s;

            var dx: f32; var dy: f32;
            if (periodic) {
                let d = minImageDisp(obsX, obsY, sx_interp, sy_interp, domW, domH, topoMode);
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
            for (var w = 0; w < 64; w++) {
                if (segK >= i32(count) - 2) { break; }
                let ni = base + ((start + u32(segK + 1)) & HISTORY_MASK);
                if (histTime[ni] > t) { break; }
                segK++;
            }
            for (var w = 0; w < 64; w++) {
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

                let loIdx = base + ((start + u32(k)) & HISTORY_MASK);
                let hiIdx = base + (((start + u32(k)) + 1u) & HISTORY_MASK);
                let tLo = histTime[loIdx];
                let segDt = histTime[hiIdx] - tLo;
                if (segDt < NR_TOLERANCE) { continue; }

                let xLo = histPosX[loIdx]; let yLo = histPosY[loIdx];
                let xHi = histPosX[hiIdx]; let yHi = histPosY[hiIdx];

                var dx: f32; var dy: f32;
                var vx: f32; var vy: f32;
                if (periodic) {
                    let d0 = minImageDisp(obsX, obsY, xLo, yLo, domW, domH, topoMode);
                    dx = d0.x; dy = d0.y;
                    let d1 = minImageDisp(xLo, yLo, xHi, yHi, domW, domH, topoMode);
                    vx = d1.x / segDt; vy = d1.y / segDt;
                } else {
                    dx = xLo - obsX; dy = yLo - obsY;
                    vx = (xHi - xLo) / segDt; vy = (yHi - yLo) / segDt;
                }

                let rSq = dx * dx + dy * dy;
                let vSq = vx * vx + vy * vy;
                let dDotV = dx * vx + dy * vy;
                let T = simTime - tLo;

                // (v² - 1)s² + 2(d·v + T)s + (r² - T²) = 0
                let a = vSq - 1.0;
                let h = dDotV + T;
                let c = rSq - T * T;
                let disc = h * h - a * c;
                if (disc < 0.0) { continue; }

                let sqrtDisc = sqrt(max(disc, 0.0));
                var s_sol: f32;
                if (abs(a) < NR_TOLERANCE) {
                    if (abs(h) < NR_TOLERANCE) { continue; }
                    s_sol = -c / (2.0 * h);
                } else {
                    let s1 = (-h + sqrtDisc) / a;
                    let s2 = (-h - sqrtDisc) / a;
                    let ok1 = s1 >= -EPSILON && s1 <= segDt + EPSILON;
                    let ok2 = s2 >= -EPSILON && s2 <= segDt + EPSILON;
                    if (ok1 && ok2) { s_sol = max(s1, s2); }
                    else if (ok1) { s_sol = s1; }
                    else if (ok2) { s_sol = s2; }
                    else { continue; }
                }

                s_sol = clamp(s_sol, 0.0, segDt);
                let frac = s_sol / segDt;

                result.x = xLo + frac * (xHi - xLo);
                result.y = yLo + frac * (yHi - yLo);
                result.vx = histVelWX[loIdx] + frac * (histVelWX[hiIdx] - histVelWX[loIdx]);
                result.vy = histVelWY[loIdx] + frac * (histVelWY[hiIdx] - histVelWY[loIdx]);
                result.angw = histAngW[loIdx] + frac * (histAngW[hiIdx] - histAngW[loIdx]);
                result.valid = true;
                return result;
            }
        }
    }

    // Dead particles: don't extrapolate past buffer
    if (isDead) { return result; }

    // ─── Phase 3: Extrapolation from oldest sample ───
    {
        let xStart = histPosX[base + start];
        let yStart = histPosY[base + start];
        var dx: f32; var dy: f32;
        if (periodic) {
            let d = minImageDisp(obsX, obsY, xStart, yStart, domW, domH, topoMode);
            dx = d.x; dy = d.y;
        } else {
            dx = xStart - obsX; dy = yStart - obsY;
        }

        let vx = histVelWX[base + start];
        let vy = histVelWY[base + start];
        let rSq = dx * dx + dy * dy;
        let vSq = vx * vx + vy * vy;
        let dDotV = dx * vx + dy * vy;
        let T = timeSpan;

        let a = vSq - 1.0;
        let h = dDotV + T;
        let c = rSq - T * T;
        let disc = h * h - a * c;
        if (disc < 0.0) { return result; }

        let sqrtDisc = sqrt(disc);
        var s_sol: f32;
        if (abs(a) < NR_TOLERANCE) {
            if (abs(h) < NR_TOLERANCE) { return result; }
            s_sol = -c / (2.0 * h);
        } else {
            let s1 = (-h + sqrtDisc) / a;
            let s2 = (-h - sqrtDisc) / a;
            let ok1 = s1 <= EPSILON;
            let ok2 = s2 <= EPSILON;
            if (ok1 && ok2) { s_sol = max(s1, s2); }
            else if (ok1) { s_sol = s1; }
            else if (ok2) { s_sol = s2; }
            else { return result; }
        }
        if (s_sol > 0.0) { s_sol = 0.0; }

        // Reject extrapolation past particle creation
        // (creationTime check done by caller using particle metadata)

        result.x = xStart + vx * s_sol;
        result.y = yStart + vy * s_sol;
        result.vx = vx;
        result.vy = vy;
        result.angw = histAngW[base + start];
        result.valid = true;
        return result;
    }
}
