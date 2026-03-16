// ─── Signal Delay Common ───
// Shared Newton-Raphson light-cone solver for interleaved history buffers.
// Prepended to consuming shaders. Callers declare histData/histMeta bindings.
//
// Requires constants: HIST_STRIDE, HIST_META_STRIDE, HISTORY_LEN, HISTORY_MASK,
// NR_TOLERANCE, NR_MAX_ITER, EPSILON, TOPO_TORUS, TOPO_KLEIN.

struct DelayedState {
    x: f32, y: f32,
    vx: f32, vy: f32,
    angw: f32,
    valid: bool,
};

// Full topology-aware minimum image displacement (Torus/Klein/RP²)
// Renamed to avoid collision with common.wgsl fullMinImage (which reads uniforms directly)
fn sdMinImageDisp(ox: f32, oy: f32, sx: f32, sy: f32,
                  domW: f32, domH: f32, topo: u32) -> vec2f {
    let halfW = domW * 0.5;
    let halfH = domH * 0.5;

    // Torus early return
    if (topo == TOPO_TORUS) {
        var dx = sx - ox;
        if (dx > halfW) { dx -= domW; } else if (dx < -halfW) { dx += domW; }
        var dy = sy - oy;
        if (dy > halfH) { dy -= domH; } else if (dy < -halfH) { dy += domH; }
        return vec2f(dx, dy);
    }

    // Candidate 0: only torus-wrap axes with translational (not glide) periodicity.
    // Klein: x periodic (period W), y glide (period 2H) — only wrap x.
    // RP²: both glide — no wrapping.
    var dx0 = sx - ox;
    var dy0 = sy - oy;
    if (topo == TOPO_KLEIN) {
        if (dx0 > halfW) { dx0 -= domW; } else if (dx0 < -halfW) { dx0 += domW; }
    }
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
        // RP²: both axes carry glide reflections (translational periods 2W, 2H)

        // Candidate 1: y-glide  (x,y) ~ (W-x, y+H) — x not wrapped
        let gx = domW - sx;
        let dxG = gx - ox;
        var dyG = (sy + domH) - oy;
        if (dyG > domH) { dyG -= 2.0 * domH; } else if (dyG < -domH) { dyG += 2.0 * domH; }
        let dSqG = dxG * dxG + dyG * dyG;
        if (dSqG < bestSq) { bestDx = dxG; bestDy = dyG; bestSq = dSqG; }

        // Candidate 2: x-glide  (x,y) ~ (x+W, H-y) — y not wrapped
        let gy = domH - sy;
        var dxH = (sx + domW) - ox;
        if (dxH > domW) { dxH -= 2.0 * domW; } else if (dxH < -domW) { dxH += 2.0 * domW; }
        let dyH = gy - oy;
        let dSqH = dxH * dxH + dyH * dyH;
        if (dSqH < bestSq) { bestDx = dxH; bestDy = dyH; bestSq = dSqH; }

        // Candidate 3: both glides  (x,y) ~ (2W-x, 2H-y)
        var dxC = (2.0 * domW - sx) - ox;
        if (dxC > domW) { dxC -= 2.0 * domW; } else if (dxC < -domW) { dxC += 2.0 * domW; }
        var dyC = (2.0 * domH - sy) - oy;
        if (dyC > domH) { dyC -= 2.0 * domH; } else if (dyC < -domH) { dyC += 2.0 * domH; }
        let dSqC = dxC * dxC + dyC * dyC;
        if (dSqC < bestSq) { bestDx = dxC; bestDy = dyC; }
    }

    return vec2f(bestDx, bestDy);
}

// Helper: compute base index into interleaved histData for a given particle and sample
fn histSampleBase(srcIdx: u32, sampleIdx: u32) -> u32 {
    return srcIdx * HISTORY_LEN * HIST_STRIDE + sampleIdx * HIST_STRIDE;
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

    let metaBase = srcIdx * HIST_META_STRIDE;
    let writeIdx = histMeta[metaBase];
    let count = histMeta[metaBase + 1u];
    if (count < 2u) { return result; }

    let start = (writeIdx - count + HISTORY_LEN) & HISTORY_MASK;
    let newest = (writeIdx - 1u + HISTORY_LEN) & HISTORY_MASK;

    // Read oldest/newest timestamps from interleaved data
    let oldestBase = histSampleBase(srcIdx, start);
    let newestBase = histSampleBase(srcIdx, newest);
    let tOldest = histData[oldestBase + 5u]; // time field at offset 5
    let tNewest = histData[newestBase + 5u];
    let timeSpan = simTime - tOldest;
    if (timeSpan < NR_TOLERANCE) { return result; }

    // Current distance to newest sample
    let nxPos = histData[newestBase + 0u];
    let nyPos = histData[newestBase + 1u];
    var cdx: f32; var cdy: f32;
    if (periodic) {
        let d = sdMinImageDisp(obsX, obsY, nxPos, nyPos, domW, domH, topoMode);
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
        } else { segK = 0; }
        segK = clamp(segK, 0, i32(count) - 2);

        // Walk to correct segment
        for (var w = 0; w < 256; w++) {
            if (segK >= i32(count) - 2) { break; }
            let nextBase = histSampleBase(srcIdx, (start + u32(segK + 1)) & HISTORY_MASK);
            if (histData[nextBase + 5u] > t) { break; }
            segK++;
        }
        for (var w = 0; w < 256; w++) {
            if (segK <= 0) { break; }
            let curBase = histSampleBase(srcIdx, (start + u32(segK)) & HISTORY_MASK);
            if (histData[curBase + 5u] <= t) { break; }
            segK--;
        }

        var prevSegK: i32 = -1;
        for (var iter = 0u; iter < NR_MAX_ITER; iter++) {
            if (segK == prevSegK) { break; }
            prevSegK = segK;

            let loBase = histSampleBase(srcIdx, (start + u32(segK)) & HISTORY_MASK);
            let hiBase = histSampleBase(srcIdx, ((start + u32(segK)) + 1u) & HISTORY_MASK);
            let tLo = histData[loBase + 5u];
            let segDt = histData[hiBase + 5u] - tLo;
            if (segDt < NR_TOLERANCE) {
                if (segK < i32(count) - 2) { segK++; prevSegK = -1; continue; }
                break;
            }

            let xLo = histData[loBase]; let yLo = histData[loBase + 1u];
            var vxEff: f32; var vyEff: f32;
            if (periodic) {
                let d = sdMinImageDisp(xLo, yLo, histData[hiBase], histData[hiBase + 1u], domW, domH, topoMode);
                vxEff = d.x / segDt; vyEff = d.y / segDt;
            } else {
                vxEff = (histData[hiBase] - xLo) / segDt;
                vyEff = (histData[hiBase + 1u] - yLo) / segDt;
            }

            let s = t - tLo;
            let sx_interp = xLo + vxEff * s;
            let sy_interp = yLo + vyEff * s;

            var dx: f32; var dy: f32;
            if (periodic) {
                let d = sdMinImageDisp(obsX, obsY, sx_interp, sy_interp, domW, domH, topoMode);
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

            for (var w2 = 0; w2 < 64; w2++) {
                if (segK >= i32(count) - 2) { break; }
                let ni = histSampleBase(srcIdx, (start + u32(segK + 1)) & HISTORY_MASK);
                if (histData[ni + 5u] > t) { break; }
                segK++;
            }
            for (var w2 = 0; w2 < 64; w2++) {
                if (segK <= 0) { break; }
                let ci = histSampleBase(srcIdx, (start + u32(segK)) & HISTORY_MASK);
                if (histData[ci + 5u] <= t) { break; }
                segK--;
            }
        }

        // ─── Phase 2: Exact quadratic on converged segment (+/- 1 neighbor) ───
        let center = segK;
        for (var offset = 0; offset <= 1; offset++) {
            for (var dir = select(-1, 1, offset == 0); dir <= 1; dir += 2) {
                let k = center + offset * dir;
                if (k < 0 || k > i32(count) - 2) { continue; }

                let loBase = histSampleBase(srcIdx, (start + u32(k)) & HISTORY_MASK);
                let hiBase = histSampleBase(srcIdx, ((start + u32(k)) + 1u) & HISTORY_MASK);
                let tLo = histData[loBase + 5u];
                let segDt = histData[hiBase + 5u] - tLo;
                if (segDt < NR_TOLERANCE) { continue; }

                let xLo = histData[loBase]; let yLo = histData[loBase + 1u];
                let xHi = histData[hiBase]; let yHi = histData[hiBase + 1u];

                var dx: f32; var dy: f32;
                var vx: f32; var vy: f32;
                if (periodic) {
                    let d0 = sdMinImageDisp(obsX, obsY, xLo, yLo, domW, domH, topoMode);
                    dx = d0.x; dy = d0.y;
                    let d1 = sdMinImageDisp(xLo, yLo, xHi, yHi, domW, domH, topoMode);
                    vx = d1.x / segDt; vy = d1.y / segDt;
                } else {
                    dx = xLo - obsX; dy = yLo - obsY;
                    vx = (xHi - xLo) / segDt; vy = (yHi - yLo) / segDt;
                }

                let rSq = dx * dx + dy * dy;
                let vSq = vx * vx + vy * vy;
                let dDotV = dx * vx + dy * vy;
                let T = simTime - tLo;

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
                // Interpolate velocity from interleaved data
                let loVx = histData[loBase + 2u]; let hiVx = histData[hiBase + 2u];
                let loVy = histData[loBase + 3u]; let hiVy = histData[hiBase + 3u];
                result.vx = loVx + frac * (hiVx - loVx);
                result.vy = loVy + frac * (hiVy - loVy);
                // Interpolate angw
                let loAngw = histData[loBase + 4u]; let hiAngw = histData[hiBase + 4u];
                result.angw = loAngw + frac * (hiAngw - loAngw);
                result.valid = true;
                return result;
            }
        }
    }

    // Dead particles: don't extrapolate past buffer
    if (isDead) { return result; }

    // ─── Phase 3: Extrapolation from oldest sample ───
    {
        let xStart = histData[oldestBase];
        let yStart = histData[oldestBase + 1u];
        var dx: f32; var dy: f32;
        if (periodic) {
            let d = sdMinImageDisp(obsX, obsY, xStart, yStart, domW, domH, topoMode);
            dx = d.x; dy = d.y;
        } else {
            dx = xStart - obsX; dy = yStart - obsY;
        }

        let vx = histData[oldestBase + 2u];
        let vy = histData[oldestBase + 3u];
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
        let creationTimeBits = histMeta[srcIdx * HIST_META_STRIDE + 2u];
        let creationTime = bitcast<f32>(creationTimeBits);
        if (tOldest + s_sol < creationTime) { return result; }

        result.x = xStart + vx * s_sol;
        result.y = yStart + vy * s_sol;
        result.vx = vx;
        result.vy = vy;
        result.angw = histData[oldestBase + 4u];
        result.valid = true;
        return result;
    }
}
