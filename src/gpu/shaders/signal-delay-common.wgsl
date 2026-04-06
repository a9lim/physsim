// ─── Signal Delay Common ───
// Shared binary-search + quadratic light-cone solver for interleaved history buffers.
// Prepended to consuming shaders (after shared-topology.wgsl which provides fullMinImageP).
// Callers declare histData/histMeta bindings.
//
// Requires constants: HIST_STRIDE, HIST_META_STRIDE, HISTORY_LEN, HISTORY_MASK,
// SOLVE_TOLERANCE, EPSILON, TOPO_TORUS, TOPO_KLEIN.

struct DelayedState {
    x: f32, y: f32,
    vx: f32, vy: f32,
    angw: f32,
    valid: bool,
};

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
    if (timeSpan < SOLVE_TOLERANCE) { return result; }

    // Current distance to newest sample
    let nxPos = histData[newestBase + 0u];
    let nyPos = histData[newestBase + 1u];
    var cdx: f32; var cdy: f32;
    if (periodic) {
        let d = fullMinImageP(obsX, obsY, nxPos, nyPos, domW, domH, topoMode);
        cdx = d.x; cdy = d.y;
    } else {
        cdx = nxPos - obsX; cdy = nyPos - obsY;
    }
    let distSq = cdx * cdx + cdy * cdy;

    // ─── Phase 1: Binary search for the segment containing the light-cone root ───
    // g(t) = |x_src(t) - x_obs| - (now - t) is strictly monotone increasing
    // (g' = d_hat·v + 1 > 0 for |v| < c), so binary search on segment boundaries works.
    if (distSq <= 4.0 * timeSpan * timeSpan) {
        var bsLo: i32 = 0;
        var bsHi: i32 = i32(count) - 2;
        for (var bsIter = 0; bsIter < 16; bsIter++) {
            if (bsLo >= bsHi) { break; }
            let mid = (bsLo + bsHi) >> 1;
            let midBase = histSampleBase(srcIdx, (start + u32(mid + 1)) & HISTORY_MASK);
            let tMid = histData[midBase + 5u];

            var bx: f32; var by: f32;
            if (periodic) {
                let d = fullMinImageP(obsX, obsY, histData[midBase], histData[midBase + 1u], domW, domH, topoMode);
                bx = d.x; by = d.y;
            } else {
                bx = histData[midBase] - obsX;
                by = histData[midBase + 1u] - obsY;
            }
            let g = sqrt(bx * bx + by * by) - (simTime - tMid);

            if (g < 0.0) { bsLo = mid + 1; }
            else { bsHi = mid; }
        }
        var segK = bsLo;

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
                if (segDt < SOLVE_TOLERANCE) { continue; }

                let xLo = histData[loBase]; let yLo = histData[loBase + 1u];
                let xHi = histData[hiBase]; let yHi = histData[hiBase + 1u];

                var dx: f32; var dy: f32;
                var vx: f32; var vy: f32;
                if (periodic) {
                    let d0 = fullMinImageP(obsX, obsY, xLo, yLo, domW, domH, topoMode);
                    dx = d0.x; dy = d0.y;
                    let d1 = fullMinImageP(xLo, yLo, xHi, yHi, domW, domH, topoMode);
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
                if (abs(a) < SOLVE_TOLERANCE) {
                    if (abs(h) < SOLVE_TOLERANCE) { continue; }
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
            let d = fullMinImageP(obsX, obsY, xStart, yStart, domW, domH, topoMode);
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
        if (abs(a) < SOLVE_TOLERANCE) {
            if (abs(h) < SOLVE_TOLERANCE) { return result; }
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
