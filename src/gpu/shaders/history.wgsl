// ─── Signal Delay History ───
// Ring buffer recording + Newton-Raphson light-cone retrieval.
// f32 precision with relative-time encoding for GPU.
//
// Standalone shader — defines own structs (NOT prepended with common.wgsl).

const HISTORY_LEN: u32 = 256u;
const HISTORY_MASK: u32 = 255u;
const NR_MAX_ITER: u32 = 8u;
const NR_TOLERANCE: f32 = 1e-5;
const EPSILON: f32 = 1e-9;

// ── Packed struct definitions ──

struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

// Must match SimUniforms byte layout in common.wgsl / writeUniforms() exactly.
// Only simTime is used here; preceding fields kept as padding for alignment.
struct SimUniforms {
    _dt: f32,               // [0] dt (unused here)
    simTime: f32,           // [1] simTime
};

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read> particles: array<ParticleState>;

// History ring buffers
@group(1) @binding(0) var<storage, read_write> histPosX: array<f32>;
@group(1) @binding(1) var<storage, read_write> histPosY: array<f32>;
@group(1) @binding(2) var<storage, read_write> histVelWX: array<f32>;
@group(1) @binding(3) var<storage, read_write> histVelWY: array<f32>;
@group(1) @binding(4) var<storage, read_write> histAngW: array<f32>;
@group(1) @binding(5) var<storage, read_write> histTime: array<f32>;
// histMeta: [writeIdx, count] per particle (u32 pairs)
@group(1) @binding(6) var<storage, read_write> histMeta: array<u32>;

const ALIVE_BIT: u32 = 1u;

@compute @workgroup_size(64)
fn recordHistory(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= arrayLength(&particles)) { return; }
    if ((particles[i].flags & ALIVE_BIT) == 0u) { return; }

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

// Topology-aware minimum image displacement (torus only for GPU hot path)
fn minImageDisp(ox: f32, oy: f32, sx: f32, sy: f32,
                domW: f32, domH: f32) -> vec2f {
    var dx = sx - ox;
    var dy = sy - oy;
    let halfW = domW * 0.5;
    let halfH = domH * 0.5;
    if (dx > halfW) { dx -= domW; } else if (dx < -halfW) { dx += domW; }
    if (dy > halfH) { dy -= domH; } else if (dy < -halfH) { dy += domH; }
    return vec2f(dx, dy);
}

fn getDelayedStateGPU(
    srcIdx: u32,
    obsX: f32, obsY: f32,
    simTime: f32,
    periodic: bool,
    domW: f32, domH: f32,
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
        let d = minImageDisp(obsX, obsY, nxPos, nyPos, domW, domH);
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
                let d = minImageDisp(xLo, yLo, histPosX[hiIdx], histPosY[hiIdx], domW, domH);
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
                let d = minImageDisp(obsX, obsY, sx_interp, sy_interp, domW, domH);
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
                    let d0 = minImageDisp(obsX, obsY, xLo, yLo, domW, domH);
                    dx = d0.x; dy = d0.y;
                    let d1 = minImageDisp(xLo, yLo, xHi, yHi, domW, domH);
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

                let sqrtDisc = sqrt(disc);
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
            let d = minImageDisp(obsX, obsY, xStart, yStart, domW, domH);
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
