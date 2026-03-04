// ─── Signal Delay ───
// Solves the light-cone equation |x_src(t_ret) - x_obs| = now - t_ret (c=1)
// via NR convergence to segment, exact quadratic on that segment, and
// constant-velocity extrapolation past the buffer.

import { HISTORY_SIZE } from './config.js';
import { TORUS, minImage } from './topology.js';

const _miOut = { x: 0, y: 0 };

// Shared return object -- caller must read before next call
const _delayedOut = { x: 0, y: 0, vx: 0, vy: 0 };

const NR_MAX_ITER = 6;

/** Solve light-cone equation; returns shared {x,y,vx,vy} or null. */
export function getDelayedState(source, observer, simTime, periodic, domW, domH, halfDomW, halfDomH, topology = TORUS) {
    if (!source.histX || source.histCount < 2) return null;

    const ox = observer.pos.x, oy = observer.pos.y;
    const N = HISTORY_SIZE;
    const count = source.histCount;
    const start = (source.histHead - count + N) % N;
    const newest = (source.histHead - 1 + N) % N;

    // Time bounds
    const tOldest = source.histTime[start];
    const tNewest = source.histTime[newest];
    const timeSpan = simTime - tOldest;
    if (timeSpan < 1e-12) return null;

    let cdx, cdy;
    if (periodic) {
        minImage(ox, oy, source.histX[newest], source.histY[newest],
                 topology, domW, domH, halfDomW, halfDomH, _miOut);
        cdx = _miOut.x; cdy = _miOut.y;
    } else {
        cdx = source.histX[newest] - ox;
        cdy = source.histY[newest] - oy;
    }
    const distSq = cdx * cdx + cdy * cdy;

    // ─── Buffer search: NR + quadratic on recorded history ───
    // Skip if current distance > 2x buffer time span (solution predates buffer).
    buffer: {
    if (distSq > 4 * timeSpan * timeSpan) break buffer;

    // ─── Phase 1: Newton-Raphson to locate the correct history segment ───
    // g' = d_hat.v + 1 > 0 for |v| < c, guaranteeing convergence.

    let t = simTime - Math.sqrt(distSq);   // initial guess: light travel time
    if (t < tOldest) t = tOldest;
    if (t > tNewest) t = tNewest;

    // O(1) proportional estimate + short walk for non-uniform spacing
    let segK = Math.floor((t - tOldest) / (tNewest - tOldest) * (count - 1));
    if (segK > count - 2) segK = count - 2;
    if (segK < 0) segK = 0;
    while (segK < count - 2 && source.histTime[(start + segK + 1) % N] <= t) segK++;
    while (segK > 0 && source.histTime[(start + segK) % N] > t) segK--;

    let prevSegK = -1;
    for (let iter = 0; iter < NR_MAX_ITER; iter++) {
        if (segK === prevSegK) break;   // segment stabilized → go to analytical phase
        prevSegK = segK;

        const loIdx = (start + segK) % N;
        const hiIdx = (loIdx + 1) % N;
        const tLo = source.histTime[loIdx];
        const segDt = source.histTime[hiIdx] - tLo;
        if (segDt < 1e-12) {
            if (segK < count - 2) { segK++; prevSegK = -1; continue; }
            break buffer;
        }

        const xLo = source.histX[loIdx], yLo = source.histY[loIdx];
        let vxEff, vyEff;
        if (periodic) {
            minImage(xLo, yLo, source.histX[hiIdx], source.histY[hiIdx],
                     topology, domW, domH, halfDomW, halfDomH, _miOut);
            vxEff = _miOut.x / segDt;
            vyEff = _miOut.y / segDt;
        } else {
            vxEff = (source.histX[hiIdx] - xLo) / segDt;
            vyEff = (source.histY[hiIdx] - yLo) / segDt;
        }

        const s = t - tLo;
        const sx = xLo + vxEff * s;
        const sy = yLo + vyEff * s;

        let dx, dy;
        if (periodic) {
            minImage(ox, oy, sx, sy, topology, domW, domH, halfDomW, halfDomH, _miOut);
            dx = _miOut.x; dy = _miOut.y;
        } else {
            dx = sx - ox; dy = sy - oy;
        }

        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 1e-12) break;               // source ≈ observer, skip to quadratic

        const g  = dist - (simTime - t);
        const gp = (dx * vxEff + dy * vyEff) / dist + 1;
        if (Math.abs(gp) < 1e-12) break;

        t -= g / gp;

        if (t < tOldest) t = tOldest;
        if (t > tNewest) t = tNewest;

        while (segK < count - 2 && source.histTime[(start + segK + 1) % N] <= t) segK++;
        while (segK > 0 && source.histTime[(start + segK) % N] > t) segK--;
    }

    // ─── Phase 2: Exact quadratic on converged segment (and +/- 1 neighbor) ───
    // Piecewise-linear trajectory makes the light-cone equation a quadratic in s.
    const center = segK;
    for (let offset = 0; offset <= 1; offset++) {
        for (let dir = (offset === 0 ? 1 : -1); dir <= 1; dir += 2) {
            const k = center + offset * dir;
            if (k < 0 || k > count - 2) continue;

            const loIdx = (start + k) % N;
            const hiIdx = (loIdx + 1) % N;
            const tLo = source.histTime[loIdx];
            const segDt = source.histTime[hiIdx] - tLo;
            if (segDt < 1e-12) continue;

            let dx, dy;
            if (periodic) {
                minImage(ox, oy, source.histX[loIdx], source.histY[loIdx],
                         topology, domW, domH, halfDomW, halfDomH, _miOut);
                dx = _miOut.x; dy = _miOut.y;
            } else {
                dx = source.histX[loIdx] - ox;
                dy = source.histY[loIdx] - oy;
            }

            let vx, vy;
            if (periodic) {
                minImage(source.histX[loIdx], source.histY[loIdx],
                         source.histX[hiIdx], source.histY[hiIdx],
                         topology, domW, domH, halfDomW, halfDomH, _miOut);
                vx = _miOut.x / segDt;
                vy = _miOut.y / segDt;
            } else {
                vx = (source.histX[hiIdx] - source.histX[loIdx]) / segDt;
                vy = (source.histY[hiIdx] - source.histY[loIdx]) / segDt;
            }

            const rSq = dx * dx + dy * dy;
            const vSq = vx * vx + vy * vy;
            const dDotV = dx * vx + dy * vy;
            const T = simTime - tLo;

            // (v^2 - 1)s^2 + 2(d.v + T)s + (r^2 - T^2) = 0, half-b form
            const a = vSq - 1;
            const h = dDotV + T;
            const c = rSq - T * T;

            const disc = h * h - a * c;
            if (disc < 0) continue;

            const sqrtDisc = Math.sqrt(disc);
            let s;
            if (Math.abs(a) < 1e-12) {
                // v ~ c: degenerate linear case
                if (Math.abs(h) < 1e-12) continue;
                s = -c / (2 * h);
            } else {
                const s1 = (-h + sqrtDisc) / a;
                const s2 = (-h - sqrtDisc) / a;
                // Prefer most recent valid root (largest s in [0, segDt])
                const ok1 = s1 >= -1e-9 && s1 <= segDt + 1e-9;
                const ok2 = s2 >= -1e-9 && s2 <= segDt + 1e-9;
                if (ok1 && ok2) s = Math.max(s1, s2);
                else if (ok1) s = s1;
                else if (ok2) s = s2;
                else continue;
            }

            if (s < 0) s = 0;
            else if (s > segDt) s = segDt;

            const frac = s / segDt;
            _delayedOut.x  = source.histX[loIdx]  + frac * (source.histX[hiIdx]  - source.histX[loIdx]);
            _delayedOut.y  = source.histY[loIdx]  + frac * (source.histY[hiIdx]  - source.histY[loIdx]);
            _delayedOut.vx = source.histVx[loIdx] + frac * (source.histVx[hiIdx] - source.histVx[loIdx]);
            _delayedOut.vy = source.histVy[loIdx] + frac * (source.histVy[hiIdx] - source.histVy[loIdx]);
            return _delayedOut;
        }
    }

    } // buffer

    // ─── Extrapolation: backward from oldest sample at constant velocity ───
    {
        let dx, dy;
        if (periodic) {
            minImage(ox, oy, source.histX[start], source.histY[start],
                     topology, domW, domH, halfDomW, halfDomH, _miOut);
            dx = _miOut.x; dy = _miOut.y;
        } else {
            dx = source.histX[start] - ox;
            dy = source.histY[start] - oy;
        }

        const vx = source.histVx[start], vy = source.histVy[start];
        const rSq = dx * dx + dy * dy;
        const vSq = vx * vx + vy * vy;
        const dDotV = dx * vx + dy * vy;
        const T = timeSpan;

        // Same quadratic, s <= 0 (backward in time)
        const a = vSq - 1;
        const h = dDotV + T;
        const c = rSq - T * T;

        const disc = h * h - a * c;
        if (disc < 0) return null;

        const sqrtDisc = Math.sqrt(disc);
        let s;
        if (Math.abs(a) < 1e-12) {
            if (Math.abs(h) < 1e-12) return null;
            s = -c / (2 * h);
        } else {
            const s1 = (-h + sqrtDisc) / a;
            const s2 = (-h - sqrtDisc) / a;
            // Pick s <= 0 root closest to 0 (most recent past)
            const ok1 = s1 <= 1e-9;
            const ok2 = s2 <= 1e-9;
            if (ok1 && ok2) s = Math.max(s1, s2);
            else if (ok1) s = s1;
            else if (ok2) s = s2;
            else return null;
        }

        if (s > 0) s = 0;

        _delayedOut.x  = source.histX[start] + vx * s;
        _delayedOut.y  = source.histY[start] + vy * s;
        _delayedOut.vx = vx;
        _delayedOut.vy = vy;
        return _delayedOut;
    }
}

/** Binary-search interpolation on a particle's circular history buffer. */
export function interpolateHistory(p, t) {
    if (!p.histX) return null;
    if (p.histCount < 2) return null;

    const N = HISTORY_SIZE;
    const start = (p.histHead - p.histCount + N) % N;

    const oldest = p.histTime[start];
    const newest = p.histTime[(p.histHead - 1 + N) % N];
    if (t < oldest || t > newest) return null;

    let lo = 0, hi = p.histCount - 2;
    while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (p.histTime[(start + mid) % N] <= t) lo = mid;
        else hi = mid - 1;
    }

    const loIdx = (start + lo) % N;
    const hiIdx = (loIdx + 1) % N;
    const dt = p.histTime[hiIdx] - p.histTime[loIdx];
    if (dt < 1e-12) return { x: p.histX[loIdx], y: p.histY[loIdx], vx: p.histVx[loIdx], vy: p.histVy[loIdx] };

    const frac = (t - p.histTime[loIdx]) / dt;
    return {
        x:  p.histX[loIdx]  + frac * (p.histX[hiIdx]  - p.histX[loIdx]),
        y:  p.histY[loIdx]  + frac * (p.histY[hiIdx]  - p.histY[loIdx]),
        vx: p.histVx[loIdx] + frac * (p.histVx[hiIdx] - p.histVx[loIdx]),
        vy: p.histVy[loIdx] + frac * (p.histVy[hiIdx] - p.histVy[loIdx]),
    };
}
