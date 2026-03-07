// ─── Signal Delay ───
// Solves the light-cone equation |x_src(t_ret) - x_obs| = now - t_ret (c=1)
// via NR convergence to segment, exact quadratic on that segment, and
// constant-velocity extrapolation past the buffer.

import { HISTORY_SIZE, NR_TOLERANCE, NR_MAX_ITER, EPSILON } from './config.js';
import { TORUS, minImage } from './topology.js';

const _miOut = { x: 0, y: 0 };

// Shared return object -- caller must read before next call
const _delayedOut = { x: 0, y: 0, vx: 0, vy: 0, angw: 0 };


/** Solve light-cone equation; returns shared {x,y,vx,vy} or null. */
export function getDelayedState(source, observer, simTime, periodic, domW, domH, halfDomW, halfDomH, topology = TORUS) {
    if (source.histCount < 2) return null;

    const ox = observer.pos.x, oy = observer.pos.y;
    const N = HISTORY_SIZE;
    const count = source.histCount;
    const start = (source.histHead - count + N) % N;
    const newest = (source.histHead - 1 + N) % N;

    // Time bounds
    const tOldest = source.histTime[start];
    const tNewest = source.histTime[newest];
    const timeSpan = simTime - tOldest;
    if (timeSpan < NR_TOLERANCE) return null;

    // Cache history array references to avoid repeated property lookups
    const histX = source.histX, histY = source.histY;
    const histVx = source.histVx, histVy = source.histVy;
    const histTime = source.histTime;

    let cdx, cdy;
    if (periodic) {
        minImage(ox, oy, histX[newest], histY[newest],
                 topology, domW, domH, halfDomW, halfDomH, _miOut);
        cdx = _miOut.x; cdy = _miOut.y;
    } else {
        cdx = histX[newest] - ox;
        cdy = histY[newest] - oy;
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
    while (segK < count - 2 && histTime[(start + segK + 1) % N] <= t) segK++;
    while (segK > 0 && histTime[(start + segK) % N] > t) segK--;

    let prevSegK = -1;
    for (let iter = 0; iter < NR_MAX_ITER; iter++) {
        if (segK === prevSegK) break;   // segment stabilized → go to analytical phase
        prevSegK = segK;

        const loIdx = (start + segK) % N;
        const hiIdx = (loIdx + 1) % N;
        const tLo = histTime[loIdx];
        const segDt = histTime[hiIdx] - tLo;
        if (segDt < NR_TOLERANCE) {
            if (segK < count - 2) { segK++; prevSegK = -1; continue; }
            break buffer;
        }

        const xLo = histX[loIdx], yLo = histY[loIdx];
        let vxEff, vyEff;
        if (periodic) {
            minImage(xLo, yLo, histX[hiIdx], histY[hiIdx],
                     topology, domW, domH, halfDomW, halfDomH, _miOut);
            vxEff = _miOut.x / segDt;
            vyEff = _miOut.y / segDt;
        } else {
            vxEff = (histX[hiIdx] - xLo) / segDt;
            vyEff = (histY[hiIdx] - yLo) / segDt;
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
        if (dist < NR_TOLERANCE) break;               // source ≈ observer, skip to quadratic

        const g  = dist - (simTime - t);
        const gp = (dx * vxEff + dy * vyEff) / dist + 1;
        if (Math.abs(gp) < NR_TOLERANCE) break;

        t -= g / gp;

        if (t < tOldest) t = tOldest;
        if (t > tNewest) t = tNewest;

        while (segK < count - 2 && histTime[(start + segK + 1) % N] <= t) segK++;
        while (segK > 0 && histTime[(start + segK) % N] > t) segK--;
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
            const tLo = histTime[loIdx];
            const segDt = histTime[hiIdx] - tLo;
            if (segDt < NR_TOLERANCE) continue;

            const xLo = histX[loIdx], yLo = histY[loIdx];
            const xHi = histX[hiIdx], yHi = histY[hiIdx];

            let dx, dy, vx, vy;
            if (periodic) {
                minImage(ox, oy, xLo, yLo,
                         topology, domW, domH, halfDomW, halfDomH, _miOut);
                dx = _miOut.x; dy = _miOut.y;
                minImage(xLo, yLo, xHi, yHi,
                         topology, domW, domH, halfDomW, halfDomH, _miOut);
                vx = _miOut.x / segDt;
                vy = _miOut.y / segDt;
            } else {
                dx = xLo - ox;
                dy = yLo - oy;
                vx = (xHi - xLo) / segDt;
                vy = (yHi - yLo) / segDt;
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
            if (Math.abs(a) < NR_TOLERANCE) {
                // v ~ c: degenerate linear case
                if (Math.abs(h) < NR_TOLERANCE) continue;
                s = -c / (2 * h);
            } else {
                const s1 = (-h + sqrtDisc) / a;
                const s2 = (-h - sqrtDisc) / a;
                // Prefer most recent valid root (largest s in [0, segDt])
                const ok1 = s1 >= -EPSILON && s1 <= segDt + EPSILON;
                const ok2 = s2 >= -EPSILON && s2 <= segDt + EPSILON;
                if (ok1 && ok2) s = Math.max(s1, s2);
                else if (ok1) s = s1;
                else if (ok2) s = s2;
                else continue;
            }

            if (s < 0) s = 0;
            else if (s > segDt) s = segDt;

            const frac = s / segDt;
            _delayedOut.x  = xLo + frac * (xHi - xLo);
            _delayedOut.y  = yLo + frac * (yHi - yLo);
            _delayedOut.vx = histVx[loIdx] + frac * (histVx[hiIdx] - histVx[loIdx]);
            _delayedOut.vy = histVy[loIdx] + frac * (histVy[hiIdx] - histVy[loIdx]);
            _delayedOut.angw = source.histAngW[loIdx] + frac * (source.histAngW[hiIdx] - source.histAngW[loIdx]);
            return _delayedOut;
        }
    }

    } // buffer

    // Dead particles have complete history — don't extrapolate past their buffer
    if (source.deathTime < Infinity) return null;

    // ─── Extrapolation: backward from oldest sample at constant velocity ───
    {
        let dx, dy;
        if (periodic) {
            minImage(ox, oy, histX[start], histY[start],
                     topology, domW, domH, halfDomW, halfDomH, _miOut);
            dx = _miOut.x; dy = _miOut.y;
        } else {
            dx = histX[start] - ox;
            dy = histY[start] - oy;
        }

        const vx = histVx[start], vy = histVy[start];
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
        if (Math.abs(a) < NR_TOLERANCE) {
            if (Math.abs(h) < NR_TOLERANCE) return null;
            s = -c / (2 * h);
        } else {
            const s1 = (-h + sqrtDisc) / a;
            const s2 = (-h - sqrtDisc) / a;
            // Pick s <= 0 root closest to 0 (most recent past)
            const ok1 = s1 <= EPSILON;
            const ok2 = s2 <= EPSILON;
            if (ok1 && ok2) s = Math.max(s1, s2);
            else if (ok1) s = s1;
            else if (ok2) s = s2;
            else return null;
        }

        if (s > 0) s = 0;

        // Reject extrapolation past particle creation (particle didn't exist yet)
        if (tOldest + s < source.creationTime) return null;

        _delayedOut.x  = histX[start] + vx * s;
        _delayedOut.y  = histY[start] + vy * s;
        _delayedOut.vx = vx;
        _delayedOut.vy = vy;
        _delayedOut.angw = source.histAngW[start];
        return _delayedOut;
    }
}
