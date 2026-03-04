// ─── Signal Delay ───
// Retarded potentials: solve for delayed time t_del such that
// |x_source(t_del) - x_observer(now)| = c·(now - t_del) where c = 1.
// Uses Newton-Raphson with 3 iterations.

import { HISTORY_SIZE } from './config.js';

/**
 * Solve light-cone equation for delayed state of source as seen by observer.
 * @param {Object} source - Source particle with history buffers
 * @param {Object} observer - Observer particle
 * @param {number} simTime - Current simulation time
 * @returns {Object|null} Interpolated {x, y, vx, vy} or null if history insufficient
 */
export function getDelayedState(source, observer, simTime) {
    const ox = observer.pos.x, oy = observer.pos.y;

    // Initial guess: t_del = now - |current separation|
    const dx0 = source.pos.x - ox, dy0 = source.pos.y - oy;
    let tDel = simTime - Math.sqrt(dx0 * dx0 + dy0 * dy0);

    for (let iter = 0; iter < 3; iter++) {
        const sp = interpolateHistory(source, tDel);
        if (!sp) return null;

        const dx = sp.x - ox, dy = sp.y - oy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const residual = dist - (simTime - tDel);
        if (Math.abs(residual) < 0.01) break;

        // Newton step: d(residual)/d(tDel) ≈ -(v·r̂)/r - 1
        const denom = 1 + (sp.vx * dx + sp.vy * dy) / (dist * dist + 1);
        tDel += residual / denom;
    }

    return interpolateHistory(source, tDel);
}

/**
 * Interpolate position/velocity from circular history buffer at time t.
 * @param {Object} p - Particle with history buffers
 * @param {number} t - Time to interpolate at
 * @returns {Object|null} {x, y, vx, vy} or null if out of range
 */
export function interpolateHistory(p, t) {
    if (!p.histX) return null;
    if (p.histCount < 2) return null;

    // Find bracketing entries via linear scan (buffer is chronological)
    const N = HISTORY_SIZE;
    const start = (p.histHead - p.histCount + N) % N;

    // Check bounds
    const oldest = p.histTime[start];
    const newest = p.histTime[(p.histHead - 1 + N) % N];
    if (t < oldest || t > newest) return null;

    // Linear scan from oldest to find bracket
    let lo = start;
    for (let k = 0; k < p.histCount - 1; k++) {
        const idx = (start + k) % N;
        const nextIdx = (start + k + 1) % N;
        if (p.histTime[idx] <= t && t <= p.histTime[nextIdx]) {
            lo = idx;
            break;
        }
    }
    const hi = (lo + 1) % N;
    const dt = p.histTime[hi] - p.histTime[lo];
    if (dt < 1e-12) return { x: p.histX[lo], y: p.histY[lo], vx: p.histVx[lo], vy: p.histVy[lo] };

    const frac = (t - p.histTime[lo]) / dt;
    return {
        x:  p.histX[lo]  + frac * (p.histX[hi]  - p.histX[lo]),
        y:  p.histY[lo]  + frac * (p.histY[hi]  - p.histY[lo]),
        vx: p.histVx[lo] + frac * (p.histVx[hi] - p.histVx[lo]),
        vy: p.histVy[lo] + frac * (p.histVy[hi] - p.histVy[lo]),
    };
}
