// ─── Topology Module ───
// Minimum-image separation and boundary wrapping for T^2, Klein bottle, and RP^2.

import { TORUS, KLEIN, RP2 } from './config.js';

const _c = { x: 0, y: 0 };

/** Wrap a signed displacement into [-half, +half]. */
function torusWrap(d, full, half) {
    if (d > half) d -= full; else if (d < -half) d += full;
    return d;
}

/**
 * Minimum-image separation from (ox,oy) to (sx,sy). Writes into out.
 * Klein/RP^2 need absolute source coords because glide reflections are position-dependent.
 */
export function minImage(ox, oy, sx, sy, topology, W, H, halfW, halfH, out) {
    let dx = sx - ox;
    let dy = sy - oy;

    if (topology === TORUS) {
        out.x = torusWrap(dx, W, halfW);
        out.y = torusWrap(dy, H, halfH);
        return;
    }

    // Klein/RP²: translational periods are 2H (Klein) or 2W & 2H (RP²).
    // Candidate 0 must NOT torus-wrap axes that only have glide reflections,
    // otherwise it creates phantom images (e.g. (sx, sy±H) instead of (W-sx, sy±H)).
    const doubleH = 2 * H;

    if (topology === KLEIN) {
        // Candidate 0: x periodic (period W), y not periodic (period 2H > domain)
        let bx = torusWrap(dx, W, halfW);
        let by = dy;
        let bestDx = bx, bestDy = by, bestSq = bx * bx + by * by;

        // Candidate 1: y-glide  (x,y) ~ (W-x, y+H)
        _c.x = torusWrap((W - sx) - ox, W, halfW);
        _c.y = torusWrap((sy + H) - oy, doubleH, H);
        let sq = _c.x * _c.x + _c.y * _c.y;
        if (sq < bestSq) { bestDx = _c.x; bestDy = _c.y; }

        out.x = bestDx;
        out.y = bestDy;
        return;
    }

    // RP²: both axes have glide reflections, translational periods 2W & 2H
    const doubleW = 2 * W;

    // Candidate 0: no wrapping (|dx| ≤ W, |dy| ≤ H, within half-periods)
    let bestDx = dx, bestDy = dy, bestSq = dx * dx + dy * dy;

    // Candidate 1: y-glide  (x,y) ~ (W-x, y+H)
    _c.x = (W - sx) - ox;
    _c.y = torusWrap((sy + H) - oy, doubleH, H);
    let sq = _c.x * _c.x + _c.y * _c.y;
    if (sq < bestSq) { bestDx = _c.x; bestDy = _c.y; bestSq = sq; }

    // Candidate 2: x-glide  (x,y) ~ (x+W, H-y)
    _c.x = torusWrap((sx + W) - ox, doubleW, W);
    _c.y = (H - sy) - oy;
    sq = _c.x * _c.x + _c.y * _c.y;
    if (sq < bestSq) { bestDx = _c.x; bestDy = _c.y; bestSq = sq; }

    // Candidate 3: both glides
    _c.x = torusWrap((2 * W - sx) - ox, doubleW, W);
    _c.y = torusWrap((2 * H - sy) - oy, doubleH, H);
    sq = _c.x * _c.x + _c.y * _c.y;
    if (sq < bestSq) { bestDx = _c.x; bestDy = _c.y; }

    out.x = bestDx;
    out.y = bestDy;
}

/**
 * Wrap position into [0,W]x[0,H], flipping velocity/spin for non-orientable crossings.
 */
export function wrapPosition(p, topology, W, H) {
    if (topology === TORUS) {
        if (p.pos.x < 0) p.pos.x += W;
        else if (p.pos.x > W) p.pos.x -= W;
        if (p.pos.y < 0) p.pos.y += H;
        else if (p.pos.y > H) p.pos.y -= H;
        return;
    }

    if (topology === KLEIN) {
        if (p.pos.x < 0) p.pos.x += W;
        else if (p.pos.x > W) p.pos.x -= W;
        // y-wrap: glide reflection mirrors x-position and negates x-velocity
        if (p.pos.y < 0) {
            p.pos.y += H;
            p.pos.x = W - p.pos.x;
            p.w.x = -p.w.x; p.vel.x = -p.vel.x;
            p.angw = -p.angw; p.angVel = -p.angVel;
        } else if (p.pos.y > H) {
            p.pos.y -= H;
            p.pos.x = W - p.pos.x;
            p.w.x = -p.w.x; p.vel.x = -p.vel.x;
            p.angw = -p.angw; p.angVel = -p.angVel;
        }
        return;
    }

    // RP2: both axes carry glide reflections
    if (p.pos.x < 0) {
        p.pos.x += W;
        p.pos.y = H - p.pos.y;
        p.w.y = -p.w.y; p.vel.y = -p.vel.y;
        p.angw = -p.angw; p.angVel = -p.angVel;
    } else if (p.pos.x > W) {
        p.pos.x -= W;
        p.pos.y = H - p.pos.y;
        p.w.y = -p.w.y; p.vel.y = -p.vel.y;
        p.angw = -p.angw; p.angVel = -p.angVel;
    }
    if (p.pos.y < 0) {
        p.pos.y += H;
        p.pos.x = W - p.pos.x;
        p.w.x = -p.w.x; p.vel.x = -p.vel.x;
        p.angw = -p.angw; p.angVel = -p.angVel;
    } else if (p.pos.y > H) {
        p.pos.y -= H;
        p.pos.x = W - p.pos.x;
        p.w.x = -p.w.x; p.vel.x = -p.vel.x;
        p.angw = -p.angw; p.angVel = -p.angVel;
    }
}
