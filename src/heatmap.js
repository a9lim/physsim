// ─── Potential Field Heatmap ───
// 48x48 offscreen canvas, diverging colormap, updates every 6 frames.
// When Barnes-Hut is enabled, uses tree walk for O(GRID² log N) instead of O(GRID² N).

import { SOFTENING_SQ, BH_THETA, YUKAWA_G2 } from './config.js';
import { getDelayedState } from './signal-delay.js';

const GRID_SIZE = 48;
const GRID_SQ = GRID_SIZE * GRID_SIZE;
const UPDATE_INTERVAL = 6;
const SENSITIVITY = 3;   // tanh scaling: phi * SENSITIVITY -> 0-1
const MAX_ALPHA = 100;

/** Fast tanh approximation: rational Padé x(27+x²)/(27+9x²), max error ~0.4% */
function fastTanh(x) {
    if (x > 4.9) return 1;
    if (x < -4.9) return -1;
    const x2 = x * x;
    return x * (27 + x2) / (27 + 9 * x2);
}

/**
 * Iterative BH tree walk for scalar potential at point (wx, wy).
 * Accumulates into _treeOut.g and _treeOut.e.
 */
let _hmStack = new Int32Array(256);

function treePotential(pool, rootIdx, wx, wy, thetaSq, softeningSq, doYukawa, yukawaMu) {
    let stackTop = 0;
    if (_hmStack.length < pool.maxNodes) _hmStack = new Int32Array(pool.maxNodes);
    _hmStack[stackTop++] = rootIdx;

    while (stackTop > 0) {
        const nodeIdx = _hmStack[--stackTop];
        if (pool.totalMass[nodeIdx] === 0 && pool.totalCharge[nodeIdx] === 0) continue;

        const dx = pool.comX[nodeIdx] - wx;
        const dy = pool.comY[nodeIdx] - wy;
        const dSq = dx * dx + dy * dy;
        const size = pool.bw[nodeIdx] * 2;

        if (!pool.divided[nodeIdx] && pool.pointCount[nodeIdx] > 0) {
            const base = nodeIdx * pool.nodeCapacity;
            let gP = 0, eP = 0, yP = 0;
            for (let i = 0; i < pool.pointCount[nodeIdx]; i++) {
                const p = pool.points[base + i];
                const pdx = wx - p.pos.x, pdy = wy - p.pos.y;
                const rSq = pdx * pdx + pdy * pdy + softeningSq;
                const invR = 1 / Math.sqrt(rSq);
                gP -= p.mass * invR;
                eP += p.charge * invR;
                if (doYukawa) {
                    const r = 1 / invR;
                    yP -= YUKAWA_G2 * p.mass * Math.exp(-yukawaMu * r) * invR;
                }
            }
            _treeOut.g += gP;
            _treeOut.e += eP;
            _treeOut.y += yP;
        } else if (pool.divided[nodeIdx] && (size * size < thetaSq * dSq)) {
            const rSq = dSq + softeningSq;
            const invR = 1 / Math.sqrt(rSq);
            _treeOut.g -= pool.totalMass[nodeIdx] * invR;
            _treeOut.e += pool.totalCharge[nodeIdx] * invR;
            if (doYukawa) {
                const r = 1 / invR;
                _treeOut.y -= YUKAWA_G2 * pool.totalMass[nodeIdx] * Math.exp(-yukawaMu * r) * invR;
            }
        } else if (pool.divided[nodeIdx]) {
            _hmStack[stackTop++] = pool.nw[nodeIdx];
            _hmStack[stackTop++] = pool.ne[nodeIdx];
            _hmStack[stackTop++] = pool.sw[nodeIdx];
            _hmStack[stackTop++] = pool.se[nodeIdx];
        }
    }
}

// Reusable output for treePotential
const _treeOut = { g: 0, e: 0, y: 0 };
// Reusable observer object for signal delay (avoids allocation per grid cell)
const _hmObs = { pos: { x: 0, y: 0 } };

// Heatmap display modes
export const HEATMAP_MODES = ['all', 'gravity', 'electric', 'yukawa'];

export default class Heatmap {
    constructor() {
        this.enabled = false;
        this.mode = 'all'; // 'all' | 'gravity' | 'electric' | 'yukawa'
        this.canvas = document.createElement('canvas');
        this.canvas.width = GRID_SIZE;
        this.canvas.height = GRID_SIZE;
        this.ctx = this.canvas.getContext('2d');
        this.frameCount = 0;
        this.gravPotential = new Float32Array(GRID_SQ);
        this.elecPotential = new Float32Array(GRID_SQ);
        this.yukawaPotential = new Float32Array(GRID_SQ);
        // Persistent ImageData — avoid creating a new one every update
        this._imgData = this.ctx.createImageData(GRID_SIZE, GRID_SIZE);
    }

    update(particles, camera, width, height, pool, root, barnesHutEnabled, relativityEnabled, simTime, periodic, domW, domH, topology, softeningSq = SOFTENING_SQ, yukawaEnabled = false, yukawaMu = 0.2) {
        if (!this.enabled) return;
        if (++this.frameCount % UPDATE_INTERVAL !== 0) return;

        const zoom = camera.zoom;
        const cx = camera.x, cy = camera.y;
        const halfW = width / (2 * zoom), halfH = height / (2 * zoom);
        const left = cx - halfW, top = cy - halfH;
        const cellW = (2 * halfW) / GRID_SIZE;
        const cellH = (2 * halfH) / GRID_SIZE;
        const n = particles.length;
        const useTree = barnesHutEnabled && root >= 0;
        const thetaSq = BH_THETA * BH_THETA;
        const useDelay = relativityEnabled;
        const halfDomW = domW * 0.5, halfDomH = domH * 0.5;
        const doYukawa = yukawaEnabled && (this.mode === 'all' || this.mode === 'yukawa');

        for (let gy = 0; gy < GRID_SIZE; gy++) {
            const wy = top + (gy + 0.5) * cellH;
            for (let gx = 0; gx < GRID_SIZE; gx++) {
                const wx = left + (gx + 0.5) * cellW;
                let gPhi = 0, ePhi = 0, yPhi = 0;

                if (useTree) {
                    _treeOut.g = 0;
                    _treeOut.e = 0;
                    _treeOut.y = 0;
                    treePotential(pool, root, wx, wy, thetaSq, softeningSq, doYukawa, yukawaMu);
                    gPhi = _treeOut.g;
                    ePhi = _treeOut.e;
                    yPhi = _treeOut.y;
                } else {
                    for (let i = 0; i < n; i++) {
                        const p = particles[i];
                        let px, py;
                        if (useDelay && p.histCount >= 2) {
                            _hmObs.pos.x = wx; _hmObs.pos.y = wy;
                            const ret = getDelayedState(p, _hmObs, simTime, periodic, domW, domH, halfDomW, halfDomH, topology);
                            if (ret) { px = ret.x; py = ret.y; }
                            else { px = p.pos.x; py = p.pos.y; }
                        } else {
                            px = p.pos.x; py = p.pos.y;
                        }
                        const dx = wx - px, dy = wy - py;
                        const rSq = dx * dx + dy * dy + softeningSq;
                        const invR = 1 / Math.sqrt(rSq);
                        gPhi -= p.mass * invR;
                        ePhi += p.charge * invR;
                        if (doYukawa) {
                            const r = 1 / invR;
                            yPhi -= YUKAWA_G2 * p.mass * Math.exp(-yukawaMu * r) * invR;
                        }
                    }
                }

                const idx = gy * GRID_SIZE + gx;
                this.gravPotential[idx] = gPhi;
                this.elecPotential[idx] = ePhi;
                this.yukawaPotential[idx] = yPhi;
            }
        }

        // Fast tanh approximation — no per-frame ImageData allocation
        const data = this._imgData.data;
        const mode = this.mode;
        const showG = mode === 'all' || mode === 'gravity';
        const showE = mode === 'all' || mode === 'electric';
        const showY = (mode === 'all' || mode === 'yukawa') && doYukawa;

        for (let i = 0; i < GRID_SQ; i++) {
            let gA = 0, eA = 0, yA = 0;
            if (showG) {
                const v = this.gravPotential[i];
                gA = fastTanh((v > 0 ? v : -v) * SENSITIVITY) * MAX_ALPHA;
            }
            if (showE) {
                const v = this.elecPotential[i];
                eA = fastTanh((v > 0 ? v : -v) * SENSITIVITY) * MAX_ALPHA;
            }
            if (showY) {
                const v = this.yukawaPotential[i];
                yA = fastTanh((v > 0 ? v : -v) * SENSITIVITY) * MAX_ALPHA;
            }

            const totalA = gA + eA + yA;
            const idx = i * 4;
            if (totalA > 0) {
                const invA = 1 / totalA;
                // Slate: 138, 126, 114 — gravity
                let r = 138 * gA, g = 126 * gA, b = 114 * gA;
                // Blue-teal (negative charge) / Red-warm (positive charge) — electric
                if (this.elecPotential[i] < 0) {
                    r += 92 * eA; g += 146 * eA; b += 168 * eA;
                } else {
                    r += 192 * eA; g += 80 * eA; b += 72 * eA;
                }
                // Green: 80, 152, 120 — yukawa
                r += 80 * yA; g += 152 * yA; b += 120 * yA;

                data[idx] = (r * invA + 0.5) | 0;
                data[idx + 1] = (g * invA + 0.5) | 0;
                data[idx + 2] = (b * invA + 0.5) | 0;
                data[idx + 3] = totalA < 120 ? (totalA + 0.5) | 0 : 120;
            } else {
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
                data[idx + 3] = 0;
            }
        }
        this.ctx.putImageData(this._imgData, 0, 0);
    }

    draw(ctx, width, height) {
        if (!this.enabled) return;
        ctx.save();
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(this.canvas, 0, 0, width, height);
        ctx.restore();
    }
}
