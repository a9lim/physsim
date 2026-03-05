// ─── Potential Field Heatmap ───
// 48x48 offscreen canvas, diverging colormap, updates every 6 frames.
// When Barnes-Hut is enabled, uses tree walk for O(GRID² log N) instead of O(GRID² N).

import { SOFTENING_SQ, BH_THETA } from './config.js';
import { getDelayedState } from './signal-delay.js';

const GRID_SIZE = 48;
const UPDATE_INTERVAL = 6;
const SENSITIVITY = 3;   // tanh scaling: phi * SENSITIVITY -> 0-1
const MAX_ALPHA = 100;

/**
 * Recursive BH tree walk for scalar potential at point (wx, wy).
 * Returns { gPhi, ePhi } — gravitational and electrostatic potentials.
 */
function treePotential(pool, nodeIdx, wx, wy, theta) {
    if (pool.totalMass[nodeIdx] === 0 && pool.totalCharge[nodeIdx] === 0) return;

    const dx = pool.comX[nodeIdx] - wx;
    const dy = pool.comY[nodeIdx] - wy;
    const dSq = dx * dx + dy * dy;
    const size = pool.bw[nodeIdx] * 2;

    if ((!pool.divided[nodeIdx] && pool.pointCount[nodeIdx] > 0) || (pool.divided[nodeIdx] && (size * size < theta * theta * dSq))) {
        if (!pool.divided[nodeIdx]) {
            // Leaf: sum individual particles
            const base = nodeIdx * pool.nodeCapacity;
            let gP = 0, eP = 0;
            for (let i = 0; i < pool.pointCount[nodeIdx]; i++) {
                const p = pool.points[base + i];
                const pdx = wx - p.pos.x, pdy = wy - p.pos.y;
                const rSq = pdx * pdx + pdy * pdy + SOFTENING_SQ;
                const invR = 1 / Math.sqrt(rSq);
                gP -= p.mass * invR;
                eP += p.charge * invR;
            }
            _treeOut.g += gP;
            _treeOut.e += eP;
        } else {
            // Distant node: use aggregate
            const rSq = dSq + SOFTENING_SQ;
            const invR = 1 / Math.sqrt(rSq);
            _treeOut.g -= pool.totalMass[nodeIdx] * invR;
            _treeOut.e += pool.totalCharge[nodeIdx] * invR;
        }
    } else if (pool.divided[nodeIdx]) {
        treePotential(pool, pool.nw[nodeIdx], wx, wy, theta);
        treePotential(pool, pool.ne[nodeIdx], wx, wy, theta);
        treePotential(pool, pool.sw[nodeIdx], wx, wy, theta);
        treePotential(pool, pool.se[nodeIdx], wx, wy, theta);
    }
}

// Reusable output for treePotential
const _treeOut = { g: 0, e: 0 };

export default class Heatmap {
    constructor() {
        this.enabled = false;
        this.canvas = document.createElement('canvas');
        this.canvas.width = GRID_SIZE;
        this.canvas.height = GRID_SIZE;
        this.ctx = this.canvas.getContext('2d');
        this.frameCount = 0;
        this.gravPotential = new Float32Array(GRID_SIZE * GRID_SIZE);
        this.elecPotential = new Float32Array(GRID_SIZE * GRID_SIZE);
    }

    update(particles, camera, width, height, pool, root, barnesHutEnabled, signalDelayEnabled, relativityEnabled, simTime, periodic, domW, domH, topology) {
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

        for (let gy = 0; gy < GRID_SIZE; gy++) {
            for (let gx = 0; gx < GRID_SIZE; gx++) {
                const wx = left + (gx + 0.5) * cellW;
                const wy = top + (gy + 0.5) * cellH;
                let gPhi = 0, ePhi = 0;

                if (useTree) {
                    _treeOut.g = 0;
                    _treeOut.e = 0;
                    treePotential(pool, root, wx, wy, BH_THETA);
                    gPhi = _treeOut.g;
                    ePhi = _treeOut.e;
                } else {
                    const useDelay = signalDelayEnabled && relativityEnabled;
                    for (let i = 0; i < n; i++) {
                        const p = particles[i];
                        let px, py;
                        if (useDelay && p.histCount >= 2) {
                            const ret = getDelayedState(p, { pos: { x: wx, y: wy } }, simTime, periodic, domW, domH, domW * 0.5, domH * 0.5, topology);
                            if (ret) { px = ret.x; py = ret.y; }
                            else { px = p.pos.x; py = p.pos.y; }
                        } else {
                            px = p.pos.x; py = p.pos.y;
                        }
                        const dx = wx - px, dy = wy - py;
                        const rSq = dx * dx + dy * dy + SOFTENING_SQ;
                        const invR = 1 / Math.sqrt(rSq);
                        gPhi -= p.mass * invR;
                        ePhi += p.charge * invR;
                    }
                }

                const idx = gy * GRID_SIZE + gx;
                this.gravPotential[idx] = gPhi;
                this.elecPotential[idx] = ePhi;
            }
        }

        // Absolute scaling via tanh — no per-frame normalization
        const imgData = this.ctx.createImageData(GRID_SIZE, GRID_SIZE);
        for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            const gInt = Math.tanh(Math.abs(this.gravPotential[i]) * SENSITIVITY);
            const eVal = this.elecPotential[i];
            const eInt = Math.tanh(Math.abs(eVal) * SENSITIVITY);

            const gA = gInt * MAX_ALPHA;
            const eA = eInt * MAX_ALPHA;
            // Slate: 138, 126, 114
            let r = 138 * gA, g = 126 * gA, b = 114 * gA;
            if (eVal < 0) {
                // Blue: 92, 146, 168
                r += 92 * eA; g += 146 * eA; b += 168 * eA;
            } else {
                // Red: 192, 80, 72
                r += 192 * eA; g += 80 * eA; b += 72 * eA;
            }
            const totalA = gA + eA;
            const idx = i * 4;
            if (totalA > 0) {
                imgData.data[idx] = Math.round(r / totalA);
                imgData.data[idx + 1] = Math.round(g / totalA);
                imgData.data[idx + 2] = Math.round(b / totalA);
                imgData.data[idx + 3] = Math.round(Math.min(totalA, 120));
            }
        }
        this.ctx.putImageData(imgData, 0, 0);
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
