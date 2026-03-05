// ─── Potential Field Heatmap ───
// 48x48 offscreen canvas, diverging colormap, updates every 6 frames.

import { SOFTENING_SQ } from './config.js';

const GRID_SIZE = 48;
const UPDATE_INTERVAL = 6;
const SENSITIVITY = 3;   // tanh scaling: phi * SENSITIVITY -> 0-1
const MAX_ALPHA = 100;

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

    update(particles, camera, width, height) {
        if (!this.enabled) return;
        if (++this.frameCount % UPDATE_INTERVAL !== 0) return;

        const zoom = camera.zoom;
        const cx = camera.x, cy = camera.y;
        const halfW = width / (2 * zoom), halfH = height / (2 * zoom);
        const left = cx - halfW, top = cy - halfH;
        const cellW = (2 * halfW) / GRID_SIZE;
        const cellH = (2 * halfH) / GRID_SIZE;

        for (let gy = 0; gy < GRID_SIZE; gy++) {
            for (let gx = 0; gx < GRID_SIZE; gx++) {
                const wx = left + (gx + 0.5) * cellW;
                const wy = top + (gy + 0.5) * cellH;
                let gPhi = 0, ePhi = 0;

                for (let i = 0; i < particles.length; i++) {
                    const p = particles[i];
                    const dx = wx - p.pos.x, dy = wy - p.pos.y;
                    const rSq = dx * dx + dy * dy + SOFTENING_SQ;
                    const invR = 1 / Math.sqrt(rSq);
                    gPhi -= p.mass * invR;   // always negative
                    ePhi += p.charge * invR;  // sign depends on charge
                }

                const idx = gy * GRID_SIZE + gx;
                this.gravPotential[idx] = gPhi;
                this.elecPotential[idx] = ePhi;
            }
        }

        // Absolute scaling via tanh — no per-frame normalization
        // Slate #8A7E72 for gravity, Blue #5C92A8 for negative electric, Red #C05048 for positive electric
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
