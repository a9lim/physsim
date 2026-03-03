import { SOFTENING_SQ } from './config.js';

const GRID_SIZE = 48;
const UPDATE_INTERVAL = 6; // update every 6 frames

export default class Heatmap {
    constructor() {
        this.enabled = false;
        this.canvas = document.createElement('canvas');
        this.canvas.width = GRID_SIZE;
        this.canvas.height = GRID_SIZE;
        this.ctx = this.canvas.getContext('2d');
        this.frameCount = 0;
        this.potential = new Float32Array(GRID_SIZE * GRID_SIZE);
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

        let minPhi = 0, maxPhi = 0;

        for (let gy = 0; gy < GRID_SIZE; gy++) {
            for (let gx = 0; gx < GRID_SIZE; gx++) {
                const wx = left + (gx + 0.5) * cellW;
                const wy = top + (gy + 0.5) * cellH;
                let phi = 0;

                for (let i = 0; i < particles.length; i++) {
                    const p = particles[i];
                    const dx = wx - p.pos.x, dy = wy - p.pos.y;
                    const rSq = dx * dx + dy * dy + SOFTENING_SQ;
                    const invR = 1 / Math.sqrt(rSq);
                    phi -= p.mass * invR;        // gravitational (attractive = negative)
                    phi += p.charge * invR;      // electrostatic (repulsive same-sign = positive)
                }

                this.potential[gy * GRID_SIZE + gx] = phi;
                if (phi < minPhi) minPhi = phi;
                if (phi > maxPhi) maxPhi = phi;
            }
        }

        // Render to offscreen canvas with diverging colormap
        const imgData = this.ctx.createImageData(GRID_SIZE, GRID_SIZE);
        const range = Math.max(Math.abs(minPhi), Math.abs(maxPhi)) || 1;

        for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            const norm = this.potential[i] / range; // -1 to 1
            const idx = i * 4;
            if (norm < 0) {
                // Negative (gravity well) → blue
                imgData.data[idx] = 40;
                imgData.data[idx + 1] = 80;
                imgData.data[idx + 2] = 200;
                imgData.data[idx + 3] = Math.round(Math.abs(norm) * 80);
            } else {
                // Positive (repulsive) → red
                imgData.data[idx] = 200;
                imgData.data[idx + 1] = 60;
                imgData.data[idx + 2] = 40;
                imgData.data[idx + 3] = Math.round(norm * 80);
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
