import { MAX_TRAIL_LENGTH, PHOTON_LIFETIME, HISTORY_SIZE, INERTIA_K } from './config.js';

const TWO_PI = Math.PI * 2;
const HALF_PI = Math.PI / 2;
const _PAL = window._PALETTE;
const _r = window._r;

// Per-force component vector colors (matching force toggle colors)
const _forceCompColors = {
    gravity:     { light: _r(_PAL.extended.red, 0.7),    dark: _r(_PAL.extended.red, 0.8) },
    coulomb:     { light: _r(_PAL.extended.blue, 0.7),   dark: _r(_PAL.extended.blue, 0.8) },
    magnetic:    { light: _r(_PAL.extended.cyan, 0.7),   dark: _r(_PAL.extended.cyan, 0.8) },
    gravitomag:  { light: _r(_PAL.extended.purple, 0.7), dark: _r(_PAL.extended.purple, 0.8) },
    onepn:       { light: _r(_PAL.extended.rose, 0.7),   dark: _r(_PAL.extended.rose, 0.8) },
    spinCurv:    { light: _r(_PAL.extended.orange, 0.7), dark: _r(_PAL.extended.orange, 0.8) },
    radiation:   { light: _r(_PAL.extended.yellow, 0.7), dark: _r(_PAL.extended.yellow, 0.8) },
    torqueSO:    { light: _PAL.extended.orange, dark: _PAL.extended.orange },
    torqueFD:    { light: _PAL.extended.purple, dark: _PAL.extended.purple },
};

// Precomputed spin ring colors: [hue][isLight ? 0 : 1]
const _spinColors = {
    pos: { light: `hsla(${_PAL.spinPos},80%,60%,0.8)`, dark: `hsla(${_PAL.spinPos},80%,60%,0.9)` },
    neg: { light: `hsla(${_PAL.spinNeg},80%,60%,0.8)`, dark: `hsla(${_PAL.spinNeg},80%,60%,0.9)` },
};

export default class Renderer {
    constructor(ctx, width, height) {
        this.ctx = ctx;
        this.width = width;
        this.height = height;
        this.domainW = width;
        this.domainH = height;
        this.trails = true;
        this.showVelocity = false;
        this.showForce = false;
        this.showForceComponents = false;
        this.showSignalDelay = true;
        this.accelScaling = false;
        this.isLight = false;
        this.trailHistory = new Map();
        this.heatmap = null;  // set externally by Simulation
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
    }

    setTheme(isLight) {
        this.isLight = isLight;
    }

    render(particles, dt = 0.016, camera, photons) {
        const ctx = this.ctx;
        const isLight = this.isLight;

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, this.width, this.height);

        // Potential field heatmap (drawn in screen space)
        if (this.heatmap) this.heatmap.draw(ctx, this.width, this.height);

        // Apply camera transform
        if (camera) {
            const z = camera.zoom;
            ctx.setTransform(z, 0, 0, z, this.width / 2 - camera.x * z, this.height / 2 - camera.y * z);
        }

        if (this.trails) {
            this.updateTrails(particles);
            this.drawTrails(ctx, particles, isLight, camera);
        } else if (this.trailHistory.size > 0) {
            this.trailHistory.clear();
        }

        if (this.showSignalDelay) this.drawDelayedPositions(ctx, particles, isLight);
        this.drawParticles(ctx, particles, isLight);
        if (photons && photons.length) this.drawPhotons(ctx, photons, isLight);

        ctx.globalCompositeOperation = 'source-over';
        ctx.shadowBlur = 0;

        const invZoom = 1 / (camera ? camera.zoom : 1);
        if (this.showVelocity) this.drawVelocityVectors(ctx, particles, invZoom, isLight);
        if (this.showForce) {
            this.drawForceVectors(ctx, particles, invZoom, isLight);
            this.drawTotalTorqueArc(ctx, particles, invZoom, isLight);
        }
        if (this.showForceComponents) {
            this.drawForceComponentVectors(ctx, particles, invZoom, isLight);
            this.drawTorqueArcs(ctx, particles, invZoom, isLight);
        }

        // Drag line drawn in world space (dragStart/currentPos are world coords)
        if (this.input && this.input.isDragging) {
            const start = this.input.dragStart;
            const end = this.input.currentPos;
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.strokeStyle = isLight ? _r(_PAL.light.text, 0.4) : _r(_PAL.dark.text, 0.5);
            ctx.lineWidth = 1 / (camera ? camera.zoom : 1);
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Reset transform
        ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    updateTrails(particles) {
        const history = this.trailHistory;
        const capacity = MAX_TRAIL_LENGTH * 2; // flat x,y pairs
        const activeIds = new Set();

        for (const p of particles) {
            activeIds.add(p.id);
            let trail = history.get(p.id);
            if (!trail) {
                trail = { data: new Float32Array(capacity), len: 0, start: 0 };
                history.set(p.id, trail);
            }
            if (trail.len < capacity) {
                // Buffer not full yet — append at (start + len) % capacity
                const writeIdx = (trail.start + trail.len) % capacity;
                trail.data[writeIdx] = p.pos.x;
                trail.data[writeIdx + 1] = p.pos.y;
                trail.len += 2;
            } else {
                // Buffer full — overwrite oldest, advance start
                trail.data[trail.start] = p.pos.x;
                trail.data[trail.start + 1] = p.pos.y;
                trail.start = (trail.start + 2) % capacity;
            }
        }

        for (const id of history.keys()) {
            if (!activeIds.has(id)) {
                history.delete(id);
            }
        }
    }

    drawTrails(ctx, particles, isLight, camera) {
        const alphaMax = isLight ? 0.7 : 0.9;
        ctx.globalCompositeOperation = isLight ? 'source-over' : 'lighter';

        // Wrap-detection threshold: half the fixed domain
        const wrapThreshX = this.domainW * 0.5;
        const wrapThreshY = this.domainH * 0.5;

        for (const p of particles) {
            const trail = this.trailHistory.get(p.id);
            if (!trail || trail.len < 4) continue; // need at least 2 points (4 values)

            const pointCount = trail.len / 2;
            const segCount = pointCount - 1;
            const capacity = trail.data.length;
            const lineWidth = Math.max(1.5, p.radius * 0.6);
            ctx.strokeStyle = p.color;
            ctx.lineWidth = lineWidth;

            const groupCount = 4;
            for (let g = 0; g < groupCount; g++) {
                const segStart = Math.floor(g * segCount / groupCount);
                const segEnd = Math.floor((g + 1) * segCount / groupCount);
                if (segEnd <= segStart) continue;

                const midSeg = (segStart + segEnd) / 2;
                ctx.globalAlpha = ((midSeg + 1) / (segCount + 1)) * alphaMax;
                ctx.beginPath();
                const i0 = (trail.start + segStart * 2) % capacity;
                let prevX = trail.data[i0], prevY = trail.data[i0 + 1];
                ctx.moveTo(prevX, prevY);
                for (let s = segStart + 1; s <= segEnd; s++) {
                    const i = (trail.start + s * 2) % capacity;
                    const x = trail.data[i], y = trail.data[i + 1];
                    if (Math.abs(x - prevX) > wrapThreshX || Math.abs(y - prevY) > wrapThreshY) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                    prevX = x;
                    prevY = y;
                }
                ctx.stroke();
            }
        }
        ctx.globalAlpha = 1.0;
    }

    drawParticles(ctx, particles, isLight) {
        const blendMode = isLight ? 'source-over' : 'lighter';
        ctx.globalCompositeOperation = blendMode;

        for (const p of particles) {
            ctx.beginPath();
            ctx.arc(p.pos.x, p.pos.y, p.radius, 0, TWO_PI);
            ctx.fillStyle = p.color;

            if (!isLight) {
                if (p.charge !== 0) {
                    ctx.shadowBlur = Math.min(Math.abs(p.charge) * 3 + 10, 50);
                    ctx.shadowColor = p.color;
                } else {
                    ctx.shadowBlur = 5;
                    ctx.shadowColor = _r(_PAL.dark.text, 0.5);
                }
            } else {
                ctx.shadowBlur = 0;
            }

            ctx.fill();

            if (p.angVel !== 0) {
                this.drawSpinRing(ctx, p, isLight, blendMode);
            }
        }
    }

    drawDelayedPositions(ctx, particles, isLight) {
        // Draw ghost circles at each particle's most recent delayed position
        // (using its own history to show where it "was" when the force was emitted)
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 0.3;
        for (const p of particles) {
            if (p.histCount < 2) continue;
            // Show oldest recorded position as ghost
            const oldest = (p.histHead - p.histCount + HISTORY_SIZE) % HISTORY_SIZE;
            const gx = p.histX[oldest], gy = p.histY[oldest];

            // Ghost circle (filled, transparency via globalAlpha)
            ctx.beginPath();
            ctx.arc(gx, gy, p.radius, 0, TWO_PI);
            ctx.fillStyle = p.color;
            ctx.fill();

            // Connecting line from ghost to current
            ctx.beginPath();
            ctx.moveTo(gx, gy);
            ctx.lineTo(p.pos.x, p.pos.y);
            ctx.strokeStyle = isLight ? _r(_PAL.light.textMuted, 0.3) : _r(_PAL.dark.textMuted, 0.3);
            ctx.lineWidth = 0.5;
            ctx.setLineDash([2, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
        ctx.globalAlpha = 1.0;
    }

    drawArrow(ctx, x1, y1, x2, y2, invZoom, color) {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2 * invZoom;
        ctx.stroke();

        // Arrowhead
        const dx = x2 - x1, dy = y2 - y1;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len < 2 * invZoom) return;
        const nx = dx / len, ny = dy / len;
        const headLen = 6 * invZoom;
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - nx * headLen + ny * headLen * 0.4, y2 - ny * headLen - nx * headLen * 0.4);
        ctx.lineTo(x2 - nx * headLen - ny * headLen * 0.4, y2 - ny * headLen + nx * headLen * 0.4);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
    }

    drawVelocityVectors(ctx, particles, invZoom, isLight) {
        const scale = 40;
        const color = isLight ? _r(_PAL.light.text, 0.5) : _r(_PAL.dark.text, 0.6);
        for (const p of particles) {
            const vx = p.vel.x * scale, vy = p.vel.y * scale;
            const mag = Math.sqrt(vx * vx + vy * vy);
            if (mag < 1 * invZoom) continue;
            this.drawArrow(ctx, p.pos.x, p.pos.y, p.pos.x + vx, p.pos.y + vy, invZoom, color);
        }
    }

    drawForceVectors(ctx, particles, invZoom, isLight) {
        const scale = 5;
        const color = isLight ? _r(_PAL.accent, 0.7) : _r(_PAL.accentLight, 0.8);
        for (const p of particles) {
            // Sum all component vectors for total force (p.force only has E-like;
            // component vectors include both E-like and Boris display forces)
            const s = this.accelScaling ? scale * 100 / p.mass : scale;
            const fx = (p.forceGravity.x + p.forceCoulomb.x + p.forceMagnetic.x + p.forceGravitomag.x + p.force1PN.x + p.forceSpinCurv.x + p.forceRadiation.x) * s;
            const fy = (p.forceGravity.y + p.forceCoulomb.y + p.forceMagnetic.y + p.forceGravitomag.y + p.force1PN.y + p.forceSpinCurv.y + p.forceRadiation.y) * s;
            const mag = Math.sqrt(fx * fx + fy * fy);
            if (mag < 1 * invZoom) continue;
            this.drawArrow(ctx, p.pos.x, p.pos.y, p.pos.x + fx, p.pos.y + fy, invZoom, color);
        }
    }

    drawForceComponentVectors(ctx, particles, invZoom, isLight) {
        const scale = 5;
        const theme = isLight ? 'light' : 'dark';
        const forces = [
            { key: 'forceGravity', color: _forceCompColors.gravity[theme] },
            { key: 'forceCoulomb', color: _forceCompColors.coulomb[theme] },
            { key: 'forceMagnetic', color: _forceCompColors.magnetic[theme] },
            { key: 'forceGravitomag', color: _forceCompColors.gravitomag[theme] },
            { key: 'force1PN', color: _forceCompColors.onepn[theme] },
            { key: 'forceSpinCurv', color: _forceCompColors.spinCurv[theme] },
            { key: 'forceRadiation', color: _forceCompColors.radiation[theme] },
        ];
        for (const { key, color } of forces) {
            for (const p of particles) {
                const s = this.accelScaling ? scale * 100 / p.mass : scale;
                const fx = p[key].x * s, fy = p[key].y * s;
                const mag = Math.sqrt(fx * fx + fy * fy);
                if (mag < 1 * invZoom) continue;
                this.drawArrow(ctx, p.pos.x, p.pos.y, p.pos.x + fx, p.pos.y + fy, invZoom, color);
            }
        }
    }

    drawTotalTorqueArc(ctx, particles, invZoom, isLight) {
        const color = isLight ? _PAL.accent : _PAL.accentLight;
        this._drawTorqueArc(ctx, particles, invZoom, color, 9, (p) => p.torqueSpinOrbit + p.torqueFrameDrag);
    }

    drawTorqueArcs(ctx, particles, invZoom, isLight) {
        const theme = isLight ? 'light' : 'dark';
        this._drawTorqueArc(ctx, particles, invZoom, _forceCompColors.torqueSO[theme], 7, (p) => p.torqueSpinOrbit);
        this._drawTorqueArc(ctx, particles, invZoom, _forceCompColors.torqueFD[theme], 5, (p) => p.torqueFrameDrag);
    }

    _drawTorqueArc(ctx, particles, invZoom, color, offset, getValue) {
        const scale = 0.3;
        const maxSweep = Math.PI * 1.5;
        const threshold = 1e-8;

        ctx.globalCompositeOperation = 'source-over';
        ctx.lineWidth = 2 * invZoom;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;

        for (const p of particles) {
            let val = getValue(p);
            if (Math.abs(val) < threshold) continue;
            if (this.accelScaling) val *= 100 / (INERTIA_K * p.mass * p.radius * p.radius);

            const ringRadius = p.radius + offset;
            const sweep = Math.min(scale * Math.abs(val), maxSweep);
            const dir = val > 0 ? 1 : -1;
            const startAngle = -HALF_PI;
            const endAngle = startAngle - dir * sweep;

            ctx.beginPath();
            ctx.arc(p.pos.x, p.pos.y, ringRadius, startAngle, endAngle, dir > 0);
            ctx.stroke();

            const ax = p.pos.x + Math.cos(endAngle) * ringRadius;
            const ay = p.pos.y + Math.sin(endAngle) * ringRadius;
            const sweepDir = endAngle - dir * HALF_PI;
            const h = 6 * invZoom;
            const tipX = ax + Math.cos(sweepDir) * h;
            const tipY = ay + Math.sin(sweepDir) * h;
            const spread = h * 0.4;
            ctx.beginPath();
            ctx.moveTo(tipX, tipY);
            ctx.lineTo(ax + Math.cos(endAngle) * spread, ay + Math.sin(endAngle) * spread);
            ctx.lineTo(ax - Math.cos(endAngle) * spread, ay - Math.sin(endAngle) * spread);
            ctx.closePath();
            ctx.fill();
        }
    }

    drawSpinRing(ctx, p, isLight, blendMode) {
        ctx.shadowBlur = 0;
        const dir = Math.sign(p.angVel);
        // Surface velocity |ω·r| < c=1, so this naturally caps at 2π
        const arcLen = Math.min(Math.abs(p.angVel) * p.radius * Math.PI * 2, Math.PI * 2);
        const ringRadius = p.radius + 2;
        const colors = p.angVel > 0 ? _spinColors.pos : _spinColors.neg;
        const style = isLight ? colors.light : colors.dark;

        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = style;
        ctx.fillStyle = style;
        ctx.lineWidth = 1;

        const startAngle = -HALF_PI;
        const endAngle = startAngle - dir * arcLen;
        ctx.beginPath();
        ctx.arc(p.pos.x, p.pos.y, ringRadius, startAngle, endAngle, dir > 0);
        ctx.stroke();

        const ax = p.pos.x + Math.cos(endAngle) * ringRadius;
        const ay = p.pos.y + Math.sin(endAngle) * ringRadius;
        const sweepDir = endAngle - dir * HALF_PI;
        const h = 3;
        const tipX = ax + Math.cos(sweepDir) * h;
        const tipY = ay + Math.sin(sweepDir) * h;
        const spread = h * 0.4;
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(ax + Math.cos(endAngle) * spread, ay + Math.sin(endAngle) * spread);
        ctx.lineTo(ax - Math.cos(endAngle) * spread, ay - Math.sin(endAngle) * spread);
        ctx.closePath();
        ctx.fill();

        ctx.globalCompositeOperation = blendMode;
    }

    drawPhotons(ctx, photons, isLight) {
        if (!photons || !photons.length) return;
        ctx.globalCompositeOperation = isLight ? 'source-over' : 'lighter';
        ctx.shadowBlur = 0;
        for (const ph of photons) {
            const alpha = 1 - ph.lifetime / PHOTON_LIFETIME;
            if (alpha <= 0) continue;
            const size = 1.5 + ph.energy * 20;
            ctx.globalAlpha = alpha * (isLight ? 0.6 : 0.8);
            ctx.fillStyle = _PAL.extended.yellow;
            ctx.beginPath();
            ctx.arc(ph.pos.x, ph.pos.y, Math.min(size, 5), 0, TWO_PI);
            ctx.fill();
            if (!isLight) {
                ctx.shadowBlur = Math.min(size * 3, 15);
                ctx.shadowColor = '#FFDC6480';
                ctx.fill();
                ctx.shadowBlur = 0;
            }
        }
        ctx.globalAlpha = 1;
    }
}
