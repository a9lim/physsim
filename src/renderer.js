import { MAX_TRAIL_LENGTH, PHOTON_LIFETIME, INERTIA_K } from './config.js';

const TWO_PI = Math.PI * 2;
const HALF_PI = Math.PI / 2;
const _PAL = window._PALETTE;
const _r = window._r;

// Per-force component colors (matching toggle colors in styles.css)
const _forceCompColors = {
    gravity:     { light: _PAL.extended.red,    dark: _PAL.extended.red },
    coulomb:     { light: _PAL.extended.blue,   dark: _PAL.extended.blue },
    magnetic:    { light: _PAL.extended.cyan,   dark: _PAL.extended.cyan },
    gravitomag:  { light: _PAL.extended.rose,   dark: _PAL.extended.rose },
    onepn:       { light: _PAL.extended.orange, dark: _PAL.extended.orange },
    onepnem:     { light: _PAL.extended.orange, dark: _PAL.extended.orange },
    spinCurv:    { light: _PAL.extended.purple, dark: _PAL.extended.purple },
    radiation:   { light: _PAL.extended.yellow, dark: _PAL.extended.yellow },
    torqueSO:    { light: _PAL.extended.purple, dark: _PAL.extended.purple },
    torqueFD:    { light: _PAL.extended.rose,   dark: _PAL.extended.rose },
    torqueTidal: { light: _PAL.extended.green,  dark: _PAL.extended.green },
};

// Spin ring colors by sign
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
        this.isLight = false;
        this.trailHistory = new Map();
        this.heatmap = null;
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

        if (this.heatmap) this.heatmap.draw(ctx, this.width, this.height);

        // Camera transform: all subsequent drawing is in world space
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

        ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    updateTrails(particles) {
        const history = this.trailHistory;
        const capacity = MAX_TRAIL_LENGTH * 2;
        const activeIds = new Set();

        for (const p of particles) {
            activeIds.add(p.id);
            let trail = history.get(p.id);
            if (!trail) {
                trail = { data: new Float32Array(capacity), len: 0, start: 0 };
                history.set(p.id, trail);
            }
            if (trail.len < capacity) {
                const writeIdx = (trail.start + trail.len) % capacity;
                trail.data[writeIdx] = p.pos.x;
                trail.data[writeIdx + 1] = p.pos.y;
                trail.len += 2;
            } else {
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

        // Skip segment if position jumps > half domain (periodic wrap)
        const wrapThreshX = this.domainW * 0.5;
        const wrapThreshY = this.domainH * 0.5;

        for (const p of particles) {
            const trail = this.trailHistory.get(p.id);
            if (!trail || trail.len < 4) continue; // need at least 2 points (4 values)

            const pointCount = trail.len / 2;
            const segCount = pointCount - 1;
            const capacity = trail.data.length;
            const lineWidth = 0.5 * p.radius;
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


    drawArrow(ctx, x1, y1, x2, y2, invZoom, color) {
        const dx = x2 - x1, dy = y2 - y1;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len < 0.5 * invZoom) return;

        const nx = dx / len, ny = dy / len;
        const headLen = len < 2 * invZoom ? 0 : 8 * invZoom;

        // Stop shaft at arrowhead base
        const shaftX = x2 - nx * headLen;
        const shaftY = y2 - ny * headLen;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(shaftX, shaftY);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3 * invZoom;
        ctx.stroke();

        if (headLen > 0) {
            ctx.beginPath();
            ctx.moveTo(x2, y2);
            ctx.lineTo(x2 - nx * headLen + ny * headLen * 0.4, y2 - ny * headLen - nx * headLen * 0.4);
            ctx.lineTo(x2 - nx * headLen - ny * headLen * 0.4, y2 - ny * headLen + nx * headLen * 0.4);
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
        }
    }

    drawVelocityVectors(ctx, particles, invZoom, isLight) {
        const scale = 40;
        const color = isLight ? _PAL.light.text : _PAL.dark.text;
        for (const p of particles) {
            const vx = p.vel.x * scale, vy = p.vel.y * scale;
            const mag = Math.sqrt(vx * vx + vy * vy);
            if (mag < 1 * invZoom) continue;
            this.drawArrow(ctx, p.pos.x, p.pos.y, p.pos.x + vx, p.pos.y + vy, invZoom, color);
        }
    }

    drawForceVectors(ctx, particles, invZoom, isLight) {
        const scale = 256;
        const color = isLight ? _PAL.accent : _PAL.accentLight;
        for (const p of particles) {
            // Sum all 8 component vectors (includes Boris display forces)
            const s = scale / p.mass;
            let fx = (p.forceGravity.x + p.forceCoulomb.x + p.forceMagnetic.x + p.forceGravitomag.x + p.force1PN.x + p.force1PNEM.x + p.forceSpinCurv.x + p.forceRadiation.x) * s;
            let fy = (p.forceGravity.y + p.forceCoulomb.y + p.forceMagnetic.y + p.forceGravitomag.y + p.force1PN.y + p.force1PNEM.y + p.forceSpinCurv.y + p.forceRadiation.y) * s;
            const mag = Math.sqrt(fx * fx + fy * fy);
            if (mag < 0.1 * invZoom) continue;
            this.drawArrow(ctx, p.pos.x, p.pos.y, p.pos.x + fx, p.pos.y + fy, invZoom, color);
        }
    }

    drawForceComponentVectors(ctx, particles, invZoom, isLight) {
        const scale = 256;
        const theme = isLight ? 'light' : 'dark';
        const forces = [
            { key: 'forceGravity', color: _forceCompColors.gravity[theme] },
            { key: 'forceCoulomb', color: _forceCompColors.coulomb[theme] },
            { key: 'forceMagnetic', color: _forceCompColors.magnetic[theme] },
            { key: 'forceGravitomag', color: _forceCompColors.gravitomag[theme] },
            { key: 'force1PN', color: _forceCompColors.onepn[theme] },
            { key: 'force1PNEM', color: _forceCompColors.onepnem[theme] },
            { key: 'forceSpinCurv', color: _forceCompColors.spinCurv[theme] },
            { key: 'forceRadiation', color: _forceCompColors.radiation[theme] },
        ];
        for (const { key, color } of forces) {
            for (const p of particles) {
                const s = scale / p.mass;
                let fx = p[key].x * s, fy = p[key].y * s;
                const mag = Math.sqrt(fx * fx + fy * fy);
                if (mag < 0.1 * invZoom) continue;
                this.drawArrow(ctx, p.pos.x, p.pos.y, p.pos.x + fx, p.pos.y + fy, invZoom, color);
            }
        }
    }

    drawTotalTorqueArc(ctx, particles, invZoom, isLight) {
        const color = isLight ? _PAL.accent : _PAL.accentLight;
        this._drawTorqueArc(ctx, particles, invZoom, color, 2.5, (p) => p.torqueSpinOrbit + p.torqueFrameDrag + p.torqueTidal);
    }

    drawTorqueArcs(ctx, particles, invZoom, isLight) {
        const theme = isLight ? 'light' : 'dark';
        this._drawTorqueArc(ctx, particles, invZoom, _forceCompColors.torqueSO[theme], 2, (p) => p.torqueSpinOrbit);
        this._drawTorqueArc(ctx, particles, invZoom, _forceCompColors.torqueFD[theme], 1.5, (p) => p.torqueFrameDrag);
        this._drawTorqueArc(ctx, particles, invZoom, _forceCompColors.torqueTidal[theme], 1, (p) => p.torqueTidal);
    }

    _drawTorqueArc(ctx, particles, invZoom, color, offset, getValue) {
        const scale = 256 / INERTIA_K;
        const maxSweep = Math.PI * 2;
        const threshold = 1e-8;

        ctx.globalCompositeOperation = 'source-over';
        ctx.lineWidth = 3 * invZoom;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;

        for (const p of particles) {
            let val = getValue(p);
            if (Math.abs(val) < threshold) continue;
            val /= INERTIA_K * p.mass * p.radius * p.radius;

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
            const h = 8 * invZoom;
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
        // Arc length proportional to surface speed; caps at full circle
        const arcLen = Math.min(Math.abs(p.angVel) * p.radius * Math.PI * 2, Math.PI * 2);
        const ringRadius = p.radius + 0.5;
        const colors = p.angVel > 0 ? _spinColors.pos : _spinColors.neg;
        const style = isLight ? colors.light : colors.dark;

        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = style;
        ctx.fillStyle = style;
        ctx.lineWidth = 0.2;

        const startAngle = -HALF_PI;
        const endAngle = startAngle - dir * arcLen;
        ctx.beginPath();
        ctx.arc(p.pos.x, p.pos.y, ringRadius, startAngle, endAngle, dir > 0);
        ctx.stroke();

        const ax = p.pos.x + Math.cos(endAngle) * ringRadius;
        const ay = p.pos.y + Math.sin(endAngle) * ringRadius;
        const sweepDir = endAngle - dir * HALF_PI;
        const h = 1;
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
            const size = 0.2 + ph.energy * 20;
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
