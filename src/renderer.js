import { MAX_TRAIL_LENGTH } from './config.js';

const TWO_PI = Math.PI * 2;
const HALF_PI = Math.PI / 2;
const _PAL = window._PALETTE;
const _r = window._r;

// Per-force component vector colors (matching force toggle colors)
const _forceCompColors = {
    gravity:     { light: _r(_PAL.extended.slate, 0.7),  dark: _r(_PAL.extended.slate, 0.8) },
    coulomb:     { light: _r(_PAL.extended.blue, 0.7),   dark: _r(_PAL.extended.blue, 0.8) },
    magnetic:    { light: _r(_PAL.extended.cyan, 0.7),   dark: _r(_PAL.extended.cyan, 0.8) },
    gravitomag:  { light: _r(_PAL.extended.purple, 0.7), dark: _r(_PAL.extended.purple, 0.8) },
};

// Precomputed spin ring colors: [hue][isLight ? 0 : 1]
const _spinColors = {
    pos: { light: `hsla(${_PAL.spinPos},80%,60%,0.6)`, dark: `hsla(${_PAL.spinPos},80%,60%,0.7)` },
    neg: { light: `hsla(${_PAL.spinNeg},80%,60%,0.6)`, dark: `hsla(${_PAL.spinNeg},80%,60%,0.7)` },
};

export default class Renderer {
    constructor(ctx, width, height) {
        this.ctx = ctx;
        this.width = width;
        this.height = height;
        this.trails = false;
        this.showVelocity = false;
        this.showForce = false;
        this.showForceComponents = false;
        this.isLight = false;
        this.trailHistory = new Map();
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
    }

    setTheme(isLight) {
        this.isLight = isLight;
    }

    render(particles, dt = 0.016, camera) {
        const ctx = this.ctx;
        const isLight = this.isLight;

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, this.width, this.height);

        // Apply camera transform
        if (camera) {
            const z = camera.zoom;
            ctx.setTransform(z, 0, 0, z, this.width / 2 - camera.x * z, this.height / 2 - camera.y * z);
        }

        if (this.trails) {
            this.updateTrails(particles);
            this.drawTrails(ctx, particles, isLight);
        } else if (this.trailHistory.size > 0) {
            this.trailHistory.clear();
        }

        this.drawParticles(ctx, particles, isLight);

        ctx.globalCompositeOperation = 'source-over';
        ctx.shadowBlur = 0;

        const invZoom = 1 / (camera ? camera.zoom : 1);
        if (this.showVelocity) this.drawVelocityVectors(ctx, particles, invZoom, isLight);
        if (this.showForce) {
            this.drawForceVectors(ctx, particles, invZoom, isLight);
            this.drawTorqueArcs(ctx, particles, invZoom, isLight);
        }
        if (this.showForceComponents) {
            this.drawForceComponentVectors(ctx, particles, invZoom, isLight);
            this.drawTorqueComponentArcs(ctx, particles, invZoom, isLight);
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

    drawTrails(ctx, particles, isLight) {
        const alphaMax = isLight ? 0.7 : 0.9;
        ctx.globalCompositeOperation = isLight ? 'source-over' : 'lighter';

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
                ctx.moveTo(trail.data[i0], trail.data[i0 + 1]);
                for (let s = segStart + 1; s <= segEnd; s++) {
                    const i = (trail.start + s * 2) % capacity;
                    ctx.lineTo(trail.data[i], trail.data[i + 1]);
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
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5 * invZoom;
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
            const fx = (p.forceGravity.x + p.forceCoulomb.x + p.forceMagnetic.x + p.forceGravitomag.x) * scale;
            const fy = (p.forceGravity.y + p.forceCoulomb.y + p.forceMagnetic.y + p.forceGravitomag.y) * scale;
            const mag = Math.sqrt(fx * fx + fy * fy);
            if (mag < 1 * invZoom) continue;
            this.drawArrow(ctx, p.pos.x, p.pos.y, p.pos.x + fx, p.pos.y + fy, invZoom, color);
        }
    }

    _drawTorqueArc(ctx, p, torque, invZoom, color, angleOffset) {
        const mag = Math.abs(torque);
        if (mag < 0.01) return;

        const dir = Math.sign(torque);
        const radius = p.radius + 8;
        const arcLen = Math.min(mag * 0.4, Math.PI * 2);
        const startAngle = -HALF_PI + angleOffset;
        const endAngle = startAngle - dir * arcLen;

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5 * invZoom;
        ctx.beginPath();
        ctx.arc(p.pos.x, p.pos.y, radius, startAngle, endAngle, dir > 0);
        ctx.stroke();

        // Arrowhead extending past arc end in sweep direction
        const ax = p.pos.x + Math.cos(endAngle) * radius;
        const ay = p.pos.y + Math.sin(endAngle) * radius;
        const sweepDir = endAngle - dir * HALF_PI;
        const h = 4 * invZoom;
        const tipX = ax + Math.cos(sweepDir) * h;
        const tipY = ay + Math.sin(sweepDir) * h;
        const spread = h * 0.4;
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(ax + Math.cos(endAngle) * spread, ay + Math.sin(endAngle) * spread);
        ctx.lineTo(ax - Math.cos(endAngle) * spread, ay - Math.sin(endAngle) * spread);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
    }

    drawTorqueArcs(ctx, particles, invZoom, isLight) {
        const color = isLight ? _r(_PAL.accent, 0.7) : _r(_PAL.accentLight, 0.8);
        for (const p of particles) {
            this._drawTorqueArc(ctx, p, p.torque, invZoom, color, 0);
        }
    }

    drawTorqueComponentArcs(ctx, particles, invZoom, isLight) {
        const theme = isLight ? 'light' : 'dark';
        const magColor = _forceCompColors.magnetic[theme];
        const gmColor = _forceCompColors.gravitomag[theme];
        for (const p of particles) {
            this._drawTorqueArc(ctx, p, p.torqueMagnetic, invZoom, magColor, 0);
            this._drawTorqueArc(ctx, p, p.torqueGravitomag, invZoom, gmColor, Math.PI);
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
        ];
        for (const { key, color } of forces) {
            for (const p of particles) {
                const fx = p[key].x * scale, fy = p[key].y * scale;
                const mag = Math.sqrt(fx * fx + fy * fy);
                if (mag < 1 * invZoom) continue;
                this.drawArrow(ctx, p.pos.x, p.pos.y, p.pos.x + fx, p.pos.y + fy, invZoom, color);
            }
        }
    }

    drawSpinRing(ctx, p, isLight, blendMode) {
        ctx.shadowBlur = 0;
        const dir = Math.sign(p.angVel);
        // Surface velocity |ω·r| < c=1, so this naturally caps at 2π
        const arcLen = Math.min(Math.abs(p.angVel) * p.radius * Math.PI * 2, Math.PI * 2);
        const ringRadius = p.radius + 3;
        const colors = p.angVel > 0 ? _spinColors.pos : _spinColors.neg;
        const style = isLight ? colors.light : colors.dark;

        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = style;
        ctx.fillStyle = style;
        ctx.lineWidth = 1.5;

        const startAngle = -HALF_PI;
        const endAngle = startAngle - dir * arcLen;
        ctx.beginPath();
        ctx.arc(p.pos.x, p.pos.y, ringRadius, startAngle, endAngle, dir > 0);
        ctx.stroke();

        const ax = p.pos.x + Math.cos(endAngle) * ringRadius;
        const ay = p.pos.y + Math.sin(endAngle) * ringRadius;
        const sweepDir = endAngle - dir * HALF_PI;
        const h = 4;
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
}
