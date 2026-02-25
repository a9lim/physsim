const TWO_PI = Math.PI * 2;
const HALF_PI = Math.PI / 2;

export default class Renderer {
    constructor(ctx, width, height) {
        this.ctx = ctx;
        this.width = width;
        this.height = height;
        this.trails = false;
        this.isLight = false;
        this.spinAngle = 0;
        this.trailHistory = new Map();
        this.maxTrailLength = 200;
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
    }

    setTheme(isLight) {
        this.isLight = isLight;
    }

    render(particles, dt = 0.016) {
        const ctx = this.ctx;
        const isLight = this.isLight;

        this.spinAngle += dt * 3;

        ctx.clearRect(0, 0, this.width, this.height);

        if (this.trails) {
            this.updateTrails(particles);
            this.drawTrails(ctx, particles, isLight);
        } else if (this.trailHistory.size > 0) {
            this.trailHistory.clear();
        }

        this.drawParticles(ctx, particles, isLight);

        ctx.globalCompositeOperation = 'source-over';
        ctx.shadowBlur = 0;

        if (this.input && this.input.isDragging) {
            const start = this.input.dragStart;
            const end = this.input.currentPos;
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.strokeStyle = isLight ? 'rgba(0, 0, 0, 0.4)' : 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    updateTrails(particles) {
        const history = this.trailHistory;
        const maxLen = this.maxTrailLength;
        const activeIds = new Set();

        for (const p of particles) {
            activeIds.add(p.id);
            let trail = history.get(p.id);
            if (!trail) {
                trail = [];
                history.set(p.id, trail);
            }
            trail.push(p.pos.x, p.pos.y); // flat x,y pairs — half the object overhead
            if (trail.length > maxLen * 2) {
                trail.splice(0, 2);
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
            if (!trail || trail.length < 4) continue; // need at least 2 points (4 values)

            const segCount = (trail.length / 2) - 1;
            const lineWidth = Math.max(1.5, p.radius * 0.6);
            ctx.strokeStyle = p.color;
            ctx.lineWidth = lineWidth;

            // Batch into fewer alpha groups (4 groups) to reduce state changes
            const groupCount = 4;
            for (let g = 0; g < groupCount; g++) {
                const segStart = Math.floor(g * segCount / groupCount);
                const segEnd = Math.floor((g + 1) * segCount / groupCount);
                if (segEnd <= segStart) continue;

                const midSeg = (segStart + segEnd) / 2;
                ctx.globalAlpha = ((midSeg + 1) / (segCount + 1)) * alphaMax;
                ctx.beginPath();
                const i0 = segStart * 2;
                ctx.moveTo(trail[i0], trail[i0 + 1]);
                for (let s = segStart + 1; s <= segEnd; s++) {
                    const i = s * 2;
                    ctx.lineTo(trail[i], trail[i + 1]);
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
                    ctx.shadowColor = 'rgba(255, 255, 255, 0.5)';
                }
            } else {
                ctx.shadowBlur = 0;
            }

            ctx.fill();

            if (p.spin !== 0) {
                this.drawSpinRing(ctx, p, isLight, blendMode);
            }
        }
    }

    drawSpinRing(ctx, p, isLight, blendMode) {
        ctx.shadowBlur = 0;
        const spinDir = Math.sign(p.spin);
        const spinMag = Math.min(Math.abs(p.spin), 50);
        const ringRadius = p.radius + 3 + spinMag * 0.05;
        const arcLen = Math.min(0.4 + spinMag * 0.03, Math.PI * 1.5);
        const baseAngle = this.spinAngle * spinDir * (0.5 + spinMag * 0.05);
        const alpha = isLight ? 0.6 : 0.7;
        const hue = p.spin > 0 ? 160 : 30;
        const style = `hsla(${hue},80%,60%,${alpha})`;

        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = style;
        ctx.fillStyle = style;
        ctx.lineWidth = 1.5;

        for (let a = 0; a < 2; a++) {
            const startAngle = baseAngle + a * Math.PI;
            ctx.beginPath();
            ctx.arc(p.pos.x, p.pos.y, ringRadius, startAngle, startAngle + arcLen);
            ctx.stroke();

            const endAngle = startAngle + arcLen;
            const arrowX = p.pos.x + Math.cos(endAngle) * ringRadius;
            const arrowY = p.pos.y + Math.sin(endAngle) * ringRadius;
            const arrowAngle = endAngle + (spinDir > 0 ? HALF_PI : -HALF_PI);
            ctx.beginPath();
            ctx.moveTo(arrowX, arrowY);
            ctx.lineTo(arrowX + Math.cos(arrowAngle - 0.5) * 4, arrowY + Math.sin(arrowAngle - 0.5) * 4);
            ctx.lineTo(arrowX + Math.cos(arrowAngle + 0.5) * 4, arrowY + Math.sin(arrowAngle + 0.5) * 4);
            ctx.closePath();
            ctx.fill();
        }
        ctx.globalCompositeOperation = blendMode;
    }
}
