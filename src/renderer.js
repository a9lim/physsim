export default class Renderer {
    constructor(ctx, width, height) {
        this.ctx = ctx;
        this.width = width;
        this.height = height;
        this.trails = false;
        this.spinAngle = 0;
        this.trailHistory = new Map(); // particle id -> array of positions
        this.maxTrailLength = 80;
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
    }

    render(particles, dt = 0.016) {
        const ctx = this.ctx;
        const isLight = document.body.classList.contains('light-theme');

        // Advance spin animation
        this.spinAngle += dt * 3;

        // Always clear fully
        ctx.clearRect(0, 0, this.width, this.height);

        // Update trail history
        if (this.trails) {
            const activeIds = new Set();
            for (const p of particles) {
                activeIds.add(p.id);
                let trail = this.trailHistory.get(p.id);
                if (!trail) {
                    trail = [];
                    this.trailHistory.set(p.id, trail);
                }
                trail.push({ x: p.pos.x, y: p.pos.y });
                if (trail.length > this.maxTrailLength) {
                    trail.shift();
                }
            }
            // Clean up trails for removed particles
            for (const id of this.trailHistory.keys()) {
                if (!activeIds.has(id)) {
                    this.trailHistory.delete(id);
                }
            }
        } else {
            this.trailHistory.clear();
        }

        // Draw trails as lines
        if (this.trails) {
            ctx.globalCompositeOperation = isLight ? 'source-over' : 'lighter';
            for (const p of particles) {
                const trail = this.trailHistory.get(p.id);
                if (!trail || trail.length < 2) continue;

                for (let i = 1; i < trail.length; i++) {
                    const alpha = (i / trail.length) * (isLight ? 0.5 : 0.6);
                    ctx.beginPath();
                    ctx.moveTo(trail[i - 1].x, trail[i - 1].y);
                    ctx.lineTo(trail[i].x, trail[i].y);
                    ctx.strokeStyle = p.color;
                    ctx.globalAlpha = alpha;
                    ctx.lineWidth = Math.max(1, p.radius * 0.5);
                    ctx.stroke();
                }
            }
            ctx.globalAlpha = 1.0;
        }

        // Additive blending for particles in dark mode
        ctx.globalCompositeOperation = isLight ? 'source-over' : 'lighter';

        // Draw particles
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.pos.x, p.pos.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = p.color;

            if (!isLight && p.charge !== 0) {
                ctx.shadowBlur = Math.min(Math.abs(p.charge) * 3 + 10, 50);
                ctx.shadowColor = p.color;
            } else if (!isLight) {
                ctx.shadowBlur = 5;
                ctx.shadowColor = 'rgba(255, 255, 255, 0.5)';
            } else {
                ctx.shadowBlur = 0;
            }

            ctx.fill();

            // Spin indicator ring
            if (p.spin !== 0) {
                ctx.shadowBlur = 0;
                const spinDir = Math.sign(p.spin);
                const spinMag = Math.min(Math.abs(p.spin), 50);
                const ringRadius = p.radius + 3 + spinMag * 0.05;
                const arcLen = Math.min(0.4 + spinMag * 0.03, Math.PI * 1.5);
                const baseAngle = this.spinAngle * spinDir * (0.5 + spinMag * 0.05);

                ctx.globalCompositeOperation = 'source-over';
                for (let a = 0; a < 2; a++) {
                    const startAngle = baseAngle + a * Math.PI;
                    ctx.beginPath();
                    ctx.arc(p.pos.x, p.pos.y, ringRadius, startAngle, startAngle + arcLen);
                    const alpha = isLight ? 0.6 : 0.7;
                    const hue = p.spin > 0 ? 160 : 30;
                    ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${alpha})`;
                    ctx.lineWidth = 1.5;
                    ctx.stroke();

                    const endAngle = startAngle + arcLen;
                    const arrowX = p.pos.x + Math.cos(endAngle) * ringRadius;
                    const arrowY = p.pos.y + Math.sin(endAngle) * ringRadius;
                    const arrowAngle = endAngle + (spinDir > 0 ? Math.PI / 2 : -Math.PI / 2);
                    ctx.beginPath();
                    ctx.moveTo(arrowX, arrowY);
                    ctx.lineTo(arrowX + Math.cos(arrowAngle - 0.5) * 4, arrowY + Math.sin(arrowAngle - 0.5) * 4);
                    ctx.lineTo(arrowX + Math.cos(arrowAngle + 0.5) * 4, arrowY + Math.sin(arrowAngle + 0.5) * 4);
                    ctx.closePath();
                    ctx.fillStyle = `hsla(${hue}, 80%, 60%, ${alpha})`;
                    ctx.fill();
                }
                ctx.globalCompositeOperation = isLight ? 'source-over' : 'lighter';
            }
        });

        // Reset state
        ctx.globalCompositeOperation = 'source-over';
        ctx.shadowBlur = 0;

        // Draw drag line
        if (this.input && this.input.isDragging) {
            const start = this.input.dragStart;
            const end = this.input.currentPos;

            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }
}
