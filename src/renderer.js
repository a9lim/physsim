export default class Renderer {
    constructor(ctx, width, height) {
        this.ctx = ctx;
        this.width = width;
        this.height = height;
        this.trails = false;
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
    }

    render(particles) {
        // Clear background
        const isLight = document.body.classList.contains('light-theme');

        if (this.trails) {
            this.ctx.fillStyle = isLight ? 'rgba(241, 245, 249, 0.3)' : 'rgba(10, 17, 40, 0.3)'; // Trail effect based on theme background
            this.ctx.fillRect(0, 0, this.width, this.height);
        } else {
            this.ctx.clearRect(0, 0, this.width, this.height);
        }

        // Use 'lighter' for additive blending (bloom) in dark mode
        this.ctx.globalCompositeOperation = isLight ? 'source-over' : 'lighter';

        // Draw particles
        particles.forEach(p => {
            this.ctx.beginPath();
            this.ctx.arc(p.pos.x, p.pos.y, p.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color;

            // Apply glow effect based on charge intensity
            if (!isLight && p.charge !== 0) {
                this.ctx.shadowBlur = Math.min(Math.abs(p.charge) * 3 + 10, 50); // Cap blur to 50
                this.ctx.shadowColor = p.color;
            } else if (!isLight) {
                this.ctx.shadowBlur = 5;
                this.ctx.shadowColor = 'rgba(255, 255, 255, 0.5)';
            } else {
                this.ctx.shadowBlur = 0;
            }

            this.ctx.fill();
        });

        // Reset state for UI or other drawings
        this.ctx.globalCompositeOperation = 'source-over';
        this.ctx.shadowBlur = 0;

        // Draw drag line if dragging
        if (this.input && this.input.isDragging) {
            const start = this.input.dragStart;
            const end = this.input.currentPos;

            this.ctx.beginPath();
            this.ctx.moveTo(start.x, start.y);
            this.ctx.lineTo(end.x, end.y);
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            this.ctx.setLineDash([5, 5]);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
        }
    }
}
