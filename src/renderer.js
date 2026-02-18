export default class Renderer {
    constructor(ctx, width, height) {
        this.ctx = ctx;
        this.width = width;
        this.height = height;
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
    }

    render(particles) {
        // Clear background
        this.ctx.fillStyle = 'rgba(15, 23, 42, 0.3)'; // Trail effect
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw particles
        particles.forEach(p => {
            this.ctx.beginPath();

            // Remove glow, just draw core
            this.ctx.arc(p.pos.x, p.pos.y, p.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color;
            this.ctx.fill();

            // Draw velocity vector (optional, maybe debug only)
        });

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
