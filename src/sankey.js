// ─── Energy Flow Sankey Overlay ───

export default class SankeyOverlay {
    constructor() {
        this.enabled = false;
        this.prevEnergy = { linearKE: 0, spinKE: 0, pe: 0, fieldE: 0, radiated: 0 };
        this.flows = { keToPe: 0, peToKe: 0, keToRad: 0, spinToOrbit: 0 };
        this.smoothing = 0.92;
        this.initialized = false;
    }

    update(linearKE, spinKE, pe, fieldE, radiated) {
        if (!this.enabled) return;
        if (!this.initialized) {
            this.prevEnergy = { linearKE, spinKE, pe, fieldE, radiated };
            this.initialized = true;
            return;
        }

        const dKE = linearKE - this.prevEnergy.linearKE;
        const dPE = pe - this.prevEnergy.pe;
        const dRad = radiated - this.prevEnergy.radiated;
        const dSpin = spinKE - this.prevEnergy.spinKE;

        const s = this.smoothing;
        this.flows.keToPe = s * this.flows.keToPe + (1 - s) * Math.max(0, dPE);
        this.flows.peToKe = s * this.flows.peToKe + (1 - s) * Math.max(0, -dPE);
        this.flows.keToRad = s * this.flows.keToRad + (1 - s) * Math.max(0, dRad);
        this.flows.spinToOrbit = s * this.flows.spinToOrbit + (1 - s) * Math.max(0, -dSpin);

        this.prevEnergy = { linearKE, spinKE, pe, fieldE, radiated };
    }

    draw(ctx, width, height, isLight) {
        if (!this.enabled) return;

        const x = width - 230, y = 12, w = 210, h = 150;

        ctx.save();
        ctx.fillStyle = isLight ? '#FCF7F2CC' : '#0C0B09CC';
        ctx.strokeStyle = isLight ? '#1A161222' : '#E8DED422';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 8);
        ctx.fill();
        ctx.stroke();

        const nodes = {
            KE:    { x: x + 35,  y: y + 40, color: '#CC8E4E' },
            PE:    { x: x + 105, y: y + 40, color: '#5C92A8' },
            Rad:   { x: x + 175, y: y + 40, color: '#CCA84C' },
            Spin:  { x: x + 35,  y: y + 100, color: '#9C7EB0' },
            Field: { x: x + 105, y: y + 100, color: '#4AACA0' },
        };

        // Draw nodes
        const textColor = isLight ? '#1A1612' : '#E8DED4';
        ctx.font = '9px Geist, sans-serif';
        ctx.textAlign = 'center';
        for (const [name, n] of Object.entries(nodes)) {
            ctx.fillStyle = n.color;
            ctx.beginPath();
            ctx.arc(n.x, n.y, 12, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = textColor;
            ctx.fillText(name, n.x, n.y + 24);
        }

        // Draw flow arrows
        const maxFlow = Math.max(
            this.flows.keToRad, this.flows.keToPe, this.flows.peToKe, this.flows.spinToOrbit, 0.001
        );

        this._drawFlow(ctx, nodes.KE, nodes.PE, this.flows.keToPe / maxFlow, '#CC8E4E88');
        this._drawFlow(ctx, nodes.PE, nodes.KE, this.flows.peToKe / maxFlow, '#5C92A888');
        this._drawFlow(ctx, nodes.KE, nodes.Rad, this.flows.keToRad / maxFlow, '#CCA84C88');
        this._drawFlow(ctx, nodes.Spin, nodes.KE, this.flows.spinToOrbit / maxFlow, '#9C7EB088');

        // Title
        ctx.fillStyle = textColor;
        ctx.font = '10px Geist, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Energy Flow', x + 8, y + 14);

        ctx.restore();
    }

    _drawFlow(ctx, from, to, magnitude, color) {
        if (magnitude < 0.01) return;
        const lineWidth = 1 + magnitude * 6;
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();

        // Arrowhead
        const dx = to.x - from.x, dy = to.y - from.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len < 1) return;
        const nx = dx / len, ny = dy / len;
        const ax = to.x - nx * 14, ay = to.y - ny * 14;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(to.x - nx * 2, to.y - ny * 2);
        ctx.lineTo(ax - ny * 4, ay + nx * 4);
        ctx.lineTo(ax + ny * 4, ay - nx * 4);
        ctx.fill();
    }
}
