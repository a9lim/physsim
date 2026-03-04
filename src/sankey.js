// ─── Energy Bar Chart ───

const BAR_H = 14;
const GAP = 6;
const LABEL_W = 56;
const PAD = 8;

const CATEGORIES = [
    { key: 'linearKE', label: 'Linear KE', color: '#CC8E4E' },
    { key: 'spinKE',   label: 'Spin KE',   color: '#9C7EB0' },
    { key: 'pe',       label: 'Potential',  color: '#5C92A8' },
    { key: 'fieldE',   label: 'Field',      color: '#4AACA0' },
    { key: 'radiated', label: 'Radiated',   color: '#CCA84C' },
];

const CSS_H = PAD * 2 + CATEGORIES.length * (BAR_H + GAP) - GAP;

export default class SankeyOverlay {
    constructor() {
        this.enabled = true;
        this.canvas = document.createElement('canvas');
        this.canvas.style.height = CSS_H + 'px';
        this.ctx = this.canvas.getContext('2d');
        this.values = { linearKE: 0, spinKE: 0, pe: 0, fieldE: 0, radiated: 0 };
    }

    update(linearKE, spinKE, pe, fieldE, radiated) {
        if (!this.enabled) return;
        this.values = { linearKE, spinKE, pe, fieldE, radiated };
    }

    draw(isLight) {
        if (!this.enabled) return;

        const dpr = devicePixelRatio || 1;
        const cssW = this.canvas.clientWidth || 280;
        const cssH = CSS_H;
        const pxW = Math.round(cssW * dpr);
        const pxH = Math.round(cssH * dpr);
        if (this.canvas.width !== pxW || this.canvas.height !== pxH) {
            this.canvas.width = pxW;
            this.canvas.height = pxH;
        }

        const ctx = this.ctx;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cssW, cssH);

        const barW = cssW - LABEL_W - PAD * 2;

        // Find max absolute value for scaling
        let maxAbs = 0;
        for (const cat of CATEGORIES) {
            const v = Math.abs(this.values[cat.key]);
            if (v > maxAbs) maxAbs = v;
        }
        if (maxAbs < 0.01) maxAbs = 1;

        const textColor = isLight ? '#1A1612' : '#E8DED4';
        const mutedColor = isLight ? '#1A161266' : '#E8DED466';

        ctx.textBaseline = 'middle';

        for (let i = 0; i < CATEGORIES.length; i++) {
            const cat = CATEGORIES[i];
            const val = this.values[cat.key];
            const y = PAD + i * (BAR_H + GAP);

            // Label
            ctx.fillStyle = textColor;
            ctx.font = '9px Noto Sans, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(cat.label, LABEL_W, y + BAR_H / 2);

            // Track background
            const bx = LABEL_W + 6;
            ctx.fillStyle = mutedColor;
            ctx.beginPath();
            ctx.roundRect(bx, y, barW, BAR_H, 3);
            ctx.fill();

            // Bar
            const ratio = Math.abs(val) / maxAbs;
            const barLen = Math.max(2, ratio * barW);
            ctx.fillStyle = cat.color;
            ctx.beginPath();
            ctx.roundRect(bx, y, barLen, BAR_H, 3);
            ctx.fill();

            // Value text
            const fmt = Math.abs(val) < 0.01 ? '0' : Math.abs(val) > 999 ? val.toExponential(1) : val.toFixed(1);
            ctx.fillStyle = textColor;
            ctx.font = '8px Noto Sans Mono, monospace';
            ctx.textAlign = 'left';
            ctx.fillText(fmt, bx + barLen + 4, y + BAR_H / 2);
        }
    }
}
