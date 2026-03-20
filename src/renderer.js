import { MAX_TRAIL_LENGTH, PHOTON_LIFETIME, INERTIA_K, TWO_PI, HALF_PI, VELOCITY_VECTOR_SCALE, FORCE_VECTOR_SCALE, FIELD_THROTTLE_MASK } from './config.js';
const _PAL = window._PALETTE;
const _r = window._r;

// Per-force component colors (matching toggle colors in styles.css)
// Extended palette is theme-independent, so no light/dark distinction needed
const _forceCompColors = {
    gravity:     _PAL.extended.red,
    coulomb:     _PAL.extended.blue,
    magnetic:    _PAL.extended.cyan,
    gravitomag:  _PAL.extended.rose,
    onepn:       _PAL.extended.orange,
    spinCurv:    _PAL.extended.purple,
    radiation:   _PAL.extended.yellow,
    torqueSO:    _PAL.extended.purple,
    torqueFD:    _PAL.extended.rose,
    torqueTidal: _PAL.extended.red,
    torqueContact: _PAL.extended.brown,
    yukawa:      _PAL.extended.green,
    external:    _PAL.extended.brown,
    higgs:       _PAL.extended.lime,
    axion:       _PAL.extended.indigo,
};

// Force component types for batched rendering: [propName, color, unused] triples
// Third element pads to 3-stride for loop indexing
const _forceCompTypes = [
    'forceGravity',    _forceCompColors.gravity,    0,
    'forceCoulomb',    _forceCompColors.coulomb,     0,
    'forceMagnetic',   _forceCompColors.magnetic,    0,
    'forceGravitomag', _forceCompColors.gravitomag,  0,
    'force1PN',        _forceCompColors.onepn,       0,
    'forceSpinCurv',   _forceCompColors.spinCurv,    0,
    'forceRadiation',  _forceCompColors.radiation,   0,
    'forceYukawa',     _forceCompColors.yukawa,      0,
    'forceExternal',   _forceCompColors.external,    0,
    'forceHiggs',      _forceCompColors.higgs,       0,
    'forceAxion',      _forceCompColors.axion,       0,
];

// Spin ring colors by sign
const _spinColors = {
    pos: { light: `hsla(${_PAL.spinPos},80%,60%,0.8)`, dark: `hsla(${_PAL.spinPos},80%,60%,0.9)` },
    neg: { light: `hsla(${_PAL.spinNeg},80%,60%,0.8)`, dark: `hsla(${_PAL.spinNeg},80%,60%,0.9)` },
};

// R1: Pre-allocated photon alpha buckets (16 levels)
const _photonBuckets = Array.from({length: 16}, () => []);

// Pre-allocated dash patterns (avoid per-call array allocation)
const _ERGO_DASH = [0.3, 0.3];
const _ANTI_DASH = [0.5, 0.3];
const _NO_DASH = [];

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
        this.higgsField = null;
        this.axionField = null;
        this._fieldFrame = 0;
        // Pre-allocated buffers for batched arrow rendering
        // 4 floats per line (x1,y1,x2,y2), 6 floats per head (3 vertices × 2)
        // Sized for 256 particles; _ensureArrowBuffers() grows if needed
        this._arrowLines = new Float32Array(256 * 4);
        this._arrowHeads = new Float32Array(256 * 6);
        // Pre-allocated buffer for spin ring / torque arc arrowhead endpoints
        // 5 floats per entry: tipX, tipY, ax, ay, endAngle (saved during arc pass)
        this._spinHeadData = new Float32Array(256 * 5);
        // Viewport culling bounds (set per-frame in render())
        this._vpLeft = 0;
        this._vpRight = 0;
        this._vpTop = 0;
        this._vpBottom = 0;
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
    }

    setTheme(isLight) {
        this.isLight = isLight;
        // Cache theme-dependent RGBA strings to avoid per-frame allocation
        this._ergoStyle = isLight ? _r(_PAL.light.text, 0.3) : _r(_PAL.dark.text, 0.4);
        this._photonEmGlow = _r(_PAL.extended.yellow, 0.5);
        this._photonGravGlow = _r(_PAL.extended.red, 0.5);
        this._pionGlow = _r(_PAL.extended.green, 0.5);
        this._dragColor = isLight
            ? _r(_PAL.light.text, 0.4) : _r(_PAL.dark.text, 0.5);
    }

    render(particles, dt = 0.016, camera, photons, pions) {
        const ctx = this.ctx;
        const isLight = this.isLight;

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, this.width, this.height);

        if (this.heatmap) this.heatmap.draw(ctx, this.width, this.height);

        // Compute viewport bounds in world space (for culling)
        if (camera) {
            const z = camera.zoom;
            const halfW = this.width / (2 * z);
            const halfH = this.height / (2 * z);
            this._vpLeft = camera.x - halfW;
            this._vpRight = camera.x + halfW;
            this._vpTop = camera.y - halfH;
            this._vpBottom = camera.y + halfH;
            ctx.setTransform(z, 0, 0, z, this.width / 2 - camera.x * z, this.height / 2 - camera.y * z);
        } else {
            this._vpLeft = 0;
            this._vpRight = this.domainW;
            this._vpTop = 0;
            this._vpBottom = this.domainH;
        }

        // Field overlays: throttle render() via bitmask (matching stats/sidebar pattern)
        const fieldRender = (++this._fieldFrame & FIELD_THROTTLE_MASK) === 0;

        if (this.higgsField && window.sim && window.sim.physics.higgsEnabled) {
            if (fieldRender) this.higgsField.render(this.isLight);
            this.higgsField.draw(ctx, this.domainW, this.domainH);
        }

        if (this.axionField && window.sim && window.sim.physics.axionEnabled) {
            if (fieldRender) this.axionField.render(this.isLight);
            this.axionField.draw(ctx, this.domainW, this.domainH);
        }

        if (this.trails) {
            this.updateTrails(particles);
            this.drawTrails(ctx, particles, isLight, camera);
        } else if (this.trailHistory.size > 0) {
            this.trailHistory.clear();
        }

        this.drawParticles(ctx, particles, isLight);
        if (photons && photons.length) this.drawPhotons(ctx, photons, isLight);
        if (pions && pions.length) this.drawPions(ctx, pions, isLight);

        ctx.globalCompositeOperation = 'source-over';
        ctx.shadowBlur = 0;

        const invZoom = 1 / (camera ? camera.zoom : 1);
        if (this.showVelocity || this.showForce || this.showForceComponents) {
            this._ensureArrowBuffers(particles.length);
        }
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
            ctx.strokeStyle = this.input._rightButton
                ? (isLight ? 'rgba(136,136,136,0.6)' : 'rgba(204,204,204,0.7)')
                : this._dragColor;
            ctx.lineWidth = 1 / (camera ? camera.zoom : 1);
            ctx.stroke();
        }

        ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    /** Draw drag indicator on a cleared canvas (for GPU overlay). */
    drawDragOverlay(camera) {
        const ctx = this.ctx;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, this.width, this.height);
        if (!this.input || !this.input.isDragging) return;
        if (camera) {
            const z = camera.zoom;
            ctx.setTransform(z, 0, 0, z, this.width / 2 - camera.x * z, this.height / 2 - camera.y * z);
        }
        const start = this.input.dragStart;
        const end = this.input.currentPos;
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.strokeStyle = this.input._rightButton
            ? (this.isLight ? 'rgba(136,136,136,0.6)' : 'rgba(204,204,204,0.7)')
            : this._dragColor;
        ctx.lineWidth = 1 / (camera ? camera.zoom : 1);
        ctx.stroke();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    updateTrails(particles) {
        const history = this.trailHistory;
        const capacity = MAX_TRAIL_LENGTH * 2;
        if (!this._activeIds) this._activeIds = new Set();
        const activeIds = this._activeIds;
        activeIds.clear();

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

            const groupCount = 2; // C21: 2 groups halves ctx.stroke() calls vs 4
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

        const bhEnabled = window.sim && window.sim.physics.blackHoleEnabled;
        const ergoStyle = this._ergoStyle;

        // Viewport culling bounds with margin for glow/ergosphere
        const vpL = this._vpLeft - 10, vpR = this._vpRight + 10;
        const vpT = this._vpTop - 10, vpB = this._vpBottom + 10;

        ctx.shadowBlur = 0;
        for (let i = 0, len = particles.length; i < len; i++) {
            const p = particles[i];
            if (p.pos.x < vpL || p.pos.x > vpR || p.pos.y < vpT || p.pos.y > vpB) continue;
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.arc(p.pos.x, p.pos.y, p.radius, 0, TWO_PI);
            ctx.fill();
        }

        // R5: Batch spin rings by sign — two passes (pos/neg), one stroke + one fill each
        // C11: Fused — arrowhead endpoint data saved during arc pass, no second particle loop
        ctx.shadowBlur = 0;
        ctx.globalCompositeOperation = 'source-over';
        ctx.lineWidth = 0.2;
        if (this._spinHeadData.length < particles.length * 5) {
            this._spinHeadData = new Float32Array(particles.length * 5);
        }
        for (let sign = 0; sign < 2; sign++) {
            const isPos = sign === 0;
            const colors = isPos ? _spinColors.pos : _spinColors.neg;
            const style = isLight ? colors.light : colors.dark;
            ctx.strokeStyle = style;

            // Single pass: draw arcs + save arrowhead data
            ctx.beginPath();
            let hasArcs = false;
            let hdc = 0; // head data count (entries of 5 floats)
            for (let i = 0, len = particles.length; i < len; i++) {
                const p = particles[i];
                if (p.angVel === 0 || (p.angVel > 0) !== isPos) continue;
                if (p.pos.x < vpL || p.pos.x > vpR || p.pos.y < vpT || p.pos.y > vpB) continue;
                const dir = -Math.sign(p.angVel);
                const arcLen = Math.min(Math.abs(p.angVel) * p.radius * TWO_PI, TWO_PI);
                const ringRadius = p.radius + 0.5;
                const startAngle = -HALF_PI;
                const endAngle = startAngle - dir * arcLen;
                ctx.moveTo(p.pos.x + Math.cos(startAngle) * ringRadius,
                           p.pos.y + Math.sin(startAngle) * ringRadius);
                ctx.arc(p.pos.x, p.pos.y, ringRadius, startAngle, endAngle, dir > 0);
                hasArcs = true;
                // Save arrowhead data: 5 floats: tipX, tipY, ax, ay, endAngle
                const ax = p.pos.x + Math.cos(endAngle) * ringRadius;
                const ay = p.pos.y + Math.sin(endAngle) * ringRadius;
                const sweepDir = endAngle - dir * HALF_PI;
                const base = hdc * 5;
                this._spinHeadData[base]     = ax + Math.cos(sweepDir);       // tipX
                this._spinHeadData[base + 1] = ay + Math.sin(sweepDir);       // tipY
                this._spinHeadData[base + 2] = ax;
                this._spinHeadData[base + 3] = ay;
                this._spinHeadData[base + 4] = endAngle;
                hdc++;
            }
            if (hasArcs) ctx.stroke();

            // Arrowheads: draw from saved data
            if (hdc > 0) {
                ctx.fillStyle = style;
                ctx.beginPath();
                const spread = 0.4;
                for (let j = 0; j < hdc; j++) {
                    const base = j * 5;
                    const tipX  = this._spinHeadData[base];
                    const tipY  = this._spinHeadData[base + 1];
                    const ax    = this._spinHeadData[base + 2];
                    const ay    = this._spinHeadData[base + 3];
                    const endAng = this._spinHeadData[base + 4];
                    ctx.moveTo(tipX, tipY);
                    ctx.lineTo(ax + Math.cos(endAng) * spread, ay + Math.sin(endAng) * spread);
                    ctx.lineTo(ax - Math.cos(endAng) * spread, ay - Math.sin(endAng) * spread);
                    ctx.closePath();
                }
                ctx.fill();
            }
        }
        ctx.globalCompositeOperation = blendMode;

        // R6: Batch ergospheres — one setLineDash pass instead of per-particle
        if (bhEnabled) {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = ergoStyle;
            ctx.lineWidth = 0.15;
            ctx.setLineDash(_ERGO_DASH);
            ctx.beginPath();
            let hasErgo = false;
            for (let i = 0, len = particles.length; i < len; i++) {
                const p = particles[i];
                if (p.pos.x < vpL || p.pos.x > vpR || p.pos.y < vpT || p.pos.y > vpB) continue;
                if (p.mass <= 0) continue;
                const M = p.mass;
                const a = INERTIA_K * p.bodyRadiusSq * Math.abs(p.angVel);
                const rErgo = M + Math.sqrt(Math.max(0, M * M - a * a));
                if (rErgo > p.radius + 0.3) {
                    ctx.moveTo(p.pos.x + rErgo, p.pos.y);
                    ctx.arc(p.pos.x, p.pos.y, rErgo, 0, TWO_PI);
                    hasErgo = true;
                }
            }
            if (hasErgo) ctx.stroke();
            ctx.setLineDash(_NO_DASH);
            ctx.globalCompositeOperation = blendMode;
        }

        // R6: Batch antimatter indicators — one setLineDash pass
        {
            ctx.beginPath();
            let hasAnti = false;
            for (let i = 0, len = particles.length; i < len; i++) {
                const p = particles[i];
                if (!p.antimatter) continue;
                if (p.pos.x < vpL || p.pos.x > vpR || p.pos.y < vpT || p.pos.y > vpB) continue;
                if (!hasAnti) {
                    ctx.globalCompositeOperation = 'source-over';
                    ctx.strokeStyle = isLight ? '#888' : '#ccc';
                    ctx.lineWidth = 0.25;
                    ctx.setLineDash(_ANTI_DASH);
                    hasAnti = true;
                }
                const r = p.radius + 0.4;
                ctx.moveTo(p.pos.x + r, p.pos.y);
                ctx.arc(p.pos.x, p.pos.y, r, 0, TWO_PI);
            }
            if (hasAnti) {
                ctx.stroke();
                ctx.setLineDash(_NO_DASH);
                ctx.globalCompositeOperation = blendMode;
            }
        }
    }

    // Ensure arrow buffers can hold n arrows
    _ensureArrowBuffers(n) {
        if (n * 4 > this._arrowLines.length) {
            this._arrowLines = new Float32Array(n * 4);
            this._arrowHeads = new Float32Array(n * 6);
        }
    }

    // Batched arrow drawing: accumulates all line segments into one path and
    // all arrowheads into another, then issues a single stroke + single fill.
    // Reduces canvas API calls from O(arrows) to O(1) per color batch.
    // C19: lc/hc counts passed directly — no subarray() allocation.
    _batchArrowsDraw(ctx, color, invZoom, lines, lc, heads, hc) {
        if (lc === 0) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 0.25;
        ctx.beginPath();
        for (let i = 0; i < lc; i += 4) {
            ctx.moveTo(lines[i], lines[i + 1]);
            ctx.lineTo(lines[i + 2], lines[i + 3]);
        }
        ctx.stroke();

        if (hc > 0) {
            ctx.fillStyle = color;
            ctx.beginPath();
            for (let i = 0; i < hc; i += 6) {
                ctx.moveTo(heads[i], heads[i + 1]);
                ctx.lineTo(heads[i + 2], heads[i + 3]);
                ctx.lineTo(heads[i + 4], heads[i + 5]);
                ctx.closePath();
            }
            ctx.fill();
        }
    }

    drawVelocityVectors(ctx, particles, invZoom, isLight) {
        const scale = VELOCITY_VECTOR_SCALE;
        const color = isLight ? _PAL.light.text : _PAL.dark.text;
        const minMag = 1 * invZoom;
        const headLen = 0.5;
        const lines = this._arrowLines;
        const heads = this._arrowHeads;
        let lc = 0, hc = 0;

        for (let i = 0, len = particles.length; i < len; i++) {
            const p = particles[i];
            const vx = p.vel.x * scale, vy = p.vel.y * scale;
            const mag = Math.sqrt(vx * vx + vy * vy);
            if (mag < minMag) continue;
            const px = p.pos.x, py = p.pos.y;
            const ex = px + vx, ey = py + vy;
            const nx = vx / mag, ny = vy / mag;
            const hasHead = mag >= 0.5;
            lines[lc++] = px; lines[lc++] = py;
            lines[lc++] = hasHead ? ex - nx * headLen : ex;
            lines[lc++] = hasHead ? ey - ny * headLen : ey;
            if (hasHead) {
                heads[hc++] = ex; heads[hc++] = ey;
                heads[hc++] = ex - nx * headLen + ny * headLen * 0.5;
                heads[hc++] = ey - ny * headLen - nx * headLen * 0.5;
                heads[hc++] = ex - nx * headLen - ny * headLen * 0.5;
                heads[hc++] = ey - ny * headLen + nx * headLen * 0.5;
            }
        }
        this._batchArrowsDraw(ctx, color, invZoom, lines, lc, heads, hc);
    }

    drawForceVectors(ctx, particles, invZoom, isLight) {
        const scale = FORCE_VECTOR_SCALE;
        const color = isLight ? _PAL.accent : _PAL.accentLight;
        // C22: minLen is strictly tighter than threshold (0.5 > 0.1) — only minLen needed
        const minLen = 0.5 * invZoom;
        const headLen = 0.5;
        const lines = this._arrowLines;
        const heads = this._arrowHeads;
        let lc = 0, hc = 0;

        for (let i = 0, len = particles.length; i < len; i++) {
            const p = particles[i];
            const s = scale / p.mass;
            const fx = p.force.x * s;
            const fy = p.force.y * s;
            const mag = Math.sqrt(fx * fx + fy * fy);
            if (mag < minLen) continue;
            const px = p.pos.x, py = p.pos.y;
            const ex = px + fx, ey = py + fy;
            const nx = fx / mag, ny = fy / mag;
            const hasHead = mag >= 0.5;
            lines[lc++] = px; lines[lc++] = py;
            lines[lc++] = hasHead ? ex - nx * headLen : ex;
            lines[lc++] = hasHead ? ey - ny * headLen : ey;
            if (hasHead) {
                heads[hc++] = ex; heads[hc++] = ey;
                heads[hc++] = ex - nx * headLen + ny * headLen * 0.5;
                heads[hc++] = ey - ny * headLen - nx * headLen * 0.5;
                heads[hc++] = ex - nx * headLen - ny * headLen * 0.5;
                heads[hc++] = ey - ny * headLen + nx * headLen * 0.5;
            }
        }
        this._batchArrowsDraw(ctx, color, invZoom, lines, lc, heads, hc);
    }

    drawForceComponentVectors(ctx, particles, invZoom, isLight) {
        const scale = FORCE_VECTOR_SCALE;
        const threshold = 0.1 * invZoom;
        const threshSq = threshold * threshold;
        const minLen = 0.5 * invZoom;
        const headLen = 0.5;
        const lines = this._arrowLines;
        const heads = this._arrowHeads;

        // Batch by force type: one stroke + one fill per force color
        // Force field name, Vec2 property pairs
        const forceTypes = _forceCompTypes;

        for (let f = 0; f < forceTypes.length; f += 3) {
            const forceProp = forceTypes[f];
            const color = forceTypes[f + 1];
            let lc = 0, hc = 0;

            for (let i = 0, len = particles.length; i < len; i++) {
                const p = particles[i];
                const s = scale / p.mass;
                const fv = p[forceProp];
                const fx = fv.x * s, fy = fv.y * s;
                // R8: Cache magnitude squared to avoid double computation
                const magSq = fx * fx + fy * fy;
                if (magSq < threshSq) continue;
                const mag = Math.sqrt(magSq);
                if (mag < minLen) continue;
                const px = p.pos.x, py = p.pos.y;
                const ex = px + fx, ey = py + fy;
                const nx = fx / mag, ny = fy / mag;
                const hasHead = mag >= 0.5;
                lines[lc++] = px; lines[lc++] = py;
                lines[lc++] = hasHead ? ex - nx * headLen : ex;
                lines[lc++] = hasHead ? ey - ny * headLen : ey;
                if (hasHead) {
                    heads[hc++] = ex; heads[hc++] = ey;
                    heads[hc++] = ex - nx * headLen + ny * headLen * 0.5;
                    heads[hc++] = ey - ny * headLen - nx * headLen * 0.5;
                    heads[hc++] = ex - nx * headLen - ny * headLen * 0.5;
                    heads[hc++] = ey - ny * headLen + nx * headLen * 0.5;
                }
            }
            if (lc > 0) {
                this._batchArrowsDraw(ctx, color, invZoom, lines, lc, heads, hc);
            }
        }
    }

    drawTotalTorqueArc(ctx, particles, invZoom, isLight) {
        const color = isLight ? _PAL.accent : _PAL.accentLight;
        this._drawTorqueArc(ctx, particles, invZoom, color, 3, (p) => p.torqueSpinOrbit + p.torqueFrameDrag + p.torqueTidal + p.torqueContact);
    }

    drawTorqueArcs(ctx, particles, invZoom, isLight) {
        this._drawTorqueArc(ctx, particles, invZoom, _forceCompColors.torqueSO, 2.5, (p) => p.torqueSpinOrbit);
        this._drawTorqueArc(ctx, particles, invZoom, _forceCompColors.torqueFD, 2, (p) => p.torqueFrameDrag);
        this._drawTorqueArc(ctx, particles, invZoom, _forceCompColors.torqueTidal, 1.5, (p) => p.torqueTidal);
        this._drawTorqueArc(ctx, particles, invZoom, _forceCompColors.torqueContact, 1, (p) => p.torqueContact);
    }

    _drawTorqueArc(ctx, particles, invZoom, color, offset, getValue) {
        const scale = FORCE_VECTOR_SCALE / INERTIA_K;
        const maxSweep = TWO_PI;
        const threshold = 1e-8;

        ctx.globalCompositeOperation = 'source-over';
        ctx.lineWidth = 0.25;
        ctx.strokeStyle = color;

        // C12: Fused — getValue called once per particle; arrowhead data saved during arc pass.
        // Head data layout per entry (5 floats): tipX, tipY, ax, ay, endAngle
        if (this._spinHeadData.length < particles.length * 5) {
            this._spinHeadData = new Float32Array(particles.length * 5);
        }
        const hd = this._spinHeadData;

        ctx.beginPath();
        let hasArcs = false;
        let hdc = 0;
        for (let i = 0, len = particles.length; i < len; i++) {
            const p = particles[i];
            let val = getValue(p);
            if (Math.abs(val) < threshold) continue;
            val /= INERTIA_K * p.mass * p.radius * p.radius;

            const ringRadius = p.radius + offset;
            const sweep = Math.min(scale * Math.abs(val), maxSweep);
            const dir = val > 0 ? -1 : 1;
            const startAngle = -HALF_PI;
            const endAngle = startAngle - dir * sweep;

            ctx.moveTo(p.pos.x + Math.cos(startAngle) * ringRadius,
                       p.pos.y + Math.sin(startAngle) * ringRadius);
            ctx.arc(p.pos.x, p.pos.y, ringRadius, startAngle, endAngle, dir > 0);
            hasArcs = true;

            // Save arrowhead data if arc is long enough to warrant a head
            if (sweep * ringRadius >= 0.5) {
                const ax = p.pos.x + Math.cos(endAngle) * ringRadius;
                const ay = p.pos.y + Math.sin(endAngle) * ringRadius;
                const sweepDir = endAngle - dir * HALF_PI;
                const h = 0.5;
                const base = hdc * 5;
                hd[base]     = ax + Math.cos(sweepDir) * h;  // tipX
                hd[base + 1] = ay + Math.sin(sweepDir) * h;  // tipY
                hd[base + 2] = ax;
                hd[base + 3] = ay;
                hd[base + 4] = endAngle;
                hdc++;
            }
        }
        if (hasArcs) ctx.stroke();

        // Arrowheads: draw from saved data
        if (hdc > 0) {
            ctx.fillStyle = color;
            ctx.beginPath();
            const spread = 0.5 * 0.4;
            for (let j = 0; j < hdc; j++) {
                const base = j * 5;
                const tipX   = hd[base];
                const tipY   = hd[base + 1];
                const ax     = hd[base + 2];
                const ay     = hd[base + 3];
                const endAng = hd[base + 4];
                ctx.moveTo(tipX, tipY);
                ctx.lineTo(ax + Math.cos(endAng) * spread, ay + Math.sin(endAng) * spread);
                ctx.lineTo(ax - Math.cos(endAng) * spread, ay - Math.sin(endAng) * spread);
                ctx.closePath();
            }
            ctx.fill();
        }
    }

    drawPhotons(ctx, photons, isLight) {
        ctx.globalCompositeOperation = isLight ? 'source-over' : 'lighter';
        ctx.shadowBlur = 0;
        const alphaScale = isLight ? 0.6 : 0.8;
        const vpL = this._vpLeft - 5, vpR = this._vpRight + 5;
        const vpT = this._vpTop - 5, vpB = this._vpBottom + 5;

        // R1: Bucket photons into 16 alpha levels — reduces up to 1024 fills to ≤32
        for (let pass = 0; pass < 2; pass++) {
            const isGrav = pass === 1;
            const color = isGrav ? _PAL.extended.red : _PAL.extended.yellow;
            ctx.fillStyle = color;

            if (!isLight) {
                ctx.shadowBlur = 12;
                ctx.shadowColor = isGrav ? this._photonGravGlow : this._photonEmGlow;
            }

            for (let b = 0; b < 16; b++) _photonBuckets[b].length = 0;
            for (let i = 0, len = photons.length; i < len; i++) {
                const ph = photons[i];
                if ((ph.type === 'grav') !== isGrav) continue;
                if (ph.pos.x < vpL || ph.pos.x > vpR || ph.pos.y < vpT || ph.pos.y > vpB) continue;
                const alpha = 1 - ph.lifetime / PHOTON_LIFETIME;
                if (alpha <= 0) continue;
                _photonBuckets[(alpha * 15.999) | 0].push(ph);
            }
            for (let b = 0; b < 16; b++) {
                const bucket = _photonBuckets[b];
                if (bucket.length === 0) continue;
                ctx.globalAlpha = ((b + 0.5) / 16) * alphaScale;
                ctx.beginPath();
                for (let j = 0; j < bucket.length; j++) {
                    const ph = bucket[j];
                    const size = 0.25 + 2 * ph.energy;
                    const r = size < 5 ? size : 5;
                    ctx.moveTo(ph.pos.x + r, ph.pos.y);
                    ctx.arc(ph.pos.x, ph.pos.y, r, 0, TWO_PI);
                }
                ctx.fill();
            }
            if (!isLight) ctx.shadowBlur = 0;
        }
        ctx.globalAlpha = 1;
    }

    drawPions(ctx, pions, isLight) {
        ctx.globalCompositeOperation = isLight ? 'source-over' : 'lighter';
        const alphaScale = isLight ? 0.7 : 0.9;
        ctx.fillStyle = _PAL.extended.green;

        if (!isLight) {
            ctx.shadowBlur = 12;
            ctx.shadowColor = this._pionGlow;
        } else {
            ctx.shadowBlur = 0;
        }

        const vpL = this._vpLeft - 5, vpR = this._vpRight + 5;
        const vpT = this._vpTop - 5, vpB = this._vpBottom + 5;

        // R3: Batch all pions into one path — constant alpha, one fill()
        ctx.globalAlpha = alphaScale;
        ctx.beginPath();
        for (let i = 0, len = pions.length; i < len; i++) {
            const pn = pions[i];
            if (pn.pos.x < vpL || pn.pos.x > vpR || pn.pos.y < vpT || pn.pos.y > vpB) continue;
            const size = 0.25 + 2 * pn.energy;
            const r = size < 5 ? size : 5;
            ctx.moveTo(pn.pos.x + r, pn.pos.y);
            ctx.arc(pn.pos.x, pn.pos.y, r, 0, TWO_PI);
        }
        ctx.fill();
        if (!isLight) ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
    }
}
