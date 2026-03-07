import { MAX_TRAIL_LENGTH, PHOTON_LIFETIME, INERTIA_K, PI, TWO_PI, HALF_PI, VELOCITY_VECTOR_SCALE, FORCE_VECTOR_SCALE } from './config.js';
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
        this.higgsField = null;
        this.axionField = null;
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
    }

    setTheme(isLight) {
        this.isLight = isLight;
    }

    render(particles, dt = 0.016, camera, photons, pions) {
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

        // Higgs field overlay (world space, behind trails)
        if (this.higgsField && window.sim && window.sim.physics.higgsEnabled) {
            this.higgsField.render(this.isLight);
            this.higgsField.draw(ctx, this.domainW, this.domainH);
        }

        // Axion field overlay (world space, behind trails)
        if (this.axionField && window.sim && window.sim.physics.axionEnabled) {
            this.axionField.render(this.isLight);
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
                : (isLight ? _r(_PAL.light.text, 0.4) : _r(_PAL.dark.text, 0.5));
            ctx.lineWidth = 1 / (camera ? camera.zoom : 1);
            ctx.setLineDash(this.input._rightButton ? [3, 3] : [5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);
        }

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

        // Cache BH state once instead of window.sim lookup per particle
        const bhEnabled = window.sim && window.sim.physics.blackHoleEnabled;
        const ergoStyle = isLight ? _r(_PAL.light.text, 0.3) : _r(_PAL.dark.text, 0.4);
        const neutralGlow = !isLight ? _r(_PAL.dark.text, 0.5) : null;

        // Batch all particle fills with same shadow state to minimize state changes
        if (isLight) {
            ctx.shadowBlur = 0;
            for (let i = 0, len = particles.length; i < len; i++) {
                const p = particles[i];
                ctx.fillStyle = p.color;
                ctx.beginPath();
                ctx.arc(p.pos.x, p.pos.y, p.radius, 0, TWO_PI);
                ctx.fill();
            }
        } else {
            // Dark mode: group by shadow state to reduce changes
            // First pass: uncharged particles (uniform shadow)
            ctx.shadowBlur = 5;
            ctx.shadowColor = neutralGlow;
            for (let i = 0, len = particles.length; i < len; i++) {
                const p = particles[i];
                if (p.charge !== 0) continue;
                ctx.fillStyle = p.color;
                ctx.beginPath();
                ctx.arc(p.pos.x, p.pos.y, p.radius, 0, TWO_PI);
                ctx.fill();
            }
            // Second pass: charged particles (per-particle shadow)
            for (let i = 0, len = particles.length; i < len; i++) {
                const p = particles[i];
                if (p.charge === 0) continue;
                const absQ = p.charge > 0 ? p.charge : -p.charge;
                ctx.shadowBlur = absQ * 3 + 10 < 50 ? absQ * 3 + 10 : 50;
                ctx.shadowColor = p.color;
                ctx.fillStyle = p.color;
                ctx.beginPath();
                ctx.arc(p.pos.x, p.pos.y, p.radius, 0, TWO_PI);
                ctx.fill();
            }
        }

        // Spin rings + ergospheres (less frequent, ok to iterate once)
        ctx.shadowBlur = 0;
        for (let i = 0, len = particles.length; i < len; i++) {
            const p = particles[i];
            if (p.angVel !== 0) {
                this.drawSpinRing(ctx, p, isLight, blendMode);
            }
            if (bhEnabled && p.mass > 0) {
                const M = p.mass;
                const a = INERTIA_K * Math.cbrt(p.mass) ** 2 * Math.abs(p.angVel);
                const rErgo = M + Math.sqrt(Math.max(0, M * M - a * a));
                if (rErgo > p.radius + 0.3) {
                    ctx.globalCompositeOperation = 'source-over';
                    ctx.beginPath();
                    ctx.arc(p.pos.x, p.pos.y, rErgo, 0, TWO_PI);
                    ctx.strokeStyle = ergoStyle;
                    ctx.lineWidth = 0.15;
                    ctx.setLineDash([0.3, 0.3]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.globalCompositeOperation = blendMode;
                }
            }

            // Antimatter indicator: dashed ring
            if (p.antimatter) {
                ctx.globalCompositeOperation = 'source-over';
                ctx.beginPath();
                ctx.arc(p.pos.x, p.pos.y, p.radius + 0.4, 0, TWO_PI);
                ctx.strokeStyle = isLight ? '#888' : '#ccc';
                ctx.lineWidth = 0.25;
                ctx.setLineDash([0.5, 0.3]);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalCompositeOperation = blendMode;
            }
        }
    }

    drawArrow(ctx, x1, y1, x2, y2, invZoom, color) {
        const dx = x2 - x1, dy = y2 - y1;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len < 0.5 * invZoom) return;

        const nx = dx / len, ny = dy / len;
        const hasHead = len >= 0.5;
        const headLen = 0.5;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(hasHead ? x2 - nx * headLen : x2, hasHead ? y2 - ny * headLen : y2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 0.25;
        ctx.stroke();

        if (hasHead) {
            ctx.beginPath();
            ctx.moveTo(x2, y2);
            ctx.lineTo(x2 - nx * headLen + ny * headLen * 0.5, y2 - ny * headLen - nx * headLen * 0.5);
            ctx.lineTo(x2 - nx * headLen - ny * headLen * 0.5, y2 - ny * headLen + nx * headLen * 0.5);
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
        }
    }

    drawVelocityVectors(ctx, particles, invZoom, isLight) {
        const scale = VELOCITY_VECTOR_SCALE;
        const color = isLight ? _PAL.light.text : _PAL.dark.text;
        for (const p of particles) {
            const vx = p.vel.x * scale, vy = p.vel.y * scale;
            const mag = Math.sqrt(vx * vx + vy * vy);
            if (mag < 1 * invZoom) continue;
            this.drawArrow(ctx, p.pos.x, p.pos.y, p.pos.x + vx, p.pos.y + vy, invZoom, color);
        }
    }

    drawForceVectors(ctx, particles, invZoom, isLight) {
        const scale = FORCE_VECTOR_SCALE;
        const color = isLight ? _PAL.accent : _PAL.accentLight;
        for (const p of particles) {
            // Sum all 8 component vectors (includes Boris display forces)
            const s = scale / p.mass;
            let fx = (p.forceGravity.x + p.forceCoulomb.x + p.forceMagnetic.x + p.forceGravitomag.x + p.force1PN.x + p.forceSpinCurv.x + p.forceRadiation.x + p.forceYukawa.x + p.forceExternal.x + p.forceHiggs.x + p.forceAxion.x) * s;
            let fy = (p.forceGravity.y + p.forceCoulomb.y + p.forceMagnetic.y + p.forceGravitomag.y + p.force1PN.y + p.forceSpinCurv.y + p.forceRadiation.y + p.forceYukawa.y + p.forceExternal.y + p.forceHiggs.y + p.forceAxion.y) * s;
            const mag = Math.sqrt(fx * fx + fy * fy);
            if (mag < 0.1 * invZoom) continue;
            this.drawArrow(ctx, p.pos.x, p.pos.y, p.pos.x + fx, p.pos.y + fy, invZoom, color);
        }
    }

    drawForceComponentVectors(ctx, particles, invZoom, isLight) {
        const scale = FORCE_VECTOR_SCALE;
        const threshold = 0.1 * invZoom;
        const threshSq = threshold * threshold;
        // Iterate particles in outer loop to maximize cache locality
        for (let i = 0, len = particles.length; i < len; i++) {
            const p = particles[i];
            const s = scale / p.mass;
            const px = p.pos.x, py = p.pos.y;
            // Inline each force to avoid dynamic property lookup
            let fx, fy;
            fx = p.forceGravity.x * s; fy = p.forceGravity.y * s;
            if (fx * fx + fy * fy >= threshSq) this.drawArrow(ctx, px, py, px + fx, py + fy, invZoom, _forceCompColors.gravity);
            fx = p.forceCoulomb.x * s; fy = p.forceCoulomb.y * s;
            if (fx * fx + fy * fy >= threshSq) this.drawArrow(ctx, px, py, px + fx, py + fy, invZoom, _forceCompColors.coulomb);
            fx = p.forceMagnetic.x * s; fy = p.forceMagnetic.y * s;
            if (fx * fx + fy * fy >= threshSq) this.drawArrow(ctx, px, py, px + fx, py + fy, invZoom, _forceCompColors.magnetic);
            fx = p.forceGravitomag.x * s; fy = p.forceGravitomag.y * s;
            if (fx * fx + fy * fy >= threshSq) this.drawArrow(ctx, px, py, px + fx, py + fy, invZoom, _forceCompColors.gravitomag);
            fx = p.force1PN.x * s; fy = p.force1PN.y * s;
            if (fx * fx + fy * fy >= threshSq) this.drawArrow(ctx, px, py, px + fx, py + fy, invZoom, _forceCompColors.onepn);
            fx = p.forceSpinCurv.x * s; fy = p.forceSpinCurv.y * s;
            if (fx * fx + fy * fy >= threshSq) this.drawArrow(ctx, px, py, px + fx, py + fy, invZoom, _forceCompColors.spinCurv);
            fx = p.forceRadiation.x * s; fy = p.forceRadiation.y * s;
            if (fx * fx + fy * fy >= threshSq) this.drawArrow(ctx, px, py, px + fx, py + fy, invZoom, _forceCompColors.radiation);
            fx = p.forceYukawa.x * s; fy = p.forceYukawa.y * s;
            if (fx * fx + fy * fy >= threshSq) this.drawArrow(ctx, px, py, px + fx, py + fy, invZoom, _forceCompColors.yukawa);
            fx = p.forceExternal.x * s; fy = p.forceExternal.y * s;
            if (fx * fx + fy * fy >= threshSq) this.drawArrow(ctx, px, py, px + fx, py + fy, invZoom, _forceCompColors.external);
            fx = p.forceHiggs.x * s; fy = p.forceHiggs.y * s;
            if (fx * fx + fy * fy >= threshSq) this.drawArrow(ctx, px, py, px + fx, py + fy, invZoom, _forceCompColors.higgs);
            fx = p.forceAxion.x * s; fy = p.forceAxion.y * s;
            if (fx * fx + fy * fy >= threshSq) this.drawArrow(ctx, px, py, px + fx, py + fy, invZoom, _forceCompColors.axion);
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
        ctx.fillStyle = color;

        for (const p of particles) {
            let val = getValue(p);
            if (Math.abs(val) < threshold) continue;
            val /= INERTIA_K * p.mass * p.radius * p.radius;

            const ringRadius = p.radius + offset;
            const sweep = Math.min(scale * Math.abs(val), maxSweep);
            const dir = val > 0 ? -1 : 1;
            const startAngle = -HALF_PI;
            const endAngle = startAngle - dir * sweep;

            ctx.beginPath();
            ctx.arc(p.pos.x, p.pos.y, ringRadius, startAngle, endAngle, dir > 0);
            ctx.stroke();

            if (sweep * ringRadius >= 0.5) {
                const ax = p.pos.x + Math.cos(endAngle) * ringRadius;
                const ay = p.pos.y + Math.sin(endAngle) * ringRadius;
                const sweepDir = endAngle - dir * HALF_PI;
                const h = 0.5;
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
    }

    drawSpinRing(ctx, p, isLight, blendMode) {
        ctx.shadowBlur = 0;
        const dir = -Math.sign(p.angVel);
        // Arc length proportional to surface speed; caps at full circle
        const arcLen = Math.min(Math.abs(p.angVel) * p.radius * TWO_PI, TWO_PI);
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
        // Caller already guards photons && photons.length
        ctx.globalCompositeOperation = isLight ? 'source-over' : 'lighter';
        ctx.shadowBlur = 0;
        const alphaScale = isLight ? 0.6 : 0.8;

        // Batch by type to minimize fillStyle/shadowColor changes
        for (let pass = 0; pass < 2; pass++) {
            const isGrav = pass === 1;
            const color = isGrav ? _PAL.extended.red : _PAL.extended.yellow;
            const glowColor = _r(color, 0.5);
            ctx.fillStyle = color;

            for (let i = 0, len = photons.length; i < len; i++) {
                const ph = photons[i];
                if ((ph.type === 'grav') !== isGrav) continue;
                const alpha = 1 - ph.lifetime / PHOTON_LIFETIME;
                if (alpha <= 0) continue;
                const size = 0.25 + 2 * ph.energy;
                const r = size < 5 ? size : 5;
                if (!isLight) {
                    ctx.shadowBlur = size * 3 < 15 ? size * 3 : 15;
                    ctx.shadowColor = glowColor;
                }
                ctx.globalAlpha = alpha * alphaScale;
                ctx.beginPath();
                ctx.arc(ph.pos.x, ph.pos.y, r, 0, TWO_PI);
                ctx.fill();
            }
            if (!isLight) ctx.shadowBlur = 0;
        }
        ctx.globalAlpha = 1;
    }

    drawPions(ctx, pions, isLight) {
        ctx.globalCompositeOperation = isLight ? 'source-over' : 'lighter';
        ctx.shadowBlur = 0;
        const alphaScale = isLight ? 0.7 : 0.9;
        const color = _PAL.extended.green;
        const glowColor = _r(color, 0.5);
        ctx.fillStyle = color;

        for (let i = 0, len = pions.length; i < len; i++) {
            const pn = pions[i];
            const size = 0.25 + 2 * pn.energy;
            const r = size < 5 ? size : 5;
            if (!isLight) {
                ctx.shadowBlur = size * 3 < 15 ? size * 3 : 15;
                ctx.shadowColor = glowColor;
            }
            ctx.globalAlpha = alphaScale;
            ctx.beginPath();
            ctx.arc(pn.pos.x, pn.pos.y, r, 0, TWO_PI);
            ctx.fill();
        }
        if (!isLight) ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
    }
}
