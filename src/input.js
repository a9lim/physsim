import Vec2 from './vec2.js';
import { PINCH_DEBOUNCE, DRAG_THRESHOLD, SHOOT_VELOCITY_SCALE } from './config.js';

export default class InputHandler {
    constructor(canvas, sim) {
        this.canvas = canvas;
        this.sim = sim;

        this.isDragging = false;
        this.dragStart = new Vec2(0, 0);
        this.currentPos = new Vec2(0, 0);
        this.canvasRect = canvas.getBoundingClientRect();

        this.massInput = document.getElementById('massInput');
        this.chargeInput = document.getElementById('chargeInput');
        this.spinInput = document.getElementById('spinInput');

        this.tooltip = document.getElementById('particle-tooltip');
        this.hoveredParticle = null;
        this._screenX = 0;
        this._screenY = 0;

        this._posOut = new Vec2(0, 0);
        this._pinching = false;
        this._wasPinching = false;
        this._lastPinchDist = 0;
        this._lastPinchCenterX = 0;
        this._lastPinchCenterY = 0;
        this._rightButton = false;
        // A8: Throttle findParticleAt — resolve in rAF instead of per-mousemove
        this._hoverPending = false;
        this._hoverE = null;

        this.setupListeners();
    }

    updateRect() {
        this.canvasRect = this.canvas.getBoundingClientRect();
    }

    /** Convert screen (client) coords to world coords via shared camera */
    getPos(clientX, clientY) {
        const sx = clientX - this.canvasRect.left;
        const sy = clientY - this.canvasRect.top;
        const w = this.sim.camera.screenToWorld(sx, sy);
        return this._posOut.set(w.x, w.y);
    }

    /** getPos that returns a new Vec2 (for values that must persist, e.g. dragStart) */
    _getPosNew(clientX, clientY) {
        const sx = clientX - this.canvasRect.left;
        const sy = clientY - this.canvasRect.top;
        const w = this.sim.camera.screenToWorld(sx, sy);
        return new Vec2(w.x, w.y);
    }

    setupListeners() {
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        this.canvas.addEventListener('mouseleave', () => {
            this.hoveredParticle = null;
            this.tooltip.hidden = true;
        });

        this.sim.camera.bindWheel(this.canvas);

        this.canvas.addEventListener('touchstart', (e) => this.onTouchStart(e), { passive: false });
        this.canvas.addEventListener('touchmove', (e) => this.onTouchMove(e), { passive: false });
        this.canvas.addEventListener('touchend', (e) => this.onTouchEnd(e), { passive: false });
    }

    _pinchDist(t1, t2) {
        const dx = t1.clientX - t2.clientX;
        const dy = t1.clientY - t2.clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    onTouchStart(e) {
        e.preventDefault();

        if (e.touches.length === 2) {
            this._pinching = true;
            this.isDragging = false;
            const t0 = e.touches[0], t1 = e.touches[1];
            this._lastPinchDist = this._pinchDist(t0, t1);
            this._lastPinchCenterX = (t0.clientX + t1.clientX) / 2;
            this._lastPinchCenterY = (t0.clientY + t1.clientY) / 2;
            return;
        }

        if (e.touches.length === 1 && !this._wasPinching) {
            const t = e.touches[0];
            this.isDragging = true;
            this.dragStart = this._getPosNew(t.clientX, t.clientY);
            this.currentPos = this.dragStart.clone();
            this.sim._dirty = true;
        }
    }

    onTouchMove(e) {
        e.preventDefault();

        if (e.touches.length === 2 && this._pinching) {
            const t0 = e.touches[0], t1 = e.touches[1];
            const dist = this._pinchDist(t0, t1);
            const cx = (t0.clientX + t1.clientX) / 2;
            const cy = (t0.clientY + t1.clientY) / 2;
            const sx = cx - this.canvasRect.left;
            const sy = cy - this.canvasRect.top;

            const factor = dist / this._lastPinchDist;
            this.sim.camera.zoomBy(factor, sx, sy);

            this.sim.camera.panBy(cx - this._lastPinchCenterX, cy - this._lastPinchCenterY);

            this._lastPinchDist = dist;
            this._lastPinchCenterX = cx;
            this._lastPinchCenterY = cy;
            return;
        }

        if (e.touches.length === 1 && this.isDragging && !this._pinching) {
            const t = e.touches[0];
            this.currentPos = this._getPosNew(t.clientX, t.clientY);
            this.sim._dirty = true;
        }
    }

    onTouchEnd(e) {
        e.preventDefault();

        if (e.touches.length === 0) {
            if (this._pinching) {
                this._pinching = false;
                this._wasPinching = true;
                // 300ms guard prevents accidental spawn after pinch
                setTimeout(() => { this._wasPinching = false; }, PINCH_DEBOUNCE);
                return;
            }

            if (this.isDragging && !this._wasPinching) {
                this.isDragging = false;
                const t = e.changedTouches[0];
                this.spawnParticle(this._getPosNew(t.clientX, t.clientY));
                this.sim._dirty = true;
                return;
            }

            this.isDragging = false;
        } else if (e.touches.length === 1 && this._pinching) {
            // Still in pinch mode; don't start drag from remaining finger
        }
    }

    _deleteParticle(p) {
        this.sim.physics._retireParticle(p);
        // A9: swap-and-pop instead of filter (O(1) vs O(N), no allocation)
        const arr = this.sim.particles;
        const idx = arr.indexOf(p);
        if (idx !== -1) {
            arr[idx] = arr[arr.length - 1];
            arr.pop();
        }
        if (this.sim.selectedParticle === p) this.sim.selectedParticle = null;
        this.sim._dirty = true;
        _haptics.trigger('light');
    }

    onMouseDown(e) {
        const isRight = e.button === 2;
        if (e.button !== 0 && !isRight) return;

        const pos = this._getPosNew(e.clientX, e.clientY);
        const hit = this.findParticleAt(pos);
        const bhOn = this.sim.physics.blackHoleEnabled;

        // Right-click on particle: select antimatter, delete matter
        // BH mode: right-click always deletes (no antimatter distinction)
        if (isRight && hit) {
            if (bhOn) this._deleteParticle(hit);
            else if (hit.antimatter) { this.sim.selectedParticle = hit; this.sim._dirty = true; }
            else this._deleteParticle(hit);
            return;
        }

        this.isDragging = true;
        this._rightButton = isRight;
        this.dragStart = pos;
        this.currentPos = pos.clone();
        this.sim._dirty = true;
    }

    onMouseMove(e) {
        const pos = this.getPos(e.clientX, e.clientY);
        this.currentPos.set(pos.x, pos.y);
        this._screenX = e.clientX;
        this._screenY = e.clientY;

        if (this.isDragging) this.sim._dirty = true;

        // A8: Defer hover search to rAF (120-240Hz mousemove → ~60Hz search)
        if (!this._hoverPending) {
            this._hoverPending = true;
            this._hoverE = e;
            requestAnimationFrame(() => {
                this._hoverPending = false;
                const ev = this._hoverE;
                const hit = this.findParticleAt(this.currentPos);
                this.hoveredParticle = hit;
                if (hit) {
                    const speed = Math.sqrt(hit.vel.x * hit.vel.x + hit.vel.y * hit.vel.y);
                    const spin = (hit.angVel * hit.radius).toFixed(3);
                    this.tooltip.textContent = `m=${hit.mass.toFixed(2)}  q=${hit.charge.toFixed(2)}  s=${spin}c  v=${speed.toFixed(3)}c`;
                    this.tooltip.style.transform = `translate(${ev.clientX + 14}px,${ev.clientY - 10}px)`;
                    this.tooltip.hidden = false;
                } else {
                    this.tooltip.hidden = true;
                }
            });
        } else {
            this._hoverE = e; // Update to latest event for pending rAF
        }
    }

    onMouseUp(e) {
        if (!this.isDragging) return;
        this.isDragging = false;
        this.sim._dirty = true;

        const endPos = this._getPosNew(e.clientX, e.clientY);

        // Short click on a particle: select same type, delete opposite type
        // BH mode: left-click selects, right-click deletes (no antimatter)
        if (this.dragStart.dist(endPos) < DRAG_THRESHOLD) {
            const hit = this.findParticleAt(endPos);
            if (hit) {
                const bhOn = this.sim.physics.blackHoleEnabled;
                if (bhOn) {
                    if (this._rightButton) this._deleteParticle(hit);
                    else this.sim.selectedParticle = hit;
                } else {
                    if (hit.antimatter === this._rightButton) this.sim.selectedParticle = hit;
                    else this._deleteParticle(hit);
                }
                return;
            }
        }

        // BH mode: never spawn antimatter
        this.spawnParticle(endPos, this._rightButton && !this.sim.physics.blackHoleEnabled);
    }

    findParticleAt(worldPos) {
        // GPU backend: fall back to CPU particle array scan
        // (GPU hit test dispatch is deferred — not needed for spawn/select)
        if (this.sim.backend === 'gpu') {
            return this._cpuFallbackHitTest(worldPos);
        }
        // CPU backend: existing O(N) scan
        let best = null;
        let bestDist = Infinity;
        for (const p of this.sim.particles) {
            const d = p.pos.dist(worldPos);
            if (d < p.radius && d < bestDist) {
                bestDist = d;
                best = p;
            }
        }
        return best;
    }

    /**
     * CPU-side fallback for immediate hit testing when GPU path active.
     * Uses the particle state array (which is periodically synced from GPU
     * readback for selected-particle display).
     */
    _cpuFallbackHitTest(worldPos) {
        // Use CPU-side particle array (always maintained alongside GPU)
        let best = null;
        let bestDist = Infinity;
        for (const p of this.sim.particles) {
            const d = p.pos.dist(worldPos);
            if (d < p.radius && d < bestDist) {
                bestDist = d;
                best = p;
            }
        }
        return best;
    }

    spawnParticle(endPos, antimatter = false) {
        const mass = parseFloat(this.massInput.value);
        const charge = parseFloat(this.chargeInput.value) * (antimatter ? -1 : 1);
        const spin = parseFloat(this.spinInput.value) * (antimatter ? -1 : 1);

        const vx = (this.dragStart.x - endPos.x) * SHOOT_VELOCITY_SCALE;
        const vy = (this.dragStart.y - endPos.y) * SHOOT_VELOCITY_SCALE;
        this.sim.addParticle(this.dragStart.x, this.dragStart.y, vx, vy, { mass, charge, spin, antimatter });
        _haptics.trigger('light');
    }
}
