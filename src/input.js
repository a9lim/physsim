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

        // Deferred click for GPU mode (1-frame async hit test)
        this._pendingClick = null; // { pos: Vec2, rightButton: bool }

        canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        canvas.addEventListener('touchstart', (e) => this.onTouchStart(e), { passive: false });
        canvas.addEventListener('touchmove', (e) => this.onTouchMove(e), { passive: false });
        canvas.addEventListener('touchend', (e) => this.onTouchEnd(e), { passive: false });
    }

    _pinchDist(t0, t1) {
        const dx = t0.clientX - t1.clientX;
        const dy = t0.clientY - t1.clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    getPos(cx, cy) {
        const rect = this.canvasRect;
        const sx = cx - rect.left;
        const sy = cy - rect.top;
        return this.sim.camera.screenToWorld(sx, sy, this._posOut);
    }

    _getPosNew(cx, cy) {
        const rect = this.canvasRect;
        const sx = cx - rect.left;
        const sy = cy - rect.top;
        return this.sim.camera.screenToWorld(sx, sy, new Vec2(0, 0));
    }

    refreshCanvasRect() { this.canvasRect = this.canvas.getBoundingClientRect(); }

    // ─── Touch events ───

    onTouchStart(e) {
        e.preventDefault();

        if (e.touches.length === 2) {
            this.isDragging = false;
            this._pinching = true;
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

    // ─── Particle deletion ───

    _deleteParticle(p) {
        this.sim.physics._retireParticle(p);
        // GPU path: mark particle dead on GPU side
        if (this.sim._gpuReady && this.sim.backend === 'gpu' && p._gpuIdx != null) {
            this.sim._gpuPhysics.removeParticle(p._gpuIdx);
        }
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

    _deleteByGpuIdx(gpuIdx) {
        this.sim._gpuPhysics.removeParticle(gpuIdx);
        // Remove CPU counterpart if it exists
        const arr = this.sim.particles;
        for (let i = arr.length - 1; i >= 0; i--) {
            if (arr[i]._gpuIdx === gpuIdx) {
                this.sim.physics._retireParticle(arr[i]);
                if (this.sim.selectedParticle === arr[i]) this.sim.selectedParticle = null;
                arr[i] = arr[arr.length - 1];
                arr.pop();
                break;
            }
        }
        this.sim._dirty = true;
        _haptics.trigger('light');
    }

    // ─── Mouse events ───

    onMouseDown(e) {
        const isRight = e.button === 2;
        if (e.button !== 0 && !isRight) return;

        const pos = this._getPosNew(e.clientX, e.clientY);

        // In GPU mode, right-click on particle handled via deferred hit test (onMouseUp)
        // In CPU mode, handle immediately
        if (isRight && this.sim.backend !== 'gpu') {
            const hit = this._cpuFindParticleAt(pos);
            const bhOn = this.sim.physics.blackHoleEnabled;
            if (hit) {
                if (bhOn) this._deleteParticle(hit);
                else if (hit.antimatter) { this.sim.selectedParticle = hit; this.sim._dirty = true; }
                else this._deleteParticle(hit);
                return;
            }
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
                // GPU mode: dispatch hit test for hover, result handled in pollGPUHitResult
                if (this.sim.backend === 'gpu' && this.sim._gpuReady) {
                    this.sim._gpuPhysics.hitTest(this.currentPos.x, this.currentPos.y);
                    // Tooltip updated when poll returns result — hide stale tooltip
                    this.hoveredParticle = null;
                    this.tooltip.hidden = true;
                } else {
                    const hit = this._cpuFindParticleAt(this.currentPos);
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

        // Short click: select, delete, or spawn
        if (this.dragStart.dist(endPos) < DRAG_THRESHOLD) {
            if (this.sim.backend === 'gpu' && this.sim._gpuReady) {
                // GPU mode: defer action until GPU hit test result arrives
                this.sim._gpuPhysics.hitTest(endPos.x, endPos.y);
                this._pendingClick = { pos: endPos, rightButton: this._rightButton };
                return;
            }
            // CPU mode: immediate
            const hit = this._cpuFindParticleAt(endPos);
            if (hit) {
                this._resolveClickHit(hit, this._rightButton);
                return;
            }
        }

        // Long drag or no hit: spawn
        this.spawnParticle(endPos, this._rightButton && !this.sim.physics.blackHoleEnabled);
    }

    /** Resolve a click that hit a particle (select or delete based on type). */
    _resolveClickHit(p, rightButton) {
        const bhOn = this.sim.physics.blackHoleEnabled;
        if (bhOn) {
            if (rightButton) this._deleteParticle(p);
            else this.sim.selectedParticle = p;
        } else {
            if (p.antimatter === rightButton) this.sim.selectedParticle = p;
            else this._deleteParticle(p);
        }
        this.sim._dirty = true;
    }

    // ─── Hit testing ───

    /** CPU-mode hit test using quadtree range query. */
    _cpuFindParticleAt(worldPos) {
        const physics = this.sim.physics;
        const root = physics._lastRoot;
        if (root < 0) return null;
        const searchR = 4; // conservative max particle radius
        const candidates = physics.pool.queryReuse(root, worldPos.x, worldPos.y, searchR, searchR);
        let best = null;
        let bestDist = Infinity;
        for (let i = 0; i < candidates.length; i++) {
            const p = candidates[i].isGhost ? candidates[i].original : candidates[i];
            const d = p.pos.dist(worldPos);
            if (d < p.radius && d < bestDist) {
                bestDist = d;
                best = p;
            }
        }
        return best;
    }

    /**
     * Poll for GPU hit test result. Call once per frame in GPU mode.
     * Completes deferred click actions and updates hover state.
     */
    pollGPUHitResult() {
        if (!this.sim._gpuReady || this.sim.backend !== 'gpu') return;
        const result = this.sim._gpuPhysics.readHitResult();
        if (result === null) return; // not ready yet

        const pending = this._pendingClick;
        this._pendingClick = null;

        if (result >= 0) {
            // GPU found a particle at this index
            const match = this.sim.particles.find(p => p._gpuIdx === result);
            if (pending && match) {
                // Complete deferred click
                this._resolveClickHit(match, pending.rightButton);
            } else if (pending && !match) {
                // GPU found particle but no CPU counterpart — delete directly by GPU index
                if (pending.rightButton) {
                    this._deleteByGpuIdx(result);
                } else {
                    // Can't select without CPU particle — ignore
                }
            } else {
                // No pending click — this was a hover hit test, update selection
                const hovered = this.sim.particles.find(p => p._gpuIdx === result);
                if (hovered) this.hoveredParticle = hovered;
            }
        } else {
            // GPU returned -1: no particle at click position
            if (pending) {
                // Complete deferred click as spawn
                this.spawnParticle(pending.pos, pending.rightButton && !this.sim.physics.blackHoleEnabled);
            } else {
                this.hoveredParticle = null;
                this.tooltip.hidden = true;
            }
        }
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
