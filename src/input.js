import Vec2 from './vec2.js';

export default class InputHandler {
    constructor(canvas, sim) {
        this.canvas = canvas;
        this.sim = sim;

        this.isDragging = false;
        this.dragStart = new Vec2(0, 0);
        this.currentPos = new Vec2(0, 0);
        this.mode = 'place';

        this.canvasRect = canvas.getBoundingClientRect();

        this.massInput = document.getElementById('massInput');
        this.chargeInput = document.getElementById('chargeInput');
        this.spinInput = document.getElementById('spinInput');

        // Multi-touch state
        this._pinching = false;
        this._wasPinching = false;
        this._lastPinchDist = 0;
        this._lastPinchCenterX = 0;
        this._lastPinchCenterY = 0;

        this.setupListeners();
    }

    updateRect() {
        this.canvasRect = this.canvas.getBoundingClientRect();
    }

    /** Convert screen (client) coords to world coords via camera */
    getPos(clientX, clientY) {
        const sx = clientX - this.canvasRect.left;
        const sy = clientY - this.canvasRect.top;
        const cam = this.sim.camera;
        const w = this.sim.width, h = this.sim.height;
        return new Vec2(
            (sx - w / 2) / cam.zoom + cam.x,
            (sy - h / 2) / cam.zoom + cam.y
        );
    }

    setupListeners() {
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });

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
            // Start pinch-to-zoom
            this._pinching = true;
            this.isDragging = false; // cancel any single-finger drag
            const t0 = e.touches[0], t1 = e.touches[1];
            this._lastPinchDist = this._pinchDist(t0, t1);
            this._lastPinchCenterX = (t0.clientX + t1.clientX) / 2;
            this._lastPinchCenterY = (t0.clientY + t1.clientY) / 2;
            return;
        }

        if (e.touches.length === 1 && !this._wasPinching) {
            // Single finger — start drag for spawn
            const t = e.touches[0];
            this.isDragging = true;
            this.dragStart = this.getPos(t.clientX, t.clientY);
            this.currentPos = this.dragStart.clone();
        }
    }

    onTouchMove(e) {
        e.preventDefault();

        if (e.touches.length === 2 && this._pinching) {
            const t0 = e.touches[0], t1 = e.touches[1];
            const dist = this._pinchDist(t0, t1);
            const cx = (t0.clientX + t1.clientX) / 2;
            const cy = (t0.clientY + t1.clientY) / 2;

            // Pinch-to-zoom: reuse onWheel math — preserve world point under pinch center
            const cam = this.sim.camera;
            const w = this.sim.width, h = this.sim.height;
            const sx = cx - this.canvasRect.left;
            const sy = cy - this.canvasRect.top;

            // World pos under pinch center before zoom
            const wx = (sx - w / 2) / cam.zoom + cam.x;
            const wy = (sy - h / 2) / cam.zoom + cam.y;

            // Scale factor from pinch delta
            const factor = dist / this._lastPinchDist;
            cam.zoom = Math.min(Math.max(cam.zoom * factor, 1), 3);

            // Adjust camera so pinch center still points at same world pos
            cam.x = wx - (sx - w / 2) / cam.zoom;
            cam.y = wy - (sy - h / 2) / cam.zoom;

            // Pan: translate by movement of pinch center
            const panDx = (cx - this._lastPinchCenterX) / cam.zoom;
            const panDy = (cy - this._lastPinchCenterY) / cam.zoom;
            cam.x -= panDx;
            cam.y -= panDy;

            this._lastPinchDist = dist;
            this._lastPinchCenterX = cx;
            this._lastPinchCenterY = cy;
            return;
        }

        if (e.touches.length === 1 && this.isDragging && !this._pinching) {
            const t = e.touches[0];
            this.currentPos = this.getPos(t.clientX, t.clientY);
        }
    }

    onTouchEnd(e) {
        e.preventDefault();

        if (e.touches.length === 0) {
            // All fingers lifted
            if (this._pinching) {
                this._pinching = false;
                this._wasPinching = true;
                // Clear wasPinching after a short delay to prevent accidental spawn
                setTimeout(() => { this._wasPinching = false; }, 300);
                return;
            }

            if (this.isDragging && !this._wasPinching) {
                this.isDragging = false;
                const t = e.changedTouches[0];
                this.spawnParticle(this.getPos(t.clientX, t.clientY));
                return;
            }

            this.isDragging = false;
        } else if (e.touches.length === 1 && this._pinching) {
            // Went from 2 fingers to 1 — still in pinch mode, don't start drag
            // Just update state to avoid jump when remaining finger moves
        }
    }

    onWheel(e) {
        e.preventDefault();
        const cam = this.sim.camera;
        const w = this.sim.width, h = this.sim.height;

        // Screen coords of mouse
        const sx = e.clientX - this.canvasRect.left;
        const sy = e.clientY - this.canvasRect.top;

        // World pos under mouse before zoom
        const wx = (sx - w / 2) / cam.zoom + cam.x;
        const wy = (sy - h / 2) / cam.zoom + cam.y;

        // Update zoom (clamp to reasonable range)
        const factor = e.deltaY > 0 ? 0.9 : 1.1;
        cam.zoom = Math.min(Math.max(cam.zoom * factor, 1), 3);

        // Adjust camera so mouse still points at same world pos
        cam.x = wx - (sx - w / 2) / cam.zoom;
        cam.y = wy - (sy - h / 2) / cam.zoom;
    }

    onMouseDown(e) {
        if (e.button === 2) {
            const pos = this.getPos(e.clientX, e.clientY);
            this.sim.particles = this.sim.particles.filter(p => p.pos.dist(pos) > p.radius + 5);
            return;
        }

        this.isDragging = true;
        this.dragStart = this.getPos(e.clientX, e.clientY);
        this.currentPos = this.dragStart.clone();
    }

    onMouseMove(e) {
        this.currentPos = this.getPos(e.clientX, e.clientY);
    }

    onMouseUp(e) {
        if (!this.isDragging) return;
        this.isDragging = false;
        if (e.button !== 0) return;
        this.spawnParticle(this.getPos(e.clientX, e.clientY));
    }

    spawnParticle(endPos) {
        const dragVector = Vec2.sub(this.dragStart, endPos);

        const mode = this.mode;
        const mass = parseFloat(this.massInput.value);
        const charge = parseFloat(this.chargeInput.value);
        const spin = parseFloat(this.spinInput.value);

        if (mode === 'shoot') {
            const velocity = dragVector.scale(0.1);
            this.sim.addParticle(this.dragStart.x, this.dragStart.y, velocity.x, velocity.y, { mass, charge, spin });
        } else if (mode === 'orbit') {
            let bestBody = null;
            let maxGForce = 0;

            for (const p of this.sim.particles) {
                const d = p.pos.dist(this.dragStart);
                if (d > 10) {
                    const force = p.mass / (d * d);
                    if (force > maxGForce) {
                        maxGForce = force;
                        bestBody = p;
                    }
                }
            }

            if (bestBody) {
                const rVec = Vec2.sub(bestBody.pos, this.dragStart);
                const r = rVec.mag();
                const dir = rVec.normalize();

                const vMag = Math.min(Math.sqrt(bestBody.mass / r), 0.99);
                const vx = -dir.y * vMag;
                const vy = dir.x * vMag;
                this.sim.addParticle(this.dragStart.x, this.dragStart.y, vx, vy, { mass, charge, spin });
            } else {
                this.sim.addParticle(this.dragStart.x, this.dragStart.y, 0, 0, { mass, charge, spin });
            }
        } else {
            this.sim.addParticle(this.dragStart.x, this.dragStart.y, 0, 0, { mass, charge, spin });
        }
    }
}
