import Vec2 from './vec2.js';

export default class InputHandler {
    constructor(canvas, sim) {
        this.canvas = canvas;
        this.sim = sim;

        this.isDragging = false;
        this.dragStart = new Vec2(0, 0);
        this.currentPos = new Vec2(0, 0);

        this.setupListeners();
        this.setupUI();
    }

    setupUI() {
        // UI toggles are handled in main.js now
    }

    setupListeners() {
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        // Prevent context menu on right click
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    getPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return new Vec2(e.clientX - rect.left, e.clientY - rect.top);
    }

    onMouseDown(e) {
        if (e.button === 2) {
            // Right click: Remove particle
            const pos = this.getPos(e);
            this.sim.particles = this.sim.particles.filter(p => p.pos.dist(pos) > p.radius + 5);
            return;
        }

        this.isDragging = true;
        this.dragStart = this.getPos(e);
        this.currentPos = this.dragStart.clone();
    }

    onMouseMove(e) {
        this.currentPos = this.getPos(e);
    }

    onMouseUp(e) {
        if (!this.isDragging) return;
        this.isDragging = false;

        if (e.button !== 0) return; // Only process left click release

        const endPos = this.getPos(e);
        const dragVector = Vec2.sub(this.dragStart, endPos); // Pull back to shoot

        const mode = document.querySelector('.mode-btn.active').dataset.mode;
        const mass = parseFloat(document.getElementById('massInput').value);
        const charge = parseFloat(document.getElementById('chargeInput').value);
        const spin = parseFloat(document.getElementById('spinInput').value);

        if (mode === 'shoot') {
            // Velocity proportional to drag distance
            const velocity = dragVector.scale(0.1);
            this.sim.addParticle(this.dragStart.x, this.dragStart.y, velocity.x, velocity.y, { mass, charge, spin });
        } else if (mode === 'orbit') {
            // Find nearest implementation of orbital velocity?
            // For now, let's just place it with a calculated tangential velocity relative to center of mass or nearest massive object
            // Simple heuristic: find massive neighbor
            let bestBody = null;
            let maxGForce = 0;

            this.sim.particles.forEach(p => {
                const d = p.pos.dist(this.dragStart);
                if (d > 10) {
                    const force = p.mass / (d * d);
                    if (force > maxGForce) {
                        maxGForce = force;
                        bestBody = p;
                    }
                }
            });

            if (bestBody) {
                const rVec = Vec2.sub(bestBody.pos, this.dragStart);
                const r = rVec.mag();
                const dir = rVec.normalize();

                // v_orbit = sqrt(G * M / r)
                let vMag = Math.sqrt(this.sim.physics.G * bestBody.mass / r);

                // Clamp orbital velocity to ~0.99c to prevent massive relativistic glitches natively
                const maxV = this.sim.physics.c * 0.99;
                if (vMag > maxV) vMag = maxV;

                // Perpendicular direction
                const vDir = new Vec2(-dir.y, dir.x); // or new Vec2(dir.y, -dir.x) for other way

                // Add initial velocity from drag if any?
                // For orbit mode, let's say drag defines the direction of orbit + extra kick

                const velocity = vDir.scale(vMag);
                this.sim.addParticle(this.dragStart.x, this.dragStart.y, velocity.x, velocity.y, { mass, charge, spin });

            } else {
                // No body to orbit, just place at rest
                this.sim.addParticle(this.dragStart.x, this.dragStart.y, 0, 0, { mass, charge, spin });
            }
        } else {
            // Place mode
            this.sim.addParticle(this.dragStart.x, this.dragStart.y, 0, 0, { mass, charge, spin });
        }
    }
}
