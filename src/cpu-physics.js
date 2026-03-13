/**
 * @fileoverview CPUPhysics — wraps existing Physics class (integrator.js)
 * to conform to the shared PhysicsBackend interface.
 *
 * This is a thin adapter. The actual physics code stays in integrator.js unchanged.
 * The wrapper exists so main.js can swap between CPU and GPU backends.
 */
import Physics from './integrator.js';

export default class CPUPhysics {
    /**
     * @param {Physics} engine - Existing Physics instance to wrap.
     */
    constructor(engine) {
        /** @type {Physics} */
        this.engine = engine;
    }

    /**
     * Expose the underlying Physics engine for code that still needs direct access
     * during the migration period (ui.js, save-load.js, etc.).
     * This will be removed once all callers use the shared interface.
     */
    get _engine() { return this.engine; }
}
