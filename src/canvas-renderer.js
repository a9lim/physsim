/**
 * @fileoverview CanvasRenderer — wraps existing Renderer class (renderer.js)
 * to conform to the shared RenderBackend interface.
 */
import Renderer from './renderer.js';

export default class CanvasRenderer {
    /**
     * @param {CanvasRenderingContext2D} ctx
     * @param {number} width
     * @param {number} height
     */
    constructor(ctx, width, height) {
        /** @type {Renderer} */
        this.engine = new Renderer(ctx, width, height);
    }

    /** Expose underlying Renderer during migration period. */
    get _engine() { return this.engine; }
}
