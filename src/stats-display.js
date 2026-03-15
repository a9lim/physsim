import { computeEnergies } from './energy.js';
import { DISPLAY_SCALE, STATS_THROTTLE_MASK, EPSILON } from './config.js';
const fmt = (v) => { v *= DISPLAY_SCALE; return Math.abs(v) > 999 ? v.toExponential(2) : v.toFixed(2); };
const fmtRaw = (v) => Math.abs(v) > 999 ? v.toExponential(2) : v.toFixed(2);
const fmtDrift = (v) => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
// R15: Set textContent only when value changed (avoids DOM text node rebuild)
function _set(el, val) { if (el.textContent !== val) el.textContent = val; }

export default class StatsDisplay {
    constructor(dom, selDom) {
        this.dom = dom;
        this.selDom = selDom;
        this.initialEnergy = null;
        this.initialMomentum = null;
        this.initialAngMom = null;
        this._frameCount = 0;
        // R14: Pre-allocate force descriptor array (avoids 11 object allocations per updateSelected)
        this._forceDescs = selDom ? [
            { row: selDom.fbGravity, val: selDom.fbGravityVal, vec: null },
            { row: selDom.fbCoulomb, val: selDom.fbCoulombVal, vec: null },
            { row: selDom.fbMagnetic, val: selDom.fbMagneticVal, vec: null },
            { row: selDom.fbGravitomag, val: selDom.fbGravitomagVal, vec: null },
            { row: selDom.fb1pn, val: selDom.fb1pnVal, vec: null },
            { row: selDom.fbSpincurv, val: selDom.fbSpincurvVal, vec: null },
            { row: selDom.fbRadiation, val: selDom.fbRadiationVal, vec: null },
            { row: selDom.fbYukawa, val: selDom.fbYukawaVal, vec: null },
            { row: selDom.fbExternal, val: selDom.fbExternalVal, vec: null },
            { row: selDom.fbHiggs, val: selDom.fbHiggsVal, vec: null },
            { row: selDom.fbAxion, val: selDom.fbAxionVal, vec: null },
        ] : [];
    }

    resetBaseline() {
        this.initialEnergy = null;
        this.initialMomentum = null;
        this.initialAngMom = null;
    }

    updateEnergy(particles, physics, sim) {
        // Throttle DOM writes to every 4th frame (stats are display-only)
        if (++this._frameCount & STATS_THROTTLE_MASK) return;

        const e = computeEnergies(particles, physics, sim);
        const angMom = e.orbitalAngMom + e.spinAngMom;

        const totalPx = e.px + e.fieldPx + sim.totalRadiatedPx;
        const totalPy = e.py + e.fieldPy + sim.totalRadiatedPy;
        const pMag = Math.sqrt(totalPx * totalPx + totalPy * totalPy);
        const total = e.linearKE + e.spinKE + e.pe + e.fieldEnergy + e.higgsFieldEnergy + e.axionFieldEnergy + sim.totalRadiated;

        if (this.initialEnergy === null && particles.length > 0) {
            this.initialEnergy = total;
            this.initialMomentum = pMag;
            this.initialAngMom = angMom;
        }

        const eDrift = this.initialEnergy !== null && this.initialEnergy !== 0
            ? ((total - this.initialEnergy) / Math.abs(this.initialEnergy) * 100) : 0;
        const pDrift = this.initialMomentum !== null && this.initialMomentum !== 0
            ? ((pMag - this.initialMomentum) / Math.abs(this.initialMomentum) * 100) : 0;
        const aDrift = this.initialAngMom !== null && this.initialAngMom !== 0
            ? ((angMom - this.initialAngMom) / Math.abs(this.initialAngMom) * 100) : 0;

        _set(this.dom.linearKE, fmt(e.linearKE));
        _set(this.dom.spinKE, fmt(e.spinKE));
        _set(this.dom.massE, fmt(e.totalMass));
        _set(this.dom.potentialE, fmt(e.pe));
        _set(this.dom.totalE, fmt(total));
        _set(this.dom.energyDrift, fmtDrift(eDrift));
        _set(this.dom.fieldE, fmt(e.fieldEnergy + e.higgsFieldEnergy + e.axionFieldEnergy));
        _set(this.dom.radiatedE, fmt(sim.totalRadiated));
        _set(this.dom.momentum, fmt(pMag));
        _set(this.dom.particleMom, fmt(Math.sqrt(e.px * e.px + e.py * e.py)));
        _set(this.dom.fieldMom, fmt(Math.sqrt(e.fieldPx * e.fieldPx + e.fieldPy * e.fieldPy)));
        _set(this.dom.radiatedMom, fmt(Math.sqrt(sim.totalRadiatedPx * sim.totalRadiatedPx + sim.totalRadiatedPy * sim.totalRadiatedPy)));
        _set(this.dom.momentumDrift, fmtDrift(pDrift));
        _set(this.dom.angularMomentum, fmt(angMom));
        _set(this.dom.orbitalAngMom, fmt(e.orbitalAngMom));
        _set(this.dom.spinAngMom, fmt(e.spinAngMom));
        _set(this.dom.angMomDrift, fmtDrift(aDrift));
    }

    /**
     * Update energy/momentum display from GPU readback data.
     * All quantities computed on GPU: KE, PE (O(N²)), Darwin field energy/momentum,
     * scalar field energy/momentum, particle-field interaction energy.
     */
    updateEnergyGPU(gpuStats, sim) {
        const e = gpuStats;
        const angMom = e.orbitalAngMom + e.spinAngMom;
        const pe = e.pe + e.pfiEnergy;
        const allFieldE = e.fieldEnergy + e.higgsFieldEnergy + e.axionFieldEnergy;
        const total = e.linearKE + e.spinKE + pe + allFieldE + sim.totalRadiated;

        const totalFieldPx = e.fieldPx + e.scalarFieldMomX;
        const totalFieldPy = e.fieldPy + e.scalarFieldMomY;
        const totalPx = e.px + totalFieldPx + sim.totalRadiatedPx;
        const totalPy = e.py + totalFieldPy + sim.totalRadiatedPy;
        const pMag = Math.sqrt(totalPx * totalPx + totalPy * totalPy);

        if (this.initialEnergy === null && e.aliveCount > 0) {
            this.initialEnergy = total;
            this.initialMomentum = pMag;
            this.initialAngMom = angMom;
        }

        const eDrift = this.initialEnergy !== null && this.initialEnergy !== 0
            ? ((total - this.initialEnergy) / Math.abs(this.initialEnergy) * 100) : 0;
        const pDrift = this.initialMomentum !== null && this.initialMomentum !== 0
            ? ((pMag - this.initialMomentum) / Math.abs(this.initialMomentum) * 100) : 0;
        const aDrift = this.initialAngMom !== null && this.initialAngMom !== 0
            ? ((angMom - this.initialAngMom) / Math.abs(this.initialAngMom) * 100) : 0;

        _set(this.dom.linearKE, fmt(e.linearKE));
        _set(this.dom.spinKE, fmt(e.spinKE));
        _set(this.dom.massE, fmt(e.totalMass));
        _set(this.dom.potentialE, fmt(pe));
        _set(this.dom.totalE, fmt(total));
        _set(this.dom.energyDrift, fmtDrift(eDrift));
        _set(this.dom.fieldE, fmt(allFieldE));
        _set(this.dom.radiatedE, fmt(sim.totalRadiated));
        _set(this.dom.momentum, fmt(pMag));
        _set(this.dom.particleMom, fmt(Math.sqrt(e.px * e.px + e.py * e.py)));
        _set(this.dom.fieldMom, fmt(Math.sqrt(totalFieldPx * totalFieldPx + totalFieldPy * totalFieldPy)));
        _set(this.dom.radiatedMom, fmt(Math.sqrt(sim.totalRadiatedPx * sim.totalRadiatedPx + sim.totalRadiatedPy * sim.totalRadiatedPy)));
        _set(this.dom.momentumDrift, fmtDrift(pDrift));
        _set(this.dom.angularMomentum, fmt(angMom));
        _set(this.dom.orbitalAngMom, fmt(e.orbitalAngMom));
        _set(this.dom.spinAngMom, fmt(e.spinAngMom));
        _set(this.dom.angMomDrift, fmtDrift(aDrift));
    }

    /**
     * Update selected particle display from GPU readback data.
     * @param {Object} sel - GPU readback selected particle object, or null.
     * @param {Object} physics - for relativityEnabled check.
     * @returns {Object|null}
     */
    updateSelectedGPU(sel, physics) {
        const dom = this.selDom;
        if (!sel || sel.mass <= 0) {
            dom.details.hidden = true;
            dom.hint.hidden = false;
            dom.phaseSection.hidden = true;
            if (dom.effPotSection) dom.effPotSection.hidden = true;
            return null;
        }

        dom.details.hidden = false;
        dom.hint.hidden = true;
        dom.phaseSection.hidden = true; // phase plot not available in GPU mode
        if (dom.effPotSection) dom.effPotSection.hidden = true; // eff potential not available
        const speed = Math.sqrt(sel.velX * sel.velX + sel.velY * sel.velY);
        const wSq = sel.velWX * sel.velWX + sel.velWY * sel.velWY;
        const gamma = physics.relativityEnabled ? Math.sqrt(1 + wSq) : 1;
        const totalFx = sel.forceGravity.x + sel.forceCoulomb.x + sel.forceMagnetic.x + sel.forceGravitomag.x + sel.force1PN.x + sel.forceSpinCurv.x + sel.forceRadiation.x + sel.forceYukawa.x + sel.forceExternal.x + sel.forceHiggs.x + sel.forceAxion.x;
        const totalFy = sel.forceGravity.y + sel.forceCoulomb.y + sel.forceMagnetic.y + sel.forceGravitomag.y + sel.force1PN.y + sel.forceSpinCurv.y + sel.forceRadiation.y + sel.forceYukawa.y + sel.forceExternal.y + sel.forceHiggs.y + sel.forceAxion.y;
        const forceMag = Math.sqrt(totalFx * totalFx + totalFy * totalFy);

        _set(dom.mass, fmtRaw(sel.mass) + (sel.antimatter ? ' (anti)' : ''));
        _set(dom.charge, fmtRaw(sel.charge));
        const surfaceV = sel.angVel * sel.radius;
        _set(dom.spin, surfaceV.toFixed(4) + 'c');
        _set(dom.speed, speed.toFixed(4) + 'c');
        _set(dom.gamma, gamma.toFixed(3));
        _set(dom.force, fmt(forceMag));

        const forces = this._forceDescs;
        forces[0].vec = sel.forceGravity;
        forces[1].vec = sel.forceCoulomb;
        forces[2].vec = sel.forceMagnetic;
        forces[3].vec = sel.forceGravitomag;
        forces[4].vec = sel.force1PN;
        forces[5].vec = sel.forceSpinCurv;
        forces[6].vec = sel.forceRadiation;
        forces[7].vec = sel.forceYukawa;
        forces[8].vec = sel.forceExternal;
        forces[9].vec = sel.forceHiggs;
        forces[10].vec = sel.forceAxion;
        for (const f of forces) {
            if (!f.row) continue;
            const mag = Math.sqrt(f.vec.x * f.vec.x + f.vec.y * f.vec.y);
            if (mag > EPSILON) {
                f.row.hidden = false;
                _set(f.val, fmt(mag));
            } else {
                f.row.hidden = true;
            }
        }

        return sel;
    }

    updateSelected(particle, particles, physics) {
        const p = particle;
        const dom = this.selDom;

        if (!p || p.mass <= 0) {
            dom.details.hidden = true;
            dom.hint.hidden = false;
            dom.phaseSection.hidden = true;
            if (dom.effPotSection) dom.effPotSection.hidden = true;
            return null; // signal to caller to clear selection
        }

        dom.details.hidden = false;
        dom.hint.hidden = true;
        dom.phaseSection.hidden = false;
        if (dom.effPotSection) dom.effPotSection.hidden = false;
        const speed = Math.sqrt(p.vel.x * p.vel.x + p.vel.y * p.vel.y);
        const gamma = physics.relativityEnabled ? Math.sqrt(1 + p.w.magSq()) : 1;
        const totalFx = p.forceGravity.x + p.forceCoulomb.x + p.forceMagnetic.x + p.forceGravitomag.x + p.force1PN.x + p.forceSpinCurv.x + p.forceRadiation.x + p.forceYukawa.x + p.forceExternal.x + p.forceHiggs.x + p.forceAxion.x;
        const totalFy = p.forceGravity.y + p.forceCoulomb.y + p.forceMagnetic.y + p.forceGravitomag.y + p.force1PN.y + p.forceSpinCurv.y + p.forceRadiation.y + p.forceYukawa.y + p.forceExternal.y + p.forceHiggs.y + p.forceAxion.y;
        const forceMag = Math.sqrt(totalFx * totalFx + totalFy * totalFy);

        _set(dom.mass, fmtRaw(p.mass) + (p.antimatter ? ' (anti)' : ''));
        _set(dom.charge, fmtRaw(p.charge));
        const surfaceV = p.angVel * p.radius;
        _set(dom.spin, surfaceV.toFixed(4) + 'c');
        _set(dom.speed, speed.toFixed(4) + 'c');
        _set(dom.gamma, gamma.toFixed(3));
        _set(dom.force, fmt(forceMag));

        // R14: Reuse pre-allocated force descriptor array
        const forces = this._forceDescs;
        forces[0].vec = p.forceGravity;
        forces[1].vec = p.forceCoulomb;
        forces[2].vec = p.forceMagnetic;
        forces[3].vec = p.forceGravitomag;
        forces[4].vec = p.force1PN;
        forces[5].vec = p.forceSpinCurv;
        forces[6].vec = p.forceRadiation;
        forces[7].vec = p.forceYukawa;
        forces[8].vec = p.forceExternal;
        forces[9].vec = p.forceHiggs;
        forces[10].vec = p.forceAxion;
        for (const f of forces) {
            if (!f.row) continue;
            const mag = Math.sqrt(f.vec.x * f.vec.x + f.vec.y * f.vec.y);
            if (mag > EPSILON) {
                f.row.hidden = false;
                _set(f.val, fmt(mag));
            } else {
                f.row.hidden = true;
            }
        }

        return p; // still valid
    }
}
