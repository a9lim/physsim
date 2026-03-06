import { computeEnergies } from './energy.js';
import { DISPLAY_SCALE, STATS_THROTTLE_MASK, EPSILON } from './config.js';
const fmt = (v) => { v *= DISPLAY_SCALE; return Math.abs(v) > 999 ? v.toExponential(2) : v.toFixed(2); };
const fmtRaw = (v) => Math.abs(v) > 999 ? v.toExponential(2) : v.toFixed(2);
const fmtDrift = (v) => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';

export default class StatsDisplay {
    constructor(dom, selDom) {
        this.dom = dom;
        this.selDom = selDom;
        this.initialEnergy = null;
        this.initialMomentum = null;
        this.initialAngMom = null;
        this._frameCount = 0;
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

        this.dom.linearKE.textContent = fmt(e.linearKE);
        this.dom.spinKE.textContent = fmt(e.spinKE);
        this.dom.potentialE.textContent = fmt(e.pe);
        this.dom.totalE.textContent = fmt(total);
        this.dom.energyDrift.textContent = fmtDrift(eDrift);
        this.dom.fieldE.textContent = fmt(e.fieldEnergy);
        if (this.dom.higgsFieldE) {
            this.dom.higgsFieldE.textContent = fmt(e.higgsFieldEnergy);
            this.dom.higgsFieldE.closest('.stat-row').hidden = e.higgsFieldEnergy === 0;
        }
        if (this.dom.axionFieldE) {
            this.dom.axionFieldE.textContent = fmt(e.axionFieldEnergy);
            this.dom.axionFieldE.closest('.stat-row').hidden = e.axionFieldEnergy === 0;
        }
        this.dom.radiatedE.textContent = fmt(sim.totalRadiated);
        this.dom.momentum.textContent = fmt(pMag);
        this.dom.particleMom.textContent = fmt(Math.sqrt(e.px * e.px + e.py * e.py));
        this.dom.fieldMom.textContent = fmt(Math.sqrt(e.fieldPx * e.fieldPx + e.fieldPy * e.fieldPy));
        this.dom.radiatedMom.textContent = fmt(Math.sqrt(sim.totalRadiatedPx * sim.totalRadiatedPx + sim.totalRadiatedPy * sim.totalRadiatedPy));
        this.dom.momentumDrift.textContent = fmtDrift(pDrift);
        this.dom.angularMomentum.textContent = fmt(angMom);
        this.dom.orbitalAngMom.textContent = fmt(e.orbitalAngMom);
        this.dom.spinAngMom.textContent = fmt(e.spinAngMom);
        this.dom.angMomDrift.textContent = fmtDrift(aDrift);
    }

    updateSelected(particle, particles, physics) {
        const p = particle;
        const dom = this.selDom;

        if (!p || p.mass <= 0 || !particles.includes(p)) {
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

        dom.mass.textContent = fmtRaw(p.mass) + (p.antimatter ? ' (anti)' : '');
        dom.charge.textContent = fmtRaw(p.charge);
        const surfaceV = p.angVel * p.radius;
        dom.spin.textContent = surfaceV.toFixed(4) + 'c';
        dom.speed.textContent = speed.toFixed(4) + 'c';
        dom.gamma.textContent = gamma.toFixed(3);
        dom.force.textContent = fmt(forceMag);

        // Force breakdown by type
        const forces = [
            { row: dom.fbGravity, val: dom.fbGravityVal, vec: p.forceGravity },
            { row: dom.fbCoulomb, val: dom.fbCoulombVal, vec: p.forceCoulomb },
            { row: dom.fbMagnetic, val: dom.fbMagneticVal, vec: p.forceMagnetic },
            { row: dom.fbGravitomag, val: dom.fbGravitomagVal, vec: p.forceGravitomag },
            { row: dom.fb1pn, val: dom.fb1pnVal, vec: p.force1PN },
            { row: dom.fbSpincurv, val: dom.fbSpincurvVal, vec: p.forceSpinCurv },
            { row: dom.fbRadiation, val: dom.fbRadiationVal, vec: p.forceRadiation },
            { row: dom.fbYukawa, val: dom.fbYukawaVal, vec: p.forceYukawa },
            { row: dom.fbExternal, val: dom.fbExternalVal, vec: p.forceExternal },
            { row: dom.fbHiggs, val: dom.fbHiggsVal, vec: p.forceHiggs },
            { row: dom.fbAxion, val: dom.fbAxionVal, vec: p.forceAxion },
        ];
        for (const f of forces) {
            if (!f.row) continue;
            const mag = Math.sqrt(f.vec.x * f.vec.x + f.vec.y * f.vec.y);
            if (mag > EPSILON) {
                f.row.hidden = false;
                f.val.textContent = fmt(mag);
            } else {
                f.row.hidden = true;
            }
        }

        return p; // still valid
    }
}
