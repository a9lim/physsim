import Vec2 from './vec2.js';
import { HISTORY_SIZE, kerrNewmanRadius } from './config.js';

const _PAL = window._PALETTE;
const _hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const _nRGB = _hex(_PAL.neutral);
const _bhLightRGB = _hex(_PAL.light.text); // BH base color in light mode
const _bhDarkRGB = _hex(_PAL.dark.text);   // BH base color in dark mode
const _posRGB = _hex(_PAL.extended.red);
const _negRGB = _hex(_PAL.extended.blue);

export default class Particle {
    static nextId = 0;

    constructor(x, y, mass = 10, charge = 0) {
        this.id = Particle.nextId++;
        this.creationTime = -Infinity; // set by addParticle / loadState
        this.deathTime = Infinity;    // set by _retireParticle
        this._deathMass = 0;          // pre-removal mass (for merged particles whose mass is zeroed)
        this._deathAngVel = 0;        // pre-removal angular velocity (for signal delay forces)
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(0, 0);
        this.w = new Vec2(0, 0);    // proper velocity (gamma*v)
        this.force = new Vec2(0, 0);
        this.jerk = new Vec2(0, 0);    // analytical dF/dt for radiation reaction
        this.forceGravity = new Vec2(0, 0);
        this.forceCoulomb = new Vec2(0, 0);
        this.forceMagnetic = new Vec2(0, 0);
        this.forceGravitomag = new Vec2(0, 0);
        this.force1PN = new Vec2(0, 0);
        this.forceSpinCurv = new Vec2(0, 0);
        this.forceRadiation = new Vec2(0, 0);
        this.forceYukawa = new Vec2(0, 0);
        this.forceExternal = new Vec2(0, 0);
        this.forceHiggs = new Vec2(0, 0);
        this.forceAxion = new Vec2(0, 0);
        this.axMod = 1; // local EM coupling modulation from axion field (scalar aF²)
        this.yukMod = 1; // local Yukawa coupling modulation from axion field (pseudoscalar PQ)
        this.higgsMod = 1; // local Higgs VEV |φ(x)| for Yukawa range modulation
        this.torqueSpinOrbit = 0;
        this.torqueFrameDrag = 0;
        this.torqueTidal = 0;
        this.torqueContact = 0;
        this.torqueSuperradiance = 0;
        this._f1pnOld = new Vec2(0, 0);

        this.mass = mass;
        this.baseMass = mass;
        this.charge = charge;
        this.antimatter = false;
        this.angw = 0;      // angular celerity (unbounded)
        this.angVel = 0;    // angular velocity (derived)
        this.magMoment = 0;   // cached: MAG_MOMENT_K * q * ω * r²
        this.angMomentum = 0; // cached: INERTIA_K * m * ω * r²

        // Accumulated B fields (Boris rotation) and gradients (spin-orbit)
        this.Bz = 0;
        this.Bgz = 0;
        this.dBzdx = 0;
        this.dBzdy = 0;
        this.dBgzdx = 0;
        this.dBgzdy = 0;
        this._tidalTorque = 0;
        this._frameDragTorque = 0;
        this._contactTorque = 0;

        // Radiation accumulators (used by integrator)
        this._radAccum = 0;
        this._hawkAccum = 0;
        this._quadAccum = 0;    // GW quadrupole
        this._emQuadAccum = 0;  // EM quadrupole
        this._radDisplayX = 0;
        this._radDisplayY = 0;
        this._yukawaRadAccum = 0; // Yukawa meson radiation accumulator
        this._schwingerAccum = 0; // Schwinger discharge accumulator

        // Signal delay history (lazy-allocated)
        this.histX = null;
        this.histY = null;
        this.histVx = null;
        this.histVy = null;
        this.histAngW = null;
        this.histTime = null;
        this.histHead = 0;
        this.histCount = 0;

        this.radius = Math.cbrt(this.mass);
        this.radiusSq = this.radius * this.radius;
        this.bodyRadiusSq = this.radiusSq; // body radius² (always ∛(mass)², never horizon)
        this.invMass = 1 / this.mass;
        this.color = this.getColor();
    }

    getColor() {
        const bh = window.sim && window.sim.physics.blackHoleEnabled;
        const isLight = document.documentElement.dataset.theme !== 'dark';
        const base = bh ? (isLight ? _bhLightRGB : _bhDarkRGB) : _nRGB;
        const t = Math.max(-1, Math.min(1, this.charge / 5));
        const absT = Math.abs(t);
        if (absT < 1e-6) {
            return bh ? (isLight ? _PAL.light.text : _PAL.dark.text) : _PAL.neutral;
        }
        const tgt = t > 0 ? _posRGB : _negRGB;
        const r = Math.round(base[0] + (tgt[0] - base[0]) * absT);
        const g = Math.round(base[1] + (tgt[1] - base[1]) * absT);
        const b = Math.round(base[2] + (tgt[2] - base[2]) * absT);
        return `rgb(${r},${g},${b})`;
    }

    updateColor() {
        const bodyR = Math.cbrt(this.mass);
        this.bodyRadiusSq = bodyR * bodyR;
        const bh = window.sim && window.sim.physics.blackHoleEnabled;
        if (bh) {
            this.radius = kerrNewmanRadius(this.mass, this.bodyRadiusSq, this.angVel, this.charge);
        } else {
            this.radius = bodyR;
        }
        this.radiusSq = this.radius * this.radius;
        this.invMass = 1 / this.mass;
        this.color = this.getColor();
    }

    _initHistory() {
        if (this.histX) return;
        this.histX = new Float64Array(HISTORY_SIZE);
        this.histY = new Float64Array(HISTORY_SIZE);
        this.histVx = new Float64Array(HISTORY_SIZE);
        this.histVy = new Float64Array(HISTORY_SIZE);
        this.histAngW = new Float64Array(HISTORY_SIZE);
        this.histTime = new Float64Array(HISTORY_SIZE);
        this.histHead = 0;
        this.histCount = 0;
    }
}
