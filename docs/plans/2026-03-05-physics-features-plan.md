# Physics Feature Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 13 new physics features — gravitational wave radiation, Kerr-Newman black holes, photon bending, Roche lobe overflow, reduced BH softening, cosmological expansion, signal-delayed potential heatmap, Yukawa potential, axion coupling, save/load, reference pages, and force equation display — to the No-Hair N-body simulation.

**Architecture:** Features are grouped into 5 phases by dependency. Phase 1 adds new forces/potentials (Yukawa, axion coupling). Phase 2 enhances existing systems (Kerr-Newman BH, reduced BH softening, photon bending). Phase 3 adds gravitational wave quadrupole radiation and Roche lobe overflow. Phase 4 adds cosmological expansion and signal-delayed heatmap. Phase 5 adds UI features (force equations, save/load, reference pages). Each feature follows the existing pattern: config constant → physics flag → force/integrator code → UI toggle → info tip → preset if applicable.

**Tech Stack:** Vanilla JS (ES6 modules), Canvas 2D, KaTeX for math rendering. No dependencies.

**Codebase orientation:** This is a zero-dependency vanilla JS simulation. No test framework, no build step, no linter. "Testing" means serving locally (`python -m http.server` from the parent `a9lim.github.io/` directory) and verifying in the browser. All physics uses natural units: c = 1, G = 1.

---

## Phase 1: New Forces

### Task 1: Yukawa Potential — Config & Physics Flag

**Files:**
- Modify: `src/config.js:1-38`
- Modify: `src/integrator.js:23-37` (Physics constructor flags)
- Modify: `src/integrator.js:50-57` (`_toggles` object)
- Modify: `src/integrator.js:66-73` (`_syncToggles()`)
- Modify: `src/particle.js:10-71` (add `forceYukawa` Vec2)

**Step 1: Add config constants**

In `src/config.js`, add after line 37 (`FRAGMENT_COUNT`):

```js
// Yukawa potential: V(r) = -g²·exp(-μr)/r
export const DEFAULT_YUKAWA_G2 = 1.0;   // coupling strength
export const DEFAULT_YUKAWA_MU = 0.2;   // mediator mass (inverse range)
```

**Step 2: Add physics flags and toggle sync**

In `src/integrator.js` constructor (around line 36), add:
```js
this.yukawaEnabled = false;
this.yukawaG2 = DEFAULT_YUKAWA_G2;
this.yukawaMu = DEFAULT_YUKAWA_MU;
```

Import the new constants at line 6.

In `_toggles` object (line 50-57), add:
```js
yukawaEnabled: false,
yukawaG2: 1.0,
yukawaMu: 0.2,
```

In `_syncToggles()` (line 66-73), add:
```js
this._toggles.yukawaEnabled = this.yukawaEnabled;
this._toggles.yukawaG2 = this.yukawaG2;
this._toggles.yukawaMu = this.yukawaMu;
```

**Step 3: Add forceYukawa Vec2 to Particle**

In `src/particle.js`, add after `forceRadiation` declaration (line 26):
```js
this.forceYukawa = new Vec2(0, 0);
```

**Step 4: Commit**
```
feat: add Yukawa potential config and physics flags
```

---

### Task 2: Yukawa Force Computation

**Files:**
- Modify: `src/forces.js:13-37` (`resetForces()` — add `forceYukawa` reset)
- Modify: `src/forces.js:134-301` (`pairForce()` — add Yukawa force block)

**Step 1: Reset forceYukawa in resetForces()**

In `src/forces.js` `resetForces()`, add after `p.forceRadiation.set(0, 0);` (line 24):
```js
p.forceYukawa.set(0, 0);
```

**Step 2: Add Yukawa force to pairForce()**

In `src/forces.js` `pairForce()`, add after the tidal locking block (after line 301):

```js
if (toggles.yukawaEnabled) {
    const g2 = toggles.yukawaG2;
    const mu = toggles.yukawaMu;
    const r = 1 / invR;
    const expMuR = Math.exp(-mu * r);
    // F = g² · exp(-μr) · (1/r² + μ/r) · r̂  (attractive, like gravity)
    const fDir = g2 * p.mass * sMass * expMuR * (invRSq + mu * invR) * invR;
    out.x += rx * fDir;
    out.y += ry * fDir;
    p.forceYukawa.x += rx * fDir;
    p.forceYukawa.y += ry * fDir;
    // Analytical jerk for radiation reaction
    // dF/dt = g²·m₁m₂·exp(-μr)·[v_rel·(1/r+μ)/r² - r·(r·v_rel)·(3/r+2μ+μ²r)/r⁴]
    const jBase = g2 * p.mass * sMass * expMuR;
    const term1 = (invRSq + mu * invR) * invR;
    const jRadial = -(3 * invRSq + 2 * mu * invR + mu * mu) * rDotVr * expMuR * g2 * p.mass * sMass * invRSq * invR;
    p.jerk.x += vrx * jBase * term1 + rx * jRadial;
    p.jerk.y += vry * jBase * term1 + ry * jRadial;
}
```

**Step 3: Commit**
```
feat: implement Yukawa force computation with analytical jerk
```

---

### Task 3: Yukawa Potential Energy

**Files:**
- Modify: `src/potential.js:88-141` (`pairPE()` — add Yukawa PE term)

**Step 1: Add Yukawa PE**

In `src/potential.js` `pairPE()`, add after the 1PN PE block (before `return pe;` at line 140):

```js
if (toggles.yukawaEnabled) {
    const r = 1 / invR;
    pe -= toggles.yukawaG2 * p.mass * sMass * Math.exp(-toggles.yukawaMu * r) * invR;
}
```

**Step 2: Commit**
```
feat: add Yukawa potential energy term
```

---

### Task 4: Yukawa UI (Toggle + Sliders + Info Tip)

**Files:**
- Modify: `index.html:199-237` (add Yukawa toggle and sliders in Settings → Forces section)
- Modify: `src/ui.js:105-125` (add Yukawa toggle binding)
- Modify: `src/ui.js:348-369` (add Yukawa info tip)

**Step 1: Add HTML toggle and sliders**

In `index.html`, after the Magnetic toggle row (line 212), add a new section:

```html
<div class="ctrl-row"><label for="yukawa-toggle">Yukawa <button class="info-trigger" data-info="yukawa" aria-label="Yukawa info">?</button></label>
    <div class="tog-wrap"><input type="checkbox" id="yukawa-toggle" role="switch" aria-checked="false">
    <label for="yukawa-toggle" class="tog tog-yukawa"><span class="tog-thumb"></span></label></div></div>
<div class="ctrl-row ctrl-sub" id="yukawa-sliders" style="display:none">
    <div class="control-group">
        <label>Coupling g² <span class="slider-value" id="yukawaG2Value">1.00</span></label>
        <input type="range" id="yukawaG2Input" min="0.1" max="10" step="0.1" value="1">
    </div>
    <div class="control-group">
        <label>Range 1/μ <span class="slider-value" id="yukawaMuValue">5.00</span></label>
        <input type="range" id="yukawaMuInput" min="0.5" max="50" step="0.5" value="5">
    </div>
</div>
```

**Step 2: Wire toggle in ui.js**

In `src/ui.js` `forceToggles` array (line 105-118), add:
```js
{ id: 'yukawa-toggle', prop: 'yukawaEnabled' },
```

After the slider listeners (around line 277), add:
```js
const yukawaToggle = document.getElementById('yukawa-toggle');
const yukawaSliders = document.getElementById('yukawa-sliders');
const yukawaG2Slider = document.getElementById('yukawaG2Input');
const yukawaG2Label = document.getElementById('yukawaG2Value');
const yukawaMuSlider = document.getElementById('yukawaMuInput');
const yukawaMuLabel = document.getElementById('yukawaMuValue');

yukawaToggle.addEventListener('change', () => {
    yukawaSliders.style.display = yukawaToggle.checked ? '' : 'none';
});
yukawaG2Slider.addEventListener('input', () => {
    sim.physics.yukawaG2 = parseFloat(yukawaG2Slider.value);
    yukawaG2Label.textContent = parseFloat(yukawaG2Slider.value).toFixed(2);
});
yukawaMuSlider.addEventListener('input', () => {
    const range = parseFloat(yukawaMuSlider.value);
    sim.physics.yukawaMu = 1 / range;
    yukawaMuLabel.textContent = range.toFixed(2);
});
```

**Step 3: Add info tip**

In `src/ui.js` `infoData` object (around line 368), add:
```js
yukawa: { title: 'Yukawa Potential', body: 'A screened potential $V(r) = -g^2 e^{-\\mu r}/r$ that falls off exponentially beyond range $1/\\mu$. Models short-range nuclear forces (pion exchange) and any interaction mediated by a massive particle. At short range it behaves like gravity; at long range it vanishes. The coupling $g^2$ sets the strength and $\\mu$ (the mediator mass) sets the range.' },
```

**Step 4: Add Yukawa to renderer force components**

In `src/renderer.js`, add to the `_forceCompColors` object (around line 10-21):
```js
yukawa: _PAL.extended.brown,
```

In `drawForceComponentVectors()` (around line 291-298), add to the `forces` array:
```js
{ key: 'forceYukawa', color: _forceCompColors.yukawa },
```

In `drawForceVectors()` (around line 281-282), add `p.forceYukawa.x` and `p.forceYukawa.y` to the total force sum.

**Step 5: Add Yukawa to presets TOGGLE_MAP and TOGGLE_ORDER**

In `src/presets.js`, add to `TOGGLE_MAP` (line 246-260):
```js
yukawa: 'yukawa-toggle',
```

Add `'yukawa'` to `TOGGLE_ORDER` (line 263-268) at the end.

Add `yukawa: false` to every existing preset's `toggles` object.

**Step 6: Commit**
```
feat: add Yukawa potential UI, rendering, and preset integration
```

---

### Task 5: Axion Coupling — Oscillating α Modulation

**Files:**
- Modify: `src/config.js` (add axion defaults)
- Modify: `src/integrator.js` (add axion flags, apply modulation)
- Modify: `src/forces.js:175-186` (apply coupling modulation to Coulomb)
- Modify: `index.html` (add toggle + sliders)
- Modify: `src/ui.js` (wire toggle, info tip)
- Modify: `src/presets.js` (add to toggle maps)

**Step 1: Config and physics flags**

In `src/config.js`, add:
```js
// Axion dark matter: oscillating EM coupling α_eff = α·(1 + g·cos(m_a·t))
export const DEFAULT_AXION_G = 0.1;     // coupling amplitude
export const DEFAULT_AXION_MASS = 0.5;  // oscillation frequency (m_a)
```

In `src/integrator.js` constructor, add:
```js
this.axionEnabled = false;
this.axionG = DEFAULT_AXION_G;
this.axionMass = DEFAULT_AXION_MASS;
```

In `_toggles`, add:
```js
axionEnabled: false,
axionModulation: 1.0,
```

In `_syncToggles()`, add:
```js
this._toggles.axionEnabled = this.axionEnabled;
if (this.axionEnabled) {
    this._toggles.axionModulation = 1 + this.axionG * Math.cos(this.axionMass * this.simTime);
} else {
    this._toggles.axionModulation = 1.0;
}
```

**Step 2: Apply modulation to Coulomb force**

In `src/forces.js` `pairForce()`, modify the Coulomb block (line 175-186). Change:
```js
const k = -(p.charge * sCharge);
```
to:
```js
const k = -(p.charge * sCharge) * toggles.axionModulation;
```

Similarly in the magnetic dipole block (line 240-259), the Bz/gradient computations, and the Coulomb PE in `potential.js` — multiply all `charge * charge` terms by `toggles.axionModulation`.

**Step 3: Add HTML toggle and sliders**

In `index.html`, after the Yukawa section, add:

```html
<div class="ctrl-row"><label for="axion-toggle">Axion <button class="info-trigger" data-info="axion" aria-label="Axion info">?</button></label>
    <div class="tog-wrap"><input type="checkbox" id="axion-toggle" role="switch" aria-checked="false">
    <label for="axion-toggle" class="tog tog-axion"><span class="tog-thumb"></span></label></div></div>
<div class="ctrl-row ctrl-sub" id="axion-sliders" style="display:none">
    <div class="control-group">
        <label>Coupling g <span class="slider-value" id="axionGValue">0.10</span></label>
        <input type="range" id="axionGInput" min="0.01" max="0.5" step="0.01" value="0.1">
    </div>
    <div class="control-group">
        <label>Axion Mass m<sub>a</sub> <span class="slider-value" id="axionMassValue">0.50</span></label>
        <input type="range" id="axionMassInput" min="0.01" max="5" step="0.01" value="0.5">
    </div>
</div>
```

**Step 4: Wire toggle and add info tip**

In `src/ui.js`, bind the axion toggle similarly to Yukawa. Add dependency: axion requires Coulomb.

Info tip:
```js
axion: { title: 'Axion Coupling', body: 'Models dark matter axions oscillating as a background field $a(t) = a_0 \\cos(m_a t)$, which modulates the electromagnetic coupling: $\\alpha_{\\text{eff}} = \\alpha(1 + g\\cos(m_a t))$. This makes Coulomb and magnetic forces oscillate periodically. The effect is the exact phenomenon that axion detection experiments (CASPEr, ABRACADABRA) search for. Energy is not conserved — the axion field is an external reservoir.' },
```

**Step 5: Add to preset maps**

Add `axion: 'axion-toggle'` to `TOGGLE_MAP`, `'axion'` to `TOGGLE_ORDER`, and `axion: false` to all preset `toggles`.

**Step 6: Commit**
```
feat: add axion dark matter coupling (oscillating α modulation)
```

---

## Phase 2: Enhanced Existing Systems

### Task 6: Kerr-Newman Black Holes

**Files:**
- Modify: `src/particle.js:85-91` (`updateColor()` — Kerr-Newman radius)
- Modify: `src/integrator.js:474-500` (Hawking radiation — Kerr-Newman temperature)
- Modify: `src/renderer.js:192-219` (`drawParticles()` — ergosphere ring)
- Modify: `src/ui.js:367` (update BH info tip)

**Step 1: Kerr-Newman event horizon radius**

In `src/particle.js` `updateColor()`, replace the BH radius line (line 87):
```js
this.radius = bh ? 2 * this.mass : Math.cbrt(this.mass);
```
with:
```js
if (bh) {
    const M = this.mass;
    const I = INERTIA_K * M * this.radiusSq;
    // Use previous radius for I computation; iterate once
    const omega = this.angVel || 0;
    const a = I * Math.abs(omega) / M;  // spin parameter J/M
    const Q = this.charge;
    const disc = M * M - a * a - Q * Q;
    this.radius = disc > 0 ? M + Math.sqrt(disc) : M * 0.5; // naked singularity floor
} else {
    this.radius = Math.cbrt(this.mass);
}
```

Import `INERTIA_K` at the top of `particle.js`.

Note: The radius computation uses the previous `radiusSq` because the new radius depends on `I` which depends on the old radius. This is a self-consistent iterative approach — one step is sufficient since the radius changes slowly.

**Step 2: Kerr-Newman Hawking radiation**

In `src/integrator.js`, replace the Hawking power formula (line 478):
```js
const power = 1 / (15360 * Math.PI * p.mass * p.mass);
```
with:
```js
const M = p.mass;
const I = INERTIA_K * M * p.radiusSq;
const a = I * Math.abs(p.angVel) / M;
const Q = p.charge;
const disc = M * M - a * a - Q * Q;
let power;
if (disc > 1e-10) {
    const rPlus = M + Math.sqrt(disc);
    const kappa = Math.sqrt(disc) / (2 * M * rPlus); // surface gravity
    const T = kappa / (2 * Math.PI);
    const A = 4 * Math.PI * (rPlus * rPlus + a * a);  // horizon area
    // Calibrate σ so that at a=Q=0, power matches 1/(15360πM²)
    // At a=Q=0: κ=1/(4M), T=1/(8πM), A=64πM², P=σ·T⁴·A = σ/(4096π³M⁴)·64πM² = σ/(64π²M²)
    // Want P = 1/(15360πM²), so σ = 64π²/(15360π) = 4π/960 = π/240
    const sigma = Math.PI / 240;
    power = sigma * T * T * T * T * A;
} else {
    power = 0; // extremal: no radiation
}
```

**Step 3: Ergosphere ring in renderer**

In `src/renderer.js` `drawParticles()`, after drawing the particle fill and spin ring (around line 218), add:

```js
// Ergosphere ring for BH mode
if (window.sim && window.sim.physics.blackHoleEnabled && p.mass > 0) {
    const M = p.mass;
    const I = INERTIA_K * M * p.radiusSq;
    const a = I * Math.abs(p.angVel) / M;
    const rErgo = M + Math.sqrt(Math.max(0, M * M - a * a));
    if (rErgo > p.radius + 0.3) {
        ctx.beginPath();
        ctx.arc(p.pos.x, p.pos.y, rErgo, 0, TWO_PI);
        ctx.strokeStyle = isLight ? _r(_PAL.extended.purple, 0.3) : _r(_PAL.extended.purple, 0.4);
        ctx.lineWidth = 0.15;
        ctx.setLineDash([0.3, 0.3]);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}
```

Import `INERTIA_K` from config at the top of `renderer.js`.

**Step 4: Update info tip**

Update the `blackhole` info tip in `src/ui.js` to mention Kerr-Newman:
```js
blackhole: { title: 'Black Hole Mode', body: 'All particles become black holes with Kerr\u2013Newman horizons: $r_+ = M + \\sqrt{M^2 - a^2 - Q^2}$ where $a = J/M$ (spin) and $Q$ is charge. Spinning or charged black holes have smaller horizons and a surrounding ergosphere (dashed ring). Hawking radiation power depends on surface gravity: extremal black holes ($M^2 = a^2 + Q^2$) have zero temperature and stop radiating. Requires Relativity.' },
```

**Step 5: Commit**
```
feat: implement Kerr-Newman black holes with ergosphere visualization
```

---

### Task 7: Reduced Softening in BH Mode

**Files:**
- Modify: `src/forces.js:143-148` (use reduced softening when BH mode on)
- Modify: `src/potential.js:97` (same)

**Step 1: Pass BH flag through to force computation**

The simplest approach: check `window.sim.physics.blackHoleEnabled` in `pairForce()` (it's already used in `particle.js`). Replace the softening computation in `pairForce()` (line 144):

```js
const rSq = rawRSq + SOFTENING_SQ;
```
with:
```js
const bhSoft = (window.sim && window.sim.physics.blackHoleEnabled) ? 1 : SOFTENING_SQ;
const rSq = rawRSq + bhSoft;
```

Do the same in `pairPE()` in `potential.js` (line 97).

Note: SOFTENING_SQ = 64 normally. In BH mode, radius = 2M (much larger than cbrt(M) for M > ~0.125), so the BH radius itself provides physical softening. We reduce to 1 (still nonzero to prevent division by zero).

**Step 2: Commit**
```
feat: reduce Plummer softening in black hole mode for realistic dynamics
```

---

### Task 8: Photon Bending (Gravitational Lensing)

**Files:**
- Modify: `src/photon.js:13-18` (`update()` — add gravitational deflection)
- Modify: `main.js:170-178` (pass particles to photon update)
- Modify: `src/config.js` (add PHOTON_SOFTENING_SQ)

**Step 1: Add photon softening constant**

In `src/config.js`, add:
```js
export const PHOTON_SOFTENING_SQ = 4;  // smaller than particle softening for tighter lensing
```

**Step 2: Modify Photon.update() to accept and deflect from particles**

In `src/photon.js`:

```js
import Vec2 from './vec2.js';
import { PHOTON_SOFTENING_SQ } from './config.js';

export default class Photon {
    constructor(x, y, vx, vy, energy, emitterId = -1) {
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(vx, vy);
        this.energy = energy;
        this.lifetime = 0;
        this.alive = true;
        this.emitterId = emitterId;
        this.age = 0;
    }

    update(dt, particles) {
        // Gravitational deflection: GR gives 2× Newtonian (null geodesic)
        if (particles) {
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                const dx = p.pos.x - this.pos.x;
                const dy = p.pos.y - this.pos.y;
                const rSq = dx * dx + dy * dy + PHOTON_SOFTENING_SQ;
                const invR3 = 1 / (rSq * Math.sqrt(rSq));
                this.vel.x += 2 * p.mass * dx * invR3 * dt;
                this.vel.y += 2 * p.mass * dy * invR3 * dt;
            }
            // Renormalize to c = 1
            const v = Math.sqrt(this.vel.x * this.vel.x + this.vel.y * this.vel.y);
            if (v > 1e-10) {
                this.vel.x /= v;
                this.vel.y /= v;
            }
        }

        this.pos.x += this.vel.x * dt;
        this.pos.y += this.vel.y * dt;
        this.lifetime += dt;
    }
}
```

**Step 3: Pass particles to photon update in main.js**

In `main.js` line 173, change:
```js
ph.update(PHYSICS_DT);
```
to:
```js
ph.update(PHYSICS_DT, this.particles);
```

**Step 4: Commit**
```
feat: implement photon gravitational lensing (2× Newtonian deflection)
```

---

## Phase 3: Radiation & Mass Transfer

### Task 9: Gravitational Quadrupole Radiation

This is the most complex single feature. It requires computing the mass quadrupole moment tensor, its 3rd time derivative, and emitting "graviton" particles.

**Files:**
- Modify: `src/photon.js` (add `type` field: `'em'` or `'gw'`)
- Modify: `src/integrator.js` (compute quadrupole radiation, emit gravitons, orbital decay kick)
- Modify: `src/renderer.js:402-423` (`drawPhotons()` — color gravitons differently)
- Modify: `src/config.js` (add GW constants)
- Modify: `index.html` (add GW radiation toggle)
- Modify: `src/ui.js` (wire toggle, info tip)
- Modify: `src/presets.js` (add to maps)

**Step 1: Extend Photon with type field**

In `src/photon.js` constructor, add:
```js
this.type = 'em'; // 'em' for electromagnetic, 'gw' for gravitational wave
```

**Step 2: Add GW config**

In `src/config.js`:
```js
// Gravitational wave radiation (quadrupole formula)
export const GW_QUADRUPOLE_K = 32 / 5;  // Peters formula coefficient
```

**Step 3: Add physics flag and quadrupole history**

In `src/integrator.js` constructor:
```js
this.gwRadiationEnabled = false;
// Quadrupole moment tensor history (for 3rd time derivative)
this._quadHistory = []; // [{Ixx, Ixy, Iyy, t}]
this._gwAccum = 0;      // accumulated GW energy for graviton emission
```

**Step 4: Compute quadrupole radiation after substep loop**

In `src/integrator.js`, after the substep loop but before boundary handling (around line 582, after the signal delay history recording), add:

```js
// Gravitational wave radiation (quadrupole formula)
if (this.gwRadiationEnabled && n >= 2 && this.sim) {
    // Compute reduced mass quadrupole moment: I_ij = Σ m·(x_i·x_j - δ_ij·r²/3)
    // In 2D: only Ixx, Ixy, Iyy matter
    let Ixx = 0, Ixy = 0, Iyy = 0;
    for (let i = 0; i < n; i++) {
        const p = particles[i];
        Ixx += p.mass * p.pos.x * p.pos.x;
        Ixy += p.mass * p.pos.x * p.pos.y;
        Iyy += p.mass * p.pos.y * p.pos.y;
    }
    // Store in history
    this._quadHistory.push({ Ixx, Ixy, Iyy, t: this.simTime });
    if (this._quadHistory.length > 5) this._quadHistory.shift();

    // Need at least 4 samples for 3rd derivative (finite differences)
    if (this._quadHistory.length >= 4) {
        const h = this._quadHistory;
        const n3 = h.length - 1;
        // 3rd derivative via finite differences (4-point backward)
        const dt1 = h[n3].t - h[n3 - 1].t;
        const dt2 = h[n3 - 1].t - h[n3 - 2].t;
        const dt3 = h[n3 - 2].t - h[n3 - 3].t;
        if (dt1 > 1e-15 && dt2 > 1e-15 && dt3 > 1e-15) {
            // 2nd derivatives at two points, then difference for 3rd
            const d2Ixx_a = (h[n3].Ixx - 2 * h[n3 - 1].Ixx + h[n3 - 2].Ixx) / (dt1 * dt2);
            const d2Ixx_b = (h[n3 - 1].Ixx - 2 * h[n3 - 2].Ixx + h[n3 - 3].Ixx) / (dt2 * dt3);
            const d3Ixx = (d2Ixx_a - d2Ixx_b) / ((dt1 + dt2) * 0.5);
            const d2Ixy_a = (h[n3].Ixy - 2 * h[n3 - 1].Ixy + h[n3 - 2].Ixy) / (dt1 * dt2);
            const d2Ixy_b = (h[n3 - 1].Ixy - 2 * h[n3 - 2].Ixy + h[n3 - 3].Ixy) / (dt2 * dt3);
            const d3Ixy = (d2Ixy_a - d2Ixy_b) / ((dt1 + dt2) * 0.5);
            const d2Iyy_a = (h[n3].Iyy - 2 * h[n3 - 1].Iyy + h[n3 - 2].Iyy) / (dt1 * dt2);
            const d2Iyy_b = (h[n3 - 1].Iyy - 2 * h[n3 - 2].Iyy + h[n3 - 3].Iyy) / (dt2 * dt3);
            const d3Iyy = (d2Ixx_a - d2Ixx_b) / ((dt1 + dt2) * 0.5);

            // P_GW = (1/5)·(d³I_ij/dt³)²  (sum over i,j)
            const power = 0.2 * (d3Ixx * d3Ixx + 2 * d3Ixy * d3Ixy + d3Iyy * d3Iyy);

            if (power > 0) {
                const dE = power * dt;
                this.sim.totalRadiated += dE;
                this._gwAccum += dE;

                // Apply orbital decay: reduce each pair's binding energy proportionally
                // Simplified: apply a small inward radial kick to each particle
                // toward the center of mass, proportional to its contribution to the quadrupole
                if (dE > 1e-10) {
                    let comX = 0, comY = 0, totalM = 0;
                    for (let i = 0; i < n; i++) {
                        comX += particles[i].mass * particles[i].pos.x;
                        comY += particles[i].mass * particles[i].pos.y;
                        totalM += particles[i].mass;
                    }
                    comX /= totalM; comY /= totalM;
                    for (let i = 0; i < n; i++) {
                        const p = particles[i];
                        const dx = comX - p.pos.x, dy = comY - p.pos.y;
                        const r = Math.sqrt(dx * dx + dy * dy);
                        if (r > 1e-10) {
                            // Radial kick proportional to m·r (quadrupole weighting)
                            const kick = dE * p.mass * r / (totalM * r * r) * dt;
                            p.w.x += kick * dx / r;
                            p.w.y += kick * dy / r;
                        }
                    }
                }

                // Emit graviton when accumulated energy exceeds threshold
                if (this._gwAccum >= MIN_MASS && this.sim.photons.length < MAX_PHOTONS) {
                    const angle = Math.random() * 6.283185307;
                    const cosA = Math.cos(angle), sinA = Math.sin(angle);
                    // Emit from system COM
                    let gComX = 0, gComY = 0, gTotalM = 0;
                    for (let i = 0; i < n; i++) {
                        gComX += particles[i].mass * particles[i].pos.x;
                        gComY += particles[i].mass * particles[i].pos.y;
                        gTotalM += particles[i].mass;
                    }
                    gComX /= gTotalM; gComY /= gTotalM;
                    const gph = new Photon(gComX + cosA * 3, gComY + sinA * 3, cosA, sinA, this._gwAccum, -1);
                    gph.type = 'gw';
                    this.sim.photons.push(gph);
                    this.sim.totalRadiatedPx += this._gwAccum * cosA;
                    this.sim.totalRadiatedPy += this._gwAccum * sinA;
                    this._gwAccum = 0;
                }
            }
        }
    }
}
```

**Step 5: Color gravitons in renderer**

In `src/renderer.js` `drawPhotons()`, change the fillStyle line:
```js
ctx.fillStyle = _PAL.extended.yellow;
```
to:
```js
ctx.fillStyle = ph.type === 'gw' ? _PAL.extended.green : _PAL.extended.yellow;
```

And change the glow color similarly:
```js
ctx.shadowColor = ph.type === 'gw' ? '#50987880' : '#FFDC6480';
```

**Step 6: Add UI toggle and info tip**

In `index.html`, add after the Radiation toggle:
```html
<div class="ctrl-row ctrl-sub"><label for="gwradiation-toggle">GW Radiation <button class="info-trigger" data-info="gwradiation" aria-label="GW radiation info">?</button></label>
    <div class="tog-wrap"><input type="checkbox" id="gwradiation-toggle" role="switch" aria-checked="false">
    <label for="gwradiation-toggle" class="tog tog-gwradiation"><span class="tog-thumb"></span></label></div></div>
```

Wire in `ui.js` with dependency on Gravity. Add info tip:
```js
gwradiation: { title: 'GW Radiation', body: 'Gravitational wave emission from the mass quadrupole moment: $P = \\frac{1}{5}|\\dddot{I}_{ij}|^2$. For circular binaries this gives $P = \\frac{32}{5} \\frac{m_1^2 m_2^2(m_1+m_2)}{r^5}$, causing orbital inspiral and merger — exactly what LIGO detects. Emitted gravitons (green) carry energy and momentum away from the system. Requires Gravity.' },
```

**Step 7: Add to preset maps**

Add `gwradiation: 'gwradiation-toggle'` to `TOGGLE_MAP`, to `TOGGLE_ORDER`, and `gwradiation: false` to all presets.

**Step 8: Commit**
```
feat: implement gravitational wave quadrupole radiation with graviton emission
```

---

### Task 10: EM Quadrupole Radiation

**Files:**
- Modify: `src/integrator.js` (compute charge quadrupole, add to existing radiation)

**Step 1: Add EM quadrupole to the GW radiation block**

The EM quadrupole power is `P = (1/180) · |d³Q_ij/dt³|²` where `Q_ij = Σ q·x_i·x_j`.

Extend the quadrupole history to also store charge quadrupole components (`Qxx`, `Qxy`, `Qyy`). Compute EM quadrupole power alongside GW power. Gate behind `radiationEnabled && coulombEnabled`. EM quadrupole photons use `type: 'em'`.

Follow the same finite difference pattern as the mass quadrupole. The EM quadrupole adds to `_radAccum` per-system (not per-particle), emitting from the charge COM.

**Step 2: Commit**
```
feat: add electromagnetic quadrupole radiation
```

---

### Task 11: Roche Lobe Overflow

**Files:**
- Modify: `src/integrator.js:707-775` (`checkTidalBreakup()` — add Roche overflow before fragmentation)
- Modify: `src/config.js` (add Roche constants)
- Modify: `main.js:180-208` (handle mass transfer particles from Roche overflow)

**Step 1: Add Roche config**

In `src/config.js`:
```js
// Roche lobe overflow
export const ROCHE_THRESHOLD = 0.9;       // overflow starts at this fraction of Roche radius
export const ROCHE_TRANSFER_RATE = 0.01;  // mass transfer rate coefficient
export const ROCHE_MIN_PACKET = 0.02;     // minimum mass for a stream particle
```

**Step 2: Add Roche overflow check in checkTidalBreakup()**

In `src/integrator.js` `checkTidalBreakup()`, before the violent fragmentation check (around line 769), add Roche lobe overflow detection:

```js
// Roche lobe overflow: continuous mass transfer when radius > Roche lobe
// Eggleton formula: r_Roche ≈ 0.462 · d · (m/(m+M))^(1/3)
let strongestTidalIdx = -1;
let strongestDist = 0;

// (find the neighbor causing the strongest tidal force — reuse the maxTidal loop)
// After finding maxTidal, also record the index and distance of that neighbor.

// If the particle fills its Roche lobe but isn't violently disrupted:
if (strongestTidalIdx >= 0 && maxTidal + centrifugal + coulombSelf <= selfGravity) {
    const other = particles[strongestTidalIdx];
    const d = strongestDist;
    const q = p.mass / (p.mass + other.mass);
    const rRoche = 0.462 * d * Math.cbrt(q);
    if (p.radius > rRoche * ROCHE_THRESHOLD && p.mass > ROCHE_MIN_PACKET * 4) {
        // Compute L1 direction (toward companion)
        let l1x = other.pos.x - p.pos.x, l1y = other.pos.y - p.pos.y;
        // ... minImage if periodic ...
        const l1Mag = Math.sqrt(l1x * l1x + l1y * l1y);
        l1x /= l1Mag; l1y /= l1Mag;
        // Mass transfer rate scales with overflow
        const overflow = p.radius / rRoche - ROCHE_THRESHOLD;
        const dM = Math.min(overflow * ROCHE_TRANSFER_RATE * p.mass, p.mass * 0.1);
        if (dM >= ROCHE_MIN_PACKET) {
            // Record transfer event: {source, target direction, mass, position}
            rocheTransfers.push({
                source: p,
                l1x, l1y,
                mass: dM,
                charge: dM * p.charge / p.mass,
                spawnX: p.pos.x + l1x * p.radius * 1.2,
                spawnY: p.pos.y + l1y * p.radius * 1.2,
                // Tangential velocity from orbital motion
                vx: p.vel.x + (-l1y) * Math.sqrt(other.mass / d) * 0.5,
                vy: p.vel.y + l1x * Math.sqrt(other.mass / d) * 0.5,
            });
        }
    }
}
```

Return both `fragments` and `rocheTransfers` from `checkTidalBreakup()`.

**Step 3: Handle Roche transfers in main.js**

In `main.js`, after the tidal breakup handling, process Roche transfers:

```js
const { fragments: toFragment, transfers: rocheTransfers } = this.physics.checkTidalBreakup(...);
// Handle Roche transfers
for (const t of rocheTransfers) {
    t.source.mass -= t.mass;
    t.source.charge -= t.charge;
    t.source.updateColor();
    this.addParticle(t.spawnX, t.spawnY, t.vx, t.vy, {
        mass: t.mass, charge: t.charge, spin: 0,
    });
}
```

**Step 4: Commit**
```
feat: implement Roche lobe overflow with continuous mass transfer
```

---

## Phase 4: Cosmology & Signal-Delayed Potential

### Task 12: Cosmological Expansion

**Files:**
- Modify: `src/config.js` (add expansion defaults)
- Modify: `src/integrator.js` (add expansion flag, apply Hubble flow + drag in drift step)
- Modify: `index.html` (add toggle + slider in Engine tab)
- Modify: `src/ui.js` (wire toggle, info tip)
- Modify: `src/presets.js` (add to maps)

**Step 1: Config and physics flags**

In `src/config.js`:
```js
export const DEFAULT_HUBBLE = 0.001;  // Hubble parameter
```

In `src/integrator.js` constructor:
```js
this.expansionEnabled = false;
this.hubbleParam = DEFAULT_HUBBLE;
```

**Step 2: Apply Hubble flow + drag in drift step**

In `src/integrator.js`, after the drift step (line 510-512), inside the substep loop, add:

```js
// Cosmological expansion: Hubble flow + drag
if (this.expansionEnabled) {
    const H = this.hubbleParam;
    const cx = this.domainW * 0.5, cy = this.domainH * 0.5;
    for (let i = 0; i < n; i++) {
        const p = particles[i];
        // Hubble flow: v_H = H · r (from domain center)
        p.pos.x += H * (p.pos.x - cx) * dtSub;
        p.pos.y += H * (p.pos.y - cy) * dtSub;
        // Hubble drag: peculiar velocity redshifts
        const decay = 1 - H * dtSub;
        p.w.x *= decay;
        p.w.y *= decay;
    }
}
```

**Step 3: Add UI**

In `index.html` Engine tab, after the Disintegration toggle (line 276), add:

```html
<div class="control-group">
    <label class="checkbox-label"><span>Expansion <button class="info-trigger" data-info="expansion" aria-label="Expansion info">?</button></span> <input type="checkbox" id="expansion-toggle"></label>
</div>
<div class="control-group" id="hubble-group" style="display:none">
    <label>Hubble H <span class="slider-value" id="hubbleValue">0.001</span></label>
    <input type="range" id="hubbleInput" min="0.0001" max="0.01" step="0.0001" value="0.001">
</div>
```

**Step 4: Wire toggle and info tip**

In `src/ui.js`:
```js
const expansionEl = document.getElementById('expansion-toggle');
const hubbleGroup = document.getElementById('hubble-group');
const hubbleSlider = document.getElementById('hubbleInput');
const hubbleLabel = document.getElementById('hubbleValue');

expansionEl.addEventListener('change', () => {
    sim.physics.expansionEnabled = expansionEl.checked;
    hubbleGroup.style.display = expansionEl.checked ? '' : 'none';
});
hubbleSlider.addEventListener('input', () => {
    sim.physics.hubbleParam = parseFloat(hubbleSlider.value);
    hubbleLabel.textContent = parseFloat(hubbleSlider.value).toFixed(4);
});
```

Info tip:
```js
expansion: { title: 'Cosmological Expansion', body: 'Adds Hubble flow $v_H = H \\cdot r$ from the domain center, causing distant particles to separate. Bound systems (where binding energy exceeds Hubble kinetic energy) resist expansion and stay together, while unbound particles drift apart — the mechanism that creates large-scale cosmic structure. Includes Hubble drag ($v_{\\text{pec}} \\propto 1/a$) to redshift peculiar velocities, matching the physics of real cosmological N-body simulations.' },
```

**Step 5: Add to preset maps**

Add `expansion: 'expansion-toggle'` to `TOGGLE_MAP` and `TOGGLE_ORDER` (independent). Add `expansion: false` to all presets.

**Step 6: Commit**
```
feat: add cosmological expansion with Hubble flow and drag
```

---

### Task 13: Signal-Delayed Potential Heatmap

**Files:**
- Modify: `src/heatmap.js:69-109` (`update()` — use delayed source positions)
- Modify: `src/heatmap.js:1-5` (import signal delay)

**Step 1: Import getDelayedState and pass delay parameters**

In `src/heatmap.js`, add import:
```js
import { getDelayedState } from './signal-delay.js';
```

Change the `update()` signature to accept signal delay parameters:
```js
update(particles, camera, width, height, pool, root, barnesHutEnabled, signalDelayEnabled, relativityEnabled, simTime, periodic, domW, domH, topology) {
```

**Step 2: Use delayed positions in the pairwise loop**

In the non-BH loop (line 94-103), when `signalDelayEnabled && relativityEnabled`, compute the delayed position for each source particle at each grid point:

```js
for (let i = 0; i < n; i++) {
    const p = particles[i];
    let px, py;
    if (signalDelayEnabled && relativityEnabled && p.histCount >= 2) {
        // Create a temporary "observer" at the grid point
        const ret = getDelayedState(p, { pos: { x: wx, y: wy } }, simTime, periodic, domW, domH, domW * 0.5, domH * 0.5, topology);
        if (ret) { px = ret.x; py = ret.y; }
        else { px = p.pos.x; py = p.pos.y; }
    } else {
        px = p.pos.x; py = p.pos.y;
    }
    const dx = wx - px, dy = wy - py;
    const rSq = dx * dx + dy * dy + SOFTENING_SQ;
    const invR = 1 / Math.sqrt(rSq);
    gPhi -= p.mass * invR;
    ePhi += p.charge * invR;
}
```

Note: this is expensive (GRID² × N delay solves). The 6-frame update interval mitigates this. For large N with BH on, the tree path doesn't use delay (consistent with the force computation).

**Step 3: Update call site in main.js**

In `main.js` line 244, pass the additional parameters:
```js
this.heatmap.update(this.particles, this.camera, this.width, this.height,
    this.physics.pool, this.physics._lastRoot, this.physics.barnesHutEnabled,
    this.physics.signalDelayEnabled, this.physics.relativityEnabled,
    this.physics.simTime, this.physics.periodic, this.domainW, this.domainH,
    this.topology);
```

**Step 4: Commit**
```
feat: add signal delay to potential field heatmap
```

---

## Phase 5: UI Features

### Task 14: Force Equation Display (Particle Tab)

**Files:**
- Modify: `index.html:331-338` (add force breakdown section in particle details)
- Modify: `src/stats-display.js:65-93` (`updateSelected()` — compute and display force magnitudes)

**Step 1: Add HTML for force breakdown**

In `index.html`, after the `sel-force` stat row (line 337), add:

```html
<div id="force-breakdown" class="stat-group" style="margin-top: 8px">
    <span class="group-label" style="font-size: 0.62rem">Force Breakdown</span>
    <div class="stat-row stat-sub" id="fb-gravity" hidden><span class="stat-label">F<sub>G</sub> = m₁m₂/r²</span><span class="stat-value" id="fb-gravity-val">0</span></div>
    <div class="stat-row stat-sub" id="fb-coulomb" hidden><span class="stat-label">F<sub>C</sub> = q₁q₂/r²</span><span class="stat-value" id="fb-coulomb-val">0</span></div>
    <div class="stat-row stat-sub" id="fb-magnetic" hidden><span class="stat-label">F<sub>B</sub> = q(v×B)</span><span class="stat-value" id="fb-magnetic-val">0</span></div>
    <div class="stat-row stat-sub" id="fb-gravitomag" hidden><span class="stat-label">F<sub>GM</sub> = 4m(v×B<sub>g</sub>)</span><span class="stat-value" id="fb-gravitomag-val">0</span></div>
    <div class="stat-row stat-sub" id="fb-1pn" hidden><span class="stat-label">F<sub>1PN</sub></span><span class="stat-value" id="fb-1pn-val">0</span></div>
    <div class="stat-row stat-sub" id="fb-spincurv" hidden><span class="stat-label">F<sub>SC</sub> = μ∇B</span><span class="stat-value" id="fb-spincurv-val">0</span></div>
    <div class="stat-row stat-sub" id="fb-radiation" hidden><span class="stat-label">F<sub>rad</sub> (LL)</span><span class="stat-value" id="fb-radiation-val">0</span></div>
    <div class="stat-row stat-sub" id="fb-yukawa" hidden><span class="stat-label">F<sub>Y</sub> = g²e<sup>-μr</sup>/r²</span><span class="stat-value" id="fb-yukawa-val">0</span></div>
</div>
```

**Step 2: Cache DOM refs in Simulation constructor**

In `main.js`, add to `this.selDom`:
```js
fbGravity: document.getElementById('fb-gravity'),
fbGravityVal: document.getElementById('fb-gravity-val'),
fbCoulomb: document.getElementById('fb-coulomb'),
fbCoulombVal: document.getElementById('fb-coulomb-val'),
fbMagnetic: document.getElementById('fb-magnetic'),
fbMagneticVal: document.getElementById('fb-magnetic-val'),
fbGravitomag: document.getElementById('fb-gravitomag'),
fbGravitomagVal: document.getElementById('fb-gravitomag-val'),
fb1pn: document.getElementById('fb-1pn'),
fb1pnVal: document.getElementById('fb-1pn-val'),
fbSpincurv: document.getElementById('fb-spincurv'),
fbSpincurvVal: document.getElementById('fb-spincurv-val'),
fbRadiation: document.getElementById('fb-radiation'),
fbRadiationVal: document.getElementById('fb-radiation-val'),
fbYukawa: document.getElementById('fb-yukawa'),
fbYukawaVal: document.getElementById('fb-yukawa-val'),
```

**Step 3: Update force breakdown in StatsDisplay.updateSelected()**

In `src/stats-display.js` `updateSelected()`, after the existing stat updates (line 91), add:

```js
const forces = [
    { row: dom.fbGravity, val: dom.fbGravityVal, vec: p.forceGravity },
    { row: dom.fbCoulomb, val: dom.fbCoulombVal, vec: p.forceCoulomb },
    { row: dom.fbMagnetic, val: dom.fbMagneticVal, vec: p.forceMagnetic },
    { row: dom.fbGravitomag, val: dom.fbGravitomagVal, vec: p.forceGravitomag },
    { row: dom.fb1pn, val: dom.fb1pnVal, vec: p.force1PN },
    { row: dom.fbSpincurv, val: dom.fbSpincurvVal, vec: p.forceSpinCurv },
    { row: dom.fbRadiation, val: dom.fbRadiationVal, vec: p.forceRadiation },
    { row: dom.fbYukawa, val: dom.fbYukawaVal, vec: p.forceYukawa },
];
for (const f of forces) {
    if (!f.row) continue;
    const mag = Math.sqrt(f.vec.x * f.vec.x + f.vec.y * f.vec.y);
    if (mag > 1e-10) {
        f.row.hidden = false;
        f.val.textContent = fmt(mag);
    } else {
        f.row.hidden = true;
    }
}
```

**Step 4: Commit**
```
feat: add per-force-type magnitude breakdown in Particle tab
```

---

### Task 15: Save/Load State

**Files:**
- Create: `src/save-load.js` (serialize/deserialize simulation state)
- Modify: `index.html` (add save/load buttons to topbar)
- Modify: `main.js` (import and wire save/load)

**Step 1: Create save-load module**

Create `src/save-load.js`:

```js
import Particle from './particle.js';
import { setVelocity, angwToAngVel } from './relativity.js';

export function saveState(sim) {
    const state = {
        version: 1,
        particles: sim.particles.map(p => ({
            x: p.pos.x, y: p.pos.y,
            wx: p.w.x, wy: p.w.y,
            mass: p.mass, charge: p.charge, angw: p.angw,
        })),
        toggles: {},
        settings: {
            collision: sim.collisionMode,
            boundary: sim.boundaryMode,
            topology: sim.topology,
            speed: sim.speedScale,
            friction: sim.physics.bounceFriction,
        },
        camera: { x: sim.camera.x, y: sim.camera.y, zoom: sim.camera.zoom },
    };
    // Capture all physics toggle states
    const ph = sim.physics;
    for (const key of ['gravityEnabled', 'coulombEnabled', 'magneticEnabled',
        'gravitomagEnabled', 'relativityEnabled', 'barnesHutEnabled',
        'radiationEnabled', 'blackHoleEnabled', 'tidalEnabled',
        'tidalLockingEnabled', 'signalDelayEnabled', 'spinOrbitEnabled',
        'onePNEnabled', 'yukawaEnabled', 'axionEnabled', 'gwRadiationEnabled',
        'expansionEnabled']) {
        state.toggles[key] = ph[key];
    }
    // Yukawa/axion/expansion params
    state.yukawaG2 = ph.yukawaG2;
    state.yukawaMu = ph.yukawaMu;
    state.axionG = ph.axionG;
    state.axionMass = ph.axionMass;
    state.hubbleParam = ph.hubbleParam;
    return state;
}

export function loadState(state, sim) {
    if (!state || state.version !== 1) return false;

    // Clear
    sim.particles = [];
    sim.photons = [];
    sim.totalRadiated = 0;
    sim.totalRadiatedPx = 0;
    sim.totalRadiatedPy = 0;
    sim.selectedParticle = null;
    sim.physics._forcesInit = false;

    // Restore toggles
    const ph = sim.physics;
    for (const [key, val] of Object.entries(state.toggles)) {
        if (key in ph) ph[key] = val;
    }
    if (state.yukawaG2 != null) ph.yukawaG2 = state.yukawaG2;
    if (state.yukawaMu != null) ph.yukawaMu = state.yukawaMu;
    if (state.axionG != null) ph.axionG = state.axionG;
    if (state.axionMass != null) ph.axionMass = state.axionMass;
    if (state.hubbleParam != null) ph.hubbleParam = state.hubbleParam;

    // Restore settings
    if (state.settings) {
        sim.collisionMode = state.settings.collision || 'pass';
        sim.boundaryMode = state.settings.boundary || 'despawn';
        sim.topology = state.settings.topology || 'torus';
        sim.speedScale = state.settings.speed || 100;
        if (state.settings.friction != null) ph.bounceFriction = state.settings.friction;
    }

    // Restore camera
    if (state.camera) {
        sim.camera.x = state.camera.x;
        sim.camera.y = state.camera.y;
        sim.camera.zoom = state.camera.zoom;
    }

    // Restore particles
    for (const pd of state.particles) {
        const p = new Particle(pd.x, pd.y, pd.mass, pd.charge);
        p.mass = pd.mass;
        p.charge = pd.charge;
        p.angw = pd.angw;
        p.w.x = pd.wx;
        p.w.y = pd.wy;
        p.updateColor();
        const invG = ph.relativityEnabled ? 1 / Math.sqrt(1 + p.w.magSq()) : 1;
        p.vel.x = p.w.x * invG;
        p.vel.y = p.w.y * invG;
        p.angVel = ph.relativityEnabled ? angwToAngVel(p.angw, p.radius) : p.angw;
        sim.particles.push(p);
    }
    sim.stats.resetBaseline();
    return true;
}

export function downloadState(sim) {
    const state = saveState(sim);
    const json = JSON.stringify(state);
    const blob = new Blob([json], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'nohair-state.json';
    a.click();
    URL.revokeObjectURL(a.href);
}

export function uploadState(sim, onComplete) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.addEventListener('change', () => {
        const file = input.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
            try {
                const state = JSON.parse(reader.result);
                if (loadState(state, sim)) {
                    showToast('State loaded');
                    if (onComplete) onComplete();
                } else {
                    showToast('Invalid state file');
                }
            } catch { showToast('Failed to parse state file'); }
        };
        reader.readAsText(file);
    });
    input.click();
}

export function quickSave(sim) {
    const state = saveState(sim);
    localStorage.setItem('nohair-quicksave', JSON.stringify(state));
    showToast('Quick saved');
}

export function quickLoad(sim, onComplete) {
    const json = localStorage.getItem('nohair-quicksave');
    if (!json) { showToast('No quick save found'); return; }
    try {
        const state = JSON.parse(json);
        if (loadState(state, sim)) {
            showToast('Quick loaded');
            if (onComplete) onComplete();
        }
    } catch { showToast('Failed to load quick save'); }
}
```

**Step 2: Add save/load buttons to topbar**

In `index.html`, after the reset button (line 110), add:
```html
<button id="saveBtn" class="tool-btn" title="Quick Save (Ctrl+S)" aria-label="Quick save">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>
    </svg>
</button>
<button id="loadBtn" class="tool-btn" title="Quick Load (Ctrl+L)" aria-label="Quick load">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
</button>
```

**Step 3: Wire in main.js**

Import and wire the save/load functions. Add keyboard shortcuts (Ctrl+S, Ctrl+L) and button click handlers.

**Step 4: Commit**
```
feat: add save/load state (quick save + JSON file export/import)
```

---

### Task 16: Reference Pages (Expanded Info Tips)

**Files:**
- Create: `src/reference.js` (full reference content for each physics concept)
- Modify: `src/ui.js` (add reference page modal trigger)
- Modify: `index.html` (add reference overlay HTML)
- Modify: `styles.css` (add reference overlay styles)

**Step 1: Create reference content module**

Create `src/reference.js` with a `REFERENCE` object mapping each info key to a `{ title, body }` where `body` is a longer HTML+KaTeX string with derivations and explanations. Cover all existing physics concepts plus the new ones (Yukawa, axion, GW radiation, Kerr-Newman, expansion).

**Step 2: Add reference overlay HTML**

In `index.html`, before the closing `</body>`, add:
```html
<div id="reference-overlay" class="reference-overlay" hidden>
    <div class="reference-panel glass">
        <div class="stats-header">
            <h2 id="reference-title" class="stats-title">Reference</h2>
            <button id="reference-close" class="tool-btn" aria-label="Close reference">
                <svg width="16" height="16" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
        </div>
        <div id="reference-body" class="reference-body scrollbar-thin"></div>
    </div>
</div>
```

**Step 3: Wire info triggers to open reference on long-press or Shift+click**

In `src/ui.js`, modify the info trigger setup to add a second handler:
```js
trigger.addEventListener('click', (e) => {
    if (e.shiftKey && REFERENCE[key]) {
        // Open full reference page
        document.getElementById('reference-title').textContent = REFERENCE[key].title;
        document.getElementById('reference-body').innerHTML = REFERENCE[key].body;
        document.getElementById('reference-overlay').hidden = false;
        // Render KaTeX in the reference body
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(document.getElementById('reference-body'), { delimiters: [{ left: '$', right: '$', display: false }] });
        }
    }
});
```

**Step 4: Add CSS for reference overlay**

In `styles.css`, add styles for `.reference-overlay`, `.reference-panel`, `.reference-body`.

**Step 5: Commit**
```
feat: add expandable reference pages for physics concepts (Shift+click info button)
```

---

### Task 17: CLAUDE.md Update

**Files:**
- Modify: `CLAUDE.md` (document all new features, toggles, forces, presets)

Update the File Map, Force Types, Toggle Dependencies, UI sections, and Key Patterns to reflect all new features.

**Step 1: Commit**
```
docs: update CLAUDE.md for new physics features
```

---

## Summary

| Phase | Tasks | Features |
|---|---|---|
| 1 | 1-5 | Yukawa potential, Axion coupling |
| 2 | 6-8 | Kerr-Newman BH, Reduced BH softening, Photon bending |
| 3 | 9-11 | GW quadrupole radiation, EM quadrupole, Roche lobe overflow |
| 4 | 12-13 | Cosmological expansion, Signal-delayed heatmap |
| 5 | 14-17 | Force equation display, Save/Load, Reference pages, CLAUDE.md |

Commit after each task. Test in browser after each phase.
