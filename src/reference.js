// ─── Reference Pages ───
// Extended content for each physics concept, shown via Shift+click on info buttons.

export const REFERENCE = {
    gravity: {
        title: 'Newtonian Gravity',
        body: `
<p>Every massive particle attracts every other massive particle. In natural units ($G = c = 1$), gravity is the simplest long-range force — and the architect of all large-scale structure in the universe.</p>

<h3>Lagrangian</h3>
<p>$$\\mathcal{L} = \\frac{1}{2}m v^2 + \\frac{m_1 m_2}{r}$$</p>
<p>The positive sign on the potential term means gravity lowers the total energy as particles approach — the system naturally tends toward collapse.</p>

<h3>Force</h3>
<p>$$\\mathbf{F} = +\\frac{m_1 m_2}{r^2}\\,\\hat{r}$$</p>
<p>Always attractive, with $1/r^2$ falloff — a geometric consequence of flux spreading over spherical surfaces. Plummer softening ($r \\to \\sqrt{r^2 + \\epsilon^2}$, $\\epsilon^2 = 64$) prevents divergence at close approach.</p>

<h3>Potential Energy</h3>
<p>$$V(r) = -\\frac{m_1 m_2}{r}$$</p>
<p>Negative and unbounded below. Total energy $E = \\text{KE} + V$ determines the orbit type:</p>
<ul>
<li>$E < 0$: bound orbit (ellipse)</li>
<li>$E = 0$: parabolic escape</li>
<li>$E > 0$: hyperbolic flyby</li>
</ul>

<h3>Kepler Orbits</h3>
<p>A test particle orbiting mass $M$ follows a conic section. For circular orbits:</p>
<p>$$v_{\\text{circ}} = \\sqrt{\\frac{M}{r}}, \\qquad T = 2\\pi\\sqrt{\\frac{r^3}{M}}$$</p>
<p>This is Kepler's third law: the orbital period squared is proportional to the semi-major axis cubed. The "Orbit" spawn mode uses this formula to give new particles exactly circular velocity.</p>

<h3>Escape Velocity</h3>
<p>$$v_{\\text{esc}} = \\sqrt{\\frac{2M}{r}} = \\sqrt{2}\\,v_{\\text{circ}}$$</p>
<p>A particle launched at exactly this speed has zero total energy and will coast to infinity, never returning.</p>

<h3>Two-Body Reduction</h3>
<p>Any two gravitating bodies can be reduced to an equivalent one-body problem using the reduced mass $\\mu = m_1 m_2/(m_1+m_2)$. The relative separation traces a Kepler ellipse. In pairwise mode with gravity only, this simulation conserves energy and angular momentum to machine precision.</p>
`,
    },

    coulomb: {
        title: 'Coulomb Force',
        body: `
<p>The electrostatic interaction between charged particles — the force that governs chemistry, electricity, and the structure of every atom.</p>

<h3>Lagrangian</h3>
<p>$$\\mathcal{L} = \\frac{1}{2}m v^2 - \\frac{q_1 q_2}{r}$$</p>

<h3>Force</h3>
<p>$$\\mathbf{F} = -\\frac{q_1 q_2}{r^2}\\,\\hat{r}$$</p>
<p>Like charges ($q_1 q_2 > 0$) repel; opposite charges ($q_1 q_2 < 0$) attract. Same $1/r^2$ falloff as gravity, but with two crucial differences: charge comes in two signs, and the coupling can be far stronger.</p>

<h3>Potential Energy</h3>
<p>$$V(r) = +\\frac{q_1 q_2}{r}$$</p>
<p>Positive for like charges (repulsive barrier), negative for opposite charges (attractive well). An electron-proton system has $V < 0$ at all distances, forming hydrogen-like bound orbits.</p>

<h3>Classical Bound States</h3>
<p>With both gravity and Coulomb active, a massive positive nucleus can bind lighter negative charges in stable orbits. The effective potential has a minimum where gravitational + Coulomb attraction balances centrifugal repulsion. These are not quantum atoms, but they capture the essential orbital mechanics.</p>

<h3>Screening & Neutrality</h3>
<p>Unlike gravity (which only attracts), opposite charges cancel. A system with equal positive and negative charges appears neutral from far away. This is why electromagnetism doesn't dominate at cosmic scales — gravity does, because mass has only one sign.</p>
`,
    },

    magnetic: {
        title: 'Magnetic Interactions',
        body: `
<p>Magnetism arises from charges in motion. This simulation models two distinct mechanisms: the Lorentz force from translating charges, and dipole-dipole interactions from spinning charges.</p>

<h3>Lorentz Force</h3>
<p>A charge $q$ moving with velocity $\\mathbf{v}$ through a magnetic field $\\mathbf{B}$ experiences:</p>
<p>$$\\mathbf{F} = q(\\mathbf{v} \\times \\mathbf{B})$$</p>
<p>This force is always perpendicular to the velocity — it changes direction but never speed. In a uniform field, charges trace circles (cyclotron motion) with:</p>
<p>$$r_L = \\frac{mv_\\perp}{qB}, \\qquad \\omega_c = \\frac{qB}{m}$$</p>
<p>Each moving charge also generates a magnetic field. The out-of-plane component from a source charge:</p>
<p>$$B_z = \\frac{q_s\\,(\\mathbf{v}_s \\times \\hat{r})_z}{r^2}$$</p>

<h3>Boris Integrator</h3>
<p>The Boris algorithm handles the Lorentz force exactly: it rotates the velocity vector around $\\mathbf{B}$ without changing its magnitude, preserving kinetic energy through every gyration. This is why cyclotron orbits remain stable indefinitely, even at large timesteps — a critical property for long simulations.</p>

<h3>Magnetic Dipole Interaction</h3>
<p>A spinning charged particle generates a magnetic dipole moment:</p>
<p>$$\\mu = \\frac{q\\omega r^2}{5}$$</p>
<p>Two dipoles interact with a force and potential:</p>
<p>$$\\mathbf{F} = \\frac{3\\mu_1\\mu_2}{r^4}\\,\\hat{r}, \\qquad V = +\\frac{\\mu_1\\mu_2}{r^3}$$</p>
<p>Aligned perpendicular-to-plane dipoles repel. This is opposite to the gravitomagnetic case, where co-rotating masses attract — a consequence of gravity being spin-2 vs. electromagnetism being spin-1.</p>

<h3>Connection to Maxwell's Equations</h3>
<p>The Biot-Savart law ($\\mathbf{B} \\propto q\\mathbf{v} \\times \\hat{r}/r^2$) generates the field; the Lorentz force law ($\\mathbf{F} = q\\mathbf{v} \\times \\mathbf{B}$) governs the response. Together they produce mutual deflection of moving charges — the principle behind electric motors.</p>
`,
    },

    gravitomag: {
        title: 'Gravitomagnetism',
        body: `
<p>General relativity predicts that moving masses generate effects analogous to magnetism. Just as electric currents create magnetic fields, mass currents create <em>gravitomagnetic</em> fields. This is the gravitoelectromagnetic (GEM) framework.</p>

<h3>The GEM Analogy</h3>
<p>Maxwell's equations for electromagnetism have gravitational analogs:</p>
<table style="width:100%;border-collapse:collapse;margin:12px 0">
<tr><th style="text-align:left;padding:4px 8px">Electromagnetism</th><th style="text-align:left;padding:4px 8px">Gravitoelectromagnetism</th></tr>
<tr><td style="padding:4px 8px">Electric field $\\mathbf{E}$</td><td style="padding:4px 8px">Gravitoelectric field $\\mathbf{g}$ (Newtonian gravity)</td></tr>
<tr><td style="padding:4px 8px">Magnetic field $\\mathbf{B}$</td><td style="padding:4px 8px">Gravitomagnetic field $\\mathbf{B}_g$</td></tr>
<tr><td style="padding:4px 8px">Charge $q$</td><td style="padding:4px 8px">Mass $m$ (one sign only)</td></tr>
</table>

<h3>Linear Gravitomagnetic Force</h3>
<p>$$\\mathbf{F} = 4m(\\mathbf{v} \\times \\mathbf{B}_g)$$</p>
<p>The factor of 4 (compared to the EM Lorentz factor of 1) arises because gravity is mediated by a spin-2 field (gravitons) rather than spin-1 (photons). This factor is a genuine prediction of general relativity.</p>

<h3>GM Dipole Interaction</h3>
<p>Spinning masses carry angular momentum $L = 2m\\omega r^2/5$, which acts as a gravitomagnetic dipole:</p>
<p>$$\\mathbf{F} = +\\frac{3L_1 L_2}{r^4}\\,\\hat{r}, \\qquad V = -\\frac{L_1 L_2}{r^3}$$</p>
<p>Co-rotating masses attract — the opposite sign from EM dipoles. This sign difference traces back to gravity being universally attractive (one sign of "charge").</p>

<h3>Frame Dragging (Lense-Thirring Effect)</h3>
<p>A spinning mass drags spacetime around it, exerting a torque on nearby objects:</p>
<p>$$\\tau = \\frac{2L_s(\\omega_s - \\omega_p)}{r^3}$$</p>
<p>This drives nearby spins toward co-rotation. In 2004, the Gravity Probe B satellite measured Earth's frame dragging at 37 milliarcseconds per year, confirming this prediction of general relativity to within 19% accuracy.</p>
`,
    },

    relativity: {
        title: 'Special Relativity',
        body: `
<p>Einstein's special relativity imposes a universal speed limit: nothing travels faster than light ($c = 1$ in natural units). This simulation enforces the limit through the proper velocity formalism.</p>

<h3>Proper Velocity (Celerity)</h3>
<p>Instead of tracking coordinate velocity $\\mathbf{v}$ directly, the simulation uses <em>proper velocity</em>:</p>
<p>$$\\mathbf{w} = \\gamma\\mathbf{v}$$</p>
<p>where $\\gamma = 1/\\sqrt{1-v^2}$. Proper velocity can grow without bound under constant force, but coordinate velocity is always subluminal:</p>
<p>$$\\mathbf{v} = \\frac{\\mathbf{w}}{\\sqrt{1+w^2}} \\quad \\Rightarrow \\quad |\\mathbf{v}| < 1 \\text{ always}$$</p>
<p>This is mathematically elegant: $\\mathbf{w}$ obeys Newton's second law ($\\mathbf{F} = m\\,d\\mathbf{w}/dt$) exactly, and the speed limit emerges from the nonlinear map to $\\mathbf{v}$.</p>

<h3>Lorentz Factor</h3>
<p>$$\\gamma = \\sqrt{1+w^2}$$</p>
<p>At low speeds ($w \\ll 1$): $\\gamma \\approx 1$, $\\mathbf{v} \\approx \\mathbf{w}$. As $w \\to \\infty$: $v \\to 1$ asymptotically. The transition becomes noticeable around $v \\gtrsim 0.3c$.</p>

<h3>Relativistic Kinetic Energy</h3>
<p>$$\\text{KE} = (\\gamma-1)mc^2 = \\frac{w^2}{\\gamma+1}\\,m$$</p>
<p>The second form avoids catastrophic cancellation when $v \\ll c$ (where $\\gamma-1 \\approx 0$). At low speeds this reduces to $\\frac{1}{2}mv^2$; at high speeds it grows as $\\gamma mc^2$.</p>

<h3>Signal Delay</h3>
<p>With relativity enabled, forces propagate at the speed of light rather than instantaneously. Each particle sees its neighbors at their <em>retarded position</em> — where they were when light left them:</p>
<p>$$|\\mathbf{x}_{\\text{src}}(t_{\\text{ret}}) - \\mathbf{x}_{\\text{obs}}| = t_{\\text{now}} - t_{\\text{ret}}$$</p>
<p>This light-cone equation is solved for each interacting pair using a three-phase algorithm:</p>
<ol>
<li><b>Newton-Raphson</b>: segment search over the history buffer (up to 8 iterations)</li>
<li><b>Quadratic solve</b>: exact solution on the converged segment</li>
<li><b>Extrapolation</b>: constant-velocity fallback for times before recorded history</li>
</ol>
<p>Signal delay causes forces to lag behind, producing aberration effects that become significant when particles move at appreciable fractions of $c$.</p>

<h3>Relativistic Spin</h3>
<p>Spin uses the same celerity pattern. Angular celerity $W$ maps to angular velocity:</p>
<p>$$\\omega = \\frac{W}{\\sqrt{1+W^2 r^2}}$$</p>
<p>This ensures the surface speed $|\\omega|r$ stays below $c$, preventing unphysical breakup of rapidly spinning particles.</p>
`,
    },

    radiation: {
        title: 'Radiation',
        body: `
<p>Accelerating charges radiate electromagnetic waves; orbiting masses radiate gravitational waves. Both carry energy and momentum out of the system, causing orbits to decay. Three radiation channels are modeled.</p>

<h3>Larmor Dipole Radiation</h3>
<p>Requires Coulomb. An accelerating charge radiates power:</p>
<p>$$P = \\frac{2q^2}{3}a^2$$</p>
<p>The reaction on the emitter is the Landau-Lifshitz force, a physically consistent alternative to the Abraham-Lorentz force that avoids runaway solutions:</p>
<p>$$\\mathbf{F}_{\\text{rad}} = \\tau\\left[\\frac{\\dot{\\mathbf{F}}}{\\gamma^3} - \\frac{\\mathbf{v}F^2}{m\\gamma^2} + \\frac{\\mathbf{F}(\\mathbf{v}\\cdot\\mathbf{F})}{m\\gamma^4}\\right]$$</p>
<p>where $\\tau = 2q^2/(3m)$ is the radiation timescale. The force is clamped to 50% of the external force for numerical stability. Emitted photons follow a dipole angular distribution with relativistic aberration beaming.</p>

<h3>EM Quadrupole Radiation</h3>
<p>Requires Coulomb. A system with a time-varying charge quadrupole moment radiates:</p>
<p>$$P_{\\text{EM}} = \\frac{1}{180}\\left|\\dddot{Q}_{ij}\\right|^2, \\qquad Q_{ij} = \\sum_k q_k\\, x_i^{(k)} x_j^{(k)}$$</p>
<p>This is the next multipole order above dipole, significant for binary charged systems. Photon emission uses a TT-projected angular pattern via rejection sampling.</p>

<h3>Gravitational Wave Radiation</h3>
<p>Requires Gravity. The mass quadrupole formula — the leading-order source of gravitational waves:</p>
<p>$$P_{\\text{GW}} = \\frac{1}{5}\\left|\\dddot{I}_{ij}\\right|^2, \\qquad I_{ij} = \\sum_k m_k\\left(x_i^{(k)} x_j^{(k)} - \\frac{\\delta_{ij}}{3}r_k^2\\right)$$</p>
<p>For a circular binary, this reduces to the Peters formula:</p>
<p>$$P_{\\text{GW}} = \\frac{32}{5}\\frac{m_1^2 m_2^2(m_1+m_2)}{r^5}$$</p>
<p>The steep $1/r^5$ dependence means power grows dramatically as the orbit shrinks, creating a runaway inspiral. This is exactly what LIGO detects: the final moments of a billion-year inspiral, as the frequency chirps up to merger.</p>

<h3>Why No Gravitational Dipole?</h3>
<p>The gravitational dipole moment $\\sum m_i\\mathbf{x}_i$ is the center of mass, which moves at constant velocity by momentum conservation. No time-varying dipole means no dipole radiation — gravitational waves require at least the quadrupole, making them intrinsically weaker than electromagnetic waves.</p>

<h3>Energy Extraction</h3>
<p>Radiated energy is extracted from the emitter's kinetic energy by scaling the tangential velocity. Emitted photons (yellow for EM, red for gravitons) carry both energy and momentum, tracked by the simulation for conservation accounting.</p>
`,
    },

    onepn: {
        title: '1PN Corrections',
        body: `
<p>First post-Newtonian corrections are $O(v^2/c^2)$ terms that appear when expanding general relativity (or its EM analog) beyond leading order. They capture the first effects of finite light speed on the equations of motion.</p>

<h3>Einstein-Infeld-Hoffmann (EIH) &mdash; Gravitational Sector</h3>
<p>Requires Gravitomagnetic. The EIH equations of motion at 1PN produce the famous perihelion precession:</p>
<p>$$\\Delta\\phi = \\frac{6\\pi M}{a(1-e^2)} \\text{ rad/orbit}$$</p>
<p>For Mercury, this gives 43 arcseconds per century — the anomaly that led Einstein to general relativity. In this simulation, the precession appears as a slowly rotating ellipse.</p>
<p>The EIH Lagrangian (two-body, symmetric remainder after subtracting the GM Lorentz piece):</p>
<p>$$\\mathcal{L}_{\\text{EIH}} = \\frac{m_1 m_2}{2r}\\!\\left[(\\mathbf{v}_1 \\!\\cdot\\! \\mathbf{v}_2) + (\\mathbf{v}_1 \\!\\cdot\\! \\hat{r})(\\mathbf{v}_2 \\!\\cdot\\! \\hat{r})\\right] - \\frac{m_1 m_2}{r}\\!\\left[\\frac{v_1^2+v_2^2}{2} - \\frac{m_1+m_2}{2r}\\right]$$</p>

<h3>Darwin EM &mdash; Electromagnetic Sector</h3>
<p>Requires Magnetic. The Darwin Lagrangian for two charges at $O(v^2/c^2)$:</p>
<p>$$\\mathcal{L}_{\\text{Darwin}} = -\\frac{q_1 q_2}{2r}\\!\\left[(\\mathbf{v}_1 \\!\\cdot\\! \\mathbf{v}_2) + (\\mathbf{v}_1 \\!\\cdot\\! \\hat{r})(\\mathbf{v}_2 \\!\\cdot\\! \\hat{r})\\right]$$</p>
<p>This adds velocity-dependent corrections beyond Coulomb + Lorentz. It modifies atomic electron orbits and is the classical counterpart to the Breit interaction in quantum mechanics.</p>

<h3>Bazanski Cross-Term &mdash; Mixed Gravity-EM Sector</h3>
<p>Requires both Gravitomagnetic and Magnetic. A mixed gravity-EM interaction Lagrangian at 1PN:</p>
<p>$$\\mathcal{L}_{\\text{Baz}} = -\\frac{q_1 q_2(m_1+m_2) - (q_1^2 m_2 + q_2^2 m_1)}{2r^2}$$</p>
<p>Unlike the EIH and Darwin terms, this Lagrangian has no velocity dependence — it is purely a position-dependent $1/r^2$ potential correction. It vanishes for identical particles (when $q_1 = q_2$ and $m_1 = m_2$) and represents the gravitational correction to electromagnetic self-energy and vice versa.</p>

<h3>Integration Scheme</h3>
<p>All three sectors use velocity-Verlet: the 1PN force is computed before and after the drift step, and the average is applied, giving second-order accuracy. This is necessary because 1PN forces depend on both position and velocity.</p>
`,
    },

    blackhole: {
        title: 'Black Hole Mode',
        body: `
<p>When enabled, every particle is treated as a Kerr-Newman black hole — a rotating, charged singularity surrounded by an event horizon. This is the most general stationary black hole solution in general relativity.</p>

<h3>The No-Hair Theorem</h3>
<p>A black hole is completely characterized by just three numbers: mass $M$, angular momentum $J$, and charge $Q$. All other information about the matter that formed it is lost behind the horizon. This theorem is the namesake of this simulation.</p>

<h3>Kerr-Newman Horizon</h3>
<p>The outer event horizon radius:</p>
<p>$$r_+ = M + \\sqrt{M^2 - a^2 - Q^2}$$</p>
<p>where $a = J/M$ is the spin parameter. The horizon exists only when $M^2 \\geq a^2 + Q^2$; violations would produce a naked singularity, forbidden by cosmic censorship (a minimum radius floor prevents this in the simulation).</p>

<h3>Ergosphere</h3>
<p>The region between the horizon $r_+$ and the static limit:</p>
<p>$$r_{\\text{ergo}} = M + \\sqrt{M^2 - a^2}$$</p>
<p>Inside the ergosphere, spacetime is dragged so strongly that nothing can remain stationary — everything must co-rotate with the black hole. The Penrose process can extract rotational energy by exploiting this region, converting spin energy to kinetic energy of escaping particles.</p>

<h3>Hawking Radiation</h3>
<p>Quantum effects near the horizon cause black holes to radiate thermally. The temperature depends on the surface gravity:</p>
<p>$$\\kappa = \\frac{\\sqrt{M^2 - a^2 - Q^2}}{r_+^2 + a^2}, \\qquad T = \\frac{\\kappa}{2\\pi}$$</p>
<p>Radiated power follows the Stefan-Boltzmann law:</p>
<p>$$P = \\sigma T^4 A, \\qquad \\sigma = \\frac{\\pi^2}{60}, \\qquad A = 4\\pi(r_+^2+a^2)$$</p>
<p>Smaller black holes are hotter and radiate faster, creating a runaway: mass decreases $\\to$ temperature rises $\\to$ radiation intensifies $\\to$ evaporation. The final instant produces a burst of photons.</p>

<h3>Extremal Limit</h3>
<p>When $M^2 = a^2 + Q^2$, the inner and outer horizons merge, surface gravity vanishes, and the temperature drops to zero — the black hole stops radiating. Extremal black holes are the most compact objects possible for their mass and charge, saturating the cosmic censorship bound.</p>
`,
    },

    spinorbit: {
        title: 'Spin-Orbit Coupling',
        body: `
<p>Spin-orbit coupling transfers energy and momentum between a particle's translational motion and its intrinsic rotation. This occurs whenever a spinning particle moves through a non-uniform field.</p>

<h3>Stern-Gerlach Force (Electromagnetic)</h3>
<p>A magnetic dipole in a non-uniform magnetic field feels a gradient force:</p>
<p>$$\\mathbf{F} = \\mu\\,\\nabla B_z$$</p>
<p>In the original 1922 experiment, Stern and Gerlach sent silver atoms through an inhomogeneous magnetic field and observed the beam split into discrete components — the first direct evidence of quantized angular momentum. This simulation shows the classical analog: spinning charged particles deflect toward or away from field concentrations depending on the sign of their spin.</p>

<h3>Mathisson-Papapetrou Force (Gravitational)</h3>
<p>$$\\mathbf{F} = -L\\,\\nabla B_{gz}$$</p>
<p>The opposite sign (the GEM sign flip) means spinning masses are deflected in the opposite direction from spinning charges in equivalent field geometries. This force produces subtle orbital corrections for spinning bodies around massive objects, such as the geodetic precession of pulsars.</p>

<h3>Spin-Orbit Energy Transfer</h3>
<p>Moving through a field gradient transfers energy between orbital kinetic energy and spin:</p>
<p>$$\\frac{dE}{dt} = -\\mu(\\mathbf{v}\\cdot\\nabla B_z) \\quad \\text{(EM)}, \\qquad \\frac{dE}{dt} = -L(\\mathbf{v}\\cdot\\nabla B_{gz}) \\quad \\text{(GM)}$$</p>
<p>This can speed up or slow down a particle's spin while adjusting its orbital energy to compensate, conserving total energy.</p>

<h3>Astrophysical Significance</h3>
<p>Spin-orbit coupling is crucial in binary pulsar systems, where rapidly spinning neutron stars precess due to gravitational spin-orbit effects (geodetic precession). Hulse-Taylor binary pulsar measurements of these effects provided some of the strongest early confirmations of general relativity.</p>
`,
    },

    yukawa: {
        title: 'Yukawa Potential',
        body: `
<p>Proposed by Hideki Yukawa in 1935 to explain the strong nuclear force binding protons and neutrons inside atomic nuclei. His key insight: if the force is mediated by a massive particle, the potential must have an exponential cutoff.</p>

<h3>Lagrangian</h3>
<p>$$\\mathcal{L} = \\frac{1}{2}mv^2 + \\frac{g^2\\,m_1 m_2\\,e^{-\\mu r}}{r}$$</p>

<h3>Potential</h3>
<p>$$V(r) = -\\frac{g^2\\,m_1 m_2\\,e^{-\\mu r}}{r}$$</p>
<p>A screened Coulomb (or gravity) potential. The coupling $g^2$ sets the strength; the mediator mass $\\mu$ sets the range $\\lambda = 1/\\mu$.</p>

<h3>Force</h3>
<p>$$\\mathbf{F} = -g^2 m_1 m_2\\,\\frac{e^{-\\mu r}}{r^2}(1+\\mu r)\\,\\hat{r}$$</p>
<p>The $(1+\\mu r)$ factor comes from differentiating $e^{-\\mu r}/r$. At short range ($r \\ll 1/\\mu$), the exponential is approximately 1 and the force looks like gravity. At long range ($r \\gg 1/\\mu$), it vanishes exponentially.</p>

<h3>Physical Interpretation</h3>
<p>In quantum field theory, every force is mediated by a virtual particle. The mediator's mass $m$ determines the range through the uncertainty principle: a virtual particle can exist for time $\\Delta t \\sim \\hbar/(mc^2)$, traveling at most $c\\Delta t \\sim \\hbar/(mc) = 1/m$ (in natural units). This is the Compton wavelength.</p>
<ul>
<li><b>Massless mediator</b> ($\\mu = 0$): infinite range $\\to$ recovers $1/r$ potential (gravity, Coulomb)</li>
<li><b>Massive mediator</b> ($\\mu > 0$): range $\\sim 1/\\mu$ $\\to$ nuclear force (~1 fm for pions)</li>
</ul>
<p>The pion, with mass $\\sim 140$ MeV/$c^2$, gives a range of $\\sim 1.4$ fm, matching the observed range of nuclear binding. Yukawa predicted the pion's existence this way — it was discovered experimentally in 1947, earning him the Nobel Prize.</p>

<h3>Scalar Breit Correction</h3>
<p>When 1PN corrections are enabled, the Yukawa force receives $O(v^2/c^2)$ relativistic corrections from the Breit equation for massive scalar boson exchange. The correction Hamiltonian is:</p>
<p>$$\\delta H = \\frac{g^2 m_1 m_2\\,e^{-\\mu r}}{2r}\\left[\\mathbf{v}_1\\!\\cdot\\!\\mathbf{v}_2 + (\\hat{r}\\!\\cdot\\!\\mathbf{v}_1)(\\hat{r}\\!\\cdot\\!\\mathbf{v}_2)(1+\\mu r)\\right]$$</p>
<p>This is positive (repulsive), weakening the attraction for fast-moving particles. Unlike EM or gravity, scalar (spin-0) exchange produces no magnetic-type force — all corrections are radial and velocity-dependent. The $(1+\\mu r)$ factor on the radial-velocity term comes from the massive propagator.</p>
<p>The resulting force has both radial and tangential components, accumulated into the 1PN display vector. A velocity-Verlet correction ensures accuracy for these velocity-dependent terms.</p>

<h3>Beyond Nuclear Physics</h3>
<p>The Yukawa form appears throughout physics: Debye screening in plasmas, screened Coulomb potentials in metals, and hypothetical fifth forces in modified gravity theories. Any massive scalar or vector boson exchange produces this characteristic exponential envelope.</p>
`,
    },

    axion: {
        title: 'Axion-Like Scalar Field',
        body: `
<p>The axion is a hypothetical particle originally proposed to solve the strong CP problem in quantum chromodynamics — the puzzle of why the strong force conserves CP symmetry despite having no reason to. It has since become a leading dark matter candidate.</p>

<h3>The Dynamical Field</h3>
<p>The axion field $a(\\mathbf{x},t)$ lives on a 64×64 grid, governed by the Klein-Gordon equation with a quadratic potential:</p>
<p>$$\\frac{\\partial^2 a}{\\partial t^2} = \\nabla^2 a - m_a^2\\,a - \\gamma\\dot{a} + \\text{source}$$</p>
<p>The potential $V(a) = \\frac{1}{2}m_a^2 a^2$ has its minimum at $a=0$ — unlike the Higgs, there is no symmetry breaking. The field oscillates around zero with frequency $m_a$, exactly as cosmological axion dark matter does. Damping $g\\,m_a\\dot{a}$ gives $Q = 1/g$, so the resonant buildup exactly compensates the coupling strength ($g \\cdot Q = 1$). In nature, the axion oscillation is essentially undamped (cosmological damping comes only from Hubble friction $3H\\dot{a}$).</p>

<h3>Scalar Coupling to Electromagnetism</h3>
<p>The QCD axion's pseudoscalar coupling $a\\,F_{\\mu\\nu}\\tilde{F}^{\\mu\\nu} \\propto a\\,\\mathbf{E}\\cdot\\mathbf{B}$ vanishes identically in 2D, where $\\mathbf{E}$ lies in the plane and $\\mathbf{B}$ is perpendicular. Instead, this simulation uses the <em>scalar</em> coupling to the EM field invariant $F_{\\mu\\nu}F^{\\mu\\nu}$, which is non-zero in 2D and is physically motivated for axion-like particles (ALPs):</p>
<p>$$\\mathcal{L}_{\\text{int}} = -\\tfrac{1}{4}\\bigl(1 + g\\,a\\bigr)\\,F_{\\mu\\nu}F^{\\mu\\nu}$$</p>
<p>This makes the fine structure constant position-dependent:</p>
<p>$$\\alpha_{\\text{eff}}(\\mathbf{x}) = \\alpha\\left(1 + a(\\mathbf{x})\\right)$$</p>
<p>All electromagnetic forces — Coulomb, magnetic dipole, Biot-Savart — use the <em>local</em> coupling evaluated at each particle's position. Spatial variation in the field creates regions of stronger and weaker EM interaction.</p>

<h3>Source and Gradient Force</h3>
<p>From the $aF^2$ vertex, the axion field equation acquires a source proportional to the local EM field energy. For point charges, the dominant contribution is the Coulomb self-energy ($\\propto q^2$), which is what particles deposit via PQS (cubic B-spline) interpolation. The gradient force arises from the position-dependence of this self-energy in the axion background:</p>
<p>$$\\text{source} = g\\,q^2, \\qquad \\mathbf{F}_a = -g\\,q^2\\,\\nabla a$$</p>
<p>Neutral particles neither source nor feel the axion field. The coupling $g = 0.2$ compensates for the field's high quality factor ($Q \\approx 20$); in nature, $g \\sim \\alpha/f_a$ is fantastically small.</p>

<h3>Detection Experiments</h3>
<p>Several major experiments search for axion-photon conversion:</p>
<ul>
<li><b>ADMX</b>: a resonant microwave cavity that converts dark matter axions to photons in a strong magnetic field</li>
<li><b>ABRACADABRA</b>: searches for oscillating magnetic flux induced by the axion-photon coupling</li>
<li><b>CASPEr</b>: looks for oscillating nuclear spin precession driven by axion-nucleon interaction</li>
</ul>

<h3>Field Visualization</h3>
<p>The field overlay shows $a > 0$ in blue and $a < 0$ in red, with opacity proportional to field amplitude. Watch how charged particles source field excitations that propagate outward, oscillating at frequency $m_a$, and modulate the local EM coupling strength.</p>
`,
    },

    higgs: {
        title: 'Higgs Scalar Field',
        body: `
<p>The Higgs mechanism is how elementary particles acquire mass. It is the only fundamental scalar field in the Standard Model, confirmed by the discovery of the Higgs boson at CERN in 2012 (mass 125 GeV/$c^2$).</p>

<h3>Mexican Hat Potential</h3>
<p>$$V(\\phi) = -\\frac{1}{2}\\mu^2\\phi^2 + \\frac{1}{4}\\lambda\\phi^4$$</p>
<p>This potential has a local maximum at $\\phi = 0$ and a ring of degenerate minima at the vacuum expectation value (VEV):</p>
<p>$$v = \\frac{\\mu}{\\sqrt{\\lambda}}$$</p>
<p>The field spontaneously "rolls" to $\\phi = v$, breaking the symmetry. In this simulation, $v = 1$.</p>

<h3>Lagrangian</h3>
<p>The Klein-Gordon Lagrangian with the Mexican hat potential:</p>
<p>$$\\mathcal{L} = \\frac{1}{2}\\dot{\\phi}^2 - \\frac{1}{2}|\\nabla\\phi|^2 + \\frac{1}{2}\\mu^2\\phi^2 - \\frac{1}{4}\\lambda\\phi^4$$</p>
<p>Small excitations around the VEV ($\\phi = v + h$) propagate as the Higgs boson, with mass $m_H = \\mu\\sqrt{2}$.</p>

<h3>Mass Generation</h3>
<p>Particles couple to the field via Yukawa-type terms. Their effective mass depends on the local field value:</p>
<p>$$m_{\\text{eff}} = m_0\\cdot|\\phi(\\mathbf{x})|$$</p>
<p>where $m_0$ is the bare coupling strength. When $\\phi = v = 1$ (the vacuum), particles have their full mass. When $\\phi \\to 0$ (symmetry restored), particles become effectively massless — they lose all inertia.</p>

<h3>Gradient Force</h3>
<p>Particles feel a force toward regions of higher field:</p>
<p>$$\\mathbf{F} = -m_0\\,\\nabla\\phi$$</p>
<p>This is the classical analog of the Higgs mechanism: particles are dragged by spatial variations of the field, and regions with depleted $\\phi$ act as effective potential wells.</p>

<h3>Field Equation</h3>
<p>$$\\ddot{\\phi} = \\nabla^2\\phi + \\mu^2\\phi - \\lambda\\phi^3 + \\rho_{\\text{source}} - 2m_H\\dot{\\phi}$$</p>
<p>The source term $\\rho$ comes from particle deposition (cubic B-spline interpolation), and the damping term $2m_H\\dot{\\phi}$ provides critical damping. The field is evolved via symplectic Euler.</p>

<h3>Electroweak Phase Transition</h3>
<p>At high temperatures, thermal corrections modify the effective potential:</p>
<p>$$\\mu^2_{\\text{eff}} = \\mu^2 - T^2_{\\text{local}}$$</p>
<p>When $T^2 > \\mu^2$, the minimum at $\\phi = v$ disappears — the only equilibrium is $\\phi = 0$. The symmetry is restored and particles lose their mass. This models the electroweak phase transition that occurred $\\sim 10^{-12}$ seconds after the Big Bang, when the universe cooled below $\\sim 160$ GeV.</p>

<h3>Field Energy</h3>
<p>$$E = \\int\\!\\left(\\frac{1}{2}\\dot{\\phi}^2 + \\frac{1}{2}|\\nabla\\phi|^2 + V(\\phi) - V(v)\\right)dA$$</p>
<p>The vacuum energy $V(v)$ is subtracted so the ground state carries zero field energy.</p>

<h3>The Mass Slider</h3>
<p>Controls $m_H$ (range 0.25&ndash;1). Smaller $m_H$ means a shallower potential well, longer interaction range ($\\sim 1/m_H$), and weaker restoring force — easier to displace from the VEV and trigger phase transitions.</p>
`,
    },

    expansion: {
        title: 'Cosmological Expansion',
        body: `
<p>The universe is expanding: distant galaxies recede from us at speeds proportional to their distance. This is Hubble's law, the observational cornerstone of modern cosmology.</p>

<h3>Hubble Flow</h3>
<p>$$\\mathbf{v}_H = H\\cdot\\mathbf{r}$$</p>
<p>where $H$ is the Hubble parameter and $\\mathbf{r}$ is measured from the domain center. Every particle acquires an outward drift proportional to its distance. This is not a force — it is the stretching of space itself.</p>

<h3>Hubble Drag (Cosmological Redshift)</h3>
<p>Peculiar velocities (motion relative to the Hubble flow) redshift over time:</p>
<p>$$\\mathbf{v}_{\\text{pec}} \\to \\mathbf{v}_{\\text{pec}}(1-H\\,dt)$$</p>
<p>A photon emitted by a distant galaxy arrives with a longer wavelength because space expanded while it was in transit. The same effect decelerates particles that are not bound together.</p>

<h3>Bound vs. Unbound Systems</h3>
<p>The key physical result: gravitationally bound systems resist expansion. A binary orbit where the binding energy exceeds the Hubble kinetic energy stays together, while unbound particles are swept apart.</p>
<p>This is exactly how large-scale cosmic structure forms — dense regions collapse under gravity while the expanding background carries diffuse matter apart, creating a cosmic web of galaxies, filaments, and voids.</p>

<h3>Limitations</h3>
<p>This implementation uses a constant $H$ (de Sitter expansion). The real universe has $H(t)$ that evolves with matter, radiation, and dark energy content. The simulation locks boundary mode to "despawn" when expansion is active, since periodic boundaries would conflict with the outward flow.</p>
`,
    },

    tidallocking: {
        title: 'Tidal Locking',
        body: `
<p>Tidal locking is the process by which a body's rotation synchronizes with its orbital period, so it always shows the same face to its companion. The Moon is tidally locked to Earth — it rotates exactly once per orbit, which is why we always see the same side.</p>

<h3>Tidal Torque</h3>
<p>$$\\tau = -\\text{TIDAL\\_STRENGTH}\\cdot\\frac{C^2\\,R^3}{r^6}\\,(\\omega_{\\text{spin}}-\\omega_{\\text{orbit}})$$</p>
<p>where the coupling $C$ combines gravitational and electrostatic tidal fields:</p>
<p>$$C = m_{\\text{other}} + \\frac{q_1 q_2}{m}$$</p>
<p>The $r^{-6}$ dependence makes tidal torque extremely sensitive to distance — halving the separation increases it 64-fold.</p>

<h3>Mechanism</h3>
<p>A non-synchronous body develops a tidal bulge that is slightly displaced from the line connecting the two bodies (due to finite viscous response time). The gravitational pull on this displaced bulge creates a torque that transfers angular momentum between the body's spin and the orbit:</p>
<ul>
<li>If $\\omega_{\\text{spin}} > \\omega_{\\text{orbit}}$: the body spins down, transferring angular momentum to the orbit (pushing the companion outward)</li>
<li>If $\\omega_{\\text{spin}} < \\omega_{\\text{orbit}}$: the body spins up, extracting angular momentum from the orbit (pulling the companion inward)</li>
</ul>
<p>Equilibrium is $\\omega_{\\text{spin}} = \\omega_{\\text{orbit}}$: tidal lock.</p>

<h3>Mixed Coupling</h3>
<p>The $C^2$ factor includes four cross-terms: gravity-gravity, gravity-EM, EM-gravity, and EM-EM tidal coupling. For purely gravitational systems, $C = m_{\\text{other}}$. For charged systems, the electrostatic tidal field can enhance or counteract the gravitational tide depending on the sign of $q_1 q_2/m$.</p>
`,
    },

    disintegration: {
        title: 'Disintegration & Roche Limit',
        body: `
<p>A body approaching a more massive companion can be torn apart by tidal forces when the differential gravity across its diameter exceeds its own self-gravity. The critical distance at which this occurs is the Roche limit.</p>

<h3>Disintegration Criterion</h3>
<p>A particle fragments when combined disruptive stresses exceed self-binding:</p>
<p>$$\\underbrace{\\frac{M_{\\text{other}}\\cdot R}{d^3}}_{\\text{tidal}} + \\underbrace{\\omega^2 R}_{\\text{centrifugal}} + \\underbrace{\\frac{q^2}{4R^2}}_{\\text{Coulomb self-repulsion}} > \\underbrace{\\frac{m}{R^2}}_{\\text{self-gravity}}$$</p>
<p>Each term represents a different mechanism trying to tear the particle apart, balanced against the gravitational binding that holds it together.</p>

<h3>Roche Lobe Overflow</h3>
<p>Before catastrophic disruption, a particle can gradually transfer mass to its companion through the <em>Roche lobe</em> — the teardrop-shaped region around each body where its own gravity dominates. The lobe radius (Eggleton formula):</p>
<p>$$r_R \\approx 0.462\\,d\\left(\\frac{m}{m+M}\\right)^{1/3}$$</p>
<p>When a particle's radius exceeds $r_R$, surface material feels a stronger pull toward the companion and flows through the inner Lagrange point (L1) at a rate proportional to the overflow.</p>

<h3>Astrophysical Context</h3>
<p>Roche lobe overflow powers some of the most dramatic phenomena in astrophysics:</p>
<ul>
<li><b>Cataclysmic variables</b>: a white dwarf accreting from a red giant companion, sometimes detonating as a Type Ia supernova</li>
<li><b>X-ray binaries</b>: a neutron star or black hole accreting material heated to X-ray temperatures</li>
<li><b>Saturn's rings</b>: likely formed when a moon crossed inside Saturn's Roche limit and was torn apart by tidal forces</li>
</ul>
`,
    },

    barneshut: {
        title: 'Barnes-Hut Algorithm',
        body: `
<p>Direct computation of all pairwise forces scales as $O(N^2)$ — doubling the particle count quadruples the work. The Barnes-Hut algorithm reduces this to $O(N\\log N)$ using a spatial tree, enabling simulations with hundreds of particles at interactive framerates.</p>

<h3>Quadtree Construction</h3>
<p>The domain is recursively subdivided into four quadrants. Each leaf holds at most 4 particles. Internal nodes store aggregate properties: total mass, total charge, center of mass, total magnetic moment, and total angular momentum.</p>

<h3>Opening Angle Criterion</h3>
<p>When computing the force on a particle, the tree is walked from the root. At each node:</p>
<p>$$\\frac{s}{d} < \\theta \\qquad (\\theta = 0.5)$$</p>
<p>If the cell size $s$ divided by the distance $d$ to the cell's center of mass is less than $\\theta$, the entire group is treated as a single body using its aggregate properties. Otherwise, the node is opened and its children examined.</p>

<h3>Accuracy vs. Performance</h3>
<table style="width:100%;border-collapse:collapse;margin:12px 0">
<tr><th style="text-align:left;padding:4px 8px">Mode</th><th style="text-align:left;padding:4px 8px">Scaling</th><th style="text-align:left;padding:4px 8px">Conservation</th></tr>
<tr><td style="padding:4px 8px">Pairwise (off)</td><td style="padding:4px 8px">$O(N^2)$</td><td style="padding:4px 8px">Machine precision</td></tr>
<tr><td style="padding:4px 8px">Barnes-Hut (on)</td><td style="padding:4px 8px">$O(N\\log N)$</td><td style="padding:4px 8px">Approximate</td></tr>
</table>
<p>With Barnes-Hut off, Newton's third law is exploited exactly (each pair computed once), so momentum and angular momentum are conserved to floating-point precision. With Barnes-Hut on, the asymmetric force evaluation breaks exact reciprocity, introducing small conservation drift.</p>

<h3>Implementation</h3>
<p>The tree uses a flat Structure-of-Arrays layout with pre-allocated typed arrays for zero garbage collection. Nodes are pooled and reset each substep rather than allocated and freed, keeping the per-substep tree rebuild fast and free of GC pauses.</p>
`,
    },

    collision: {
        title: 'Collision Modes',
        body: `
<h3>Pass</h3>
<p>Particles move through each other freely — no collision detection. Useful for studying pure force dynamics without contact effects, or for maximizing performance with many particles.</p>

<h3>Bounce (Hertz Contact)</h3>
<p>Overlapping particles feel a repulsive contact force modeled by Hertz contact mechanics:</p>
<p>$$F = K\\,\\delta^{3/2}$$</p>
<p>where $\\delta$ is the overlap depth and $K$ is the stiffness. The 3/2 exponent comes from the elastic deformation of spheres — it produces a stiffer response at deeper overlaps, preventing excessive penetration. Tangential friction transfers angular momentum between spinning particles during contact.</p>

<h3>Merge (Inelastic Collision)</h3>
<p>Overlapping particles combine into a single particle, conserving:</p>
<ul>
<li><b>Mass</b>: $m = m_1 + m_2$</li>
<li><b>Charge</b>: $q = q_1 + q_2$</li>
<li><b>Momentum</b>: $m\\mathbf{w} = m_1\\mathbf{w}_1 + m_2\\mathbf{w}_2$</li>
<li><b>Angular momentum</b>: $I\\omega = I_1\\omega_1 + I_2\\omega_2 + \\text{orbital}$</li>
</ul>

<h3>Antimatter Annihilation</h3>
<p>When a matter and antimatter particle merge, the lesser mass is annihilated from both, converting rest mass energy $E = 2m_{\\text{annihilated}}$ into a burst of photons.</p>
`,
    },

    boundary: {
        title: 'Boundary Modes',
        body: `
<h3>Despawn</h3>
<p>Particles are removed when they leave the viewport. Models an open system where particles can escape to infinity. Required when cosmological expansion is active.</p>

<h3>Loop (Periodic Boundaries)</h3>
<p>Particles exiting one side re-enter from the opposite side, creating a topologically closed space. Forces use the <em>minimum image convention</em>: each particle interacts with the nearest copy (real or periodic ghost) of every other particle. Three topologies are available when using loop boundaries.</p>

<h3>Bounce (Elastic Walls)</h3>
<p>Walls exert Hertz contact repulsion on approaching particles:</p>
<p>$$F = K\\,\\delta^{3/2}$$</p>
<p>where $\\delta$ is the depth of wall penetration. Tangential friction from wall sliding transfers torque to spinning particles, gradually slowing their rotation. Creates a bounded billiard-like domain.</p>
`,
    },

    topology: {
        title: 'Surface Topology',
        body: `
<p>When boundaries are set to "Loop," the simulation domain becomes a closed 2D surface. The choice of topology determines how the edges are identified — and whether the space is orientable.</p>

<h3>Torus ($T^2$)</h3>
<p>Both pairs of edges wrap normally: right$\\leftrightarrow$left, top$\\leftrightarrow$bottom. The familiar "Pac-Man" topology. The torus is orientable — a clockwise-spinning particle remains clockwise after wrapping. Forces use the minimum image convention.</p>

<h3>Klein Bottle ($K^2$)</h3>
<p>The x-axis wraps normally, but the y-axis wraps with a reflection: exiting the top re-enters from the bottom with x-coordinate mirrored and horizontal velocity reversed. The Klein bottle is <em>non-orientable</em> — a clockwise-spinning particle becomes counterclockwise after a y-boundary crossing. This surface cannot be embedded in 3D without self-intersection.</p>

<h3>Real Projective Plane ($\\mathbb{RP}^2$)</h3>
<p>Both axes wrap with a perpendicular flip — each crossing reverses the perpendicular velocity component and reflects the perpendicular coordinate. $\\mathbb{RP}^2$ is the most exotic topology: non-orientable, and the only closed 2D surface where <em>every</em> closed loop is orientation-reversing. Force computation requires checking 4 minimum-image candidates.</p>
`,
    },

    external: {
        title: 'External Background Fields',
        body: `
<p>Uniform fields that pervade the entire domain, acting on every particle independently of the particle-particle force toggles.</p>

<h3>Uniform Gravitational Field</h3>
<p>$$\\mathbf{F} = m\\mathbf{g}$$</p>
<p>All particles experience the same acceleration $\\mathbf{g}$ regardless of mass — the equivalence principle in action. Direction is set in degrees (0$^\\circ$ = right, 90$^\\circ$ = down). Models surface gravity, projectile motion, or any uniform gravitational environment.</p>

<h3>Uniform Electric Field</h3>
<p>$$\\mathbf{F} = q\\mathbf{E}$$</p>
<p>Accelerates particles proportional to their charge. Opposite charges deflect in opposite directions, enabling beam separation and drift velocity experiments. Neutral particles are unaffected.</p>

<h3>Uniform Magnetic Field ($B_z$)</h3>
<p>An out-of-plane magnetic field produces cyclotron motion:</p>
<p>$$\\omega_c = \\frac{qB}{m}, \\qquad r_L = \\frac{mv_\\perp}{qB}$$</p>
<p>The Larmor radius $r_L$ grows with mass and speed; heavier or faster particles orbit in larger circles. The Boris integrator handles the rotation exactly, preserving kinetic energy through every gyration.</p>

<h3>$\\mathbf{E}\\times\\mathbf{B}$ Drift</h3>
<p>Combining electric and magnetic fields produces a drift perpendicular to both: all particles drift at speed $E/B$ regardless of charge or mass. This is a fundamental result of plasma physics, responsible for particle confinement in tokamaks and the dynamics of the magnetosphere.</p>
`,
    },

    interaction: {
        title: 'Spawn Modes',
        body: `
<h3>Place</h3>
<p>Click to spawn a particle at rest at the cursor position. The simplest mode for building static configurations.</p>

<h3>Shoot</h3>
<p>Click and drag to set the initial velocity vector. Direction points from click to release; speed scales with drag distance. Use this to launch projectiles, create eccentric orbits, or collide particles at controlled speeds.</p>

<h3>Orbit</h3>
<p>Spawns a particle in a circular orbit around the nearest massive body. The velocity is automatically set to:</p>
<p>$$v = \\sqrt{\\frac{M}{r}}$$</p>
<p>where $M$ is the mass of the nearest body and $r$ is the distance to it. This gives a perfectly circular Keplerian orbit. With relativistic corrections or other forces, the orbit will precess or decay.</p>
`,
    },

    spin: {
        title: 'Particle Spin',
        body: `
<p>Each particle rotates as a uniform-density solid sphere with moment of inertia:</p>
<p>$$I = \\frac{2}{5}mr^2$$</p>
<p>The spin slider sets the angular celerity $W$ (range $-0.99$ to $+0.99$). Positive values mean counterclockwise rotation.</p>

<h3>Relativistic Spin</h3>
<p>Angular celerity maps to angular velocity:</p>
<p>$$\\omega = \\frac{W}{\\sqrt{1+W^2 r^2}}$$</p>
<p>This guarantees $|\\omega|r < c$, preventing unphysical surface speeds even at extreme spin values. At low spin, $\\omega \\approx W$.</p>

<h3>Derived Quantities</h3>
<p>Spin determines two important physical properties:</p>
<ul>
<li><b>Magnetic moment</b>: $\\mu = q\\omega r^2/5$ — the source of magnetic dipole interactions</li>
<li><b>Angular momentum</b>: $L = 2m\\omega r^2/5$ — the source of gravitomagnetic dipole interactions</li>
</ul>
<p>Spin evolves dynamically under torques from tidal locking, frame dragging, and spin-orbit energy transfer.</p>
`,
    },

    energy: {
        title: 'Energy Conservation',
        body: `
<p>Total energy is tracked as the sum of five components:</p>
<ul>
<li><b>Linear KE</b>: $\\sum_i(\\gamma_i-1)m_i$ — relativistic translational kinetic energy</li>
<li><b>Spin KE</b>: $\\sum_i\\frac{1}{2}I_i\\omega_i^2$ — rotational kinetic energy</li>
<li><b>Potential</b>: gravitational + Coulomb + magnetic/GM dipole + 1PN corrections + Yukawa</li>
<li><b>Field</b>: Darwin velocity-dependent corrections at $O(v^2/c^2)$</li>
<li><b>Radiated</b>: cumulative energy carried away by photons and gravitons</li>
</ul>

<h3>Conservation Quality</h3>
<p>Energy is exactly conserved (to machine precision) with gravity + Coulomb only, in pairwise mode. Additional forces affect conservation differently:</p>
<ul>
<li><b>Magnetic / GM</b>: velocity-dependent forces carry energy in fields not fully modeled — small drift expected</li>
<li><b>Radiation</b>: energy leaves the system (tracked in the "Radiated" line)</li>
<li><b>Barnes-Hut</b>: approximate force evaluation breaks exact symmetry — small drift</li>
<li><b>Axion</b>: external oscillating field injects and extracts energy — no conservation expected</li>
<li><b>Expansion</b>: Hubble drag dissipates peculiar kinetic energy — no conservation expected</li>
</ul>
<p>The "Drift" line tracks cumulative numerical error as a percentage of the initial total energy, giving a real-time measure of simulation accuracy.</p>
`,
    },

    pion: {
        title: 'Pion Exchange (Yukawa Force Carriers)',
        body: `
<p>The Yukawa potential was proposed in 1935 by Hideki Yukawa, who predicted that the short-range nuclear force must be mediated by a massive particle — the <b>pion</b>. In this simulation, pions are emitted automatically during Yukawa interactions as massive force carriers, analogous to how photons mediate electromagnetic radiation.</p>

<h3>Emission: Scalar Larmor Radiation</h3>
<p>A particle accelerating under the Yukawa force radiates pions with power:</p>
<p>$$P = \\frac{1}{3}\\,g^2 m^2 a^2 = \\frac{1}{3}\\,g^2 F_{\\text{Yuk}}^2$$</p>
<p>The scalar charge is $Q = gm$ (since the Yukawa coupling is proportional to mass), so $Q^2 a^2 = g^2 m^2 (F/m)^2 = g^2 F^2$. The factor of $\\frac{1}{3}$ comes from integrating the $\\cos^2\\theta$ angular pattern of spin-0 radiation over the sphere — compared to $\\frac{2}{3}$ for the $\\sin^2\\theta$ dipole pattern of spin-1 (EM) Larmor radiation. The ratio $1:2$ reflects the single polarization state of a scalar vs. two for a photon.</p>

<h3>Pion Mass</h3>
<p>The pion rest mass equals the Yukawa range parameter $\\mu$:</p>
<p>$$m_\\pi = \\mu, \\qquad V(r) = -g^2 \\frac{e^{-\\mu r}}{r}$$</p>
<p>This is Yukawa's key insight: the range of the force ($\\sim 1/\\mu$) is inversely proportional to the mediator mass. Heavier pions mean shorter-range forces.</p>

<h3>Kinematics</h3>
<p>Unlike massless photons ($|v| = c$), pions travel at $v < c$ using proper velocity:</p>
<p>$$\\mathbf{v} = \\frac{\\mathbf{w}}{\\sqrt{1 + w^2}}$$</p>
<p>Gravitational deflection uses the massive-particle geodesic factor $(1 + v^2)$, which correctly reduces to $2\\times$ (null geodesic) as $v \\to c$ and $1\\times$ (Newtonian) as $v \\to 0$.</p>

<h3>Decay</h3>
<p>Pions decay after a finite lifetime:</p>
<ul>
<li>$\\pi^0 \\to 2\\gamma$ — two photons emitted back-to-back perpendicular to the flight direction</li>
<li>$\\pi^\\pm \\to \\gamma$ — one photon along the flight direction (simplified from $\\mu\\nu$)</li>
</ul>

<h3>Radiation Reaction</h3>
<p>When a pion is emitted, the emitting particle's kinetic energy is reduced by the pion's total energy (rest mass + kinetic). This prevents double-counting: the Yukawa force is already computed directly between particles, so the pion emission represents the radiation channel only.</p>
`,
    },

    fieldExcitation: {
        title: 'Field Excitations (Higgs Boson & Axion Particle)',
        body: `
<p>When particles merge, kinetic energy lost in the inelastic collision excites the active scalar fields (Higgs and/or Axion). These excitations propagate as wave packets — the simulation's analog of <b>Higgs bosons</b> and <b>axion particles</b>.</p>

<h3>Mechanism</h3>
<p>The kinetic energy before and after an inelastic merge determines the excitation energy:</p>
<p>$$\\Delta E = \\text{KE}_{\\text{before}} - \\text{KE}_{\\text{after}}$$</p>
<p>This energy is deposited as a Gaussian bump in the field's time derivative $\\dot{\\phi}$ (or $\\dot{a}$):</p>
<p>$$\\dot{\\phi}(\\mathbf{x}) \\mathrel{+}= A \\exp\\!\\left(-\\frac{|\\mathbf{x} - \\mathbf{x}_0|^2}{2\\sigma^2}\\right)$$</p>
<p>where $A = 0.5\\sqrt{\\Delta E}$ and $\\sigma = 2$ grid cells. The existing Klein-Gordon wave equation then propagates the excitation naturally.</p>

<h3>Higgs Boson Analog</h3>
<p>When the Higgs field is active, merge energy creates oscillations around the vacuum expectation value $\\langle\\phi\\rangle = 1$. These ripples are the 2D analog of the Higgs boson — excitations of the field that gives particles their mass. The Mexican hat potential:</p>
<p>$$V(\\phi) = -\\frac{1}{2}\\mu^2\\phi^2 + \\frac{1}{4}\\lambda\\phi^4$$</p>
<p>determines the oscillation frequency: $\\omega = m_H = \\mu\\sqrt{2}$.</p>

<h3>Axion Particle Analog</h3>
<p>When the Axion field is active, merge energy creates oscillations around the vacuum $\\langle a \\rangle = 0$. These propagating wave packets are the simulation's analog of axion particles — quanta of the axion-like scalar field. The quadratic potential:</p>
<p>$$V(a) = \\frac{1}{2}m_a^2 a^2$$</p>
<p>gives simple harmonic oscillation at frequency $\\omega = m_a$.</p>

<h3>Physical Motivation</h3>
<p>In quantum field theory, particles <em>are</em> field excitations. The Higgs boson discovered at the LHC in 2012 is a quantum of the Higgs field; axions (if they exist) would be quanta of an axion-like field. This simulation captures the classical wave analog: localized disturbances that propagate, disperse, and interact with particles through the same coupling that governs the background field.</p>
`,
    },

    conserved: {
        title: 'Conserved Quantities',
        body: `
<h3>Linear Momentum</h3>
<p>$$\\mathbf{P} = \\sum_i m_i\\mathbf{w}_i + \\mathbf{P}_{\\text{field}} + \\mathbf{P}_{\\text{radiated}}$$</p>
<p>Conserved by Noether's theorem from translational symmetry of space. Particle momentum uses proper velocity $\\mathbf{w}$ (which reduces to $m\\mathbf{v}$ when relativity is off). Field and radiated contributions are tracked separately.</p>

<h3>Angular Momentum</h3>
<p>$$J = \\underbrace{\\sum_i\\mathbf{r}_i \\times m_i\\mathbf{w}_i}_{\\text{orbital}} + \\underbrace{\\sum_i I_i W_i}_{\\text{spin}}$$</p>
<p>Conserved from rotational symmetry of space. Computed about the center of mass. Tidal locking, frame dragging, and spin-orbit coupling transfer angular momentum between orbital and spin reservoirs, but the total is preserved.</p>

<h3>When Conservation Breaks</h3>
<p>Velocity-dependent forces (magnetic, gravitomagnetic) carry momentum in electromagnetic and gravitomagnetic fields that the simulation doesn't fully track, producing small drift. This is a fundamental limitation of the particle-only approach — a full field theory would restore exact conservation. The drift percentage quantifies this effect.</p>
`,
    },
};
