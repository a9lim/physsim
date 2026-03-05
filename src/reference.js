// ─── Reference Pages ───
// Extended content for each physics concept, shown via Shift+click on info buttons.

export const REFERENCE = {
    gravity: {
        title: 'Gravity',
        body: `
<p>Newtonian gravity between two massive particles:</p>
<p>$$\\mathbf{F} = \\frac{m_1 m_2}{r^2} \\hat{r}$$</p>
<p>With $G = 1$ in natural units. Always attractive between all massive particles. Plummer-softened with $\\epsilon^2 = 64$ to prevent singularities at close approach.</p>
<h3>Potential Energy</h3>
<p>$$U = -\\frac{m_1 m_2}{r}$$</p>
<h3>Circular Orbits</h3>
<p>For a test particle orbiting mass $M$: $v_{\\text{circ}} = \\sqrt{M/r}$. The period is $T = 2\\pi\\sqrt{r^3/M}$ (Kepler's third law).</p>
`,
    },
    coulomb: {
        title: 'Coulomb Force',
        body: `
<p>Electrostatic force between two charged particles:</p>
<p>$$\\mathbf{F} = -\\frac{q_1 q_2}{r^2} \\hat{r}$$</p>
<p>Like charges repel, opposite charges attract. Combined with gravity, this enables atom-like bound states.</p>
<h3>Potential Energy</h3>
<p>$$U = +\\frac{q_1 q_2}{r}$$</p>
`,
    },
    magnetic: {
        title: 'Magnetic Interactions',
        body: `
<p>Two components of magnetic interaction:</p>
<h3>Lorentz Force</h3>
<p>Moving charges generate magnetic fields. A charge $q$ moving through field $\\mathbf{B}$ feels:</p>
<p>$$\\mathbf{F} = q(\\mathbf{v} \\times \\mathbf{B})$$</p>
<p>Handled exactly by the Boris integrator, which preserves $|\\mathbf{v}|$ through rotation.</p>
<h3>Magnetic Dipole Interaction</h3>
<p>Spinning charged particles create dipole moments $\\mu = q\\omega r^2/5$:</p>
<p>$$F = \\frac{3\\mu_1 \\mu_2}{r^4}$$</p>
<p>Aligned perpendicular-to-plane dipoles repel (unlike gravitomagnetic, where they attract).</p>
`,
    },
    gravitomag: {
        title: 'Gravitomagnetism',
        body: `
<p>The gravitational analog of magnetism from general relativity's gravitoelectromagnetic (GEM) framework.</p>
<h3>Linear Gravitomagnetic Force</h3>
<p>Moving masses generate a gravitomagnetic field $\\mathbf{B}_g$. A mass $m$ moving through it feels:</p>
<p>$$\\mathbf{F} = 4m(\\mathbf{v} \\times \\mathbf{B}_g)$$</p>
<p>The factor of 4 (vs 1 for EM) arises from gravity being spin-2.</p>
<h3>GM Dipole Interaction</h3>
<p>$$F = +\\frac{3L_1 L_2}{r^4}$$</p>
<p>Co-rotating masses attract (opposite sign from EM dipoles).</p>
<h3>Frame-Dragging</h3>
<p>Lense-Thirring torque drives nearby spins toward co-rotation:</p>
<p>$$\\tau = \\frac{2L_s(\\omega_s - \\omega_p)}{r^3}$$</p>
`,
    },
    relativity: {
        title: 'Special Relativity',
        body: `
<p>When enabled, the simulation uses proper velocity $\\mathbf{w} = \\gamma\\mathbf{v}$ as the fundamental state variable.</p>
<p>Coordinate velocity is derived as:</p>
<p>$$\\mathbf{v} = \\frac{\\mathbf{w}}{\\sqrt{1 + w^2}}$$</p>
<p>This automatically enforces $|v| < c$ regardless of applied forces. The Lorentz factor is $\\gamma = \\sqrt{1 + w^2}$.</p>
<h3>Relativistic Kinetic Energy</h3>
<p>$$\\text{KE} = (\\gamma - 1)mc^2 = \\frac{w^2}{\\gamma + 1}m$$</p>
<p>The second form avoids catastrophic cancellation at low velocities.</p>
`,
    },
    radiation: {
        title: 'Electromagnetic Radiation',
        body: `
<p>Accelerating charges radiate energy via the Larmor formula:</p>
<p>$$P = \\frac{2q^2 a^2}{3}$$</p>
<h3>Landau-Lifshitz Reaction Force</h3>
<p>The radiation reaction is modeled with three terms:</p>
<p>$$\\mathbf{F}_{\\text{rad}} = \\tau\\left[\\frac{\\dot{\\mathbf{F}}}{\\gamma^3} - \\frac{\\mathbf{v}F^2}{m\\gamma^2} + \\frac{\\mathbf{F}(\\mathbf{v}\\cdot\\mathbf{F})}{m\\gamma^4}\\right]$$</p>
<p>where $\\tau = 2q^2/(3m)$.</p>
<p>The jerk term uses analytical derivatives for gravity and Coulomb, plus finite differences for other forces. Clamped to 50% of external force for perturbative validity.</p>
`,
    },
    onepn: {
        title: '1PN Corrections',
        body: `
<p>First post-Newtonian $O(v^2/c^2)$ corrections from three sectors:</p>
<h3>EIH (Gravity)</h3>
<p>Einstein-Infeld-Hoffmann force produces perihelion precession:</p>
<p>$$\\Delta\\phi \\approx \\frac{6\\pi M}{a(1-e^2)} \\text{ rad/orbit}$$</p>
<h3>Darwin EM</h3>
<p>Velocity-dependent corrections from the Darwin Lagrangian, beyond the Lorentz force.</p>
<h3>Bazanski Cross-Term</h3>
<p>Mixed gravity-EM interaction:</p>
<p>$$F = \\frac{q_1 q_2(m_1+m_2) - (q_1^2 m_2 + q_2^2 m_1)}{r^3}\\hat{r}$$</p>
<p>All three use velocity-Verlet integration for second-order accuracy.</p>
`,
    },
    signaldelay: {
        title: 'Signal Delay',
        body: `
<p>Forces propagate at the speed of light ($c = 1$) instead of acting instantaneously.</p>
<p>The light-cone equation solved for each pair:</p>
<p>$$|\\mathbf{x}_{\\text{src}}(t_{\\text{ret}}) - \\mathbf{x}_{\\text{obs}}| = t_{\\text{now}} - t_{\\text{ret}}$$</p>
<h3>Three-Phase Solver</h3>
<ol>
<li><b>Newton-Raphson</b>: 6 iterations to locate the correct history segment</li>
<li><b>Quadratic solve</b>: exact solution on the piecewise-linear trajectory segment</li>
<li><b>Extrapolation</b>: constant-velocity fallback when history is exhausted</li>
</ol>
<p>History buffers: 256 entries per particle, recorded every 64 update calls.</p>
`,
    },
    blackhole: {
        title: 'Black Hole Mode',
        body: `
<p>All particles become black holes with Kerr-Newman horizons:</p>
<p>$$r_+ = M + \\sqrt{M^2 - a^2 - Q^2}$$</p>
<p>where $a = J/M$ is the spin parameter and $Q$ is charge.</p>
<h3>Hawking Radiation</h3>
<p>Temperature depends on surface gravity:</p>
<p>$$T = \\frac{\\kappa}{2\\pi}, \\quad \\kappa = \\frac{r_+ - r_-}{2(r_+^2 + a^2)}$$</p>
<p>Extremal black holes ($M^2 = a^2 + Q^2$) have zero temperature and stop radiating.</p>
<h3>Ergosphere</h3>
<p>Shown as a dashed purple ring at:</p>
<p>$$r_{\\text{ergo}} = M + \\sqrt{M^2 - a^2}$$</p>
`,
    },
    spinorbit: {
        title: 'Spin-Orbit Coupling',
        body: `
<p>Couples translational and rotational motion through field gradients.</p>
<h3>Stern-Gerlach Force (EM)</h3>
<p>$$\\mathbf{F} = \\mu \\nabla B_z$$</p>
<h3>Mathisson-Papapetrou Force (Gravitational)</h3>
<p>$$\\mathbf{F} = -L \\nabla B_{gz}$$</p>
<p>The GEM sign flip means spinning masses are deflected opposite to spinning charges.</p>
<h3>Energy Transfer</h3>
<p>Moving through non-uniform fields transfers energy between orbit and spin:</p>
<p>$$dE = -\\mu(\\mathbf{v} \\cdot \\nabla B_z)dt$$</p>
`,
    },
    yukawa: {
        title: 'Yukawa Potential',
        body: `
<p>A screened potential modeling short-range nuclear forces:</p>
<p>$$V(r) = -\\frac{g^2 e^{-\\mu r}}{r}$$</p>
<p>$$\\mathbf{F} = -g^2 m_1 m_2 \\frac{e^{-\\mu r}}{r^2}(1 + \\mu r)\\hat{r}$$</p>
<p>At short range ($\\mu r \\ll 1$) it behaves like gravity. At long range ($\\mu r \\gg 1$) it vanishes exponentially.</p>
<p>The coupling $g^2$ sets the strength and $\\mu$ (the mediator mass) sets the range $\\lambda = 1/\\mu$.</p>
`,
    },
    axion: {
        title: 'Axion Dark Matter Coupling',
        body: `
<p>Models dark matter axions as a coherently oscillating background field:</p>
<p>$$a(t) = a_0 \\cos(m_a t)$$</p>
<p>This modulates the electromagnetic coupling constant:</p>
<p>$$\\alpha_{\\text{eff}} = \\alpha(1 + g\\cos(m_a t))$$</p>
<p>All Coulomb and magnetic forces oscillate with the axion field. Energy is not conserved since the axion field acts as an external reservoir.</p>
<p>This is the exact effect searched for by axion detection experiments like CASPEr and ABRACADABRA.</p>
`,
    },
    gwradiation: {
        title: 'Gravitational Wave Radiation',
        body: `
<p>Gravitational wave emission from the mass quadrupole moment:</p>
<p>$$P_{\\text{GW}} = \\frac{1}{5}|\\dddot{I}_{ij}|^2$$</p>
<p>where $I_{ij} = \\sum m(x_i x_j - \\delta_{ij}r^2/3)$ is the reduced quadrupole moment tensor.</p>
<h3>Peters Formula (Circular Binary)</h3>
<p>$$P = \\frac{32}{5}\\frac{m_1^2 m_2^2(m_1+m_2)}{r^5}$$</p>
<p>This causes orbital inspiral and eventual merger, exactly as detected by LIGO.</p>
<h3>EM Quadrupole</h3>
<p>When radiation is enabled, also computes:</p>
<p>$$P_{\\text{EM}} = \\frac{1}{180}|\\dddot{Q}_{ij}|^2$$</p>
<p>where $Q_{ij} = \\sum q \\cdot x_i x_j$ is the charge quadrupole.</p>
`,
    },
    expansion: {
        title: 'Cosmological Expansion',
        body: `
<p>Implements Hubble flow from the domain center:</p>
<p>$$\\mathbf{v}_H = H \\cdot \\mathbf{r}$$</p>
<p>Distant particles separate while bound systems resist expansion.</p>
<h3>Hubble Drag</h3>
<p>Peculiar velocities redshift over time:</p>
<p>$$\\mathbf{v}_{\\text{pec}} \\to \\mathbf{v}_{\\text{pec}}(1 - H\\,dt)$$</p>
<p>This models the cosmological redshift of non-comoving particles, matching real N-body cosmological simulations.</p>
`,
    },
    tidallocking: {
        title: 'Tidal Locking',
        body: `
<p>Dissipative torque drives spin toward synchronous rotation:</p>
<p>$$\\tau \\propto -\\frac{(M + q_1 q_2/m)^2 R^3}{r^6}\\Delta\\omega$$</p>
<p>where $\\Delta\\omega = \\omega_{\\text{spin}} - \\omega_{\\text{orbit}}$.</p>
<p>The mixed coupling $(M + q_1 q_2/m)^2$ captures all four cross-terms between gravitational and electrostatic tidal fields.</p>
`,
    },
    tidal: {
        title: 'Tidal Disintegration',
        body: `
<p>Particles break apart when disruptive forces exceed self-gravity:</p>
<p>$$\\frac{\\text{TIDAL\\_STRENGTH} \\cdot M_{\\text{other}} \\cdot r}{d^3} + \\omega^2 r + \\frac{q^2}{4r^2} > \\frac{m}{r^2}$$</p>
<p>Also includes Roche lobe overflow: when a particle fills its Roche lobe (Eggleton formula $r_R \\approx 0.462\\,d\\,(m/(m+M))^{1/3}$), it continuously transfers mass toward the companion through the L1 point.</p>
`,
    },
    barneshut: {
        title: 'Barnes-Hut Algorithm',
        body: `
<p>An $O(N\\log N)$ approximation using a quadtree. Distant groups of particles are treated as single bodies when:</p>
<p>$$\\frac{\\text{size}}{d} < \\theta \\quad (\\theta = 0.5)$$</p>
<p>Enables hundreds of particles to run smoothly. When off, every pair is computed exactly ($O(N^2)$), giving perfect conservation.</p>
`,
    },
    collision: {
        title: 'Collision Modes',
        body: `
<h3>Pass</h3><p>Particles move through each other freely.</p>
<h3>Bounce</h3><p>Elastic collision with spin-dependent surface friction. Relativistic bounces use Lorentz-boosted normal components.</p>
<h3>Merge</h3><p>Overlapping particles combine, conserving total mass, charge, momentum, and angular momentum.</p>
`,
    },
    boundary: {
        title: 'Boundary Modes',
        body: `
<h3>Despawn</h3><p>Particles removed when leaving viewport.</p>
<h3>Loop</h3><p>Periodic boundaries with topology selection.</p>
<h3>Bounce</h3><p>Elastic reflection off edges.</p>
`,
    },
    topology: {
        title: 'Surface Topology',
        body: `
<h3>Torus ($T^2$)</h3><p>Both axes wrap normally (Pac-Man style).</p>
<h3>Klein Bottle ($K$)</h3><p>y-wrap mirrors x-coordinate and reverses horizontal velocity. Non-orientable.</p>
<h3>Real Projective Plane ($\\mathbb{RP}^2$)</h3><p>Both axes wrap with perpendicular flip. The only closed 2D surface where every loop is orientation-reversing.</p>
`,
    },
    interaction: {
        title: 'Spawn Modes',
        body: `
<h3>Place</h3><p>Click to spawn at rest.</p>
<h3>Shoot</h3><p>Drag to set initial velocity (scale: 0.02).</p>
<h3>Orbit</h3><p>Spawns in circular orbit around nearest massive body at $v = \\sqrt{M/r}$.</p>
`,
    },
    spin: {
        title: 'Particle Spin',
        body: `
<p>Each particle spins as a solid sphere with moment of inertia $I = \\frac{2}{5}mr^2$.</p>
<p>The angular celerity $W$ maps to angular velocity:</p>
<p>$$\\omega = \\frac{W}{\\sqrt{1 + W^2 r^2}}$$</p>
<p>This keeps surface speed below $c$. Spin determines:</p>
<ul>
<li>Magnetic moment: $\\mu = q\\omega r^2/5$</li>
<li>Angular momentum: $L = 2m\\omega r^2/5$</li>
</ul>
`,
    },
    energy: {
        title: 'Energy Conservation',
        body: `
<p>Total energy is the sum of five components:</p>
<ul>
<li><b>Linear KE</b>: $\\sum(\\gamma - 1)m$</li>
<li><b>Spin KE</b>: $\\sum \\frac{1}{2}I\\omega^2$ (relativistic form uses angular celerity)</li>
<li><b>Potential</b>: gravity + Coulomb + dipole + 1PN corrections</li>
<li><b>Field</b>: Darwin corrections at $O(v^2/c^2)$</li>
<li><b>Radiated</b>: energy carried by photons</li>
</ul>
<p>Exactly conserved with gravity + Coulomb only in pairwise mode.</p>
`,
    },
    conserved: {
        title: 'Conserved Quantities',
        body: `
<p>Tracked quantities with particle, field, and radiated contributions:</p>
<ul>
<li><b>Momentum</b>: $\\sum m_i \\mathbf{w}_i$ + field + radiated</li>
<li><b>Angular momentum</b>: $\\sum \\mathbf{r}_i \\times m_i \\mathbf{w}_i + \\sum I_i W_i$ about COM</li>
</ul>
<p>Velocity-dependent forces carry momentum in fields the sim doesn't fully model, so small drift is expected.</p>
`,
    },
};
