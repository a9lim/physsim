# Geon

Interactive particle physics simulator running entirely in the browser. Spawn particles, fling them into orbit, toggle forces on and off, and watch the physics play out in real time. Covers gravity, electromagnetism, nuclear forces, scalar fields, relativistic corrections, black hole mechanics, and more -- all rendered live with a WebGPU compute backend (automatic CPU fallback).

**[Try it](https://a9l.im/geon)** | Part of the [a9l.im](https://a9l.im) portfolio

## What You Can Explore

**19 built-in presets** across four categories:

| Category | Presets |
|----------|---------|
| Gravity | Kepler Orbits, Precession, Binary Inspiral, Tidal Lock, Roche Limit, Hawking Evaporation |
| Electromagnetism | Atom, Bremsstrahlung, Magnetic Dipoles |
| Exotic | Atomic Nucleus, Axion Field, Pion Exchange, Higgs Mechanism, Higgs Boson, Axion Burst, Peccei-Quinn, Phase Transition |
| Cosmological | Galaxy, Expanding Universe |

Each preset configures the relevant forces, topology, boundary conditions, and initial particles so the physics is immediately visible. Every parameter is adjustable from the sidebar.

## Physics

Geon simulates 2D relativistic N-body dynamics in natural units (c = G = h = 1). Particles store proper velocity (gamma * v), so the speed of light is enforced automatically. Boris integrator with adaptive substepping (up to 32 substeps per frame).

### Forces

15 force types, all with interdependent toggle logic:

- **Newtonian gravity** -- inverse-square with softening
- **Coulomb** -- electrostatic attraction/repulsion
- **Magnetic dipole** -- spin-dependent B-field interactions
- **Gravitomagnetic** -- frame-dragging / Lense-Thirring effect
- **Lorentz** -- velocity-dependent magnetic force
- **Tidal locking** -- torque synchronizing spin to orbital frequency
- **Yukawa** -- screened nuclear force with exponential decay (massive pion mediator)
- **Spin-orbit coupling** -- spin-curvature and Thomas precession
- **External fields** -- uniform gravity, electric field, and magnetic Bz
- **Higgs field gradients** -- mass modulation from scalar field coupling
- **Axion field coupling** -- position-dependent electromagnetic coupling (aF^2)
- **Boson-particle gravity** -- photons and pions gravitationally lensed by massive particles
- **Boson-boson gravity** -- mutual gravitational interaction between bosons
- **Pion-pion Coulomb** -- charged pion electrostatic interactions
- **Cosmological expansion** -- Hubble flow with adjustable H parameter

### Relativistic Corrections

- **1PN (post-Newtonian)** -- Einstein-Infeld-Hoffmann for gravity, Darwin for EM, Bazanski for gravity-EM cross-terms, scalar Breit for Yukawa
- **Signal delay** -- finite-speed force propagation solving for the retarded position on the past light cone, with history buffers (256 snapshots, stride 64)

### Scalar Fields

Two scalar fields on smooth PQS grids (cubic B-spline, 64x64 CPU / 128x128 GPU) with self-gravity via FFT convolution:

- **Higgs field** -- spontaneous symmetry breaking (Mexican hat potential), mass generation via geometric-mean coupling, field excitation on particle collisions, symmetry restoration at high energy
- **Axion field** -- position-dependent EM coupling constants, superradiant amplification by spinning black holes (stimulated + spontaneous), Peccei-Quinn CP-violation dynamics

Both fields evolve via Stormer-Verlet KDK integration, interact with particles and each other, and produce visible wave dynamics.

### Radiation

- **Larmor dipole** -- electromagnetic radiation from accelerating charges
- **EM quadrupole** -- higher-order electromagnetic radiation
- **Gravitational wave quadrupole** -- energy and angular momentum loss from mass quadrupole oscillation
- **Pion emission** -- Yukawa-mediated boson exchange between nuclear-range particles
- **Photon and pion gravitational lensing** -- bosons follow curved trajectories near massive bodies

### Black Holes

- **Kerr-Newman horizons** -- mass, spin, and charge determine horizon radius with cosmic censorship (super-extremal case clamps to extremal r+ = M)
- **Ergosphere visualization** -- rendered as a distinct region around spinning black holes
- **Hawking radiation** -- thermal emission with inverse-mass-squared rate, evaporating small black holes
- **Schwinger discharge** -- vacuum pair production at charged black hole horizons (rate depends on Kerr-Newman area factor)
- **Pair production** -- high-energy photon conversion to particle-antiparticle pairs
- **Superradiance** -- axion field amplification extracting rotational energy from spinning black holes, with natural saturation when horizon angular velocity drops below axion mass

### Topology

Boundary conditions and non-trivial topologies:

- **Despawn** -- particles removed at edges
- **Bounce** -- elastic reflection with friction
- **Loop** -- periodic wrapping (three topologies below)
  - **Torus (T^2)** -- standard periodic boundaries
  - **Klein bottle** -- periodic x, glide-reflected y (non-orientable, velocity/spin flips)
  - **Real projective plane (RP^2)** -- both axes glide-reflected (non-orientable)

Ghost particles generated for all periodic topologies to handle cross-boundary forces correctly via minimum-image separation.

### Quantized Charge

All charges quantized in units of BOSON_CHARGE (0.1). Emission, absorption, decay, and disintegration transfer charge in discrete quanta. Conservation enforced throughout.

## Controls

**Mouse**: Click to spawn particles. Drag to launch with velocity. Right-click to select or delete. Scroll to zoom, middle-drag to pan. Hover for particle tooltip (mass, charge, spin, speed).

**Touch**: Tap to spawn. Two-finger pinch to zoom. Drag to pan.

**Keyboard**: Space for play/pause. X to toggle antimatter mode. `?` for the full shortcut list.

## Sidebar

Four tabs with full simulation control:

- **Settings** -- 16 physics toggles with declarative dependency graph, spawn parameter sliders (mass, charge, spin), preset selector, external field controls, Yukawa/Higgs/axion mass sliders
- **Engine** -- GPU/CPU toggle, collision mode (pass/merge/bounce), boundary mode (despawn/bounce/loop), topology (torus/Klein/RP^2), sim speed, overlays (trails, velocity vectors, acceleration vectors, acceleration components, potential field heatmap)
- **Stats** -- energy breakdown (linear KE, spin KE, rest mass, potential, field, radiated) with conservation drift %, momentum (particle + field + radiated) with drift %, angular momentum (orbital + spin) with drift %, 11-component per-particle force breakdown
- **Particle** -- selected particle details, phase space plot (r vs v_r, 512-sample ring buffer), effective potential V_eff(r) curve (200 samples, all active forces)

## Analysis Tools

- **Heatmap overlay** -- real-time potential field visualization (64x64 CPU / 128x128 GPU) with modes for gravity, electric, Yukawa, or combined. Diverging colormaps, box blur smoothing, signal-delay-aware when relativity is on. Barnes-Hut tree-walk accelerated.
- **Phase space plot** -- radial distance vs radial velocity relative to the most massive body. 512-sample ring buffer traces orbital evolution, showing ellipticity and precession.
- **Effective potential** -- V_eff(r) = V(r) + L^2/(2 mu r^2) computed over 0.5x to 4x current separation. Includes centrifugal barrier, potential wells, turning points, and current position marker.
- **Conservation monitor** -- energy, momentum, and angular momentum tracked with drift percentages to diagnose numerical issues.

## Save & Load

Quick-save/quick-load (in-memory), download as JSON, or upload via file picker and drag-drop. Saves particle state, all 16 physics toggles, slider values, collision/boundary/topology modes, and camera position. GPU state is async-readback compatible with field downsampling (128x128 to 64x64).

## Running Locally

```bash
cd path/to/a9lim.github.io && python -m http.server
# -> http://localhost:8000/geon/
```

Serve from the repository root -- shared design files load via absolute paths. No build step, no dependencies. ES6 modules require HTTP (not `file://`). Force CPU backend with `?cpu=1`.

## Tech

Zero-dependency vanilla JavaScript. 40 ES6 modules (~20,000 lines) and 52 WGSL compute/render shaders (~12,000 lines).

**CPU backend**: Canvas 2D rendering, pairwise or Barnes-Hut O(N log N) force evaluation, up to 128 particles + 1,024 photons + 256 pions + 256 leptons.

**GPU backend**: WebGPU compute with instanced rendering (dual light/dark pipelines). 5-phase compute pipeline covering forces, tree build, advanced physics, scalar fields, and boson lifecycle. Up to 512 particles + 4,096 photons + 1,024 pions (leptons share pion pool). 128x128 scalar field and heatmap grids. Lock-free CAS tree insertion, visitor-flag bottom-up aggregation, async statistics readback.

**Integrator**: Boris velocity-preserving magnetic rotation with adaptive substepping. Substep size: min(sqrt(softening/a_max), (2 pi / omega_c) / 8). Stormer-Verlet KDK for scalar field evolution. 1PN velocity-Verlet correction pass.

**Memory**: Zero per-frame allocation. Pre-allocated particle pools, typed array buffers, swap-and-pop deletion, circular history buffers for signal delay.

## Sibling Projects

- [Cyano](https://github.com/a9lim/cyano) -- [a9l.im/cyano](https://a9l.im/cyano)
- [Gerry](https://github.com/a9lim/gerry) -- [a9l.im/gerry](https://a9l.im/gerry)
- [Scripture](https://github.com/a9lim/scripture) -- [a9l.im/scripture](https://a9l.im/scripture)
- [Shoals](https://github.com/a9lim/shoals) -- [a9l.im/shoals](https://a9l.im/shoals)

## License

[AGPL-3.0](LICENSE)
