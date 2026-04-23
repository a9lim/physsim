# Geon

Interactive particle physics simulator that runs entirely in the browser. You can spawn particles, toggle a variety of forces on and off, and watch the physics play out in real time. It covers gravity, electromagnetism, nuclear forces, scalar fields, relativistic corrections, black hole mechanics, and more. Everything runs on a WebGPU compute backend with an automatic CPU fallback.

**[Try it](https://a9l.im/geon)** | Part of the [a9l.im](https://a9l.im) portfolio

## Physics

2D optionally-relativistic N-body dynamics in natural units (c = G = ℏ = 1). Particles store proper velocity (gamma * v), so the speed of light is enforced automatically. Boris integrator with adaptive substepping (up to 32 substeps per frame).

### Forces

- **Newtonian gravity**: inverse-square with softening
- **Coulomb**: electrostatic attraction or repulsion
- **Magnetic dipole**: spin-dependent B-field interactions
- **Gravitomagnetic**: frame-dragging (also called the Lense-Thirring effect)
- **Lorentz**: velocity-dependent magnetic force
- **Tidal locking**: torque synchronizing spin to orbital frequency
- **Yukawa**: screened nuclear force with exponential decay (massive pion mediator)
- **Spin-orbit coupling**: spin-curvature and Thomas precession
- **External fields**: uniform gravity, electric field, and magnetic Bz
- **Higgs field gradients**: mass modulation from scalar field coupling
- **Axion field coupling**: position-dependent electromagnetic coupling (aF^2)
- **Boson-particle gravity**: photons and pions gravitationally lensed by massive particles
- **Boson-boson gravity**: mutual gravitational interaction between bosons
- **Pion-pion Coulomb**: charged pion electrostatic interactions
- **Cosmological expansion**: Hubble flow with adjustable H parameter

### Relativistic Corrections

- **1PN (post-Newtonian)**: Einstein-Infeld-Hoffmann for gravity, Darwin for EM, Bazanski for gravity-EM cross-terms, scalar Breit for Yukawa
- **Signal delay**: finite-speed force propagation solving for the delayed position on the past light cone, with history buffers (256 snapshots, stride 64)

### Scalar Fields

Two scalar fields on smooth PQS grids (cubic B-spline, 64×64 on CPU, 128×128 on GPU) with self-gravity via FFT convolution:

- **Higgs field**: spontaneous symmetry breaking (Mexican hat potential), mass generation via geometric-mean coupling, field excitation on particle collisions, symmetry restoration at high energy
- **Axion field**: position-dependent EM coupling constants, superradiant amplification by spinning black holes (stimulated and spontaneous), Peccei-Quinn CP-violation dynamics

Both fields evolve via Stormer-Verlet KDK integration, interact with particles and each other, and produce visible wave dynamics.

### Radiation

- **Larmor dipole**: electromagnetic radiation from accelerating charges
- **EM quadrupole**: higher-order electromagnetic radiation
- **Gravitational wave quadrupole**: energy and angular momentum loss from mass quadrupole oscillation
- **Pion emission**: Yukawa-mediated boson exchange between nuclear-range particles
- **Photon and pion gravitational lensing**: bosons follow curved trajectories near massive bodies

### Black Holes

- **Kerr-Newman horizons**: mass, spin, and charge determine horizon radius with cosmic censorship
- **Ergosphere visualization**: rendered as a distinct region around spinning black holes
- **Hawking radiation**: thermal emission with inverse-mass-squared rate, evaporating small black holes
- **Schwinger discharge**: vacuum pair production at charged black hole horizons 
- **Pair production**: high-energy photon conversion to particle-antiparticle pairs
- **Superradiance**: axion field amplification extracting rotational energy from spinning black holes, with natural saturation when horizon angular velocity drops below axion mass

### Topology and Boundary Conditions

- **Despawn**: particles removed at edges
- **Bounce**: elastic reflection with friction
- **Loop**: periodic wrapping (three topologies below)
  - **Torus (T^2)**: standard periodic boundaries
  - **Klein bottle**: periodic x, glide-reflected y (non-orientable, velocity and spin both flip)
  - **Real projective plane (RP^2)**: both axes glide-reflected (non-orientable)

Ghost particles generated for all periodic topologies to handle cross-boundary forces correctly via minimum-image separation.

### Quantized Charge

All charges quantized in units of BOSON_CHARGE (0.1). Emission, absorption, decay, and disintegration transfer charge in discrete quanta. Conservation enforced throughout.

## Controls

**Mouse**: Click to spawn particles. Drag to launch with velocity. Right-click to select or delete. Scroll to zoom, middle-drag to pan. Hover for a particle tooltip (mass, charge, spin, speed).

**Touch**: Tap to spawn. Two-finger pinch to zoom. Drag to pan.

**Keyboard**: Space to play or pause. X to toggle antimatter mode. `?` for the full shortcut list.

## Sidebar

Four tabs with full simulation control:

- **Settings**: 16 physics toggles with declarative dependency graph, spawn parameter sliders (mass, charge, spin), preset selector, external field controls, and mass sliders for the Yukawa, Higgs, and axion fields
- **Engine**: backend toggle (GPU or CPU), collision mode (pass, merge, or bounce), boundary mode (despawn, bounce, or loop), topology (torus, Klein, or RP^2), sim speed, overlays (trails, velocity vectors, acceleration vectors, acceleration components, potential field heatmap)
- **Stats**: energy breakdown (linear KE, spin KE, rest mass, potential, field, radiated) with conservation drift %, momentum (particle, field, and radiated) with drift %, angular momentum (orbital and spin) with drift %, 11-component per-particle force breakdown
- **Particle**: selected particle details, phase space plot (r vs v_r, 512-sample ring buffer), effective potential V_eff(r) curve (200 samples, all active forces)

## Analysis Tools

- **Heatmap overlay**: real-time potential field visualization (64×64 on CPU, 128×128 on GPU) with modes for gravity, electric, Yukawa, or combined. Diverging colormaps, box blur smoothing, signal-delay-aware when relativity is on. Barnes-Hut tree-walk accelerated.
- **Phase space plot**: radial distance vs radial velocity relative to the most massive body. 512-sample ring buffer traces orbital evolution, showing ellipticity and precession.
- **Effective potential**: V_eff(r) = V(r) + L^2/(2 mu r^2) computed over 0.5x to 4x current separation. Includes centrifugal barrier, potential wells, turning points, and current position marker.
- **Conservation monitor**: energy, momentum, and angular momentum tracked with drift percentages to diagnose numerical issues.

## Save & Load

Quick-save and quick-load (in-memory), download as JSON, or upload via file picker and drag-drop. Saves particle state, all 16 physics toggles, slider values, collision, boundary, and topology modes, and camera position. GPU state is async-readback compatible with field downsampling (128×128 to 64×64).

## Running Locally

```bash
cd path/to/a9lim.github.io && python -m http.server
# -> http://localhost:8000/geon/
```

Serve from the repository root, because shared design files load via absolute paths. There's no build step and no dependencies. ES6 modules require HTTP (not `file://`). Force the CPU backend with `?cpu=1`.

## Tech

Vanilla JavaScript with no dependencies. 40 ES6 modules (~20,000 lines) and 52 WGSL compute and render shaders (~12,000 lines).

**CPU backend**: Canvas 2D rendering, pairwise or Barnes-Hut O(N log N) force evaluation, up to 128 particles, 1,024 photons, 256 pions, and 256 leptons.

**GPU backend**: WebGPU compute with instanced rendering (dual pipelines for light and dark themes). 5-phase compute pipeline covering forces, tree build, advanced physics, scalar fields, and boson lifecycle. Up to 512 particles, 4,096 photons, and 1,024 pions (leptons share the pion pool). 128×128 scalar field and heatmap grids. Lock-free CAS tree insertion, visitor-flag bottom-up aggregation, async statistics readback.

**Integrator**: Boris velocity-preserving magnetic rotation with adaptive substepping. Substep size: min(sqrt(softening/a_max), (2 pi / omega_c) / 8). Stormer-Verlet KDK for scalar field evolution. 1PN velocity-Verlet correction pass.

**Memory**: Zero per-frame allocation. Pre-allocated particle pools, typed array buffers, swap-and-pop deletion, circular history buffers for signal delay.

## Sibling Projects

- [Cyano](https://a9l.im/cyano) ([GitHub](https://github.com/a9lim/cyano))
- [Gerry](https://a9l.im/gerry) ([GitHub](https://github.com/a9lim/gerry))
- [Scripture](https://a9l.im/scripture) ([GitHub](https://github.com/a9lim/scripture))
- [Shoals](https://a9l.im/shoals) ([GitHub](https://github.com/a9lim/shoals))

## License

[AGPL-3.0](LICENSE)
