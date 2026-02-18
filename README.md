# Relativistic N-Body Physics Simulation

A high-performance, interactive physics simulation that models gravity and electromagnetism with relativistic effects. It uses the **Barnes-Hut algorithm** for efficient force calculation, allowing for simulations with thousands of particles.

## Features

- **Relativistic Physics**: Particles respect the speed of light ($c$)..
- **Forces**: 
  - **Gravity**: All particles attract each other based on mass.
  - **Electromagnetism**: Particles with charge repel (like charges) or attract (opposites).
- **Barnes-Hut Optimization**: Uses a QuadTree to approximate long-range forces ($O(N \log N)$), enabling high particle counts.
- **Interactive Modes**:
  - **Place**: Click to spawn particles at rest.
  - **Shoot**: Drag and release to launch particles.
  - **Orbit**: Automatically calculates velocity for a stable circular orbit around the nearest massive body.
- **Presets**: Instantly load scenarios like Solar System, Binary Star, Galaxy, and Collision.
- **Visuals**:
  - **Trails**: Visualize particle paths.
  - **Dynamic Color**: Particle color shifts based on charge.
  - **Glow Effects**: Stylized rendering.

## Controls

### Simulation
- **Play/Pause**: Toggle simulation.
- **Step**: Advance simulation by one frame (when paused).
- **Speed**: Adjust time step ($dt$).
- **Clear**: Remove all particles.

### Physics Parameters
- **Mass/Charge/Spin**: Properties for the next particle you spawn.
- **Interaction Mode**:
  - `Place`: Spawn at cursor.
  - `Shoot`: Drag to set velocity.
  - `Orbit`: Auto-orbit valid gravitating bodies.
- **Collision Mode**:
  - `Pass`: Particles pass through each other.
  - `Bounce`: Elastic collisions.
  - `Merge`: Particles combine mass/charge/momentum upon contact.
- **Boundary Mode**:
  - `Despawn`: Particles are removed when far from screen.
  - `Loop`: Toroidal topology (wrap around edges).
  - `Bounce`: Particles bounce off screen edges.

### Other
- **Trails**: Toggle particle trails.
- **Presets**: Load pre-defined scenarios.

## Technical Details

The simulation implements a **Relativistic Euler Integration** scheme:
1. Forces are calculated using **Barnes-Hut** (approximating distant clusters as single bodies).
2. Momentum is updated: $\vec{p}_{new} = \vec{p}_{old} + \vec{F} \cdot dt$
3. Velocity is derived from momentum: $\vec{v} = \frac{\vec{p}}{m \sqrt{1 + \frac{p^2}{m^2 c^2}}}$
4. Position is updated: $\vec{x}_{new} = \vec{x}_{old} + \vec{v} \cdot dt$

This ensures that no particle can ever exceed the speed limit $c$, providing inherent stability for high-energy interactions.

## Running the Simulation

Simply open `index.html` in a modern web browser. 

For the best experience (to avoid CORS issues with Modules if run locally without a server), it is recommended to serve the directory:

```bash
# Python 3
python -m http.server
# Then navigate to http://localhost:8000
```
