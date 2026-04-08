# Kugelblitz Collapse

When boson interaction and gravity are both enabled, photon/pion/lepton energy concentrations that exceed the gravitational collapse threshold condense into a new massive particle.

## Physics

**Hoop conjecture** (natural units, G=c=1): a region of size r containing total energy E collapses when E > r/2.

The boson tree aggregates `_srcMass` (= `energy` for photons, `gamma*m` for massive bosons — no receiver GR factor). So `totalMass` in the tree equals total energy directly.

The resulting particle is a normal massive particle. It becomes a BH only if the BH toggle is already on.

## Detection

Walk the existing boson BH tree (built during boson interaction phase). For each internal node:

```
nodeEnergy = totalMass   (_srcMass = energy for photons, gamma*m for pions/leptons)
nodeSize = max(bw, bh)   (bounding box dimensions from quadtree)
collapse if nodeEnergy > nodeSize / 2
```

Find the **smallest** collapsing node (deepest in tree) to get the most compact cluster. This avoids collapsing the entire boson population when a dense core exists inside a larger node.

**Boson count**: `pointCount` only tracks leaf-level direct points (max 4). For the minimum-count guard, add a `totalCount` array to `QuadTreePool`, computed alongside `totalMass` in `calculateBosonDistribution()`. Leaf: `totalCount = pointCount`. Internal: `totalCount = sum of children's totalCount`.

**Guards**:
- Minimum `MIN_KUGELBLITZ_COUNT` (4) bosons in the collapsing node (prevents trivial 2-photon collapses from softening proximity)
- Minimum total energy floor: `MIN_KUGELBLITZ_ENERGY` (tentatively `4 * MIN_MASS = 0.2`) so dust-energy photons don't form microscopic particles
- Maximum 1 collapse event per substep (prevents cascade)

## Collapse Resolution

1. Collect all alive photons, pions, and leptons whose positions fall within the collapsing node's bounding box
2. Compute:
   - **COM position**: energy-weighted centroid
   - **Total energy**: sum of photon energies + relativistic energies of massive bosons (`gamma * mass`)
   - **Total momentum**: vector sum (photon: `energy * vel`; massive: `mass * w`)
   - **Angular momentum**: sum of `r_i x p_i` about COM for all consumed bosons
   - **Total charge**: sum of pion/lepton charges (photons contribute 0)
3. Spawn new particle via `sim.addParticle()`:
   - `mass = totalEnergy`
   - `baseMass = totalEnergy`
   - Position at COM
   - Velocity from `totalMomentum / totalEnergy` (capped at MAX_SPEED_RATIO)
   - `angw` from `L_total / I` where `I = (2/5) * mass * radius^2`, `radius = cbrt(mass)`
   - `charge` = totalCharge (already quantized since pion/lepton charges are multiples of BOSON_CHARGE)
   - `antimatter = false`
4. Kill all consumed bosons (set `alive = false`)
5. Track energy accounting: subtract consumed boson energy from `sim.totalRadiated` (radiation returning to mass)

## Integration Point

CPU (`integrator.js`): immediately after the boson interaction block (~line 1393), inside the `if (this.bosonInterEnabled && this.sim)` guard. The boson tree (`bRoot`, `this._bosonPool`) is already available.

New method: `Integrator.prototype._checkKugelblitz(bRoot)`.

## GPU

Same 1-frame readback pattern as merge/disintegration events:

1. Compute shader walks boson tree nodes, checks collapse condition
2. Writes a `KugelblitzEvent` struct (COM, momentum, energy, charge, angL) via `atomicAdd` on event counter
3. CPU reads back on next frame, spawns particle via `addParticle()`
4. Consumed bosons flagged dead in the same dispatch

`KugelblitzEvent` struct (~40 bytes): `x, y, px, py, energy, charge, angL, count` (8 f32 fields). Max 1 event per substep, buffer sized for 1.

Requires a new compute pipeline in `gpu-pipelines.js` and a new shader file `kugelblitz.wgsl`. Bump `SHADER_VERSION`.

## What This Does NOT Include

- No UI toggle (consequence of gravity + boson interaction, not independent)
- No geon (metastable bound state) mechanics — those emerge naturally from boson-boson gravity if photon lifetimes are sufficient
- No preset — can be added later once the mechanic is tuned
- No visual collapse effect (flash, ring, etc.) — the particle just appears. Can be added later.
- No changes to photon lifetime

## Files Modified

| File | Change |
|------|--------|
| `src/quadtree.js` | Add `totalCount` array, compute in `calculateBosonDistribution()` |
| `src/integrator.js` | `_checkKugelblitz()` method, call site after boson interaction |
| `src/config.js` | `MIN_KUGELBLITZ_ENERGY`, `MIN_KUGELBLITZ_COUNT` constants |
| `src/gpu/gpu-physics.js` | Kugelblitz event readback, buffer creation |
| `src/gpu/gpu-pipelines.js` | New pipeline, shader fetch, SHADER_VERSION bump |
| `src/gpu/shaders/kugelblitz.wgsl` | Collapse detection + boson kill dispatch |
| `src/reference.js` | Documentation entry for kugelblitz |
| `CLAUDE.md` | Document the new mechanic |
