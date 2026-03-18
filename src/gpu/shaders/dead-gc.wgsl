// ─── Dead Particle Garbage Collection ───
// Transitions RETIRED particles to FREE when their signal delay history expires.
// Runs once per frame (not per substep).

// Constants provided by generated wgslConstants block.

// Struct definitions (ParticleState, ParticleAux, SimUniforms) provided by shared-structs.wgsl.

@group(0) @binding(0) var<storage, read_write> particleState: array<ParticleState>;
@group(0) @binding(1) var<storage, read_write> particleAux: array<ParticleAux>;
@group(0) @binding(2) var<uniform> uniforms: SimUniforms;
@group(0) @binding(3) var<storage, read_write> freeStack: array<u32>;
@group(0) @binding(4) var<storage, read_write> freeTop: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    // Scan all particle slots (not just alive count — retired particles may be anywhere)
    let maxSlots = uniforms.aliveCount; // High-water mark: aliveCount is never decremented, so all ever-allocated slots (alive, retired, or freed) live in [0, aliveCount)
    if (idx >= maxSlots) { return; }

    var ps = particleState[idx];

    // Skip alive particles
    if ((ps.flags & FLAG_ALIVE) != 0u) { return; }
    // Skip already-freed particles (flags == 0)
    if (ps.flags == 0u) { return; }

    let isRetired = (ps.flags & FLAG_RETIRED) != 0u;

    if (isRetired) {
        let deathT = particleAux[idx].deathTime;
        // All retired particles (collision, boundary) have finite deathTime set at retirement.
        // Wait for signal-delay expiry (2 × domain diagonal) before freeing the slot.
        if (deathT < 1e30) {
            let domainDiag = sqrt(uniforms.domainW * uniforms.domainW + uniforms.domainH * uniforms.domainH);
            let expiry = 2.0 * domainDiag;
            if (uniforms.simTime - deathT <= expiry) { return; }
        }
    }

    // Transition to FREE: clear all flags
    ps.flags = 0u;
    particleState[idx] = ps;

    // Push slot to free stack
    let slot = atomicAdd(&freeTop, 1u);
    freeStack[slot] = idx;
}
