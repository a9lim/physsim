// ─── Signal Delay History ───
// Ring buffer recording for interleaved history format.
// f32 precision with relative-time encoding for GPU.
//
// Standalone shader — defines own structs (NOT prepended with common.wgsl).
// getDelayedStateGPU moved to signal-delay-common.wgsl.

// Constants provided by generated wgslConstants block.

// ── Packed struct definitions ──

struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

// Must match SimUniforms byte layout in common.wgsl / writeUniforms() exactly.
// Only a few fields used; preceding fields kept as padding for alignment.
struct SimUniforms {
    _dt: f32,               // [0] dt
    simTime: f32,           // [1] simTime
    domainW: f32,           // [2] domainW
    domainH: f32,           // [3] domainH
    _pad0: f32,             // [4] speedScale
    _pad1: f32,             // [5] softening
    _pad2: f32,             // [6] softeningSq
    _pad3: u32,             // [7] toggles0
    _pad4: u32,             // [8] toggles1
    _pad5: f32,             // [9] yukawaCoupling
    _pad6: f32,             // [10] yukawaMu
    _pad7: f32,             // [11] higgsMass
    _pad8: f32,             // [12] axionMass
    boundaryMode: u32,      // [13] boundaryMode
    topologyMode: u32,      // [14] topologyMode
};

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState>;
// Note: particles bound as read_write so we can clear FLAG_REBORN

// History buffers (interleaved format)
@group(1) @binding(0) var<storage, read_write> histData: array<f32>;
@group(1) @binding(1) var<storage, read_write> histMeta: array<u32>;

@compute @workgroup_size(64)
fn recordHistory(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= arrayLength(&particles)) { return; }

    let flags = particles[i].flags;

    // Newly-retired particles: record one final history snapshot at death time,
    // then set FLAG_DEATH_HIST to prevent re-recording on subsequent frames
    if ((flags & FLAG_RETIRED) != 0u && (flags & FLAG_ALIVE) == 0u) {
        if ((flags & FLAG_DEATH_HIST) != 0u) { return; }
        particles[i].flags = flags | FLAG_DEATH_HIST;

        let metaBase = i * HIST_META_STRIDE;
        var writeIdx = histMeta[metaBase];
        var count = histMeta[metaBase + 1u];

        let sampleBase = i * HISTORY_LEN * HIST_STRIDE
                       + (writeIdx & HISTORY_MASK) * HIST_STRIDE;

        histData[sampleBase + 0u] = particles[i].posX;
        histData[sampleBase + 1u] = particles[i].posY;
        let wx = particles[i].velWX;
        let wy = particles[i].velWY;
        let gamma = sqrt(1.0 + wx * wx + wy * wy);
        let invG = 1.0 / gamma;
        histData[sampleBase + 2u] = wx * invG;
        histData[sampleBase + 3u] = wy * invG;
        histData[sampleBase + 4u] = particles[i].angW;
        histData[sampleBase + 5u] = u.simTime;

        histMeta[metaBase] = (writeIdx + 1u) & HISTORY_MASK;
        histMeta[metaBase + 1u] = min(count + 1u, HISTORY_LEN);
        return;
    }

    if ((flags & FLAG_ALIVE) == 0u) { return; }

    let metaBase = i * HIST_META_STRIDE;

    // Reborn particles (new particle in recycled slot after merge):
    // clear stale history from the old particle, set creationTime, and reset the flag
    if ((flags & FLAG_REBORN) != 0u) {
        histMeta[metaBase] = 0u;              // writeIdx
        histMeta[metaBase + 1u] = 0u;         // count
        histMeta[metaBase + 2u] = bitcast<u32>(u.simTime); // creationTime
        histMeta[metaBase + 3u] = 0u;         // _pad
        particles[i].flags = flags & ~FLAG_REBORN;
    }

    var writeIdx = histMeta[metaBase];
    var count = histMeta[metaBase + 1u];

    let sampleBase = i * HISTORY_LEN * HIST_STRIDE
                   + (writeIdx & HISTORY_MASK) * HIST_STRIDE;

    histData[sampleBase + 0u] = particles[i].posX;
    histData[sampleBase + 1u] = particles[i].posY;

    // Store coordinate velocity (vel = w / sqrt(1 + w²))
    let wx = particles[i].velWX;
    let wy = particles[i].velWY;
    let gamma = sqrt(1.0 + wx * wx + wy * wy);
    let invG = 1.0 / gamma;
    histData[sampleBase + 2u] = wx * invG;
    histData[sampleBase + 3u] = wy * invG;

    histData[sampleBase + 4u] = particles[i].angW;
    histData[sampleBase + 5u] = u.simTime;

    writeIdx = (writeIdx + 1u) & HISTORY_MASK;
    count = min(count + 1u, HISTORY_LEN);
    histMeta[metaBase] = writeIdx;
    histMeta[metaBase + 1u] = count;
}
