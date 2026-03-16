// ─── Field Excitations ───
// Deposit Gaussian wave packets from collision events into fieldDot.
// Amplitude = MERGE_EXCITATION_SCALE * sqrt(energy), σ = FIELD_EXCITATION_SIGMA cells.

// Excitation events are written to a small append buffer by the collision resolve pass.
struct ExcitationEvent {
    x: f32,
    y: f32,
    energy: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read_write> fieldDot: array<f32>;
@group(0) @binding(1) var<storage, read> events: array<ExcitationEvent>;
@group(0) @binding(2) var<uniform> uniforms: FieldUniforms;
@group(0) @binding(3) var<storage, read> eventCount: array<u32>;  // [0] = count

// One thread per grid cell. Iterates over all excitation events.
@compute @workgroup_size(8, 8)
fn depositExcitations(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= GRID || iy >= GRID) { return; }

    let nEvents = eventCount[0];
    if (nEvents == 0u) { return; }

    let cellW = uniforms.domainW / f32(GRID);
    let cellH = uniforms.domainH / f32(GRID);
    if (cellW < EPSILON || cellH < EPSILON) { return; }

    let sigma = FIELD_EXCITATION_SIGMA;
    let sigmaSq = sigma * sigma;
    let cutoffSq = 9.0 * sigmaSq;  // 3σ cutoff
    let idx = iy * GRID + ix;
    var total: f32 = 0.0;

    for (var e = 0u; e < nEvents; e++) {
        let evt = events[e];
        if (evt.energy < EPSILON) { continue; }

        let gxEvt = evt.x / cellW;
        let gyEvt = evt.y / cellH;
        let amplitude = min(MERGE_EXCITATION_SCALE * sqrt(evt.energy), EXCITATION_MAX_AMPLITUDE);

        let dx = f32(ix) - gxEvt;
        let dy = f32(iy) - gyEvt;
        let rSq = dx * dx + dy * dy;

        if (rSq <= cutoffSq) {
            total += amplitude * exp(-rSq / (2.0 * sigmaSq));
        }
    }

    fieldDot[idx] += total;
}
