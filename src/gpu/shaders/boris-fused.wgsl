// Fused Boris integrator: halfKick → rotation → halfKick in a single pass.
// Eliminates 2 extra compute passes + 4 redundant global memory loads/stores.
// Each thread processes its own particle with no cross-thread dependency.
//
// Replaces the separate: borisHalfKick → borisRotate → borisHalfKick dispatch sequence.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read_write> allForces: array<AllForces>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }
    if ((particles[idx].flags & FLAG_ALIVE) == 0u) { return; }

    let m = particles[idx].mass;
    if (m < EPSILON) { return; }
    let invM = 1.0 / m;
    let dt = uniforms.dt;
    let halfDtOverM = dt * 0.5 * invM;

    // Load velocity once from global memory (saved: 2 extra struct loads)
    var wx = particles[idx].velWX;
    var wy = particles[idx].velWY;

    // Load forces once (saved: 2 extra struct loads)
    let af = allForces[idx];
    let fx = af.totalForce.x;
    let fy = af.totalForce.y;

    // ── Step 1: Half-kick ──
    wx += fx * halfDtOverM;
    wy += fy * halfDtOverM;

    // ── Step 2: Boris rotation ──
    let hasMag = hasToggle0(MAGNETIC_BIT) || uniforms.extBz != 0.0;
    let hasGM = hasToggle0(GRAVITOMAG_BIT);

    if (hasMag || hasGM) {
        let q = particles[idx].charge;
        let totalBz = af.bFields.x;
        let totalBgz = af.bFields.y;

        var t: f32 = 0.0;
        if (hasMag) { t += q * 0.5 * invM * totalBz; }
        if (hasGM) { t += 2.0 * totalBgz; }

        if (t != 0.0) {
            let relOn = hasToggle0(RELATIVITY_BIT);
            let gamma = select(1.0, sqrt(1.0 + wx * wx + wy * wy), relOn);
            t *= dt / gamma;
            let s = 2.0 * t / (1.0 + t * t);

            let wpx = wx + wy * t;
            let wpy = wy - wx * t;
            wx = wx + wpy * s;
            wy = wy - wpx * s;
        }
    }

    // ── Step 3: Second half-kick ──
    wx += fx * halfDtOverM;
    wy += fy * halfDtOverM;

    // NaN guard — single check covers all three stages
    if (wx != wx || wy != wy) { wx = 0.0; wy = 0.0; }

    // Write velocity back once (saved: 2 extra struct stores)
    particles[idx].velWX = wx;
    particles[idx].velWY = wy;
}
