// Boris rotation: rotate w in the combined Bz + Bgz + extBz magnetic plane.
// Preserves |w| exactly (symplectic rotation).
// Reads bFields from packed AllForces struct.
// Reads/writes proper velocity from packed ParticleState struct.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read_write> allForces: array<AllForces>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }
    if ((particles[idx].flags & FLAG_ALIVE) == 0u) { return; }

    let hasMag = hasToggle0(MAGNETIC_BIT) || uniforms.extBz != 0.0;
    let hasGM = hasToggle0(GRAVITOMAG_BIT);
    if (!hasMag && !hasGM) { return; }

    let m = particles[idx].mass;
    if (m < EPSILON) { return; }
    let invM = 1.0 / m;
    let q = particles[idx].charge;
    let bf = allForces[idx].bFields;
    let totalBz = bf.x;    // accumulated Bz (includes extBz)
    let totalBgz = bf.y;   // accumulated Bgz

    // Compute Boris parameter t
    var t: f32 = 0.0;
    if (hasMag) {
        t += q * 0.5 * invM * totalBz;
    }
    if (hasGM) {
        t += 2.0 * totalBgz;
    }

    if (t == 0.0) { return; }

    let wx = particles[idx].velWX;
    let wy = particles[idx].velWY;
    let relOn = hasToggle0(RELATIVITY_BIT);
    let gamma = select(1.0, sqrt(1.0 + wx * wx + wy * wy), relOn);

    t *= uniforms.dt / gamma;
    let s = 2.0 * t / (1.0 + t * t);

    // Boris rotation: w- -> w+ via two cross products
    let wpx = wx + wy * t;
    let wpy = wy - wx * t;
    var newWx = wx + wpy * s;
    var newWy = wy - wpx * s;

    // NaN guard (Boris rotation is exact in exact arithmetic, but FP drift possible)
    if (newWx != newWx || newWy != newWy) { newWx = 0.0; newWy = 0.0; }

    particles[idx].velWX = newWx;
    particles[idx].velWY = newWy;
}
