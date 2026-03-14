// Apply uniform external background fields to all particles.
// Gravity: F = m*g (uniform gravitational field)
// Electric: F = q*E (uniform electric field)
// Magnetic: adds extBz to accumulated Bz for Boris rotation
// Direction vectors precomputed in uniforms (extGx, extGy, extEx, extEy).
// Uses packed ParticleState + AllForces structs.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read_write> allForces: array<AllForces>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }
    if ((particles[idx].flags & FLAG_ALIVE) == 0u) { return; }

    let g = uniforms.extGravity;
    let E = uniforms.extElectric;
    let Bext = uniforms.extBz;

    if (g == 0.0 && E == 0.0 && Bext == 0.0) { return; }

    var extFx: f32 = 0.0;
    var extFy: f32 = 0.0;

    if (g != 0.0) {
        let m = particles[idx].mass;
        extFx += m * uniforms.extGx;
        extFy += m * uniforms.extGy;
    }

    if (E != 0.0) {
        let q = particles[idx].charge;
        extFx += q * uniforms.extEx;
        extFy += q * uniforms.extEy;
    }

    // Write external force to allForces.f4.xy and update totalForce
    var af = allForces[idx];
    af.f4.x += extFx;
    af.f4.y += extFy;
    af.totalForce.x += extFx;
    af.totalForce.y += extFy;

    // Add external Bz
    if (Bext != 0.0) {
        af.bFields.z = Bext;  // extBz slot
        af.bFields.x += Bext;  // add to total Bz for Boris rotation
    }
    allForces[idx] = af;
}
