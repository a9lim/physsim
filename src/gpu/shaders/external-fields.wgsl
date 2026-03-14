// Apply uniform external background fields to all particles.
// Gravity: F = m*g (uniform gravitational field)
// Electric: F = q*E (uniform electric field)
// Magnetic: adds extBz to accumulated Bz for Boris rotation
// Direction vectors precomputed in uniforms (extGx, extGy, extEx, extEy).

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read> mass: array<f32>;
@group(0) @binding(2) var<storage, read> charge: array<f32>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> forces4: array<vec4<f32>>;   // external.xy, higgs.xy
@group(0) @binding(5) var<storage, read_write> totalForce: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> bFields: array<vec4<f32>>;   // Bz, Bgz, extBz, pad

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }
    if ((flags[idx] & FLAG_ALIVE) == 0u) { return; }

    let g = uniforms.extGravity;
    let E = uniforms.extElectric;
    let Bext = uniforms.extBz;

    if (g == 0.0 && E == 0.0 && Bext == 0.0) { return; }

    var extFx: f32 = 0.0;
    var extFy: f32 = 0.0;

    if (g != 0.0) {
        let m = mass[idx];
        extFx += m * uniforms.extGx;
        extFy += m * uniforms.extGy;
    }

    if (E != 0.0) {
        let q = charge[idx];
        extFx += q * uniforms.extEx;
        extFy += q * uniforms.extEy;
    }

    // Write external force (first two components of forces4)
    var f4 = forces4[idx];
    f4.x += extFx;
    f4.y += extFy;
    forces4[idx] = f4;

    // Add to total force
    var tf = totalForce[idx];
    tf.x += extFx;
    tf.y += extFy;
    totalForce[idx] = tf;

    // Add external Bz
    if (Bext != 0.0) {
        var bf = bFields[idx];
        bf.z = Bext;  // extBz slot
        bf.x += Bext;  // add to total Bz for Boris rotation
        bFields[idx] = bf;
    }
}
