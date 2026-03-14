// update-colors.wgsl — Recompute per-particle packed RGBA from charge/mass/antimatter.
//
// Matches CPU Particle.updateColor() logic:
//   neutral → slate (#8A7E72)
//   positive charge → lerp toward red (#C05048), intensity = |q|/5
//   negative charge → lerp toward blue (#5C92A8), intensity = |q|/5
//   BH mode → override to pure white
//   antimatter → invert RGB channels

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

struct ColorUniforms {
    blackHoleEnabled: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: ColorUniforms;
@group(0) @binding(1) var<storage, read> particles: array<ParticleState>;
@group(0) @binding(2) var<storage, read_write> color: array<u32>;

// Constants (FLAG_ALIVE, FLAG_ANTIMATTER, COLOR_SLATE, COLOR_RED, COLOR_BLUE)
// provided by generated wgslConstants block.

fn packRGBA(r: f32, g: f32, b: f32, a: f32) -> u32 {
    let ri = u32(clamp(r * 255.0, 0.0, 255.0));
    let gi = u32(clamp(g * 255.0, 0.0, 255.0));
    let bi = u32(clamp(b * 255.0, 0.0, 255.0));
    let ai = u32(clamp(a * 255.0, 0.0, 255.0));
    return ri | (gi << 8u) | (bi << 16u) | (ai << 24u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= arrayLength(&particles)) { return; }
    let p = particles[idx];
    if ((p.flags & FLAG_ALIVE) == 0u) { return; }

    let q = p.charge;
    let absQ = abs(q);
    let intensity = clamp(absQ / 5.0, 0.0, 1.0);

    var r: f32;
    var g: f32;
    var b: f32;

    if (absQ < 0.001) {
        // Neutral
        r = COLOR_SLATE.r; g = COLOR_SLATE.g; b = COLOR_SLATE.b;
    } else if (q > 0.0) {
        // Positive: lerp slate → red
        r = mix(COLOR_SLATE.r, COLOR_RED.r, intensity);
        g = mix(COLOR_SLATE.g, COLOR_RED.g, intensity);
        b = mix(COLOR_SLATE.b, COLOR_RED.b, intensity);
    } else {
        // Negative: lerp slate → blue
        r = mix(COLOR_SLATE.r, COLOR_BLUE.r, intensity);
        g = mix(COLOR_SLATE.g, COLOR_BLUE.g, intensity);
        b = mix(COLOR_SLATE.b, COLOR_BLUE.b, intensity);
    }

    // Antimatter: no color inversion — charge is already negated at spawn time,
    // which naturally produces the opposite color. CPU uses same getColor() for both.
    // The visual distinction is the dashed ring overlay (rendered separately).

    color[idx] = packRGBA(r, g, b, 1.0);
}
