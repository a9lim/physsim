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

const ALIVE_BIT: u32 = 1u;
const ANTIMATTER_BIT: u32 = 4u;

// Slate (neutral): #8A7E72 = (138, 126, 114)
const SLATE_R: f32 = 138.0 / 255.0;
const SLATE_G: f32 = 126.0 / 255.0;
const SLATE_B: f32 = 114.0 / 255.0;

// Positive (red): #C05048 = (192, 80, 72)
const POS_R: f32 = 192.0 / 255.0;
const POS_G: f32 = 80.0 / 255.0;
const POS_B: f32 = 72.0 / 255.0;

// Negative (blue): #5C92A8 = (92, 146, 168)
const NEG_R: f32 = 92.0 / 255.0;
const NEG_G: f32 = 146.0 / 255.0;
const NEG_B: f32 = 168.0 / 255.0;

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
    if ((p.flags & ALIVE_BIT) == 0u) { return; }

    // BH mode: all particles are white
    if (params.blackHoleEnabled != 0u) {
        color[idx] = packRGBA(1.0, 1.0, 1.0, 1.0);
        return;
    }

    let q = p.charge;
    let absQ = abs(q);
    let intensity = clamp(absQ / 5.0, 0.0, 1.0);

    var r: f32;
    var g: f32;
    var b: f32;

    if (absQ < 0.001) {
        // Neutral
        r = SLATE_R; g = SLATE_G; b = SLATE_B;
    } else if (q > 0.0) {
        // Positive: lerp slate → red
        r = mix(SLATE_R, POS_R, intensity);
        g = mix(SLATE_G, POS_G, intensity);
        b = mix(SLATE_B, POS_B, intensity);
    } else {
        // Negative: lerp slate → blue
        r = mix(SLATE_R, NEG_R, intensity);
        g = mix(SLATE_G, NEG_G, intensity);
        b = mix(SLATE_B, NEG_B, intensity);
    }

    // Antimatter: invert
    if ((p.flags & ANTIMATTER_BIT) != 0u) {
        r = 1.0 - r;
        g = 1.0 - g;
        b = 1.0 - b;
    }

    color[idx] = packRGBA(r, g, b, 1.0);
}
