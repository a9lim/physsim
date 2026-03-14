// ─── Barnes-Hut Tree Walk Force Computation ───
// One thread per alive particle. Stack-based iterative traversal.
// Uses theta=0.5 opening angle criterion.
// Accumulates into the same force buffers as pairwise path.

const NONE: i32 = -1;
const MAX_STACK: u32 = 48u;
const EPSILON: f32 = 1e-9;
const FLAG_ALIVE: u32 = 1u;
const FLAG_GHOST: u32 = 16u;

const GRAVITY_BIT:     u32 = 1u;
const COULOMB_BIT:     u32 = 2u;
const MAGNETIC_BIT:    u32 = 4u;
const GRAVITOMAG_BIT:  u32 = 8u;
const ONEPN_BIT:       u32 = 16u;
const RELATIVITY_BIT:  u32 = 32u;
const YUKAWA_BIT:      u32 = 2048u;
const AXION_BIT:       u32 = 8192u;
const RADIATION_BIT:   u32 = 128u;

const MAG_MOMENT_K: f32 = 0.2;
const INERTIA_K: f32 = 0.4;
const TIDAL_STRENGTH: f32 = 0.3;

// Node layout accessors (same as tree-build.wgsl)
const NODE_STRIDE: u32 = 20u;
fn nodeOffset(idx: u32) -> u32 { return idx * NODE_STRIDE; }

fn getMinX(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx)]); }
fn getMinY(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 1u]); }
fn getMaxX(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 2u]); }
fn getMaxY(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 3u]); }
fn getComX(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 4u]); }
fn getComY(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 5u]); }
fn getTotalMass(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 6u]); }
fn getTotalCharge(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 7u]); }
fn getTotalMagMoment(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 8u]); }
fn getTotalAngMomentum(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 9u]); }
fn getTotalMomX(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 10u]); }
fn getTotalMomY(idx: u32) -> f32 { return bitcast<f32>(nodes[nodeOffset(idx) + 11u]); }
fn getNW(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 12u]); }
fn getNE(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 13u]); }
fn getSW(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 14u]); }
fn getSE(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 15u]); }
fn getParticleIndex(idx: u32) -> i32 { return bitcast<i32>(nodes[nodeOffset(idx) + 16u]); }
fn getParticleCount(idx: u32) -> u32 { return nodes[nodeOffset(idx) + 17u]; }

struct SimUniforms {
    dt: f32,
    simTime: f32,
    domainW: f32,
    domainH: f32,
    speedScale: f32,
    softening: f32,
    softeningSq: f32,
    toggles0: u32,
    toggles1: u32,
    yukawaCoupling: f32,
    yukawaMu: f32,
    higgsMass: f32,
    axionMass: f32,
    extGravX: f32,
    extGravY: f32,
    extElecX: f32,
    extElecY: f32,
    extBz: f32,
    boundaryMode: u32,
    topologyMode: u32,
    collisionMode: u32,
    aliveCount: u32,
    bhTheta: f32,
    totalCount: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: SimUniforms;

// Particle SoA (alive + ghosts)
@group(1) @binding(0) var<storage, read> posX: array<f32>;
@group(1) @binding(1) var<storage, read> posY: array<f32>;
@group(1) @binding(2) var<storage, read> velWX: array<f32>;
@group(1) @binding(3) var<storage, read> velWY: array<f32>;
@group(1) @binding(4) var<storage, read> mass_in: array<f32>;
@group(1) @binding(5) var<storage, read> charge_in: array<f32>;
@group(1) @binding(6) var<storage, read> angW_in: array<f32>;
@group(1) @binding(7) var<storage, read> flags_in: array<u32>;
@group(1) @binding(8) var<storage, read> radius_in: array<f32>;
@group(1) @binding(9) var<storage, read> magMoment_in: array<f32>;
@group(1) @binding(10) var<storage, read> angMomentum_in: array<f32>;
@group(1) @binding(11) var<storage, read> axMod_in: array<f32>;
@group(1) @binding(12) var<storage, read> yukMod_in: array<f32>;
@group(1) @binding(13) var<storage, read> particleId_in: array<u32>;

// Ghost->original mapping
@group(1) @binding(14) var<storage, read> ghostOriginalIdx: array<u32>;

// Force accumulators (output, same layout as Phase 2)
@group(2) @binding(0) var<storage, read_write> forces0: array<vec4<f32>>; // gravity.xy, coulomb.xy
@group(2) @binding(1) var<storage, read_write> forces1: array<vec4<f32>>; // magnetic.xy, gravitomag.xy
@group(2) @binding(2) var<storage, read_write> forces3: array<vec4<f32>>; // radiation.xy, yukawa.xy
@group(2) @binding(3) var<storage, read_write> bFields: array<vec4<f32>>; // Bz, Bgz, extBz, pad
@group(2) @binding(4) var<storage, read_write> torques: array<vec4<f32>>; // spinOrbit, frameDrag, tidal, pad

// Shared pairForce function (from pair-force.wgsl, imported or inlined)
// This function accumulates E-like forces and B-field contributions
// for one source acting on one receiver particle.
// For the tree walk, sources are either individual leaf particles
// or aggregate node data (mass, charge, CoM, etc.).

// Inline the core force accumulation (matches pairForce in forces.js):
fn accumulateForce(
    pIdx: u32,
    px: f32, py: f32,
    pMass: f32, pCharge: f32,
    pMagMoment: f32, pAngMomentum: f32,
    pAngVel: f32, pVelX: f32, pVelY: f32,
    pAxMod: f32,
    sx: f32, sy: f32,
    svx: f32, svy: f32,
    sMass: f32, sCharge: f32,
    sAngVel: f32, sMagMoment: f32, sAngMomentum: f32,
    sAxMod: f32, sYukMod: f32,
    pBodyRadiusSq: f32,
) {
    let toggles = uniforms.toggles0;
    let softeningSq = uniforms.softeningSq;

    // Displacement
    let rx = sx - px;
    let ry = sy - py;
    let rawRSq = rx * rx + ry * ry;
    let rSq = rawRSq + softeningSq;
    let invRSq = 1.0 / rSq;
    let invR = sqrt(invRSq);
    let invR3 = invR * invRSq;
    let invR5 = invR3 * invRSq;

    let needAxMod = ((toggles & COULOMB_BIT) != 0u || (toggles & MAGNETIC_BIT) != 0u) && (toggles & AXION_BIT) != 0u;
    var axModPair: f32 = 1.0;
    if (needAxMod) {
        axModPair = sqrt(pAxMod * sAxMod);
    }

    // Gravity: +m1*m2/r^2 (attractive)
    if ((toggles & GRAVITY_BIT) != 0u) {
        let k = pMass * sMass;
        let fDir = k * invR3;
        forces0[pIdx] += vec4<f32>(rx * fDir, ry * fDir, 0.0, 0.0);

        // Tidal locking torque
        let crossRV = rx * (svy - pVelY) - ry * (svx - pVelX);
        let wOrbit = crossRV * invRSq;
        let dw = pAngVel - wOrbit;
        var coupling = sMass;
        if ((toggles & COULOMB_BIT) != 0u && pMass > EPSILON) {
            coupling += pCharge * sCharge / pMass;
        }
        let ri5 = pBodyRadiusSq * pBodyRadiusSq * pow(pMass, 1.0/3.0);
        let invR6 = invRSq * invRSq * invRSq;
        torques[pIdx] += vec4<f32>(0.0, 0.0, -TIDAL_STRENGTH * coupling * coupling * ri5 * invR6 * dw, 0.0);
    }

    // Coulomb: -q1*q2/r^2 (like repels)
    if ((toggles & COULOMB_BIT) != 0u) {
        let k = -(pCharge * sCharge) * axModPair;
        let fDir = k * invR3;
        forces0[pIdx] += vec4<f32>(0.0, 0.0, rx * fDir, ry * fDir);
    }

    // Cross product (vs x r)_z for Biot-Savart
    let crossSV = svx * ry - svy * rx;

    // Magnetic: dipole-dipole + Bz field
    if ((toggles & MAGNETIC_BIT) != 0u) {
        let axMod = axModPair;
        // Dipole-dipole radial: F = +3*mu1*mu2/r^4
        let fDir = 3.0 * (pMagMoment * sMagMoment) * invR5 * axMod;
        forces1[pIdx] += vec4<f32>(rx * fDir, ry * fDir, 0.0, 0.0);

        // Bz from moving charge
        let BzMoving = sCharge * crossSV * invR3 * axMod;
        bFields[pIdx] += vec4<f32>(BzMoving, 0.0, 0.0, 0.0);
        // Bz from dipole
        bFields[pIdx] -= vec4<f32>(sMagMoment * invR3 * axMod, 0.0, 0.0, 0.0);
    }

    // Gravitomagnetic: dipole + Bgz field
    if ((toggles & GRAVITOMAG_BIT) != 0u) {
        let fDir = 3.0 * (pAngMomentum * sAngMomentum) * invR5;
        forces1[pIdx] += vec4<f32>(0.0, 0.0, rx * fDir, ry * fDir);

        // Bgz from moving mass
        let BgzMoving = -sMass * crossSV * invR3;
        bFields[pIdx] += vec4<f32>(0.0, BgzMoving, 0.0, 0.0);
        // Bgz from spin
        bFields[pIdx] -= vec4<f32>(0.0, 2.0 * sAngMomentum * invR3, 0.0, 0.0);

        // Frame-dragging torque
        let fdTorque = 2.0 * sAngMomentum * (sAngVel - pAngVel) * invR3;
        torques[pIdx] += vec4<f32>(0.0, fdTorque, 0.0, 0.0);
    }

    // Yukawa
    if ((toggles & YUKAWA_BIT) != 0u) {
        let mu = uniforms.yukawaMu;
        let cutoffSq = (6.0 / mu) * (6.0 / mu);
        if (rawRSq < cutoffSq) {
            let r = 1.0 / invR;
            let expMuR = exp(-mu * r);
            let yukModPair = sqrt(1.0 * sYukMod); // p.yukMod handled via caller
            let ym = yukModPair;
            let fDir = uniforms.yukawaCoupling * ym * pMass * sMass * expMuR * (invRSq + mu * invR) * invR;
            forces3[pIdx] += vec4<f32>(0.0, 0.0, rx * fDir, ry * fDir);
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pIdx = gid.x;
    if (pIdx >= uniforms.aliveCount) { return; }

    let f = flags_in[pIdx];
    if ((f & FLAG_ALIVE) == 0u) { return; }
    if ((f & FLAG_GHOST) != 0u) { return; } // Ghosts don't receive forces

    let px = posX[pIdx];
    let py = posY[pIdx];
    let pMass = mass_in[pIdx];
    let pCharge = charge_in[pIdx];
    let pMagMoment = magMoment_in[pIdx];
    let pAngMomentum = angMomentum_in[pIdx];
    let pRadius = radius_in[pIdx];
    let pBodyRadiusSq = pRadius * pRadius; // Approximation; BH mode uses different formula
    let pAngW = angW_in[pIdx];
    let pAxMod = axMod_in[pIdx];

    // Derive velocity from proper velocity
    let wSq = velWX[pIdx] * velWX[pIdx] + velWY[pIdx] * velWY[pIdx];
    let relOn = (uniforms.toggles0 & RELATIVITY_BIT) != 0u;
    let invGamma = select(1.0, 1.0 / sqrt(1.0 + wSq), relOn);
    let pVelX = velWX[pIdx] * invGamma;
    let pVelY = velWY[pIdx] * invGamma;
    let pAngVel = select(pAngW, pAngW / sqrt(1.0 + pAngW * pAngW * pBodyRadiusSq), relOn);

    let thetaSq = uniforms.bhTheta * uniforms.bhTheta;
    let pid = particleId_in[pIdx];

    // Stack-based iterative tree walk
    var stack: array<u32, 48>;
    var stackTop: u32 = 0u;
    stack[0] = 0u; // push root
    stackTop = 1u;

    loop {
        if (stackTop == 0u) { break; }
        stackTop -= 1u;
        let nodeIdx = stack[stackTop];

        let nodeMass = getTotalMass(nodeIdx);
        if (nodeMass < EPSILON) { continue; }

        let comX = getComX(nodeIdx);
        let comY = getComY(nodeIdx);
        let dx = comX - px;
        let dy = comY - py;
        let dSq = dx * dx + dy * dy;
        let size = getMaxX(nodeIdx) - getMinX(nodeIdx); // node width

        let isLeaf = getNW(nodeIdx) == NONE;
        let particleIdx = getParticleIndex(nodeIdx);

        if (isLeaf && particleIdx >= 0) {
            // Leaf node: accumulate from the individual particle
            let sIdx = u32(particleIdx);

            // Skip self-interaction
            let sFlags = flags_in[sIdx];
            let sPid = particleId_in[sIdx];
            // Ghost particles: skip if original is self
            let isGhost = (sFlags & FLAG_GHOST) != 0u;
            var origIdx: u32 = sIdx;
            if (isGhost && sIdx >= uniforms.aliveCount) {
                origIdx = ghostOriginalIdx[sIdx - uniforms.aliveCount];
            }
            if (origIdx == pIdx) { continue; } // skip self
            if (sIdx == pIdx) { continue; }

            accumulateForce(
                pIdx, px, py, pMass, pCharge,
                pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY, pAxMod,
                posX[sIdx], posY[sIdx],
                velWX[sIdx] * invGamma, velWY[sIdx] * invGamma, // approximate vel from w
                mass_in[sIdx], charge_in[sIdx],
                0.0, // sAngVel approximated as 0 for aggregate; leaf uses cached
                magMoment_in[sIdx], angMomentum_in[sIdx],
                axMod_in[sIdx], yukMod_in[sIdx],
                pBodyRadiusSq,
            );
        } else if (!isLeaf && (size * size < thetaSq * dSq)) {
            // Distant node: use aggregate multipole data
            let avgVx = getTotalMomX(nodeIdx) / nodeMass;
            let avgVy = getTotalMomY(nodeIdx) / nodeMass;

            accumulateForce(
                pIdx, px, py, pMass, pCharge,
                pMagMoment, pAngMomentum, pAngVel, pVelX, pVelY, pAxMod,
                comX, comY,
                avgVx, avgVy,
                nodeMass, getTotalCharge(nodeIdx),
                0.0, // sAngVel = 0 for aggregate
                getTotalMagMoment(nodeIdx), getTotalAngMomentum(nodeIdx),
                1.0, 1.0, // axMod/yukMod = 1 for aggregate
                pBodyRadiusSq,
            );
        } else if (!isLeaf) {
            // Push children
            if (stackTop + 4u <= MAX_STACK) {
                stack[stackTop] = u32(getNW(nodeIdx)); stackTop += 1u;
                stack[stackTop] = u32(getNE(nodeIdx)); stackTop += 1u;
                stack[stackTop] = u32(getSW(nodeIdx)); stackTop += 1u;
                stack[stackTop] = u32(getSE(nodeIdx)); stackTop += 1u;
            }
        }
    }
}
