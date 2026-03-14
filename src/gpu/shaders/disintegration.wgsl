// ─── Tidal Disintegration & Roche Lobe Overflow ───
// One thread per particle. Checks tidal stress > self-gravity.
// Writes events to an append buffer for CPU-side fragment spawning.
// Roche: Eggleton (1983) L1 mass transfer.

const EPSILON: f32 = 1e-9;

// Packed particle state struct (matches common.wgsl ParticleState)
struct ParticleState_DI {
    posX: f32, posY: f32,
    velWX: f32, velWY: f32,
    mass: f32, charge: f32, angW: f32,
    baseMass: f32,
    flags: u32,
};

// Packed auxiliary struct (matches common.wgsl ParticleAux)
struct ParticleAux_DI {
    radius: f32,
    particleId: u32,
    deathTime: f32,
    deathMass: f32,
    deathAngVel: f32,
};

// Packed struct (mirrors common.wgsl ParticleDerived)
struct DisintDerived {
    magMoment: f32,
    angMomentum: f32,
    invMass: f32,
    radiusSq: f32,
    velX: f32,
    velY: f32,
    angVel: f32,
    _pad: f32,
};

struct DisintUniforms {
    softeningSq: f32,
    domainW: f32,
    domainH: f32,
    tidalStrength: f32,  // 0.3
    rocheThreshold: f32, // 0.9
    rocheTransferRate: f32, // 0.01
    minMass: f32,        // MIN_MASS
    spawnCount: u32,     // 4
    particleCount: u32,
    periodic: u32,
    topologyMode: u32,
    _pad: f32,
};

const DISINT_FRAGMENT: u32 = 0u;
const DISINT_TRANSFER: u32 = 1u;

struct DisintEvent {
    particleIdx: u32,
    eventType: u32,       // 0=fragment, 1=transfer
    targetIdx: u32,       // strongest other (for transfer)
    transferMass: f32,
    spawnX: f32,
    spawnY: f32,
    spawnVX: f32,
    spawnVY: f32,
};

// Group 0: particleState (ro) + particleAux (ro) + derived (ro)
@group(0) @binding(0) var<storage, read> particles: array<ParticleState_DI>;
@group(0) @binding(1) var<storage, read> particleAux: array<ParticleAux_DI>;
@group(0) @binding(2) var<storage, read> derived: array<DisintDerived>;

@group(1) @binding(0) var<storage, read_write> events: array<DisintEvent>;
@group(1) @binding(1) var<storage, read_write> eventCounter: atomic<u32>;
@group(1) @binding(2) var<uniform> du: DisintUniforms;

const MAX_DISINT_EVENTS: u32 = 64u;

@compute @workgroup_size(256)
fn checkDisintegration(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= du.particleCount) { return; }
    let p = particles[pid];
    let flag = p.flags;
    if ((flag & 1u) == 0u) { return; }

    let m = p.mass;
    if (m < du.minMass * f32(du.spawnCount)) { return; }

    let d = derived[pid];
    let rSq = d.radiusSq;
    let r = particleAux[pid].radius;
    let selfGravity = m / rSq;
    let w = d.angVel;
    let centrifugal = w * w * r;
    let q = p.charge;
    let coulombSelf = (q * q) / (4.0 * rSq);

    // Pure centrifugal/Coulomb breakup
    if (centrifugal + coulombSelf > selfGravity) {
        let slot = atomicAdd(&eventCounter, 1u);
        if (slot < MAX_DISINT_EVENTS) {
            var evt: DisintEvent;
            evt.particleIdx = pid;
            evt.eventType = DISINT_FRAGMENT;
            events[slot] = evt;
        }
        return;
    }

    // Find strongest tidal source (O(N) scan — acceptable for N < 4096)
    var maxTidal: f32 = 0.0;
    var strongestIdx: u32 = 0u;
    var strongestDx: f32 = 0.0;
    var strongestDy: f32 = 0.0;
    var strongestDist: f32 = 0.0;
    let px = p.posX;
    let py = p.posY;
    let halfDomW = du.domainW * 0.5;
    let halfDomH = du.domainH * 0.5;

    for (var oi = 0u; oi < du.particleCount; oi++) {
        if (oi == pid) { continue; }
        let op = particles[oi];
        let oflag = op.flags;
        if ((oflag & 1u) == 0u) { continue; }

        var dx = op.posX - px;
        var dy = op.posY - py;

        // Minimum image (torus only for simplicity)
        if (du.periodic != 0u) {
            if (dx > halfDomW) { dx -= du.domainW; }
            else if (dx < -halfDomW) { dx += du.domainW; }
            if (dy > halfDomH) { dy -= du.domainH; }
            else if (dy < -halfDomH) { dy += du.domainH; }
        }

        let distSq = dx * dx + dy * dy + du.softeningSq;
        let invDistSq = 1.0 / distSq;
        let tidalAccel = du.tidalStrength * op.mass * r * sqrt(invDistSq) * invDistSq;

        if (tidalAccel > maxTidal) {
            maxTidal = tidalAccel;
            strongestIdx = oi;
            strongestDx = dx;
            strongestDy = dy;
            strongestDist = sqrt(distSq - du.softeningSq);
        }
    }

    // Tidal breakup
    if (maxTidal + centrifugal + coulombSelf > selfGravity) {
        let slot = atomicAdd(&eventCounter, 1u);
        if (slot < MAX_DISINT_EVENTS) {
            var evt: DisintEvent;
            evt.particleIdx = pid;
            evt.eventType = DISINT_FRAGMENT;
            events[slot] = evt;
        }
        return;
    }

    // Roche lobe overflow (Eggleton 1983)
    if (strongestDist > EPSILON && m > du.minMass * 4.0) {
        let rocheDist = strongestDist;
        let oMass = particles[strongestIdx].mass;
        let qRatio = m / (m + oMass);
        let q13 = pow(qRatio, 1.0 / 3.0);
        let q23 = q13 * q13;
        let rRoche = rocheDist * 0.49 * q23 / (0.6 * q23 + log(1.0 + q13));

        if (r > rRoche * du.rocheThreshold) {
            let l1Mag = sqrt(strongestDx * strongestDx + strongestDy * strongestDy);
            if (l1Mag > EPSILON) {
                let l1x = strongestDx / l1Mag;
                let l1y = strongestDy / l1Mag;
                let overflow = r / rRoche - du.rocheThreshold;
                let dM = min(overflow * du.rocheTransferRate * m, m * 0.1);

                if (dM >= du.minMass) {
                    let slot = atomicAdd(&eventCounter, 1u);
                    if (slot < MAX_DISINT_EVENTS) {
                        var evt: DisintEvent;
                        evt.particleIdx = pid;
                        evt.eventType = DISINT_TRANSFER;
                        evt.targetIdx = strongestIdx;
                        evt.transferMass = dM;
                        evt.spawnX = px + l1x * r * 1.2;
                        evt.spawnY = py + l1y * r * 1.2;
                        let dvd = derived[pid];
                        let dv = vec2<f32>(dvd.velX, dvd.velY);
                        evt.spawnVX = dv.x + (-l1y) * sqrt(oMass / rocheDist) * 0.5;
                        evt.spawnVY = dv.y + l1x * sqrt(oMass / rocheDist) * 0.5;
                        events[slot] = evt;
                    }
                }
            }
        }
    }
}
