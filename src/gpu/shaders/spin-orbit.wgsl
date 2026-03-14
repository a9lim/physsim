// Spin-orbit coupling: energy transfer + translational kicks.
// Stern-Gerlach: F = +mu * grad(Bz) (magnetic moment in B-field gradient)
// Mathisson-Papapetrou: F = -L * grad(Bgz) (angular momentum in gravitomagnetic gradient)
// Energy coupling: transfers KE between translational and rotational DOFs.
//
// 9 storage buffers (was 12). Uses packed AllForces + ParticleDerived.

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;
@group(0) @binding(1) var<storage, read_write> velWX: array<f32>;
@group(0) @binding(2) var<storage, read_write> velWY: array<f32>;
@group(0) @binding(3) var<storage, read_write> angW: array<f32>;
@group(0) @binding(4) var<storage, read> mass: array<f32>;
@group(0) @binding(5) var<storage, read> charge: array<f32>;
@group(0) @binding(6) var<storage, read> radius: array<f32>;
@group(0) @binding(7) var<storage, read_write> derived: array<ParticleDerived>;
@group(0) @binding(8) var<storage, read> flags: array<u32>;
@group(0) @binding(9) var<storage, read_write> allForces: array<AllForces>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.aliveCount) { return; }
    if ((flags[idx] & FLAG_ALIVE) == 0u) { return; }

    let spinOrbitOn = hasToggle0(SPIN_ORBIT_BIT);
    let hasMag = hasToggle0(MAGNETIC_BIT);
    let hasGM = hasToggle0(GRAVITOMAG_BIT);
    if (!spinOrbitOn || (!hasMag && !hasGM)) { return; }

    let aw = angW[idx];
    let d = derived[idx];
    let aVel = d.angVel;
    if (abs(aVel) < EPSILON) { return; }

    let angMom = d.angMomentum;
    if (abs(angMom) < EPSILON) { return; }

    let m = mass[idx];
    let q = charge[idx];
    let invM = select(0.0, 1.0 / m, m > EPSILON);
    let dtOverM = uniforms.dt * invM;
    let r = radius[idx];
    let relOn = hasToggle0(RELATIVITY_BIT);

    let vx = d.velX;
    let vy = d.velY;
    let mu = d.magMoment;
    let L = angMom;

    let af = allForces[idx];
    let dBzdx = af.bFieldGrads.x;
    let dBzdy = af.bFieldGrads.y;
    let dBgzdx = af.bFieldGrads.z;
    let dBgzdy = af.bFieldGrads.w;

    var newAngW = aw;
    var kickX: f32 = 0.0;
    var kickY: f32 = 0.0;
    var scX: f32 = 0.0;  // spin-curvature display force
    var scY: f32 = 0.0;

    if (hasMag && abs(q) > EPSILON) {
        // Energy transfer: angw -= mu * (v . grad(Bz)) * dt / (I*omega)
        newAngW -= mu * (vx * dBzdx + vy * dBzdy) * uniforms.dt / angMom;
        // Stern-Gerlach translational kick
        let sgX = mu * dBzdx * dtOverM;
        let sgY = mu * dBzdy * dtOverM;
        kickX += sgX;
        kickY += sgY;
        scX += mu * dBzdx;
        scY += mu * dBzdy;
    }

    if (hasGM) {
        // Energy transfer: angw -= L * (v . grad(Bgz)) * dt / (I*omega)
        newAngW -= L * (vx * dBgzdx + vy * dBgzdy) * uniforms.dt / angMom;
        // Mathisson-Papapetrou translational kick (GEM sign flip)
        let mpX = -L * dBgzdx * dtOverM;
        let mpY = -L * dBgzdy * dtOverM;
        kickX += mpX;
        kickY += mpY;
        scX += -L * dBgzdx;
        scY += -L * dBgzdy;
    }

    // NaN guard
    if (newAngW != newAngW) { newAngW = 0.0; }

    // Derive angular velocity from angular proper velocity
    let sr = newAngW * r;
    let newAngVel = select(newAngW, newAngW / sqrt(1.0 + sr * sr), relOn);

    angW[idx] = newAngW;

    // Update angVel in derived struct
    var dOut = derived[idx];
    dOut.angVel = newAngVel;
    derived[idx] = dOut;

    // Apply translational kicks to proper velocity
    velWX[idx] = velWX[idx] + kickX;
    velWY[idx] = velWY[idx] + kickY;

    // Store spin-curvature display force in allForces.f2.zw
    var afOut = allForces[idx];
    afOut.f2.z = scX;
    afOut.f2.w = scY;
    allForces[idx] = afOut;
}
