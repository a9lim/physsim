// ─── Preset Definitions ───

export const PRESETS = {
    solar(sim) {
        const cx = sim.width / 2, cy = sim.height / 2;
        sim.addParticle(cx, cy, 0, 0, { mass: 80, charge: 0, spin: 0 });
        for (let i = 0; i < 5; i++) {
            const dist = 100 + i * 60;
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.sqrt(80 / dist);
            const cos = Math.cos(angle), sin = Math.sin(angle);
            sim.addParticle(cx + cos * dist, cy + sin * dist, -sin * speed, cos * speed,
                { mass: 0.5 + Math.random() * 1.5, charge: 0, spin: 0 });
        }
    },

    binary(sim) {
        const cx = sim.width / 2, cy = sim.height / 2;
        const dist = 100;
        const starMass = 50;
        const speed = Math.sqrt(starMass / (2 * dist));
        sim.addParticle(cx - dist, cy, 0, speed, { mass: starMass, charge: 0, spin: 10 });
        sim.addParticle(cx + dist, cy, 0, -speed, { mass: starMass, charge: 0, spin: 10 });
    },

    galaxy(sim) {
        const cx = sim.width / 2, cy = sim.height / 2;
        sim.addParticle(cx, cy, 0, 0, { mass: 150, charge: 0, spin: 30 });
        for (let i = 0; i < 200; i++) {
            const dist = 150 + Math.random() * 300;
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.sqrt(150 / dist);
            const cos = Math.cos(angle), sin = Math.sin(angle);
            sim.addParticle(cx + cos * dist, cy + sin * dist, -sin * speed, cos * speed, {
                mass: 0.1 + Math.random() * 0.4,
                charge: (Math.random() - 0.5) * 5,
                spin: (Math.random() - 0.5) * 10
            });
        }
    },

    collision(sim) {
        const cx = sim.width / 2, cy = sim.height / 2;
        for (let i = 0; i < 50; i++) {
            sim.addParticle(cx - 200 + Math.random() * 50, cy + Math.random() * 50, 0.5, 0, { mass: 1, charge: 0, spin: 0 });
            sim.addParticle(cx + 200 + Math.random() * 50, cy + Math.random() * 50, -0.5, 0, { mass: 1, charge: 0, spin: 0 });
        }
    },

    magnetic(sim) {
        const cx = sim.width / 2, cy = sim.height / 2;
        const spacing = 80;
        for (let i = -2; i <= 2; i++) {
            for (let j = -2; j <= 2; j++) {
                sim.addParticle(
                    cx + i * spacing + (Math.random() - 0.5) * 20,
                    cy + j * spacing + (Math.random() - 0.5) * 20,
                    (Math.random() - 0.5) * 0.1,
                    (Math.random() - 0.5) * 0.1,
                    { mass: 3 + Math.random() * 2, charge: 5 + Math.random() * 5, spin: 20 + Math.random() * 10 }
                );
            }
        }
    },
};

export function loadPreset(name, sim) {
    sim.particles = [];
    sim.camera.x = sim.width / 2;
    sim.camera.y = sim.height / 2;
    sim.camera.zoom = 1;
    sim.updateZoomDisplay();

    const preset = PRESETS[name];
    if (preset) preset(sim);
}
