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
        const spin = 0.8 / Math.cbrt(starMass); // 80% of surface-velocity cap
        sim.addParticle(cx - dist, cy, 0, speed, { mass: starMass, charge: 0, spin });
        sim.addParticle(cx + dist, cy, 0, -speed, { mass: starMass, charge: 0, spin });
    },

    galaxy(sim) {
        const cx = sim.width / 2, cy = sim.height / 2;
        const coreMass = 150;
        sim.addParticle(cx, cy, 0, 0, { mass: coreMass, charge: 0, spin: 0.8 / Math.cbrt(coreMass) });
        for (let i = 0; i < 200; i++) {
            const dist = 150 + Math.random() * 300;
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.sqrt(coreMass / dist);
            const cos = Math.cos(angle), sin = Math.sin(angle);
            const m = 0.1 + Math.random() * 0.4;
            sim.addParticle(cx + cos * dist, cy + sin * dist, -sin * speed, cos * speed, {
                mass: m,
                charge: (Math.random() - 0.5) * 5,
                spin: (Math.random() - 0.5) * 1.5 / Math.cbrt(m)
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
                const m = 3 + Math.random() * 2;
                sim.addParticle(
                    cx + i * spacing + (Math.random() - 0.5) * 20,
                    cy + j * spacing + (Math.random() - 0.5) * 20,
                    (Math.random() - 0.5) * 0.1,
                    (Math.random() - 0.5) * 0.1,
                    { mass: m, charge: 5 + Math.random() * 5, spin: 0.8 / Math.cbrt(m) }
                );
            }
        }
    },
};

const PRESET_LABELS = {
    solar: 'Solar System',
    binary: 'Binary Stars',
    galaxy: 'Galaxy',
    collision: 'Collision',
    magnetic: 'Magnetic',
};

export function loadPreset(name, sim) {
    sim.particles = [];
    sim.camera.reset(sim.width / 2, sim.height / 2, 1);

    const preset = PRESETS[name];
    if (preset) {
        preset(sim);
        showToast(PRESET_LABELS[name] || name);
    }
}
