import Vec2 from './vec2.js';

class Rect {
    constructor(x, y, w, h) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
    }

    contains(point) {
        return (point.x >= this.x - this.w &&
            point.x <= this.x + this.w &&
            point.y >= this.y - this.h &&
            point.y <= this.y + this.h);
    }

    intersects(range) {
        return !(range.x - range.w > this.x + this.w ||
            range.x + range.w < this.x - this.w ||
            range.y - range.h > this.y + this.h ||
            range.y + range.h < this.y - this.h);
    }
}

export default class QuadTree {
    constructor(boundary, capacity) {
        this.boundary = boundary;
        this.capacity = capacity;
        this.points = [];
        this.divided = false;

        this.totalMass = 0;
        this.totalCharge = 0;
        this.totalMagneticMoment = 0;
        this.totalAngularMomentum = 0;
        this.centerOfMass = new Vec2(boundary.x, boundary.y);
    }

    subdivide() {
        const { x, y, w, h } = this.boundary;
        const hw = w / 2, hh = h / 2;

        this.northwest = new QuadTree(new Rect(x - hw, y - hh, hw, hh), this.capacity);
        this.northeast = new QuadTree(new Rect(x + hw, y - hh, hw, hh), this.capacity);
        this.southwest = new QuadTree(new Rect(x - hw, y + hh, hw, hh), this.capacity);
        this.southeast = new QuadTree(new Rect(x + hw, y + hh, hw, hh), this.capacity);

        this.divided = true;
    }

    insert(particle) {
        if (!this.boundary.contains(particle.pos)) {
            return false;
        }

        if (this.points.length < this.capacity && !this.divided) {
            this.points.push(particle);
            return true;
        }

        if (!this.divided) {
            this.subdivide();
            for (const p of this.points) {
                this.northwest.insert(p) ||
                this.northeast.insert(p) ||
                this.southwest.insert(p) ||
                this.southeast.insert(p);
            }
            this.points = [];
        }

        return this.northwest.insert(particle) ||
            this.northeast.insert(particle) ||
            this.southwest.insert(particle) ||
            this.southeast.insert(particle);
    }

    calculateMassDistribution() {
        if (!this.divided) {
            const pts = this.points;
            if (pts.length === 0) return;

            let mass = 0, charge = 0, magMom = 0, angMom = 0;
            let comX = 0, comY = 0;

            for (const p of pts) {
                mass += p.mass;
                charge += p.charge;
                magMom += p.charge * p.spin;
                angMom += p.mass * p.spin;
                comX += p.pos.x * p.mass;
                comY += p.pos.y * p.mass;
            }

            this.totalMass = mass;
            this.totalCharge = charge;
            this.totalMagneticMoment = magMom;
            this.totalAngularMomentum = angMom;

            if (mass > 0) {
                this.centerOfMass.set(comX / mass, comY / mass);
            }
        } else {
            const children = [this.northwest, this.northeast, this.southwest, this.southeast];
            for (const child of children) {
                child.calculateMassDistribution();
            }

            let mass = 0, charge = 0, magMom = 0, angMom = 0;
            let comX = 0, comY = 0;

            for (const c of children) {
                mass += c.totalMass;
                charge += c.totalCharge;
                magMom += c.totalMagneticMoment;
                angMom += c.totalAngularMomentum;
                comX += c.centerOfMass.x * c.totalMass;
                comY += c.centerOfMass.y * c.totalMass;
            }

            this.totalMass = mass;
            this.totalCharge = charge;
            this.totalMagneticMoment = magMom;
            this.totalAngularMomentum = angMom;

            if (mass > 0) {
                this.centerOfMass.set(comX / mass, comY / mass);
            }
        }
    }

    query(range, found) {
        if (!found) found = [];
        if (!this.boundary.intersects(range)) {
            return found;
        }

        if (!this.divided) {
            for (const p of this.points) {
                if (range.contains(p.pos)) {
                    found.push(p);
                }
            }
        } else {
            this.northwest.query(range, found);
            this.northeast.query(range, found);
            this.southwest.query(range, found);
            this.southeast.query(range, found);
        }

        return found;
    }
}
export { Rect };
