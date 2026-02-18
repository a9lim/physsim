import Vec2 from './vec2.js';

class Rect {
    constructor(x, y, w, h) {
        this.x = x; // Center x
        this.y = y; // Center y
        this.w = w; // Half width
        this.h = h; // Half height
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
    }

    subdivide() {
        const x = this.boundary.x;
        const y = this.boundary.y;
        const w = this.boundary.w;
        const h = this.boundary.h;

        this.northwest = new QuadTree(new Rect(x - w / 2, y - h / 2, w / 2, h / 2), this.capacity);
        this.northeast = new QuadTree(new Rect(x + w / 2, y - h / 2, w / 2, h / 2), this.capacity);
        this.southwest = new QuadTree(new Rect(x - w / 2, y + h / 2, w / 2, h / 2), this.capacity);
        this.southeast = new QuadTree(new Rect(x + w / 2, y + h / 2, w / 2, h / 2), this.capacity);

        this.divided = true;
    }

    insert(particle) {
        if (!this.boundary.contains(particle.pos)) {
            return false; // Point not in this node's boundary
        }

        if (this.points.length < this.capacity && !this.divided) {
            this.points.push(particle);
            return true;
        }

        if (!this.divided) {
            this.subdivide();
            // Move existing points to children
            for (let p of this.points) {
                if (this.northwest.insert(p)) continue;
                if (this.northeast.insert(p)) continue;
                if (this.southwest.insert(p)) continue;
                if (this.southeast.insert(p)) continue;
            }
            this.points = []; // Points are now in children
        }

        if (this.northwest.insert(particle)) return true;
        if (this.northeast.insert(particle)) return true;
        if (this.southwest.insert(particle)) return true;
        if (this.southeast.insert(particle)) return true;

        return false;
    }

    calculateMassDistribution() {
        if (!this.divided) {
            this.totalMass = 0;
            this.totalCharge = 0;
            let centerOfMassX = 0;
            let centerOfMassY = 0;

            if (this.points.length > 0) {
                for (let p of this.points) {
                    this.totalMass += p.mass;
                    this.totalCharge += p.charge;
                    centerOfMassX += p.pos.x * p.mass;
                    centerOfMassY += p.pos.y * p.mass;
                }
                if (this.totalMass > 0) {
                    this.centerOfMass = new Vec2(centerOfMassX / this.totalMass, centerOfMassY / this.totalMass);
                } else {
                    this.centerOfMass = new Vec2(this.boundary.x, this.boundary.y);
                }
            } else {
                this.centerOfMass = new Vec2(this.boundary.x, this.boundary.y);
            }
        } else {
            this.northwest.calculateMassDistribution();
            this.northeast.calculateMassDistribution();
            this.southwest.calculateMassDistribution();
            this.southeast.calculateMassDistribution();

            this.totalMass = this.northwest.totalMass + this.northeast.totalMass + this.southwest.totalMass + this.southeast.totalMass;
            this.totalCharge = this.northwest.totalCharge + this.northeast.totalCharge + this.southwest.totalCharge + this.southeast.totalCharge;

            if (this.totalMass > 0) {
                const comX = (this.northwest.centerOfMass.x * this.northwest.totalMass +
                    this.northeast.centerOfMass.x * this.northeast.totalMass +
                    this.southwest.centerOfMass.x * this.southwest.totalMass +
                    this.southeast.centerOfMass.x * this.southeast.totalMass) / this.totalMass;

                const comY = (this.northwest.centerOfMass.y * this.northwest.totalMass +
                    this.northeast.centerOfMass.y * this.northeast.totalMass +
                    this.southwest.centerOfMass.y * this.southwest.totalMass +
                    this.southeast.centerOfMass.y * this.southeast.totalMass) / this.totalMass;

                this.centerOfMass = new Vec2(comX, comY);
            } else {
                this.centerOfMass = new Vec2(this.boundary.x, this.boundary.y);
            }
        }
    }

    query(range, found) {
        if (!found) found = [];
        if (!this.boundary.intersects(range)) {
            return found;
        }

        if (!this.divided) {
            for (let p of this.points) {
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
