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

        if (this.points.length < this.capacity) {
            this.points.push(particle);
            return true;
        }

        if (!this.divided) {
            this.subdivide();
        }

        if (this.northwest.insert(particle)) return true;
        if (this.northeast.insert(particle)) return true;
        if (this.southwest.insert(particle)) return true;
        if (this.southeast.insert(particle)) return true;

        return false;
    }

    query(range, found) {
        if (!found) found = [];
        if (!this.boundary.intersects(range)) {
            return found;
        }

        for (let p of this.points) {
            if (range.contains(p.pos)) {
                found.push(p);
            }
        }

        if (this.divided) {
            this.northwest.query(range, found);
            this.northeast.query(range, found);
            this.southwest.query(range, found);
            this.southeast.query(range, found);
        }

        return found;
    }
}
export { Rect };
