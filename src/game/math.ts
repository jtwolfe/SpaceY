/** Pad-ENU linear algebra. x east, y north, z up. */

export class Vec3 {
  constructor(
    public x = 0,
    public y = 0,
    public z = 0,
  ) {}

  static readonly ZERO = new Vec3(0, 0, 0);
  static readonly X = new Vec3(1, 0, 0);
  static readonly Y = new Vec3(0, 1, 0);
  static readonly Z = new Vec3(0, 0, 1);

  clone() {
    return new Vec3(this.x, this.y, this.z);
  }
  set(x: number, y: number, z: number) {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }
  copy(o: Vec3) {
    this.x = o.x;
    this.y = o.y;
    this.z = o.z;
    return this;
  }
  add(o: Vec3) {
    return new Vec3(this.x + o.x, this.y + o.y, this.z + o.z);
  }
  sub(o: Vec3) {
    return new Vec3(this.x - o.x, this.y - o.y, this.z - o.z);
  }
  scale(s: number) {
    return new Vec3(this.x * s, this.y * s, this.z * s);
  }
  neg() {
    return new Vec3(-this.x, -this.y, -this.z);
  }
  dot(o: Vec3) {
    return this.x * o.x + this.y * o.y + this.z * o.z;
  }
  cross(o: Vec3) {
    return new Vec3(
      this.y * o.z - this.z * o.y,
      this.z * o.x - this.x * o.z,
      this.x * o.y - this.y * o.x,
    );
  }
  lenSq() {
    return this.dot(this);
  }
  len() {
    return Math.hypot(this.x, this.y, this.z);
  }
  normalized() {
    const n = this.len();
    return n < 1e-14 ? new Vec3() : this.scale(1 / n);
  }
  clampLen(max: number) {
    const n = this.len();
    return n > max ? this.scale(max / n) : this.clone();
  }
  lerp(o: Vec3, t: number) {
    return this.scale(1 - t).add(o.scale(t));
  }
}

export class Quat {
  constructor(
    public w = 1,
    public x = 0,
    public y = 0,
    public z = 0,
  ) {}

  static readonly IDENTITY = new Quat(1, 0, 0, 0);

  clone() {
    return new Quat(this.w, this.x, this.y, this.z);
  }
  hemisphere() {
    return this.w < 0 ? new Quat(-this.w, -this.x, -this.y, -this.z) : this;
  }
  copy(o: Quat) {
    this.w = o.w;
    this.x = o.x;
    this.y = o.y;
    this.z = o.z;
    return this;
  }

  static fromAxisAngle(axis: Vec3, angle: number) {
    const a = axis.normalized();
    const h = angle * 0.5;
    const s = Math.sin(h);
    return new Quat(Math.cos(h), a.x * s, a.y * s, a.z * s);
  }

  /** Body→world from images of body axes (columns of R). */
  static fromAxes(x: Vec3, y: Vec3, z: Vec3) {
    const m00 = x.x,
      m10 = x.y,
      m20 = x.z;
    const m01 = y.x,
      m11 = y.y,
      m21 = y.z;
    const m02 = z.x,
      m12 = z.y,
      m22 = z.z;
    const tr = m00 + m11 + m22;
    let q: Quat;
    if (tr > 0) {
      const s = 0.5 / Math.sqrt(tr + 1);
      q = new Quat(0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s);
    } else if (m00 >= m11 && m00 >= m22) {
      const s = 2 * Math.sqrt(Math.max(0, 1 + m00 - m11 - m22));
      q =
        s < 1e-18
          ? new Quat()
          : new Quat((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s);
    } else if (m11 >= m22) {
      const s = 2 * Math.sqrt(Math.max(0, 1 + m11 - m00 - m22));
      q =
        s < 1e-18
          ? new Quat()
          : new Quat((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s);
    } else {
      const s = 2 * Math.sqrt(Math.max(0, 1 + m22 - m00 - m11));
      q =
        s < 1e-18
          ? new Quat()
          : new Quat((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s);
    }
    return q.normalized();
  }

  static fromRotationArc(from: Vec3, to: Vec3) {
    const f = from.normalized();
    const t = to.normalized();
    const d = f.dot(t);
    if (d > 0.999999) return new Quat();
    if (d < -0.999999) {
      const axis = (Math.abs(f.x) < 0.9 ? f.cross(Vec3.X) : f.cross(Vec3.Y)).normalized();
      return Quat.fromAxisAngle(axis, Math.PI);
    }
    const c = f.cross(t);
    return new Quat(1 + d, c.x, c.y, c.z).normalized();
  }

  normalized() {
    const n = Math.hypot(this.w, this.x, this.y, this.z);
    if (n < 1e-18) return new Quat();
    return new Quat(this.w / n, this.x / n, this.y / n, this.z / n);
  }

  mul(o: Quat) {
    return new Quat(
      this.w * o.w - this.x * o.x - this.y * o.y - this.z * o.z,
      this.w * o.x + this.x * o.w + this.y * o.z - this.z * o.y,
      this.w * o.y - this.x * o.z + this.y * o.w + this.z * o.x,
      this.w * o.z + this.x * o.y - this.y * o.x + this.z * o.w,
    );
  }

  conjugate() {
    return new Quat(this.w, -this.x, -this.y, -this.z);
  }

  rotate(v: Vec3) {
    const qv = new Vec3(this.x, this.y, this.z);
    const t = qv.cross(v).scale(2);
    return v.add(t.scale(this.w)).add(qv.cross(t));
  }

  /** Semi-implicit integrate body rate ω (rad/s). */
  integrate(omega: Vec3, dt: number) {
    const ow = new Quat(0, omega.x, omega.y, omega.z);
    const dq = this.mul(ow);
    return new Quat(
      this.w + 0.5 * dq.w * dt,
      this.x + 0.5 * dq.x * dt,
      this.y + 0.5 * dq.y * dt,
      this.z + 0.5 * dq.z * dt,
    ).normalized();
  }
}

export function clamp(x: number, a: number, b: number) {
  return Math.min(b, Math.max(a, x));
}
export function saturate(x: number) {
  return clamp(x, 0, 1);
}
export function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}
export function angleBetween(a: Vec3, b: Vec3) {
  const d = clamp(a.normalized().dot(b.normalized()), -1, 1);
  return Math.acos(d);
}
export function slew(cur: number, tgt: number, rate: number, dt: number) {
  const d = tgt - cur;
  const max = rate * dt;
  if (d > max) return cur + max;
  if (d < -max) return cur - max;
  return tgt;
}

/** Mulberry32. */
export function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
