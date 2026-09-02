//! Small f64 linear-algebra helpers. Positions are Earth-radius scale; f64 is required.

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    pub const X: Vec3 = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
    pub const Y: Vec3 = Vec3 { x: 0.0, y: 1.0, z: 0.0 };
    pub const Z: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 1.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }

    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n < 1e-14 {
            Self::ZERO
        } else {
            self * (1.0 / n)
        }
    }

    pub fn clamp_norm(self, max: f64) -> Self {
        let n = self.norm();
        if n > max {
            self * (max / n)
        } else {
            self
        }
    }

    pub fn lerp(self, o: Self, t: f64) -> Self {
        self * (1.0 - t) + o * t
    }

    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}
impl std::ops::Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, s: f64) -> Self {
        Self::new(self.x / s, self.y / s, self.z / s)
    }
}
impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl std::ops::SubAssign for Vec3 {
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

/// Unit quaternion, Hamilton convention, stored as (w, x, y, z).
#[derive(Clone, Copy, Debug)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Quat = Quat {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let a = axis.normalized();
        let h = angle * 0.5;
        let s = h.sin();
        Self {
            w: h.cos(),
            x: a.x * s,
            y: a.y * s,
            z: a.z * s,
        }
    }

    /// Rotate `from` onto `to` (both treated as directions).
    pub fn from_rotation_arc(from: Vec3, to: Vec3) -> Self {
        let f = from.normalized();
        let t = to.normalized();
        let d = f.dot(t);
        if d > 0.999999 {
            return Self::IDENTITY;
        }
        if d < -0.999999 {
            let axis = if f.x.abs() < 0.9 {
                f.cross(Vec3::X)
            } else {
                f.cross(Vec3::Y)
            }
            .normalized();
            return Self::from_axis_angle(axis, std::f64::consts::PI);
        }
        let c = f.cross(t);
        let w = 1.0 + d;
        Self::new(w, c.x, c.y, c.z).normalized()
    }

    pub fn normalized(self) -> Self {
        let n = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if n < 1e-18 {
            Self::IDENTITY
        } else {
            Self {
                w: self.w / n,
                x: self.x / n,
                y: self.y / n,
                z: self.z / n,
            }
        }
    }

    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn mul(self, o: Self) -> Self {
        Self {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }

    pub fn rotate(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t = qv.cross(v) * 2.0;
        v + t * self.w + qv.cross(t)
    }

    /// Body rates (rad/s) integrated with first-order multiplicative update.
    pub fn integrate(self, omega_body: Vec3, dt: f64) -> Self {
        let ang = omega_body.norm();
        if ang < 1e-14 {
            return self;
        }
        let dq = Quat::from_axis_angle(omega_body, ang * dt);
        self.mul(dq).normalized()
    }

    /// 3-2-1 yaw/pitch/roll of body X/Y/Z relative to a reference frame.
    pub fn yaw_pitch_roll(self) -> (f64, f64, f64) {
        let x = self.rotate(Vec3::X);
        let y = self.rotate(Vec3::Y);
        let yaw = y.x.atan2(x.x);
        let pitch = -x.z.atan2((x.x * x.x + x.y * x.y).sqrt());
        let roll = self.rotate(Vec3::Z).y.atan2(self.rotate(Vec3::Z).z);
        (yaw, pitch, roll)
    }
}

pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

pub fn saturate(x: f64) -> f64 {
    clamp(x, 0.0, 1.0)
}

pub fn wrap_pi(a: f64) -> f64 {
    let mut x = (a + std::f64::consts::PI) % (2.0 * std::f64::consts::PI);
    if x < 0.0 {
        x += 2.0 * std::f64::consts::PI;
    }
    x - std::f64::consts::PI
}

pub fn angle_between(a: Vec3, b: Vec3) -> f64 {
    let d = clamp(a.normalized().dot(b.normalized()), -1.0, 1.0);
    d.acos()
}
