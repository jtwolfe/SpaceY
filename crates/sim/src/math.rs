//! Small f64 linear-algebra helpers. Positions are Earth-radius scale; f64 is required.
//!
//! Transcendentals go through the `libm` crate on **every** target so
//! native tests and the wasm32 browser share one numeric path.

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
        sqrt(self.norm_squared())
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
        let s = sin(h);
        Self {
            w: cos(h),
            x: a.x * s,
            y: a.y * s,
            z: a.z * s,
        }
    }

    /// Body→world quaternion from the images of the body axes (columns of R).
    pub fn from_axes(x: Vec3, y: Vec3, z: Vec3) -> Self {
        let m00 = x.x;
        let m10 = x.y;
        let m20 = x.z;
        let m01 = y.x;
        let m11 = y.y;
        let m21 = y.z;
        let m02 = z.x;
        let m12 = z.y;
        let m22 = z.z;
        let tr = m00 + m11 + m22;
        let q = if tr > 0.0 {
            let s = 0.5 / sqrt(tr + 1.0);
            Self::new(0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s)
        } else if m00 >= m11 && m00 >= m22 {
            let s = 2.0 * sqrt((1.0 + m00 - m11 - m22).max(0.0));
            if s < 1e-18 {
                Self::IDENTITY
            } else {
                Self::new((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
            }
        } else if m11 >= m22 {
            let s = 2.0 * sqrt((1.0 + m11 - m00 - m22).max(0.0));
            if s < 1e-18 {
                Self::IDENTITY
            } else {
                Self::new((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
            }
        } else {
            let s = 2.0 * sqrt((1.0 + m22 - m00 - m11).max(0.0));
            if s < 1e-18 {
                Self::IDENTITY
            } else {
                Self::new((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
            }
        };
        q.normalized()
    }

    /// Flip the double-cover so `w >= 0` (stable as an MLP input).
    pub fn hemisphere(self) -> Self {
        if self.w < 0.0 {
            Self::new(-self.w, -self.x, -self.y, -self.z)
        } else {
            self
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
        let n = sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z);
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
        let yaw = atan2(y.x, x.x);
        let pitch = -atan2(x.z, sqrt(x.x * x.x + x.y * x.y));
        let roll = {
            let z = self.rotate(Vec3::Z);
            atan2(z.y, z.z)
        };
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
    acos(d)
}

/// Portable libm wrappers. Used by Earth frames, atmosphere, aero, and
/// guidance so native and wasm32 integrate the same trajectory.
#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}
#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}
#[inline]
pub fn tan(x: f64) -> f64 {
    libm::tan(x)
}
#[inline]
pub fn asin(x: f64) -> f64 {
    libm::asin(x)
}
#[inline]
pub fn acos(x: f64) -> f64 {
    libm::acos(x)
}
#[inline]
pub fn atan(x: f64) -> f64 {
    libm::atan(x)
}
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}
#[inline]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}
#[inline]
pub fn powf(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}
#[inline]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}
#[inline]
pub fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}

#[cfg(test)]
mod tests {
    #[test]
    fn portable_trig_is_finite() {
        let x = super::sin(0.2) * super::cos(-0.2);
        assert!(x.is_finite());
        assert!((super::sqrt(4.0) - 2.0).abs() < 1e-15);
        assert!((super::exp(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn from_axes_identity_and_known_rotation() {
        let q0 = super::Quat::from_axes(super::Vec3::X, super::Vec3::Y, super::Vec3::Z);
        assert!((q0.w - 1.0).abs() < 1e-12);
        assert!(q0.x.abs() < 1e-12 && q0.y.abs() < 1e-12 && q0.z.abs() < 1e-12);

        let q = super::Quat::from_axis_angle(super::Vec3::Y, 0.4);
        let x = q.rotate(super::Vec3::X);
        let y = q.rotate(super::Vec3::Y);
        let z = q.rotate(super::Vec3::Z);
        let q2 = super::Quat::from_axes(x, y, z).hemisphere();
        let q1 = q.hemisphere();
        assert!((q1.w - q2.w).abs() < 1e-9);
        assert!((q1.x - q2.x).abs() < 1e-9);
        assert!((q1.y - q2.y).abs() < 1e-9);
        assert!((q1.z - q2.z).abs() < 1e-9);
    }
}
