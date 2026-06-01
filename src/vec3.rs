//! Minimal 3D vector for Brownian dynamics.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    #[must_use]
    pub fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    #[must_use]
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// Squared Euclidean norm.
    #[must_use]
    pub fn norm2(self) -> f64 {
        self.dot(self)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 0.0, -1.0);
        assert_eq!(a + b, Vec3::new(5.0, 2.0, 2.0));
        assert_eq!(a - b, Vec3::new(-3.0, 2.0, 4.0));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert!((a.dot(b) - 1.0).abs() < 1e-12); // 4 + 0 - 3
        assert!((a.norm2() - 14.0).abs() < 1e-12); // 1 + 4 + 9
    }
}
