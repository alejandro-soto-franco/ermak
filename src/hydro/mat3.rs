//! Minimal 3x3 matrix for RPY mobility blocks: row-major `[f64; 9]` with the
//! handful of operations the tensor algebra needs (identity, outer product,
//! scale, add, matrix-vector). Kept local so the crate stays nalgebra-free.

use crate::vec3::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3(pub [f64; 9]);

impl Mat3 {
    pub const ZERO: Mat3 = Mat3([0.0; 9]);

    #[must_use]
    pub fn identity() -> Mat3 {
        Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    /// Outer product `e e^T` of a vector with itself.
    #[must_use]
    pub fn outer(e: Vec3) -> Mat3 {
        Mat3([
            e.x * e.x,
            e.x * e.y,
            e.x * e.z,
            e.y * e.x,
            e.y * e.y,
            e.y * e.z,
            e.z * e.x,
            e.z * e.y,
            e.z * e.z,
        ])
    }

    #[must_use]
    pub fn scale(self, s: f64) -> Mat3 {
        let mut m = self.0;
        for v in &mut m {
            *v *= s;
        }
        Mat3(m)
    }

    // Named `add` (not the `Add` trait) deliberately: the tensor algebra reads as
    // chained `.scale(..).add(..)` calls, which is clearer here than operators.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn add(self, other: Mat3) -> Mat3 {
        let mut m = self.0;
        for (mi, oi) in m.iter_mut().zip(other.0.iter()) {
            *mi += *oi;
        }
        Mat3(m)
    }

    /// Matrix-vector product `M v`.
    #[must_use]
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        let m = self.0;
        Vec3::new(
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[3] * v.x + m[4] * v.y + m[5] * v.z,
            m[6] * v.x + m[7] * v.y + m[8] * v.z,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_outer_scale_add_and_mulvec() {
        let i = Mat3::identity();
        assert_eq!(
            i.mul_vec(Vec3::new(2.0, 3.0, 4.0)),
            Vec3::new(2.0, 3.0, 4.0)
        );
        // e e^T applied to e gives |e|^2 e; for a unit vector that is e itself.
        let e = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(Mat3::outer(e).mul_vec(e), e);
        // (2 I + 0) scales by 2
        let m = i.scale(2.0).add(Mat3::ZERO);
        assert_eq!(
            m.mul_vec(Vec3::new(1.0, 1.0, 1.0)),
            Vec3::new(2.0, 2.0, 2.0)
        );
    }
}
