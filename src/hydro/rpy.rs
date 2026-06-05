//! Rotne-Prager-Yamakawa pair mobility blocks (equal radius in this task; the
//! unequal-radius RPYW generalization is added in Task A8).

use crate::hydro::mat3::Mat3;
use crate::vec3::Vec3;
use std::f64::consts::PI;

/// Equal-radius RPY mobility block coupling velocity of `i` to force on `j`,
/// for separation `r_vec = r_i - r_j`. `a` is the common radius, `eta` the
/// viscosity. Caller supplies the (possibly minimum-image) separation.
#[must_use]
pub fn rpy_pair_equal(r_vec: Vec3, a: f64, eta: f64) -> Mat3 {
    let r2 = r_vec.norm2();
    let r = r2.sqrt();
    let e = r_vec.scale(1.0 / r);
    let ee = Mat3::outer(e);
    let id = Mat3::identity();
    if r >= 2.0 * a {
        let pref = 1.0 / (8.0 * PI * eta * r);
        let c_i = 1.0 + 2.0 * a * a / (3.0 * r2);
        let c_e = 1.0 - 2.0 * a * a / r2;
        id.scale(pref * c_i).add(ee.scale(pref * c_e))
    } else {
        let mu0 = 1.0 / (6.0 * PI * eta * a);
        let c_i = 1.0 - 9.0 * r / (32.0 * a);
        let c_e = 3.0 * r / (32.0 * a);
        id.scale(mu0 * c_i).add(ee.scale(mu0 * c_e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: f64 = 1.0;
    const ETA: f64 = 1.0 / (6.0 * PI); // makes mu0 = 1 exactly

    #[test]
    fn far_field_decays_like_oseen() {
        // At large r the leading term is the Oseen tensor (1/(8 pi eta r)).
        // Along x (e = x_hat): block_xx = pref (c_i + c_e), block_yy = pref c_i.
        // r must be deep in the far field for the xx/yy -> 2 limit to hold to 1e-3
        // (the deviation is O(a^2/r^2)).
        let r = 500.0;
        let m = rpy_pair_equal(Vec3::new(r, 0.0, 0.0), A, ETA);
        let pref = 1.0 / (8.0 * PI * ETA * r);
        let c_i = 1.0 + 2.0 * A * A / (3.0 * r * r);
        let c_e = 1.0 - 2.0 * A * A / (r * r);
        assert!((m.0[0] - pref * (c_i + c_e)).abs() < 1e-12, "xx block");
        assert!((m.0[4] - pref * c_i).abs() < 1e-12, "yy block");
        // ratio xx/yy -> 2 as r -> infinity (Oseen anisotropy)
        assert!((m.0[0] / m.0[4] - 2.0).abs() < 1e-3, "Oseen ratio");
    }

    #[test]
    fn continuous_at_contact() {
        // The two branches must agree at r = 2a.
        let eps = 1e-7;
        let inner = rpy_pair_equal(Vec3::new(2.0 * A - eps, 0.0, 0.0), A, ETA);
        let outer = rpy_pair_equal(Vec3::new(2.0 * A + eps, 0.0, 0.0), A, ETA);
        for k in 0..9 {
            assert!(
                (inner.0[k] - outer.0[k]).abs() < 1e-5,
                "branch {k}: {} vs {}",
                inner.0[k],
                outer.0[k]
            );
        }
    }
}

/// Unequal-radius RPYW mobility block (Zuk, Wajnryb, Mizerski, Szymczak,
/// J. Fluid Mech. 741 (2014), R5). Reduces to `rpy_pair_equal` when `ai == aj`.
#[must_use]
pub fn rpyw_pair(r_vec: Vec3, ai: f64, aj: f64, eta: f64) -> Mat3 {
    let r = r_vec.norm2().sqrt();
    let e = r_vec.scale(1.0 / r);
    let ee = Mat3::outer(e);
    let id = Mat3::identity();
    let sum = ai + aj;
    let diff = (ai - aj).abs();
    if r >= sum {
        let pref = 1.0 / (8.0 * PI * eta * r);
        let c_i = 1.0 + (ai * ai + aj * aj) / (3.0 * r * r);
        let c_e = 1.0 - (ai * ai + aj * aj) / (r * r);
        id.scale(pref * c_i).add(ee.scale(pref * c_e))
    } else if r > diff {
        // overlapping, neither centre inside the other
        let s = r;
        let base = 1.0 / (6.0 * PI * eta * ai * aj);
        let a_coeff = base
            * (16.0 * s.powi(3) * sum - ((ai - aj).powi(2) + 3.0 * s * s).powi(2))
            / (32.0 * s.powi(3));
        let b_coeff = base * (3.0 * ((ai - aj).powi(2) - s * s).powi(2)) / (32.0 * s.powi(3));
        id.scale(a_coeff).add(ee.scale(b_coeff))
    } else {
        // one centre inside the other: the larger sphere's self mobility (isotropic)
        let amax = ai.max(aj);
        id.scale(1.0 / (6.0 * PI * eta * amax))
    }
}

#[cfg(test)]
mod rpyw_tests {
    use super::*;

    const ETA: f64 = 1.0 / (6.0 * PI);

    #[test]
    fn reduces_to_equal_radius_kernel() {
        let a = 1.3;
        for &r in &[3.0, 2.6, 2.0, 1.5, 0.7] {
            let v = Vec3::new(r, 0.0, 0.0);
            let eq = rpy_pair_equal(v, a, ETA);
            let gw = rpyw_pair(v, a, a, ETA);
            for k in 0..9 {
                assert!((eq.0[k] - gw.0[k]).abs() < 1e-9, "r={r} block {k}: {} vs {}", eq.0[k], gw.0[k]);
            }
        }
    }

    #[test]
    fn unequal_block_is_continuous_at_outer_contact() {
        let (ai, aj) = (1.0, 2.0);
        let eps = 1e-7;
        let inner = rpyw_pair(Vec3::new(ai + aj - eps, 0.0, 0.0), ai, aj, ETA);
        let outer = rpyw_pair(Vec3::new(ai + aj + eps, 0.0, 0.0), ai, aj, ETA);
        for k in 0..9 {
            assert!((inner.0[k] - outer.0[k]).abs() < 1e-4, "block {k}: {} vs {}", inner.0[k], outer.0[k]);
        }
    }
}
