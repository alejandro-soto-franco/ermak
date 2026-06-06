//! CPU dense reference for the positively-split sinc^2 / Hasimoto periodic RPY
//! mobility (Milestone G4 step 2 follow-up (c), Task 1). This is a SECOND
//! mobility kernel, distinct from the GRPerY/Beenakker polynomial split in
//! [`crate::hydro::ewald`]: it uses the shape factor `sinc^2(ka)` (always >= 0)
//! and the Hasimoto splitting function `H(k, xi)` (always in `[0, 1]`), so the
//! real-space and wave-space halves are EACH symmetric positive-definite. That
//! is what makes a single-FFT positive-split Brownian draw possible, which the
//! indefinite GRPerY split forbids (`pse_wave::grpery_wave_split_is_indefinite`).
//!
//! Forms verified against Fiore, Balboa Usabiaga, Donev, Swan, J. Chem. Phys.
//! 146, 124116 (2017) (arXiv:1611.09322): shape factor `sinc^2(ka)` (eq 8),
//! split `H(k, xi) = (1 + k^2/(4 xi^2)) e^{-k^2/(4 xi^2)}` (eq 11), wave part
//! (eq 9), real-space scalars F,G (eqs A2/A3 + Appendix A coefficients), self
//! real part (eq A4). Parameter mapping to [`crate::hydro::ewald::EwaldParams`]:
//! Fiore's `xi = alpha = 1/(sigma sqrt 2)` (so the wave screen
//! `e^{-k^2/(4 xi^2)}` equals GRPerY's `e^{-k^2 sigma^2 / 2}`). GRPerY units
//! (`M_grpery = pi eta M_physical`): the wave prefactor is `pi/V` and the real
//! part is `(1/(6a)) [F (I - r_hat r_hat) + G r_hat r_hat]`. The k=0 mode is
//! dropped; there is no separate `mu0` or backflow term (the self mobility lives
//! entirely in the real+wave split via A4).

use crate::hydro::ewald::{EwaldParams, erfc};
use crate::hydro::mat3::Mat3;
use crate::vec3::Vec3;
use std::f64::consts::PI;

/// Fiore's splitting/screening parameter from the GRPerY width: `xi = 1/(sigma sqrt2)`.
#[inline]
#[must_use]
pub fn xi_of(ep: &EwaldParams) -> f64 {
    1.0 / (ep.sigma * std::f64::consts::SQRT_2)
}

/// Hasimoto splitting function `H(k, xi) = (1 + k^2/(4 xi^2)) e^{-k^2/(4 xi^2)}`
/// (eq 11), in `[0, 1]` for all k.
#[inline]
#[must_use]
pub fn hasimoto_h(k2: f64, xi: f64) -> f64 {
    let q = k2 / (4.0 * xi * xi);
    (1.0 + q) * (-q).exp()
}

/// Real-space scalar mobility functions `F(r), G(r)` (eqs A2/A3, Appendix A),
/// dimensionless (the `1/(6a)` prefactor is applied by the caller). `xi` is the
/// split parameter, `a` the radius.
#[must_use]
pub fn fg_scalars(r: f64, a: f64, xi: f64) -> (f64, f64) {
    let sqrt_pi = PI.sqrt();
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let xi2 = xi * xi;
    let xi3 = xi2 * xi;
    let xi4 = xi2 * xi2;
    let e_r = (-r2 * xi2).exp();
    let e_rm = (-(r - 2.0 * a).powi(2) * xi2).exp();
    let e_rp = (-(r + 2.0 * a).powi(2) * xi2).exp();
    let erfc_r = erfc(r * xi);
    let erfc_rm = erfc((r - 2.0 * a) * xi);
    let erfc_rp = erfc((r + 2.0 * a) * xi);

    let f1 = (18.0 * r2 * xi2 + 3.0) / (64.0 * sqrt_pi * a * r2 * xi3);
    let f2 =
        (2.0 * xi2 * (2.0 * a - r) * (4.0 * a * a + 4.0 * a * r + 9.0 * r2) - 2.0 * a - 3.0 * r)
            / (128.0 * sqrt_pi * a * r3 * xi3);
    let f3 = (-2.0 * xi2 * (2.0 * a + r) * (4.0 * a * a - 4.0 * a * r + 9.0 * r2) + 2.0 * a
        - 3.0 * r)
        / (128.0 * sqrt_pi * a * r3 * xi3);
    let f4 = (3.0 - 36.0 * r4 * xi4) / (128.0 * a * r3 * xi4);
    let f5 = (4.0 * xi4 * (r - 2.0 * a).powi(2) * (4.0 * a * a + 4.0 * a * r + 9.0 * r2) - 3.0)
        / (256.0 * a * r3 * xi4);
    let f6 = (4.0 * xi4 * (2.0 * a + r).powi(2) * (4.0 * a * a - 4.0 * a * r + 9.0 * r2) - 3.0)
        / (256.0 * a * r3 * xi4);

    let g1 = (6.0 * r2 * xi2 - 3.0) / (32.0 * sqrt_pi * a * r2 * xi3);
    let g2 = (-2.0 * xi2 * (r - 2.0 * a).powi(2) * (2.0 * a + 3.0 * r) + 2.0 * a + 3.0 * r)
        / (64.0 * sqrt_pi * a * r3 * xi3);
    let g3 = (2.0 * xi2 * (2.0 * a + r).powi(2) * (2.0 * a - 3.0 * r) - 2.0 * a + 3.0 * r)
        / (64.0 * sqrt_pi * a * r3 * xi3);
    let g4 = -3.0 * (4.0 * r4 * xi4 + 1.0) / (64.0 * a * r3 * xi4);
    let g5 =
        (3.0 - 4.0 * xi4 * (2.0 * a - r).powi(3) * (2.0 * a + 3.0 * r)) / (128.0 * a * r3 * xi4);
    let g6 =
        (3.0 - 4.0 * xi4 * (2.0 * a - 3.0 * r) * (2.0 * a + r).powi(3)) / (128.0 * a * r3 * xi4);

    let (f0, g0) = if r > 2.0 * a {
        (0.0, 0.0)
    } else {
        (
            -(r - 2.0 * a).powi(2) * (4.0 * a * a + 4.0 * a * r + 9.0 * r2) / (32.0 * a * r3),
            (2.0 * a - r).powi(3) * (2.0 * a + 3.0 * r) / (16.0 * a * r3),
        )
    };

    let f = f0 + f1 * e_r + f2 * e_rm + f3 * e_rp + f4 * erfc_r + f5 * erfc_rm + f6 * erfc_rp;
    let g = g0 + g1 * e_r + g2 * e_rm + g3 * e_rp + g4 * erfc_r + g5 * erfc_rm + g6 * erfc_rp;
    (f, g)
}

/// Wave-space (reciprocal) sinc^2 block coupling i to j. With `screen = true`
/// this is the screened wave part `M^(w)` (carries `H`, eq 9); with
/// `screen = false` it is the complete unsplit periodic mobility (eq 8, the
/// inverse-FT of which is the free-space RPY tensor). GRPerY units; k=0 dropped.
#[must_use]
pub fn recip_sinc_block(rij: Vec3, ep: &EwaldParams, screen: bool) -> Mat3 {
    let l = ep.box_l;
    let vol = l * l * l;
    let a = ep.a;
    let xi = xi_of(ep);
    let two_pi_l = 2.0 * PI / l;
    let id = Mat3::identity();
    let mut acc = Mat3::ZERO;
    for kx in -ep.k_max..=ep.k_max {
        for ky in -ep.k_max..=ep.k_max {
            for kz in -ep.k_max..=ep.k_max {
                if kx == 0 && ky == 0 && kz == 0 {
                    continue;
                }
                let kvec = Vec3::new(
                    f64::from(kx) * two_pi_l,
                    f64::from(ky) * two_pi_l,
                    f64::from(kz) * two_pi_l,
                );
                let k2 = kvec.norm2();
                let k = k2.sqrt();
                let kk = Mat3::outer(kvec.scale(1.0 / k));
                let ka = k * a;
                let sinc = ka.sin() / ka;
                let s2 = sinc * sinc;
                let mut scale = (PI / vol) * s2 / k2;
                if screen {
                    scale *= hasimoto_h(k2, xi);
                }
                // I - k_hat k_hat
                let tensor = id.add(kk.scale(-1.0));
                let phase = (kvec.x * rij.x + kvec.y * rij.y + kvec.z * rij.z).cos();
                acc = acc.add(tensor.scale(scale * phase));
            }
        }
    }
    acc
}

/// Real-space sinc^2 block coupling i to j, summed over images (|r| < r_cut),
/// skipping the n=0 image when `is_self`. GRPerY units. The n=0 self term is the
/// finite A4 limit, supplied separately by [`self_real_sinc`].
#[must_use]
pub fn real_sinc_block(rij: Vec3, ep: &EwaldParams, is_self: bool) -> Mat3 {
    let l = ep.box_l;
    let a = ep.a;
    let xi = xi_of(ep);
    let inv6a = 1.0 / (6.0 * a);
    let id = Mat3::identity();
    let mut acc = Mat3::ZERO;
    let span = (ep.r_cut / l).ceil() as i32 + 1;
    for nx in -span..=span {
        for ny in -span..=span {
            for nz in -span..=span {
                if is_self && nx == 0 && ny == 0 && nz == 0 {
                    continue;
                }
                let d = Vec3::new(
                    rij.x + f64::from(nx) * l,
                    rij.y + f64::from(ny) * l,
                    rij.z + f64::from(nz) * l,
                );
                let r = d.norm2().sqrt();
                if r > ep.r_cut || r == 0.0 {
                    continue;
                }
                let rr = Mat3::outer(d.scale(1.0 / r));
                let (f, g) = fg_scalars(r, a, xi);
                // (1/6a) [ F (I - r_hat r_hat) + G r_hat r_hat ]
                let block = id.add(rr.scale(-1.0)).scale(f).add(rr.scale(g));
                acc = acc.add(block.scale(inv6a));
            }
        }
    }
    acc
}

/// Real-space self mobility (eq A4, the `r -> 0` limit of F), GRPerY units:
/// `(1/(6a)) (1/(4 sqrt(pi) xi a)) [1 - e^{-4 a^2 xi^2} + 4 sqrt(pi) a xi erfc(2 a xi)] I`.
#[must_use]
pub fn self_real_sinc(ep: &EwaldParams) -> Mat3 {
    let a = ep.a;
    let xi = xi_of(ep);
    let sqrt_pi = PI.sqrt();
    // A4 bracket, in 6 pi eta a units (= F(r->0)).
    let a4 = (1.0 / (4.0 * sqrt_pi * xi * a))
        * (1.0 - (-4.0 * a * a * xi * xi).exp() + 4.0 * sqrt_pi * a * xi * erfc(2.0 * a * xi));
    Mat3::identity().scale(a4 / (6.0 * a))
}

/// Assemble the full periodic sinc^2 / Hasimoto grand mobility (equal radius),
/// GRPerY units, row-major `3N x 3N`. Each block is `M^(r) + M^(w)`: the
/// short-range real part ([`real_sinc_block`]) plus the screened wave part
/// ([`recip_sinc_block`]); the diagonal also carries the A4 self term
/// ([`self_real_sinc`]). Unlike [`crate::hydro::ewald::periodic_grand_mobility`]
/// there is no separate `mu0` and no k=0 backflow: the positive sinc^2 split puts
/// the entire self mobility in the real+wave decomposition, and the result is
/// independent of the splitting width `sigma` (the `xi`-invariance test).
#[must_use]
pub fn periodic_grand_mobility_sinc(pos: &[Vec3], ep: &EwaldParams) -> Vec<f64> {
    let n = pos.len();
    let dim = 3 * n;
    let mut m = vec![0.0f64; dim * dim];
    let diag = self_real_sinc(ep)
        .add(real_sinc_block(Vec3::ZERO, ep, true))
        .add(recip_sinc_block(Vec3::ZERO, ep, true));
    for i in 0..n {
        for r in 0..3 {
            for c in 0..3 {
                m[(3 * i + r) * dim + (3 * i + c)] = diag.0[3 * r + c];
            }
        }
        for j in (i + 1)..n {
            let rij = pos[i] - pos[j];
            let block = real_sinc_block(rij, ep, false).add(recip_sinc_block(rij, ep, true));
            for r in 0..3 {
                for c in 0..3 {
                    let v = block.0[3 * r + c];
                    m[(3 * i + r) * dim + (3 * j + c)] = v;
                    m[(3 * j + c) * dim + (3 * i + r)] = v;
                }
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydro::mobility::cholesky;
    use crate::vec3::Vec3;

    // Reference values from scripts/verify_sinc_rpy.py (mpmath, 40 digits),
    // sigma = 2.5 -> xi = 1/(sigma sqrt 2), a = 1. The Rust erfc is the
    // Abramowitz-Stegun approximation (|err| < 1.5e-7), so the realistic
    // agreement floor against these references is ~1e-6.
    const SIGMA: f64 = 2.5;
    const A: f64 = 1.0;

    fn xi() -> f64 {
        1.0 / (SIGMA * std::f64::consts::SQRT_2)
    }

    /// The closed-form scalars F,G must match the values produced by the
    /// independently verified mpmath integral evaluation (eqs 12,13 -> A2,A3).
    #[test]
    fn fg_scalars_match_verified_values() {
        let cases = [
            (1.3_f64, 0.227_536_695_142_518_7, 0.333_262_566_733_396_6),
            (3.5_f64, -0.038_350_554_145_461_1, 0.064_266_834_106_467_6),
        ];
        for (r, f_ref, g_ref) in cases {
            let (f, g) = fg_scalars(r, A, xi());
            eprintln!("r={r}: F={f:.15} (ref {f_ref:.15})  G={g:.15} (ref {g_ref:.15})");
            assert!((f - f_ref).abs() < 1e-6, "F(r={r}) {f} vs ref {f_ref}");
            assert!((g - g_ref).abs() < 1e-6, "G(r={r}) {g} vs ref {g_ref}");
        }
    }

    /// The real-space self block (eq A4) must match `lim_{r->0} F(r)` scaled by
    /// the GRPerY `1/(6a)` and be isotropic. mpmath A4 scalar = 0.560274206717
    /// (in 6 pi eta a units); GRPerY self = (1/6a) * A4.
    #[test]
    fn self_real_matches_a4() {
        let ep = EwaldParams {
            box_l: 10.0,
            sigma: SIGMA,
            r_cut: 13.0,
            k_max: 14,
            a: A,
        };
        let m = self_real_sinc(&ep);
        let expect = 0.560_274_206_716_730_8 / (6.0 * A);
        eprintln!("self_real diag = {:.15} (ref {expect:.15})", m.0[0]);
        assert!(
            (m.0[0] - expect).abs() < 1e-6,
            "self diag {} vs {expect}",
            m.0[0]
        );
        assert!((m.0[4] - expect).abs() < 1e-6 && (m.0[8] - expect).abs() < 1e-6);
        // off-diagonal must vanish (isotropic)
        for &k in &[1usize, 2, 3, 5, 6, 7] {
            assert!(
                m.0[k].abs() < 1e-15,
                "self off-diagonal {k} nonzero: {}",
                m.0[k]
            );
        }
    }

    /// Exact transcription consistency: for a pair (i != j), the positively-split
    /// mobility `M^(w)(screened, H) + M^(r)(closed form F,G)` must equal the
    /// complete unsplit periodic mobility (eq 8, one converged k-sum). If F or G
    /// is wrong the two routes diverge. The floor is the erfc approximation.
    #[test]
    fn sinc_split_pair_matches_unsplit() {
        let split = EwaldParams {
            box_l: 10.0,
            sigma: SIGMA,
            r_cut: 13.0,
            k_max: 14,
            a: A,
        };
        let full = EwaldParams { k_max: 40, ..split };
        for &rij in &[
            Vec3::new(3.5, 0.0, 0.0), // r > 2a
            Vec3::new(1.3, 0.4, 0.2), // r < 2a (overlap regime)
            Vec3::new(2.0, 3.0, 1.0),
        ] {
            let lhs = recip_sinc_block(rij, &split, true).add(real_sinc_block(rij, &split, false));
            let rhs = recip_sinc_block(rij, &full, false);
            let mut max_abs = 0.0f64;
            for k in 0..9 {
                max_abs = max_abs.max((lhs.0[k] - rhs.0[k]).abs());
            }
            eprintln!("rij={rij:?} split-vs-unsplit max_abs={max_abs:.3e}");
            assert!(
                max_abs < 1e-4,
                "split != unsplit for rij={rij:?}; {max_abs:.2e}"
            );
        }
    }

    /// The defining contrast with `pse_wave::grpery_wave_split_is_indefinite`:
    /// the sinc^2 wave-only grand mobility is positive-definite (Cholesky ok),
    /// because every mode is `(positive scalar)(I - k_hat k_hat)`. This is what
    /// makes a single-FFT positive-split Brownian draw possible.
    #[test]
    fn sinc_wave_split_is_positive_definite() {
        let l = 10.0;
        let ep = EwaldParams {
            box_l: l,
            sigma: SIGMA,
            r_cut: 13.0,
            k_max: 12,
            a: A,
        };
        let pos = [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(5.0, 6.0, 4.0),
            Vec3::new(8.0, 1.0, 7.0),
            Vec3::new(3.0, 8.0, 2.0),
        ];
        let n = pos.len();
        let dim = 3 * n;
        let mut m = vec![0.0f64; dim * dim];
        let diag = recip_sinc_block(Vec3::ZERO, &ep, true);
        for i in 0..n {
            for r in 0..3 {
                for c in 0..3 {
                    m[(3 * i + r) * dim + (3 * i + c)] = diag.0[3 * r + c];
                }
            }
            for j in (i + 1)..n {
                let blk = recip_sinc_block(pos[i] - pos[j], &ep, true);
                for r in 0..3 {
                    for c in 0..3 {
                        let v = blk.0[3 * r + c];
                        m[(3 * i + r) * dim + (3 * j + c)] = v;
                        m[(3 * j + c) * dim + (3 * i + r)] = v;
                    }
                }
            }
        }
        assert!(
            cholesky(&m, dim).is_ok(),
            "sinc^2 wave-only grand mobility must be SPD (positive split)"
        );
    }

    fn two_particle_box() -> Vec<Vec3> {
        vec![Vec3::new(2.0, 3.0, 4.0), Vec3::new(6.0, 5.0, 7.0)]
    }

    /// The full sinc^2 grand mobility (real + wave + A4 self) is SPD.
    #[test]
    fn sinc_full_mobility_is_spd() {
        let ep = EwaldParams {
            box_l: 10.0,
            sigma: SIGMA,
            r_cut: 13.0,
            k_max: 14,
            a: A,
        };
        let m = periodic_grand_mobility_sinc(&two_particle_box(), &ep);
        assert!(cholesky(&m, 6).is_ok(), "full sinc^2 mobility not SPD");
    }

    /// xi-invariance (the strong consistency check): the assembled mobility must
    /// not depend on the splitting width sigma, since real(sigma) + wave(sigma)
    /// reconstruct the sigma-free unsplit sum. If F,G, the wave H, or the A4 self
    /// were mutually inconsistent, the two sigma values would disagree. The floor
    /// is the erfc approximation (the real-space screen), so the tolerance is ~1e-3.
    #[test]
    fn sinc_mobility_is_independent_of_sigma() {
        let pos = two_particle_box();
        let base = EwaldParams {
            box_l: 10.0,
            sigma: 2.0,
            r_cut: 14.0,
            k_max: 16,
            a: 1.0,
        };
        let alt = EwaldParams { sigma: 3.0, ..base };
        let m1 = periodic_grand_mobility_sinc(&pos, &base);
        let m2 = periodic_grand_mobility_sinc(&pos, &alt);
        let mut max_abs = 0.0f64;
        for k in 0..36 {
            max_abs = max_abs.max((m1[k] - m2[k]).abs());
        }
        eprintln!("sinc xi-invariance max diff = {max_abs:.3e}");
        assert!(
            max_abs < 1e-3,
            "sinc^2 mobility must be sigma-invariant; {max_abs:.2e}"
        );
    }
}
