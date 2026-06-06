//! Pair translational mobility versus separation: the GRPerY (Beenakker) and the
//! sinc^2 / Hasimoto periodic kernels against the analytic free-space
//! Rotne-Prager-Yamakawa tensor, evaluated in a large box so the periodic
//! corrections are negligible. CPU only. Emits CSV to stdout:
//!   r_over_a, grpery_par, grpery_perp, sinc_par, sinc_perp, rpy_par, rpy_perp
//! with every mobility in units of the self mobility mu0 = 1/(6a). The off-diagonal
//! block for a pair separated along z has parallel component (zz) and perpendicular
//! component (xx). Run: cargo run --release --example hi_pair_mobility

use ermak::hydro::ewald::{EwaldParams, periodic_grand_mobility};
use ermak::hydro::pse_wave_hasimoto::periodic_grand_mobility_sinc;
use ermak::vec3::Vec3;

/// (perpendicular = xx, parallel = zz) of the 0,1 off-diagonal block, row-major.
fn off_block(m: &[f64], dim: usize) -> (f64, f64) {
    (m[3], m[2 * dim + 5])
}

fn main() {
    let a = 1.0_f64;
    let l = 400.0_f64; // large box: the periodic finite-size shift (~2.837 a/L) is sub-percent
    let mu0 = 1.0 / (6.0 * a);
    let ep = EwaldParams {
        box_l: l,
        sigma: 8.0,
        r_cut: 50.0,
        k_max: 28,
        a,
    };
    println!("r_over_a,grpery_par,grpery_perp,sinc_par,sinc_perp,rpy_par,rpy_perp");
    let mut r = 2.0_f64;
    while r <= 12.0 + 1e-9 {
        let pos = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, r)];
        let mg = periodic_grand_mobility(&pos, &ep);
        let ms = periodic_grand_mobility_sinc(&pos, &ep);
        let (gp_perp, gp_par) = off_block(&mg, 6);
        let (sp_perp, sp_par) = off_block(&ms, 6);
        // analytic free-space RPY (GRPerY units), r >= 2a
        let rpy_par = (3.0 * a / (2.0 * r) - a * a * a / (r * r * r)) / (6.0 * a);
        let rpy_perp = (3.0 * a / (4.0 * r) + a * a * a / (2.0 * r * r * r)) / (6.0 * a);
        println!(
            "{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            r / a,
            gp_par / mu0,
            gp_perp / mu0,
            sp_par / mu0,
            sp_perp / mu0,
            rpy_par / mu0,
            rpy_perp / mu0,
        );
        r += 0.5;
    }
}
