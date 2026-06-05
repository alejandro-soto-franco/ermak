//! Two-sphere hydrodynamic signature: the cross-mobility coupling that the new
//! RPY engine adds and that the previous free-draining ermak lacks.
//!
//! For two equal spheres (radius a) separated by d along x, the RPY cross block
//! couples their motions. In units of the self-mobility mu0 = 1/(6 pi eta a),
//! the analytic far-field values (r = d >= 2a) are
//!   longitudinal (along the line of centres): mu_cross_par / mu0 = 3a/(2d) - a^3/d^3
//!   transverse  (perpendicular):              mu_cross_perp/ mu0 = 3a/(4d) + a^3/(2d^3)
//! Free-draining (old ermak) has mu_cross = 0 at every separation.
//!
//! Physical reading: with HI, two particles' motions are CORRELATED. The
//! relative (approach) diffusion D_rel,par/(2 D0) = 1 - mu_cross_par/mu0 is
//! SLOWED by HI; the collective drift is enhanced. None of this exists without HI.
//!
//! RPY is the far-field approximation: it is accurate for d >~ 3a but, lacking
//! near-contact lubrication, it underestimates the relative-motion resistance as
//! d -> 2a (the exact two-sphere functions of Jeffrey & Onishi / Kim & Karrila
//! diverge there). The table flags that regime.
//!
//! Run: cargo run --release --example hi_two_sphere

use ermak::hydro::rpy::rpy_pair_equal;
use ermak::vec3::Vec3;

fn main() {
    let a = 1.0_f64;
    let eta = 1.0_f64;
    let mu0 = 1.0 / (6.0 * std::f64::consts::PI * eta * a);

    println!("# two-sphere RPY hydrodynamic signature (mu0 units), a = {a}");
    println!(
        "# d/a  par_engine  par_analytic  perp_engine  perp_analytic  Drel_par/2D0(HI)  Drel_par/2D0(free)  regime"
    );
    for &doa in &[2.0_f64, 2.5, 3.0, 4.0, 6.0, 10.0] {
        let d = doa * a;
        let block = rpy_pair_equal(Vec3::new(d, 0.0, 0.0), a, eta);
        // separation along x: block[0,0] is longitudinal, block[1,1] transverse
        let par = block.0[0] / mu0;
        let perp = block.0[4] / mu0;
        let par_an = 1.5 * a / d - a.powi(3) / d.powi(3);
        let perp_an = 0.75 * a / d + 0.5 * a.powi(3) / d.powi(3);
        let drel_hi = 1.0 - par; // relative approach diffusion, normalized
        let drel_free = 1.0; // free-draining: no coupling
        let regime = if doa < 2.5 {
            "near-contact: RPY misses lubrication"
        } else {
            "far-field: RPY reliable"
        };
        println!(
            "{doa:4.1}  {par:10.5}  {par_an:12.5}  {perp:11.5}  {perp_an:13.5}  {drel_hi:16.5}  {drel_free:18.5}  {regime}"
        );
    }
    println!();
    println!("# free-draining (previous ermak) cross-mobility is identically 0 at all d:");
    println!("# the entire 'par_engine'/'perp_engine' column is the NEW physics HI adds.");
}
