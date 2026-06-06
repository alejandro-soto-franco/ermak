//! On-device Ermak-McCammon step for a periodic box (feature `gpu`):
//! r' = r + M_phys F dt + sqrt(2 kT dt) M_phys^{1/2} xi, periodic-wrapped.
//! Drift uses the G1 device apply, noise the G2 Lanczos draw; conservative
//! forces (WCA + screened Coulomb) are periodic min-image on the host. The
//! mobility is in GRPerY units (M_grpery = pi eta M_phys), so the step folds in
//! inv_pi_eta = 1/(pi eta): drift = (M_grpery F) inv_pi_eta, and the noise uses
//! kT_eff = kT inv_pi_eta so its covariance is 2 kT dt M_phys.

use crate::error::ErmakError;
use crate::hydro::ForceParams;
use crate::hydro::ewald::{EwaldParams, periodic_grand_mobility};
use crate::hydro::gpu_ewald::GpuEwald;
use crate::hydro::gpu_noise::brownian_noise_gpu;
use crate::hydro::gpu_pse_wave::GpuPseWave;
use crate::potential::{wca_pair_force, yukawa_pair_force};
use crate::vec3::Vec3;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Minimum-image periodic conservative forces (WCA excluded volume + screened
/// Coulomb), cubic box side `box_l`.
#[must_use]
pub fn periodic_pair_forces(
    pos: &[Vec3],
    charge: &[f64],
    box_l: f64,
    fp: ForceParams,
) -> Vec<Vec3> {
    let n = pos.len();
    let mut f = vec![Vec3::ZERO; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut rij = pos[i] - pos[j];
            rij = Vec3::new(
                rij.x - box_l * (rij.x / box_l).round(),
                rij.y - box_l * (rij.y / box_l).round(),
                rij.z - box_l * (rij.z / box_l).round(),
            );
            let mut fij = wca_pair_force(rij, fp.sigma, fp.eps);
            fij += yukawa_pair_force(rij, charge[i], charge[j], fp.k_e, fp.kappa, fp.cut);
            f[i] += fij;
            f[j] += fij.scale(-1.0);
        }
    }
    f
}

/// Isotropic periodic self-mobility in GRPerY units. The diagonal self block of
/// the periodic grand mobility is configuration-independent (a particle with its
/// own periodic images), so a single-particle evaluation suffices.
#[must_use]
pub fn periodic_self_mobility(ep: &EwaldParams) -> f64 {
    let m = periodic_grand_mobility(&[Vec3::ZERO], ep);
    m[0] // m[0][0]; the block is mu_self * I by cubic symmetry
}

fn wrap(x: f64, l: f64) -> f64 {
    let y = x % l;
    if y < 0.0 { y + l } else { y }
}

/// Advance one Ermak-McCammon step on the GPU path (periodic box).
/// `hydro_on=false` uses only the periodic self-mobility (free-draining limit).
/// `m_iters` is the Lanczos depth for the noise (use `3 * pos.len()` for an exact
/// square root at small N).
///
/// # Errors
/// [`ErmakError::Gpu`] on any device error.
#[allow(clippy::too_many_arguments)]
pub fn em_step_hi_gpu<R: Rng + ?Sized>(
    dev: &GpuEwald,
    pos: &mut [Vec3],
    charge: &[f64],
    ep: &EwaldParams,
    fp: ForceParams,
    eta: f64,
    kt: f64,
    dt: f64,
    hydro_on: bool,
    m_iters: usize,
    rng: &mut R,
) -> Result<(), ErmakError> {
    let n = pos.len();
    let inv_pi_eta = 1.0 / (PI * eta);
    let forces = periodic_pair_forces(pos, charge, ep.box_l, fp);

    let (drift, noise): (Vec<Vec3>, Vec<Vec3>) = if hydro_on {
        let d_grpery = dev.apply_mobility_gpu(pos, &forces, ep)?;
        let drift = d_grpery.iter().map(|u| u.scale(inv_pi_eta)).collect();
        let noise = brownian_noise_gpu(dev, pos, ep, kt * inv_pi_eta, dt, m_iters, rng)?;
        (drift, noise)
    } else {
        let mu_self = periodic_self_mobility(ep) * inv_pi_eta;
        let drift = forces.iter().map(|f| f.scale(mu_self)).collect();
        let sd = (2.0 * kt * dt * mu_self).sqrt();
        let normal = Normal::new(0.0, 1.0).expect("unit normal");
        let noise = (0..n)
            .map(|_| {
                Vec3::new(
                    sd * normal.sample(rng),
                    sd * normal.sample(rng),
                    sd * normal.sample(rng),
                )
            })
            .collect();
        (drift, noise)
    };

    for i in 0..n {
        pos[i] += drift[i].scale(dt) + noise[i];
        pos[i] = Vec3::new(
            wrap(pos[i].x, ep.box_l),
            wrap(pos[i].y, ep.box_l),
            wrap(pos[i].z, ep.box_l),
        );
    }
    Ok(())
}

/// Advance one Ermak-McCammon step on the GPU PSE path (periodic box): the drift
/// uses the FFT particle-mesh mobility ([`GpuPseWave::full_apply`], O(N log N))
/// and the noise the Lanczos draw on that same apply ([`GpuPseWave::brownian_noise_pse`]),
/// in place of the dense O(N^2 k^3) lattice sum of [`em_step_hi_gpu`]. Same GRPerY
/// to physical unit folding (`inv_pi_eta`, `kt inv_pi_eta`), min-image conservative
/// forces, and periodic wrap. Always hydrodynamic (the free-draining limit is the
/// dense path's `hydro_on = false`). `ng` is the wave-grid size; `m_iters` the
/// Lanczos depth (`3 * pos.len()` for an exact root at small N).
///
/// # Errors
/// [`ErmakError::Gpu`] on any device or cuFFT error.
#[allow(clippy::too_many_arguments)]
pub fn em_step_hi_pse_gpu<R: Rng + ?Sized>(
    dev: &GpuPseWave,
    pos: &mut [Vec3],
    charge: &[f64],
    ep: &EwaldParams,
    fp: ForceParams,
    eta: f64,
    kt: f64,
    dt: f64,
    ng: usize,
    m_iters: usize,
    rng: &mut R,
) -> Result<(), ErmakError> {
    let n = pos.len();
    let inv_pi_eta = 1.0 / (PI * eta);
    let forces = periodic_pair_forces(pos, charge, ep.box_l, fp);

    let d_grpery = dev.full_apply(pos, &forces, ep, ng)?;
    let drift: Vec<Vec3> = d_grpery.iter().map(|u| u.scale(inv_pi_eta)).collect();
    let noise = dev.brownian_noise_pse(pos, ep, ng, kt * inv_pi_eta, dt, m_iters, rng)?;

    for i in 0..n {
        pos[i] += drift[i].scale(dt) + noise[i];
        pos[i] = Vec3::new(
            wrap(pos[i].x, ep.box_l),
            wrap(pos[i].y, ep.box_l),
            wrap(pos[i].z, ep.box_l),
        );
    }
    Ok(())
}

/// Advance one Ermak-McCammon step on the positively-split sinc^2 / Hasimoto PSE
/// path: the drift uses [`GpuPseWave::full_apply_sinc`] and the noise the
/// single-FFT wave draw plus the Lanczos real draw ([`GpuPseWave::brownian_noise_sinc`]),
/// in place of [`em_step_hi_pse_gpu`]'s GRPerY polynomial kernel. The sinc^2
/// mobility is in the same GRPerY units (`M_grpery = pi eta M_phys`, self
/// `1/(6a)`), so the `inv_pi_eta` / `kt inv_pi_eta` folding is identical. The
/// sinc^2 and GRPerY kernels are both valid far-field RPY; they differ only in
/// the high-k (overlap) regularization, and only sinc^2 admits the single-FFT
/// positive-split noise. `ng` is the wave-grid size; `m_iters` the Lanczos depth
/// for the SHORT-RANGE real part only (the wave noise needs no iteration).
///
/// # Errors
/// [`ErmakError::Gpu`] on any device or cuFFT error.
#[allow(clippy::too_many_arguments)]
pub fn em_step_hi_pse_sinc_gpu<R: Rng + ?Sized>(
    dev: &GpuPseWave,
    pos: &mut [Vec3],
    charge: &[f64],
    ep: &EwaldParams,
    fp: ForceParams,
    eta: f64,
    kt: f64,
    dt: f64,
    ng: usize,
    m_iters: usize,
    rng: &mut R,
) -> Result<(), ErmakError> {
    let n = pos.len();
    let inv_pi_eta = 1.0 / (PI * eta);
    let forces = periodic_pair_forces(pos, charge, ep.box_l, fp);

    let d_grpery = dev.full_apply_sinc(pos, &forces, ep, ng)?;
    let drift: Vec<Vec3> = d_grpery.iter().map(|u| u.scale(inv_pi_eta)).collect();
    let noise = dev.brownian_noise_sinc(pos, ep, ng, kt * inv_pi_eta, dt, m_iters, rng)?;

    for i in 0..n {
        pos[i] += drift[i].scale(dt) + noise[i];
        pos[i] = Vec3::new(
            wrap(pos[i].x, ep.box_l),
            wrap(pos[i].y, ep.box_l),
            wrap(pos[i].z, ep.box_l),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const FP: ForceParams = ForceParams {
        sigma: 1.0,
        eps: 1.0,
        k_e: 0.0,
        kappa: 1.0,
        cut: 0.0,
    };

    fn ep10() -> EwaldParams {
        EwaldParams {
            box_l: 10.0,
            sigma: 2.5,
            r_cut: 13.0,
            k_max: 6,
            a: 1.0,
        }
    }

    // (host) min-image: two particles whose direct separation exceeds the WCA
    // cutoff but whose nearest image is in contact must feel a force.
    #[test]
    fn periodic_force_uses_min_image() {
        let l = 10.0;
        let pos = vec![Vec3::new(0.3, 5.0, 5.0), Vec3::new(9.8, 5.0, 5.0)];
        let charge = vec![0.0, 0.0];
        let f = periodic_pair_forces(&pos, &charge, l, FP);
        // Particle 0 is at x=0.3; the nearest image of particle 1 is at
        // x = 9.8 - 10 = -0.2, a min-image separation of 0.5 (< 2^(1/6)). The
        // repulsion therefore pushes particle 0 toward +x, away from that image.
        assert!(
            f[0].norm2().sqrt() > 1e-6,
            "min-image force should be non-zero"
        );
        assert!(
            f[0].x > 0.0,
            "particle 0 should be pushed in +x, got {:?}",
            f[0]
        );
    }

    // (host) wrap: a displacement past the boundary lands back in [0, L).
    #[test]
    fn wrap_returns_into_box() {
        assert!((wrap(10.3, 10.0) - 0.3).abs() < 1e-12);
        assert!((wrap(-0.4, 10.0) - 9.6).abs() < 1e-12);
        let w = wrap(25.0, 10.0);
        assert!((0.0..10.0).contains(&w), "wrap out of range: {w}");
    }

    // (GPU) drift + units wiring: at kT=0, hydro_on, one step's displacement is
    // exactly (M_grpery F) inv_pi_eta dt.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_step_drift_matches_apply() {
        let ep = ep10();
        let eta = 0.31;
        let inv_pi_eta = 1.0 / (PI * eta);
        let dt = 0.001;
        // keep particles away from the boundary so no wrap interferes
        let mut pos = vec![Vec3::new(4.0, 5.0, 5.0), Vec3::new(5.2, 5.0, 5.0)];
        let charge = vec![0.0, 0.0];
        let start = pos.clone();
        let forces = periodic_pair_forces(&pos, &charge, ep.box_l, FP);

        let dev = GpuEwald::new().expect("cuda");
        let u = dev.apply_mobility_gpu(&pos, &forces, &ep).expect("apply");
        let mut rng = StdRng::seed_from_u64(1);
        em_step_hi_gpu(
            &dev, &mut pos, &charge, &ep, FP, eta, 0.0, dt, true, 6, &mut rng,
        )
        .expect("step");

        let mut max_abs = 0.0f64;
        for i in 0..pos.len() {
            let expect = start[i] + u[i].scale(inv_pi_eta * dt);
            for (a, b) in [
                (pos[i].x, expect.x),
                (pos[i].y, expect.y),
                (pos[i].z, expect.z),
            ] {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        eprintln!("step-drift vs apply: max_abs={max_abs:.3e}");
        assert!(
            max_abs < 1e-12,
            "drift step must equal M_phys F dt; {max_abs:.3e}"
        );
    }

    // (GPU) PSE step drift + units: at kT=0, one PSE step's displacement equals
    // (full_apply(F)) inv_pi_eta dt, i.e. the FFT mobility wired with the GRPerY ->
    // physical conversion. Validates the wiring; the PSE apply and noise are pinned
    // in gpu_pse_wave.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_pse_step_drift_matches_apply() {
        use crate::hydro::gpu_pse_wave::GpuPseWave;
        let ep = ep10();
        let eta = 0.31;
        let inv_pi_eta = 1.0 / (PI * eta);
        let dt = 0.001;
        let ng = 24usize;
        let mut pos = vec![Vec3::new(4.0, 5.0, 5.0), Vec3::new(5.2, 5.0, 5.0)];
        let charge = vec![0.0, 0.0];
        let start = pos.clone();
        let forces = periodic_pair_forces(&pos, &charge, ep.box_l, FP);

        let dev = GpuPseWave::new().expect("cuda");
        let u = dev.full_apply(&pos, &forces, &ep, ng).expect("full_apply");
        let mut rng = StdRng::seed_from_u64(1);
        em_step_hi_pse_gpu(
            &dev, &mut pos, &charge, &ep, FP, eta, 0.0, dt, ng, 6, &mut rng,
        )
        .expect("pse step");

        let mut max_abs = 0.0f64;
        for i in 0..pos.len() {
            let expect = start[i] + u[i].scale(inv_pi_eta * dt);
            for (a, b) in [
                (pos[i].x, expect.x),
                (pos[i].y, expect.y),
                (pos[i].z, expect.z),
            ] {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        eprintln!("pse step-drift vs full_apply: max_abs={max_abs:.3e}");
        assert!(
            max_abs < 1e-12,
            "PSE drift step must equal M_phys F dt; {max_abs:.3e}"
        );
    }

    // cargo test --features gpu -- --ignored gpu_pse_sinc_step_drift --nocapture
    // At kT=0 the sinc^2 step is pure drift: pos must advance by M_phys F dt with
    // M_phys = full_apply_sinc * inv_pi_eta (bit-exact, no noise).
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_pse_sinc_step_drift_matches_apply() {
        use crate::hydro::gpu_pse_wave::GpuPseWave;
        let ep = ep10();
        let eta = 0.31;
        let inv_pi_eta = 1.0 / (PI * eta);
        let dt = 0.001;
        let ng = 24usize;
        let mut pos = vec![Vec3::new(4.0, 5.0, 5.0), Vec3::new(5.2, 5.0, 5.0)];
        let charge = vec![0.0, 0.0];
        let start = pos.clone();
        let forces = periodic_pair_forces(&pos, &charge, ep.box_l, FP);

        let dev = GpuPseWave::new().expect("cuda");
        let u = dev
            .full_apply_sinc(&pos, &forces, &ep, ng)
            .expect("full_apply_sinc");
        let mut rng = StdRng::seed_from_u64(1);
        em_step_hi_pse_sinc_gpu(
            &dev, &mut pos, &charge, &ep, FP, eta, 0.0, dt, ng, 6, &mut rng,
        )
        .expect("pse sinc step");

        let mut max_abs = 0.0f64;
        for i in 0..pos.len() {
            let expect = start[i] + u[i].scale(inv_pi_eta * dt);
            for (a, b) in [
                (pos[i].x, expect.x),
                (pos[i].y, expect.y),
                (pos[i].z, expect.z),
            ] {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        eprintln!("pse sinc step-drift vs full_apply_sinc: max_abs={max_abs:.3e}");
        assert!(
            max_abs < 1e-12,
            "sinc^2 PSE drift step must equal M_phys F dt; {max_abs:.3e}"
        );
    }

    // (GPU) HI-off decouples: at kT=0, hydro_on=false, a force on a charged pair
    // moves each by mu_self_phys f_i dt, with no cross coupling.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_step_hi_off_decouples() {
        let ep = ep10();
        let eta = 0.31;
        let inv_pi_eta = 1.0 / (PI * eta);
        let dt = 0.01;
        let mut pos = vec![Vec3::new(4.0, 5.0, 5.0), Vec3::new(5.5, 5.0, 5.0)];
        let charge = vec![1.0, 1.0];
        let fp = ForceParams {
            k_e: 1.0,
            cut: 4.0,
            ..FP
        };
        let start = pos.clone();
        let forces = periodic_pair_forces(&pos, &charge, ep.box_l, fp);
        let mu_self = periodic_self_mobility(&ep) * inv_pi_eta;

        let dev = GpuEwald::new().expect("cuda");
        let mut rng = StdRng::seed_from_u64(2);
        em_step_hi_gpu(
            &dev, &mut pos, &charge, &ep, fp, eta, 0.0, dt, false, 6, &mut rng,
        )
        .expect("step");

        // free-draining: dr_i = mu_self_phys f_i dt, no cross coupling
        for i in 0..2 {
            let expect = start[i] + forces[i].scale(mu_self * dt);
            let d = (pos[i] - expect).norm2().sqrt();
            assert!(d < 1e-12, "HI-off particle {i} off by {d:.3e}");
        }
    }

    // (GPU) Stokes-Einstein on the device step: a single particle with no force
    // diffuses with the periodic self-mobility. One step's mean squared
    // displacement is 6 kT dt mu_self_phys.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_step_stokes_einstein_self_mobility() {
        let ep = ep10();
        let eta = 0.31;
        let inv_pi_eta = 1.0 / (PI * eta);
        let (kt, dt) = (1.2_f64, 0.005_f64);
        let mu_self = periodic_self_mobility(&ep) * inv_pi_eta;
        let target = 6.0 * kt * dt * mu_self;

        let dev = GpuEwald::new().expect("cuda");
        let mut rng = StdRng::seed_from_u64(3);
        let charge = vec![0.0];
        let samples = 8000usize;
        let mut msd = 0.0;
        for _ in 0..samples {
            let mut pos = vec![Vec3::new(5.0, 5.0, 5.0)];
            em_step_hi_gpu(
                &dev, &mut pos, &charge, &ep, FP, eta, kt, dt, true, 3, &mut rng,
            )
            .expect("step");
            let mut d = pos[0] - Vec3::new(5.0, 5.0, 5.0);
            d = Vec3::new(
                d.x - ep.box_l * (d.x / ep.box_l).round(),
                d.y - ep.box_l * (d.y / ep.box_l).round(),
                d.z - ep.box_l * (d.z / ep.box_l).round(),
            );
            msd += d.norm2();
        }
        let est = msd / samples as f64;
        let rel = (est - target).abs() / target;
        eprintln!("SE step MSD est={est:.5} target={target:.5} rel={rel:.3}");
        assert!(rel < 0.08, "step MSD vs 6 kT dt mu_self off by {rel:.3}");
    }
}
