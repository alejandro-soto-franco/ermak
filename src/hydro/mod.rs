//! Coupled N-body Brownian dynamics with hydrodynamic interactions (the full
//! Ermak-McCammon propagator). Unlike `crate::backend::Scenario`, which runs
//! independent single-tracer walkers, every particle here couples to every
//! other through one shared mobility matrix.

pub mod mat3;
pub mod mobility;
pub mod noise;
pub mod rpy;

use crate::vec3::Vec3;

/// Coupled N-body state. `box_l = None` is an unbounded (open) domain;
/// `Some(l)` is a cubic periodic box (Milestone B).
#[derive(Debug, Clone)]
pub struct HydroSystem {
    /// Particle centres.
    pub pos: Vec<Vec3>,
    /// Hydrodynamic radius per particle (polydisperse).
    pub radius: Vec<f64>,
    /// Charge per particle (for screened Coulomb; zeros = uncharged).
    pub charge: Vec<f64>,
    /// Solvent viscosity (sets self-mobility `1 / (6 pi eta a)`).
    pub eta: f64,
    /// Thermal energy `kB T`.
    pub kt: f64,
    /// Cubic box side, or `None` for open boundaries.
    pub box_l: Option<f64>,
}

impl HydroSystem {
    #[must_use]
    pub fn n(&self) -> usize {
        self.pos.len()
    }

    /// Self-mobility scalar `mu0_i = 1 / (6 pi eta a_i)` (so `D0_i = kT mu0_i`).
    #[must_use]
    pub fn self_mobility(&self, i: usize) -> f64 {
        1.0 / (6.0 * std::f64::consts::PI * self.eta * self.radius[i])
    }
}

use crate::hydro::mobility::{apply_mobility, cholesky, grand_mobility};
use crate::hydro::noise::correlated_noise;
use crate::potential::{wca_pair_force, yukawa_pair_force};
use rand::Rng;

/// Force-field parameters for one run.
#[derive(Debug, Clone, Copy)]
pub struct ForceParams {
    pub sigma: f64,
    pub eps: f64,
    pub k_e: f64,
    pub kappa: f64,
    pub cut: f64,
}

/// Pairwise conservative force on every particle: WCA excluded volume + screened
/// Coulomb. Open boundary in this milestone (no minimum image).
fn pair_forces(sys: &HydroSystem, fp: ForceParams) -> Vec<Vec3> {
    let n = sys.n();
    let mut f = vec![Vec3::ZERO; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let rij = sys.pos[i] - sys.pos[j];
            let mut fij = wca_pair_force(rij, fp.sigma, fp.eps);
            fij += yukawa_pair_force(rij, sys.charge[i], sys.charge[j], fp.k_e, fp.kappa, fp.cut);
            f[i] += fij;
            f[j] += fij.scale(-1.0);
        }
    }
    f
}

impl HydroSystem {
    /// Advance one Ermak-McCammon step. `hydro_on=false` zeroes hydrodynamic
    /// coupling (diagonal mobility only), the free-draining limit.
    pub fn step<R: Rng + ?Sized>(&mut self, dt: f64, fp: ForceParams, hydro_on: bool, rng: &mut R) {
        let n = self.n();
        let dim = 3 * n;
        let forces = pair_forces(self, fp);
        let mut m = grand_mobility(self);
        if !hydro_on {
            // zero every off-diagonal block: keep only the 3x3 self blocks
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    for r in 0..3 {
                        for c in 0..3 {
                            m[(3 * i + r) * dim + (3 * j + c)] = 0.0;
                        }
                    }
                }
            }
        }
        let drift = apply_mobility(&m, &forces);
        let l = cholesky(&m, dim).expect("mobility SPD");
        let noise = correlated_noise(&l, dim, self.kt, dt, rng);
        for i in 0..n {
            self.pos[i] += drift[i].scale(dt) + noise[i];
        }
    }
}

#[cfg(test)]
mod step_tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const FREE: ForceParams = ForceParams { sigma: 1.0, eps: 1.0, k_e: 0.0, kappa: 1.0, cut: 0.0 };

    fn one_particle(eta: f64, a: f64) -> HydroSystem {
        HydroSystem { pos: vec![Vec3::ZERO], radius: vec![a], charge: vec![0.0], eta, kt: 1.0, box_l: None }
    }

    #[test]
    fn single_particle_recovers_stokes_einstein() {
        // D_eff = MSD/(6 t) should equal D0 = kT/(6 pi eta a).
        let (eta, a) = (0.3, 1.5);
        let dt = 0.01;
        let steps = 1500usize;
        let d0 = 1.0 / (6.0 * std::f64::consts::PI * eta * a); // kT = 1
        let replicas = 4000usize;
        let mut msd = 0.0;
        for rep in 0..replicas {
            let mut sys = one_particle(eta, a);
            let mut rng = StdRng::seed_from_u64(100 + rep as u64);
            for _ in 0..steps {
                sys.step(dt, FREE, true, &mut rng);
            }
            msd += sys.pos[0].norm2();
        }
        let deff = msd / (replicas as f64 * 6.0 * steps as f64 * dt);
        let rel = (deff - d0).abs() / d0;
        assert!(rel < 0.05, "single-particle D_eff {deff:.4} vs D0 {d0:.4} (rel {rel:.3})");
    }

    #[test]
    fn hi_off_two_particles_are_independent() {
        // With hydro_on=false and no forces, each particle diffuses with its own
        // mu0; the cross block of the Cholesky factor must vanish.
        let sys = HydroSystem {
            pos: vec![Vec3::ZERO, Vec3::new(2.5, 0.0, 0.0)], radius: vec![1.0, 1.0],
            charge: vec![0.0, 0.0], eta: 1.0 / (6.0 * std::f64::consts::PI), kt: 1.0, box_l: None,
        };
        let dim = 6;
        let mut m = grand_mobility(&sys);
        for i in 0..2 { for j in 0..2 { if i != j {
            for r in 0..3 { for c in 0..3 { m[(3*i+r)*dim+(3*j+c)] = 0.0; }}
        }}}
        let l = cholesky(&m, dim).unwrap();
        for r in 0..3 { for c in 0..3 {
            assert!(l[(3*1+r)*dim + c].abs() < 1e-15, "HI-off L cross block nonzero");
        }}
    }
}
