//! Coupled N-body Brownian dynamics with hydrodynamic interactions (the full
//! Ermak-McCammon propagator). Unlike `crate::backend::Scenario`, which runs
//! independent single-tracer walkers, every particle here couples to every
//! other through one shared mobility matrix.

pub mod mat3;
pub mod mobility;
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
