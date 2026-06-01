//! Backend abstraction for running a walker ensemble.
//!
//! The memory budget and batching are enforced here, in the backend-agnostic
//! driver, so every backend (the CPU reference and the GPU accelerator) streams
//! the ensemble in bounded batches and cannot over-allocate. A backend that
//! ignored the budget would be the bug the [`crate::memory`] guardrails exist to
//! prevent, so the budget is part of the trait contract, not an afterthought.

use crate::error::ErmakError;
use crate::integrator::em_step;
use crate::memory::{MemoryBudget, batch_spans};
use crate::potential::wca_pair_force;
use crate::rng::brownian_displacement;
use crate::vec3::Vec3;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// Per-walker memory footprint used to size a batch against the budget. A
/// conservative estimate of the device-side state (position + RNG counter +
/// scratch); the GPU backend allocates roughly this per walker.
pub const WALKER_BYTES: usize = 64;

/// A crowded-diffusion run: the parameters every walker shares.
#[derive(Debug, Clone)]
pub struct Scenario {
    pub d0: f64,
    pub dt: f64,
    pub steps: usize,
    pub box_l: f64,
    pub sigma: f64,
    pub eps: f64,
    pub crowders: Vec<Vec3>,
}

impl Scenario {
    /// One trajectory's final squared displacement. Deterministic in
    /// `(seed, walker)` so the result is independent of how walkers are batched.
    #[must_use]
    pub fn walk(&self, seed: u64, walker: usize) -> f64 {
        let mut rng =
            StdRng::seed_from_u64(seed ^ (walker as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut r = Vec3::ZERO;
        for _ in 0..self.steps {
            let mut force = Vec3::ZERO;
            for &c in &self.crowders {
                let d = min_image(r - c, self.box_l);
                force += wca_pair_force(d, self.sigma, self.eps);
            }
            let noise = brownian_displacement(self.d0, self.dt, &mut rng);
            r = em_step(r, force, self.d0, self.dt, noise);
        }
        r.norm2()
    }

    /// One tracer trajectory (unwrapped positions, sampled every `stride`
    /// steps), for visualising diffusion through the crowder matrix.
    #[must_use]
    pub fn path(&self, seed: u64, stride: usize) -> Vec<Vec3> {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5DEE_CE66_D5C3_1A11);
        let mut r = Vec3::ZERO;
        let mut path = vec![r];
        for step in 0..self.steps {
            let mut force = Vec3::ZERO;
            for &c in &self.crowders {
                let d = min_image(r - c, self.box_l);
                force += wca_pair_force(d, self.sigma, self.eps);
            }
            let noise = brownian_displacement(self.d0, self.dt, &mut rng);
            r = em_step(r, force, self.d0, self.dt, noise);
            if stride > 0 && step % stride == 0 {
                path.push(r);
            }
        }
        path
    }
}

/// Minimum-image displacement of `d` under a cubic box of side `l`.
pub(crate) fn min_image(d: Vec3, l: f64) -> Vec3 {
    Vec3::new(
        d.x - l * (d.x / l).round(),
        d.y - l * (d.y / l).round(),
        d.z - l * (d.z / l).round(),
    )
}

/// Runs an ensemble of independent walkers and returns the sum of their final
/// squared displacements. Implementations MUST stream within `budget`.
pub trait EnsembleBackend {
    /// # Errors
    /// [`ErmakError::MemoryBudgetExceeded`] if even one walker cannot fit the
    /// budget; backend errors otherwise.
    fn msd_sum(
        &self,
        scenario: &Scenario,
        n_walkers: usize,
        seed: u64,
        budget: &MemoryBudget,
    ) -> Result<f64, ErmakError>;
}

/// CPU reference backend: rayon over each batch, batches sized to the budget.
pub struct CpuBackend;

impl EnsembleBackend for CpuBackend {
    fn msd_sum(
        &self,
        scenario: &Scenario,
        n_walkers: usize,
        seed: u64,
        budget: &MemoryBudget,
    ) -> Result<f64, ErmakError> {
        // Guardrail: a single walker must fit the budget, then each streamed
        // batch is sized so its footprint stays under the cap.
        budget.ensure_fits(WALKER_BYTES)?;
        let batch = budget.max_items(WALKER_BYTES).max(1);
        let mut sum = 0.0;
        for (start, len) in batch_spans(n_walkers, batch) {
            sum += (start..start + len)
                .into_par_iter()
                .map(|w| scenario.walk(seed, w))
                .sum::<f64>();
        }
        Ok(sum)
    }
}

/// Effective diffusion coefficient from an ensemble: `MSD / (6 t)`.
///
/// # Errors
/// Propagates the backend's budget/backend errors.
pub fn ensemble_deff(
    scenario: &Scenario,
    n_walkers: usize,
    seed: u64,
    backend: &dyn EnsembleBackend,
    budget: &MemoryBudget,
) -> Result<f64, ErmakError> {
    let sum = backend.msd_sum(scenario, n_walkers, seed, budget)?;
    let t = scenario.steps as f64 * scenario.dt;
    Ok(sum / (n_walkers as f64 * 6.0 * t))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_scenario() -> Scenario {
        Scenario {
            d0: 1.0,
            dt: 0.01,
            steps: 50,
            box_l: 8.0,
            sigma: 1.0,
            eps: 1.0,
            crowders: vec![],
        }
    }

    #[test]
    fn batching_does_not_change_result() {
        let s = tiny_scenario();
        let cpu = CpuBackend;
        let big = MemoryBudget::new(1 << 30, "host");
        let small = MemoryBudget::new(WALKER_BYTES * 8, "host"); // batches of 8
        let one_shot = cpu.msd_sum(&s, 200, 7, &big).unwrap();
        let streamed = cpu.msd_sum(&s, 200, 7, &small).unwrap();
        assert!(
            (one_shot - streamed).abs() < 1e-9,
            "batching must not change the result: {one_shot} vs {streamed}"
        );
    }

    #[test]
    fn rejects_when_one_walker_exceeds_budget() {
        let s = tiny_scenario();
        let cpu = CpuBackend;
        let budget = MemoryBudget::new(WALKER_BYTES - 1, "host"); // cannot fit one walker
        match cpu.msd_sum(&s, 10, 7, &budget) {
            Err(ErmakError::MemoryBudgetExceeded { .. }) => {}
            other => panic!("expected budget rejection, got {other:?}"),
        }
    }
}
