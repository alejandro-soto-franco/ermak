//! Ermak-McCammon overdamped Langevin (Brownian dynamics) propagator.
//!
//! For a free-draining particle with isotropic diffusion `D`, one step is
//! `r' = r + (D / kB T) F dt + R`, where the random displacement `R` has
//! independent Gaussian components of variance `2 D dt`. The deterministic
//! drift and the random kick are separated so the step is a pure, reproducible
//! function: the caller draws `noise` and passes it in.

use crate::vec3::Vec3;

/// One Ermak-McCammon step. `mobility` is `D / kB T`; `noise` is the random
/// displacement the caller drew for this step (see [`crate::rng`] once added).
#[must_use]
pub fn em_step(r: Vec3, force: Vec3, mobility: f64, dt: f64, noise: Vec3) -> Vec3 {
    r + force.scale(mobility * dt) + noise
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_only_step_is_mobility_force_dt() {
        // With zero noise the step is pure drift: r' = r + mobility * force * dt.
        let r = Vec3::new(1.0, 2.0, 3.0);
        let force = Vec3::new(0.0, 0.0, -4.0);
        let mobility = 0.5; // D / kB T
        let dt = 0.1;
        let r1 = em_step(r, force, mobility, dt, Vec3::ZERO);
        assert!(
            (r1.z - 2.8).abs() < 1e-12,
            "z drift: 3 + 0.5*-4*0.1 = 2.8, got {}",
            r1.z
        );
        assert!((r1.x - 1.0).abs() < 1e-12);
        assert!((r1.y - 2.0).abs() < 1e-12);
    }
}
