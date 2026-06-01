//! Weeks-Chandler-Andersen (WCA) excluded-volume interaction: the purely
//! repulsive part of a Lennard-Jones potential, shifted to zero at the minimum
//! `r_c = 2^(1/6) sigma`. Models the hard core of crowders and the tracer.

use crate::vec3::Vec3;

/// WCA cutoff radius for a given `sigma`.
#[must_use]
pub fn wca_cutoff(sigma: f64) -> f64 {
    sigma * 2f64.powf(1.0 / 6.0)
}

/// WCA pair energy as a function of centre-to-centre distance `r`.
#[must_use]
pub fn wca_energy(r: f64, sigma: f64, eps: f64) -> f64 {
    if r >= wca_cutoff(sigma) || r <= 0.0 {
        return 0.0;
    }
    let sr6 = (sigma / r).powi(6);
    4.0 * eps * (sr6 * sr6 - sr6) + eps
}

/// Force on particle `i` due to `j`, where `rij = r_i - r_j`. Points along
/// `rij` (repulsive) inside the cutoff, zero outside.
#[must_use]
pub fn wca_pair_force(rij: Vec3, sigma: f64, eps: f64) -> Vec3 {
    let r2 = rij.norm2();
    let rc = wca_cutoff(sigma);
    if r2 >= rc * rc || r2 <= 0.0 {
        return Vec3::ZERO;
    }
    let sr6 = (sigma * sigma / r2).powi(3); // (sigma/r)^6
    // |F| = -dU/dr = 24 eps [2 (sigma/r)^12 - (sigma/r)^6] / r, directed along rij/r.
    // The 1/r from the magnitude and the 1/r from the unit vector combine to 1/r2.
    let coeff = 24.0 * eps * (2.0 * sr6 * sr6 - sr6) / r2;
    rij.scale(coeff)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGMA: f64 = 1.0;
    const EPS: f64 = 1.0;

    #[test]
    fn zero_beyond_cutoff() {
        let r = wca_cutoff(SIGMA) + 0.1;
        assert_eq!(wca_energy(r, SIGMA, EPS), 0.0);
        assert_eq!(
            wca_pair_force(Vec3::new(r, 0.0, 0.0), SIGMA, EPS),
            Vec3::ZERO
        );
    }

    #[test]
    fn force_matches_negative_gradient() {
        // Inside the cutoff, F = -dU/dr along the separation. Central difference.
        let r = 1.0; // < cutoff (~1.1225)
        let h = 1e-6;
        let dudr = (wca_energy(r + h, SIGMA, EPS) - wca_energy(r - h, SIGMA, EPS)) / (2.0 * h);
        let f = wca_pair_force(Vec3::new(r, 0.0, 0.0), SIGMA, EPS);
        assert!(
            (f.x - (-dudr)).abs() < 1e-4,
            "F_x should equal -dU/dr: F_x={}, -dU/dr={}",
            f.x,
            -dudr
        );
    }

    #[test]
    fn repulsive_when_overlapping() {
        // i sits at +x relative to j and they overlap: force pushes i further +x.
        let f = wca_pair_force(Vec3::new(0.8, 0.0, 0.0), SIGMA, EPS);
        assert!(
            f.x > 0.0,
            "overlap force should be repulsive (+x), got {}",
            f.x
        );
    }
}
