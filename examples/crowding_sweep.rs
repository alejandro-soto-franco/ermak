//! Sweep crowder volume fraction and print the effective diffusion coefficient
//! as CSV. Reproduces, qualitatively, the crowder-slowed small-molecule
//! diffusion of Dey et al. 2022.
//!
//!   cargo run --release --example crowding_sweep > crowding.csv
//!   python scripts/plot_crowding.py crowding.csv
//!
//! Reduced Lennard-Jones units (kB T = 1, sigma = 1, bare D_0 = 1).

use ermak::crowding::{crowded_diffusion_deff, cubic_lattice, volume_fraction};

fn main() {
    let d0 = 1.0;
    let dt = 0.0002;
    let steps = 8_000;
    let replicas = 300;
    let seed = 7;
    let box_l = 8.0;
    let sigma = 1.0;
    let eps = 1.0;

    // Free baseline (no crowders); normalise the curve by this to cancel bias.
    let d_free = crowded_diffusion_deff(d0, dt, steps, replicas, seed, box_l, &[], sigma, eps);

    println!("phi,n_crowders,d_eff,d_eff_over_d0");
    println!("0.0000,0,{d_free:.5},1.00000");

    for n in [2usize, 3, 4, 5, 6] {
        let crowders = cubic_lattice(box_l, n);
        let phi = volume_fraction(crowders.len(), sigma, box_l);
        let deff =
            crowded_diffusion_deff(d0, dt, steps, replicas, seed, box_l, &crowders, sigma, eps);
        println!("{phi:.4},{},{deff:.5},{:.5}", crowders.len(), deff / d_free);
    }
}
