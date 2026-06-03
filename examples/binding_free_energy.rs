//! Binding thermodynamics: the configurational binding free energy from the
//! pocket well. Given a well of depth `epsilon` over the bottleneck radius
//! `r_b`, ermak evaluates the Boltzmann configurational integral
//!
//!   dG = -kT ln( (1/V0) integral exp(-U_well(r)/kT) 4 pi r^2 dr ),
//!
//! so a deeper or wider well binds more tightly (more negative dG), and a
//! vanishing well returns the ideal reference. This is the thermodynamic
//! companion to the dissociation kinetics in `ligand_escape`: one pocket, both
//! how fast a ligand leaves and how tightly it is held.
//!
//!   cargo run --release --example binding_free_energy > binding.csv
//!   python scripts/plot_binding.py binding.csv
//!
//! kT = 0.593 kcal/mol (T = 298 K), so dG reads in kcal/mol.

use ermak::potential::binding_free_energy;

fn main() {
    let kt = 0.593; // kcal/mol at 298 K
    let v0 = 1.0; // standard-state reference volume
    let n = 4000; // integration nodes

    println!("well_depth,rb_1.0,rb_1.5,rb_2.0");
    let mut depth = 0.0;
    while depth <= 10.0 + 1e-9 {
        let g_narrow = binding_free_energy(1.0, depth, kt, v0, n);
        let g_mid = binding_free_energy(1.5, depth, kt, v0, n);
        let g_wide = binding_free_energy(2.0, depth, kt, v0, n);
        println!("{depth:.2},{g_narrow:.4},{g_mid:.4},{g_wide:.4}");
        depth += 0.5;
    }
}
