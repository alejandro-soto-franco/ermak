//! Predict log k_off from system descriptors with a random forest.
//!
//! The kinetics engine labels a grid of systems (bottleneck barrier, pocket
//! radius, ligand diffusivity) with their residence times; the forest learns
//! log k_off = -log(residence) from the three descriptors and reports held-out
//! R^2 and permutation importance. This is the predict-dissociation-rates-with-
//! ML thread of Nunes-Alves's work, in miniature: the simulator is the data
//! generator, and the model predicts k_off at high R^2 while permutation
//! importance ranks the descriptors. In this coarse-grained, diffusion-dominated
//! regime the ligand diffusivity and pocket size lead and the shallow bottleneck
//! barrier follows; a deeper barrier would promote it, as Kramers implies.
//!
//!   cargo run --release --example koff_prediction > koff_parity.csv
//!   python scripts/plot_koff.py koff_parity.csv
//!
//! Reduced units (kB T = 1).

use ermak::kinetics::mean_residence_time;
use ermak::ml::{Forest, ForestParams, permutation_importance, r2_score, train_test_split};

fn main() {
    let (dt, max_steps, reps) = (0.001, 80_000usize, 120usize);
    let barriers = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
    let radii = [1.5, 2.0, 2.5];
    let diffusivities = [0.5, 1.0, 2.0];

    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    let mut seed = 1u64;
    for &barrier in &barriers {
        for &r_b in &radii {
            for &d0 in &diffusivities {
                let residence = mean_residence_time(barrier, r_b, d0, dt, max_steps, reps, seed);
                seed += 1;
                x.push(vec![barrier, r_b, d0]);
                y.push(-residence.ln()); // log k_off
            }
        }
    }

    let (train, test) = train_test_split(x.len(), 0.3, 42);
    let pick = |idx: &[usize]| -> (Vec<Vec<f64>>, Vec<f64>) {
        (
            idx.iter().map(|&i| x[i].clone()).collect(),
            idx.iter().map(|&i| y[i]).collect(),
        )
    };
    let (xtr, ytr) = pick(&train);
    let (xte, yte) = pick(&test);

    let forest = Forest::fit(&xtr, &ytr, &ForestParams::default(), 7);
    let pred = forest.predict_many(&xte);
    let r2 = r2_score(&yte, &pred);
    let imp = permutation_importance(&forest, &xte, &yte, 11);

    eprintln!("n = {} ({} train / {} test)", x.len(), xtr.len(), xte.len());
    eprintln!("held-out R^2 = {r2:.3}");
    eprintln!(
        "permutation importance: barrier = {:.3}, r_b = {:.3}, D_0 = {:.3}",
        imp[0], imp[1], imp[2]
    );

    println!("true_log_koff,pred_log_koff");
    for (t, p) in yte.iter().zip(pred.iter()) {
        println!("{t:.4},{p:.4}");
    }
}
