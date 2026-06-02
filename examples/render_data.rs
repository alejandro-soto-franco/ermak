//! Emit coarse-grained "molecular" geometry for visualisation.
//!
//!   cargo run --release --example render_data paths > paths.csv   # egress pathways
//!   cargo run --release --example render_data crowd > crowd.csv   # tracer in a crowded box
//!
//! `paths` writes several ligand escape trajectories out of a buried pocket
//! (the multiple-pathway dissociation picture); `crowd` writes the crowder
//! lattice plus one tracer trajectory through it.

use ermak::backend::Scenario;
use ermak::crowding::cubic_lattice;
use ermak::kinetics::escape_path;
use ermak::vec3::Vec3;

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("paths") => {
            let (barrier, r_b, d0, dt) = (3.0, 2.0, 1.0, 0.001);
            let (accel, reorient) = (6.0, 80usize);
            println!("path,x,y,z");
            for seed in 0..12u64 {
                for p in escape_path(barrier, r_b, d0, dt, accel, reorient, 200_000, 40, seed) {
                    println!("{seed},{:.4},{:.4},{:.4}", p.x, p.y, p.z);
                }
            }
            eprintln!("r_b = {r_b}");
        }
        Some("crowd") => {
            let box_l = 8.0;
            // Centre the lattice on the origin so the tracer (which starts at
            // the origin) begins *inside* the matrix rather than at a box
            // corner. An even lattice count puts the origin at the symmetric
            // centre of an empty cell, caged by eight crowders at distance
            // a*sqrt(3)/2 well outside the WCA core, so the tracer starts in an
            // interstitial void and is not ejected. Minimum-image forces are
            // translation-invariant on the torus, so this only shifts the frame.
            let half = box_l / 2.0;
            let crowders: Vec<Vec3> = cubic_lattice(box_l, 4)
                .into_iter()
                .map(|c| Vec3::new(c.x - half, c.y - half, c.z - half))
                .collect();
            let scenario = Scenario {
                d0: 1.0,
                dt: 0.0005,
                steps: 6_000,
                box_l,
                sigma: 1.0,
                eps: 1.0,
                crowders: crowders.clone(),
            };
            let path = scenario.path(7, 10);
            println!("kind,x,y,z");
            for c in &crowders {
                println!("crowder,{:.4},{:.4},{:.4}", c.x, c.y, c.z);
            }
            for p in &path {
                println!("tracer,{:.4},{:.4},{:.4}", p.x, p.y, p.z);
            }
            eprintln!("box_l = {box_l}, crowders = {}", crowders.len());
        }
        _ => eprintln!("usage: render_data paths|crowd"),
    }
}
