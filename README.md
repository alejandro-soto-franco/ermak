# ermak

Brownian dynamics of ligand diffusion and binding kinetics in crowded
environments, in Rust.

`ermak` integrates the overdamped Langevin equation with the Ermak-McCammon
(1978) propagator and uses it to measure, on one engine, the three quantities a
binding-kinetics study cares about: how molecular crowding slows diffusion, how
fast a ligand finds its target, and how long it stays bound. Particles are
coarse-grained spheres in implicit solvent, so the simulations run in seconds on
a laptop while still reproducing the right analytical and experimental limits.

Named for the Ermak-McCammon Brownian-dynamics algorithm.

## Result: crowders slow tracer diffusion

A Brownian tracer diffusing among fixed crowder spheres (excluded volume via a
Weeks-Chandler-Andersen potential) in a periodic box. As the crowder volume
fraction grows, the effective diffusion coefficient falls toward zero, the
qualitative result of Dey et al. 2022 on crowder-slowed small-molecule
diffusion, with a percolation-like arrest as the obstacle matrix closes the
channels.

![Effective diffusion vs crowder volume fraction](docs/crowding.png)

```
cargo run --release --example crowding_sweep > crowding.csv
python scripts/plot_crowding.py crowding.csv
```

## Validation

Every physical claim is pinned to a closed-form limit as a test:

| Limit | Check | Module |
| --- | --- | --- |
| Free diffusion | `MSD = 6 D_0 t`, so estimated `D_eff` equals `D_0` | `diffusion` |
| Fluctuation-dissipation | random step variance is `2 D dt` per axis | `rng` |
| Force consistency | WCA `force == -grad energy` (central difference) | `potential` |
| Crowding | `D_eff` decreases monotonically with volume fraction | `crowding` |

```
cargo test
```

## Design

- `integrator` : the Ermak-McCammon step as a pure, reproducible function
  (`r' = r + (D / kB T) F dt + R`); the caller supplies the random kick.
- `potential`  : `Wca` excluded volume (`force == -grad energy`, tested).
- `rng`        : Gaussian Brownian displacements, `R ~ N(0, 2 D dt)` per axis.
- `diffusion`  : free-tracer `D_eff` from the ensemble MSD (the `D_0` baseline).
- `crowding`   : tracer among fixed crowders, periodic minimum image, `D_eff(phi)`.

Replicas are independent and independently seeded, so ensembles are
embarrassingly parallel (`rayon`) and reproducible for a fixed seed.

Reduced Lennard-Jones units throughout (`kB T = 1`, `sigma = 1`, bare `D_0 = 1`).

## Roadmap

This is phase one of three (see the `ermak-planning` repo for the design spec):

1. **CPU engine + validation** (this release): crowded-environment diffusion,
   analytical-limit tests.
2. **GPU backend**: a feature-gated `cuda-oxide` ensemble propagator (one walker
   per thread); the CPU backend stays the correctness reference.
3. **Binding kinetics + ML**: association rate (Smoluchowski limit), residence
   time / dissociation rate (Kramers limit) with a tauRAMD-style egress
   protocol, and an ML layer predicting `k_off` from system descriptors.

Phase one holds the crowders fixed (a quenched obstacle matrix) and uses
free-draining, isotropic diffusion; mobile crowders and hydrodynamic
interactions (Rotne-Prager-Yamakawa) are the planned extensions.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
