# ermak (Python)

Python bindings for [`ermak`](https://github.com/alejandro-soto-franco/ermak), a
Brownian-dynamics engine for ligand diffusion and dissociation kinetics in
crowded environments. The core integrates the overdamped Langevin equation with
the Ermak-McCammon propagator; particles are coarse-grained spheres in implicit
solvent, in reduced Lennard-Jones units (`kB T = 1`, `sigma = 1`, bare `D0 = 1`).

```
pip install ermak
```

```python
import ermak

# Free diffusion recovers D0.
ermak.free_diffusion_deff(d0=1.0, dt=0.01, steps=200, replicas=4000)

# Crowders slow it down.
box = 8.0
crowders = ermak.cubic_lattice(box, 4)            # 4^3 obstacles
ermak.crowded_diffusion_deff(
    d0=1.0, dt=0.01, steps=200, replicas=2000,
    box_l=box, crowders=crowders, sigma=1.0, eps=1.0,
)

# Residence time (1/k_off) climbs with the pocket barrier (Kramers).
ermak.mean_residence_time(barrier=4.0, r_b=2.0, d0=1.0, dt=1e-3,
                          max_steps=80_000, replicas=400)

# tauRAMD egress times rank the true residence times.
ermak.tauramd_egress_time(barrier=2.0, r_b=2.0, d0=1.0, dt=1e-3, accel=6.0,
                          reorient_steps=100, max_steps=200_000, replicas=400)

# Predict log k_off from descriptors with a random forest.
forest = ermak.Forest.fit(rows, log_koff, n_trees=200)
forest.predict_many(rows)
```

The wheel is pure Rust with no CUDA dependency; the GPU backend lives in the Rust
crate behind an opt-in feature. Dual-licensed under MIT or Apache-2.0.
