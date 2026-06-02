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

## GPU

The Linux x86-64 wheel is GPU-accelerated through the CUDA driver API. It needs
no CUDA toolkit to install (the driver is loaded at runtime); `gpu_available()`
reports whether a usable device is present.

```python
import ermak

if ermak.gpu_available():
    box = 8.0
    crowders = ermak.cubic_lattice(box, 5)
    deff = ermak.crowded_diffusion_deff_gpu(
        d0=1.0, dt=2e-4, steps=10_000, replicas=200_000,
        box_l=box, crowders=crowders, sigma=1.0, eps=1.0,
        precision="f32",     # "f64" for the reference path
    )
```

`gpu_available()` and `crowded_diffusion_deff_gpu(...)` exist on every wheel; on a
build or machine without GPU support the latter raises a clear `RuntimeError`, so
calling code can branch on `gpu_available()`. Dual-licensed under MIT or
Apache-2.0.
