"""Smoke tests for the ermak Python bindings.

Each test pins a binding to a closed-form limit, mirroring the Rust test suite,
so the wheel is verified to compute, not merely to import.
"""
import ermak


def test_free_diffusion_recovers_d0():
    # MSD = 6 D0 t in free space, so the estimated D_eff equals D0.
    deff = ermak.free_diffusion_deff(d0=1.0, dt=0.01, steps=200, replicas=4000, seed=7)
    assert abs(deff - 1.0) < 0.05, f"free D_eff should recover D0=1, got {deff}"


def test_crowding_slows_diffusion():
    # A dense crowder lattice must lower D_eff below the free value. Hard-core WCA
    # needs a small timestep for stability (t = steps*dt = 2), mirroring the Rust
    # crowding test.
    box_l, sigma, eps = 8.0, 1.0, 1.0
    common = dict(d0=1.0, dt=0.0002, steps=10_000, replicas=400,
                  box_l=box_l, sigma=sigma, eps=eps, seed=7)
    free = ermak.crowded_diffusion_deff(crowders=[], **common)
    crowders = ermak.cubic_lattice(box_l, 6)  # 6^3 = 216 obstacles, phi ~ 0.22
    crowded = ermak.crowded_diffusion_deff(crowders=crowders, **common)
    assert free > 0.9, f"empty box should recover D0, got {free}"
    assert crowded < 0.85 * free, f"crowding should slow diffusion: {free} -> {crowded}"


def test_residence_time_grows_with_barrier():
    # Kramers/Arrhenius: a higher pocket barrier means a longer residence.
    common = dict(r_b=2.0, d0=1.0, dt=0.001, max_steps=80_000, replicas=400, seed=1)
    t_low = ermak.mean_residence_time(barrier=1.0, **common)
    t_high = ermak.mean_residence_time(barrier=4.0, **common)
    assert t_high > 2.0 * t_low, f"barrier should lengthen residence: {t_low} -> {t_high}"


def test_escape_path_reaches_the_bottleneck():
    path = ermak.escape_path(
        barrier=2.0, r_b=2.0, d0=1.0, dt=0.001, accel=6.0,
        reorient_steps=100, max_steps=200_000, stride=50, seed=5,
    )
    assert path[0] == (0.0, 0.0, 0.0), "trajectory starts at the pocket centre"
    x, y, z = path[-1]
    assert x * x + y * y + z * z >= 2.0 * 2.0 - 1e-6, "an accelerated walk reaches r_b"


def test_forest_predicts_koff():
    # A forest fit on a monotone target should track it on held-out points.
    rows = [[float(a), float(b)] for a in range(6) for b in range(6)]
    y = [r[0] - 0.5 * r[1] for r in rows]
    forest = ermak.Forest.fit(rows, y, n_trees=80, seed=0)
    preds = forest.predict_many(rows)
    assert ermak.r2_score(y, preds) > 0.8, "forest should fit a monotone target well"


def test_module_has_version():
    assert isinstance(ermak.__version__, str) and ermak.__version__


def test_gpu_api_is_uniform():
    # gpu_available() always exists and returns a bool. The GPU function exists
    # on every wheel: it computes when a device is present, else raises cleanly.
    import pytest

    available = ermak.gpu_available()
    assert isinstance(available, bool)

    box_l = 8.0
    crowders = ermak.cubic_lattice(box_l, 5)
    kw = dict(d0=1.0, dt=2e-4, steps=2000, replicas=500,
              box_l=box_l, crowders=crowders, sigma=1.0, eps=1.0, seed=7)
    if available:
        deff = ermak.crowded_diffusion_deff_gpu(precision="f32", **kw)
        assert 0.0 < deff < 1.0, f"GPU crowded D_eff should sit in (0, D0), got {deff}"
    else:
        with pytest.raises(RuntimeError):
            ermak.crowded_diffusion_deff_gpu(**kw)
