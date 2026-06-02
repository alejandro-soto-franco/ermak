//! Python bindings for `ermak`.
//!
//! A thin, dependency-light wrapper over the high-level entry points of the
//! Rust crate: free and crowded effective diffusion, pocket residence times and
//! the tauRAMD egress protocol, a recorded escape trajectory, and the
//! random-forest `k_off` predictor. Vectors come in and out as plain Python
//! lists and `(x, y, z)` tuples, so the module has no runtime dependency beyond
//! the interpreter.

use ermak_core::ml::{Forest as RsForest, ForestParams};
use ermak_core::vec3::Vec3;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn to_vec3(points: Vec<(f64, f64, f64)>) -> Vec<Vec3> {
    points
        .into_iter()
        .map(|(x, y, z)| Vec3::new(x, y, z))
        .collect()
}

fn from_vec3(points: Vec<Vec3>) -> Vec<(f64, f64, f64)> {
    points.into_iter().map(|v| (v.x, v.y, v.z)).collect()
}

/// Free-tracer effective diffusion coefficient, `MSD / (6 t)`, which recovers
/// `d0` in free space.
#[pyfunction]
#[pyo3(signature = (d0, dt, steps, replicas, seed = 0))]
fn free_diffusion_deff(d0: f64, dt: f64, steps: usize, replicas: usize, seed: u64) -> f64 {
    ermak_core::diffusion::free_diffusion_deff(d0, dt, steps, replicas, seed)
}

/// Effective diffusion of a tracer among fixed crowder spheres (excluded volume
/// via a Weeks-Chandler-Andersen core), under the periodic minimum image.
#[pyfunction]
#[pyo3(signature = (d0, dt, steps, replicas, box_l, crowders, sigma, eps, seed = 0))]
#[allow(clippy::too_many_arguments)]
fn crowded_diffusion_deff(
    d0: f64,
    dt: f64,
    steps: usize,
    replicas: usize,
    box_l: f64,
    crowders: Vec<(f64, f64, f64)>,
    sigma: f64,
    eps: f64,
    seed: u64,
) -> f64 {
    ermak_core::crowding::crowded_diffusion_deff(
        d0,
        dt,
        steps,
        replicas,
        seed,
        box_l,
        &to_vec3(crowders),
        sigma,
        eps,
    )
}

/// `n^3` crowder centres on a cubic lattice spanning a box of side `box_l`.
#[pyfunction]
fn cubic_lattice(box_l: f64, n: usize) -> Vec<(f64, f64, f64)> {
    from_vec3(ermak_core::crowding::cubic_lattice(box_l, n))
}

/// Crowder volume fraction `phi` for `n` spheres of diameter `sigma` in the box.
#[pyfunction]
fn volume_fraction(n_crowders: usize, sigma: f64, box_l: f64) -> f64 {
    ermak_core::crowding::volume_fraction(n_crowders, sigma, box_l)
}

/// Mean residence time (`1 / k_off`): the average first-passage time out of the
/// pocket under plain Brownian dynamics.
#[pyfunction]
#[pyo3(signature = (barrier, r_b, d0, dt, max_steps, replicas, seed = 0))]
fn mean_residence_time(
    barrier: f64,
    r_b: f64,
    d0: f64,
    dt: f64,
    max_steps: usize,
    replicas: usize,
    seed: u64,
) -> f64 {
    ermak_core::kinetics::mean_residence_time(barrier, r_b, d0, dt, max_steps, replicas, seed)
}

/// Mean tauRAMD egress time under a reoriented random-acceleration force of
/// magnitude `accel`; the egress times rank the true residence times.
#[pyfunction]
#[pyo3(signature = (barrier, r_b, d0, dt, accel, reorient_steps, max_steps, replicas, seed = 0))]
#[allow(clippy::too_many_arguments)]
fn tauramd_egress_time(
    barrier: f64,
    r_b: f64,
    d0: f64,
    dt: f64,
    accel: f64,
    reorient_steps: usize,
    max_steps: usize,
    replicas: usize,
    seed: u64,
) -> f64 {
    ermak_core::kinetics::tauramd_egress_time(
        barrier,
        r_b,
        d0,
        dt,
        accel,
        reorient_steps,
        max_steps,
        replicas,
        seed,
    )
}

/// Record one escape trajectory (positions every `stride` steps) from the pocket
/// centre until the ligand crosses the bottleneck, for visualising egress paths.
#[pyfunction]
#[pyo3(signature = (barrier, r_b, d0, dt, accel, reorient_steps, max_steps, stride, seed = 0))]
#[allow(clippy::too_many_arguments)]
fn escape_path(
    barrier: f64,
    r_b: f64,
    d0: f64,
    dt: f64,
    accel: f64,
    reorient_steps: usize,
    max_steps: usize,
    stride: usize,
    seed: u64,
) -> Vec<(f64, f64, f64)> {
    from_vec3(ermak_core::kinetics::escape_path(
        barrier,
        r_b,
        d0,
        dt,
        accel,
        reorient_steps,
        max_steps,
        stride,
        seed,
    ))
}

/// Coefficient of determination of a prediction against the truth.
#[pyfunction]
fn r2_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> f64 {
    ermak_core::ml::r2_score(&y_true, &y_pred)
}

/// Whether GPU acceleration is usable from this wheel: it was built with the
/// `gpu` feature AND a CUDA device initialises now. Always present, so callers
/// can branch on it without catching an exception.
#[pyfunction]
fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        ermak_core::gpu::GpuBackend::new().is_ok()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// GPU-accelerated crowded effective diffusion: the same observable as
/// [`crowded_diffusion_deff`], run on the device. `precision` is `"f32"`
/// (throughput, default) or `"f64"` (the correctness reference). The batch is
/// sized to `vram_fraction` of free device memory. Raises `RuntimeError` if the
/// wheel lacks GPU support or no CUDA device is present.
#[pyfunction]
#[pyo3(signature = (d0, dt, steps, replicas, box_l, crowders, sigma, eps, seed = 0, precision = "f32", vram_fraction = 0.5))]
#[allow(clippy::too_many_arguments)]
fn crowded_diffusion_deff_gpu(
    d0: f64,
    dt: f64,
    steps: usize,
    replicas: usize,
    box_l: f64,
    crowders: Vec<(f64, f64, f64)>,
    sigma: f64,
    eps: f64,
    seed: u64,
    precision: &str,
    vram_fraction: f64,
) -> PyResult<f64> {
    #[cfg(feature = "gpu")]
    {
        use ermak_core::backend::{EnsembleBackend, Scenario};
        let scenario = Scenario {
            d0,
            dt,
            steps,
            box_l,
            sigma,
            eps,
            crowders: to_vec3(crowders),
        };
        let gpu = ermak_core::gpu::GpuBackend::new()
            .map_err(|e| PyRuntimeError::new_err(format!("no CUDA device: {e}")))?;
        let budget = ermak_core::gpu::device_budget(vram_fraction)
            .map_err(|e| PyRuntimeError::new_err(format!("device budget: {e}")))?;
        let t = scenario.steps as f64 * scenario.dt;
        let sum = match precision {
            "f64" => gpu.msd_sum(&scenario, replicas, seed, &budget),
            "f32" => gpu.msd_sum_f32(&scenario, replicas, seed, &budget),
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "precision must be \"f32\" or \"f64\", got {other:?}"
                )));
            }
        }
        .map_err(|e| PyRuntimeError::new_err(format!("gpu run: {e}")))?;
        Ok(sum / (replicas as f64 * 6.0 * t))
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (
            d0,
            dt,
            steps,
            replicas,
            box_l,
            crowders,
            sigma,
            eps,
            seed,
            precision,
            vram_fraction,
        );
        Err(PyRuntimeError::new_err(
            "this ermak wheel was built without GPU support (the `gpu` feature); \
             use crowded_diffusion_deff for the CPU path",
        ))
    }
}

/// A random-forest regressor over CART trees, for predicting `log k_off` (or any
/// scalar target) from system descriptors.
#[pyclass]
struct Forest {
    inner: RsForest,
}

#[pymethods]
impl Forest {
    /// Fit a forest of `n_trees` bootstrap-resampled CART trees on rows `x` with
    /// targets `y`. `mtry = 0` considers every feature at each split.
    #[staticmethod]
    #[pyo3(signature = (x, y, n_trees = 200, max_depth = 8, min_split = 4, mtry = 0, seed = 0))]
    fn fit(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        n_trees: usize,
        max_depth: usize,
        min_split: usize,
        mtry: usize,
        seed: u64,
    ) -> Forest {
        let params = ForestParams {
            n_trees,
            max_depth,
            min_split,
            mtry,
        };
        Forest {
            inner: RsForest::fit(&x, &y, &params, seed),
        }
    }

    /// Predict the target for one descriptor row.
    fn predict(&self, x: Vec<f64>) -> f64 {
        self.inner.predict(&x)
    }

    /// Predict the target for many descriptor rows.
    fn predict_many(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        self.inner.predict_many(&x)
    }
}

#[pymodule]
fn ermak(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(free_diffusion_deff, m)?)?;
    m.add_function(wrap_pyfunction!(crowded_diffusion_deff, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(volume_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(mean_residence_time, m)?)?;
    m.add_function(wrap_pyfunction!(tauramd_egress_time, m)?)?;
    m.add_function(wrap_pyfunction!(escape_path, m)?)?;
    m.add_function(wrap_pyfunction!(r2_score, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(crowded_diffusion_deff_gpu, m)?)?;
    m.add_class::<Forest>()?;
    Ok(())
}
