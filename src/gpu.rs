//! GPU backend (feature `gpu`) via the CUDA driver API (cudarc).
//!
//! Guardrails first: this GPU has only ~8 GiB of VRAM, so a GPU batch is sized
//! to a fraction of *free* VRAM (queried from `nvidia-smi`) and the ensemble is
//! streamed in bounded batches, exactly like the CPU backend, so the device can
//! never be over-committed.
//!
//! The kernel runs one walker per thread: the full Brownian-dynamics trajectory
//! (crowder forces under the minimum image, drift, and a Gaussian kick from a
//! per-walker xoshiro256++ Box-Muller stream), writing each walker's final
//! squared displacement. The CPU backend stays the correctness reference; the
//! GPU reproduces its `D_eff` within a statistical tolerance (the RNG differs,
//! so the match is statistical, not bit-exact).

use crate::backend::{EnsembleBackend, Scenario, WALKER_BYTES};
use crate::error::ErmakError;
use crate::memory::{MemoryBudget, batch_spans};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::process::Command;
use std::sync::Arc;

/// Parse free VRAM in MiB from `nvidia-smi` output (the first integer found,
/// tolerating units and trailing whitespace).
#[must_use]
pub fn parse_free_vram_mib(out: &str) -> Option<usize> {
    out.split(|c: char| !c.is_ascii_digit())
        .find(|s| !s.is_empty())
        .and_then(|s| s.parse().ok())
}

/// Query free VRAM via `nvidia-smi`.
///
/// # Errors
/// [`ErmakError::Gpu`] if `nvidia-smi` is missing or its output is unparseable.
pub fn free_vram_bytes() -> Result<usize, ErmakError> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|e| ErmakError::Gpu(format!("nvidia-smi failed to launch: {e}")))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mib = parse_free_vram_mib(&text)
        .ok_or_else(|| ErmakError::Gpu(format!("could not parse free VRAM from: {text:?}")))?;
    Ok(mib * 1024 * 1024)
}

/// A device-memory budget capped at `fraction` of free VRAM, so a GPU batch can
/// never claim all of the 8 GiB device. `fraction` is clamped to `(0, 1]`.
///
/// # Errors
/// Propagates [`free_vram_bytes`] errors.
pub fn device_budget(fraction: f64) -> Result<MemoryBudget, ErmakError> {
    let frac = fraction.clamp(f64::MIN_POSITIVE, 1.0);
    let free = free_vram_bytes()?;
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let cap = (free as f64 * frac) as usize;
    Ok(MemoryBudget::new(cap, "device VRAM"))
}

fn driver_err(e: DriverError) -> ErmakError {
    ErmakError::Gpu(format!("cuda driver: {e:?}"))
}

/// CUDA C kernel: one walker per thread runs the full BD trajectory.
const KERNEL_SRC: &str = r#"
extern "C" {

// splitmix64: used only to seed the per-walker xoshiro state (the recommended
// pattern). Drawing bulk randoms directly from splitmix leaves consecutive
// outputs slightly correlated, which biases Box-Muller; xoshiro256++ does not.
__device__ __forceinline__ unsigned long long sm64(unsigned long long *s) {
    unsigned long long z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ unsigned long long rotl(unsigned long long x, int k) {
    return (x << k) | (x >> (64 - k));
}

// xoshiro256++ : high-quality bulk PRNG, state s[4].
__device__ __forceinline__ unsigned long long xnext(unsigned long long *s) {
    unsigned long long result = rotl(s[0] + s[3], 23) + s[0];
    unsigned long long t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

__device__ __forceinline__ double u01(unsigned long long *s) {
    // 53-bit uniform in (0, 1), never exactly 0 (so log is finite).
    return ((double)(xnext(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

__global__ void bd_walk(
    double *out_msd, const double *crowders, int n_crowders, int n_walkers,
    int steps, double d0, double dt, double box_l, double sigma, double eps,
    unsigned long long seed, unsigned int walker_offset)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_walkers) return;

    // Seed a per-walker xoshiro256++ from splitmix (decorrelates and fixes
    // low-index seeding artifacts).
    unsigned long long ss =
        seed ^ ((unsigned long long)(walker_offset + (unsigned int)w) * 0x9E3779B97F4A7C15ULL);
    unsigned long long s[4];
    s[0] = sm64(&ss);
    s[1] = sm64(&ss);
    s[2] = sm64(&ss);
    s[3] = sm64(&ss);

    double rx = 0.0, ry = 0.0, rz = 0.0;
    double rc = sigma * 1.1224620483093730; // 2^(1/6) sigma
    double rc2 = rc * rc;
    double sigma2 = sigma * sigma;
    double sdev = sqrt(2.0 * d0 * dt);
    const double TWO_PI = 6.283185307179586;

    for (int step = 0; step < steps; step++) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (int c = 0; c < n_crowders; c++) {
            double dx = rx - crowders[3 * c + 0];
            double dy = ry - crowders[3 * c + 1];
            double dz = rz - crowders[3 * c + 2];
            dx -= box_l * rint(dx / box_l); // minimum image
            dy -= box_l * rint(dy / box_l);
            dz -= box_l * rint(dz / box_l);
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < rc2 && r2 > 0.0) {
                double sr2 = sigma2 / r2;
                double sr6 = sr2 * sr2 * sr2;
                double coeff = 24.0 * eps * (2.0 * sr6 * sr6 - sr6) / r2;
                fx += coeff * dx;
                fy += coeff * dy;
                fz += coeff * dz;
            }
        }
        // Three N(0, 2 D dt) increments via Box-Muller.
        double u1 = u01(s), u2 = u01(s);
        double rr = sqrt(-2.0 * log(u1));
        double n0 = sdev * rr * cos(TWO_PI * u2);
        double n1 = sdev * rr * sin(TWO_PI * u2);
        double u3 = u01(s), u4 = u01(s);
        double n2 = sdev * sqrt(-2.0 * log(u3)) * cos(TWO_PI * u4);

        rx += d0 * fx * dt + n0; // mobility = d0 (kB T = 1)
        ry += d0 * fy * dt + n1;
        rz += d0 * fz * dt + n2;
    }
    out_msd[w] = rx * rx + ry * ry + rz * rz;
}

} // extern "C"
"#;

/// GPU ensemble backend. Holds the CUDA context, stream, and compiled kernel.
pub struct GpuBackend {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: CudaFunction,
}

impl GpuBackend {
    /// Initialise device 0 and compile the BD kernel (nvrtc).
    ///
    /// # Errors
    /// [`ErmakError::Gpu`] if no CUDA device is available or the kernel fails to
    /// compile or load.
    pub fn new() -> Result<Self, ErmakError> {
        let ctx = CudaContext::new(0).map_err(driver_err)?;
        let stream = ctx.default_stream();
        let ptx = compile_ptx(KERNEL_SRC)
            .map_err(|e| ErmakError::Gpu(format!("nvrtc compile: {e:?}")))?;
        let module = ctx.load_module(ptx).map_err(driver_err)?;
        let func = module.load_function("bd_walk").map_err(driver_err)?;
        Ok(Self {
            _ctx: ctx,
            stream,
            func,
        })
    }
}

impl EnsembleBackend for GpuBackend {
    fn msd_sum(
        &self,
        scenario: &Scenario,
        n_walkers: usize,
        seed: u64,
        budget: &MemoryBudget,
    ) -> Result<f64, ErmakError> {
        // Device guardrail: one walker must fit, batches are sized to the budget.
        budget.ensure_fits(WALKER_BYTES)?;
        let batch = budget.max_items(WALKER_BYTES).max(1);

        // Upload crowders once (a non-empty buffer even when there are none, so
        // the device pointer is always valid; the kernel reads n_crowders).
        let flat: Vec<f64> = scenario
            .crowders
            .iter()
            .flat_map(|c| [c.x, c.y, c.z])
            .collect();
        let host_crowders = if flat.is_empty() { vec![0.0f64] } else { flat };
        let d_crowders = self.stream.clone_htod(&host_crowders).map_err(driver_err)?;

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let n_cr = scenario.crowders.len() as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let steps_i = scenario.steps as i32;

        let mut total = 0.0f64;
        for (start, len) in batch_spans(n_walkers, batch) {
            let mut d_out = self.stream.alloc_zeros::<f64>(len).map_err(driver_err)?;
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let len_i = len as i32;
            #[allow(clippy::cast_possible_truncation)]
            let offset = start as u32;
            let cfg = LaunchConfig::for_num_elems(len as u32);

            let mut b = self.stream.launch_builder(&self.func);
            b.arg(&mut d_out);
            b.arg(&d_crowders);
            b.arg(&n_cr);
            b.arg(&len_i);
            b.arg(&steps_i);
            b.arg(&scenario.d0);
            b.arg(&scenario.dt);
            b.arg(&scenario.box_l);
            b.arg(&scenario.sigma);
            b.arg(&scenario.eps);
            b.arg(&seed);
            b.arg(&offset);
            unsafe { b.launch(cfg) }.map_err(driver_err)?;

            let out = self.stream.clone_dtoh(&d_out).map_err(driver_err)?;
            total += out.iter().sum::<f64>();
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_free_vram_with_or_without_units() {
        assert_eq!(parse_free_vram_mib("7636\n"), Some(7636));
        assert_eq!(parse_free_vram_mib(" 7636 MiB\n"), Some(7636));
        assert_eq!(parse_free_vram_mib("7636, 8151"), Some(7636));
        assert_eq!(parse_free_vram_mib("garbage"), None);
        assert_eq!(parse_free_vram_mib(""), None);
    }

    // Requires a CUDA GPU; run with:
    //   scripts/run-bounded.sh cargo test --features gpu -- --ignored gpu_
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_free_diffusion_matches_cpu() {
        use crate::backend::CpuBackend;

        let s = Scenario {
            d0: 1.0,
            dt: 0.01,
            steps: 500,
            box_l: 8.0,
            sigma: 1.0,
            eps: 1.0,
            crowders: vec![],
        };
        let (n, seed) = (20_000usize, 7u64);
        let budget = MemoryBudget::new(1 << 34, "test");
        let t = s.steps as f64 * s.dt;

        let cpu_deff = CpuBackend.msd_sum(&s, n, seed, &budget).unwrap() / (n as f64 * 6.0 * t);
        let gpu = GpuBackend::new().unwrap();
        let gpu_deff = gpu.msd_sum(&s, n, seed, &budget).unwrap() / (n as f64 * 6.0 * t);

        let rel = (cpu_deff - gpu_deff).abs() / cpu_deff;
        assert!(
            rel < 0.05,
            "GPU D_eff {gpu_deff:.4} should match CPU {cpu_deff:.4} (rel {rel:.3})"
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_crowding_matches_cpu() {
        use crate::backend::CpuBackend;
        use crate::crowding::cubic_lattice;

        let s = Scenario {
            d0: 1.0,
            dt: 0.0002,
            steps: 4000,
            box_l: 8.0,
            sigma: 1.0,
            eps: 1.0,
            crowders: cubic_lattice(8.0, 5), // phi ~ 0.13
        };
        let (n, seed) = (4000usize, 11u64);
        let budget = MemoryBudget::new(1 << 34, "test");
        let t = s.steps as f64 * s.dt;

        let cpu_deff = CpuBackend.msd_sum(&s, n, seed, &budget).unwrap() / (n as f64 * 6.0 * t);
        let gpu = GpuBackend::new().unwrap();
        let gpu_deff = gpu.msd_sum(&s, n, seed, &budget).unwrap() / (n as f64 * 6.0 * t);

        let rel = (cpu_deff - gpu_deff).abs() / cpu_deff;
        assert!(
            rel < 0.10,
            "GPU crowded D_eff {gpu_deff:.4} should match CPU {cpu_deff:.4} (rel {rel:.3})"
        );
    }
}
