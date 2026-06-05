//! GPU periodic RPY mobility-apply (feature `gpu`): U = M F on the device, where
//! M is the periodic Beenakker-Ewald grand mobility of `crate::hydro::ewald`.
//! The CUDA C kernel ports the exact real-space and reciprocal-space lattice
//! sums from `ewald.rs` (GRPerY units), one thread per particle, accumulating
//! U_i = sum_j M_ij F_j with no matrix assembled. The CPU dense path
//! `apply_mobility(periodic_grand_mobility(...))` is the correctness oracle.

use crate::error::ErmakError;
use crate::hydro::ewald::EwaldParams;
use crate::vec3::Vec3;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

fn driver_err(e: DriverError) -> ErmakError {
    ErmakError::Gpu(format!("cuda driver: {e:?}"))
}

/// CUDA C kernel: one thread per particle `i` accumulates `U_i = sum_j M_ij F_j`
/// for the periodic Beenakker-Ewald RPY grand mobility, transcribing the
/// real-space and reciprocal-space lattice sums of `ewald.rs` block by block.
const KERNEL_SRC: &str = r#"
// Abramowitz-Stegun 7.1.26 erfc, identical to `ewald.rs::erfc` (|error| < 1.5e-7).
// Used in place of CUDA's correctly-rounded built-in so the device path matches
// the CPU oracle to floating-point round-off rather than to the ~1e-7 gap between
// the two erfc implementations.
__device__ __forceinline__ double erfc_as(double x) {
    double z = fabs(x);
    double t = 1.0 / (1.0 + 0.3275911 * z);
    double y = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
    return x >= 0.0 ? y : 2.0 - y;
}

extern "C" __global__ void ewald_apply(
    double *u, const double *pos, const double *force, int n,
    double box_l, double sigma, double r_cut, int k_max, double a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double PI = 3.14159265358979323846;
    const double s = sigma;
    const double s2 = s * s;
    const double a2 = a * a;
    const double vol = box_l * box_l * box_l;
    const double two_a2 = 2.0 * a2;
    const double p23a2 = two_a2 / 3.0;
    const double sqrt2 = 1.41421356237309504880;
    const double sqrt2pi = 2.50662827463100050242; // sqrt(2 pi)
    const double backflow = -(s2 * PI / (2.0 * vol));
    const double mu0 = 1.0 / (6.0 * a);
    const double self_analytic = (a2 / (9.0 * s2) - 1.0) / (4.0 * sqrt2pi * s);
    int span = (int)ceil(r_cut / box_l) + 1;

    double xi = pos[3*i+0], yi = pos[3*i+1], zi = pos[3*i+2];
    double ux = 0.0, uy = 0.0, uz = 0.0;

    for (int j = 0; j < n; j++) {
        int is_self = (j == i);
        double rijx = xi - pos[3*j+0];
        double rijy = yi - pos[3*j+1];
        double rijz = zi - pos[3*j+2];

        // 3x3 block m, row-major.
        double m00=0,m01=0,m02=0,m10=0,m11=0,m12=0,m20=0,m21=0,m22=0;

        // ---- real-space lattice sum ----
        for (int nx = -span; nx <= span; nx++)
        for (int ny = -span; ny <= span; ny++)
        for (int nz = -span; nz <= span; nz++) {
            if (is_self && nx==0 && ny==0 && nz==0) continue;
            double dx = rijx + nx * box_l;
            double dy = rijy + ny * box_l;
            double dz = rijz + nz * box_l;
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = sqrt(r2);
            if (r > r_cut || r == 0.0) continue;
            double r3 = r2 * r;
            double ex = dx / r, ey = dy / r, ez = dz / r; // unit displacement
            double ec = erfc_as(r / (s * sqrt2));
            double id_coef = (1.0/r + p23a2/r3) / 8.0 * ec;
            double rr_coef = (1.0/r - 3.0*p23a2/r3) / 8.0 * ec;
            double p_i = two_a2 * (1.0/(6.0*s2) + 1.0/(3.0*r2));
            double p_rr = two_a2 * (r2/(6.0*s2*s2) - 1.0/(3.0*s2) - 1.0/r2) + 1.0;
            double pre = exp(-r2/(2.0*s2)) / (4.0 * sqrt2pi * s);
            double id_part = id_coef + p_i * pre;   // coefficient on I
            double rr_part = rr_coef + p_rr * pre;  // coefficient on (e e^T)
            m00 += id_part + rr_part*ex*ex; m01 += rr_part*ex*ey; m02 += rr_part*ex*ez;
            m10 += rr_part*ey*ex; m11 += id_part + rr_part*ey*ey; m12 += rr_part*ey*ez;
            m20 += rr_part*ez*ex; m21 += rr_part*ez*ey; m22 += id_part + rr_part*ez*ez;
        }

        // ---- reciprocal-space sum (phase uses rij, not images) ----
        double two_pi_l = 2.0 * PI / box_l;
        for (int kx = -k_max; kx <= k_max; kx++)
        for (int ky = -k_max; ky <= k_max; ky++)
        for (int kz = -k_max; kz <= k_max; kz++) {
            if (kx==0 && ky==0 && kz==0) continue;
            double kvx = kx * two_pi_l, kvy = ky * two_pi_l, kvz = kz * two_pi_l;
            double k2 = kvx*kvx + kvy*kvy + kvz*kvz;
            double k = sqrt(k2);
            double hx = kvx/k, hy = kvy/k, hz = kvz/k; // k_hat
            double pre = (PI/vol) * (1.0/k2 - a2/3.0) * exp(-k2*s2/2.0);
            double coef = 1.0 + s2*k2/2.0; // factor on k_hat k_hat
            double phase = cos(kvx*rijx + kvy*rijy + kvz*rijz);
            double w = pre * phase;
            // tensor = I - coef * (h h^T)
            m00 += w*(1.0 - coef*hx*hx); m01 += w*(-coef*hx*hy); m02 += w*(-coef*hx*hz);
            m10 += w*(-coef*hy*hx); m11 += w*(1.0 - coef*hy*hy); m12 += w*(-coef*hy*hz);
            m20 += w*(-coef*hz*hx); m21 += w*(-coef*hz*hy); m22 += w*(1.0 - coef*hz*hz);
        }

        // ---- diagonal scalar terms + k=0 backflow on every block ----
        if (is_self) {
            double d = mu0 + self_analytic + backflow;
            m00 += d; m11 += d; m22 += d;
        } else {
            m00 += backflow; m11 += backflow; m22 += backflow;
        }

        // accumulate U_i += M_ij . F_j
        double fx = force[3*j+0], fy = force[3*j+1], fz = force[3*j+2];
        ux += m00*fx + m01*fy + m02*fz;
        uy += m10*fx + m11*fy + m12*fz;
        uz += m20*fx + m21*fy + m22*fz;
    }
    u[3*i+0] = ux; u[3*i+1] = uy; u[3*i+2] = uz;
}
"#;

/// GPU periodic RPY mobility-apply. Holds the CUDA context, stream, and kernel.
pub struct GpuEwald {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: CudaFunction,
}

impl GpuEwald {
    /// Initialise device 0 and compile the mobility-apply kernel (nvrtc).
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
        let func = module.load_function("ewald_apply").map_err(driver_err)?;
        Ok(Self {
            _ctx: ctx,
            stream,
            func,
        })
    }

    /// Compute `U_i = sum_j M_ij F_j` for the periodic RPY grand mobility `M`
    /// (GRPerY units, as in [`crate::hydro::ewald::periodic_grand_mobility`]).
    ///
    /// # Panics
    /// If `forces.len() != pos.len()`.
    ///
    /// # Errors
    /// [`ErmakError::Gpu`] on any driver error.
    pub fn apply_mobility_gpu(
        &self,
        pos: &[Vec3],
        forces: &[Vec3],
        ep: &EwaldParams,
    ) -> Result<Vec<Vec3>, ErmakError> {
        let n = pos.len();
        assert_eq!(forces.len(), n, "forces and positions length mismatch");
        let host_pos: Vec<f64> = pos.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let host_f: Vec<f64> = forces.iter().flat_map(|f| [f.x, f.y, f.z]).collect();
        let d_pos = self.stream.clone_htod(&host_pos).map_err(driver_err)?;
        let d_f = self.stream.clone_htod(&host_f).map_err(driver_err)?;
        let mut d_u = self.stream.alloc_zeros::<f64>(3 * n).map_err(driver_err)?;

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let n_i = n as i32;
        let (box_l, sigma, r_cut, a) = (ep.box_l, ep.sigma, ep.r_cut, ep.a);
        let k_max = ep.k_max;

        // Small block: this kernel is register-heavy (the full 3x3 block math plus
        // the real/reciprocal lattice sums per thread), so a 1024-thread block
        // over-subscribes the per-block register file on sm_120 and the launch
        // fails with OUT_OF_RESOURCES. 64 threads/block keeps register pressure
        // in bounds; grid covers all particles.
        #[allow(clippy::cast_sign_loss)]
        let n_u = n_i as u32;
        let block = 64u32;
        let cfg = LaunchConfig {
            grid_dim: (n_u.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.func);
        b.arg(&mut d_u);
        b.arg(&d_pos);
        b.arg(&d_f);
        b.arg(&n_i);
        b.arg(&box_l);
        b.arg(&sigma);
        b.arg(&r_cut);
        b.arg(&k_max);
        b.arg(&a);
        unsafe { b.launch(cfg) }.map_err(driver_err)?;

        let out = self.stream.clone_dtoh(&d_u).map_err(driver_err)?;
        Ok((0..n)
            .map(|i| Vec3::new(out[3 * i], out[3 * i + 1], out[3 * i + 2]))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydro::ewald::periodic_grand_mobility;
    use crate::hydro::mobility::apply_mobility;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_box(n: usize, l: f64, seed: u64) -> Vec<Vec3> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pos = Vec::with_capacity(n);
        let mut tries = 0;
        while pos.len() < n && tries < 1_000_000 {
            tries += 1;
            let p = Vec3::new(
                rng.gen_range(0.0..l),
                rng.gen_range(0.0..l),
                rng.gen_range(0.0..l),
            );
            let ok = pos.iter().all(|q: &Vec3| {
                let mut d = p - *q;
                d = Vec3::new(
                    d.x - l * (d.x / l).round(),
                    d.y - l * (d.y / l).round(),
                    d.z - l * (d.z / l).round(),
                );
                d.norm2().sqrt() > 2.5
            });
            if ok {
                pos.push(p);
            }
        }
        pos
    }

    // Requires a CUDA GPU. Run with:
    //   cargo test --features gpu -- --ignored gpu_apply
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_apply_matches_cpu_oracle() {
        let l = 10.0;
        let ep = EwaldParams {
            box_l: l,
            sigma: 2.5,
            r_cut: 13.0,
            k_max: 8,
            a: 1.0,
        };
        let n = 12usize;
        let pos = random_box(n, l, 7);
        let mut rng = StdRng::seed_from_u64(99);
        let forces: Vec<Vec3> = (0..n)
            .map(|_| {
                Vec3::new(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                )
            })
            .collect();

        let m = periodic_grand_mobility(&pos, &ep);
        let u_cpu = apply_mobility(&m, &forces);

        let dev = GpuEwald::new().expect("CUDA init");
        let u_gpu = dev
            .apply_mobility_gpu(&pos, &forces, &ep)
            .expect("gpu apply");

        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        for i in 0..n {
            for (a, b) in [
                (u_cpu[i].x, u_gpu[i].x),
                (u_cpu[i].y, u_gpu[i].y),
                (u_cpu[i].z, u_gpu[i].z),
            ] {
                let d = (a - b).abs();
                max_abs = max_abs.max(d);
                max_rel = max_rel.max(d / (a.abs() + 1e-300));
            }
        }
        eprintln!("gpu-vs-cpu apply: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
        assert!(
            max_abs < 1e-9,
            "GPU mobility-apply must match CPU oracle; max_abs {max_abs:.3e}"
        );
    }
}
