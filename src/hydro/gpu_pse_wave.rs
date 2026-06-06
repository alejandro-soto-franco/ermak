//! GPU wave-space (reciprocal) RPY mobility via Spectral Ewald (feature `gpu`,
//! Milestone G4 step 2 Task 2). The device counterpart of
//! [`crate::hydro::pse_wave::recip_apply_pse`]: spread particle forces to a grid
//! with a Gaussian window (NVRTC kernel), forward FFT (cuFFT [`super::gpu_fft`]),
//! scale each mode by the wave-space Green's tensor (k=0 dropped), inverse FFT,
//! and gather back to particles. The CPU reference and the dense
//! [`crate::hydro::ewald::recip_space_block`] apply are the correctness oracles.
//!
//! The spread and gather kernels mirror the CPU reference exactly: one thread per
//! grid point (spread) or per particle (gather), looping over the full set with
//! the full periodic Gaussian, so the GPU result equals the CPU result to
//! round-off and no atomics are needed. The truncated-support O(N log N)
//! spreading is the perf follow-up; this milestone establishes the validated
//! device path.

use crate::error::ErmakError;
use crate::hydro::gpu_fft::Fft3d;
use crate::hydro::pse_wave::WaveParams;
use crate::vec3::Vec3;
use cudarc::cufft::sys::double2;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

fn driver_err(e: DriverError) -> ErmakError {
    ErmakError::Gpu(format!("cuda driver: {e:?}"))
}

/// NVRTC kernels: Gaussian spread, k-space Green scaling, Gaussian gather. The
/// grid is row-major `g = (gx*ng + gy)*ng + gz`, matching the CPU reference and
/// cuFFT's natural layout. `double2` is the CUDA built-in complex type.
const KERNEL_SRC: &str = r#"
// Spread each force component to the grid with the periodic Gaussian window
// gamma(d) = (2 pi eta^2)^{-3/2} sum_{image shell} exp(-|d|^2 / 2 eta^2). One
// thread per grid point accumulates over all particles (mirror of the CPU spread).
extern "C" __global__ void pse_spread(
    double2 *hgx, double2 *hgy, double2 *hgz,
    const double *pos, const double *force,
    int n, int ng, double h, double eta, double l)
{
    const double PI = 3.14159265358979323846;
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int ng3 = ng * ng * ng;
    if (g >= ng3) return;
    int gx = g / (ng * ng);
    int gy = (g / ng) % ng;
    int gz = g % ng;
    double xg = h * gx, yg = h * gy, zg = h * gz;
    double inv2e2 = 1.0 / (2.0 * eta * eta);
    double norm = pow(2.0 * PI * eta * eta, -1.5);

    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int j = 0; j < n; j++) {
        double dx = xg - pos[3*j+0];
        double dy = yg - pos[3*j+1];
        double dz = zg - pos[3*j+2];
        dx -= l * round(dx / l);
        dy -= l * round(dy / l);
        dz -= l * round(dz / l);
        double w = 0.0;
        for (int ix = -1; ix <= 1; ix++) { double ex = dx + ix * l;
        for (int iy = -1; iy <= 1; iy++) { double ey = dy + iy * l;
        for (int iz = -1; iz <= 1; iz++) { double ez = dz + iz * l;
            w += exp(-(ex*ex + ey*ey + ez*ez) * inv2e2);
        }}}
        w *= norm;
        sx += w * force[3*j+0];
        sy += w * force[3*j+1];
        sz += w * force[3*j+2];
    }
    hgx[g].x = sx; hgx[g].y = 0.0;
    hgy[g].x = sy; hgy[g].y = 0.0;
    hgz[g].x = sz; hgz[g].y = 0.0;
}

// Scale each Fourier mode by D(k) = h^3 exp(k^2 eta^2) PRE(k) (I - B(k) k_hat k_hat),
// PRE(k) = (pi/V)(1/k^2 - a^2/3) exp(-k^2 s^2/2), B(k) = 1 + s^2 k^2/2; k=0 zeroed.
// One thread per mode; signed frequency via the fftfreq layout.
extern "C" __global__ void pse_scale(
    double2 *vkx, double2 *vky, double2 *vkz,
    const double2 *fkx, const double2 *fky, const double2 *fkz,
    int ng, double two_pi_l, double h3, double eta, double s, double a2, double vol)
{
    const double PI = 3.14159265358979323846;
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int ng3 = ng * ng * ng;
    if (g >= ng3) return;
    int px = g / (ng * ng);
    int py = (g / ng) % ng;
    int pz = g % ng;
    int mx = (px < ng / 2) ? px : px - ng;
    int my = (py < ng / 2) ? py : py - ng;
    int mz = (pz < ng / 2) ? pz : pz - ng;
    if (mx == 0 && my == 0 && mz == 0) {
        vkx[g].x = 0.0; vkx[g].y = 0.0;
        vky[g].x = 0.0; vky[g].y = 0.0;
        vkz[g].x = 0.0; vkz[g].y = 0.0;
        return;
    }
    double kx = mx * two_pi_l, ky = my * two_pi_l, kz = mz * two_pi_l;
    double k2 = kx*kx + ky*ky + kz*kz;
    double invk = 1.0 / sqrt(k2);
    double hkx = kx * invk, hky = ky * invk, hkz = kz * invk;
    double pre = (PI / vol) * (1.0 / k2 - a2 / 3.0) * exp(-k2 * s * s / 2.0);
    double b = 1.0 + s * s * k2 / 2.0;
    double dc = h3 * exp(k2 * eta * eta) * pre;
    double d00 = dc * (1.0 - b*hkx*hkx);
    double d01 = dc * (-b*hkx*hky);
    double d02 = dc * (-b*hkx*hkz);
    double d11 = dc * (1.0 - b*hky*hky);
    double d12 = dc * (-b*hky*hkz);
    double d22 = dc * (1.0 - b*hkz*hkz);

    double2 fx = fkx[g], fy = fky[g], fz = fkz[g];
    vkx[g].x = d00*fx.x + d01*fy.x + d02*fz.x;
    vkx[g].y = d00*fx.y + d01*fy.y + d02*fz.y;
    vky[g].x = d01*fx.x + d11*fy.x + d12*fz.x;
    vky[g].y = d01*fx.y + d11*fy.y + d12*fz.y;
    vkz[g].x = d02*fx.x + d12*fy.x + d22*fz.x;
    vkz[g].y = d02*fx.y + d12*fy.y + d22*fz.y;
}

// Gather the grid velocity back to particles (adjoint of spread, weight h^3).
// One thread per particle loops over all grid points (mirror of the CPU gather).
extern "C" __global__ void pse_gather(
    double *u, const double2 *vgx, const double2 *vgy, const double2 *vgz,
    const double *pos, int n, int ng, double h, double h3, double eta, double l)
{
    const double PI = 3.14159265358979323846;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ng3 = ng * ng * ng;
    double xi = pos[3*i+0], yi = pos[3*i+1], zi = pos[3*i+2];
    double inv2e2 = 1.0 / (2.0 * eta * eta);
    double norm = pow(2.0 * PI * eta * eta, -1.5);

    double ux = 0.0, uy = 0.0, uz = 0.0;
    for (int g = 0; g < ng3; g++) {
        int gx = g / (ng * ng);
        int gy = (g / ng) % ng;
        int gz = g % ng;
        double dx = h * gx - xi;
        double dy = h * gy - yi;
        double dz = h * gz - zi;
        dx -= l * round(dx / l);
        dy -= l * round(dy / l);
        dz -= l * round(dz / l);
        double w = 0.0;
        for (int ix = -1; ix <= 1; ix++) { double ex = dx + ix * l;
        for (int iy = -1; iy <= 1; iy++) { double ey = dy + iy * l;
        for (int iz = -1; iz <= 1; iz++) { double ez = dz + iz * l;
            w += exp(-(ex*ex + ey*ey + ez*ez) * inv2e2);
        }}}
        w *= norm;
        ux += w * vgx[g].x;
        uy += w * vgy[g].x;
        uz += w * vgz[g].x;
    }
    u[3*i+0] = ux * h3;
    u[3*i+1] = uy * h3;
    u[3*i+2] = uz * h3;
}
"#;

/// GPU Spectral-Ewald reciprocal mobility. Holds the context, stream, and the
/// spread/scale/gather kernels; the cuFFT plan is built per call (sized to `ng`).
pub struct GpuPseWave {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    spread: CudaFunction,
    scale: CudaFunction,
    gather: CudaFunction,
}

impl GpuPseWave {
    /// Initialise device 0 and compile the spread/scale/gather kernels.
    ///
    /// # Errors
    /// [`ErmakError::Gpu`] if no CUDA device is available or the kernels fail to
    /// compile or load.
    pub fn new() -> Result<Self, ErmakError> {
        let ctx = CudaContext::new(0).map_err(driver_err)?;
        let stream = ctx.default_stream();
        let ptx = compile_ptx(KERNEL_SRC)
            .map_err(|e| ErmakError::Gpu(format!("nvrtc compile: {e:?}")))?;
        let module = ctx.load_module(ptx).map_err(driver_err)?;
        let spread = module.load_function("pse_spread").map_err(driver_err)?;
        let scale = module.load_function("pse_scale").map_err(driver_err)?;
        let gather = module.load_function("pse_gather").map_err(driver_err)?;
        Ok(Self {
            _ctx: ctx,
            stream,
            spread,
            scale,
            gather,
        })
    }

    /// Wave-space reciprocal apply on the device: `U_recip_i = sum_j M_recip(r_ij) F_j`,
    /// matching [`crate::hydro::pse_wave::recip_apply_pse`] (GRPerY units).
    ///
    /// # Panics
    /// If `forces.len() != pos.len()`.
    ///
    /// # Errors
    /// [`ErmakError::Gpu`] on any driver or cuFFT error.
    #[allow(clippy::too_many_lines)]
    pub fn recip_apply_pse_gpu(
        &self,
        pos: &[Vec3],
        forces: &[Vec3],
        wp: &WaveParams,
    ) -> Result<Vec<Vec3>, ErmakError> {
        let n = pos.len();
        assert_eq!(forces.len(), n, "forces and positions length mismatch");
        let ng = wp.ng;
        let ng3 = ng * ng * ng;
        let l = wp.box_l;
        #[allow(clippy::cast_precision_loss)]
        let h = l / ng as f64;
        let h3 = h * h * h;
        let eta = wp.eta;
        let s = wp.sigma;
        let a2 = wp.a * wp.a;
        let vol = l * l * l;
        let two_pi_l = 2.0 * std::f64::consts::PI / l;

        let host_pos: Vec<f64> = pos.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let host_f: Vec<f64> = forces.iter().flat_map(|f| [f.x, f.y, f.z]).collect();
        let d_pos = self.stream.clone_htod(&host_pos).map_err(driver_err)?;
        let d_f = self.stream.clone_htod(&host_f).map_err(driver_err)?;

        // Complex grids: spread output (also cuFFT scratch), forward transforms,
        // scaled modes, inverse transforms (velocity field).
        let mut hgx = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut hgy = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut hgz = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut fkx = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut fky = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut fkz = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut vkx = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut vky = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut vkz = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut vgx = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut vgy = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut vgz = self
            .stream
            .alloc_zeros::<double2>(ng3)
            .map_err(driver_err)?;
        let mut d_u = self.stream.alloc_zeros::<f64>(3 * n).map_err(driver_err)?;

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let n_i = n as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let ng_i = ng as i32;

        // --- 1. spread (one thread per grid point) ---
        #[allow(clippy::cast_possible_truncation)]
        let ng3_u = ng3 as u32;
        let block = 128u32;
        let grid_cfg = LaunchConfig {
            grid_dim: (ng3_u.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.spread);
        b.arg(&mut hgx);
        b.arg(&mut hgy);
        b.arg(&mut hgz);
        b.arg(&d_pos);
        b.arg(&d_f);
        b.arg(&n_i);
        b.arg(&ng_i);
        b.arg(&h);
        b.arg(&eta);
        b.arg(&l);
        unsafe { b.launch(grid_cfg) }.map_err(driver_err)?;

        // --- 2. forward FFT each component (spread grids reused as scratch) ---
        let fft = Fft3d::new(self.stream.clone(), ng, ng, ng)?;
        fft.forward(&mut hgx, &mut fkx)?;
        fft.forward(&mut hgy, &mut fky)?;
        fft.forward(&mut hgz, &mut fkz)?;

        // --- 3. scale each mode by D(k) ---
        let mut b = self.stream.launch_builder(&self.scale);
        b.arg(&mut vkx);
        b.arg(&mut vky);
        b.arg(&mut vkz);
        b.arg(&fkx);
        b.arg(&fky);
        b.arg(&fkz);
        b.arg(&ng_i);
        b.arg(&two_pi_l);
        b.arg(&h3);
        b.arg(&eta);
        b.arg(&s);
        b.arg(&a2);
        b.arg(&vol);
        unsafe { b.launch(grid_cfg) }.map_err(driver_err)?;

        // --- 4. inverse FFT each component ---
        fft.inverse(&mut vkx, &mut vgx)?;
        fft.inverse(&mut vky, &mut vgy)?;
        fft.inverse(&mut vkz, &mut vgz)?;

        // --- 5. gather (one thread per particle) ---
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let n_u = n as u32;
        let gcfg = LaunchConfig {
            grid_dim: (n_u.div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.gather);
        b.arg(&mut d_u);
        b.arg(&vgx);
        b.arg(&vgy);
        b.arg(&vgz);
        b.arg(&d_pos);
        b.arg(&n_i);
        b.arg(&ng_i);
        b.arg(&h);
        b.arg(&h3);
        b.arg(&eta);
        b.arg(&l);
        unsafe { b.launch(gcfg) }.map_err(driver_err)?;

        let out = self.stream.clone_dtoh(&d_u).map_err(driver_err)?;
        Ok((0..n)
            .map(|i| Vec3::new(out[3 * i], out[3 * i + 1], out[3 * i + 2]))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydro::ewald::{EwaldParams, recip_space_block};
    use crate::hydro::pse_wave::recip_apply_pse;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_box(n: usize, l: f64, seed: u64) -> Vec<Vec3> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pos: Vec<Vec3> = Vec::with_capacity(n);
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

    fn dense_recip_apply(pos: &[Vec3], forces: &[Vec3], ep: &EwaldParams) -> Vec<Vec3> {
        let n = pos.len();
        let mut out = vec![Vec3::ZERO; n];
        for i in 0..n {
            let mut acc = Vec3::ZERO;
            for j in 0..n {
                let blk = recip_space_block(pos[i] - pos[j], ep);
                let f = forces[j];
                acc += Vec3::new(
                    blk.0[0] * f.x + blk.0[1] * f.y + blk.0[2] * f.z,
                    blk.0[3] * f.x + blk.0[4] * f.y + blk.0[5] * f.z,
                    blk.0[6] * f.x + blk.0[7] * f.y + blk.0[8] * f.z,
                );
            }
            out[i] = acc;
        }
        out
    }

    fn setup() -> (f64, f64, f64, Vec<Vec3>, Vec<Vec3>) {
        let l = 10.0;
        let (sigma, a) = (2.5, 1.0);
        let n = 6usize;
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
        (l, sigma, a, pos, forces)
    }

    fn max_abs(a: &[Vec3], b: &[Vec3]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(u, v)| {
                (u.x - v.x)
                    .abs()
                    .max((u.y - v.y).abs())
                    .max((u.z - v.z).abs())
            })
            .fold(0.0, f64::max)
    }

    // cargo test --features gpu -- --ignored gpu_pse_matches_cpu_reference --nocapture
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_pse_matches_cpu_reference() {
        let (l, sigma, a, pos, forces) = setup();
        let wp = WaveParams::new(l, sigma, a, 24);
        let u_cpu = recip_apply_pse(&pos, &forces, &wp);
        let dev = GpuPseWave::new().expect("cuda");
        let u_gpu = dev
            .recip_apply_pse_gpu(&pos, &forces, &wp)
            .expect("gpu pse");
        let err = max_abs(&u_cpu, &u_gpu);
        eprintln!("gpu-vs-cpu pse: max_abs={err:.3e}");
        assert!(
            err < 1e-10,
            "GPU PSE must match the CPU reference; {err:.3e}"
        );
    }

    // cargo test --features gpu -- --ignored gpu_pse_matches_dense --nocapture
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_pse_matches_dense_reciprocal() {
        let (l, sigma, a, pos, forces) = setup();
        let ep = EwaldParams {
            box_l: l,
            sigma,
            r_cut: 13.0,
            k_max: 12,
            a,
        };
        let u_dense = dense_recip_apply(&pos, &forces, &ep);
        let dev = GpuPseWave::new().expect("cuda");
        let wp = WaveParams::new(l, sigma, a, 32);
        let u_gpu = dev
            .recip_apply_pse_gpu(&pos, &forces, &wp)
            .expect("gpu pse");
        let err = max_abs(&u_dense, &u_gpu);
        eprintln!("gpu-pse-vs-dense reciprocal: max_abs={err:.3e}");
        assert!(
            err < 1e-4,
            "GPU PSE must match the dense reciprocal Ewald sum; {err:.3e}"
        );
    }
}
