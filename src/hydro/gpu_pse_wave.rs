//! GPU wave-space (reciprocal) RPY mobility via Spectral Ewald (feature `gpu`,
//! Milestone G4 step 2 Task 2). The device counterpart of
//! [`crate::hydro::pse_wave::recip_apply_pse`]: spread particle forces to a grid
//! with a Gaussian window (NVRTC kernel), forward FFT (cuFFT [`super::gpu_fft`]),
//! scale each mode by the wave-space Green's tensor (k=0 dropped), inverse FFT,
//! and gather back to particles. The CPU reference and the dense
//! [`crate::hydro::ewald::recip_space_block`] apply are the correctness oracles.
//!
//! Two spread/gather paths share the FFT and k-space scaling:
//! - **full grid** (`WaveParams::new`): one thread per grid point (spread) or per
//!   particle (gather) over the whole grid with the full periodic Gaussian, so the
//!   GPU result equals the CPU reference to round-off, no atomics. The validation
//!   path.
//! - **truncated support** (`WaveParams::truncated`): one thread per particle
//!   scatters/gathers only the `(2*support+1)^3` nearest nodes (atomicAdd for the
//!   scatter), the O(N P^3) Spectral-Ewald window whose per-particle cost is
//!   independent of `ng`. With `eta ~ h` the aliasing (`~exp(-2 pi^2)`) and the
//!   truncation (`~exp(-support^2/2)`) are negligible, so it converges to the dense
//!   reciprocal sum (support 5 reaches ~5e-9). The O(N log N) production path.

use crate::error::ErmakError;
use crate::hydro::ewald::EwaldParams;
use crate::hydro::gpu_fft::Fft3d;
use crate::hydro::gpu_noise::lanczos_half_apply;
use crate::hydro::pse_wave::WaveParams;
use crate::vec3::Vec3;
use cudarc::cufft::sys::double2;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::cell::RefCell;
use std::collections::HashMap;
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

// Real-space (short-range) part of the periodic RPY mobility apply: the
// erfc-screened bare RPY plus GRPerY Gaussian polynomial lattice sum, the self
// scalar terms (mu0 + analytic n=0 replacement) on the diagonal block, and the
// k=0 backflow on every block. This is `ewald_apply` of `gpu_ewald` with the
// reciprocal loop removed; the FFT wave path supplies the reciprocal half, so
// `pse_real + wave = full periodic mobility`.
__device__ __forceinline__ double erfc_as(double x) {
    double z = fabs(x);
    double t = 1.0 / (1.0 + 0.3275911 * z);
    double y = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
    return x >= 0.0 ? y : 2.0 - y;
}

extern "C" __global__ void pse_real(
    double *u, const double *pos, const double *force, int n,
    double box_l, double sigma, double r_cut, double a)
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
    const double sqrt2pi = 2.50662827463100050242;
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
        double m00=0,m01=0,m02=0,m10=0,m11=0,m12=0,m20=0,m21=0,m22=0;

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
            double ex = dx / r, ey = dy / r, ez = dz / r;
            double ec = erfc_as(r / (s * sqrt2));
            double id_coef = (1.0/r + p23a2/r3) / 8.0 * ec;
            double rr_coef = (1.0/r - 3.0*p23a2/r3) / 8.0 * ec;
            double p_i = two_a2 * (1.0/(6.0*s2) + 1.0/(3.0*r2));
            double p_rr = two_a2 * (r2/(6.0*s2*s2) - 1.0/(3.0*s2) - 1.0/r2) + 1.0;
            double pre = exp(-r2/(2.0*s2)) / (4.0 * sqrt2pi * s);
            double id_part = id_coef + p_i * pre;
            double rr_part = rr_coef + p_rr * pre;
            m00 += id_part + rr_part*ex*ex; m01 += rr_part*ex*ey; m02 += rr_part*ex*ez;
            m10 += rr_part*ey*ex; m11 += id_part + rr_part*ey*ey; m12 += rr_part*ey*ez;
            m20 += rr_part*ez*ex; m21 += rr_part*ez*ey; m22 += id_part + rr_part*ez*ez;
        }

        if (is_self) {
            double d = mu0 + self_analytic + backflow;
            m00 += d; m11 += d; m22 += d;
        } else {
            m00 += backflow; m11 += backflow; m22 += backflow;
        }

        double fx = force[3*j+0], fy = force[3*j+1], fz = force[3*j+2];
        ux += m00*fx + m01*fy + m02*fz;
        uy += m10*fx + m11*fy + m12*fz;
        uz += m20*fx + m21*fy + m22*fz;
    }
    u[3*i+0] = ux; u[3*i+1] = uy; u[3*i+2] = uz;
}

// Truncated-support spread: one thread per PARTICLE scatters its force to the
// (2*support+1)^3 nearest grid nodes with the Gaussian window, via atomicAdd. The
// per-particle cost is O(support^3), independent of ng (the O(N log N) spread).
// The grids MUST be zeroed before launch (only the window is written). Node
// indices wrap periodically; the Gaussian uses the unwrapped node position so the
// displacement to the particle stays small (no image sum needed).
extern "C" __global__ void pse_spread_trunc(
    double2 *hgx, double2 *hgy, double2 *hgz,
    const double *pos, const double *force,
    int n, int ng, int support, double h, double eta)
{
    const double PI = 3.14159265358979323846;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double px = pos[3*j+0], py = pos[3*j+1], pz = pos[3*j+2];
    double fx = force[3*j+0], fy = force[3*j+1], fz = force[3*j+2];
    double inv2e2 = 1.0 / (2.0 * eta * eta);
    double norm = pow(2.0 * PI * eta * eta, -1.5);
    long cx = lround(px / h), cy = lround(py / h), cz = lround(pz / h);
    long ngl = ng;
    for (int ox = -support; ox <= support; ox++) {
        long gxu = cx + ox; double dx = gxu * h - px; double wx = exp(-dx*dx*inv2e2);
        long gx = ((gxu % ngl) + ngl) % ngl;
        for (int oy = -support; oy <= support; oy++) {
            long gyu = cy + oy; double dy = gyu * h - py; double wy = exp(-dy*dy*inv2e2);
            long gy = ((gyu % ngl) + ngl) % ngl;
            for (int oz = -support; oz <= support; oz++) {
                long gzu = cz + oz; double dz = gzu * h - pz;
                long gz = ((gzu % ngl) + ngl) % ngl;
                double w = norm * wx * wy * exp(-dz*dz*inv2e2);
                long g = (gx * ngl + gy) * ngl + gz;
                atomicAdd(&hgx[g].x, w * fx);
                atomicAdd(&hgy[g].x, w * fy);
                atomicAdd(&hgz[g].x, w * fz);
            }
        }
    }
}

// Truncated-support gather: one thread per particle reads the grid velocity from
// its (2*support+1)^3 window (adjoint of pse_spread_trunc, weight h^3). O(support^3).
extern "C" __global__ void pse_gather_trunc(
    double *u, const double2 *vgx, const double2 *vgy, const double2 *vgz,
    const double *pos, int n, int ng, int support, double h, double h3, double eta)
{
    const double PI = 3.14159265358979323846;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double px = pos[3*i+0], py = pos[3*i+1], pz = pos[3*i+2];
    double inv2e2 = 1.0 / (2.0 * eta * eta);
    double norm = pow(2.0 * PI * eta * eta, -1.5);
    long cx = lround(px / h), cy = lround(py / h), cz = lround(pz / h);
    long ngl = ng;
    double ux = 0.0, uy = 0.0, uz = 0.0;
    for (int ox = -support; ox <= support; ox++) {
        long gxu = cx + ox; double dx = gxu * h - px; double wx = exp(-dx*dx*inv2e2);
        long gx = ((gxu % ngl) + ngl) % ngl;
        for (int oy = -support; oy <= support; oy++) {
            long gyu = cy + oy; double dy = gyu * h - py; double wy = exp(-dy*dy*inv2e2);
            long gy = ((gyu % ngl) + ngl) % ngl;
            for (int oz = -support; oz <= support; oz++) {
                long gzu = cz + oz; double dz = gzu * h - pz;
                long gz = ((gzu % ngl) + ngl) % ngl;
                double w = norm * wx * wy * exp(-dz*dz*inv2e2);
                long g = (gx * ngl + gy) * ngl + gz;
                ux += w * vgx[g].x;
                uy += w * vgy[g].x;
                uz += w * vgz[g].x;
            }
        }
    }
    u[3*i+0] = ux * h3;
    u[3*i+1] = uy * h3;
    u[3*i+2] = uz * h3;
}
"#;

/// Reusable device buffers for the wave-space apply, sized to `(n, ng)`. Cached so
/// a Lanczos noise solve (3N matvecs, each a `full_apply`) does not reallocate the
/// twelve `ng^3` complex grids on every matvec. The grids are never zeroed: every
/// element is written by spread / scale / FFT / gather before it is read.
struct Scratch {
    n: usize,
    ng: usize,
    d_pos: CudaSlice<f64>,
    d_f: CudaSlice<f64>,
    d_u: CudaSlice<f64>,
    hgx: CudaSlice<double2>,
    hgy: CudaSlice<double2>,
    hgz: CudaSlice<double2>,
    fkx: CudaSlice<double2>,
    fky: CudaSlice<double2>,
    fkz: CudaSlice<double2>,
    vkx: CudaSlice<double2>,
    vky: CudaSlice<double2>,
    vkz: CudaSlice<double2>,
    vgx: CudaSlice<double2>,
    vgy: CudaSlice<double2>,
    vgz: CudaSlice<double2>,
}

impl Scratch {
    fn new(stream: &Arc<CudaStream>, n: usize, ng: usize) -> Result<Self, ErmakError> {
        let ng3 = ng * ng * ng;
        let cx = || stream.alloc_zeros::<double2>(ng3).map_err(driver_err);
        let f3 = || stream.alloc_zeros::<f64>(3 * n).map_err(driver_err);
        Ok(Self {
            n,
            ng,
            d_pos: f3()?,
            d_f: f3()?,
            d_u: f3()?,
            hgx: cx()?,
            hgy: cx()?,
            hgz: cx()?,
            fkx: cx()?,
            fky: cx()?,
            fkz: cx()?,
            vkx: cx()?,
            vky: cx()?,
            vkz: cx()?,
            vgx: cx()?,
            vgy: cx()?,
            vgz: cx()?,
        })
    }
}

/// GPU Spectral-Ewald reciprocal mobility. Holds the context, stream, and the
/// spread/scale/gather kernels; cuFFT plans and device buffers are cached and
/// reused across applies (keyed by grid size).
pub struct GpuPseWave {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    spread: CudaFunction,
    scale: CudaFunction,
    gather: CudaFunction,
    real: CudaFunction,
    spread_trunc: CudaFunction,
    gather_trunc: CudaFunction,
    /// Reusable device buffers (the twelve `ng^3` grids plus the `3n` vectors),
    /// keyed by `(n, ng)`; rebuilt only when the size changes. `RefCell` because
    /// the apply takes `&self`.
    scratch: RefCell<Option<Scratch>>,
    /// cuFFT plans cached by grid size `ng`: a plan is built once per `ng` and
    /// reused across every `recip_apply_pse_gpu` call, rather than re-planned on
    /// each apply. A Lanczos noise draw makes 3N matvecs (each a `full_apply`), so
    /// the engine plans many times for one `ng`; the cache avoids that redundant
    /// planning and the work-area reallocation it carries (the win grows with `ng`,
    /// where planning is pricier). `RefCell` because the apply takes `&self`.
    fft_cache: RefCell<HashMap<usize, Fft3d>>,
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
        let real = module.load_function("pse_real").map_err(driver_err)?;
        let spread_trunc = module
            .load_function("pse_spread_trunc")
            .map_err(driver_err)?;
        let gather_trunc = module
            .load_function("pse_gather_trunc")
            .map_err(driver_err)?;
        Ok(Self {
            _ctx: ctx,
            stream,
            spread,
            scale,
            gather,
            real,
            spread_trunc,
            gather_trunc,
            scratch: RefCell::new(None),
            fft_cache: RefCell::new(HashMap::new()),
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

        // Reusable device buffers sized to (n, ng); rebuilt only on a size change
        // (see `Scratch`). Positions and forces are copied into the cached buffers
        // rather than reallocated each call.
        let mut guard = self.scratch.borrow_mut();
        if guard.as_ref().is_none_or(|sc| sc.n != n || sc.ng != ng) {
            *guard = Some(Scratch::new(&self.stream, n, ng)?);
        }
        let sc = guard.as_mut().expect("scratch present");
        self.stream
            .memcpy_htod(&host_pos, &mut sc.d_pos)
            .map_err(driver_err)?;
        self.stream
            .memcpy_htod(&host_f, &mut sc.d_f)
            .map_err(driver_err)?;

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let n_i = n as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let ng_i = ng as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let support_i = wp.support as i32;
        let truncated = wp.is_truncated();
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let n_u = n as u32;
        // Per-grid-point launch (full path) and per-particle launch (truncated path).
        #[allow(clippy::cast_possible_truncation)]
        let ng3_u = ng3 as u32;
        let block = 128u32;
        let grid_cfg = LaunchConfig {
            grid_dim: (ng3_u.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let particle_cfg = LaunchConfig {
            grid_dim: (n_u.div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };

        // --- 1. spread ---
        if truncated {
            // Truncated path scatters only the window per particle, so the spread
            // grids must start at zero.
            self.stream.memset_zeros(&mut sc.hgx).map_err(driver_err)?;
            self.stream.memset_zeros(&mut sc.hgy).map_err(driver_err)?;
            self.stream.memset_zeros(&mut sc.hgz).map_err(driver_err)?;
            let mut b = self.stream.launch_builder(&self.spread_trunc);
            b.arg(&mut sc.hgx);
            b.arg(&mut sc.hgy);
            b.arg(&mut sc.hgz);
            b.arg(&sc.d_pos);
            b.arg(&sc.d_f);
            b.arg(&n_i);
            b.arg(&ng_i);
            b.arg(&support_i);
            b.arg(&h);
            b.arg(&eta);
            unsafe { b.launch(particle_cfg) }.map_err(driver_err)?;
        } else {
            let mut b = self.stream.launch_builder(&self.spread);
            b.arg(&mut sc.hgx);
            b.arg(&mut sc.hgy);
            b.arg(&mut sc.hgz);
            b.arg(&sc.d_pos);
            b.arg(&sc.d_f);
            b.arg(&n_i);
            b.arg(&ng_i);
            b.arg(&h);
            b.arg(&eta);
            b.arg(&l);
            unsafe { b.launch(grid_cfg) }.map_err(driver_err)?;
        }

        // --- 2. forward FFT each component (spread grids reused as scratch) ---
        // Reuse the cached cuFFT plan for this grid size, building it on first use.
        let mut cache = self.fft_cache.borrow_mut();
        let fft = match cache.entry(ng) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(Fft3d::new(self.stream.clone(), ng, ng, ng)?)
            }
        };
        fft.forward(&mut sc.hgx, &mut sc.fkx)?;
        fft.forward(&mut sc.hgy, &mut sc.fky)?;
        fft.forward(&mut sc.hgz, &mut sc.fkz)?;

        // --- 3. scale each mode by D(k) ---
        let mut b = self.stream.launch_builder(&self.scale);
        b.arg(&mut sc.vkx);
        b.arg(&mut sc.vky);
        b.arg(&mut sc.vkz);
        b.arg(&sc.fkx);
        b.arg(&sc.fky);
        b.arg(&sc.fkz);
        b.arg(&ng_i);
        b.arg(&two_pi_l);
        b.arg(&h3);
        b.arg(&eta);
        b.arg(&s);
        b.arg(&a2);
        b.arg(&vol);
        unsafe { b.launch(grid_cfg) }.map_err(driver_err)?;

        // --- 4. inverse FFT each component ---
        fft.inverse(&mut sc.vkx, &mut sc.vgx)?;
        fft.inverse(&mut sc.vky, &mut sc.vgy)?;
        fft.inverse(&mut sc.vkz, &mut sc.vgz)?;

        // --- 5. gather (one thread per particle) ---
        if truncated {
            let mut b = self.stream.launch_builder(&self.gather_trunc);
            b.arg(&mut sc.d_u);
            b.arg(&sc.vgx);
            b.arg(&sc.vgy);
            b.arg(&sc.vgz);
            b.arg(&sc.d_pos);
            b.arg(&n_i);
            b.arg(&ng_i);
            b.arg(&support_i);
            b.arg(&h);
            b.arg(&h3);
            b.arg(&eta);
            unsafe { b.launch(particle_cfg) }.map_err(driver_err)?;
        } else {
            let mut b = self.stream.launch_builder(&self.gather);
            b.arg(&mut sc.d_u);
            b.arg(&sc.vgx);
            b.arg(&sc.vgy);
            b.arg(&sc.vgz);
            b.arg(&sc.d_pos);
            b.arg(&n_i);
            b.arg(&ng_i);
            b.arg(&h);
            b.arg(&h3);
            b.arg(&eta);
            b.arg(&l);
            unsafe { b.launch(particle_cfg) }.map_err(driver_err)?;
        }

        let out = self.stream.clone_dtoh(&sc.d_u).map_err(driver_err)?;
        Ok((0..n)
            .map(|i| Vec3::new(out[3 * i], out[3 * i + 1], out[3 * i + 2]))
            .collect())
    }

    /// Real-space (short-range) part of the periodic mobility apply on the device:
    /// the erfc-screened RPY lattice sum plus the self scalar and k=0 backflow
    /// terms (the `pse_real` kernel). `ep.k_max` is unused here.
    ///
    /// # Errors
    /// [`ErmakError::Gpu`] on any driver error.
    pub fn real_apply_gpu(
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
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let n_u = n as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_u.div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.real);
        b.arg(&mut d_u);
        b.arg(&d_pos);
        b.arg(&d_f);
        b.arg(&n_i);
        b.arg(&box_l);
        b.arg(&sigma);
        b.arg(&r_cut);
        b.arg(&a);
        unsafe { b.launch(cfg) }.map_err(driver_err)?;

        let out = self.stream.clone_dtoh(&d_u).map_err(driver_err)?;
        Ok((0..n)
            .map(|i| Vec3::new(out[3 * i], out[3 * i + 1], out[3 * i + 2]))
            .collect())
    }

    /// Full periodic RPY mobility apply on the device via Positively-Split Ewald:
    /// real-space part (`pse_real`) plus the FFT wave-space part
    /// ([`Self::recip_apply_pse_gpu`]) sum to the full grand mobility
    /// `U_i = sum_j M_ij F_j` of [`crate::hydro::ewald::periodic_grand_mobility`].
    /// The reciprocal half is the O(N log N) particle-mesh solve rather than the
    /// dense O(N^2 k_max^3) lattice sum. `ng` is the cubic wave-grid size.
    ///
    /// # Errors
    /// [`ErmakError::Gpu`] on any driver or cuFFT error.
    pub fn full_apply(
        &self,
        pos: &[Vec3],
        forces: &[Vec3],
        ep: &EwaldParams,
        ng: usize,
    ) -> Result<Vec<Vec3>, ErmakError> {
        let u_real = self.real_apply_gpu(pos, forces, ep)?;
        let wp = WaveParams::new(ep.box_l, ep.sigma, ep.a, ng);
        let u_wave = self.recip_apply_pse_gpu(pos, forces, &wp)?;
        Ok(u_real
            .iter()
            .zip(u_wave.iter())
            .map(|(r, w)| *r + *w)
            .collect())
    }

    /// Full-mobility matvec on a flat `3N` vector via [`Self::full_apply`], for the
    /// Lanczos noise driver.
    fn full_matvec(
        &self,
        pos: &[Vec3],
        ep: &EwaldParams,
        ng: usize,
        z: &[f64],
    ) -> Result<Vec<f64>, ErmakError> {
        let n = pos.len();
        let zf: Vec<Vec3> = (0..n)
            .map(|i| Vec3::new(z[3 * i], z[3 * i + 1], z[3 * i + 2]))
            .collect();
        let out = self.full_apply(pos, &zf, ep, ng)?;
        let mut flat = vec![0.0f64; 3 * n];
        for i in 0..n {
            flat[3 * i] = out[i].x;
            flat[3 * i + 1] = out[i].y;
            flat[3 * i + 2] = out[i].z;
        }
        Ok(flat)
    }

    /// Mobility square-root apply `M^{1/2} z` driven by the PSE [`Self::full_apply`]
    /// matvec (Lanczos/Krylov, [`lanczos_half_apply`]). Each Krylov step is one
    /// FFT-accelerated full apply, so the noise is O(N log N) like the drift, with
    /// no dense factorization. Exact to round-off at `m_iters = 3N`.
    ///
    /// # Errors
    /// [`ErmakError::Gpu`] on any device or cuFFT error in the matvecs.
    pub fn mobility_half_apply_pse(
        &self,
        pos: &[Vec3],
        ep: &EwaldParams,
        ng: usize,
        z: &[f64],
        m_iters: usize,
    ) -> Result<Vec<f64>, ErmakError> {
        lanczos_half_apply(|v| self.full_matvec(pos, ep, ng, v), z, m_iters)
    }

    /// Correlated Brownian displacement via the PSE path: `sqrt(2 kT dt) M^{1/2} xi`,
    /// `xi ~ N(0, I_{3N})`, with `M` the full periodic RPY mobility applied through
    /// the FFT wave-space solve. Same fluctuation-dissipation covariance
    /// `2 kT dt M` as the dense Lanczos path, generated in O(N log N) per matvec.
    ///
    /// # Errors
    /// [`ErmakError::Gpu`] on any device or cuFFT error.
    #[allow(clippy::too_many_arguments)]
    pub fn brownian_noise_pse<R: Rng + ?Sized>(
        &self,
        pos: &[Vec3],
        ep: &EwaldParams,
        ng: usize,
        kt: f64,
        dt: f64,
        m_iters: usize,
        rng: &mut R,
    ) -> Result<Vec<Vec3>, ErmakError> {
        let n = pos.len();
        let dim = 3 * n;
        let normal = Normal::new(0.0, 1.0).expect("unit normal");
        let xi: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();
        let half = self.mobility_half_apply_pse(pos, ep, ng, &xi, m_iters)?;
        let scale = (2.0 * kt * dt).sqrt();
        Ok((0..n)
            .map(|i| {
                Vec3::new(
                    scale * half[3 * i],
                    scale * half[3 * i + 1],
                    scale * half[3 * i + 2],
                )
            })
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

    // cargo test --features gpu -- --ignored gpu_pse_truncated --nocapture
    // The truncated-support spreading (O(N P^3) window) converges to the dense
    // reciprocal Ewald sum as the support grows: with eta = h the aliasing is
    // ~exp(-2 pi^2) and the truncation is ~exp(-(support)^2 / 2), so a small
    // support already matches the dense oracle.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_pse_truncated_converges_to_dense() {
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
        let ng = 32usize;
        let h = l / ng as f64;
        let eta = h; // ~1 cell: tiny aliasing, compact window
        let mut prev = f64::INFINITY;
        let mut finest = f64::INFINITY;
        for &support in &[3usize, 5, 8] {
            let wp = WaveParams::truncated(l, sigma, a, ng, eta, support);
            assert!(
                wp.is_truncated(),
                "support {support} must truncate at ng={ng}"
            );
            let u = dev.recip_apply_pse_gpu(&pos, &forces, &wp).expect("trunc");
            let err = max_abs(&u_dense, &u);
            eprintln!(
                "support={support:>2} (window {}^3) max_abs={err:.3e}",
                2 * support + 1
            );
            assert!(
                err <= prev,
                "error must fall with support ({support}: {err:.3e})"
            );
            prev = err;
            finest = err;
        }
        assert!(
            finest < 1e-5,
            "truncated spreading must match the dense reciprocal sum; {finest:.3e}"
        );
    }

    // cargo test --features gpu -- --ignored gpu_pse_sqrt_squared --nocapture
    // The PSE half-apply is a valid square root: sqrt(M)(sqrt(M) z) = M z (exact at
    // m = 3N). With Task 3 (full_apply = dense M) and the host-side
    // `gpu_noise::lanczos_noise_fd_covariance` pin (Lanczos sqrt -> 2 kT dt M
    // covariance for any SPD matvec), this gives the FFT noise path the right
    // fluctuation-dissipation covariance.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_pse_sqrt_squared_recovers_mobility() {
        let (l, sigma, a, pos, _f) = setup();
        let ep = EwaldParams {
            box_l: l,
            sigma,
            r_cut: 13.0,
            k_max: 12,
            a,
        };
        let n = pos.len();
        let dim = 3 * n;
        let ng = 24usize;
        let dev = GpuPseWave::new().expect("cuda");
        let mut rng = StdRng::seed_from_u64(5);
        let z: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let half = dev
            .mobility_half_apply_pse(&pos, &ep, ng, &z, dim)
            .expect("half");
        let twice = dev
            .mobility_half_apply_pse(&pos, &ep, ng, &half, dim)
            .expect("half2");
        let zf: Vec<Vec3> = (0..n)
            .map(|i| Vec3::new(z[3 * i], z[3 * i + 1], z[3 * i + 2]))
            .collect();
        let mz = dev.full_apply(&pos, &zf, &ep, ng).expect("mz");

        let mut max_abs = 0.0f64;
        for i in 0..n {
            for (p, q) in [
                (twice[3 * i], mz[i].x),
                (twice[3 * i + 1], mz[i].y),
                (twice[3 * i + 2], mz[i].z),
            ] {
                max_abs = max_abs.max((p - q).abs());
            }
        }
        eprintln!("pse sqrt-squared vs M: max_abs={max_abs:.3e}");
        assert!(max_abs < 1e-7, "sqrt(M)^2 must recover M; {max_abs:.3e}");
    }

    // cargo test --features gpu -- --ignored gpu_pse_noise_fd --nocapture
    // Direct fluctuation-dissipation pin through the FFT noise path: the sampled
    // covariance of brownian_noise_pse equals 2 kT dt M (M the dense oracle). Small
    // N and a modest sample count keep it runnable: each matvec is a full_apply
    // that currently rebuilds the cuFFT plan (the caching follow-up), so this is
    // the slow pin. The rigorous covariance claim is also covered by transitivity:
    // gpu_pse_sqrt_squared_recovers_mobility (sqrt(M)^2 = M) + full_apply = dense M
    // (gpu_pse_full_matches_dense_mobility) + gpu_noise::lanczos_noise_fd_covariance
    // (the Lanczos sqrt yields 2 kT dt M for any SPD matvec).
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_pse_noise_fd_covariance() {
        use crate::hydro::ewald::periodic_grand_mobility;
        let ep = EwaldParams {
            box_l: 10.0,
            sigma: 2.5,
            r_cut: 13.0,
            k_max: 12,
            a: 1.0,
        };
        let pos = vec![Vec3::new(3.0, 3.0, 3.0), Vec3::new(6.0, 3.0, 3.0)];
        let n = pos.len();
        let dim = 3 * n;
        let (kt, dt) = (1.3_f64, 0.5_f64);
        let ng = 16usize;
        let m = periodic_grand_mobility(&pos, &ep);

        let dev = GpuPseWave::new().expect("cuda");
        let mut rng = StdRng::seed_from_u64(20_260_606);
        let samples = 2_500usize;
        let mut cov = vec![0.0f64; dim * dim];
        for _ in 0..samples {
            let dr = dev
                .brownian_noise_pse(&pos, &ep, ng, kt, dt, dim, &mut rng)
                .expect("pse noise");
            let mut flat = vec![0.0f64; dim];
            for i in 0..n {
                flat[3 * i] = dr[i].x;
                flat[3 * i + 1] = dr[i].y;
                flat[3 * i + 2] = dr[i].z;
            }
            for a in 0..dim {
                for b in 0..dim {
                    cov[a * dim + b] += flat[a] * flat[b];
                }
            }
        }
        let mut max_rel = 0.0f64;
        for a in 0..dim {
            for b in 0..dim {
                let est = cov[a * dim + b] / samples as f64;
                let target = 2.0 * kt * dt * m[a * dim + b];
                let denom = (2.0 * kt * dt * m[a * dim + a]).max(1e-12);
                max_rel = max_rel.max((est - target).abs() / denom);
            }
        }
        eprintln!("pse FD covariance max_rel={max_rel:.4} over {samples} draws");
        assert!(max_rel < 0.08, "PSE FD covariance off by {max_rel:.3}");
    }

    // cargo test --features gpu -- --ignored gpu_pse_full_matches_dense --nocapture
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_pse_full_matches_dense_mobility() {
        use crate::hydro::ewald::periodic_grand_mobility;
        use crate::hydro::mobility::apply_mobility;
        let (l, sigma, a, pos, forces) = setup();
        let ep = EwaldParams {
            box_l: l,
            sigma,
            r_cut: 13.0,
            k_max: 12,
            a,
        };
        // Dense oracle: full periodic grand mobility apply.
        let m = periodic_grand_mobility(&pos, &ep);
        let u_dense = apply_mobility(&m, &forces);

        let dev = GpuPseWave::new().expect("cuda");
        let u_pse = dev.full_apply(&pos, &forces, &ep, 32).expect("full apply");
        let err = max_abs(&u_dense, &u_pse);
        eprintln!("gpu-pse-full-vs-dense mobility: max_abs={err:.3e}");
        assert!(
            err < 1e-4,
            "PSE (real + FFT wave) must match the dense periodic mobility; {err:.3e}"
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
