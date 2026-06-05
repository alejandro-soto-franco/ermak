//! cuFFT 3D double-precision transform for the wave-space PSE (feature `gpu`).
//! G4 step 1: the transform wrapper plus a round-trip validation that the cudarc
//! cuFFT binding links and runs here. The Gaussian spread/interpolate and the
//! Hasimoto Green's-function scaling that make this the O(N log N) wave-space
//! mobility are G4 step 2.

use crate::error::ErmakError;
use cudarc::cufft::sys::{cufftType, double2};
use cudarc::cufft::{CudaFft, FftDirection};
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

fn fft_err(e: impl core::fmt::Debug) -> ErmakError {
    ErmakError::Gpu(format!("cufft: {e:?}"))
}

/// A 3D complex-to-complex double-precision cuFFT plan.
pub struct Fft3d {
    plan: CudaFft,
    /// Total grid points nx*ny*nz (the cuFFT inverse is unnormalized by this).
    pub n: usize,
}

impl Fft3d {
    /// Plan a 3D Z2Z transform on the given stream.
    ///
    /// # Errors
    /// [`ErmakError::Gpu`] if the plan cannot be created.
    pub fn new(
        stream: Arc<CudaStream>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<Self, ErmakError> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let plan = CudaFft::plan_3d(
            nx as i32,
            ny as i32,
            nz as i32,
            cufftType::CUFFT_Z2Z,
            stream,
        )
        .map_err(fft_err)?;
        Ok(Self {
            plan,
            n: nx * ny * nz,
        })
    }

    /// Forward transform `src -> dst` (out of place).
    ///
    /// # Errors
    /// [`ErmakError::Gpu`] on a cuFFT error.
    pub fn forward(
        &self,
        src: &mut CudaSlice<double2>,
        dst: &mut CudaSlice<double2>,
    ) -> Result<(), ErmakError> {
        self.plan
            .exec_z2z(src, dst, FftDirection::Forward)
            .map_err(fft_err)
    }

    /// Inverse transform `src -> dst` (out of place, unnormalized: result is N x input).
    ///
    /// # Errors
    /// [`ErmakError::Gpu`] on a cuFFT error.
    pub fn inverse(
        &self,
        src: &mut CudaSlice<double2>,
        dst: &mut CudaSlice<double2>,
    ) -> Result<(), ErmakError> {
        self.plan
            .exec_z2z(src, dst, FftDirection::Inverse)
            .map_err(fft_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    // Requires a CUDA GPU. cargo test --features gpu -- --ignored cufft_3d_roundtrip
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn cufft_3d_roundtrip() {
        let ctx = CudaContext::new(0).expect("cuda");
        let stream = ctx.default_stream();
        let (nx, ny, nz) = (8usize, 8, 8);
        let n = nx * ny * nz;
        let fft = Fft3d::new(stream.clone(), nx, ny, nz).expect("plan");

        #[allow(clippy::cast_precision_loss)]
        let host: Vec<double2> = (0..n)
            .map(|i| double2 {
                x: (i as f64).sin(),
                y: (i as f64 * 0.5).cos(),
            })
            .collect();
        let mut d_in = stream.clone_htod(&host).expect("htod");
        let mut d_k = stream.alloc_zeros::<double2>(n).expect("alloc k");
        let mut d_out = stream.alloc_zeros::<double2>(n).expect("alloc out");

        fft.forward(&mut d_in, &mut d_k).expect("fwd");
        fft.inverse(&mut d_k, &mut d_out).expect("inv");
        let out = stream.clone_dtoh(&d_out).expect("dtoh");

        #[allow(clippy::cast_precision_loss)]
        let inv_n = 1.0 / n as f64;
        let mut max_abs = 0.0f64;
        for i in 0..n {
            max_abs = max_abs.max((out[i].x * inv_n - host[i].x).abs());
            max_abs = max_abs.max((out[i].y * inv_n - host[i].y).abs());
        }
        eprintln!("cufft roundtrip max_abs={max_abs:.3e}");
        assert!(
            max_abs < 1e-12,
            "FFT^-1 FFT must recover input; {max_abs:.3e}"
        );
    }
}
