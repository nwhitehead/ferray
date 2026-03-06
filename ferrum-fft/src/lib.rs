// ferrum-fft: Complete numpy.fft parity with plan caching
//
//! FFT operations for the ferrum numeric computing library.
//!
//! This crate provides the full `numpy.fft` surface:
//!
//! - **Complex FFTs**: [`fft`], [`ifft`], [`fft2`], [`ifft2`], [`fftn`], [`ifftn`]
//! - **Real FFTs**: [`rfft`], [`irfft`], [`rfft2`], [`irfft2`], [`rfftn`], [`irfftn`]
//! - **Hermitian FFTs**: [`hfft`], [`ihfft`]
//! - **Frequency utilities**: [`fftfreq`], [`rfftfreq`]
//! - **Shift utilities**: [`fftshift`], [`ifftshift`]
//! - **Plan caching**: [`FftPlan`] for efficient repeated transforms
//! - **Normalization**: [`FftNorm`] enum matching NumPy's `norm` parameter
//!
//! Internally powered by [`rustfft`](https://crates.io/crates/rustfft) with
//! automatic plan caching for repeated transforms of the same size.

pub mod complex;
pub mod freq;
pub mod hermitian;
mod nd;
pub mod norm;
pub mod plan;
pub mod real;
pub mod shift;

// Re-export public API at crate root for ergonomic access.

// Normalization
pub use norm::FftNorm;

// Plan caching
pub use plan::FftPlan;

// Complex FFTs (REQ-1..REQ-4)
pub use complex::{fft, fft2, fftn, ifft, ifft2, ifftn};

// Real FFTs (REQ-5..REQ-7)
pub use real::{irfft, irfft2, irfftn, rfft, rfft2, rfftn};

// Hermitian FFTs (REQ-8)
pub use hermitian::{hfft, ihfft};

// Frequency utilities (REQ-9, REQ-10)
pub use freq::{fftfreq, rfftfreq};

// Shift utilities (REQ-11)
pub use shift::{fftshift, ifftshift};
