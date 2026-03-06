//! Window functions and functional programming utilities for ferrum.
//!
//! This crate provides NumPy-equivalent window functions for signal processing
//! and spectral analysis (`bartlett`, `blackman`, `hamming`, `hanning`, `kaiser`),
//! along with functional programming utilities (`vectorize`, `piecewise`,
//! `apply_along_axis`, `apply_over_axes`).
//!
//! All window functions return `Array1<f64>` and match NumPy's output to
//! high precision.

pub mod functional;
pub mod windows;

// Re-export window functions at crate root for convenience
pub use windows::{bartlett, blackman, hamming, hanning, kaiser};

// Re-export functional utilities at crate root
pub use functional::{
    apply_along_axis, apply_over_axes, piecewise, sum_axis_keepdims, vectorize, vectorize_nd,
};
