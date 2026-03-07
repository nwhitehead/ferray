# ferray-fft

FFT operations for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- **1D**: `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `ihfft`
- **2D**: `fft2`, `ifft2`, `rfft2`, `irfft2`
- **ND**: `fftn`, `ifftn`, `rfftn`, `irfftn`
- **Frequencies**: `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`
- **Plan caching**: thread-local FFT plan reuse for repeated transforms
- **Normalization**: `Backward`, `Forward`, `Ortho` modes matching NumPy
- Powered by [rustfft](https://crates.io/crates/rustfft) with AVX2+FMA SIMD butterflies

## Usage

```rust
use ferray_fft::{fft, fftfreq};
use ferray_core::prelude::*;

let signal = Array1::<f64>::linspace(0.0, 1.0, 1024)?;
let spectrum = fft(&signal, None, None, None)?;
let freqs = fftfreq(1024, 1.0 / 1024.0)?;
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate with the `fft` feature.

## License

MIT OR Apache-2.0
