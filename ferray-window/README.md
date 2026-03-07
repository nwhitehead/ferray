# ferray-window

Window functions and functional programming utilities for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- **Window functions**: `hann`, `hamming`, `blackman`, `kaiser`, `bartlett`, `gaussian`, `tukey`, etc.
- **Functional**: `vectorize`, `piecewise`, `apply_along_axis`, `apply_over_axes`

## Usage

```rust
use ferray_window::{hann, hamming};

let w = hann(256, true)?;
let h = hamming(256, true)?;
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate with the `window` feature (enabled by default).

## License

MIT OR Apache-2.0
