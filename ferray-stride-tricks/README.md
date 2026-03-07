# ferray-stride-tricks

Low-level stride manipulation for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- `sliding_window_view` — create overlapping window views without copying
- `as_strided` — construct array views with arbitrary strides and shapes
- Overlap detection for safe view construction

Implements `numpy.lib.stride_tricks` for Rust.

## Usage

```rust
use ferray_stride_tricks::sliding_window_view;
use ferray_core::prelude::*;

let a = Array1::<f64>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0])?;
let windows = sliding_window_view(&a, 3)?; // [[1,2,3], [2,3,4], [3,4,5]]
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate with the `stride-tricks` feature.

## License

MIT OR Apache-2.0
