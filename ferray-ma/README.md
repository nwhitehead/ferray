# ferray-ma

Masked arrays with mask propagation for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- `MaskedArray<T, D>` — array with boolean mask for missing/invalid data
- **Mask propagation**: arithmetic and ufuncs automatically propagate masks
- **Masked reductions**: `masked_mean`, `masked_sum`, `masked_var`, `masked_std`, etc.
- **Masked sorting**: operations that respect the mask

Implements `numpy.ma` for Rust.

## Usage

```rust
use ferray_ma::MaskedArray;
use ferray_core::prelude::*;

let data = Array1::<f64>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0])?;
let mask = vec![false, false, true, false]; // mask out index 2
let ma = MaskedArray::new(data, mask)?;
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate with the `ma` feature (enabled by default).

## License

MIT OR Apache-2.0
