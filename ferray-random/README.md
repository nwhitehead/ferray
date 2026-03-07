# ferray-random

Random number generation and distributions for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- **Generator API**: `Generator` type with pluggable `BitGenerator` backends
- **BitGenerators**: PCG64, Philox, SFC64, MT19937
- **30+ distributions**: Normal, Uniform, Exponential, Poisson, Binomial, Gamma, Beta, Chi-squared, Student-t, etc.
- **Permutations**: `shuffle`, `permutation`, `choice` (with/without replacement)
- **Array generation**: `random`, `standard_normal`, `integers`, etc.

## Usage

```rust
use ferray_random::{Generator, PCG64};

let mut rng = Generator::new(PCG64::seed(42));
let samples = rng.standard_normal([1000])?;
let uniform = rng.random([3, 4])?;
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate with the `random` feature.

## License

MIT OR Apache-2.0
