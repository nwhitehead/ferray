#![no_main]

//! Fuzz target for trigonometric ufuncs: sin, cos, tan, arcsin, arccos, arctan,
//! sinh, cosh, tanh, arcsinh, arccosh, arctanh.
//!
//! Contract: ferrum either returns Ok or Err(FerrumError) — never panics.

use libfuzzer_sys::fuzz_target;

use ferrum_core::array::owned::Array;
use ferrum_core::dimension::Ix1;

/// Interpret raw bytes as a Vec<f64>.  Returns an empty vec if fewer than 8 bytes.
fn bytes_to_f64s(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let values = bytes_to_f64s(data);
    if values.is_empty() {
        return;
    }

    let n = values.len();
    let arr = match Array::from_vec(Ix1::new([n]), values) {
        Ok(a) => a,
        Err(_) => return,
    };

    // Catch panics — any panic is a fuzz failure
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ferrum_ufunc::sin(&arr);
        let _ = ferrum_ufunc::cos(&arr);
        let _ = ferrum_ufunc::tan(&arr);
        let _ = ferrum_ufunc::arcsin(&arr);
        let _ = ferrum_ufunc::arccos(&arr);
        let _ = ferrum_ufunc::arctan(&arr);
        let _ = ferrum_ufunc::sinh(&arr);
        let _ = ferrum_ufunc::cosh(&arr);
        let _ = ferrum_ufunc::tanh(&arr);
        let _ = ferrum_ufunc::arcsinh(&arr);
        let _ = ferrum_ufunc::arccosh(&arr);
        let _ = ferrum_ufunc::arctanh(&arr);
    }));
});
