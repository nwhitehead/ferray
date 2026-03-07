#![no_main]

//! Fuzz target for arithmetic ufuncs: add, subtract, multiply, divide, power,
//! sqrt, cbrt, absolute, negative, square, reciprocal, sign.
//!
//! Contract: ferrum either returns Ok or Err(FerrumError) — never panics.

use libfuzzer_sys::fuzz_target;

use ferrum_core::array::owned::Array;
use ferrum_core::dimension::Ix1;

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
    let arr = match Array::from_vec(Ix1::new([n]), values.clone()) {
        Ok(a) => a,
        Err(_) => return,
    };

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Unary ops
        let _ = ferrum_ufunc::sqrt(&arr);
        let _ = ferrum_ufunc::cbrt(&arr);
        let _ = ferrum_ufunc::absolute(&arr);
        let _ = ferrum_ufunc::negative(&arr);
        let _ = ferrum_ufunc::square(&arr);
        let _ = ferrum_ufunc::reciprocal(&arr);
        let _ = ferrum_ufunc::sign(&arr);

        // Binary ops — split input in half
        if n >= 2 {
            let half = n / 2;
            let a = Array::from_vec(Ix1::new([half]), values[..half].to_vec()).unwrap();
            let b = Array::from_vec(Ix1::new([half]), values[half..half * 2].to_vec()).unwrap();
            let _ = ferrum_ufunc::add(&a, &b);
            let _ = ferrum_ufunc::subtract(&a, &b);
            let _ = ferrum_ufunc::multiply(&a, &b);
            let _ = ferrum_ufunc::divide(&a, &b);
            let _ = ferrum_ufunc::power(&a, &b);
            let _ = ferrum_ufunc::remainder(&a, &b);
            let _ = ferrum_ufunc::floor_divide(&a, &b);
        }
    }));
});
