// ferrum-core: Type aliases for common array types (REQ-2)

use num_complex::Complex;

use super::owned::Array;
use super::view::ArrayView;
use super::view_mut::ArrayViewMut;
use super::arc::ArcArray;
use super::cow::CowArray;
use crate::dimension::{Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

// ---------------------------------------------------------------------------
// Generic aliases by rank
// ---------------------------------------------------------------------------

/// 1-dimensional owned array.
pub type Array1<T> = Array<T, Ix1>;
/// 2-dimensional owned array.
pub type Array2<T> = Array<T, Ix2>;
/// 3-dimensional owned array.
pub type Array3<T> = Array<T, Ix3>;
/// 4-dimensional owned array.
pub type Array4<T> = Array<T, Ix4>;
/// 5-dimensional owned array.
pub type Array5<T> = Array<T, Ix5>;
/// 6-dimensional owned array.
pub type Array6<T> = Array<T, Ix6>;
/// Dynamic-rank owned array.
pub type ArrayD<T> = Array<T, IxDyn>;

// ---------------------------------------------------------------------------
// Generic view aliases by rank
// ---------------------------------------------------------------------------

/// 1-dimensional immutable view.
pub type ArrayView1<'a, T> = ArrayView<'a, T, Ix1>;
/// 2-dimensional immutable view.
pub type ArrayView2<'a, T> = ArrayView<'a, T, Ix2>;
/// 3-dimensional immutable view.
pub type ArrayView3<'a, T> = ArrayView<'a, T, Ix3>;
/// Dynamic-rank immutable view.
pub type ArrayViewD<'a, T> = ArrayView<'a, T, IxDyn>;

/// 1-dimensional mutable view.
pub type ArrayViewMut1<'a, T> = ArrayViewMut<'a, T, Ix1>;
/// 2-dimensional mutable view.
pub type ArrayViewMut2<'a, T> = ArrayViewMut<'a, T, Ix2>;
/// 3-dimensional mutable view.
pub type ArrayViewMut3<'a, T> = ArrayViewMut<'a, T, Ix3>;
/// Dynamic-rank mutable view.
pub type ArrayViewMutD<'a, T> = ArrayViewMut<'a, T, IxDyn>;

// ---------------------------------------------------------------------------
// ArcArray aliases
// ---------------------------------------------------------------------------

/// 1-dimensional reference-counted array.
pub type ArcArray1<T> = ArcArray<T, Ix1>;
/// 2-dimensional reference-counted array.
pub type ArcArray2<T> = ArcArray<T, Ix2>;
/// Dynamic-rank reference-counted array.
pub type ArcArrayD<T> = ArcArray<T, IxDyn>;

// ---------------------------------------------------------------------------
// CowArray aliases
// ---------------------------------------------------------------------------

/// 1-dimensional copy-on-write array.
pub type CowArray1<'a, T> = CowArray<'a, T, Ix1>;
/// 2-dimensional copy-on-write array.
pub type CowArray2<'a, T> = CowArray<'a, T, Ix2>;
/// Dynamic-rank copy-on-write array.
pub type CowArrayD<'a, T> = CowArray<'a, T, IxDyn>;

// ---------------------------------------------------------------------------
// Float-specialized aliases (REQ-2)
// ---------------------------------------------------------------------------

/// `Array1<f32>`
pub type F32Array1 = Array1<f32>;
/// `Array2<f32>`
pub type F32Array2 = Array2<f32>;
/// `Array3<f32>`
pub type F32Array3 = Array3<f32>;
/// `ArrayD<f32>`
pub type F32ArrayD = ArrayD<f32>;

/// `Array1<f64>`
pub type F64Array1 = Array1<f64>;
/// `Array2<f64>`
pub type F64Array2 = Array2<f64>;
/// `Array3<f64>`
pub type F64Array3 = Array3<f64>;
/// `ArrayD<f64>`
pub type F64ArrayD = ArrayD<f64>;

// ---------------------------------------------------------------------------
// Integer-specialized aliases
// ---------------------------------------------------------------------------

/// `Array1<i32>`
pub type I32Array1 = Array1<i32>;
/// `Array2<i32>`
pub type I32Array2 = Array2<i32>;
/// `Array1<i64>`
pub type I64Array1 = Array1<i64>;
/// `Array2<i64>`
pub type I64Array2 = Array2<i64>;

/// `Array1<u8>`
pub type U8Array1 = Array1<u8>;
/// `Array2<u8>`
pub type U8Array2 = Array2<u8>;

// ---------------------------------------------------------------------------
// Complex-specialized aliases
// ---------------------------------------------------------------------------

/// `Array1<Complex<f32>>`
pub type C32Array1 = Array1<Complex<f32>>;
/// `Array2<Complex<f32>>`
pub type C32Array2 = Array2<Complex<f32>>;
/// `Array1<Complex<f64>>`
pub type C64Array1 = Array1<Complex<f64>>;
/// `Array2<Complex<f64>>`
pub type C64Array2 = Array2<Complex<f64>>;

// ---------------------------------------------------------------------------
// Bool aliases
// ---------------------------------------------------------------------------

/// `Array1<bool>`
pub type BoolArray1 = Array1<bool>;
/// `Array2<bool>`
pub type BoolArray2 = Array2<bool>;
/// `ArrayD<bool>`
pub type BoolArrayD = ArrayD<bool>;
