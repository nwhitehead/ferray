// ferrum: Prelude — `use ferrum::prelude::*` covers 95% of use cases (REQ-3)

// Core array types
pub use ferrum_core::ArcArray;
pub use ferrum_core::Array;
pub use ferrum_core::ArrayFlags;
pub use ferrum_core::ArrayView;
pub use ferrum_core::ArrayViewMut;
pub use ferrum_core::CowArray;

// Common aliases
pub use ferrum_core::aliases::{
    Array1, Array2, Array3, Array4, Array5, Array6, ArrayD, ArrayView1, ArrayView2, ArrayView3,
    ArrayViewD, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMutD, F32Array1, F32Array2,
    F64Array1, F64Array2,
};

// Dimension types
pub use ferrum_core::dimension::{Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

// Dtype system
pub use ferrum_core::{DType, Element, SliceInfoElem};

// Error handling
pub use ferrum_core::{FerrumError, FerrumResult};

// Macros
pub use ferrum_core::{promoted_type, s};

// Memory layout
pub use ferrum_core::MemoryLayout;

// Runtime-typed array
pub use ferrum_core::DynArray;

// Array creation functions
pub use ferrum_core::creation::{
    arange, array, asarray, eye, fromiter, full, full_like, geomspace, identity, linspace,
    logspace, ones, ones_like, zeros, zeros_like,
};

// Common math functions (ufuncs)
pub use ferrum_ufunc::{
    absolute,
    // Arithmetic
    add,
    all,
    allclose,
    any,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    around,
    array_equal,
    ceil,
    clip,
    cos,
    cosh,
    cumprod,
    cumsum,
    deg2rad,
    degrees,
    diff,
    divide,
    // Comparison
    equal,
    // Exp/log
    exp,
    exp2,
    expm1,
    fix,
    floor,
    fmod,
    gradient,
    greater,
    greater_equal,
    hypot,
    isclose,
    isfinite,
    isinf,
    // Float
    isnan,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    // Logical
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    mod_,
    multiply,
    nan_to_num,
    negative,
    not_equal,
    power,
    rad2deg,
    radians,
    remainder,
    rint,
    // Rounding
    round,
    sign,
    // Trig
    sin,
    // Special
    sinc,
    sinh,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    trunc,
};

// Stats (reductions)
pub use ferrum_stats::{
    argmax, argmin, max, mean, median, min, nanmax, nanmean, nanmin, nansum, prod, std_, sum, var,
};
