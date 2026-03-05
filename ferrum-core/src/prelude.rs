// ferrum-core: Prelude — convenient glob import
//
// Usage: `use ferrum_core::prelude::*;`

// Core array types
pub use crate::array::owned::Array;
pub use crate::array::view::ArrayView;
pub use crate::array::view_mut::ArrayViewMut;
pub use crate::array::arc::ArcArray;
pub use crate::array::cow::CowArray;
pub use crate::array::ArrayFlags;

// Common aliases
pub use crate::array::aliases::{
    Array1, Array2, Array3, Array4, Array5, Array6, ArrayD,
    ArrayView1, ArrayView2, ArrayView3, ArrayViewD,
    ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMutD,
    F32Array1, F32Array2, F64Array1, F64Array2,
};

// Dimension types
pub use crate::dimension::{Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

// Dtype system
pub use crate::dtype::{DType, Element, SliceInfoElem};

// Type casting and promotion
pub use crate::dtype::casting::{AsType, CastKind};
pub use crate::dtype::promotion::{Promoted, PromoteTo};

// Proc macros
pub use ferrum_core_macros::{s, promoted_type, FerrumRecord};

// Error handling
pub use crate::error::{FerrumError, FerrumResult};

// Memory layout
pub use crate::layout::MemoryLayout;

// Buffer interop
pub use crate::buffer::AsRawBuffer;

// Runtime-typed array
pub use crate::dynarray::DynArray;

// Record types
pub use crate::record::FerrumRecord;

// Display configuration
pub use crate::array::display::{set_print_options, get_print_options};
