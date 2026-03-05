// ferrum-core: N-dimensional array type and foundational primitives
//
// This is the root crate for the ferrum workspace. It provides NdArray<T, D>,
// the full ownership model (Array, ArrayView, ArrayViewMut, ArcArray, CowArray),
// the Element trait, DType system, FerrumError, and array introspection/iteration.
//
// ndarray is used internally but is NOT part of the public API.

pub mod array;
pub mod buffer;
pub mod dimension;
pub mod dtype;
pub mod dynarray;
pub mod error;
pub mod layout;
pub mod prelude;
pub mod record;

// Re-export key types at crate root for ergonomics
pub use array::owned::Array;
pub use array::view::ArrayView;
pub use array::view_mut::ArrayViewMut;
pub use array::arc::ArcArray;
pub use array::cow::CowArray;
pub use array::ArrayFlags;
pub use array::aliases;
pub use array::display::{set_print_options, get_print_options};

pub use dimension::{Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

pub use dtype::{DType, Element};

pub use error::{FerrumError, FerrumResult};

pub use layout::MemoryLayout;

pub use buffer::AsRawBuffer;

pub use dynarray::DynArray;

pub use record::{FerrumRecord, FieldDescriptor};
