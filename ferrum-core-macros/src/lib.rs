// ferrum-core-macros: Procedural macros for ferrum-core
//
// Agent 1d will implement the following macros here:
// - #[derive(FerrumRecord)] — generates FerrumRecord trait impl for #[repr(C)] structs
// - s![] — NumPy-style slice indexing macro
// - promoted_type!() — compile-time type promotion macro
//
// The support types that these macros generate impls for are defined in
// ferrum-core/src/record.rs (FerrumRecord trait, FieldDescriptor).
//
// For now this crate is intentionally empty so that ferrum-core compiles.

extern crate proc_macro;
