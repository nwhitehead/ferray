// ferray-linalg: Decomposition module
//
// Matrix decompositions: Cholesky, QR, SVD, LU, Eigen.

/// Cholesky decomposition.
pub mod cholesky;
/// Eigendecomposition for general and symmetric matrices.
pub mod eigen;
/// LU decomposition with partial pivoting.
pub mod lu;
/// QR decomposition with reduced and complete modes.
pub mod qr;
/// Singular Value Decomposition.
pub mod svd;

pub use cholesky::cholesky;
pub use eigen::{eig, eigh, eigvals, eigvalsh};
pub use lu::lu;
pub use qr::{QrMode, qr};
pub use svd::svd;
