// ferrum-polynomial: Poly, ToPowerBasis, FromPowerBasis traits (REQ-7, REQ-8, REQ-9, REQ-10, REQ-11)

use ferrum_core::error::FerrumError;
use num_complex::Complex;

/// Common trait for polynomial types in all bases.
///
/// Every polynomial class (power, Chebyshev, Legendre, Laguerre, Hermite,
/// HermiteE) implements this trait, providing evaluation, calculus,
/// arithmetic, root-finding, and fitting.
pub trait Poly: Clone + Sized {
    /// Evaluate the polynomial at a single point.
    ///
    /// # Errors
    /// Returns `FerrumError::InvalidValue` if computation fails.
    fn eval(&self, x: f64) -> Result<f64, FerrumError>;

    /// Evaluate the polynomial at multiple points.
    ///
    /// # Errors
    /// Returns `FerrumError::InvalidValue` if computation fails.
    fn eval_many(&self, x: &[f64]) -> Result<Vec<f64>, FerrumError> {
        x.iter().map(|&xi| self.eval(xi)).collect()
    }

    /// Differentiate the polynomial `m` times.
    ///
    /// # Errors
    /// Returns `FerrumError::InvalidValue` if `m` is invalid.
    fn deriv(&self, m: usize) -> Result<Self, FerrumError>;

    /// Integrate the polynomial `m` times with integration constants `k`.
    ///
    /// `k` should have exactly `m` elements. If fewer are provided,
    /// zeros are used for the missing constants.
    ///
    /// # Errors
    /// Returns `FerrumError::InvalidValue` if computation fails.
    fn integ(&self, m: usize, k: &[f64]) -> Result<Self, FerrumError>;

    /// Return the roots of the polynomial as complex values.
    ///
    /// Uses companion matrix eigenvalues via `ferrum-linalg`.
    ///
    /// # Errors
    /// Returns an error if root-finding fails or linalg is unavailable.
    fn roots(&self) -> Result<Vec<Complex<f64>>, FerrumError>;

    /// Return the degree of the polynomial.
    fn degree(&self) -> usize;

    /// Return the coefficients of this polynomial.
    fn coeffs(&self) -> &[f64];

    /// Remove trailing coefficients below tolerance.
    ///
    /// # Errors
    /// Returns `FerrumError::InvalidValue` if `tol` is negative.
    fn trim(&self, tol: f64) -> Result<Self, FerrumError>;

    /// Truncate the polynomial to the given number of terms.
    ///
    /// # Errors
    /// Returns `FerrumError::InvalidValue` if `size` is zero.
    fn truncate(&self, size: usize) -> Result<Self, FerrumError>;

    /// Add two polynomials of the same basis.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    fn add(&self, other: &Self) -> Result<Self, FerrumError>;

    /// Subtract another polynomial of the same basis.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    fn sub(&self, other: &Self) -> Result<Self, FerrumError>;

    /// Multiply two polynomials of the same basis.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    fn mul(&self, other: &Self) -> Result<Self, FerrumError>;

    /// Raise the polynomial to the `n`-th power.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    fn pow(&self, n: usize) -> Result<Self, FerrumError>;

    /// Divide this polynomial by another, returning (quotient, remainder).
    ///
    /// # Errors
    /// Returns `FerrumError::InvalidValue` if the divisor is zero.
    fn divmod(&self, other: &Self) -> Result<(Self, Self), FerrumError>;

    /// Least-squares polynomial fit of degree `deg` to the given data.
    ///
    /// # Errors
    /// Returns an error if the fit fails (e.g., singular matrix).
    fn fit(x: &[f64], y: &[f64], deg: usize) -> Result<Self, FerrumError>;

    /// Weighted least-squares polynomial fit.
    ///
    /// # Errors
    /// Returns an error if the fit fails.
    fn fit_weighted(x: &[f64], y: &[f64], deg: usize, w: &[f64]) -> Result<Self, FerrumError>;

    /// Construct the polynomial from the given coefficients.
    fn from_coeffs(coeffs: &[f64]) -> Self;
}

/// Trait for converting a polynomial to its power basis representation.
pub trait ToPowerBasis: Poly {
    /// Convert this polynomial to power basis coefficients.
    ///
    /// # Errors
    /// Returns an error if conversion fails.
    fn to_power_basis(&self) -> Result<Vec<f64>, FerrumError>;
}

/// Trait for constructing a polynomial from power basis coefficients.
pub trait FromPowerBasis: Poly {
    /// Create this polynomial type from power basis coefficients.
    ///
    /// # Errors
    /// Returns an error if conversion fails.
    fn from_power_basis(coeffs: &[f64]) -> Result<Self, FerrumError>;
}

/// Extension trait providing `.convert::<TargetType>()` for basis conversion.
///
/// This uses power basis as a canonical pivot: source -> power -> target.
/// This avoids the N^2 pairwise conversion problem and the coherence
/// issues with blanket `From` impls.
pub trait ConvertBasis: ToPowerBasis {
    /// Convert this polynomial to a different basis type.
    ///
    /// # Errors
    /// Returns an error if conversion fails.
    fn convert<T: FromPowerBasis>(&self) -> Result<T, FerrumError> {
        let power_coeffs = self.to_power_basis()?;
        T::from_power_basis(&power_coeffs)
    }
}

// Blanket implementation: anything that implements ToPowerBasis automatically
// gets ConvertBasis.
impl<P: ToPowerBasis> ConvertBasis for P {}
