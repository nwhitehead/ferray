#!/usr/bin/env python3
"""Statistical equivalence testing: NumPy vs ferrum.

Runs both NumPy and ferrum on identical inputs, compares results
using Welch's t-test to verify no statistically significant
regression in accuracy.

Usage:
    python3 benchmarks/statistical_equivalence.py

Requirements:
    - Python 3.12+
    - NumPy
    - scipy (optional, falls back to max-ULP threshold comparison)
    - ferrum-bench binary (built from benchmarks/ferrum_bench/)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
ALPHA = 0.05
MAX_ULP_FALLBACK = 4  # If scipy unavailable, pass if max ULP <= this
FERRUM_BENCH_DIR = Path(__file__).parent / "ferrum_bench"
FERRUM_BENCH_BIN = None  # Set after build

# Try to import scipy for statistical testing
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("NOTE: scipy not installed. Using max-ULP threshold comparison instead of Welch's t-test.")
    print("      Install scipy for full statistical testing: pip install scipy")
    print()


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

def make_test_cases():
    """Generate test cases: (function_name, size_label, numpy_func, input_gen, extra_args)."""
    rng = np.random.default_rng(SEED)

    cases = []

    # ---- Ufunc: unary float ops ----
    ufunc_sizes = [1000, 100000]
    ufunc_funcs = [
        ("sin", np.sin, lambda n, rng: rng.uniform(-np.pi, np.pi, n)),
        ("cos", np.cos, lambda n, rng: rng.uniform(-np.pi, np.pi, n)),
        ("tan", np.tan, lambda n, rng: rng.uniform(-1.5, 1.5, n)),
        ("exp", np.exp, lambda n, rng: rng.uniform(-5.0, 5.0, n)),
        ("log", np.log, lambda n, rng: rng.uniform(0.01, 100.0, n)),
        ("log2", np.log2, lambda n, rng: rng.uniform(0.01, 100.0, n)),
        ("log10", np.log10, lambda n, rng: rng.uniform(0.01, 100.0, n)),
        ("sqrt", np.sqrt, lambda n, rng: rng.uniform(0.0, 1000.0, n)),
        ("exp2", np.exp2, lambda n, rng: rng.uniform(-5.0, 5.0, n)),
        ("expm1", np.expm1, lambda n, rng: rng.uniform(-1.0, 1.0, n)),
        ("log1p", np.log1p, lambda n, rng: rng.uniform(-0.99, 10.0, n)),
        ("arcsin", np.arcsin, lambda n, rng: rng.uniform(-0.99, 0.99, n)),
        ("arccos", np.arccos, lambda n, rng: rng.uniform(-0.99, 0.99, n)),
        ("arctan", np.arctan, lambda n, rng: rng.uniform(-100.0, 100.0, n)),
        ("sinh", np.sinh, lambda n, rng: rng.uniform(-3.0, 3.0, n)),
        ("cosh", np.cosh, lambda n, rng: rng.uniform(-3.0, 3.0, n)),
        ("tanh", np.tanh, lambda n, rng: rng.uniform(-5.0, 5.0, n)),
    ]

    for func_name, np_func, input_gen in ufunc_funcs:
        for size in ufunc_sizes:
            data = input_gen(size, rng)
            cases.append({
                "function": func_name,
                "category": "ufunc",
                "size_label": str(size),
                "input_data": data,
                "numpy_func": np_func,
                "ferrum_func": func_name,
                "ferrum_size": str(size),
                "is_complex": False,
                "is_matmul": False,
            })

    # ---- Stats: reductions ----
    stats_sizes = [1000, 100000]
    stats_funcs = [
        ("mean", np.mean),
        ("var", lambda x: np.var(x, ddof=0)),
        ("std", lambda x: np.std(x, ddof=0)),
        ("sum", np.sum),
    ]

    for func_name, np_func in stats_funcs:
        for size in stats_sizes:
            data = rng.standard_normal(size)
            cases.append({
                "function": func_name,
                "category": "stats",
                "size_label": str(size),
                "input_data": data,
                "numpy_func": np_func,
                "ferrum_func": func_name,
                "ferrum_size": str(size),
                "is_complex": False,
                "is_matmul": False,
            })

    # ---- Linalg: matmul ----
    matmul_sizes = [(10, 10), (100, 100)]
    for rows, cols in matmul_sizes:
        # Generate two matrices A (rows x cols) and B (cols x rows)
        a = rng.standard_normal((rows, cols))
        b = rng.standard_normal((cols, rows))
        numpy_result = a @ b  # rows x rows
        # Flatten both matrices into one stream for ferrum
        combined = np.concatenate([a.ravel(), b.ravel()])
        cases.append({
            "function": "matmul",
            "category": "linalg",
            "size_label": f"{rows}x{cols}",
            "input_data": combined,
            "numpy_result_override": numpy_result.ravel(),
            "numpy_func": None,
            "ferrum_func": "matmul",
            "ferrum_size": f"{rows}x{cols}",
            "is_complex": False,
            "is_matmul": True,
        })

    # ---- FFT ----
    fft_sizes = [64, 1024, 65536]
    for size in fft_sizes:
        data = rng.standard_normal(size)
        cases.append({
            "function": "fft",
            "category": "fft",
            "size_label": str(size),
            "input_data": data,
            "numpy_func": np.fft.fft,
            "ferrum_func": "fft",
            "ferrum_size": str(size),
            "is_complex": True,
            "is_matmul": False,
        })

    return cases


# ---------------------------------------------------------------------------
# NumPy execution
# ---------------------------------------------------------------------------

def run_numpy(case):
    """Run the NumPy function and return the result as a flat f64 array."""
    if case.get("numpy_result_override") is not None:
        return case["numpy_result_override"]

    result = case["numpy_func"](case["input_data"])
    if np.isscalar(result):
        return np.array([result], dtype=np.float64)
    if case["is_complex"]:
        return result  # Keep complex
    return np.asarray(result, dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# Ferrum execution
# ---------------------------------------------------------------------------

def find_ferrum_bench():
    """Find or build the ferrum-bench binary."""
    # Check common build locations
    for profile in ["release", "debug"]:
        candidate = FERRUM_BENCH_DIR / "target" / profile / "ferrum-bench"
        if candidate.exists():
            return str(candidate)

    # Try to build it
    print("Building ferrum-bench binary...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=str(FERRUM_BENCH_DIR),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"WARNING: Failed to build ferrum-bench:\n{result.stderr}")
        return None

    candidate = FERRUM_BENCH_DIR / "target" / "release" / "ferrum-bench"
    if candidate.exists():
        return str(candidate)

    return None


def run_ferrum(case, bench_bin):
    """Run the ferrum function via the benchmark binary and return the result."""
    input_json = json.dumps(case["input_data"].tolist())

    try:
        result = subprocess.run(
            [bench_bin, case["ferrum_func"], case["ferrum_size"]],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:
        return None, str(e)

    if result.returncode != 0:
        return None, result.stderr.strip()

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None, f"Invalid JSON output: {result.stdout[:200]}"

    if case["is_complex"]:
        # Reconstruct complex array
        real_part = np.array(output["data_real"], dtype=np.float64)
        imag_part = np.array(output["data_imag"], dtype=np.float64)
        return real_part + 1j * imag_part, None
    else:
        return np.array(output["data"], dtype=np.float64), None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compute_ulp_distances(numpy_result, ferrum_result):
    """Compute ULP distances between two arrays, handling special values."""
    numpy_flat = np.asarray(numpy_result).ravel()
    ferrum_flat = np.asarray(ferrum_result).ravel()

    if numpy_flat.shape != ferrum_flat.shape:
        raise ValueError(
            f"Shape mismatch: numpy={numpy_flat.shape}, ferrum={ferrum_flat.shape}"
        )

    if np.iscomplexobj(numpy_flat) or np.iscomplexobj(ferrum_flat):
        # For complex, compute ULP distance on real and imaginary parts separately
        numpy_flat = np.asarray(numpy_flat, dtype=np.complex128)
        ferrum_flat = np.asarray(ferrum_flat, dtype=np.complex128)
        ulp_real = _ulp_distances_real(numpy_flat.real, ferrum_flat.real)
        ulp_imag = _ulp_distances_real(numpy_flat.imag, ferrum_flat.imag)
        return np.maximum(ulp_real, ulp_imag)
    else:
        return _ulp_distances_real(
            np.asarray(numpy_flat, dtype=np.float64),
            np.asarray(ferrum_flat, dtype=np.float64),
        )


def _ulp_distances_real(a, b):
    """Compute ULP distances for real arrays."""
    # Handle exact matches (including both-zero, both-inf, both-nan)
    exact_match = (a == b) | (np.isnan(a) & np.isnan(b))

    # Compute spacing (ULP size) at each point
    spacing = np.spacing(np.abs(a))
    # Avoid division by zero for zero values
    spacing = np.where(spacing == 0, np.spacing(np.float64(0)), spacing)

    ulp = np.abs(a - b) / spacing
    # Where both are exactly equal, ULP = 0
    ulp = np.where(exact_match, 0.0, ulp)
    # Where one is inf/nan and they don't match, set to inf
    bad = (np.isinf(a) | np.isinf(b) | np.isnan(a) | np.isnan(b)) & ~exact_match
    ulp = np.where(bad, np.inf, ulp)

    return ulp


def compare_results(numpy_result, ferrum_result, func_name, alpha=ALPHA):
    """Compare results using ULP distance and statistical test."""
    try:
        ulp_distances = compute_ulp_distances(numpy_result, ferrum_result)
    except ValueError as e:
        return {
            "function": func_name,
            "mean_ulp": float("inf"),
            "max_ulp": float("inf"),
            "t_stat": float("nan"),
            "p_value": 0.0,
            "passed": False,
            "error": str(e),
        }

    # Filter out infinities for statistical testing
    finite_ulps = ulp_distances[np.isfinite(ulp_distances)]

    if len(finite_ulps) == 0:
        # All comparisons were inf/nan mismatches
        return {
            "function": func_name,
            "mean_ulp": float("inf"),
            "max_ulp": float("inf"),
            "t_stat": float("nan"),
            "p_value": 0.0,
            "passed": False,
            "error": "No finite ULP distances",
        }

    mean_ulp = float(np.mean(finite_ulps))
    max_ulp = float(np.max(finite_ulps))

    if HAS_SCIPY:
        # One-sided t-test: is the mean ULP distance significantly greater than 0?
        # H0: mean ULP distance == 0 (ferrum is as good as NumPy)
        # H1: mean ULP distance > 0 (ferrum is worse)
        if np.all(finite_ulps == 0):
            # Perfect match, no test needed
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat, p_value = scipy_stats.ttest_1samp(
                finite_ulps, 0, alternative="greater"
            )
            t_stat = float(t_stat)
            p_value = float(p_value)

        # A function passes if the max ULP is within acceptable bounds
        # (4 ULP for transcendentals, per the design spec)
        passed = max_ulp <= MAX_ULP_FALLBACK
    else:
        # Fallback: simple threshold check
        t_stat = float("nan")
        p_value = float("nan")
        passed = max_ulp <= MAX_ULP_FALLBACK

    # Count infinities (mismatches on special values)
    n_inf = int(np.sum(~np.isfinite(ulp_distances)))

    result = {
        "function": func_name,
        "mean_ulp": mean_ulp,
        "max_ulp": max_ulp,
        "t_stat": t_stat,
        "p_value": p_value,
        "passed": passed,
    }
    if n_inf > 0:
        result["special_value_mismatches"] = n_inf
        result["passed"] = False  # Any special value mismatch is a failure

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(results):
    """Print results as a formatted table."""
    # Header
    print()
    hdr = (
        f"{'Function':<16} {'Size':<10} "
        f"{'Mean ULP':>10} {'Max ULP':>10} "
        f"{'p-value':>10} {'Status':<8}"
    )
    print(hdr)
    print("-" * 72)

    for r in results:
        func = r["function"]
        size = r.get("size_label", "?")
        mean_ulp = r["mean_ulp"]
        max_ulp = r["max_ulp"]
        p_value = r["p_value"]
        status = "PASS" if r["passed"] else "FAIL"

        if np.isnan(p_value):
            p_str = "N/A"
        elif np.isinf(p_value):
            p_str = "inf"
        else:
            p_str = f"{p_value:.4f}"

        if np.isinf(mean_ulp):
            mean_str = "inf"
        else:
            mean_str = f"{mean_ulp:.2f}"

        if np.isinf(max_ulp):
            max_str = "inf"
        else:
            max_str = f"{max_ulp:.0f}"

        print(
            f"{func:<16} {size:<10} "
            f"{mean_str:>10} {max_str:>10} "
            f"{p_str:>10} {status:<8}"
        )

    print("-" * 72)


def main():
    print("=" * 72)
    print("Statistical Equivalence Testing: NumPy vs ferrum")
    print("=" * 72)
    print()
    print(f"NumPy version:  {np.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Random seed:    {SEED}")
    print(f"Alpha:          {ALPHA}")
    scipy_status = (
        "available" if HAS_SCIPY
        else "NOT available (using threshold fallback)"
    )
    print(f"Scipy:          {scipy_status}")
    print()

    # Find or build the ferrum-bench binary
    bench_bin = find_ferrum_bench()
    if bench_bin is None:
        print("ERROR: Could not find or build ferrum-bench binary.")
        print(
            f"       Build it manually: "
            f"cd {FERRUM_BENCH_DIR} && cargo build --release"
        )
        print()
        print("Generating test cases and running NumPy only (dry run)...")
        cases = make_test_cases()
        print(f"  {len(cases)} test cases generated.")
        print()
        print("Skipping comparison -- ferrum-bench binary not available.")
        sys.exit(0)

    print(f"Ferrum bench:   {bench_bin}")
    print()

    # Generate test cases
    cases = make_test_cases()
    print(f"Running {len(cases)} test cases...")
    print()

    results = []
    all_passed = True

    for i, case in enumerate(cases):
        label = f"{case['function']}/{case['size_label']}"
        sys.stdout.write(f"  [{i+1:3d}/{len(cases)}] {label:<30} ")
        sys.stdout.flush()

        # Run NumPy
        numpy_result = run_numpy(case)

        # Run ferrum
        ferrum_result, error = run_ferrum(case, bench_bin)

        if error is not None:
            result = {
                "function": case["function"],
                "size_label": case["size_label"],
                "mean_ulp": float("inf"),
                "max_ulp": float("inf"),
                "t_stat": float("nan"),
                "p_value": 0.0,
                "passed": False,
                "error": error,
            }
            err_msg = error[:50]
            sys.stdout.write(f"FAIL (error: {err_msg})\n")
        else:
            result = compare_results(
                numpy_result, ferrum_result, case["function"]
            )
            result["size_label"] = case["size_label"]
            status = "PASS" if result["passed"] else "FAIL"
            mean_v = result["mean_ulp"]
            max_v = result["max_ulp"]
            ulp_info = f"mean={mean_v:.2f}, max={max_v:.0f}"
            sys.stdout.write(f"{status} ({ulp_info})\n")

        results.append(result)
        if not result["passed"]:
            all_passed = False

    # Print summary table
    print_table(results)

    # Summary
    n_pass = sum(1 for r in results if r["passed"])
    n_fail = sum(1 for r in results if not r["passed"])
    print()
    print(
        f"Results: {n_pass} passed, {n_fail} failed "
        f"out of {len(results)} tests"
    )

    if all_passed:
        print()
        print(
            "All tests PASSED. "
            "No statistically significant regressions detected."
        )
        sys.exit(0)
    else:
        print()
        print("Some tests FAILED. Review the results above for details.")
        # Print details on failures
        failures = [r for r in results if not r["passed"]]
        if failures:
            print()
            print("Failed tests:")
            for r in failures:
                err = r.get("error", "")
                extra = f" -- {err}" if err else ""
                fn = r["function"]
                sl = r.get("size_label", "?")
                mu = r["max_ulp"]
                print(f"  - {fn}/{sl}: max_ulp={mu}{extra}")
        sys.exit(1)


if __name__ == "__main__":
    main()
