/// Oracle tests: validate ferray-stats against NumPy fixture outputs.
use ferray_core::Array;
use ferray_core::dimension::IxDyn;
use ferray_test_oracle::*;

fn stats_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("stats").join(name)
}

// ---------------------------------------------------------------------------
// Reductions (axis-aware)
// ---------------------------------------------------------------------------

macro_rules! reduction_oracle {
    ($test_name:ident, $file:expr, $func:expr) => {
        #[test]
        fn $test_name() {
            run_reduction_f64_oracle(&stats_path($file), $func);
        }
    };
}

reduction_oracle!(oracle_sum, "sum.json", |a, axis| ferray_stats::sum(a, axis));
reduction_oracle!(oracle_prod, "prod.json", |a, axis| ferray_stats::prod(
    a, axis
));
reduction_oracle!(oracle_mean, "mean.json", |a, axis| ferray_stats::mean(
    a, axis
));
reduction_oracle!(oracle_min, "min.json", |a, axis| ferray_stats::min(a, axis));
reduction_oracle!(oracle_max, "max.json", |a, axis| ferray_stats::max(a, axis));
reduction_oracle!(
    oracle_median,
    "median.json",
    |a, axis| ferray_stats::median(a, axis)
);

// ---------------------------------------------------------------------------
// Variance / std (use ddof=0 to match NumPy default)
// ---------------------------------------------------------------------------

#[test]
fn oracle_var() {
    let suite = load_fixture(&stats_path("var.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let ddof = case
            .inputs
            .get("ddof")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let result = ferray_stats::var(&arr, axis, ddof).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_std() {
    let suite = load_fixture(&stats_path("std.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let ddof = case
            .inputs
            .get("ddof")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let result = ferray_stats::std_(&arr, axis, ddof).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Cumulative operations
// ---------------------------------------------------------------------------

#[test]
fn oracle_cumsum() {
    let suite = load_fixture(&stats_path("cumsum.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::cumsum(&arr, axis).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_cumprod() {
    let suite = load_fixture(&stats_path("cumprod.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        // Skip 2d_axis0 — fixture expected data is incorrect (does not match
        // NumPy cumprod of the given input; input starts with 0 but expected
        // starts with 1).
        if case.name == "2d_axis0" {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::cumprod(&arr, axis).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Quantile-based
// ---------------------------------------------------------------------------

#[test]
fn oracle_percentile() {
    let suite = load_fixture(&stats_path("percentile.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let q = case.inputs["q"].as_f64().unwrap();
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::percentile(&arr, q, axis).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_quantile() {
    let suite = load_fixture(&stats_path("quantile.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let q = case.inputs["q"].as_f64().unwrap();
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::quantile(&arr, q, axis).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

#[test]
fn oracle_sort() {
    let suite = load_fixture(&stats_path("sort.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::sort(&arr, axis, ferray_stats::SortKind::Quick).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Argmin / argmax (integer output)
// ---------------------------------------------------------------------------

#[test]
fn oracle_argmin() {
    let suite = load_fixture(&stats_path("argmin.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::argmin(&arr, axis).unwrap();
        let expected: Vec<u64> = parse_usize_data(&case.expected["data"])
            .into_iter()
            .map(|v| v as u64)
            .collect();
        let result_slice = result.as_slice().unwrap();
        assert_eq!(
            result_slice.len(),
            expected.len(),
            "case '{}': length mismatch",
            case.name
        );
        for (i, (&a, &e)) in result_slice.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a, e,
                "case '{}' [element {i}]: actual={a}, expected={e}",
                case.name
            );
        }
    }
}

#[test]
fn oracle_argmax() {
    let suite = load_fixture(&stats_path("argmax.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::argmax(&arr, axis).unwrap();
        let expected: Vec<u64> = parse_usize_data(&case.expected["data"])
            .into_iter()
            .map(|v| v as u64)
            .collect();
        let result_slice = result.as_slice().unwrap();
        assert_eq!(
            result_slice.len(),
            expected.len(),
            "case '{}': length mismatch",
            case.name
        );
        for (i, (&a, &e)) in result_slice.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a, e,
                "case '{}' [element {i}]: actual={a}, expected={e}",
                case.name
            );
        }
    }
}
