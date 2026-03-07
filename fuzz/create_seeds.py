#!/usr/bin/env python3
"""Generate seed corpus files for ferrum fuzz targets.

Each seed file contains raw bytes that the fuzz targets interpret as f64 values.
The seeds cover important edge cases: zeros, NaN, infinities, subnormals,
negative zero, and mixed values.
"""

import struct
import os

def main():
    corpus_dir = os.path.join(os.path.dirname(__file__), "corpus", "common")
    os.makedirs(corpus_dir, exist_ok=True)

    seeds = {
        "all_zero": struct.pack("<d", 0.0) * 4,
        "all_nan": struct.pack("<d", float("nan")) * 4,
        "single": struct.pack("<d", 1.0),
        "max": struct.pack("<d", 1.7976931348623157e+308),
        "min_pos": struct.pack("<d", 5e-324),
        "neg_zero": struct.pack("<d", -0.0),
        "inf": struct.pack("<d", float("inf")),
        "neg_inf": struct.pack("<d", float("-inf")),
        "mixed": struct.pack("<4d", 0.0, float("nan"), float("inf"), -1.0),
        "small_ints": struct.pack("<4d", 1.0, 2.0, 3.0, 4.0),
        "negatives": struct.pack("<4d", -1.0, -2.0, -3.0, -4.0),
        "unit_range": struct.pack("<4d", 0.0, 0.25, 0.5, 0.75),
        "subnormal": struct.pack("<d", 2.2250738585072014e-308),
        "large_array": struct.pack("<8d", *[float(i) for i in range(8)]),
        "pi_values": struct.pack("<4d", 3.141592653589793, 1.5707963267948966,
                                  0.7853981633974483, 6.283185307179586),
        "trig_domain": struct.pack("<4d", -1.0, -0.5, 0.5, 1.0),
        "log_domain": struct.pack("<4d", 0.001, 1.0, 2.718281828459045, 100.0),
        "matrix_2x2": struct.pack("<4d", 1.0, 0.0, 0.0, 1.0),
        "matrix_singular": struct.pack("<4d", 1.0, 2.0, 2.0, 4.0),
        "complex_pairs": struct.pack("<4d", 1.0, 0.0, 0.0, 1.0),
    }

    for name, data in seeds.items():
        path = os.path.join(corpus_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"  created {path} ({len(data)} bytes)")

    print(f"\nGenerated {len(seeds)} seed files in {corpus_dir}")


if __name__ == "__main__":
    main()
