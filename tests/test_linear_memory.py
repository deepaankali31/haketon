"""
tests/test_linear_memory.py

Run with:  python -m pytest tests/ -v
(or, with no pytest installed:  python tests/test_linear_memory.py)

These tests verify:
  1. Under light load, retrieval is (near) perfect.
  2. The central claim: accuracy degrades monotonically-ish as load factor
     (num_written / n) increases well past 1.0.
  3. State size truly stays constant while sequence length grows (the O(1)
     vs O(N) memory claim), verified on actual byte counts.
  4. Higher sparsity (fewer active units per key) reduces interference at a
     fixed high load, matching BDH's rationale for sparse activations.
  5. Read-time noise degrades accuracy (sanity check on the read path).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.linear_memory import run_associative_memory, run_kv_cache_baseline  # noqa: E402


def test_light_load_near_perfect_recall():
    res = run_associative_memory(n=256, num_classes=32, num_written=25, sparsity=0.05, seed=0)
    assert res.accuracy >= 0.95, f"expected near-perfect recall under light load, got {res.accuracy}"


def test_central_claim_accuracy_degrades_with_load():
    n = 128
    loads = [0.25, 1.0, 3.0, 6.0]
    accs = []
    for load in loads:
        N = int(load * n)
        res = run_associative_memory(n=n, num_classes=32, num_written=N, sparsity=0.05, seed=5)
        accs.append(res.accuracy)
    assert accs[0] > accs[-1], f"expected accuracy to drop as load increases, got {accs}"
    assert accs[0] >= 0.95
    assert accs[-1] <= 0.6, f"expected substantial degradation at 6x load, got {accs[-1]}"


def test_state_size_is_constant_kv_cache_grows():
    n, num_classes = 128, 32
    sizes = []
    for N in [10, 100, 1000]:
        res = run_associative_memory(n=n, num_classes=num_classes, num_written=N, sparsity=0.05, seed=1)
        sizes.append(res.state_bytes)
    assert len(set(sizes)) == 1, f"fixed state size should not depend on N, got {sizes}"

    kv_sizes = [run_kv_cache_baseline(N)["bytes"] for N in [10, 100, 1000]]
    assert kv_sizes[0] < kv_sizes[1] < kv_sizes[2], "KV-cache baseline should grow with N"


def test_sparsity_reduces_interference_at_high_load():
    n, num_classes, N = 128, 32, 256
    acc_sparse = run_associative_memory(n=n, num_classes=num_classes, num_written=N, sparsity=0.05, seed=3).accuracy
    acc_dense = run_associative_memory(n=n, num_classes=num_classes, num_written=N, sparsity=1.0, seed=3).accuracy
    assert acc_sparse > acc_dense, (
        f"sparse keys (5% active) should interfere less than dense keys (100% active) at the same load, "
        f"got sparse={acc_sparse} dense={acc_dense}"
    )


def test_read_noise_degrades_accuracy():
    n, num_classes, N = 128, 32, 64
    acc_clean = run_associative_memory(n=n, num_classes=num_classes, num_written=N, sparsity=0.05, noise=0.0, seed=9).accuracy
    acc_noisy = run_associative_memory(n=n, num_classes=num_classes, num_written=N, sparsity=0.05, noise=0.8, seed=9).accuracy
    assert acc_clean >= acc_noisy, f"noisy retrieval cue should not outperform clean cue, got clean={acc_clean} noisy={acc_noisy}"


if __name__ == "__main__":
    tests = [
        test_light_load_near_perfect_recall,
        test_central_claim_accuracy_degrades_with_load,
        test_state_size_is_constant_kv_cache_grows,
        test_sparsity_reduces_interference_at_high_load,
        test_read_noise_degrades_accuracy,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print("\nAll tests passed.")
