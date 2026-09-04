"""
core/linear_memory.py

REAL COMPUTATIONAL SUBSTRATE for the DataForge 2026 Pathway Track submission.

CENTRAL CLAIM (falsifiable):
    A fixed-size (n x d) associative memory, written one token at a time with
    a single Hebbian outer-product update  S <- S + phi(k) v^T  (no growth in
    parameters, no backward pass), can retrieve key->value associations from
    a sequence of unbounded length -- but as the number of associations N
    written into it grows past roughly the key dimension n, retrieval
    accuracy degrades due to interference between overlapping sparse keys.
    A conventional growing key-value cache, by contrast, keeps perfect
    recall by paying O(N) memory instead of O(1).

WHY THIS IS NOT A GENERIC "MEMORY DEMO":
    The write rule, the non-negative ReLU feature map, and the sparsity level
    (~5% active units) are chosen to mirror the mechanism reported for BDH's
    synaptic memory (Kosowski et al., 2025, "The Dragon Hatchling"): attention
    reformulated as a local Hebbian edge-reweighting rule over sparse,
    non-negative neuron activations, rather than a dense softmax over a
    growing token history. This module is an EDUCATIONAL TOY MODEL of that
    mechanism -- it is NOT the official BDH implementation and makes no claim
    to reproduce BDH's trained weights or benchmark numbers. Those numbers
    (e.g. the Sudoku-Extreme and ARC-AGI-1 results) are cited from the
    primary papers in SOURCES.md and are never invented or reproduced here.

Everything in this file is real, deterministic (given a seed) numpy
computation -- nothing here is a scripted animation.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def make_sparse_key(n: int, sparsity: float, rng: np.random.Generator) -> np.ndarray:
    """
    A single 'concept' key: a non-negative, sparse vector of dimension n.
    This mirrors BDH's reported regime of ~5% active neurons per step
    (sparsity is a *variable* here, not a hard cap, matching the PDF's note
    that activity "varies with predictability rather than a fixed budget" --
    we expose it as a control precisely so the learner can test that).
    """
    raw = rng.normal(size=n)
    k = relu(raw)
    n_active = max(1, int(round(sparsity * n)))
    if n_active < n:
        keep_idx = rng.choice(n, size=n_active, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[keep_idx] = True
        k = k * mask
    norm = np.linalg.norm(k)
    return k / norm if norm > 0 else k


@dataclass
class AssociativeMemoryResult:
    n: int                      # key / neuron dimension (fixed memory capacity axis)
    num_classes: int
    num_written: int            # N, number of associations actually written
    sparsity: float
    noise: float
    accuracy: float             # fraction of associations correctly retrieved
    load_factor: float          # N / n
    state_bytes: int            # size of the fixed-size synaptic matrix S
    kv_cache_bytes: int         # size an equivalent exact KV-cache would need
    keys: np.ndarray = field(repr=False)
    S: np.ndarray = field(repr=False)


def run_associative_memory(
    n: int,
    num_classes: int,
    num_written: int,
    sparsity: float = 0.05,
    noise: float = 0.0,
    seed: int = 0,
    value_dim: Optional[int] = None,
) -> AssociativeMemoryResult:
    """
    Simulates a stream of `num_written` tokens. Each token i is a
    (key_i, class_i) association: class_i is drawn from `num_classes`
    possible "meanings" and key_i is a fresh sparse random vector bound to
    that class for this run (imagine re-encountering a concept with a
    slightly different but overlapping neural signature).

    WRITE (Hebbian / linear-attention accumulation, O(1) memory):
        S <- S + phi(k_i) (one_hot(class_i))^T

    READ (associative recall, given a possibly-noisy cue):
        score = phi(k_i + noise) @ S
        predicted_class = argmax(score)

    Returns accuracy of recovering class_i for every written token, plus a
    literal byte-size comparison against a growing KV-cache baseline that
    would store every (key, class) pair explicitly (perfect recall, O(N)
    memory).
    """
    rng = np.random.default_rng(seed)
    value_dim = value_dim or num_classes

    S = np.zeros((n, value_dim))
    keys = np.zeros((num_written, n))
    classes = rng.integers(0, num_classes, size=num_written)

    for i in range(num_written):
        k = make_sparse_key(n, sparsity, rng)
        keys[i] = k
        v = np.zeros(value_dim)
        v[classes[i] % value_dim] = 1.0
        S += np.outer(k, v)  # <-- the single local Hebbian write, O(1) extra memory

    correct = 0
    for i in range(num_written):
        cue = keys[i] + rng.normal(scale=noise, size=n) if noise > 0 else keys[i]
        cue = relu(cue)
        scores = cue @ S
        pred = int(np.argmax(scores))
        true = int(classes[i] % value_dim)
        correct += int(pred == true)

    accuracy = correct / num_written if num_written > 0 else 1.0

    return AssociativeMemoryResult(
        n=n,
        num_classes=num_classes,
        num_written=num_written,
        sparsity=sparsity,
        noise=noise,
        accuracy=accuracy,
        load_factor=num_written / n,
        state_bytes=S.nbytes,
        kv_cache_bytes=keys.nbytes + classes.nbytes,
        keys=keys,
        S=S,
    )


def run_kv_cache_baseline(num_written: int, key_bytes_per_token: int = 8 * 4) -> dict:
    """
    A conventional attention KV-cache keeps every token's key/value exactly,
    so recall is always perfect (accuracy = 1.0) but memory grows linearly
    with sequence length. This is the O(N) alternative the fixed-size
    associative state (above) is being contrasted with -- see 'Key-Value
    Caching, Limitations, and Alternate Approaches' in the official problem
    statement.
    """
    return {
        "num_written": num_written,
        "accuracy": 1.0,
        "bytes": num_written * key_bytes_per_token,
    }


if __name__ == "__main__":
    res = run_associative_memory(n=256, num_classes=64, num_written=200, sparsity=0.05, seed=1)
    print(f"n={res.n} N={res.num_written} load={res.load_factor:.2f} "
          f"accuracy={res.accuracy:.3f} state_bytes={res.state_bytes} "
          f"kv_cache_bytes={res.kv_cache_bytes}")
