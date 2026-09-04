# Fixed-Size Synaptic Memory — Explain the Frontier

**DataForge 2026 · Pathway Track submission**

## One-sentence claim

> A fixed-size `n×d` associative state, written one token at a time with a
> single Hebbian outer-product update `S ← S + φ(k)vᵀ`, can retrieve
> key→value associations from a sequence of unbounded length without
> allocating new memory per token — but once the number of associations
> written exceeds roughly the key dimension `n`, retrieval accuracy degrades
> because overlapping sparse keys interfere with each other in the same
> fixed slots.

This is falsifiable and is tested empirically in `tests/test_linear_memory.py`
(run it — it fails loudly if the claim stops holding) and interactively in
`web/index.html`.

## Selected concept

**Associative Memory & Fast Weights**, with a direct architectural connection
to **Dragon Hatchling (BDH)**'s synaptic-memory mechanism and a research
comparison to **BDH-CQ**'s inference-time memory writes. (Approved-topics
list: "Associative Memory and Fast Weights", cross-referenced against
"Synaptic Plasticity as Short-Term Memory" and "Linear Attention".)

## Intended audience & prerequisites

Data scientists / ML engineers who know what a dot product and a softmax are,
and have heard of attention, but have not necessarily seen the "attention as
a fixed-size recurrent state" (linear attention / fast weights) reframing.
No GPU, no deep learning framework, and no prior BDH knowledge required.

## Learning objectives

After using the artifact, a learner should be able to:
1. State the write/read rule of a Hebbian associative memory from memory.
2. Explain *why* accuracy degrades past a load factor of ~1.0 (key overlap).
3. Explain why BDH's sparse, non-negative activations reduce that
   interference relative to dense activations (reproduced in §4 of the demo).
4. Say, precisely, which part of BDH / BDH-CQ this connects to, and what is
   *not* demonstrated (we do not run BDH itself).

## Architecture — what is real, precomputed, or a toy model

| Component | Status |
|---|---|
| `core/linear_memory.py` | **Live, real computation.** Pure numpy. Implements the Hebbian write/read rule described above. This is an **educational toy model**, not the official BDH implementation. |
| `experiments/run_experiments.py` | **Real computation**, run once, output saved as data (the rules explicitly allow precomputing expensive sweeps). Produces `precomputed/precomputed_results.json`. |
| `web/index.html` | Self-contained interactive artifact. All slider interactions (§§1–3, 5) run the **same math live in JavaScript** on every input — nothing is a scripted animation. §4's reference failure curve is the **precomputed** sweep, with your current live setting overlaid as a red dot so you can see live vs. precomputed side by side. |
| §6 BDH connection | **Real published claims**, cited to primary sources (see `SOURCES.md`), clearly separated from the toy model. |
| `tests/test_linear_memory.py` | Automated tests that assert the central claim actually holds, that state size is genuinely constant in `N`, and that the sparsity/noise mechanisms behave as claimed. |

Nothing in this project claims to run or reproduce BDH or BDH-CQ itself.

## How to run locally

Requirements: Python 3.9+, `numpy`. No GPU, no other dependencies.

```bash
git clone <this-repo-url>
cd bdh_hackathon
pip install -r requirements.txt

# 1. Regenerate the precomputed sweep (optional — already included)
python experiments/run_experiments.py

# 2. Run the tests (verifies the central claim empirically)
python -m pytest tests/ -v
# or, without pytest:
python tests/test_linear_memory.py

# 3. Open the interactive artifact
#    (needs a local static server so the fetch() for the precomputed
#     JSON works — opening the raw file also works via the built-in fallback)
python -m http.server 8000
# then visit http://localhost:8000/web/index.html
```

## Reproduction instructions

`experiments/run_experiments.py` is deterministic given its seeds and will
regenerate `precomputed/precomputed_results.json` byte-for-byte from
`core/linear_memory.py`. To verify the central claim independently:

```bash
python -c "
from core.linear_memory import run_associative_memory
for load in [0.25, 1.0, 3.0, 6.0]:
    n=128; N=int(load*n)
    r = run_associative_memory(n=n, num_classes=32, num_written=N, sparsity=0.05, seed=5)
    print(load, r.accuracy)
"
```

## Limitations (stated plainly)

- The toy model is a **classification-style** associative memory (recover a
  discrete class from a key), not a full sequence model — it demonstrates
  the write/read/interference mechanism, not language modeling.
- Keys are synthetic random sparse vectors, not learned embeddings — this
  isolates the memory mechanism from representation learning, which is a
  deliberate simplification, not an oversight.
- We do not run BDH or BDH-CQ; all BDH-specific numbers are cited from
  primary sources (`SOURCES.md`), never reproduced independently by us.
- The failure curve in §4 is a **5-seed average** at fixed hyperparameters
  (`num_classes=32`); it is illustrative of the trend, not a precision
  benchmark.

## Known issues

- Very large `n` values on the "Memory capacity" slider combined with a high
  load factor (e.g. n=384, load=8×) recompute ~3,000 associations live in
  JS on every slider tick; on low-power mobile devices this can take
  noticeably longer than the "under one second" target. A debounce is not
  yet implemented.

## Credits & licenses

- All code in this repository is original, written for this submission.
- No third-party datasets, weights, or copyrighted assets are used.
- BDH and BDH-CQ claims are attributed to Pathway's published papers and
  repository (see `SOURCES.md`); no BDH code or weights are included or
  redistributed here.

## AI assistance

See `AI_DISCLOSURE.md`.
