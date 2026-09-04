# One-Page Concept Summary: Associative Memory as Fixed-Size Synaptic State

**Claim.** A fixed-size `n×d` matrix can serve as a sequence model's entire
memory of everything it has read, updated with one additive outer-product
write per token (`S ← S + φ(k)vᵀ`), retrieved with one dot product per
query, and never grown — at the cost that retrieval degrades once the
number of stored associations exceeds roughly `n`, because overlapping keys
interfere in the same fixed slots.

**The design pressure.** Standard (softmax) attention keeps every past
token's key and value in an explicit cache and compares a new query against
all of them. This gives exact recall, but memory and per-step compute grow
linearly with sequence length — the "growing key-value cache" problem.
Linear attention removes the per-pair comparison by replacing the softmax
with a feature map `φ` such that `softmax(QKᵀ)V ≈ φ(Q)(φ(K)ᵀV)`; the term
`φ(K)ᵀV` can then be accumulated incrementally into one matrix as tokens
arrive (Katharopoulos et al., 2020), and reading becomes `φ(q)·S` instead of
a scan over history (Schlag et al., 2021, frame this explicitly as a
"fast weight" write). The mechanism has no built-in forgetting: as more
associations are packed into the same `n` dimensions, their keys start to
overlap, and the accumulated matrix increasingly returns a blend rather than
a clean lookup. This project isolates that trade-off — constant memory
against a capacity-dependent failure mode — as a controllable, measurable
phenomenon, rather than stating it as prose.

**What differs from a Transformer.** A Transformer's KV-cache is exact but
`O(N)` in memory; this associative state is `O(1)` in memory but not exact
past a load factor of roughly one association per key dimension. The
interactive artifact in this submission measures accuracy as a function of
that load factor directly (own experiment, five-seed average,
`experiments/run_experiments.py`), and separately measures byte counts for
both approaches — confirming the trade-off is real, not just plausible.

**Representative systems.** (1) *Linear attention / fast weights*
(Katharopoulos et al. 2020; Schlag et al. 2021) — the general mechanism this
project implements. (2) *Dragon Hatchling (BDH)* (Kosowski et al., Pathway,
2025) — a brain-inspired architecture whose attention is derived from a
local, Hebbian-like edge-reweighting rule over a scale-free graph of neuron
particles rather than dense matrix attention; its GPU-friendly formulation,
BDH-GPU, is explicitly built from ReLU-low-rank transformations with linear
attention, and BDH's reported activations are sparse and non-negative
(~5% active), which is the property this project's own `sparsity_sweep`
experiment shows measurably reduces interference at fixed load — a concrete,
reproduced explanation for why BDH's authors made that design choice, not
just a stated analogy. (3) *BDH-CQ* (Pathway, 2026), a later system in the
same family: task demonstrations update recurrent memory additively at
inference time (no gradient step, no evaluation-task demonstrations seen in
training), and the model reasons in a continuous latent workspace rather
than a written chain of thought. Reported result: a 150M-parameter
configuration reaches 29.5% pass@2 on the public ARC-AGI-1 benchmark at an
estimated $0.0007/task — a **developer-reported benchmark result** that
Pathway states was independently reproduced by Bielik AI and NYU; this
project did not verify that reproduction first-hand and labels it as
Pathway's own reporting rather than as independent evidence it collected.

**Where the concept has demonstrated an advantage, and where it has not.**
Constant memory and O(1) per-token update cost are demonstrated
architectural properties, not claims requiring benchmarks. Accuracy under
load is the trade-off, and it is real: this project's own tests
(`tests/test_linear_memory.py`) show accuracy falling from ≥0.95 at low load
to ≤0.6 at 6× load, at fixed sparsity. Where BDH itself sits on that curve
in practice — at production scale, on language rather than this project's
synthetic classification task — is **not established by this project** and
is not claimed to be; BDH's reported evaluations (e.g. Sudoku-Extreme,
ARC-AGI-1 via BDH-CQ) are benchmark results reported by the developer, not
independent third-party deployments, and this summary does not conflate the
two.

**Open question.** The clearest unresolved issue, both in this toy model and
in BDH's own framing, is *consolidation*: how (or whether) useful information
accumulated in fast, overwritable state should be moved into durable, slow
parameters before interference erases it — a problem the approved-topics
list names directly under "Parametric Memory in LLMs" and one this project's
fixed-capacity memory demonstrates but does not solve.
