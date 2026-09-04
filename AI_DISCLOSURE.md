# AI_DISCLOSURE.md

## What AI assistance was used

Claude (Anthropic) was used as an AI coding/research assistant to draft:
- `core/linear_memory.py`, `experiments/run_experiments.py`,
  `tests/test_linear_memory.py`, and `web/index.html`.
- `README.md`, `SOURCES.md`, and `docs/one-page-concept-summary.md`.

## What it helped generate

Full first-draft implementations of the files above, based on a
human-specified design: the choice of concept (associative memory / fast
weights, connected to BDH's Hebbian synaptic mechanism), the central
falsifiable claim, the write/read equations, and the requirement that every
number shown in the interactive artifact be either genuinely live-computed
or explicitly labeled precomputed.

## What was independently verified

- Every citation in `SOURCES.md` was checked against a live web search
  against the primary source (arXiv abstract page / official repository)
  before being included; no citation, author list, or reported number was
  taken from memory alone.
- The central claim was verified computationally, not just asserted: the
  test suite in `tests/test_linear_memory.py` runs the actual code and
  checks that the claimed degradation, the constant-vs-growing memory
  behavior, and the sparsity effect are all empirically true of the
  implementation, and the suite was run to confirm all tests pass (5/5)
  before this submission was written up.
- The BDH-CQ ARC-AGI-1 number (29.5% pass@2, ~$0.0007/task) and the claim
  that it was independently reproduced by Bielik AI / NYU are both
  attributed explicitly as Pathway's own reporting, not verified
  first-hand by this project, and are labeled as such in the artifact.

## What code was modified / reviewed

All generated code was read line-by-line, run, and checked against the
central claim before inclusion. The interference/sparsity sweep values in
`experiments/run_experiments.py` were inspected numerically (not just
visually) to confirm the trend matches what §6 of the artifact claims about
BDH's rationale for sparse activations.

## What remains to be done for a full competition-ready package

This delivered core (real computational substrate, tests, interactive
artifact, sourced BDH module, README) is the substantive, load-bearing part
of the submission. Still to produce before final submission: the full blog
PDF, `docs/judging-checklist.md`, `docs/architecture.md`, deployment to a
public URL (Vercel/GitHub Pages), and a final end-to-end accessibility/mobile
pass. These are documented as open items rather than silently omitted.
