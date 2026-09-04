# SOURCES.md

Every technical claim about BDH, BDH-CQ, or the broader linear-attention /
fast-weight literature in this project is grounded in one of the primary
sources below. Secondary write-ups (blogs, news coverage) are listed only
where they were used to locate or corroborate a primary source, and are
never cited as the sole basis for a technical claim.

## Primary BDH / BDH-CQ sources

1. **Kosowski, A., Uznański, P., Chorowski, J., Stamirowska, Z., Bartoszkiewicz, M.**
   *The Dragon Hatchling: The Missing Link between the Transformer and Models
   of the Brain.* arXiv:2509.26507 (2025). https://arxiv.org/abs/2509.26507
   - **Primary.**
   - Used for: the BDH architecture description (§6 of the artifact), the
     claim that BDH's attention arises from a local Hebbian edge-reweighting
     rule over sparse, non-negative neuron activations, the ~5%-active-neurons
     sparsity figure, and the description of BDH-GPU as a ReLU-low-rank /
     linear-attention formulation.

2. **Pathway.** *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.*
   arXiv:2608.09888 (2026). https://arxiv.org/abs/2608.09888 — see also
   Pathway's own summary, https://pathway.com/research/introducing-bdh-cq
   - **Primary** (arXiv preprint) with a **primary developer blog post**
     summary.
   - Used for: the claim that BDH-CQ updates recurrent memory from
     demonstrations at inference time rather than via a backward pass, that
     it reasons in a continuous latent workspace instead of a verbalized
     chain of thought, and the reported 29.5% pass@2 / $0.0007-per-task result
     on the public ARC-AGI-1 evaluation. This is a **developer-reported
     benchmark result**; the artifact explicitly labels it as such rather
     than as an independent evaluation or a deployment claim.
   - The independent-reproduction claim (Bielik AI and NYU) is reported
     *by Pathway*, not independently verified by this project; the artifact
     attributes it as "per Pathway's reporting" for that reason.

3. **pathwaycom/bdh** — official (toy) BDH implementation and README.
   https://github.com/pathwaycom/bdh
   - **Primary** (public source code / documentation released by the
     authors).
   - Used to confirm that the publicly released repository implements the
     *baseline* BDH variant and does not itself reproduce the Sudoku-Extreme
     97.4% figure (that figure is attributed to Pathway's internal
     implementation in the repository's own README, and is treated in this
     project only as a reported, non-reproduced number — never as something
     we ran).

## Primary papers on the selected concept (Associative Memory & Fast Weights / Linear Attention), 2022–2026

4. **Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F.**
   *Transformers are RNNs: Fast Autoregressive Transformers with Linear
   Attention.* ICML 2020. (Foundational; included because BDH-GPU and this
   project's toy memory both build on the linear-attention view it
   introduced — cited here as background for how attention can be rewritten
   as a recurrent, fixed-size state update.)
   - **Primary.**

5. **Schlag, I., Irie, K., Schmidhuber, J.**
   *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021 /
   arXiv:2102.11174.
   - **Primary.**
   - Used for: the "fast weights" framing of linear attention as an
     outer-product write into a matrix-valued state — the same write rule
     (`S += φ(k) vᵀ`) implemented in `core/linear_memory.py`, and for the
     observation (independently reproduced by this project's own
     `sparsity_sweep` experiment) that key overlap causes interference in
     such associative stores.

6. **Behrouz, A., Zhong, P., Mirrokni, V.**
   *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663 (2025).
   - **Primary**, 2022–2026 window.
   - Used as recent (2025) evidence that fixed-capacity, continually-updated
     memory for sequence models is an active research direction beyond BDH,
     supporting the "why this concept matters now" framing in the one-page
     summary — not used for any numeric claim in the interactive artifact.

## Where each source is used

| Source | Used in |
|---|---|
| #1 Dragon Hatchling | `web/index.html` §6, `docs/one-page-concept-summary.md`, README |
| #2 BDH-CQ | `web/index.html` §6, `docs/one-page-concept-summary.md` |
| #3 pathwaycom/bdh repo | `docs/one-page-concept-summary.md` (evidence labeling) |
| #4 Katharopoulos et al. 2020 | `docs/one-page-concept-summary.md` (mechanism background) |
| #5 Schlag et al. 2021 | `core/linear_memory.py` docstring, `docs/one-page-concept-summary.md` |
| #6 Behrouz et al. 2025 (Titans) | `docs/one-page-concept-summary.md` (research-context / "why now") |

## Note on evidence discipline

- A **benchmark result reported by the model's own developer** (BDH-CQ's
  ARC-AGI-1 number) is never presented in this project as an independent
  evaluation or a deployment claim.
- No number in `web/index.html` §§1–5 (accuracy, byte counts, load factors)
  is drawn from any external source — all of it is produced live by this
  project's own code (`core/linear_memory.py`, `experiments/run_experiments.py`)
  and is explicitly our own toy-model measurement, not a BDH measurement.
- Where evidence for a claim was not found in a primary source, that gap is
  stated plainly in `docs/one-page-concept-summary.md` rather than filled in.
