# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Thesis research repo: subdomain-restricted PCA/LDA compression of frozen
vision-foundation-model embeddings (CLIP, DINOv2), and zero-shot transfer of
a linear projection fit on one subdomain to a different subdomain. It
contains **three loosely-coupled parts** that are easy to mix up:

| Dir | Status | Purpose |
|---|---|---|
| `src/` | active, tracked in git | current pipeline: generate embeddings -> eigen/PCA diagnostics -> eval |
| `Preliminary-experiment/` | gitignored, own subproject | earlier/exploratory work: MTEB benchmark wrapper, cross-domain transfer experiments, all `PostProcessor` implementations |
| `proposal/` | gitignored, **read-only** | LaTeX thesis proposal |

`plans/` (tracked) holds forward-looking experiment/design docs that don't
correspond to code yet — read them for intent, don't assume they're
implemented.

## Hard constraints — read before touching anything

- **`proposal/**` is read-only.** Do not edit `.tex`/figures there even if a
  task references proposal content — design/write elsewhere (e.g. `plans/`)
  and point back at it.
- **`Preliminary-experiment/**` scripts must not be auto-executed.** Its own
  `README.md` states this explicitly: `main.py`, `dog-2-cat.py`, `n-2-cat.py`,
  `n-2-cat2.py` trigger long-running GPU/streaming extraction and can
  overwrite existing benchmark result JSON/CSVs. Only run them if the user
  explicitly asks for that specific run.
- `MINI_IMAGE_NET/`, `Preliminary-experiment/`, `proposal/`, `.venv/` are all
  gitignored (see `.gitignore`) — `git status`/`git log` will never show
  changes inside them; don't rely on git history to understand edits there.

## Setup & running (root `src/` pipeline only)

No build system, linter, or test suite is configured at the repo root or in
`Preliminary-experiment/` — don't invent `pytest`/`ruff` commands.

Root project is a plain venv + `requirements.txt` (not uv):
```
pip install -r requirements.txt
```

Eval scripts (`src/eval_embeddings.py`, `src/eval_online.py`,
`src/eval_abtt.py`, `src/main.py`) all expect pre-generated embedding caches
in `MINI_IMAGE_NET/*.npz` (gitignored — not present from a fresh clone).
Generate them first with `src/generate_embeddings.py`; see that file's
module docstring for the three supported invocation modes (plain CLI,
Colab notebook import, `colab exec` persistent-kernel exec) — the
`_is_cli_invocation()` guard exists specifically so the same file works
under all three without crashing on `argparse` when exec'd into a notebook
kernel.

`Preliminary-experiment/` is a **separate** uv-managed project
(`pyproject.toml` + `uv.lock`, Python >=3.13, its own `.venv`) — `uv sync`
then `uv run python <script>.py`, but see the execution constraint above
before running anything there.

### Running experiments on Colab (preferred for anything GPU-heavy)

Most new experiments (embedding generation, anything training an
`AutoEncoder`/`AdaptivePostProcessor`, `n-2-cat*.py`-style streaming
extraction) should run on a **Colab GPU runtime**, driven from this repo via
[`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) rather
than a local CPU `.venv`. The `colab` CLI is already installed in the root
`.venv` (`.venv/bin/colab`, backed by the `colab_cli` package).

- `colab exec -f <path>` sends a file's source into a persistent Colab
  kernel — top-level code runs with `__name__ == "__main__"` but **no
  `__file__` global**, unlike a real `python script.py` run.
  `src/generate_embeddings.py::_is_cli_invocation()` exists specifically to
  detect this and skip `argparse`/`__main__` execution on that path — any
  new script meant to run this way needs the same guard, or it'll crash on
  import.
- Typical two-step pattern (see `src/generate_embeddings.py`'s module
  docstring for the full version): `colab exec -f script.py` first
  (defines functions/registries only), then a second `colab exec -s
  <session>` call with the actual invocation line, e.g.:
  ```
  echo "_maybe_mount_drive_outputs(); run('siglip-base-patch16', 'cifar100', 'train', 64, None, 'outputs/embeddings', 'cuda')" | colab exec -s <session>
  ```
- Drive persistence convention: `_maybe_mount_drive_outputs()` mounts
  `/content/drive`, then symlinks `outputs/` to
  `/content/drive/MyDrive/eigen-feature/output-embedding-generation` so
  results survive a Colab session reset. Only fires under
  `google.colab` (import guarded, no-ops locally) — reuse this helper
  rather than writing a new Drive-mount path per script.
- `notebooks/generate_embeddings_colab.ipynb` was the original notebook-based
  entry point (imports `MODEL_REGISTRY`/`DATASET_REGISTRY`/`run()` directly
  into cells) — it's gone from the working tree (deleted, uncommitted) but
  still recoverable from git history (`git show d45d4bf:notebooks/generate_embeddings_colab.ipynb`)
  if a notebook-driven flow is needed instead of the CLI.

## Architecture — `src/` pipeline

All eval scripts share `src/utils.py`, which implements two parallel ways to
get a PCA-like subspace:
- `fit_train_subspace()` — homegrown: builds the cosine-similarity
  (Gram) matrix and eigendecomposes that instead of the `d x d` covariance
  matrix — cheaper when `m << d`, and returns eigenvectors in the original
  embedding's coordinate space via `project_topk()`.
- `fit_train_pca_sklearn()` — `sklearn.PCA` baseline.

Neither takes `k` as a fixed hyperparameter — both pick
`k = round(effective_rank(eigenvalues))`, where `effective_rank` is the
exponential of the Shannon entropy of the normalized eigenvalue spectrum.
This effective-rank-driven `k` selection is the one piece of logic every
eval script depends on; read it before changing any script's compression
behavior.

Given that shared base, the four entry points diverge in what they measure:
- **`main.py`** — standalone eigenvalue-spectrum diagnostic (train/test/val),
  plots `outputs/eigenvalues.png`, prints effective dimension at 90/95/99%
  variance thresholds (`effective_dimension()`) and effective rank per
  split. Doesn't use `load_split`/projectors machinery below.
- **`eval_embeddings.py`** — builds a `{name: projector_fn}` dict (`full`,
  `eigen-uncentered`, `eigen-centered`, `sklearn-pca`) fit once on train,
  then runs the same dict through k-NN classification, few-shot prototype
  classification, retrieval-overlap-vs-full, and pairwise subspace-angle
  comparison (`principal_angle_cosines`). Adding a new compression method to
  this file means adding one entry to `build_projectors()`.
- **`eval_online.py`** — streaming variant: replays train data in chunks
  (`CHECKPOINTS`), refits each method at each chunk size, and looks for the
  sample count at which a method's k-NN accuracy stably overtakes the
  uncompressed baseline (`_find_stable_crossover`). Plots
  `outputs/online_<encoder>.png`.
- **`eval_abtt.py`** — inverse ablation: instead of keeping the top-`k`
  components, it *removes* the top `D` (`remove_topd`, `D_VALUES`) and
  checks whether k-NN accuracy survives — evidence for whether the leading
  eigendirections are signal or a shared nuisance mean.

`utils.py` also has a second, independent group of helpers
(`cluster_metrics`, `assign_domains`, `split_fit_eval`) for a different
evaluation style — K-means clustering ACC/NMI/ARI/V-measure (matching the
thesis's own proxy-task protocol) over pseudo-domains carved out of
MINI_IMAGE_NET's single 100-class pool, rather than train/test/val k-NN.
These back three scripts implementing `plans/PROPOSAL_FIX_EXPERIMENTS.md`'s
experiment designs on cached embeddings instead of live ImageNet streaming
(needs torch — run on Colab, not the local venv; see the Colab-CLI section
above):
- **`eval_same_domain_gain.py`** (Experiment 1) — fits PCA/LDA on each
  pseudo-domain and zero-shot-transforms a disjoint held-out target domain,
  plus a same-domain fit/eval condition as the gain ceiling; reports
  `TransferRetention = cross-domain gain / same-domain gain`.
- **`eval_pca_lda_family.py`** (Experiment 2) — compares family add-ins
  (`KernelPCA`, `SparsePCA`, `FactorAnalysis`, `ShrinkageLDA`, and a
  `KernelPCA -> LDA` stand-in for Kernel LDA) against plain `PCA`/`LDA` at
  matched dimension, per pseudo-domain. Skips PPCA (analytically identical
  subspace to PCA) and Robust PCA (no sklearn primitive) on purpose.
- **`eval_linear_vs_nonlinear.py`** (Experiment 3) — trains a small local
  `TinyAutoEncoder` (torch) and compares it against `PCA`/`LDA`/
  `RandomProjection` at matched dimension on the target pseudo-domain only,
  to keep training cost bounded.

## `Preliminary-experiment/` — what it's for

Read `Preliminary-experiment/README.md` first — it has an accurate file
inventory and a table mapping every `PostProcessor` subclass to its file/line
range; don't duplicate that here. Two things worth knowing that aren't
obvious from a single file:
- `postprocessing.py`'s `PostProcessor` subclasses are shared by both
  `main.py` (MTEB clustering benchmark, via a dynamically-subclassed
  `MTEBWrapper` that monkeypatches `.encode()` to apply the fitted
  postprocessor) and the `*-2-cat*.py` transfer scripts (direct
  `fit`/`transform` calls, no MTEB).
- The `*-2-cat*.py` scripts fit a supervised projection (LDA, usually) on one
  ImageNet label-range ("domain") and zero-shot-apply it to a disjoint
  label-range, then K-Means-cluster the transformed target and score against
  ground truth. `n-2-cat2.py` is the most recent iteration (diverse
  per-class sampling to avoid LDA rank collapse from imbalanced classes);
  `dog_2_cat_methodology.md` has the formal writeup of the protocol.

## `plans/` and root notes

- `plans/PCA_AE_PLAN.md` — design for an AE-PCA hybrid (concatenate a
  learned nonlinear encoding with the raw embedding, then PCA) not yet
  implemented anywhere in `src/` or `Preliminary-experiment/`.
- `plans/PROPOSAL_FIX_EXPERIMENTS.md` — experiment designs responding to
  `Proposal_Fix.md` feedback (same-domain vs. cross-domain transfer gain,
  PCA/LDA variance-explained profiling, aggregating existing linear-vs-AE
  results). Designs only — none of the scripts it proposes exist yet.
- `GPT_PCA_FORMALIZE.md` — background taxonomy of the PCA-family and
  manifold-learning landscape; context for why the thesis scope is
  restricted to PCA/LDA rather than non-linear DR.
