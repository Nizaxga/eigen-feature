# Experiment Designs for Proposal_Fix.md

Source of truth for these designs: `Proposal_Fix.md` (Discussion addition x2,
Experiment x1). Proposal `.tex` files are read-only; this doc only designs
the experiments/scripts that would produce the numbers and figures those
fixes need. Nothing here touches `proposal/`.

---

## Experiment 1 — Same-Domain Performance Gain (the explicit ask)

**Maps to:** "Using the domain-transfer experiment, experiment about same
domain performance gain."

**Base:** `Preliminary-experiment/n-2-cat2.py` (diverse per-class sampler,
`DOMAINS = {Dogs, Birds, Vehicles, Household}`, all fit zero-shot and
evaluated on a single Cats target).

**Gap:** the existing script never fits on Cats itself, and never scores the
*raw* (uncompressed) Cats baseline inside the same run — so there is no
number to answer "how much of the gain from compression comes from crossing
domains, vs. just from compressing at all?"

**Design:**

1. Add a same-domain condition: pull two **disjoint** Cat samples from the
   stream (different shuffle seed or a `.skip(N)` offset so no image
   appears in both) —
   - `cat_fit_X, cat_fit_y` (fit set, same role as a "source domain")
   - `cat_eval_X, cat_eval_y` (eval set, same role as the existing "target")
   Reusing the same disjoint sample for both same-domain and cross-domain
   conditions keeps the eval side identical across all rows, which is what
   makes the comparison valid.
2. Run the existing `pp.fit(source_X, source_y)` → `pp.transform(target)` →
   `evaluate()` loop with `source = cat_fit_{X,y}` labeled
   `"Cats (same-domain)"`, appended to the same results table as
   `Dogs->Cat`, `Birds->Cat`, etc.
3. Add the missing raw baseline row (`evaluate(cat_eval_X, cat_eval_y)`,
   uncompressed) once per `(dim, seed)` — `dog-2-cat.py` already does this
   ("Base Model (Raw)"); `n-2-cat.py`/`n-2-cat2.py` currently don't.
4. Derive two metrics per `(domain, dim, seed)`:
   - `Gain(domain, dim) = metric_compressed(domain, dim) - metric_raw`
   - `TransferRetention(domain, dim) = Gain(domain, dim) / Gain("Cats (same-domain)", dim)`
     for `domain != Cats` — the fraction of the achievable
     same-domain ceiling that zero-shot cross-domain transfer actually
     recovers. This is the number Objective 2 needs but never computed.

**Deliverable:** `Preliminary-experiment/same-domain-gain.py` (fork of
`n-2-cat2.py`), writing `same_domain_gain_results.csv`
(`Source Domain, Target Dimension, Seed, NMI/ARI/ACC, Gain, TransferRetention`)
and a plot (`same_domain_gain_plot.png`) with one line per source domain plus
a "Cats (same-domain)" reference line, x-axis = target dimension.

**Caveat to flag in the write-up:** `TransferRetention` can exceed 1.0 or go
negative (cross-domain transfer occasionally beating same-domain fitting, or
hurting relative to raw) — report both directions rather than only the
success case.

---

## Experiment 2 — PCA/LDA Family Ablation ("variance" = variants, not explained variance)

**Maps to:** "Add the variance of PCA, LDA techniques." — clarified: this
means the *family* of each method (PPCA, Kernel PCA, Sparse PCA, ... =
PCA + some add-in; shrinkage/kernel variants = LDA + some add-in), not
explained-variance curves. (Superseded the earlier misreading of this line.)

**Gap:** `Preliminary-experiment/postprocessing.py` only implements plain
`PCA_PP` and `LDA`. `GPT_PCA_FORMALIZE.md` already catalogs the wider family
(PPCA, Factor Analysis, Kernel PCA, Sparse PCA, Robust PCA, Generalized PCA
on the PCA side) but nothing in the repo empirically checks whether any
family member's "add-in" actually changes downstream subdomain-compression
quality versus the plain base method.

**Design:**

1. **Skip PPCA analytically, don't run it.** `GPT_PCA_FORMALIZE.md` §2
   ("Key Result") already states that under maximum-likelihood fitting on
   complete data, PPCA's principal subspace is identical to ordinary PCA's —
   it only adds a scalar isotropic noise term, no new eigenvectors. An
   empirical PPCA-vs-PCA downstream comparison would just reproduce PCA_PP's
   existing numbers; cite the equivalence in the discussion instead of
   spending a run on it.
2. **PCA-family add-ins worth actually running** (each a small new
   `PostProcessor` subclass in `postprocessing.py`, same
   `fit`/`transform` shape as `PCA_PP`):
   - `KernelPCA_PP` — `sklearn.decomposition.KernelPCA(kernel="rbf")`. The
     non-linear add-in; also reusable as a genuine non-linear baseline for
     Experiment 3 (one fit, two write-ups).
   - `SparsePCA_PP` — `sklearn.decomposition.SparsePCA` (or
     `MiniBatchSparsePCA` for speed). The sparsity/interpretability add-in.
   - `FactorAnalysis_PP` — `sklearn.decomposition.FactorAnalysis`. The
     anisotropic-noise add-in (the one PPCA variant that is *not*
     analytically equivalent to plain PCA, unlike step 1).
   - Robust PCA — **flag as out of scope for this pilot**: no drop-in
     sklearn primitive, needs a custom low-rank + sparse solver (IALM/ADMM).
     Note it as future work rather than implementing from scratch here.
3. **LDA-family add-ins:**
   - `ShrinkageLDA` — `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`.
     Directly targets the Small-Sample-Size / singular-`S_W` problem the
     literature review already flags
     (`proposal/gradschool-newformat/3-literature-review.tex:79`) — cheapest
     and most on-topic add-in to test.
   - Kernel LDA analog — sklearn has no built-in Kernel Discriminant
     Analysis; approximate with a `KernelPCA -> LDA` pipeline (project with
     the same fitted `KernelPCA_PP` from step 2, then fit plain `LDA` on the
     projected features) as the non-linear LDA stand-in.
4. **Protocol:** reuse whichever subdomain-fit → clustering-metric pipeline
   Experiment 1/3 already runs (same domains, same target dims), swap in
   each variant class via `main.py`'s `build_postprocessor`/`TASK_REGISTRY`,
   and compare against the existing plain `PCA_PP`/`LDA` result JSONs already
   sitting in `PCA/results/` and `LDA/results/` — no need to rerun the
   baseline.

**Deliverable:** new subclasses in `postprocessing.py`
(`KernelPCA_PP`, `SparsePCA_PP`, `FactorAnalysis_PP`, `ShrinkageLDA`) +
`Preliminary-experiment/variance_family_ablation.py` that runs them and
aggregates into `variance_family_results.csv` + a per-backbone plot
(variant vs. V-Measure/ACC at matched dim), same shape as Experiment 3's
output so both can sit in one discussion figure.

**Caveat to flag:** Kernel/Sparse/Factor variants each carry their own
hyperparameter (`gamma`, `alpha`, ...) — fix a sensible value or do a light
per-backbone sweep rather than trusting library defaults; otherwise "this
family member underperforms" may just mean "badly tuned," not a real
finding about the add-in itself.

---

## Experiment 3 — Why Not Non-linear DR (evidence, not just citation)

**Maps to:** "Add why we aren't experimenting with non-linear dimension
reduction techniques."

**Gap:** the current draft justifies skipping non-linear DR purely by
citing the linear-probe literature (Alain & Bengio). But the repo already
*has* non-linear baselines run and scored — `postprocessing.AutoEncoder`
(non-linear bottleneck) and `postprocessing.AdaptivePostProcessor` /
`DisentangledAdaptivePostProcessor` (relation-distillation) — with MTEB-JSON
results sitting in `AUTO_ENCODER_0..4/results/`, `RANDOM_0..4/results/`,
`PCA/results/`, `LDA/results/`, `NEW_RUN_0..4/results/`. Nobody has
aggregated them into one table, so the "non-linear doesn't help here" claim
currently has no empirical backing even though the data exists.

**Design (aggregation only, no new data collection):**

1. Walk `*/results/**/*.json` under `Preliminary-experiment/` (MTEB format:
   `scores.test[0].{v_measure,nmi,ari,cluster_accuracy}`), parse the method
   and target dim out of the model-id folder name (same regex idea as
   `plot-pca-0-100.py::parse_model_info`, generalized beyond `PCA_PP`).
2. Group by `(method, backbone, dataset, target_dim)`, mean±std across the
   run replicates (`_0`.._4` dirs / seeds).
3. At matched target dimension, compare `AutoEncoder` against `PCA_PP` /
   `LDA` / `RandomProjection` on the same three clustering benchmarks
   (`ImageNet10Clustering`, `ImageNetDog15Clustering`,
   `TinyImageNetClustering`). If `AutoEncoder` doesn't beat `PCA`/`LDA` at
   equal dim despite its non-linear capacity, that's the direct empirical
   support for the discussion paragraph — stronger than the theoretical
   argument alone.

**Deliverable:** `Preliminary-experiment/aggregate_linear_vs_nonlinear.py`,
outputs `linear_vs_nonlinear_summary.csv` + a grouped-bar or line plot
(V-Measure vs. target dim, one series per method) per backbone/dataset.

**Caveat to flag:** check coverage before drawing the conclusion —
`AUTO_ENCODER_*` dims (102, 153, 204, 256, 307, ...) don't obviously line up
with `PCA`/`LDA`'s dims; confirm matched-dim rows actually exist before
claiming the comparison, and note any dims that had to be interpolated or
dropped.

---

## Priority / sequencing

1. **Experiment 1** first — it's the literal ask and needs new data
   collection (ImageNet streaming + CLIP inference), so it has the longest
   lead time.
2. **Experiment 3** next — pure aggregation of data that already exists on
   disk, cheapest to produce, and it's needed before the "why no non-linear
   DR" paragraph can cite numbers instead of only theory.
3. **Experiment 2** last — needs new `PostProcessor` subclasses (small,
   sklearn-backed) before it can run at all, so it can't start before
   Experiment 3's aggregation harness exists to reuse; its `KernelPCA_PP`
   fit doubles as a non-linear baseline point for Experiment 3, so run them
   back-to-back rather than fully sequentially.
