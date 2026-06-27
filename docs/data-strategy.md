# Final Architecture Recommendation for Full M5 Training

## Background

The Phase 1 implementation and repository refactoring have been successfully completed and verified on the CA_1 subset.

Subsequent scalability testing on the full M5 dataset revealed that the repository no longer fails during preprocessing after the following optimizations:

* Incremental per-store preprocessing.
* Per-store Parquet caching.
* Removal of unused high-memory columns (`date`, `d`, `wm_yr_wk`).

After these optimizations, the cached DataFrame memory was reduced from approximately **10.3 GB** to **3.6 GB**. However, construction of a single `TimeSeriesDataSet` for the entire M5 dataset still exceeds the available memory on Kaggle (30 GB RAM).

This behaviour is consistent with the documented design of PyTorch Forecasting, where `TimeSeriesDataSet` performs in-memory processing and the maintainers recommend using subsets with shared encoders/scalers for very large datasets.

Therefore, the implementation strategy—not the research methodology—must change.

---

# Design Principles

The implementation must satisfy the following requirements.

## Scientific methodology (must remain unchanged)

The implementation **must preserve**:

* Full M5 dataset.
* Chronological Train / Validation / ID Test / OOD Test splits.
* Same feature engineering.
* Same TFT teacher.
* Same Transformer student.
* Same Knowledge Distillation formulation.
* Same hyperparameter tuning procedure.
* Same evaluation metrics.
* Same experimental protocol.

No data sampling, window reduction, or methodological simplification is permitted.

---

## Engineering principles

The implementation should:

* Fit within commodity cloud hardware (e.g. Kaggle or DICC).
* Avoid loading the entire processed dataset into memory simultaneously.
* Minimize custom infrastructure.
* Prefer officially supported PyTorch Forecasting APIs wherever possible.
* Keep repository structure unchanged where possible.

---

# Rejected Architectures

## Option A — Single Full-M5 TimeSeriesDataSet

```
Full DataFrame
      ↓
TimeSeriesDataSet
      ↓
Trainer.fit()
```

Status:

**Rejected**

Reason:

Memory profiling demonstrates that this approach exceeds available RAM even after preprocessing optimizations.

---

## Option B — ConcatDataset(TimeSeriesDataSet)

Status:

**Not recommended**

Reason:

Although conceptually attractive, this approach:

* still requires multiple `TimeSeriesDataSet` objects to exist simultaneously;
* is not documented by PyTorch Forecasting;
* introduces uncertainty regarding samplers and collate behaviour;
* has not been validated by the library maintainers.

This option should only be considered if demonstrated to work through a prototype.

---

## Option C — Immediate custom streaming implementation

Status:

**Deferred**

Reason:

Although technically feasible, implementing a custom streaming framework before evaluating supported approaches introduces unnecessary engineering risk.

---

# Recommended Architecture

The recommended solution is a **partitioned training architecture** that preserves the scientific experiment while changing only the implementation strategy.

The guiding principle is:

> Partition the implementation, not the methodology.

---

# Partition Strategy

The existing per-store Parquet files shall remain the canonical processed dataset.

```
artifacts/

    data/

        preprocessed_CA_1.parquet

        ...

        preprocessed_WI_3.parquet
```

No additional preprocessing changes are required.

---

# Global Metadata Stage

Before training begins, construct global preprocessing metadata from the full training dataset.

This stage should establish:

* categorical encoders;
* normalisation parameters;
* feature schema.

This metadata must **not** be fitted from a single store such as CA_1.

Instead, it must represent the complete training dataset so that every partition uses identical preprocessing behaviour.

---

# Partitioned Dataset Construction

Training data should be constructed one partition at a time.

Conceptually:

```
Load one partition

↓

Construct TimeSeriesDataSet

↓

Create DataLoader

↓

Consume batches

↓

Release memory

↓

Load next partition
```

At no point should all store partitions exist simultaneously as `TimeSeriesDataSet` objects.

---

# Epoch Definition

One logical training epoch shall consist of one complete pass through every store partition.

Conceptually:

```
Epoch

↓

CA_1

↓

CA_2

↓

...

↓

WI_3

↓

Epoch complete
```

The implementation should preserve:

* one model;
* one optimizer;
* one learning-rate scheduler;
* one checkpoint sequence;
* one early stopping process.

Only the mechanism by which batches are supplied should differ.

---

# Validation and Evaluation

Validation, ID testing, and OOD testing should follow the same partitioned strategy.

Each partition may be evaluated independently.

Metrics should then be aggregated to produce the final dataset-wide results.

This preserves the methodology because every series is still evaluated exactly once.

---

# Implementation Strategy

Implementation should proceed incrementally.

## Phase 1 — Design Investigation

Compare the following implementations:

1. ConcatDataset.
2. Lightning CombinedLoader.
3. Officially documented PyTorch Forecasting approaches.
4. Custom partition manager.

Evaluate each against:

* memory usage;
* Lightning compatibility;
* checkpoint behaviour;
* early stopping;
* validation scheduling;
* implementation complexity;
* reproducibility.

No implementation should be selected without evidence.

---

## Phase 2 — Prototype

Construct a two-store prototype using:

```
CA_1
CA_2
```

Verify:

* continuous optimizer state;
* checkpoint saving;
* validation;
* epoch accounting;
* memory remains bounded.

Only after successful verification should scaling continue.

---

## Phase 3 — Full Implementation

Extend the validated architecture to all ten stores.

No further methodological changes should occur after this stage.

---

# Repository Impact

The following components should remain unchanged:

* preprocessing pipeline;
* feature engineering;
* configuration structure;
* models;
* evaluation methodology;
* experiment design;
* repository layout.

Changes should be isolated to the training data orchestration layer.

---

# Success Criteria

The implementation is considered successful if it satisfies all of the following:

* Full M5 dataset is used.
* Memory remains within hardware limits.
* Scientific methodology remains identical to Chapter 4.
* One optimizer state is maintained.
* One checkpoint sequence is maintained.
* ID/OOD evaluation remains unchanged.
* Results are fully reproducible.

---

# Final Recommendation

The objective is **not** to invent a new forecasting framework.

The objective is to build the smallest possible orchestration layer that enables PyTorch Forecasting to train on the complete M5 dataset while preserving the methodology described in Chapter 4.

Every implementation decision should therefore prioritize:

1. Scientific fidelity.
2. Simplicity.
3. Reproducibility.
4. Official library support wherever possible.
5. Minimal custom infrastructure.

Only if officially supported approaches prove inadequate should additional custom training infrastructure be introduced.
