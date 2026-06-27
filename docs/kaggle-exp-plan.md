# Task: Prepare Repository for Kaggle Experiment Workflow and Create Kaggle Notebooks

The repository has successfully passed both local and Kaggle smoke tests.

I am now transitioning from development into actual research experiments.

The objective is to create a clean Kaggle workflow while making **minimal changes** to the repository.

The existing architecture is considered stable. Avoid unnecessary refactoring.

---

# Overall Design Principles

1. Preserve the existing repository architecture.
2. Do not rewrite or duplicate any repository logic inside notebooks.
3. Notebooks should simply orchestrate the existing CLI scripts.
4. Prefer configuration changes over code changes.
5. Keep all modifications backwards compatible.
6. Minimize regression risk.

---

# Kaggle Experiment Workflow

The workflow will consist of five notebooks.

## Notebook 0

Prepare Dataset

Runs:

```python
!python scripts/prepare_dataset.py \
    --env kaggle
```

Outputs:

* preprocessed parquet
* metadata
* cache

After Notebook 0 completes, I will manually publish the generated artifacts as a Kaggle Dataset.

Dataset name:

```
kd-transformer-demand-forecasting-dataset
```

---

## Notebook 1

Train Teacher

↓

Generate Soft Targets

Runs:

```python
!python scripts/train_teacher.py ...
```

followed by

```python
!python scripts/generate_soft_targets.py ...
```

Outputs:

* teacher checkpoint
* teacher metrics
* soft targets

After Notebook 1 completes, I will manually publish the outputs as another Kaggle Dataset.

Dataset name:

```
kd-transformer-demand-forecasting-experiments
```

---

## Notebook 2

Train Student (No KD)

Runs only

```python
!python scripts/train_student.py ...
```

Outputs:

* baseline student checkpoint

---

## Notebook 3

Train Student (KD)

Runs only

```python
!python scripts/train_student.py ...
```

with KD enabled.

This notebook should verify that teacher checkpoint and soft targets exist before training.

Outputs:

* KD student checkpoint

---

## Notebook 4

Evaluate Models

Runs only

```python
!python scripts/evaluate_models.py ...
```

Outputs:

* evaluation metrics
* comparison tables
* exported CSV/JSON

No training.

---

# Shared Notebook Setup

Every notebook should begin with the same setup.

Clone repository if missing.

Otherwise pull latest changes.

Example:

```python
import os

if not os.path.exists("/kaggle/working/kd-transformer-demand-forecasting"):
    !git clone https://github.com/jingweii97/kd-transformer-demand-forecasting

%cd /kaggle/working/kd-transformer-demand-forecasting

!git pull

!pip install -r requirements.txt
```

Use Kaggle notebook syntax (`!` and `%cd`) throughout.

---

# IMPORTANT Repository Change

I would like to slightly improve the Kaggle environment configuration.

Currently the repository assumes artifacts live inside the repository.

Instead, for Kaggle experiments, I want preprocessing artifacts to be read from mounted Kaggle Datasets while keeping output paths unchanged.

The repository should continue using configuration-driven paths.

For the Kaggle environment only:

```
dataset_artifacts_dir:
/kaggle/input/kd-transformer-demand-forecasting-dataset/artifacts
```

```
experiment_artifacts_dir:
/kaggle/input/kd-transformer-demand-forecasting-experiments/artifacts
```

```
outputs_dir:
/kaggle/working/outputs
```

The code should continue reading paths from configuration.

Avoid hardcoded Kaggle paths.

---

# Repository Behaviour

Notebook 0

writes

```
/kaggle/working/artifacts/
```

Notebook 1

reads preprocessing artifacts from

```
dataset_artifacts_dir
```

writes

```
/kaggle/working/outputs/
```

and

```
/kaggle/working/artifacts/soft_targets/
```

Notebook 2

reads preprocessing artifacts from

```
dataset_artifacts_dir
```

writes outputs to

```
/kaggle/working/outputs/
```

Notebook 3

reads preprocessing artifacts from

```
dataset_artifacts_dir
```

reads soft targets from

```
experiment_artifacts_dir
```

writes outputs to

```
/kaggle/working/outputs/
```

Notebook 4

reads everything from the mounted datasets and writes only evaluation outputs to

```
/kaggle/working/outputs/
```

---

# Desired Property

The repository code should **not care** whether artifacts come from

```
artifacts/
```

or

```
/kaggle/input/...
```

It should simply read paths from configuration.

The same repository should continue to support:

* local
* Kaggle
* DICC cluster

using only different environment configurations.

---

# Deliverables

1. Make the minimal repository changes required to support this Kaggle workflow.
2. Explain every repository modification and why it is needed.
3. Ensure backwards compatibility with existing smoke tests.
4. Create five Kaggle notebooks:

   * Prepare Dataset
   * Train Teacher + Generate Soft Targets
   * Train Student (No KD)
   * Train Student (KD)
   * Evaluate Models
5. Each notebook should:

   * use Kaggle notebook syntax (`!`, `%cd`)
   * contain minimal duplicated code
   * execute existing repository scripts
   * print generated artifacts at the end
   * include a short markdown section describing:

     * purpose
     * required input datasets
     * generated outputs
6. Do **not** reimplement repository logic inside the notebooks.

Before implementing, first review whether the proposed configuration changes are sufficient or whether there is a simpler solution that better fits the existing repository architecture while preserving the same workflow.
