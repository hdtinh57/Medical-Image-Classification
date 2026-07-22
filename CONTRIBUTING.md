# Contributing

## Scope and branching

- Work in the current repository workspace and use a focused branch for each change.
- Do not commit datasets, `.env` files, credentials, checkpoints, ONNX binaries, MLflow databases or
  generated artifacts. `.gitignore` protects these paths but contributors must verify changes before
  staging.
- Do not modify, deduplicate, move or relabel the raw Kaggle benchmark. SHA-256 same-content groups
  are leakage controls for internal Train/Validation splitting, not proof of invalid labels.
- Never push or commit another contributor's unreviewed work.

## Local quality checks

Run these before opening a pull request:

```powershell
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest
pre-commit run --all-files
```

`ci.yml` repeats lint, tests and infrastructure validation. It does not train a model. Avoid adding
GPU, Kaggle download or Docker-runtime dependencies to unit tests.

## Model contract

The shared source of truth is `training/model.py`:

- exactly 9 classes and their order;
- `224 × 224` ImageNet-normalized RGB input;
- output logits shape `(batch, 9)`;
- self-describing checkpoint schema.

Changes to model architecture, labels, preprocessing, checkpoint fields, ONNX export or Triton
configuration must be made together with tests. Do not introduce an independent class mapping in a
new component.

## Candidate lifecycle

1. Use `python -m training.pipeline` to train a versioned candidate locally or let the trusted
   `Train candidate model` workflow run it.
2. Candidate promotion uses **Validation** quality gates. Kaggle Test reports are audit artifacts and
   must not be used to tune thresholds.
3. `Release model` validates a candidate artifact and pauses at `model-production` approval.
4. Only the protected deployment job may use production MinIO/Triton secrets. Do not add credentials
   to GitHub-hosted jobs, workflow artifacts, logs, README examples or source files.
5. Do not overwrite model version objects. If deployment fails, inspect its audit JSON and use a new
   version after correction.

## Infrastructure changes

- Validate `docker compose config --quiet` after changing Compose.
- Validate Prometheus config/rules with `promtool`; keep Grafana dashboard files valid JSON.
- Do not restart unrelated local services. When diagnosing the serving stack, identify a specific
  container or port and stop only that target.
- Use pinned image versions/digests for production hardening; current Compose credentials and some
  image tags are development defaults.

## Pull request guidance

A pull request should describe:

- the system behavior changed and why;
- contract, data-leakage, security and rollback impact;
- commands/tests run and their results;
- any remaining manual setup, GitHub Environment configuration or known limitations.

Keep functions small and testable; project code targets cognitive complexity below 15 per function.
