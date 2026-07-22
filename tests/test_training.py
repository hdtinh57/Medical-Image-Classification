"""Tests for the local MLflow tracking configuration."""

from __future__ import annotations

from training.train import DEFAULT_MLFLOW_DATABASE_PATH, default_mlflow_tracking_uri


def test_default_mlflow_tracking_uri_uses_sqlite() -> None:
    expected = f"sqlite:///{DEFAULT_MLFLOW_DATABASE_PATH.resolve().as_posix()}"

    assert default_mlflow_tracking_uri() == expected
