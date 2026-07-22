"""Tests for trusted candidate verification and targeted deployment rollback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.deploy import (
    CandidateBundle,
    DeployConfig,
    _rollback_candidate,
    _smoke_request_payload,
    _validate_monotonic_version,
    _validate_smoke_response,
    config_from_env,
    load_candidate_bundle,
    run_deployment,
    wait_for_triton_ready,
)
from training.model import MODEL_NAME, NUM_CLASSES
from training.model_registry import RegistrySettings


def _bundle_checksums(bundle_dir: Path) -> dict[str, str]:
    """Build a checksum manifest matching the production bundle contract."""
    return {
        path.relative_to(bundle_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    }


class FakeS3Client:
    """Minimal fake for rollback object deletion and status persistence."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str,
        Delimiter: str | None = None,
    ) -> dict[str, object]:
        keys = [key for bucket, key in self.objects if bucket == Bucket and key.startswith(Prefix)]
        if Delimiter is None:
            return {"Contents": [{"Key": key} for key in keys]}
        children = {
            f"{Prefix}{key[len(Prefix):].split(Delimiter, 1)[0]}{Delimiter}"
            for key in keys
            if Delimiter in key[len(Prefix) :]
        }
        return {"CommonPrefixes": [{"Prefix": child} for child in sorted(children)]}

    def delete_objects(self, *, Bucket: str, Delete: dict[str, object]) -> None:
        for item in Delete["Objects"]:  # type: ignore[index]
            self.objects.pop((Bucket, item["Key"]), None)  # type: ignore[index]

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: object) -> None:
        self.objects[(Bucket, Key)] = Body


def _candidate_bundle(tmp_path) -> Path:
    bundle = tmp_path / "candidate"
    bundle.mkdir()
    checkpoint = bundle / "best_checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metrics = {"accuracy": 0.8, "balanced_accuracy": 0.7, "macro_f1": 0.6}
    (bundle / "candidate.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_version": 4,
                "git_sha": "abc123",
                "checkpoint_sha256": checksum,
                "validation_metrics": metrics,
                "quality_gate_passed": True,
            }
        ),
        encoding="utf-8",
    )
    (bundle / "quality_gate.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "passed": True,
                "candidate_metrics": metrics,
                "thresholds": {
                    "min_accuracy": 0.5,
                    "min_balanced_accuracy": 0.5,
                    "min_macro_f1": 0.5,
                    "max_macro_f1_regression": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle / "checksums.json").write_text(
        json.dumps(_bundle_checksums(bundle)),
        encoding="utf-8",
    )
    return bundle


def test_load_candidate_bundle_verifies_checksums_and_gate(tmp_path) -> None:
    candidate = load_candidate_bundle(_candidate_bundle(tmp_path))

    assert candidate.version == 4
    assert candidate.validation_metrics["macro_f1"] == 0.6


def test_load_candidate_bundle_rejects_tampered_checkpoint(tmp_path) -> None:
    bundle = _candidate_bundle(tmp_path)
    (bundle / "best_checkpoint.pth").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Checksum"):
        load_candidate_bundle(bundle)


def test_rollback_only_deletes_the_failed_triton_version(monkeypatch, tmp_path) -> None:
    bundle = _candidate_bundle(tmp_path)
    loaded = load_candidate_bundle(bundle)
    candidate = CandidateBundle(
        version=loaded.version,
        checkpoint_path=loaded.checkpoint_path,
        quality_gate_path=loaded.quality_gate_path,
        metadata=loaded.metadata,
        validation_metrics=loaded.validation_metrics,
        thresholds=loaded.thresholds,
    )
    client = FakeS3Client()
    client.objects[("models", f"{MODEL_NAME}/3/model.onnx")] = b"champion"
    client.objects[("models", f"{MODEL_NAME}/4/model.onnx")] = b"candidate"
    config = DeployConfig(
        bundle_dir=bundle,
        registry=RegistrySettings(registry_bucket="registry", triton_bucket="models"),
        triton_http_url="http://triton:8000",
    )
    monkeypatch.setattr("training.deploy.triton_version_ready", lambda *_: True)

    deleted = _rollback_candidate(
        client,
        config,
        candidate,
        {"version": 3},
        registered=True,
        published=True,
        error=RuntimeError("smoke failed"),
    )

    assert deleted == (f"{MODEL_NAME}/4/model.onnx",)
    assert ("models", f"{MODEL_NAME}/3/model.onnx") in client.objects
    assert ("models", f"{MODEL_NAME}/4/model.onnx") not in client.objects
    assert ("registry", f"{MODEL_NAME}/4/deployment.json") in client.objects


def test_monotonic_version_rejects_older_candidate(tmp_path) -> None:
    candidate = load_candidate_bundle(_candidate_bundle(tmp_path))
    client = FakeS3Client()
    client.objects[("registry", f"{MODEL_NAME}/5/best_checkpoint.pth")] = b"newer"
    settings = RegistrySettings(registry_bucket="registry", triton_bucket="models")

    with pytest.raises(ValueError, match="lớn hơn"):
        _validate_monotonic_version(candidate, client, settings)


def test_partial_publish_failure_removes_candidate_prefix(monkeypatch, tmp_path) -> None:
    bundle = _candidate_bundle(tmp_path)
    client = FakeS3Client()
    config = DeployConfig(
        bundle_dir=bundle,
        registry=RegistrySettings(registry_bucket="registry", triton_bucket="models"),
        triton_http_url="http://triton:8000",
    )
    monkeypatch.setattr("training.deploy.create_s3_client", lambda _: client)
    monkeypatch.setattr("training.deploy.ensure_bucket", lambda *_: None)
    monkeypatch.setattr("training.deploy.get_json", lambda *_: None)
    monkeypatch.setattr("training.deploy._validate_monotonic_version", lambda *_: None)
    monkeypatch.setattr("training.deploy.register_checkpoint", lambda *_args, **_kwargs: None)

    def fail_after_partial_upload(*_args, **_kwargs):
        client.objects[("models", f"{MODEL_NAME}/4/model.onnx")] = b"partial"
        raise OSError("upload interrupted")

    monkeypatch.setattr("training.deploy.run_export", fail_after_partial_upload)

    with pytest.raises(RuntimeError, match="upload interrupted"):
        run_deployment(config)

    assert ("models", f"{MODEL_NAME}/4/model.onnx") not in client.objects
    assert ("registry", f"{MODEL_NAME}/4/deployment.json") in client.objects


def test_wait_for_triton_ready_retries_then_returns(monkeypatch, tmp_path) -> None:
    attempts = iter([False, True])
    config = DeployConfig(
        bundle_dir=tmp_path,
        registry=RegistrySettings(),
        triton_http_url="http://triton:8000",
        timeout_seconds=5,
        poll_seconds=1,
    )
    monkeypatch.setattr("training.deploy.triton_version_ready", lambda *_: next(attempts))
    monkeypatch.setattr("training.deploy.time.sleep", lambda _: None)

    wait_for_triton_ready(config, 2)


def test_smoke_contract_requires_nine_logits() -> None:
    request = _smoke_request_payload()
    assert request["inputs"][0]["shape"] == [1, 3, 224, 224]
    assert len(request["inputs"][0]["data"]) == 3 * 224 * 224

    _validate_smoke_response(
        {"outputs": [{"name": "logits", "shape": [1, NUM_CLASSES], "data": [0.0] * 9}]}
    )
    with pytest.raises(ValueError, match="shape"):
        _validate_smoke_response(
            {"outputs": [{"name": "logits", "shape": [1, 8], "data": [0.0] * 8}]}
        )


def test_protected_deploy_requires_explicit_environment_secrets(monkeypatch, tmp_path) -> None:
    for name in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "TRITON_HTTP_URL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="MINIO_ENDPOINT"):
        config_from_env(tmp_path, timeout_seconds=60, poll_seconds=3)
