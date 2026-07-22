"""Deploy a verified candidate bundle to MinIO and Triton with targeted rollback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from botocore.exceptions import ClientError

from training.export_triton import ExportConfig, run_export
from training.model import MODEL_NAME, NUM_CLASSES
from training.model_registry import (
    ObjectAlreadyExistsError,
    RegistrySettings,
    create_s3_client,
    delete_prefix,
    deployment_key,
    ensure_bucket,
    get_json,
    list_integer_versions,
    production_key,
    put_json,
    register_checkpoint,
    triton_version_prefix,
)
from training.quality_gate import GateThresholds, evaluate_quality_gate, extract_metrics

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_POLL_SECONDS = 3
REQUIRED_DEPLOY_ENV = (
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "TRITON_HTTP_URL",
)


@dataclass(frozen=True)
class DeployConfig:
    """Validated runtime inputs for one protected deployment transaction."""

    bundle_dir: Path
    registry: RegistrySettings
    triton_http_url: str
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    poll_seconds: int = DEFAULT_POLL_SECONDS

    def validate(self) -> None:
        """Validate finite deployment timeout and endpoint inputs."""
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds phải lớn hơn 0.")
        if self.poll_seconds < 1:
            raise ValueError("poll_seconds phải lớn hơn 0.")
        if not self.triton_http_url.startswith(("http://", "https://")):
            raise ValueError("TRITON_HTTP_URL phải bắt đầu bằng http:// hoặc https://.")


@dataclass(frozen=True)
class CandidateBundle:
    """Trusted metadata and local paths verified from a pipeline artifact."""

    version: int
    checkpoint_path: Path
    quality_gate_path: Path
    metadata: dict[str, Any]
    validation_metrics: dict[str, float]
    thresholds: GateThresholds


@dataclass(frozen=True)
class DeploymentResult:
    """Final deployment result with its immutable version and audit status."""

    version: int
    status: str
    rollback_keys: tuple[str, ...] = ()


def run_deployment(config: DeployConfig) -> DeploymentResult:
    """Deploy one verified bundle, updating production only after Triton smoke passes."""
    config.validate()
    candidate = load_candidate_bundle(config.bundle_dir)
    client = create_s3_client(config.registry)
    ensure_bucket(client, config.registry.registry_bucket)
    ensure_bucket(client, config.registry.triton_bucket)
    champion = get_json(client, config.registry.registry_bucket, production_key())
    _validate_against_champion(candidate, champion)
    _validate_monotonic_version(candidate, client, config.registry)
    registered = False
    published = False
    try:
        register_checkpoint(
            client,
            config.registry,
            version=candidate.version,
            checkpoint_path=candidate.checkpoint_path,
            candidate_metadata=candidate.metadata,
            quality_gate_path=candidate.quality_gate_path,
        )
        registered = True
        published = True
        run_export(
            ExportConfig(
                version=candidate.version,
                local_checkpoint=candidate.checkpoint_path,
                upload=True,
                registry=config.registry,
            )
        )
        wait_for_triton_ready(config, candidate.version)
        smoke_inference(config.triton_http_url, candidate.version)
    except Exception as exc:
        _rollback_candidate(
            client,
            config,
            candidate,
            champion,
            registered=registered,
            published=published,
            error=exc,
        )
        raise RuntimeError(f"Deploy model version {candidate.version} thất bại: {exc}") from exc

    status = _successful_status(candidate)
    put_json(client, config.registry.registry_bucket, deployment_key(candidate.version), status)
    put_json(client, config.registry.registry_bucket, production_key(), status)
    return DeploymentResult(version=candidate.version, status="deployed")


def load_candidate_bundle(bundle_dir: Path) -> CandidateBundle:
    """Validate an immutable candidate artifact before loading its checkpoint."""
    bundle_dir = bundle_dir.expanduser().resolve()
    _verify_bundle_checksums(bundle_dir)
    metadata = _read_json(bundle_dir / "candidate.json")
    gate = _read_json(bundle_dir / "quality_gate.json")
    _validate_bundle_schema(metadata, gate)
    version = _bundle_version(metadata)
    checkpoint_path = bundle_dir / "best_checkpoint.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if _sha256(checkpoint_path) != metadata["checkpoint_sha256"]:
        raise ValueError("Checksum checkpoint không khớp candidate metadata.")
    thresholds = _thresholds_from_gate(gate)
    metrics = _validation_metrics(metadata)
    gate_metrics = extract_metrics(gate["candidate_metrics"])
    if gate_metrics != metrics:
        raise ValueError("Validation metrics không nhất quán giữa candidate và quality gate.")
    if not evaluate_quality_gate(metrics, thresholds).passed:
        raise ValueError("Candidate metrics không còn thỏa quality gate.")
    return CandidateBundle(
        version=version,
        checkpoint_path=checkpoint_path,
        quality_gate_path=bundle_dir / "quality_gate.json",
        metadata=metadata,
        validation_metrics=metrics,
        thresholds=thresholds,
    )


def _verify_bundle_checksums(bundle_dir: Path) -> None:
    """Verify each recorded artifact and reject missing or extra bundle files."""
    checksum_path = bundle_dir / "checksums.json"
    recorded = _read_json(checksum_path)
    actual_files = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    }
    if set(recorded) != actual_files:
        raise ValueError("Candidate bundle có file thiếu hoặc không được checksum.")
    for relative_path, expected in recorded.items():
        path = bundle_dir / relative_path
        if _sha256(path) != expected:
            raise ValueError(f"Checksum bundle không khớp: {relative_path}")


def _validate_bundle_schema(metadata: dict[str, Any], gate: dict[str, Any]) -> None:
    """Reject incomplete or unsupported candidate metadata before deployment."""
    if metadata.get("format_version") != 1 or gate.get("format_version") != 1:
        raise ValueError("Candidate bundle format_version không được hỗ trợ.")
    if metadata.get("quality_gate_passed") is not True or gate.get("passed") is not True:
        raise ValueError("Candidate không vượt quality gate Validation.")
    git_sha = metadata.get("git_sha")
    if not isinstance(git_sha, str) or not git_sha.strip():
        raise ValueError("candidate.json thiếu git_sha hợp lệ.")
    checkpoint_sha = metadata.get("checkpoint_sha256")
    if not _is_sha256(checkpoint_sha):
        raise ValueError("candidate.json thiếu checkpoint_sha256 hợp lệ.")
    if not isinstance(gate.get("candidate_metrics"), dict):
        raise ValueError("quality_gate.json thiếu candidate_metrics.")


def _is_sha256(value: Any) -> bool:
    """Return whether a value is one lowercase or uppercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _bundle_version(metadata: dict[str, Any]) -> int:
    """Validate the positive integer model version supplied by the pipeline."""
    version = metadata.get("model_version")
    if not isinstance(version, int) or version < 1:
        raise ValueError("candidate.json có model_version không hợp lệ.")
    return version


def _thresholds_from_gate(gate: dict[str, Any]) -> GateThresholds:
    """Restore the original thresholds so deploy repeats the same gate policy."""
    values = gate.get("thresholds")
    if not isinstance(values, dict):
        raise ValueError("quality_gate.json thiếu thresholds.")
    try:
        thresholds = GateThresholds(
            min_accuracy=float(values["min_accuracy"]),
            min_balanced_accuracy=float(values["min_balanced_accuracy"]),
            min_macro_f1=float(values["min_macro_f1"]),
            max_macro_f1_regression=float(values["max_macro_f1_regression"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("quality_gate.json có thresholds không hợp lệ.") from exc
    thresholds.validate()
    return thresholds


def _validation_metrics(metadata: dict[str, Any]) -> dict[str, float]:
    """Return the validated metric set embedded by the trusted pipeline."""
    metrics = metadata.get("validation_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("candidate.json thiếu validation_metrics.")
    return extract_metrics(metrics)


def _validate_against_champion(candidate: CandidateBundle, champion: dict[str, Any] | None) -> None:
    """Repeat the candidate gate with current production as the reference when present."""
    reference = None if champion is None else champion.get("validation_metrics")
    decision = evaluate_quality_gate(candidate.validation_metrics, candidate.thresholds, reference)
    if not decision.passed:
        raise ValueError("Candidate regression vượt champion quality gate.")


def _validate_monotonic_version(
    candidate: CandidateBundle,
    client: Any,
    settings: RegistrySettings,
) -> None:
    """Ensure Triton's latest-version routing cannot select an older candidate."""
    registered = list_integer_versions(client, settings.registry_bucket)
    serving = list_integer_versions(client, settings.triton_bucket)
    known_versions = registered + serving
    if known_versions and candidate.version <= max(known_versions):
        raise ValueError("Candidate version phải lớn hơn mọi registry/Triton version hiện có.")


def wait_for_triton_ready(config: DeployConfig, version: int) -> None:
    """Wait only up to the configured deadline for the new model version to load."""
    deadline = time.monotonic() + config.timeout_seconds
    while time.monotonic() < deadline:
        if triton_version_ready(config.triton_http_url, version):
            return
        time.sleep(config.poll_seconds)
    message = f"Triton không ready model version {version} " f"trong {config.timeout_seconds}s."
    raise TimeoutError(message)


def triton_version_ready(base_url: str, version: int) -> bool:
    """Check one exact Triton model-version readiness endpoint."""
    url = f"{base_url.rstrip('/')}/v2/models/{MODEL_NAME}/versions/{version}/ready"
    try:
        with urlopen(Request(url, method="GET"), timeout=5) as response:
            return response.status == 200
    except (HTTPError, URLError, TimeoutError):
        return False


def smoke_inference(base_url: str, version: int) -> None:
    """Submit one JSON tensor request and enforce the nine-logit serving contract."""
    url = f"{base_url.rstrip('/')}/v2/models/{MODEL_NAME}/versions/{version}/infer"
    payload = _smoke_request_payload()
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Triton smoke inference thất bại: {exc}") from exc
    _validate_smoke_response(response_payload)


def _smoke_request_payload() -> dict[str, Any]:
    """Create one contract-valid zero tensor without relying on client SDKs."""
    tensor = np.zeros((1, 3, 224, 224), dtype=np.float32)
    return {
        "inputs": [
            {
                "name": "input",
                "shape": list(tensor.shape),
                "datatype": "FP32",
                "data": tensor.ravel().tolist(),
            }
        ]
    }


def _validate_smoke_response(payload: dict[str, Any]) -> None:
    """Validate Triton's JSON response shape before advancing production metadata."""
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("Triton smoke response thiếu output logits.")
    output = outputs[0]
    if output.get("name") != "logits" or output.get("shape") != [1, NUM_CLASSES]:
        raise ValueError("Triton output contract không khớp shape (1, 9).")
    data = output.get("data")
    if not isinstance(data, list) or len(data) != NUM_CLASSES:
        raise ValueError("Triton output logits không đủ 9 giá trị.")


def _rollback_candidate(
    client: Any,
    config: DeployConfig,
    candidate: CandidateBundle,
    champion: dict[str, Any] | None,
    *,
    registered: bool,
    published: bool,
    error: Exception,
) -> tuple[str, ...]:
    """Remove only the failed Triton version and retain registry evidence for audit."""
    deleted = ()
    if published:
        deleted = tuple(
            delete_prefix(
                client,
                config.registry.triton_bucket,
                triton_version_prefix(candidate.version),
            )
        )
    if registered:
        put_json(
            client,
            config.registry.registry_bucket,
            deployment_key(candidate.version),
            {
                "version": candidate.version,
                "status": "failed",
                "error": str(error),
                "rolled_back_triton_keys": list(deleted),
                "previous_production_version": champion.get("version") if champion else None,
                "failed_at": datetime.now(UTC).isoformat(),
            },
        )
    if champion is not None and not triton_version_ready(
        config.triton_http_url, int(champion["version"])
    ):
        raise RuntimeError("Rollback không xác nhận được previous production model ready.")
    return deleted


def _successful_status(candidate: CandidateBundle) -> dict[str, Any]:
    """Build the pointer/audit payload written only after readiness and smoke pass."""
    return {
        "format_version": 1,
        "version": candidate.version,
        "status": "deployed",
        "checkpoint_sha256": candidate.metadata["checkpoint_sha256"],
        "mlflow_run_id": candidate.metadata.get("mlflow_run_id"),
        "git_sha": candidate.metadata["git_sha"],
        "validation_metrics": candidate.validation_metrics,
        "deployed_at": datetime.now(UTC).isoformat(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON artifact and reject non-object payloads."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact không phải dictionary: {path}")
    return payload


def _sha256(path: Path) -> str:
    """Calculate one streaming file checksum without retaining model bytes in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_from_env(bundle_dir: Path, timeout_seconds: int, poll_seconds: int) -> DeployConfig:
    """Load production-only settings and reject unsafe local default credentials."""
    missing = [name for name in REQUIRED_DEPLOY_ENV if not os.getenv(name)]
    if missing:
        raise ValueError(f"Thiếu protected deploy environment variables: {', '.join(missing)}")
    registry = RegistrySettings(
        endpoint=os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        registry_bucket=os.getenv("REGISTRY_BUCKET", "model-registry"),
        triton_bucket=os.getenv("TRITON_BUCKET", "models"),
    )
    return DeployConfig(
        bundle_dir=bundle_dir.expanduser().resolve(),
        registry=registry,
        triton_http_url=os.environ["TRITON_HTTP_URL"],
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


def parse_args() -> argparse.Namespace:
    """Parse protected deployment arguments; credentials only come from environment."""
    parser = argparse.ArgumentParser(description="Deploy one quality-gated model bundle.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for a GitHub Environment-protected deployment job."""
    args = parse_args()
    try:
        result = run_deployment(
            config_from_env(args.bundle_dir, args.timeout_seconds, args.poll_seconds)
        )
    except (
        ClientError,
        ObjectAlreadyExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Deploy lỗi: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
