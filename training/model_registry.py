"""MinIO-backed immutable checkpoint registry and deployment metadata helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from training.model import MODEL_NAME

DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
DEFAULT_MINIO_ACCESS_KEY = "minioadmin"
DEFAULT_MINIO_SECRET_KEY = "minioadmin"
DEFAULT_REGISTRY_BUCKET = "model-registry"
DEFAULT_TRITON_BUCKET = "models"


class ObjectAlreadyExistsError(RuntimeError):
    """Raised when an immutable model-version object already exists."""


@dataclass(frozen=True)
class RegistrySettings:
    """Connection and bucket settings shared by export and deployment."""

    endpoint: str = DEFAULT_MINIO_ENDPOINT
    access_key: str = DEFAULT_MINIO_ACCESS_KEY
    secret_key: str = DEFAULT_MINIO_SECRET_KEY
    registry_bucket: str = DEFAULT_REGISTRY_BUCKET
    triton_bucket: str = DEFAULT_TRITON_BUCKET

    @classmethod
    def from_env(cls) -> RegistrySettings:
        """Load local-development defaults with optional environment overrides."""
        return cls(
            endpoint=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT),
            access_key=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY),
            secret_key=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY),
            registry_bucket=os.getenv("REGISTRY_BUCKET", DEFAULT_REGISTRY_BUCKET),
            triton_bucket=os.getenv("TRITON_BUCKET", DEFAULT_TRITON_BUCKET),
        )


def create_s3_client(settings: RegistrySettings) -> Any:
    """Create one S3-compatible client without exposing credentials."""
    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint,
        aws_access_key_id=settings.access_key,
        aws_secret_access_key=settings.secret_key,
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket(client: Any, bucket: str) -> None:
    """Create a bucket only when it does not already exist."""
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        if _error_code(exc) not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        client.create_bucket(Bucket=bucket)


def object_exists(client: Any, bucket: str, key: str) -> bool:
    """Return whether one exact object exists without listing the bucket."""
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def _error_code(exc: ClientError) -> str:
    """Extract the stable service error code from botocore exceptions."""
    return str(exc.response.get("Error", {}).get("Code", ""))


def checkpoint_prefix(version: int) -> str:
    """Return the immutable model-registry prefix for one positive version."""
    return f"{MODEL_NAME}/{validate_version(version)}"


def triton_version_prefix(version: int) -> str:
    """Return the Triton model-repository prefix for one positive version."""
    return f"{MODEL_NAME}/{validate_version(version)}"


def validate_version(version: int) -> int:
    """Validate Triton-compatible positive integer model versions."""
    if version < 1:
        raise ValueError("Model version phải là số nguyên dương.")
    return version


def list_integer_versions(client: Any, bucket: str, model_name: str = MODEL_NAME) -> list[int]:
    """List sorted integer child prefixes for one registered model."""
    prefix = f"{model_name}/"
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    versions = []
    for common_prefix in response.get("CommonPrefixes", []):
        part = common_prefix["Prefix"][len(prefix) :].strip("/")
        if part.isdigit():
            versions.append(int(part))
    return sorted(versions)


def latest_version(client: Any, bucket: str, model_name: str = MODEL_NAME) -> int:
    """Return the latest integer version or fail clearly for an empty registry."""
    versions = list_integer_versions(client, bucket, model_name)
    if not versions:
        raise RuntimeError(f"Không có checkpoint version trong s3://{bucket}/{model_name}/")
    return versions[-1]


def upload_file_immutable(client: Any, bucket: str, key: str, path: Path) -> None:
    """Upload one file only when its destination key is unused."""
    if object_exists(client, bucket, key):
        raise ObjectAlreadyExistsError(f"Object immutable đã tồn tại: s3://{bucket}/{key}")
    client.upload_file(str(path), bucket, key)


def put_json(client: Any, bucket: str, key: str, payload: dict[str, Any]) -> None:
    """Write one JSON metadata object, allowing pointer/status updates."""
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def get_json(client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    """Read one JSON object, returning None when it is absent."""
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    payload = json.loads(response["Body"].read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object không phải dictionary: s3://{bucket}/{key}")
    return payload


def register_checkpoint(
    client: Any,
    settings: RegistrySettings,
    *,
    version: int,
    checkpoint_path: Path,
    candidate_metadata: dict[str, Any],
    quality_gate_path: Path,
) -> None:
    """Register checkpoint and immutable candidate metadata as one version."""
    prefix = checkpoint_prefix(version)
    destinations = {
        f"{prefix}/best_checkpoint.pth": checkpoint_path,
        f"{prefix}/quality_gate.json": quality_gate_path,
    }
    _ensure_destinations_unused(client, settings.registry_bucket, destinations)
    candidate_key = f"{prefix}/candidate.json"
    if object_exists(client, settings.registry_bucket, candidate_key):
        raise ObjectAlreadyExistsError(
            f"Object immutable đã tồn tại: s3://{settings.registry_bucket}/{candidate_key}"
        )
    for key, path in destinations.items():
        client.upload_file(str(path), settings.registry_bucket, key)
    put_json(client, settings.registry_bucket, candidate_key, candidate_metadata)


def _ensure_destinations_unused(
    client: Any,
    bucket: str,
    destinations: dict[str, Path],
) -> None:
    """Preflight all immutable keys before starting a multi-object upload."""
    existing = [key for key in destinations if object_exists(client, bucket, key)]
    if existing:
        joined = ", ".join(f"s3://{bucket}/{key}" for key in existing)
        raise ObjectAlreadyExistsError(f"Model version đã tồn tại: {joined}")


def delete_prefix(client: Any, bucket: str, prefix: str) -> list[str]:
    """Delete only objects below one exact prefix and return deleted keys."""
    normalized_prefix = prefix.rstrip("/") + "/"
    response = client.list_objects_v2(Bucket=bucket, Prefix=normalized_prefix)
    keys = [item["Key"] for item in response.get("Contents", [])]
    for start in range(0, len(keys), 1000):
        batch = [{"Key": key} for key in keys[start : start + 1000]]
        client.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
    return keys


def production_key() -> str:
    """Return the mutable pointer to the deployed champion metadata."""
    return f"{MODEL_NAME}/production.json"


def deployment_key(version: int) -> str:
    """Return the immutable/audit deployment status key for one version."""
    return f"{checkpoint_prefix(version)}/deployment.json"
