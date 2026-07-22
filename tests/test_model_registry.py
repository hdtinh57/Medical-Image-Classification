"""Tests for immutable MinIO registry operations using an in-memory S3 fake."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from training.model import MODEL_NAME
from training.model_registry import (
    ObjectAlreadyExistsError,
    RegistrySettings,
    delete_prefix,
    get_json,
    latest_version,
    put_json,
    register_checkpoint,
)


class FakeS3Client:
    """Minimal S3 fake covering registry behavior without Docker or MinIO."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.buckets: set[str] = set()

    def head_bucket(self, *, Bucket: str) -> None:
        if Bucket not in self.buckets:
            raise _not_found("HeadBucket")

    def create_bucket(self, *, Bucket: str) -> None:
        self.buckets.add(Bucket)

    def head_object(self, *, Bucket: str, Key: str) -> None:
        if (Bucket, Key) not in self.objects:
            raise _not_found("HeadObject")

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        self.buckets.add(bucket)
        self.objects[(bucket, key)] = Path(path).read_bytes()

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: object) -> None:
        self.buckets.add(Bucket)
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, BytesIO]:
        try:
            return {"Body": BytesIO(self.objects[(Bucket, Key)])}
        except KeyError as exc:
            raise _not_found("GetObject") from exc

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


def _not_found(operation: str) -> ClientError:
    return ClientError({"Error": {"Code": "404", "Message": "not found"}}, operation)


def test_register_checkpoint_is_immutable_and_persists_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "best_checkpoint.pth"
    gate = tmp_path / "quality_gate.json"
    checkpoint.write_bytes(b"checkpoint")
    gate.write_text("{}", encoding="utf-8")
    client = FakeS3Client()
    settings = RegistrySettings(registry_bucket="registry")

    register_checkpoint(
        client,
        settings,
        version=2,
        checkpoint_path=checkpoint,
        candidate_metadata={"model_version": 2},
        quality_gate_path=gate,
    )

    assert latest_version(client, "registry") == 2
    assert get_json(client, "registry", f"{MODEL_NAME}/2/candidate.json") == {"model_version": 2}
    with pytest.raises(ObjectAlreadyExistsError):
        register_checkpoint(
            client,
            settings,
            version=2,
            checkpoint_path=checkpoint,
            candidate_metadata={"model_version": 2},
            quality_gate_path=gate,
        )


def test_delete_prefix_removes_only_failed_candidate_version() -> None:
    client = FakeS3Client()
    bucket = "models"
    client.objects[(bucket, f"{MODEL_NAME}/1/model.onnx")] = b"previous"
    client.objects[(bucket, f"{MODEL_NAME}/2/model.onnx")] = b"failed"
    client.objects[(bucket, f"{MODEL_NAME}/config.pbtxt")] = b"config"

    deleted = delete_prefix(client, bucket, f"{MODEL_NAME}/2")

    assert deleted == [f"{MODEL_NAME}/2/model.onnx"]
    assert (bucket, f"{MODEL_NAME}/1/model.onnx") in client.objects
    assert (bucket, f"{MODEL_NAME}/config.pbtxt") in client.objects


def test_get_json_returns_none_for_missing_production_pointer() -> None:
    client = FakeS3Client()
    assert get_json(client, "registry", f"{MODEL_NAME}/production.json") is None

    put_json(client, "registry", f"{MODEL_NAME}/production.json", {"version": 1})
    assert get_json(client, "registry", f"{MODEL_NAME}/production.json") == {"version": 1}
