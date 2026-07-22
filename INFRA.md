# INFRA — MinIO, Triton và Protected Model Deployment

## Purpose

Docker Compose supplies the persistent serving and observability runtime:

```text
MinIO → Triton (ONNX Runtime) → Prometheus → Grafana
```

Training and candidate selection are performed by `training/pipeline.py` on a trusted CUDA runner.
The release workflow then promotes a checksummed candidate through a protected deployment job. The
Compose stack does **not** train models, hold CI artifacts, or restart automatically on every model
release.

## Buckets and version contract

| Bucket | Key layout | Writer | Rule |
|---|---|---|---|
| `model-registry` | `skin_classifier/<N>/best_checkpoint.pth` | `training.deploy` | Immutable checkpoint version |
| `model-registry` | `skin_classifier/<N>/{candidate,quality_gate,deployment}.json` | pipeline/deployer | Candidate provenance and audit |
| `model-registry` | `skin_classifier/production.json` | deployer | Mutable pointer written only after successful smoke test |
| `models` | `skin_classifier/config.pbtxt`, `labels.txt`, `<N>/model.onnx` | exporter/deployer | Triton model repository |

`<N>` is a positive, monotonically increasing integer. Deployment refuses to overwrite a checkpoint or ONNX version, and rejects a candidate that is not newer than every registry/Triton version so unversioned `latest` routing cannot move backward. Triton is configured with `version_policy` to keep two latest versions, so the previous model remains available while a new candidate is validated.

## Start the local runtime

Requirements: Docker Desktop is running and no unrelated process is using ports 9000, 9001, 8000,
8001, 8002, 9090 or 3000.

```powershell
docker compose up -d
docker compose ps
```

Expected state:

- `minio`, `triton`, `prometheus`, `grafana`: running;
- `minio-init`: exited with code `0` after creating buckets;
- Triton may be healthy but return model `404` before a model version is published. This is normal:
  it polls MinIO every 30 seconds and uses `--exit-on-error=false`.

Development URLs:

| Service | URL |
|---|---|
| MinIO console | http://localhost:9001 |
| Triton REST | http://localhost:8000 |
| Triton metrics | http://localhost:8002/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

The Compose credentials are development-only defaults. Do not reuse them in a public or production
deployment.

## Normal automated promotion path

```text
Train candidate model workflow (trusted CUDA runner)
→ candidate-bundle artifact
→ Release model package job (checksum + quality gate + ONNX parity)
→ GitHub Environment model-production approval
→ protected deploy runner
→ MinIO registry + Triton repository
→ Triton ready + `(1, 9)` smoke inference
→ production.json updated
```

The protected deploy job requires these **environment-scoped** secrets, configured in GitHub
Environment `model-production`:

```text
MINIO_ENDPOINT
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
TRITON_HTTP_URL
```

The release package job never receives these secrets. It verifies the artifact from a successful
trusted `main` training workflow before the protected job is even eligible to run.

## Local operator recovery path

Use this only after the runtime is already up and when an authorized operator has received explicit
credentials. The command refuses to use fallback demo credentials:

```powershell
$env:MINIO_ENDPOINT = "https://minio.example.internal"
$env:MINIO_ACCESS_KEY = "<authorized-access-key>"
$env:MINIO_SECRET_KEY = "<authorized-secret>"
$env:TRITON_HTTP_URL = "https://triton.example.internal"

python -m training.deploy --bundle-dir artifacts\pipeline\candidate\v2
```

The deployment transaction does this in order:

1. validates every candidate-bundle checksum and repeats the Validation quality gate;
2. compares macro F1 with `production.json` when a champion exists;
3. registers immutable checkpoint/provenance/gate objects in `model-registry`;
4. exports ONNX and verifies PyTorch/ONNX Runtime parity;
5. publishes the exact ONNX version to `models`;
6. waits no more than 60 seconds for
   `/v2/models/skin_classifier/versions/<N>/ready`;
7. calls version-specific inference and requires output `(1, 9)`;
8. updates `production.json` only if every previous step succeeds.

If Triton readiness or smoke inference fails, the transaction deletes only
`models/skin_classifier/<N>/`. It retains the candidate checkpoint and writes a failed deployment
record under `model-registry` for audit. It never kills/restarts the Compose stack and never deletes
the prior production version.

## Health verification

```powershell
curl.exe http://localhost:8000/v2/health/ready
curl.exe http://localhost:8000/v2/models/skin_classifier/ready
curl.exe http://localhost:8000/v2/models/skin_classifier
curl.exe http://localhost:8002/metrics
```

To target a known version after it is deployed, use:

```text
/v2/models/skin_classifier/versions/<N>/ready
/v2/models/skin_classifier/versions/<N>/infer
```

Prometheus scrapes Triton every 15 seconds. Grafana provisions the serving dashboard at startup.
`GatewayDown` is currently expected because the FastAPI gateway has not been implemented or added
to Compose.

## Stop the local runtime

```powershell
docker compose down       # retains MinIO and Grafana volumes
docker compose down -v    # destructive: removes local MinIO/Grafana volumes
```

Only run the second command when an authorized operator intentionally wants to discard local model
and dashboard state.
