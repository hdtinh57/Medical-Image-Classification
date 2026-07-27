# Responsible AI Report — Medical Image Classification

## 1. Overview

This report documents the ethical, fairness, and privacy considerations for the
9-class skin-lesion classification system. The model is designed as a
**decision-support tool** to assist dermatologists in screening skin lesions —
it is **not** an autonomous diagnostic device.

### Scope

| Aspect | Description |
|---|---|
| **Task** | Classify dermatoscopic images into 9 skin lesion categories |
| **Dataset** | ISIC Skin Cancer — 2,357 images, 9 classes |
| **Users** | Dermatologists and clinical support staff |
| **Deployment** | REST API behind hospital network; clinician reviews all predictions |

---

## 2. Fairness Analysis and Bias Detection

### 2.1 Class-Level Disparity

The model's performance varies significantly across the 9 diagnostic categories.
We measure fairness through **per-class precision, recall, and F1-score** since
different classes carry different clinical risks.

**Clinical risk classification:**

| Risk Level | Classes | Concern |
|---|---|---|
| 🔴 **High** | melanoma, basal cell carcinoma, squamous cell carcinoma | False negatives delay cancer treatment |
| 🟡 **Medium** | actinic keratosis | Precancerous; monitoring required |
| 🟢 **Low** | nevus, dermatofibroma, pigmented/seborrheic keratosis, vascular lesion | Benign; false positives cause unnecessary anxiety |

**Key fairness metrics** (computed by `responsible_ai/fairness.py`):

- **F1 range**: measures the gap between best and worst performing classes.
  A large range (>0.3) indicates the model systematically underperforms on
  certain conditions.
- **High-risk recall**: the minimum recall among malignant classes. Values below
  0.5 are clinically concerning.
- **Dangerous confusion pairs**: cases where a malignant lesion (high-risk) is
  misclassified as benign (low-risk), potentially delaying diagnosis.

### 2.2 Dataset Bias Sources

| Bias Type | Present? | Mitigation |
|---|---|---|
| **Class imbalance** | ✅ Yes — `nevus` dominates, rare classes underrepresented | Weighted cross-entropy loss, balanced accuracy metric |
| **Skin tone diversity** | ⚠️ Limited — ISIC dataset skews toward lighter skin | Acknowledged limitation; needs diverse validation set |
| **Imaging conditions** | ⚠️ Variable — dermatoscope vs clinical photo quality | Moderate augmentation (crop, flip, rotation, color jitter) |
| **Geographic representation** | ⚠️ Unknown — dataset provenance not fully documented | Cannot assess; needs external validation |
| **Label quality** | ⚠️ SHA-256 hash groups reveal duplicate content across splits | Grouped split prevents train/val leakage |

### 2.3 Mitigation Strategies Implemented

1. **Balanced class weights** in loss function (`training/train.py`)
2. **StratifiedGroupKFold** preventing content leakage across train/val
3. **Balanced accuracy** as a gating metric alongside macro F1
4. **Quality gate** with minimum thresholds before any deployment
5. **Per-class confusion matrix** in every evaluation for audit

---

## 3. Model Explainability

### 3.1 Grad-CAM Visual Explanations

The system implements **Gradient-weighted Class Activation Mapping (Grad-CAM)**
(`gateway/gradcam.py`) to provide visual explanations of model predictions.

**How it works:**

1. Hook into the last convolutional layer of the classifier (ResNet `layer4` or
   ConvNeXt `stages[-1]`).
2. Compute gradients of the target class logit with respect to feature-map activations.
3. Weight each feature channel by its average gradient, producing a spatial heatmap.
4. Overlay the heatmap on the original image to show which regions drove the prediction.

**Why Grad-CAM for medical imaging:**

- Clinicians need to verify the model is looking at the **lesion**, not artifacts
  (ruler marks, ink, hair, skin markers).
- Heatmaps build trust: if the model highlights the dermoscopic pattern a doctor
  would examine, the prediction is more credible.
- Regulatory requirements (EU AI Act, FDA guidance) increasingly require
  explainability for AI-assisted clinical decisions.

### 3.2 Confidence Distribution Monitoring

The gateway exposes a Prometheus histogram (`gateway_prediction_confidence`)
that tracks the distribution of top-1 confidence scores. A sudden shift toward
low-confidence predictions may indicate:

- Data drift (new imaging equipment, different patient population)
- Model degradation requiring retraining

---

## 4. Data Privacy Considerations

### 4.1 Dataset Privacy

| Concern | Assessment |
|---|---|
| **Patient identifiable information (PII)** | ISIC images are deidentified; no patient names, dates, or facility markers |
| **Image metadata** | EXIF data should be stripped before training; EDA does not extract EXIF |
| **Consent** | ISIC dataset is publicly available under research terms |

### 4.2 Deployment Privacy

| Measure | Implementation |
|---|---|
| **No image storage** | Gateway processes images in-memory; no uploaded image is persisted to disk or object storage |
| **Network isolation** | Gateway communicates with Triton via internal Docker network; no external API keys |
| **Credential protection** | MinIO/Triton credentials are environment-scoped secrets, never logged or committed |
| **Audit trail** | Model registry keeps immutable deployment records without patient data |

### 4.3 HIPAA Considerations

If deployed in a clinical setting:

- ⚠️ Images uploaded through the API may constitute Protected Health Information (PHI)
- ⚠️ The current system lacks encryption at rest, access logging, and BAA compliance
- **Recommendation**: Deploy behind a HIPAA-compliant API gateway with TLS, audit
  logging, and data retention policies before any clinical use

---

## 5. Ethical Implications

### 5.1 Clinical Safety

| Risk | Severity | Mitigation |
|---|---|---|
| **False negative melanoma** | 🔴 Critical — delayed cancer diagnosis | Quality gate requires minimum recall; model is advisory only |
| **False positive benign** | 🟡 Medium — unnecessary biopsy/anxiety | Confidence thresholds; clinician confirms all results |
| **Over-reliance** | 🟡 Medium — clinician defers to model | Clear UI labeling: "AI-assisted screening — clinician review required" |
| **Scope creep** | 🟡 Medium — using model outside intended population | Training data limited to dermatoscopic images; reject non-dermatology input |

### 5.2 Deployment Guardrails

1. **Human-in-the-loop**: Every prediction requires clinician review and confirmation
2. **Scope labeling**: The API response does not say "diagnosis" — it says "predicted class"
3. **Confidence disclosure**: Full probability distribution is returned, not just top-1
4. **Model versioning**: Immutable registry ensures traceability of which model produced each prediction
5. **Performance monitoring**: Prometheus/Grafana dashboards detect degradation before patient impact

### 5.3 Equity Concerns

- **Skin tone bias**: Dermatology AI trained predominantly on lighter skin tones may
  underperform on darker skin. The ISIC dataset does not include Fitzpatrick skin type
  metadata, preventing direct assessment.
- **Access inequality**: The system requires hospital infrastructure (GPU serving, network);
  rural or under-resourced clinics may not benefit equally.
- **Recommendation**: Before clinical deployment, validate on a geographically and
  demographically diverse dataset with Fitzpatrick type annotations.

---

## 6. Limitations and Future Work

| Limitation | Status | Plan |
|---|---|---|
| No Fitzpatrick skin type stratification | Not possible with current dataset | Validate on diverse dataset |
| No model drift detection | ⚠️ Partial (confidence histogram only) | Add statistical drift tests |
| No adversarial robustness testing | Not implemented | Add perturbation tests |
| No patient outcome tracking | Not implemented | Would require clinical integration |
| No A/B testing framework | Not implemented | Add canary deployment support |

---

## 7. Compliance Checklist

| Requirement | Status |
|---|---|
| ✅ Fairness analysis code (`responsible_ai/fairness.py`) | Implemented |
| ✅ Per-class disparity metrics and visualization | Implemented |
| ✅ Explainability via Grad-CAM (`gateway/gradcam.py`) | Implemented |
| ✅ Data privacy considerations documented | This report |
| ✅ Ethical implications discussed | This report |
| ✅ Clinical safety guardrails defined | This report |
| ⚠️ Fitzpatrick skin-type stratified evaluation | Dataset limitation |
| ⚠️ HIPAA-compliant deployment architecture | Requires infrastructure upgrade |
