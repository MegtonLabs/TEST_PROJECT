# Cheque Signature Verification System

A modular, production-ready pipeline that combines two state-of-the-art AI
components to verify handwritten signatures on bank cheques:

| Component | Role |
|---|---|
| [Gemma4-Visual-Agent](https://github.com/PromtEngineer/Gemma4-Visual-Agent/tree/dgx-spark-gb10) (Gemma 4 E4B + Falcon Perception) | Stage 1 — Cheque field extraction & signature localisation |
| [siddharth-magesh/siamese-signature-verification](https://huggingface.co/siddharth-magesh/siamese-signature-verification) | Stage 2 — Biometric signature comparison |

---

## Architecture

```
                  ┌─────────────────────────────────────────┐
  cheque.jpg ──▶  │  Stage 1: ChequeAnalyzer                │
                  │  ┌─────────────────────────────────────┐ │
                  │  │  Gemma 4 E4B (VLM)                  │ │
                  │  │  • Is this a cheque?                 │ │
                  │  │  • Extract: bank, date, payee,       │ │
                  │  │    amount (figures + words), MICR    │ │
                  │  │  • Return signature bounding box     │ │
                  │  └──────────────────┬──────────────────┘ │
                  │                     │ JSON output         │
                  │  ┌──────────────────▼──────────────────┐ │
                  │  │  Falcon Perception (fallback)        │ │
                  │  │  • Detects "handwritten signature"   │ │
                  │  │    if VLM bbox is missing            │ │
                  │  └──────────────────┬──────────────────┘ │
                  └─────────────────────┼───────────────────┘
                                        │ BoundingBox
                  ┌─────────────────────▼───────────────────┐
                  │  crop + validate signature region        │
                  │  (variance check, ink-ratio, erosion)    │
                  └─────────────────────┬───────────────────┘
                                        │ cropped PIL Image
                  ┌─────────────────────▼───────────────────┐
  ref_sigs[] ──▶  │  Stage 2: SignatureVerifier             │
                  │  ┌─────────────────────────────────────┐ │
                  │  │  Siamese CNN                        │ │
                  │  │  • Embed query + references         │ │
                  │  │  • Cosine similarity                │ │
                  │  │  • Threshold → genuine / forged     │ │
                  │  └─────────────────────────────────────┘ │
                  └─────────────────────────────────────────┘
                                        │
                              VerificationResult
```

---

## Repository layout

```
.
├── config.py                   # All tuneable settings (model IDs, thresholds, …)
├── pipeline.py                 # End-to-end orchestration + CLI entry point
├── requirements.txt
│
├── stages/
│   ├── cheque_analyzer.py      # Stage 1 — Visual Agent wrapper
│   └── signature_verifier.py  # Stage 2 — Siamese verifier
│
└── utils/
    ├── image_utils.py          # Cropping, preprocessing, validation
    └── result_types.py         # ChequeData, VerificationResult, …
```

---

## Setup

### 1. System requirements

- Python 3.10+
- NVIDIA GPU with CUDA (Gemma 4 E4B requires ~8 GB VRAM; 16 GB recommended)
- HuggingFace account — accept the [Gemma licence](https://huggingface.co/google/gemma-4-E4B-it) and run `huggingface-cli login`

### 2. Clone the Visual Agent

The Visual Agent code is used **exactly as-is** — no modifications.

```bash
git clone https://github.com/PromtEngineer/Gemma4-Visual-Agent.git
export VISUAL_AGENT_REPO_PATH=/path/to/Gemma4-Visual-Agent/dgx_spark_gb10
```

### 3. Install dependencies

```bash
# PyTorch (adjust the CUDA version tag to match your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Core dependencies
pip install -r requirements.txt

# Falcon Perception (Visual Agent dependency)
pip install "falcon-perception[torch] @ git+https://github.com/tiiuae/falcon-perception.git"
```

### 4. (Optional) Verify the Visual Agent

```bash
cd $VISUAL_AGENT_REPO_PATH
python3 -m py_compile agent_studio.py
```

---

## Quick start

### Python API

```python
from pipeline import ChequeVerificationPipeline
from utils.image_utils import load_image

pipeline = ChequeVerificationPipeline()

result = pipeline.run(
    cheque_image=load_image("cheque.jpg"),
    reference_images=[
        load_image("reference_sig_1.png"),
        load_image("reference_sig_2.png"),
    ],
)

import json
print(json.dumps(result.to_dict(), indent=2))
```

**Example output**

```json
{
  "cheque_data": {
    "is_cheque": true,
    "bank_name": "State Bank of India",
    "date": "01/05/2026",
    "payee_name": "John Doe",
    "amount_numeric": "25,000.00",
    "amount_words": "Twenty Five Thousand Only",
    "micr_line": "000123456789 001234 56789012",
    "account_number": "1234567890",
    "cheque_number": "000456",
    "signature_region": {
      "bbox": {"x1": 540, "y1": 380, "x2": 780, "y2": 460},
      "confidence": 1.0,
      "validation_passed": true,
      "validation_note": "Signature region validation passed."
    }
  },
  "reference_scores": [0.923456, 0.918234],
  "similarity_score": 0.920845,
  "verdict": "genuine",
  "confidence": 0.472532,
  "reason": "Aggregated similarity 0.9208 >= threshold 0.85 (mean of 2 reference(s)).",
  "error": null
}
```

### Command-line interface

```bash
python pipeline.py cheque.jpg ref1.png ref2.png
# With a custom threshold:
python pipeline.py cheque.jpg ref1.png --threshold 0.80
# Enable VLM signature-region validation:
python pipeline.py cheque.jpg ref1.png --vlm-validate
```

---

## Configuration

All settings live in `config.py`.  Override them via environment variables or
by editing the dataclass defaults:

| Environment variable | Default | Description |
|---|---|---|
| `VISUAL_AGENT_REPO_PATH` | `./Gemma4-Visual-Agent/dgx_spark_gb10` | Path to the cloned Visual Agent |
| `GEMMA_HF_MODEL_ID` | `google/gemma-4-E4B-it` | HuggingFace model ID for Gemma |
| `FALCON_HF_MODEL_ID` | `tiiuae/Falcon-Perception` | HuggingFace model ID for Falcon |
| `SIAMESE_MODEL_ID` | `siddharth-magesh/siamese-signature-verification` | Siamese model |
| `SIAMESE_DEVICE` | `auto` | `cuda` / `cpu` / `auto` |
| `SIG_SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity threshold |
| `SIG_MULTI_REF_AGG` | `mean` | `mean` / `max` / `min` |
| `SIG_MIN_VARIANCE_RATIO` | `0.001` | Blank-region rejection threshold |
| `SIG_VLM_VALIDATION` | `0` | `1` to enable VLM signature validation |

---

## Pipeline stages in detail

### Stage 1 — ChequeAnalyzer (`stages/cheque_analyzer.py`)

1. Calls `run_gemma_reasoning(image, prompt)` from the Visual Agent with a
   structured JSON prompt requesting all cheque fields **and** the signature
   bounding box.
2. Parses the JSON response via a robust multi-strategy extractor.
3. If the VLM does not return a valid bounding box, falls back to
   `run_falcon_perception(image, "handwritten signature")` to detect the
   signature region.
4. Returns a `ChequeData` object with all fields populated.

> **No Visual Agent code is modified.**  The stage only calls the two public
> functions (`run_gemma_reasoning`, `run_falcon_perception`) exposed by
> `agent_studio.py` from the `dgx_spark_gb10/` directory.

### Stage 2 — SignatureVerifier (`stages/signature_verifier.py`)

1. Loads `siddharth-magesh/siamese-signature-verification` from HuggingFace.
   Three loader strategies are tried in order:
   - `transformers.AutoModel.from_pretrained`
   - Plain `torch.load` of `model.safetensors` / `pytorch_model.bin`
   - Fallback to a built-in `_SiameseCNN` backbone (architecture mirrors
     common signature verification networks)
2. Computes L2-normalised embedding vectors for the query and all reference
   signatures via `embed()`.
3. Computes cosine similarities and aggregates (mean/max/min).
4. Applies the configurable threshold to produce a `"genuine"` / `"forged"`
   / `"undetermined"` verdict with a confidence score.

### Signature validation (`utils/image_utils.py`)

Before the Siamese comparison the cropped region is validated with three
lightweight checks:

| Check | What it catches |
|---|---|
| Pixel variance | Blank / near-uniform regions |
| Dark-ink ratio | Empty or fully black images |
| Morphological erosion | Noise / very thin disconnected pixels |

An optional fourth check (`--vlm-validate`) uses the VLM to confirm the crop
contains a handwritten signature.

---

## Tuning the similarity threshold

The default threshold of **0.85** is a conservative starting point.  To tune
it on your own labelled dataset:

```python
from stages.signature_verifier import SignatureVerifier
from config import SiameseConfig, VerificationConfig

verifier = SignatureVerifier(SiameseConfig(), VerificationConfig())

# Build embeddings for all genuine and forged pairs
# genuine_scores = [verifier.verify(q, refs)["similarity_score"] for q, refs in genuine_pairs]
# forged_scores  = [verifier.verify(q, refs)["similarity_score"] for q, refs in forged_pairs]

# Choose the threshold that maximises F1 or minimises EER on your validation set.
```

---

## Extending the pipeline

- **Custom backbone** — subclass `torch.nn.Module`, implement `forward(x) → embedding`, and pass an instance to `SignatureVerifier._model`.
- **Different VLM** — swap the model ID in `VisualAgentConfig.gemma_model_id`.
- **REST API** — wrap `ChequeVerificationPipeline.run()` in a FastAPI endpoint; call `pipeline.warm_up()` in the `lifespan` startup hook.
