# NeuroClassify

//frontend:  streamlit run app.py
//backend:   uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

Decision-support system to classify brain MRI into four classes: `glioma`, `meningioma`, `pituitary`, and `normal`.

This repository now provides two complementary paths:

- API-based app (recommended):
  - FastAPI backend that proxies to an external inference API or OpenRouter.
  - Streamlit frontend for clinicians to upload an image, view probabilities, and see explainability info.

- Legacy local training (optional):
  - ResNet50 fine-tuning pipeline with preprocessing, augmentation, CV, metrics, and Captum Grad-CAM.

> Disclaimer: For clinical decision support only — not a diagnostic replacement. Clinical decisions must be made by qualified physicians.

## Project Structure
```
NeuroClassify/
├─ api_app/
│  ├─ backend/
│  │  ├─ main.py              # FastAPI app (/health, /predict). Calls external API or OpenRouter
│  │  └─ settings.py          # Reads env vars (.env) for configuration
│  └─ frontend/
│     └─ app.py               # Streamlit UI: upload image, get probabilities
│
├─ configs/
│  └─ config.yaml              # Main config (paths, CV, training, logging)
├─ data/
│  ├─ raw/                     # Put class folders here (or use dummy generator)
│  └─ processed/               # Preprocessed output
├─ results/
│  ├─ checkpoints/             # Saved models
│  ├─ logs/                    # TensorBoard logs
│  ├─ runs/                    # Misc run artifacts
│  └─ figures/                 # Grad-CAM and plots
├─ src/
│  ├─ preprocess.py            # Preprocess (skull-strip heuristic) CLI
│  ├─ data.py                  # Dataset + transforms + preprocessing helpers
│  ├─ model.py                 # ResNet50Classifier
│  ├─ train.py                 # K-fold training loop + early stopping
│  ├─ evaluate.py              # Metrics and confusion matrix helper
│  ├─ explain.py               # Captum LayerGradCam visualizations
│  └─ make_dummy_data.py       # Synthetic data generator for quick testing
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## Setup (Windows)
You can use either Command Prompt (CMD) or PowerShell. Use one shell consistently in a session.

### 1) Create a Python 3.10 virtual environment
- CMD
```cmd
cd /d d:\git_projects\NeuroClassify
py -3.10 -m venv .venv310
.\.venv310\Scripts\activate.bat
```
- PowerShell
```powershell
Set-Location d:\git_projects\NeuroClassify
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
# If activation is blocked, run once per session:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2) Install dependencies
```bash
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

CPU-only PyTorch (if CUDA is not available):
```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
python -m pip install -r requirements.txt
```

CUDA 11.8 wheels (only if you have matching NVIDIA drivers):
```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
python -m pip install -r requirements.txt
```

## API-based App (Backend + Frontend)

### 1) Configure environment (.env)
Create `api_app/.env` (or copy from `api_app/.env.example`) and fill values. Example for OpenRouter:

```bash
OPENROUTER_API_KEY=sk-or-...your key...
OPENROUTER_MODEL=google/gemini-2.5-pro
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_APP_TITLE=NeuroClassify-API
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_MAX_TOKENS=128
```

Or, to use your own inference API instead of OpenRouter:
```bash
MODEL_API_URL=https://your-inference-provider/api/v1/predict
MODEL_API_KEY=your-provider-api-key
```

### 2) Run backend (FastAPI)
Run from the repository root with module path:
```powershell
Set-Location d:\git_projects\NeuroClassify
.\.venv310\Scripts\Activate.ps1
uvicorn api_app.backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Health check and docs:
```bash
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

Tip: If you run from inside `api_app/`, use `uvicorn backend.main:app ...` so Python resolves the package correctly.

### 3) Run frontend (Streamlit)
In a second terminal:
```powershell
Set-Location d:\git_projects\NeuroClassify
.\.venv310\Scripts\Activate.ps1
streamlit run api_app/frontend/app.py
```
In the sidebar set Backend URL = `http://127.0.0.1:8000`, ping `/health`, upload a PNG/JPG, and click Predict.

### OpenRouter notes
- The backend caps `max_tokens` via `OPENROUTER_MAX_TOKENS` (default 128) to reduce credit usage.
- If provider returns non-JSON or empty content, the backend now gracefully falls back to simulated probabilities so the UI remains functional.
- For deterministic production inference, prefer a dedicated vision inference API via `MODEL_API_URL`.

## Data Preparation
The training code expects a 2D image dataset organized as:
```
data/raw/
├─ glioma/
├─ meningioma/
├─ pituitary/
└─ normal/
```
with PNG/JPG images in each folder. For DICOM/NIfTI volumes, convert representative slices or adapt the loader.

Quick start without real data (dummy generator):
```bash
python -m src.make_dummy_data --out data/raw --n 100 --size 256
```

## Preprocessing
Runs a fast skull-stripping heuristic (Otsu + morphology) for 2D images and writes results to `data/processed/`.
```bash
python -m src.preprocess --raw data/raw --out data/processed
# To skip skull-stripping:
# python -m src.preprocess --raw data/raw --out data/processed --no_skull_strip
```

## Training
By default, uses 5-fold stratified cross-validation, early stopping on macro-F1, cosine LR scheduling, and AMP on CUDA.
```bash
python -m src.train
```
Outputs:
- Checkpoints per fold: `results/checkpoints/resnet50_fold{K}.pt`
- TensorBoard logs: `results/logs/`

View training curves with TensorBoard:
```bash
tensorboard --logdir results/logs
```

If you are on CPU only, the code automatically falls back to CPU. You can also set `device: cpu` in `configs/config.yaml`.

## Evaluation
Evaluate a saved checkpoint on a directory of images.
```bash
python -m src.evaluate --ckpt results/checkpoints/resnet50_fold0.pt \
  --data_dir data/processed \
  --out results/metrics_fold0.json
```
The JSON includes accuracy, macro/weighted F1, per-class metrics, macro AUC, and the confusion matrix.

## Explainability (Grad-CAM via Captum)
Generate and save a Grad-CAM heatmap overlay for an image using a trained checkpoint.
```bash
python -m src.explain --ckpt results/checkpoints/resnet50_fold0.pt \
  --image data/processed/glioma/glioma_00000.png \
  --out results/figures/cam.png
```

## Configuration
Edit `configs/config.yaml` to control:
- `paths`: raw/processed data and results directories
- `cv`: k-folds, shuffle, seed
- `img_size`, `batch_size`, `epochs`, `num_workers`
- `optimizer`: AdamW or SGD; `lr`, `weight_decay`
- `scheduler`: CosineAnnealingLR or ReduceLROnPlateau
- `train`:
  - `early_stopping_patience`
  - `gradient_clip_norm`
  - `amp` (automatic mixed precision on CUDA)
  - `label_smoothing`
  - `freeze_until_layer` (0 = full fine-tune)
- `logging`: TensorBoard / (opt) Weights & Biases

## Tips to Improve Performance (on real data)
- Use patient-level splits to avoid leakage.
- Increase `img_size` (e.g., 320–384) if GPU memory allows.
- Add stronger augmentation (GaussianNoise/Blur, CoarseDropout) in `src/data.py`.
- Consider MixUp/CutMix, warmup + cosine LR, and model EMA (can be added if needed).
- Validate on external holdout data for generalization.

## Ethics & Safety
- Address dataset bias: report subgroup metrics (demographics, scanner/institution) if available.
- Privacy: de-identify DICOM headers; avoid storing PHI in logs or filenames; secure raw data.
- Disclaimer: This tool is for clinical decision support only.

## Troubleshooting
- **Backend root shows Not Found**: open `/health` or `/docs`, or use the Streamlit frontend which calls `/predict`.
- **ModuleNotFoundError: api_app**: run Uvicorn from repo root with `api_app.backend.main:app`, or from inside `api_app/` with `backend.main:app`.
- **OpenRouter 400/402/500**: lower `OPENROUTER_MAX_TOKENS` (64–128), try a cheaper model (e.g., `google/gemini-flash-1.5`), or remove `OPENROUTER_API_KEY` to use simulated mode.
- **No images found during training**: ensure `data/processed/` has images; run preprocessing.
- **CUDA not available**: install CPU-only PyTorch or set `device: cpu`.
- **Large files in Git push**: rely on `.gitignore` or use Git LFS for model artifacts.

## Git & Large Files
`.gitignore` excludes `data/` and `results/` by default. If you need to version model files, use Git LFS:
```bash
git lfs install
git lfs track "*.pt" "*.pth" "*.onnx" "*.ckpt" "*.bin"
# commit .gitattributes added by LFS
```

## License
This project is provided under the LICENSE included in this repository.
