# NeuroClassify

ResNet50-based, explainable decision-support pipeline to classify brain MRI into four classes: `glioma`, `meningioma`, `pituitary`, and `normal`. The system is designed for clinical decision support (not a diagnostic replacement) with end-to-end steps: preprocessing, augmentation, training with cross-validation, metrics, and Grad-CAM explanations.

> Disclaimer: For clinical decision support only — not a diagnostic replacement. Clinical decisions must be made by qualified physicians.

## Key Features
- Transfer learning with ResNet50 (ImageNet weights), custom head for 4 classes.
- Optional skull-stripping heuristic for 2D images during preprocessing.
- Albumentations-based data augmentation (train) and normalization.
- 5-fold stratified cross-validation, early stopping, cosine LR, AMP (CUDA).
- Metrics: Accuracy, macro/weighted F1, per-class report, macro AUC, confusion matrix.
- Explainability: Grad-CAM using Captum (no extra dependencies).
- Clean structure, configurable via `configs/config.yaml`.

## Repository Structure
```
NeuroClassify/
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
- No images found during training:
  - Ensure `data/processed/` has PNG/JPG images under each class. Run preprocessing first.
- CUDA errors / slow training:
  - If you lack CUDA, install CPU-only PyTorch (see Setup) or switch `device: cpu`.
- Large files in Git push:
  - Checkpoints are big. Either stop tracking `results/` (default in `.gitignore`) or use Git LFS to version large files.
- Grad-CAM errors:
  - Ensure the checkpoint file path exists and the image path points to a processed image.

## Git & Large Files
`.gitignore` excludes `data/` and `results/` by default. If you need to version model files, use Git LFS:
```bash
git lfs install
git lfs track "*.pt" "*.pth" "*.onnx" "*.ckpt" "*.bin"
# commit .gitattributes added by LFS
```

## License
This project is provided under the LICENSE included in this repository.
