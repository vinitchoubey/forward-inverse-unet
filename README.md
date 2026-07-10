# ME504 – Forward + Inverse UNet Pipeline with Cyclic Physics Validation

A deep-learning pipeline that learns the two-way mapping between beam
**thickness distribution** and **Von Mises stress** using U-Net architectures,
and validates physical consistency via a closed-loop cyclic test:

```
Stress → Inverse UNet → Predicted Thickness → Forward UNet → Reconstructed Stress
```

## Repository Structure

```
ME504_UNet_Pipeline/
├── dataset/                  # <- put your data here (NOT tracked by git, see below)
│   ├── output.xlsx           # thickness values per sample
│   ├── cord.txt               # FEM node coordinates
│   └── stress/                # per-sample stress .txt files
├── src/
│   ├── config.py              # paths & hyper-parameters
│   ├── utils.py                # scatter_to_grid, denorm, add_img helpers
│   ├── data_processing.py     # Section A: load, grid, normalise, split
│   ├── unet_architecture.py   # U-Net model, combined_loss, callbacks
│   ├── forward_model.py       # Section B: Thickness -> Stress
│   ├── inverse_model.py       # Section C: Stress -> Thickness
│   └── cyclic_validation.py   # Section D: cyclic physics validation
├── outputs/                   # generated plots & saved models (git-ignored)
│   ├── models/                # best_forward.h5, best_inverse.h5
│   ├── forward/                # training curve, prediction grid, scatter
│   ├── inverse/                 # training curve, prediction grid
│   └── cyclic/                  # cyclic grid, R² distribution, scatter, summary
├── main.py                     # entry point — runs the full pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset Setup (Local Disk — No Google Drive Needed)

This pipeline was refactored to run entirely from local disk instead of a
Google Drive mount. Clone/place this repo anywhere, e.g. `D:\ME504_UNet_Pipeline\`,
then drop your data into the `dataset/` folder so it looks like this:

```
D:\ME504_UNet_Pipeline\dataset\
├── output.xlsx
├── cord.txt
└── stress\
    ├── stress_1.txt
    ├── stress_2.txt
    └── ... (one file per sample)
```

All paths are resolved automatically relative to the repo root (see
`src/config.py`) — you do **not** need to edit any paths by hand, regardless
of which drive/folder you clone the repo into.

> **Note:** the dataset (>1 GB) is intentionally excluded from git via
> `.gitignore`. Only the folder structure (`.gitkeep` placeholders) is
> committed. You must copy your data in manually after cloning.

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Running the Pipeline

```bash
python main.py
```

This runs, in order:

1. **Section A – Data Processing**: loads thickness (`.xlsx`) and stress
   (`.txt`) data, interpolates both onto a shared `64×128` pixel grid,
   applies log1p + global min-max normalisation, and splits into
   train/val/test (~73/12/15%).
2. **Section B – Forward UNet** (Thickness → Stress): trained with a
   combined loss (MSE + MAE + gradient penalty). Saves training curve,
   a 3-sample prediction grid, and a pixel-level scatter plot to
   `outputs/forward/`.
3. **Section C – Inverse UNet** (Stress → Thickness): an intentionally
   ill-posed one-to-many problem — low direct R² is expected. RNG is
   reseeded before this stage so its Dropout layers start from a fresh,
   reproducible state. Saves outputs to `outputs/inverse/`.
4. **Section D – Cyclic Validation**: runs the full
   Stress → Inverse → Thickness → Forward → Stress loop, clips the
   inverse output to `[0, 1]` before feeding it back into the forward
   model, and reports R² in both log-norm (primary) and physical
   (audit) space, plus a per-sample R² distribution. Saves outputs to
   `outputs/cyclic/`.

All figures are saved as `.png` at 150 dpi; trained model weights are
saved as `.h5` under `outputs/models/`.

## Key Design Notes

- **`combined_loss`** (`0.4·MSE + 0.4·MAE + 0.2·gradient_penalty`) is used
  for both models — the gradient-penalty term is the single biggest driver
  of higher R² on the forward test set and, downstream, on the cyclic
  reconstruction.
- **Seed reset** between Forward and Inverse training (`SEED_FORWARD=42`,
  `SEED_INVERSE=123`) ensures the Inverse model's Dropout layers aren't
  biased by RNG state consumed during the Forward model's training.
- **Clipping** of `inv_pred_norm` to `[0, 1]` before the cyclic forward
  pass keeps inputs in-distribution for the sigmoid-activated forward model.
- **Dual-space cyclic R²**: log-norm space (primary, undistorted by the
  heavy tail of physical stress) and physical space (for audit/comparison).

## Requirements

See `requirements.txt`. Core dependencies: TensorFlow/Keras, scikit-learn,
pandas, scipy, matplotlib, openpyxl.
