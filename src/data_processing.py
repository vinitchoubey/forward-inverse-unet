
#  SECTION A  –  DATA LOADING, GRID SETUP & NORMALISATION


import os
import glob
import numpy as np
import pandas as pd
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

from . import config as cfg
from .utils import scatter_to_grid


def load_and_prepare_data():


    if not os.path.isdir(cfg.DATA_DIR):
        raise FileNotFoundError(
            f"Dataset folder not found at '{cfg.DATA_DIR}'.\n"
            f"Place output.xlsx, cord.txt and the stress/ folder there."
        )

    #Beam Geometry & Design-Variable Grid 
    cord_matches = glob.glob(os.path.join(cfg.DATA_DIR, '*cord*'))
    if not cord_matches:
        raise FileNotFoundError(f"No cord*.txt file found in {cfg.DATA_DIR}")
    cord_file = cord_matches[0]
    cord_raw  = np.loadtxt(cord_file)
    XY_fem    = cord_raw[:, 1:] if cord_raw.shape[1] == 3 else cord_raw

    x_min, x_max = XY_fem[:, 0].min(), XY_fem[:, 0].max()
    y_min, y_max = XY_fem[:, 1].min(), XY_fem[:, 1].max()
    beam_aspect  = (x_max - x_min) / (y_max - y_min)

    best_nx, best_ny, best_score = 1, cfg.N_DESIGN, float('inf')
    for nx in range(2, cfg.N_DESIGN):
        ny    = max(1, round(cfg.N_DESIGN / nx))
        score = abs(nx * ny - cfg.N_DESIGN) + 3.0 * abs(nx / ny - beam_aspect)
        if score < best_score:
            best_score, best_nx, best_ny = score, nx, ny

    dn_x, dn_y = np.linspace(x_min, x_max, best_nx), np.linspace(y_min, y_max, best_ny)
    DN_X, DN_Y = np.meshgrid(dn_x, dn_y)
    design_xy  = np.column_stack([DN_X.ravel(), DN_Y.ravel()])
    n_design   = len(design_xy)

    #Pixel Grid 
    xi = np.linspace(x_min, x_max, cfg.GRID_W)
    yi = np.linspace(y_min, y_max, cfg.GRID_H)
    XI, YI = np.meshgrid(xi, yi)

    #Load Thickness (xlsx) & Stress (.txt) 
    if not os.path.isfile(cfg.XLSX_PATH):
        raise FileNotFoundError(f"Thickness file not found: {cfg.XLSX_PATH}")
    thickness_raw = pd.read_excel(cfg.XLSX_PATH, header=None).values.astype(np.float32)
    n_use = min(n_design, thickness_raw.shape[1])

    stress_files = sorted(
        glob.glob(os.path.join(cfg.STRESS_DIR, '*.txt')),
        key=lambda p: int(
            os.path.splitext(os.path.basename(p))[0]
            .replace('stress_', '').split()[0]
        )
    )
    if not stress_files:
        raise FileNotFoundError(f"No stress *.txt files found in {cfg.STRESS_DIR}")

    n_samples = min(cfg.N_SAMPLES, thickness_raw.shape[0], len(stress_files))

    t_imgs = np.zeros((n_samples, cfg.GRID_H, cfg.GRID_W, 1), dtype=np.float32)
    s_imgs = np.zeros((n_samples, cfg.GRID_H, cfg.GRID_W, 1), dtype=np.float32)

    for i in range(n_samples):
        t_imgs[i, :, :, 0] = scatter_to_grid(
            thickness_raw[i, :n_use],
            design_xy[:n_use, 0],
            design_xy[:n_use, 1],
            XI, YI
        )
        try:
            sd = np.loadtxt(stress_files[i])
        except ValueError:
            sd = np.loadtxt(stress_files[i], skiprows=1)
        s_imgs[i, :, :, 0] = scatter_to_grid(
            sd[:, 0].astype(np.float32),
            sd[:, 1].astype(np.float32),
            sd[:, 2].astype(np.float32),
            XI, YI
        )

    # Log Transform & Global Min-Max Normalisation 
    s_imgs_log = np.log1p(s_imgs)

    t_lo, t_hi = t_imgs.min(),     t_imgs.max()
    s_lo, s_hi = s_imgs_log.min(), s_imgs_log.max()

    t_norm = (t_imgs     - t_lo) / (t_hi - t_lo + 1e-8)
    s_norm = (s_imgs_log - s_lo) / (s_hi - s_lo + 1e-8)

    # ── A-6  Train / Val / Test Split  (≈73% / 12% / 15%) 
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        t_norm, s_norm, test_size=0.15, random_state=42
    )
    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X_tr, Y_tr, test_size=0.12, random_state=42
    )


    plot_ids = np.random.default_rng(77).choice(len(X_te), cfg.N_PLOT, replace=False)

    return SimpleNamespace(
        X_tr=X_tr, X_val=X_val, X_te=X_te,
        Y_tr=Y_tr, Y_val=Y_val, Y_te=Y_te,
        t_lo=t_lo, t_hi=t_hi, s_lo=s_lo, s_hi=s_hi,
        XI=XI, YI=YI, plot_ids=plot_ids,
        n_samples=n_samples,
    )
