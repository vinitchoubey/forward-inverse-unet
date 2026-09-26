import os
import glob
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg
from .utils import scatter_to_grid, load_to_grid


def _numeric_key(p):
    return int(os.path.splitext(os.path.basename(p))[0].split('_')[-1].split()[0])


def _read_load_file(path):
    try:
        ld = np.loadtxt(path)
    except ValueError:
        ld = np.loadtxt(path, skiprows=1)
    if ld.ndim == 1:
        ld = ld[None, :]
    return ld


def load_and_prepare_data():
    if not os.path.isdir(cfg.DATA_DIR):
        raise FileNotFoundError(
            f"Dataset folder not found at '{cfg.DATA_DIR}'.\n"
            f"Place output.xlsx, cord.txt, stress/ and loaddata/ there."
        )

    cord_matches = glob.glob(os.path.join(cfg.DATA_DIR, '*cord*'))
    if not cord_matches:
        raise FileNotFoundError(f"No cord*.txt file found in {cfg.DATA_DIR}")
    cord_raw = np.loadtxt(cord_matches[0])
    XY_fem = cord_raw[:, 1:] if cord_raw.shape[1] == 3 else cord_raw

    x_min, x_max = XY_fem[:, 0].min(), XY_fem[:, 0].max()
    y_min, y_max = XY_fem[:, 1].min(), XY_fem[:, 1].max()
    beam_aspect = (x_max - x_min) / (y_max - y_min)

    best_nx, best_ny, best_score = 1, cfg.N_DESIGN, float('inf')
    for nx in range(2, cfg.N_DESIGN):
        ny = max(1, round(cfg.N_DESIGN / nx))
        score = abs(nx * ny - cfg.N_DESIGN) + 3.0 * abs(nx / ny - beam_aspect)
        if score < best_score:
            best_score, best_nx, best_ny = score, nx, ny

    dn_x, dn_y = np.linspace(x_min, x_max, best_nx), np.linspace(y_min, y_max, best_ny)
    DN_X, DN_Y = np.meshgrid(dn_x, dn_y)
    design_xy = np.column_stack([DN_X.ravel(), DN_Y.ravel()])
    n_design = len(design_xy)

    xi = np.linspace(x_min, x_max, cfg.GRID_W)
    yi = np.linspace(y_min, y_max, cfg.GRID_H)
    XI, YI = np.meshgrid(xi, yi)

    if not os.path.isfile(cfg.XLSX_PATH):
        raise FileNotFoundError(f"Thickness file not found: {cfg.XLSX_PATH}")
    thickness_raw = pd.read_excel(cfg.XLSX_PATH, header=None).values.astype(np.float32)
    n_use = min(n_design, thickness_raw.shape[1])

    stress_files_all = sorted(glob.glob(os.path.join(cfg.STRESS_DIR, '*.txt')))
    load_files_all = sorted(glob.glob(os.path.join(cfg.LOAD_DIR, '*.txt')))
    if not stress_files_all:
        raise FileNotFoundError(f"No stress *.txt files found in {cfg.STRESS_DIR}")
    if not load_files_all:
        raise FileNotFoundError(f"No loaddata *.txt files found in {cfg.LOAD_DIR}")

    common_keys = sorted(set(_numeric_key(p) for p in stress_files_all) &
                          set(_numeric_key(p) for p in load_files_all))
    stress_by_key = {_numeric_key(p): p for p in stress_files_all}
    load_by_key = {_numeric_key(p): p for p in load_files_all}
    stress_files = [stress_by_key[k] for k in common_keys]
    load_files = [load_by_key[k] for k in common_keys]

    n_samples = min(cfg.N_SAMPLES, len(stress_files), len(load_files), thickness_raw.shape[0])
    stress_files = stress_files[:n_samples]
    load_files = load_files[:n_samples]

    scan_n = min(50, len(load_files))
    all_ids = []
    for p in load_files[:scan_n]:
        ld = _read_load_file(p)
        all_ids.extend(ld[:, 0].tolist())
    all_ids = np.array(all_ids)
    id_max = all_ids.max()
    id_unique = len(np.unique(all_ids))

    if id_max > len(design_xy):
        node_xy_ref = XY_fem                     # ids clearly index the full FEM mesh
    elif id_max <= len(design_xy) and id_unique <= len(design_xy) * 0.5:
        node_xy_ref = design_xy                  # sparse subset of the coarse design grid
    else:
        node_xy_ref = XY_fem                     # fallback heuristic

    t_imgs = np.zeros((n_samples, cfg.GRID_H, cfg.GRID_W, 1), dtype=np.float32)
    s_imgs = np.zeros((n_samples, cfg.GRID_H, cfg.GRID_W, 1), dtype=np.float32)
    l_imgs = np.zeros((n_samples, cfg.GRID_H, cfg.GRID_W, 1), dtype=np.float32)

    for i in range(n_samples):
        t_imgs[i, :, :, 0] = scatter_to_grid(
            thickness_raw[i, :n_use], design_xy[:n_use, 0], design_xy[:n_use, 1], XI, YI
        )
        try:
            sd = np.loadtxt(stress_files[i])
        except ValueError:
            sd = np.loadtxt(stress_files[i], skiprows=1)
        s_imgs[i, :, :, 0] = scatter_to_grid(
            sd[:, 0].astype(np.float32), sd[:, 1].astype(np.float32), sd[:, 2].astype(np.float32), XI, YI
        )

        ld = _read_load_file(load_files[i])
        l_grid, _ = load_to_grid(
            ld[:, 0], ld[:, 2], node_xy_ref, xi, yi, cfg.GRID_H, cfg.GRID_W, cfg.LOAD_SPLAT_SIGMA
        )
        l_imgs[i, :, :, 0] = l_grid

    if l_imgs.max() <= 0:
        raise ValueError("Load maps are entirely zero -- check loaddata files and the node-id reference.")

    s_imgs_log = np.log1p(s_imgs)
    l_imgs_log = np.log1p(l_imgs)

    s_clip_hi = np.percentile(s_imgs_log, cfg.STRESS_CLIP_PCTL)
    s_imgs_log_clipped = np.clip(s_imgs_log, s_imgs_log.min(), s_clip_hi)

    t_lo, t_hi = t_imgs.min(), t_imgs.max()
    s_lo, s_hi = s_imgs_log_clipped.min(), s_imgs_log_clipped.max()
    l_lo, l_hi = l_imgs_log.min(), l_imgs_log.max()

    t_norm = (t_imgs - t_lo) / (t_hi - t_lo + 1e-8)
    s_norm = (s_imgs_log_clipped - s_lo) / (s_hi - s_lo + 1e-8)
    l_norm = (l_imgs_log - l_lo) / (l_hi - l_lo + 1e-8)

    T_tr, T_te, S_tr, S_te, L_tr, L_te = train_test_split(
        t_norm, s_norm, l_norm, test_size=0.15, random_state=42
    )
    T_tr, T_val, S_tr, S_val, L_tr, L_val = train_test_split(
        T_tr, S_tr, L_tr, test_size=0.12, random_state=42
    )

    fwd_X_tr = np.concatenate([T_tr, L_tr], -1)
    fwd_X_val = np.concatenate([T_val, L_val], -1)
    fwd_X_te = np.concatenate([T_te, L_te], -1)
    inv_X_tr = np.concatenate([S_tr, L_tr], -1)
    inv_X_val = np.concatenate([S_val, L_val], -1)
    inv_X_te = np.concatenate([S_te, L_te], -1)

    plot_ids = np.random.default_rng(77).choice(len(T_te), cfg.N_PLOT, replace=False)

    return SimpleNamespace(
        T_tr=T_tr, T_val=T_val, T_te=T_te,
        S_tr=S_tr, S_val=S_val, S_te=S_te,
        L_tr=L_tr, L_val=L_val, L_te=L_te,
        fwd_X_tr=fwd_X_tr, fwd_X_val=fwd_X_val, fwd_X_te=fwd_X_te,
        inv_X_tr=inv_X_tr, inv_X_val=inv_X_val, inv_X_te=inv_X_te,
        t_lo=t_lo, t_hi=t_hi, s_lo=s_lo, s_hi=s_hi, l_lo=l_lo, l_hi=l_hi,
        XI=XI, YI=YI, xi=xi, yi=yi, node_xy_ref=node_xy_ref,
        plot_ids=plot_ids, n_samples=n_samples,
    )