import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def scatter_to_grid(values, xs, ys, XI, YI):
    pts = np.column_stack([xs, ys])
    grid = griddata(pts, values, (XI, YI), method='linear', fill_value=np.nan)
    if np.any(np.isnan(grid)):
        fill = griddata(pts, values, (XI, YI), method='nearest')
        grid = np.where(np.isnan(grid), fill, grid)
    return grid.astype(np.float32)


def load_to_grid(load_node_ids, load_mags, node_xy_ref, xi, yi, grid_h, grid_w, sigma):
    grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    rr, cc = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
    n_skipped = 0
    for nid, mag in zip(load_node_ids, load_mags):
        nid = int(nid)
        if nid < 1 or nid > len(node_xy_ref):
            n_skipped += 1
            continue
        x, y = node_xy_ref[nid - 1]
        col = int(np.argmin(np.abs(xi - x)))
        row = int(np.argmin(np.abs(yi - y)))
        splat = abs(float(mag)) * np.exp(-(((rr - row) ** 2 + (cc - col) ** 2) / (2 * sigma ** 2)))
        grid += splat
    return grid.astype(np.float32), n_skipped


def denorm(arr, lo, hi, log=False):
    out = arr * (hi - lo) + lo
    return np.expm1(out) if log else out


def add_img(ax, img, cmap):
    lo = np.percentile(img, 1)
    hi = np.percentile(img, 99)
    if hi - lo < 1e-10:
        lo, hi = img.min(), img.max() + 1e-8
    im = ax.imshow(img, cmap=cmap, origin='lower', aspect='auto', vmin=lo, vmax=hi)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.axis('off')


def predict_stress_for_custom_load(forward_model, thickness_field_norm, node_ids, magnitudes,
                                    node_xy_ref, xi, yi, grid_h, grid_w, sigma,
                                    l_lo, l_hi, s_lo, s_hi):
    load_map, _ = load_to_grid(node_ids, magnitudes, node_xy_ref, xi, yi, grid_h, grid_w, sigma)
    load_map_log = np.log1p(load_map)
    load_map_norm = (load_map_log - l_lo) / (l_hi - l_lo + 1e-8)
    x_in = np.stack([thickness_field_norm, load_map_norm], axis=-1)[None, ...]
    pred_norm = forward_model.predict(x_in, verbose=0)[0, :, :, 0]
    return denorm(pred_norm, s_lo, s_hi, log=True)
