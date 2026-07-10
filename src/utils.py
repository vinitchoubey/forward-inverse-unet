
#  UTILS  –  Shared helper functions


import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def scatter_to_grid(values, xs, ys, XI, YI):
    pts  = np.column_stack([xs, ys])
    grid = griddata(pts, values, (XI, YI), method='linear', fill_value=np.nan)
    if np.any(np.isnan(grid)):
        fill = griddata(pts, values, (XI, YI), method='nearest')
        grid = np.where(np.isnan(grid), fill, grid)
    return grid.astype(np.float32)


def denorm(arr, lo, hi, log=False):
    out = arr * (hi - lo) + lo
    return np.expm1(out) if log else out


def add_img(ax, img, cmap):
    lo = np.percentile(img, 1)
    hi = np.percentile(img, 99)
    if hi - lo < 1e-10:
        lo, hi = img.min(), img.max() + 1e-8
    im = ax.imshow(img, cmap=cmap, origin='lower', aspect='auto',
                    vmin=lo, vmax=hi)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.axis('off')
