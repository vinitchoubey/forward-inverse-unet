
#  SECTION D  –  OPTIMISED CYCLIC VALIDATION  (THE PHYSICS PROOF)


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from . import config as cfg
from .utils import denorm, add_img


def run_cyclic_validation(data, fwd_result, inv_result):
   
    fwd_model = fwd_result['model']
    inv_pred_norm = inv_result['pred_norm']
    inv_pred_phys = inv_result['pred_phys']

    inv_pred_clip = np.clip(inv_pred_norm, 0.0, 1.0)
    recon_norm = fwd_model.predict(inv_pred_clip, batch_size=cfg.BATCH_SIZE, verbose=0)

    orig_phys  = denorm(data.Y_te, data.s_lo, data.s_hi, log=True)
    recon_phys = denorm(recon_norm, data.s_lo, data.s_hi, log=True)

    #   Dual-space metrics 
    cyc_r2_norm  = r2_score(data.Y_te.ravel(), recon_norm.ravel())
    cyc_mae_norm = mean_absolute_error(data.Y_te.ravel(), recon_norm.ravel())

    cyc_r2_phys  = r2_score(orig_phys.ravel(), recon_phys.ravel())
    cyc_mae_phys = mean_absolute_error(orig_phys.ravel(), recon_phys.ravel())
    cyc_mse_phys = mean_squared_error(orig_phys.ravel(), recon_phys.ravel())

    #  Per-sample cyclic r2
    per_sample_r2 = np.array([
        r2_score(data.Y_te[i].ravel(), recon_norm[i].ravel())
        for i in range(len(data.Y_te))
    ])

    #  Cyclic prediction grid  
    D_COLS  = ['Original Stress  (Input to Inverse)',
               'Inverse → Predicted Thickness',
               'Forward(Pred. Thickness) → Reconstructed Stress']
    D_CMAPS = ['jet', 'viridis', 'jet']

    fig, axes = plt.subplots(cfg.N_PLOT, 3, figsize=(17, 6 * cfg.N_PLOT))
    for col_i, title in enumerate(D_COLS):
        axes[0, col_i].set_title(title, fontsize=10, fontweight='bold')

    for row, sid in enumerate(data.plot_ids):
        orig_s  = orig_phys[sid, :, :, 0]
        pred_t  = inv_pred_phys[sid, :, :, 0]
        recon_s = recon_phys[sid, :, :, 0]
        s_r2    = per_sample_r2[sid]

        for col_i, (img, cmap) in enumerate(zip([orig_s, pred_t, recon_s], D_CMAPS)):
            add_img(axes[row, col_i], img, cmap)

        axes[row, 2].text(
            0.03, 0.04, f'sample R² = {s_r2:.3f}',
            transform=axes[row, 2].transAxes,
            fontsize=9, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.55)
        )

    plt.suptitle(
        'Cyclic Validation:  Stress → Inverse → Thickness → Forward → Stress\n'
        f'Cyclic R² (log-norm, PRIMARY) = {cyc_r2_norm:.4f}  |  '
        f'Cyclic R² (physical) = {cyc_r2_phys:.4f}  |  MAE = {cyc_mae_phys:.4f}',
        fontsize=11, y=1.002
    )
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.CYC_DIR, 'D1_cyclic_validation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Per-sample cyclic R² distribution 
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(per_sample_r2, bins=30, color='seagreen', edgecolor='black', alpha=0.82)
    ax.axvline(np.median(per_sample_r2), color='crimson', ls='--', lw=2,
               label=f'Median = {np.median(per_sample_r2):.4f}')
    ax.axvline(cyc_r2_norm, color='navy', ls='-.', lw=2,
               label=f'Global R² = {cyc_r2_norm:.4f}')
    ax.set_xlabel('Per-sample Cyclic R²  (log-norm space)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Per-sample Cyclic R²  (log-norm space)', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.CYC_DIR, 'D2_cyclic_r2_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

   
    rng_sc2    = np.random.default_rng(88)
    flat_orig  = orig_phys.ravel()
    flat_recon = recon_phys.ravel()
    idx_sc2    = rng_sc2.choice(len(flat_orig), min(5_000, len(flat_orig)), replace=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(flat_orig[idx_sc2], flat_recon[idx_sc2],
               s=6, alpha=0.4, color='seagreen', label='Pixel reconstructions')
    lims = [min(flat_orig.min(), flat_recon.min()), max(flat_orig.max(), flat_recon.max())]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect reconstruction')
    ax.set_xlabel('Original Von Mises Stress')
    ax.set_ylabel('Reconstructed Stress (Cyclic)')
    ax.set_title(f'Cyclic Scatter  |  R² (physical) = {cyc_r2_phys:.4f}')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.3e7); ax.set_ylim(0, 0.3e7)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.CYC_DIR, 'D3_cyclic_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    bar_labels = ['Forward\n(direct)', 'Inverse\n(direct)',
                  'Cyclic\n(physical)', 'Cyclic\n(log-norm\nPRIMARY)']
    bar_vals   = [fwd_result['r2'], inv_result['r2'], cyc_r2_phys, cyc_r2_norm]
    bar_colors = ['royalblue', 'tomato', 'goldenrod', 'seagreen']

    bars = ax.bar(bar_labels, bar_vals, color=bar_colors, edgecolor='black', width=0.45, zorder=3)
    ax.axhline(0, color='black', lw=1.0, ls='--')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(
        'Model Performance Summary\n'
        '(Cyclic log-norm R² is the primary physics-validity metric;\n'
        ' inverse direct R² is expected to be low – ill-posed by design)',
        fontsize=10, fontweight='bold'
    )
    ax.set_ylim(min(bar_vals) - 0.15, 1.05)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    for bar, val in zip(bars, bar_vals):
        offset = 0.03 if val >= 0 else -0.07
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.annotate('ill-posed\none-to-many',
                xy=(1, inv_result['r2']), xytext=(1.65, inv_result['r2'] - 0.15),
                fontsize=9, color='tomato',
                arrowprops=dict(arrowstyle='->', color='tomato'))

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.CYC_DIR, 'D4_summary_r2_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes_tc = plt.subplots(1, 2, figsize=(16, 5))
    for ax, mode, color, hist in zip(
            axes_tc,
            ['Forward (combined_loss)', 'Inverse (combined_loss)'],
            ['royalblue', 'tomato'],
            [fwd_result['history'], inv_result['history']]):
        ax.plot(hist.history['loss'],     color=color, lw=2, label='Train Loss')
        ax.plot(hist.history['val_loss'], color=color, lw=2, ls='--', alpha=0.8, label='Val Loss')
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(f'{mode} – Training Curve', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.suptitle('Model Convergence Comparison', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.CYC_DIR, 'D5_combined_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'cyc_r2_norm': cyc_r2_norm, 'cyc_mae_norm': cyc_mae_norm,
        'cyc_r2_phys': cyc_r2_phys, 'cyc_mae_phys': cyc_mae_phys,
        'cyc_mse_phys': cyc_mse_phys, 'per_sample_r2': per_sample_r2,
    }
