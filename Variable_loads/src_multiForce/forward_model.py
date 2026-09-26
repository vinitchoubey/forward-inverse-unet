import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from . import config as cfg
from .unet_architecture import build_unet, combined_loss, get_callbacks
from .utils import denorm, add_img


def train_forward_model(data):
    np.random.seed(cfg.SEED_FORWARD)
    tf.random.set_seed(cfg.SEED_FORWARD)

    fwd_model = build_unet(name='UNet_Forward', in_channels=2)
    fwd_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=['mae']
    )

    fwd_hist = fwd_model.fit(
        data.fwd_X_tr, data.S_tr,
        validation_data=(data.fwd_X_val, data.S_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=get_callbacks(os.path.join(cfg.MODEL_DIR, 'best_forward.h5')),
        verbose=1
    )

    fwd_pred_norm = fwd_model.predict(data.fwd_X_te, batch_size=cfg.BATCH_SIZE, verbose=0)
    fwd_true = denorm(data.S_te, data.s_lo, data.s_hi, log=True)
    fwd_pred = denorm(fwd_pred_norm, data.s_lo, data.s_hi, log=True)

    fwd_r2 = r2_score(fwd_true.ravel(), fwd_pred.ravel())
    fwd_r2_norm = r2_score(data.S_te.ravel(), fwd_pred_norm.ravel())
    fwd_mse = mean_squared_error(fwd_true.ravel(), fwd_pred.ravel())
    fwd_mae = mean_absolute_error(fwd_true.ravel(), fwd_pred.ravel())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fwd_hist.history['loss'], color='royalblue', lw=2, label='Train Loss')
    ax.plot(fwd_hist.history['val_loss'], color='royalblue', lw=2, ls='--', alpha=0.8, label='Val Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Combined Loss')
    ax.set_title('Forward UNet - Training Curve', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.FWD_DIR, 'B1_forward_training_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(cfg.N_PLOT, 4, figsize=(22, 6 * cfg.N_PLOT))
    for col_i, title in enumerate(['Input: Thickness (norm.)', 'Input: Load-map (norm.)',
                                    'True Stress (Physical)', 'Predicted Stress (Physical)']):
        axes[0, col_i].set_title(title, fontsize=11, fontweight='bold')

    for row, sid in enumerate(data.plot_ids):
        add_img(axes[row, 0], data.T_te[sid, :, :, 0], 'viridis')
        add_img(axes[row, 1], data.L_te[sid, :, :, 0], 'inferno')
        add_img(axes[row, 2], fwd_true[sid, :, :, 0], 'jet')
        add_img(axes[row, 3], fwd_pred[sid, :, :, 0], 'jet')

    plt.suptitle(f'Forward UNet | R2 = {fwd_r2:.4f} | MAE = {fwd_mae:.4f}', fontsize=12, y=1.002)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.FWD_DIR, 'B2_forward_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    rng_sc = np.random.default_rng(99)
    flat_true = fwd_true.ravel()
    flat_pred = fwd_pred.ravel()
    idx_sc = rng_sc.choice(len(flat_true), min(5000, len(flat_true)), replace=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(flat_true[idx_sc], flat_pred[idx_sc], s=6, alpha=0.4, color='steelblue', label='Pixel predictions')
    lims = [min(flat_true.min(), flat_pred.min()), max(flat_true.max(), flat_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('True Von Mises Stress'); ax.set_ylabel('Predicted Von Mises Stress')
    ax.set_title(f'Forward UNet - Pixel Scatter | R2 = {fwd_r2:.4f}')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.FWD_DIR, 'B3_forward_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'model': fwd_model, 'history': fwd_hist,
        'r2': fwd_r2, 'r2_norm': fwd_r2_norm, 'mse': fwd_mse, 'mae': fwd_mae,
        'pred_norm': fwd_pred_norm, 'pred_phys': fwd_pred,
    }
