import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

from . import config as cfg
from .unet_architecture import build_unet, combined_loss, get_callbacks
from .utils import denorm, add_img


def train_inverse_model(data):
    np.random.seed(cfg.SEED_INVERSE)
    tf.random.set_seed(cfg.SEED_INVERSE)

    inv_model = build_unet(name='UNet_Inverse', in_channels=2)
    inv_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=['mae']
    )

    inv_hist = inv_model.fit(
        data.inv_X_tr, data.T_tr,
        validation_data=(data.inv_X_val, data.T_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=get_callbacks(os.path.join(cfg.MODEL_DIR, 'best_inverse.h5')),
        verbose=1
    )

    inv_pred_norm = inv_model.predict(data.inv_X_te, batch_size=cfg.BATCH_SIZE, verbose=0)
    inv_true = denorm(data.T_te, data.t_lo, data.t_hi, log=False)
    inv_pred = denorm(inv_pred_norm, data.t_lo, data.t_hi, log=False)

    inv_r2 = r2_score(inv_true.ravel(), inv_pred.ravel())
    inv_mae = mean_absolute_error(inv_true.ravel(), inv_pred.ravel())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(inv_hist.history['loss'], color='tomato', lw=2, label='Train Loss')
    ax.plot(inv_hist.history['val_loss'], color='tomato', lw=2, ls='--', alpha=0.8, label='Val Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Combined Loss')
    ax.set_title('Inverse UNet - Training Curve', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.INV_DIR, 'C1_inverse_training_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(cfg.N_PLOT, 4, figsize=(22, 6 * cfg.N_PLOT))
    for col_i, title in enumerate(['Input: Stress (norm.)', 'Input: Load-map (norm.)',
                                    'True Thickness (Physical)', 'Predicted Thickness (Physical)']):
        axes[0, col_i].set_title(title, fontsize=11, fontweight='bold')

    for row, sid in enumerate(data.plot_ids):
        add_img(axes[row, 0], data.S_te[sid, :, :, 0], 'jet')
        add_img(axes[row, 1], data.L_te[sid, :, :, 0], 'inferno')
        add_img(axes[row, 2], inv_true[sid, :, :, 0], 'jet')
        add_img(axes[row, 3], inv_pred[sid, :, :, 0], 'jet')

    plt.suptitle(f'Inverse UNet | R2 = {inv_r2:.4f} | MAE = {inv_mae:.6f}', fontsize=12, y=1.002)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.INV_DIR, 'C2_inverse_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'model': inv_model, 'history': inv_hist,
        'r2': inv_r2, 'mae': inv_mae,
        'pred_norm': inv_pred_norm, 'pred_phys': inv_pred,
    }
