#  SECTION C  –  INVERSE UNET   (Von Mises Stress  →  Thickness)


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

from . import config as cfg
from .unet_architecture import build_unet, combined_loss, get_callbacks
from .utils import denorm


def train_inverse_model(data):
    

    np.random.seed(cfg.SEED_INVERSE)
    tf.random.set_seed(cfg.SEED_INVERSE)

    inv_model = build_unet(name='UNet_Inverse')
    inv_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=['mae']
    )

    inv_hist = inv_model.fit(
        data.Y_tr, data.X_tr,
        validation_data=(data.Y_val, data.X_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=get_callbacks(os.path.join(cfg.MODEL_DIR, 'best_inverse.h5')),
        verbose=1
    )

    inv_pred_norm = inv_model.predict(data.Y_te, batch_size=cfg.BATCH_SIZE, verbose=0)
    inv_true = denorm(data.X_te, data.t_lo, data.t_hi, log=False)
    inv_pred = denorm(inv_pred_norm, data.t_lo, data.t_hi, log=False)

    inv_r2  = r2_score(inv_true.ravel(), inv_pred.ravel())
    inv_mae = mean_absolute_error(inv_true.ravel(), inv_pred.ravel())

    # Training curve 
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(inv_hist.history['loss'],     color='tomato', lw=2, label='Train Loss')
    ax.plot(inv_hist.history['val_loss'], color='tomato', lw=2, ls='--', alpha=0.8, label='Val Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Combined Loss')
    ax.set_title('Inverse UNet – Training Curve', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.INV_DIR, 'C1_inverse_training_curve.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    #  Prediction grid  
    fig, axes = plt.subplots(cfg.N_PLOT, 3, figsize=(17, 6 * cfg.N_PLOT))
    for col_i, title in enumerate(['Input: Stress (normalised)',
                                    'True Thickness (Physical)',
                                    'Predicted Thickness (Physical)']):
        axes[0, col_i].set_title(title, fontsize=11, fontweight='bold')

    for row, sid in enumerate(data.plot_ids):
        inp  = data.Y_te[sid, :, :, 0]
        true = inv_true[sid, :, :, 0]
        pred = inv_pred[sid, :, :, 0]

        im0 = axes[row, 0].imshow(inp, cmap='jet', origin='lower',
                                   aspect='auto', vmin=inp.min(), vmax=inp.max())
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.02)
        axes[row, 0].axis('off')

        for col_i, img in ((1, true), (2, pred)):
            lo_p, hi_p = np.percentile(img, 1), np.percentile(img, 99)
            if hi_p - lo_p < 1e-10:
                lo_p, hi_p = img.min(), img.max() + 1e-8
            im = axes[row, col_i].imshow(img, cmap='jet', origin='lower',
                                          aspect='auto', vmin=lo_p, vmax=hi_p)
            plt.colorbar(im, ax=axes[row, col_i], fraction=0.046, pad=0.02)
            axes[row, col_i].axis('off')

    plt.suptitle(
        f'Inverse UNet (Stress → Thickness)  |  R² = {inv_r2:.4f}  |  MAE = {inv_mae:.6f}\n'
        f'(Low R² reflects ill-posedness; Cyclic Validation is the true metric)',
        fontsize=11, y=1.002
    )
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.INV_DIR, 'C2_inverse_predictions.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'model': inv_model, 'history': inv_hist,
        'r2': inv_r2, 'mae': inv_mae,
        'pred_norm': inv_pred_norm, 'pred_phys': inv_pred,
    }
