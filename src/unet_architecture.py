#  SECTION  –  U-NET ARCHITECTURE, LOSSES & CALLBACKS
#  Shared between the Forward and Inverse models.


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from . import config as cfg


# ── U-Net building blocks 
def conv_block(x, filters, drop=0.10):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    return layers.Dropout(drop)(x)


def build_unet(name='UNet', base=32):
    inp = layers.Input((cfg.GRID_H, cfg.GRID_W, 1))

    # Encoder
    c1 = conv_block(inp, base);      p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1,  base * 2);  p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2,  base * 4);  p3 = layers.MaxPooling2D(2)(c3)
    c4 = conv_block(p3,  base * 8);  p4 = layers.MaxPooling2D(2)(c4)

    # Bottleneck
    b = conv_block(p4, base * 16)

    # Decoder
    def up_block(x, skip, f):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip])
        return conv_block(x, f)

    x = up_block(b, c4, base * 8)
    x = up_block(x, c3, base * 4)
    x = up_block(x, c2, base * 2)
    x = up_block(x, c1, base)

    out = layers.Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inp, out, name=name)


# ── Losses 
def gradient_penalty(y_true, y_pred):
    dy_t = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    dx_t = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    dy_p = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    dx_p = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    return (tf.reduce_mean(tf.square(dy_t - dy_p)) +
            tf.reduce_mean(tf.square(dx_t - dx_p)))


def combined_loss(y_true, y_pred):
    """MSE + MAE + gradient-penalty. The gradient term explicitly penalises
    differences in spatial gradients, which is where stress concentrations
    live — empirically the biggest driver of forward-model (and therefore
    cyclic) R²."""
    mse  = tf.reduce_mean(tf.square(y_true - y_pred))
    mae  = tf.reduce_mean(tf.abs(y_true - y_pred))
    grad = gradient_penalty(y_true, y_pred)
    return 0.4 * mse + 0.4 * mae + 0.2 * grad


# Callbacks 
def get_callbacks(ckpt_path):
    return [
        EarlyStopping(monitor='val_loss', patience=25,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                           patience=6, min_lr=1e-6, verbose=0),
        ModelCheckpoint(ckpt_path, monitor='val_loss',
                         save_best_only=True, verbose=0),
    ]
