#  FORWARD + INVERSE UNET 
#   WITH OPTIMISED CYCLIC PHYSICS VALIDATION
#     A. Load & preprocess data           (src/data_processing.py)
#     B. Train + evaluate Forward UNet    (src/forward_model.py)
#     C. Train + evaluate Inverse UNet    (src/inverse_model.py)
#     D. Optimised cyclic validation      (src/cyclic_validation.py)

import gc
import warnings

from src import config as cfg
from src.data_processing import load_and_prepare_data
from src.forward_model import train_forward_model
from src.inverse_model import train_inverse_model
from src.cyclic_validation import run_cyclic_validation

warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print(" SECTION A – Loading & preprocessing data")
    print("=" * 60)
    data = load_and_prepare_data()
    print(f"Loaded {data.n_samples} samples | "
          f"Train={len(data.X_tr)}  Val={len(data.X_val)}  Test={len(data.X_te)}")

    print("=" * 60)
    print(" SECTION B – Training Forward UNet (Thickness -> Stress)")
    print("=" * 60)
    fwd_result = train_forward_model(data)
    print(f"Forward R² = {fwd_result['r2']:.4f}")

    print("=" * 60)
    print(" SECTION C – Training Inverse UNet (Stress -> Thickness)")
    print("=" * 60)
    inv_result = train_inverse_model(data)
    print(f"Inverse R² = {inv_result['r2']:.4f}  (low is expected — ill-posed)")

    print("=" * 60)
    print(" SECTION D – Optimised Cyclic Validation")
    print("=" * 60)
    cyc_result = run_cyclic_validation(data, fwd_result, inv_result)

    gc.collect()

    print("=" * 60)
    print(" FINAL SUMMARY")
    print("=" * 60)
    print(f"Forward  R²  (direct)   = {fwd_result['r2']:.4f}")
    print(f"Inverse  R²  (direct)   = {inv_result['r2']:.4f}")
    print(f"Cyclic   R²  (log-norm) = {cyc_result['cyc_r2_norm']:.4f}")
    print(f"Cyclic   R²  (physical) = {cyc_result['cyc_r2_phys']:.4f}")
    print(f"\nAll plots saved under: {cfg.SAVE_DIR}")


if __name__ == '__main__':
    main()
