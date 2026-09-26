import gc

from src_multiForce.data_processing import load_and_prepare_data
from src_multiForce.forward_model import train_forward_model
from src_multiForce.inverse_model import train_inverse_model
from src_multiForce.cyclic_validation import run_cyclic_validation


def main():
    print("Loading and preparing data...")
    data = load_and_prepare_data()

    print("Training forward model (Thickness + Load -> Stress)...")
    fwd_result = train_forward_model(data)

    print("Training inverse model (Stress + Load -> Thickness)...")
    inv_result = train_inverse_model(data)

    print("Running cyclic validation...")
    cyc_result = run_cyclic_validation(data, fwd_result, inv_result)

    print("Pipeline complete.")
    print(f"  Forward R2 (physical) = {fwd_result['r2']:.4f}")
    print(f"  Forward R2 (log-norm) = {fwd_result['r2_norm']:.4f}")
    print(f"  Inverse R2            = {inv_result['r2']:.4f}")
    print(f"  Cyclic R2 (log-norm)  = {cyc_result['cyc_r2_norm']:.4f}")
    print(f"  Cyclic R2 (physical)  = {cyc_result['cyc_r2_phys']:.4f}")

    gc.collect()


if __name__ == '__main__':
    main()
