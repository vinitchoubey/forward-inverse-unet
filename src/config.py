

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Dataset paths ────────────────────────────────────────────────
DATA_DIR   = os.path.join(BASE_DIR, 'dataset')
XLSX_PATH  = os.path.join(DATA_DIR, 'output.xlsx')
STRESS_DIR = os.path.join(DATA_DIR, 'stress')
# cord*.txt is searched for directly inside DATA_DIR (glob pattern)


SAVE_DIR  = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(SAVE_DIR, 'models')
FWD_DIR   = os.path.join(SAVE_DIR, 'forward')
INV_DIR   = os.path.join(SAVE_DIR, 'inverse')
CYC_DIR   = os.path.join(SAVE_DIR, 'cyclic')

for _d in (DATA_DIR, STRESS_DIR, SAVE_DIR, MODEL_DIR, FWD_DIR, INV_DIR, CYC_DIR):
    os.makedirs(_d, exist_ok=True)

#  Hyper-parameters 
N_SAMPLES  = 4500
GRID_H     = 64
GRID_W     = 128
EPOCHS     = 60
BATCH_SIZE = 16
N_PLOT     = 3          # rows (samples) in every visualisation grid
N_DESIGN   = 226

SEED_FORWARD = 42
SEED_INVERSE = 123
