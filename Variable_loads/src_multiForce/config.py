import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = os.path.join(BASE_DIR, 'datasets_multiForce')
XLSX_PATH  = os.path.join(DATA_DIR, 'output.xlsx')
STRESS_DIR = os.path.join(DATA_DIR, 'stress')
LOAD_DIR   = os.path.join(DATA_DIR, 'loaddata')

SAVE_DIR  = os.path.join(BASE_DIR, 'outputs-multiforce')
MODEL_DIR = os.path.join(SAVE_DIR, 'models')
FWD_DIR   = os.path.join(SAVE_DIR, 'forward')
INV_DIR   = os.path.join(SAVE_DIR, 'inverse')
CYC_DIR   = os.path.join(SAVE_DIR, 'cyclic')

RESULTS_DIR     = os.path.join(BASE_DIR, 'results-multiforce')
RESULTS_FWD_DIR = os.path.join(RESULTS_DIR, 'forward')
RESULTS_INV_DIR = os.path.join(RESULTS_DIR, 'inverse')
RESULTS_CYC_DIR = os.path.join(RESULTS_DIR, 'cyclic')

for _d in (DATA_DIR, STRESS_DIR, LOAD_DIR, SAVE_DIR, MODEL_DIR, FWD_DIR, INV_DIR, CYC_DIR,
           RESULTS_DIR, RESULTS_FWD_DIR, RESULTS_INV_DIR, RESULTS_CYC_DIR):
    os.makedirs(_d, exist_ok=True)

N_SAMPLES  = 4900
GRID_H     = 64
GRID_W     = 128
EPOCHS     = 80
BATCH_SIZE = 16
N_PLOT     = 3
N_DESIGN   = 226

LOAD_SPLAT_SIGMA  = 3.0
L2_REG            = 1e-5
STRESS_CLIP_PCTL  = 99.5

SEED_FORWARD = 42
SEED_INVERSE = 123
