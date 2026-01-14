"""
Cau hinh tong the cho project
"""

import os
from pathlib import Path

# Duong dan
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data' / 'Audio Files'
SRC_DIR = PROJECT_ROOT / 'src'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
RESULTS_DIR = OUTPUTS_DIR / 'results'

# Tao thu muc neu chua co
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Audio config
TAN_SO_MAU = 16000  # Sample rate
THOI_LUONG = 4  # Thoi luong audio (giay)
N_FFT = 2048  # FFT size
DO_NHAY = 512  # Hop length
SO_MEL = 128  # So mel bins

# Preprocessing config
TAN_SO_PHOI_THAP = 50  # Hz
TAN_SO_PHOI_CAO = 4000  # Hz
NGUONG_DB_TRIM = 20  # dB
TARGET_RMS = 0.1  # RMS target

# Model config
BATCH_SIZE = 16  
EPOCHS = 200
LEARNING_RATE = 5e-5 
SEED = 42

# Data split
TY_LE_TRAIN = 0.8
TY_LE_VAL = 0.2

# Model paths
MODEL_PATH = MODELS_DIR / 'lung_model_balanced.keras'

# Output paths
CONFUSION_MATRIX_PATH = RESULTS_DIR / 'confusion_matrix.png'
TRAINING_HISTORY_PATH = RESULTS_DIR / 'training_history.png'
EVALUATION_RESULTS_PATH = RESULTS_DIR / 'evaluation_results.json'
EDA_RESULTS_PATH = RESULTS_DIR / 'eda_results.json'

# Preprocessing flags
SU_DUNG_GIAM_NHIEU = True
SU_DUNG_LOC_BANG_THONG = True
SU_DUNG_CHUAN_HOA = True
SU_DUNG_CAT_LANG = True
SU_DUNG_PRE_EMPHASIS = True

# Feature engineering config - Data Augmentation
# Các kỹ thuật tăng cường dữ liệu để cải thiện khả năng tổng quát của mô hình
AUGMENTATION_CONFIG = {
    # Pitch Shift - Thay đổi cao độ âm thanh
    'pitch_shift': True,  # Bật/tắt pitch shift
    'pitch_shift_range': (-2, 2),  # Thay đổi cao độ từ -2 đến +2 semitones
    
    # Time Stretch - Thay đổi tốc độ phát âm thanh
    'time_stretch': True,  # Bật/tắt time stretch
    'time_stretch_range': (0.9, 1.1),  # Thay đổi tốc độ từ 0.9x đến 1.1x
    
    # Noise Injection - Thêm nhiễu vào âm thanh
    'noise_injection': True,  # Bật/tắt noise injection
    'noise_factor': 0.005,  # Mức độ nhiễu (0.005 = 0.5% biên độ tín hiệu)
    
    # Dynamic Range Compression - Nén dải động
    'dynamic_range_compression': True,  # Bật/tắt compression
    'compression_ratio': 4,  # Tỷ lệ nén 4:1 (giảm biên độ 4 lần)
}
