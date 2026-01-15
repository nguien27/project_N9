# Main script - Orchestrate toan bo quy trinh training va evaluation

import os
import sys
import random
import numpy as np
import json
from pathlib import Path

# Add parent to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_PATH, RESULTS_DIR, CONFUSION_MATRIX_PATH, TRAINING_HISTORY_PATH, EVALUATION_RESULTS_PATH, EDA_RESULTS_PATH

# Import cac module
from CNN_preprocessing import tai_du_lieu_audio_nang_cao as tai_du_lieu_audio, chia_du_lieu
from CNN_NHNguyen import thiet_lap_seed_lap_lai, huan_luyen_mo_hinh, tao_mo_hinh_cnn
from eda import chay_eda_day_du
from evaluation import danh_gia_mo_hinh, ve_confusion_matrix, ve_lich_su_huan_luyen, luu_ket_qua_json
from feature_engineering import chuyen_sang_mel
from gradcam_feature_extraction import GradCAMFeatureExtractor

# CAU HINH
THU_MUC_AUDIO = 'project_N9/data/Audio Files'
TAN_SO_MAU = 16000
N_FFT = 2048
DO_NHAY = 512
THOI_LUONG = 4
SO_MEL = 128
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 5e-5
SEED = 42

def chuan_bi_du_lieu(du_lieu_file, file_train, file_val):
    """Chuan bi du lieu cho training - CO DATA AUGMENTATION """
    print("\nCHUAN BI DU LIEU")
    print("="*50)
    
    # BUOC 1: Tinh mean/std tu tap train (khong tang cuong)
    print("Buoc 1: Tinh toan thong ke tap train...")
    mel_train_tam = []
    for ten_file in file_train:
        tin_hieu, _ = du_lieu_file[ten_file]
        mel, _ = chuyen_sang_mel(tin_hieu, TAN_SO_MAU, N_FFT, DO_NHAY, SO_MEL, 
                             tang_cuong=False)
        mel_train_tam.append(mel)
    
    mel_train_tam = np.array(mel_train_tam)
    trung_binh_train = mel_train_tam.mean()
    do_lech_train = mel_train_tam.std()
    print(f'Trung binh train: {trung_binh_train:.4f}, Do lech train: {do_lech_train:.4f}')
    
    # BUOC 2: Tao tap train voi tang cuong (4 mau augment moi file)
    print('\nBuoc 2: Tao tap train (co tang cuong)...')
    X_train_mel = []
    y_train_mel = []
    for i, ten_file in enumerate(file_train):
        tin_hieu, nhan = du_lieu_file[ten_file]
        
        # Mau goc
        mel, _ = chuyen_sang_mel(tin_hieu, TAN_SO_MAU, N_FFT, DO_NHAY, SO_MEL,
                             tang_cuong=False, trung_binh=trung_binh_train, 
                             do_lech=do_lech_train)
        X_train_mel.append(mel)
        y_train_mel.append(nhan)
        
        # 4 mau tang cuong voi muc do khac nhau
        for muc_do in range(4):
            md = min(muc_do, 2)  # Gioi han muc_do toi da la 2
            mel_aug, _ = chuyen_sang_mel(tin_hieu, TAN_SO_MAU, N_FFT, DO_NHAY, SO_MEL,
                                     tang_cuong=True, muc_do=md, 
                                     trung_binh=trung_binh_train, 
                                     do_lech=do_lech_train)
            X_train_mel.append(mel_aug)
            y_train_mel.append(nhan)
        
        if (i+1) % 50 == 0:
            print(f'Train: {i+1}/{len(file_train)}')
    
    X_train = np.array(X_train_mel)
    y_train = np.array(y_train_mel)
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train: {np.sum(y_train==0)} binh thuong, {np.sum(y_train==1)} bat thuong')
    
    # BUOC 3: Tao tap val voi cung mean/std (khong tang cuong)
    print('\nBuoc 3: Tao tap val...')
    X_val_list = []
    for f in file_val:
        mel, _ = chuyen_sang_mel(du_lieu_file[f][0], TAN_SO_MAU, N_FFT, DO_NHAY, SO_MEL,
                                      tang_cuong=False, trung_binh=trung_binh_train, 
                                      do_lech=do_lech_train)
        X_val_list.append(mel)
    X_val = np.array(X_val_list)
    y_val = np.array([du_lieu_file[f][1] for f in file_val])
    print(f'X_val shape: {X_val.shape}')
    print(f'y_val: {np.sum(y_val==0)} binh thuong, {np.sum(y_val==1)} bat thuong')
    
    return X_train, y_train, X_val, y_val, trung_binh_train, do_lech_train

def main():
    """Main function"""
    print("="*60)
    print("PHAN LOAI AM THANH PHOI - PIPELINE TRAINING")
    print("="*60)
    
    # 1. Thiet lap seed
    thiet_lap_seed_lap_lai(SEED)
    
    # 2. Tai du lieu audio
    print("\n1. TAI DU LIEU AUDIO")
    print("="*50)
    
    # Buoc 1: Lay danh sach tat ca file .wav tu thu muc
    cac_file_wav = sorted([f for f in os.listdir(THU_MUC_AUDIO) if f.endswith('.wav')])
    print(f'Tong so file: {len(cac_file_wav)}')
    
    # Buoc 2: Chia file thanh train/val 80/20
    print("\n2. CHIA DU LIEU (THEO FILE)")
    print("="*50)
    
    # Phan loai file theo nhan
    file_binh_thuong = sorted([f for f in cac_file_wav if f.split('_', 1)[1].split(',')[0].strip() == 'N'])
    file_bat_thuong = sorted([f for f in cac_file_wav if f.split('_', 1)[1].split(',')[0].strip() != 'N'])
    
    print(f'Binh thuong: {len(file_binh_thuong)}, Bat thuong: {len(file_bat_thuong)}')
    
    # Xao tron va chia file
    random.seed(SEED)
    random.shuffle(file_binh_thuong)
    random.shuffle(file_bat_thuong)
    
    # Chia tung loai file
    n_train_0 = int(len(file_binh_thuong) * 0.8)
    n_train_1 = int(len(file_bat_thuong) * 0.8)
    
    file_train_list = file_binh_thuong[:n_train_0] + file_bat_thuong[:n_train_1]
    file_val_list = file_binh_thuong[n_train_0:] + file_bat_thuong[n_train_1:]
    
    random.shuffle(file_train_list)
    random.shuffle(file_val_list)
    
    print(f'Train files: {len(file_train_list)} ({n_train_0} binh thuong + {n_train_1} bat thuong)')
    print(f'Val files: {len(file_val_list)} ({len(file_binh_thuong)-n_train_0} binh thuong + {len(file_bat_thuong)-n_train_1} bat thuong)')
    
    # Buoc 3: Tai va cat file train
    print('\nTai va cat file TRAIN...')
    du_lieu_train, thong_ke_train = tai_du_lieu_audio(file_train_list, THU_MUC_AUDIO, TAN_SO_MAU)
    
    # Buoc 4: Tai va cat file val
    print('\nTai va cat file VAL...')
    du_lieu_val, thong_ke_val = tai_du_lieu_audio(file_val_list, THU_MUC_AUDIO, TAN_SO_MAU)
    
    # Gop du lieu train va val
    du_lieu_file = {**du_lieu_train, **du_lieu_val}
    
    # Tao danh sach file segment cho chuan_bi_du_lieu
    file_train = list(du_lieu_train.keys())
    file_val = list(du_lieu_val.keys())
    
    # Thong ke tong hop
    thong_ke_cat = {
        'tong_so_file_goc': thong_ke_train['tong_so_file_goc'] + thong_ke_val['tong_so_file_goc'],
        'tong_so_doan_cat': thong_ke_train['tong_so_doan_cat'] + thong_ke_val['tong_so_doan_cat'],
        'chi_tiet_cat': thong_ke_train['chi_tiet_cat'] + thong_ke_val['chi_tiet_cat']
    }
    
    # 3. Chay EDA (phan tich du lieu)
    print("\n3. PHAN TICH DU LIEU (EDA)")
    print("="*50)
    chay_eda_day_du(du_lieu_file, THU_MUC_AUDIO, TAN_SO_MAU, duong_dan_luu=str(EDA_RESULTS_PATH), thong_ke_cat=thong_ke_cat)
    
    # 4. Chuan bi du lieu
    print("\n4. CHUAN BI DU LIEU")
    print("="*50)
    X_train, y_train, X_val, y_val, _, _ = chuan_bi_du_lieu(du_lieu_file, file_train, file_val)
    
    # 5. Oversample de can bang
    print("\n5. OVERSAMPLE DE CAN BANG")
    print("="*50)
    from sklearn.utils import resample
    
    idx_0 = np.where(y_train == 0)[0]
    idx_1 = np.where(y_train == 1)[0]
    n0, n1 = len(idx_0), len(idx_1)
    print(f'Truoc oversample: {n0} binh thuong, {n1} bat thuong')
    
    if n0 < n1:
        idx_0_os = resample(idx_0, replace=True, n_samples=n1, random_state=SEED)
        idx_new = np.concatenate([idx_1, idx_0_os])
    elif n1 < n0:
        idx_1_os = resample(idx_1, replace=True, n_samples=n0, random_state=SEED)
        idx_new = np.concatenate([idx_0, idx_1_os])
    else:
        idx_new = np.concatenate([idx_0, idx_1])
    
    np.random.seed(SEED)
    np.random.shuffle(idx_new)
    X_train = X_train[idx_new]
    y_train = y_train[idx_new]
    print(f'Sau oversample: {np.sum(y_train==0)} binh thuong, {np.sum(y_train==1)} bat thuong')
    
    # 6. Huan luyen mo hinh
    print("\n6. HUAN LUYEN MO HINH")
    print("="*50)
    model, history = huan_luyen_mo_hinh(X_train, y_train, X_val, y_val, 
                                        duong_dan_luu=str(MODEL_PATH))
    
    # 7. Danh gia mo hinh
    print("\n7. DANH GIA MO HINH")
    print("="*50)
    results = danh_gia_mo_hinh(model, X_val, y_val)
    
    # 8. Ve bieu do
    print("\n8. VE BIEU DO")
    print("="*50)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    ve_confusion_matrix(
        np.array(results['confusion_matrix']), 
        save_path=str(CONFUSION_MATRIX_PATH)
    )
    
    ve_lich_su_huan_luyen(
        history, 
        save_path=str(TRAINING_HISTORY_PATH)
    )
    
    # 9. Luu ket qua
    print("\n9. LUU KET QUA")
    print("="*50)
    luu_ket_qua_json(results, str(EVALUATION_RESULTS_PATH))
    
    # 10. Trich xuat dac trung bang Grad-CAM
    print("\n10. TRICH XUAT DAC TRUNG BANG GRAD-CAM")
    print("="*50)
    try:
        gradcam_output_dir = 'project_N9/outputs/gradcam'
        extractor = GradCAMFeatureExtractor(str(MODEL_PATH))
        extractor.extract_features_batch(THU_MUC_AUDIO, gradcam_output_dir, num_samples=12)
    except Exception as e:
        print(f"Loi khi trich xuat Grad-CAM: {e}")
    
    print("\n" + "="*60)
    print("HOAN THANH TRAINING VA EVALUATION")
    print("="*60)

if __name__ == "__main__":
    main()
