import os
import random
import numpy as np
import librosa
from scipy import signal
from config import (SU_DUNG_GIAM_NHIEU, SU_DUNG_LOC_BANG_THONG, 
                    SU_DUNG_CHUAN_HOA, SU_DUNG_CAT_LANG)

# Import noisereduce (can cai: pip install noisereduce)
try:
    import noisereduce as nr
    CO_NOISEREDUCE = True
except ImportError:
    CO_NOISEREDUCE = False
    print('Chua cai noisereduce. Chay: pip install noisereduce')

# CAU HINH TIEN XU LY RAW AUDIO
TAN_SO_PHOI_THAP = 50
TAN_SO_PHOI_CAO = 4000

def giam_nhieu_nhe(tin_hieu, tan_so):
    # Giam nhieu nhe cho tin hieu audio
    if not CO_NOISEREDUCE:
        return tin_hieu
    try:
        return nr.reduce_noise(y=tin_hieu, sr=tan_so, prop_decrease=0.3, stationary=True)
    except:
        return tin_hieu

def loc_bang_thong(tin_hieu, tan_so, thap=TAN_SO_PHOI_THAP, cao=TAN_SO_PHOI_CAO):
    # Loc tan so trong khoang phu hop voi am thanh phoi
    nyquist = tan_so / 2
    thap_chuan = max(0.01, thap / nyquist)
    cao_chuan = min(0.99, cao / nyquist)
    b, a = signal.butter(3, [thap_chuan, cao_chuan], btype='band')
    return signal.filtfilt(b, a, tin_hieu)

def chuan_hoa_audio(tin_hieu):
    # Chuan hoa tin hieu audio ve khoang [-1, 1] 
    gia_tri_max = np.max(np.abs(tin_hieu))
    return tin_hieu / gia_tri_max if gia_tri_max > 0 else tin_hieu

def cat_khoang_lang(tin_hieu, nguong_db=20):
    # Cat bo khoang lang o dau va cuoi
    do_dai_truoc = len(tin_hieu)
    tin_hieu_da_cat, _ = librosa.effects.trim(tin_hieu, top_db=nguong_db)
    do_dai_sau = len(tin_hieu_da_cat)
    
    # In thong tin
    ti_le_cat = (1 - do_dai_sau / do_dai_truoc) * 100
    print(f"  Trim silence: {do_dai_truoc} -> {do_dai_sau} samples ({ti_le_cat:.1f}% cat)")
    
    return tin_hieu_da_cat





def tien_xu_ly_audio_tho(tin_hieu, tan_so, tan_so_mau=16000, thoi_luong=4):
    # Tien xu ly tong hop cho tin hieu audio tho 

    if SU_DUNG_GIAM_NHIEU:
        tin_hieu = giam_nhieu_nhe(tin_hieu, tan_so)
    if SU_DUNG_LOC_BANG_THONG:
        tin_hieu = loc_bang_thong(tin_hieu, tan_so)
    if SU_DUNG_CHUAN_HOA:
        tin_hieu = chuan_hoa_audio(tin_hieu)  # Max normalization
    if SU_DUNG_CAT_LANG:
        tin_hieu = cat_khoang_lang(tin_hieu)
    
    return tin_hieu

def cat_audio_thanh_doan(tin_hieu, tan_so_mau=16000, thoi_luong_doan=4):
    # Cat audio thanh nhieu doan, moi doan co do dai chuan
    so_mau_moi_doan = tan_so_mau * thoi_luong_doan
    so_doan = len(tin_hieu) // so_mau_moi_doan
    
    cac_doan = []
    for i in range(so_doan):
        start = i * so_mau_moi_doan
        end = start + so_mau_moi_doan
        doan = tin_hieu[start:end]
        cac_doan.append(doan)
    
    return cac_doan

def tai_du_lieu_audio(thu_muc_audio, tan_so_mau=16000, thoi_luong_doan=4):
    # Load tat ca file audio tu thu muc, tien xu ly, va cat thanh nhieu doan
    du_lieu_file = {}
    thong_ke_cat = {
        'tong_so_file_goc': 0,
        'tong_so_doan_cat': 0,
        'chi_tiet_cat': []  # List chua chi tiet tung file
    }
    bo_qua = 0
    cac_file = sorted([f for f in os.listdir(thu_muc_audio) if f.endswith('.wav')])
    
    for ten_file in cac_file:
        try:
            # Trich xuat nhan tu ten file
            chan_doan = ten_file.split('_', 1)[1].split(',')[0].strip()
            nhan = 0 if chan_doan == 'N' else 1
            
            # Load va tien xu ly audio
            duong_dan = os.path.join(thu_muc_audio, ten_file)
            tin_hieu, _ = librosa.load(duong_dan, sr=tan_so_mau)
            tin_hieu = tien_xu_ly_audio_tho(tin_hieu, tan_so_mau)
            
            # Cat thanh nhieu doan
            cac_doan = cat_audio_thanh_doan(tin_hieu, tan_so_mau, thoi_luong_doan)
            
            # Luu tung doan vao dict
            for i, doan in enumerate(cac_doan):
                ten_doan = f"{ten_file[:-4]}_doan_{i}.wav"  # Loai bo .wav va them _doan_i
                du_lieu_file[ten_doan] = (doan, nhan)
            
            # Cap nhat thong ke
            thong_ke_cat['tong_so_file_goc'] += 1
            thong_ke_cat['tong_so_doan_cat'] += len(cac_doan)
            thong_ke_cat['chi_tiet_cat'].append({
                'file_goc': ten_file,
                'so_doan': len(cac_doan),
                'nhan': 'Binh thuong' if nhan == 0 else 'Bat thuong'
            })
            
            print(f"  {ten_file}: {len(cac_doan)} doan ({len(tin_hieu)} samples)")
            
        except Exception as e:
            print(f"Loi khi xu ly file {ten_file}: {e}")
            bo_qua += 1
    
    print(f'\nTong: {thong_ke_cat["tong_so_file_goc"]} file goc -> {thong_ke_cat["tong_so_doan_cat"]} doan')
    print(f'Da tai {len(du_lieu_file)} doan, bo qua {bo_qua} file')
    return du_lieu_file, thong_ke_cat

def chia_du_lieu(du_lieu_file, ty_le_train=0.8, seed=42):
    # Chia du lieu thanh train va validation
    # Phan loai file theo nhan
    file_binh_thuong = sorted([f for f, (_, nhan) in du_lieu_file.items() if nhan == 0])
    file_bat_thuong = sorted([f for f, (_, nhan) in du_lieu_file.items() if nhan == 1])
    
    print(f'Binh thuong: {len(file_binh_thuong)}, Bat thuong: {len(file_bat_thuong)}')
    
    # Shuffle voi seed co dinh
    random.seed(seed)
    random.shuffle(file_binh_thuong)
    random.shuffle(file_bat_thuong)
    
    def chia_file(cac_file, ty_le_train):
        n = len(cac_file)
        cuoi_train = int(n * ty_le_train)
        return cac_file[:cuoi_train], cac_file[cuoi_train:]
    
    # Chia tung loai
    train_0, val_0 = chia_file(file_binh_thuong, ty_le_train)
    train_1, val_1 = chia_file(file_bat_thuong, ty_le_train)
    
    # Gop va shuffle
    file_train = train_0 + train_1
    file_val = val_0 + val_1
    
    random.seed(seed)
    random.shuffle(file_train)
    random.shuffle(file_val)
    
    print(f'Train: {len(file_train)} ({len(train_0)} binh thuong + {len(train_1)} bat thuong)')
    print(f'Val: {len(file_val)} ({len(val_0)} binh thuong + {len(val_1)} bat thuong)')
    
    return file_train, file_val

if __name__ == "__main__":
    print('Preprocessing functions ready!')
