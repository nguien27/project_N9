import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Mapping de chuan hoa ten chan doan (gop cac ten giong nhau)
MAPPING_CHAN_DOAN = {
    'N': 'Normal',
    'Asthma': 'Asthma',
    'asthma': 'Asthma',
    'Asthma and lung fibrosis': 'Asthma + Lung Fibrosis',
    'Heart Failure': 'Heart Failure',
    'heart failure': 'Heart Failure',
    'Heart Failure + COPD': 'Heart Failure + COPD',
    'Heart Failure + Lung Fibrosis': 'Heart Failure + Lung Fibrosis',
    'COPD': 'COPD',
    'copd': 'COPD',
    'Pneumonia': 'Pneumonia',
    'pneumonia': 'Pneumonia',
    'Lung Fibrosis': 'Lung Fibrosis',
    'Bronchitis': 'Bronchitis',
    'BRON': 'Bronchitis',
    'Pleural Effusion': 'Pleural Effusion',
    'Crep': 'Crepitations',
}

def tong_quan_dataset(thu_muc_audio):
    # Tong quan dataset - load danh sach file va parse thong tin
    print("="*60)
    print("1. TONG QUAN DATASET")
    print("="*60)
    
    # Load danh sach file
    cac_file = sorted([f for f in os.listdir(thu_muc_audio) if f.endswith('.wav')])
    print(f'\nTong so file audio: {len(cac_file)}')
    print(f'\n5 file dau tien:')
    for f in cac_file[:5]:
        print(f'  - {f}')
    
    # Parse thong tin tu ten file
    # Format: ID_ChanDoan,AmThanh,ViTri,Tuoi,GioiTinh.wav
    data = []
    for f in cac_file:
        try:
            parts = f.replace('.wav', '').split('_', 1)
            file_id = parts[0]
            info = parts[1].split(',')
            
            chan_doan = info[0].strip()
            am_thanh = info[1].strip() if len(info) > 1 else 'N/A'
            vi_tri = info[2].strip() if len(info) > 2 else 'N/A'
            tuoi = int(info[3].strip()) if len(info) > 3 and info[3].strip().isdigit() else None
            gioi_tinh = info[4].strip() if len(info) > 4 else 'N/A'
            
            # Phan loai binh thuong / bat thuong
            label = 'Binh thuong' if chan_doan == 'N' else 'Bat thuong'
            
            # Ap dung mapping de chuan hoa ten chan doan
            chan_doan_chuan = MAPPING_CHAN_DOAN.get(chan_doan, chan_doan)
            
            data.append({
                'file': f,
                'id': file_id,
                'chan_doan': chan_doan,
                'chan_doan_chuan': chan_doan_chuan,
                'am_thanh': am_thanh,
                'vi_tri': vi_tri,
                'tuoi': tuoi,
                'gioi_tinh': gioi_tinh,
                'label': label
            })
        except Exception as e:
            print(f'Loi parse file {f}: {e}')
    
    df = pd.DataFrame(data)
    print(f'\nDataFrame shape: {df.shape}')
    print(f'\nDau tien 10 hang:')
    print(df.head(10))
    
    return df

def phan_tich_phan_phoi_chan_doan(df, thu_muc_luu=None):
    # Phan tich phan phoi chan doan
    print("\n" + "="*60)
    print("2. PHAN TICH PHAN PHOI CHAN DOAN")
    print("="*60)
    
    # Dem so luong (su dung chan_doan_chuan da chuan hoa)
    chan_doan_counts = df['chan_doan_chuan'].value_counts().sort_values(ascending=False)
    print(f'\nPhan phoi chan doan (sau khi chuan hoa):')
    print(chan_doan_counts)
    
    # Ve bieu do - chi bar chart voi font size lon hon
    fig, ax = plt.subplots(figsize=(14, 6))
    
    chan_doan_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title('So luong theo chan doan', fontsize=14, fontweight='bold')
    ax.set_ylabel('So luong', fontsize=12)
    ax.set_xlabel('Chan doan', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    plt.tight_layout()
    
    # Luu bieu do
    if thu_muc_luu:
        duong_dan = thu_muc_luu / 'phan_phoi_chan_doan.png'
        plt.savefig(str(duong_dan), dpi=300, bbox_inches='tight')
        print(f"Da luu: {duong_dan}")
    
    plt.close()
    
    return chan_doan_counts

def phan_tich_phan_phoi_label(df, thu_muc_luu=None):
    # Phan tich phan phoi nhan (Binh thuong vs Bat thuong)
    print("\n" + "="*60)
    print("3. PHAN TICH PHAN PHOI NHAN")
    print("="*60)
    
    label_counts = df['label'].value_counts()
    print(f'\nPhan phoi nhan:')
    print(label_counts)
    print(f'\nTi le:')
    print(df['label'].value_counts(normalize=True))
    
    # Ve bieu do
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    colors = ['lightblue', 'lightcoral']
    axes[0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', colors=colors)
    axes[0].set_title('Phan phoi nhan', fontsize=12, fontweight='bold')
    
    # Bar chart
    label_counts.plot(kind='bar', ax=axes[1], color=colors, edgecolor='black')
    axes[1].set_title('So luong theo nhan', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('So luong')
    axes[1].set_xlabel('Nhan')
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    # Luu bieu do
    if thu_muc_luu:
        duong_dan = thu_muc_luu / 'phan_phoi_nhan.png'
        plt.savefig(str(duong_dan), dpi=300, bbox_inches='tight')
        print(f"Da luu: {duong_dan}")
    
    plt.close()
    
    return label_counts

def phan_tich_tuoi_gioi_tinh(df, thu_muc_luu=None):
    # Phan tich tuoi va gioi tinh
    print("\n" + "="*60)
    print("4. PHAN TICH TUOI VA GIOI TINH")
    print("="*60)
    
    # Thong ke tuoi
    print(f'\nThong ke tuoi:')
    print(f'  Trung binh: {df["tuoi"].mean():.1f}')
    print(f'  Median: {df["tuoi"].median():.1f}')
    print(f'  Min/Max: {df["tuoi"].min()} / {df["tuoi"].max()}')
    print(f'  Std: {df["tuoi"].std():.1f}')
    
    # Thong ke gioi tinh
    print(f'\nThong ke gioi tinh:')
    print(df['gioi_tinh'].value_counts())
    
    # Ve bieu do
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram tuoi
    axes[0, 0].hist(df['tuoi'].dropna(), bins=20, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Phan phoi tuoi', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Tuoi')
    axes[0, 0].set_ylabel('So luong')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Bar chart gioi tinh
    df['gioi_tinh'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['pink', 'lightblue'], edgecolor='black')
    axes[0, 1].set_title('Phan phoi gioi tinh', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('So luong')
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # Tuoi theo nhan
    df.boxplot(column='tuoi', by='label', ax=axes[1, 0])
    axes[1, 0].set_title('Tuoi theo nhan', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Nhan')
    axes[1, 0].set_ylabel('Tuoi')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=0)
    
    # Tuoi theo gioi tinh
    df.boxplot(column='tuoi', by='gioi_tinh', ax=axes[1, 1])
    axes[1, 1].set_title('Tuoi theo gioi tinh', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Gioi tinh')
    axes[1, 1].set_ylabel('Tuoi')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    # Luu bieu do
    if thu_muc_luu:
        duong_dan = thu_muc_luu / 'tuoi_gioi_tinh.png'
        plt.savefig(str(duong_dan), dpi=300, bbox_inches='tight')
        print(f"Da luu: {duong_dan}")
    
    plt.close()

def hien_thi_mau_audio(du_lieu_file, so_mau=4, tan_so_mau=16000, thu_muc_luu=None):
    # Hien thi mau tin hieu audio va mel spectrogram - 4 loai benh
    print("\n" + "="*60)
    print("6. HIEN THI MAU TIN HIEU")
    print("="*60)
    
    # Phan loai file theo chan doan
    file_binh_thuong = [f for f, (_, l) in du_lieu_file.items() if l == 0]
    file_bat_thuong = [f for f, (_, l) in du_lieu_file.items() if l == 1]
    
    # Phan loai bat thuong theo ten file
    file_asthma = [f for f in file_bat_thuong if 'asthma' in f.lower()]
    file_heart = [f for f in file_bat_thuong if 'heart' in f.lower()]
    file_copd = [f for f in file_bat_thuong if 'copd' in f.lower()]
    
    # Lay mau tu moi loai
    np.random.seed(42)
    
    mau_binh_thuong = np.random.choice(file_binh_thuong, min(so_mau, len(file_binh_thuong)), replace=False) if file_binh_thuong else []
    mau_asthma = np.random.choice(file_asthma, min(so_mau, len(file_asthma)), replace=False) if file_asthma else []
    mau_heart = np.random.choice(file_heart, min(so_mau, len(file_heart)), replace=False) if file_heart else []
    mau_copd = np.random.choice(file_copd, min(so_mau, len(file_copd)), replace=False) if file_copd else []
    
    # Tao 4 khung, moi khung 4 hàng (4 loai benh)
    loai_benh = [
        ('Binh thuong', list(mau_binh_thuong)),
        ('Asthma', list(mau_asthma)),
        ('Heart Failure', list(mau_heart)),
        ('COPD', list(mau_copd))
    ]
    
    for khung_idx, (ten_loai, cac_file) in enumerate(loai_benh):
        if not cac_file:
            print(f"Khong co mau cho loai: {ten_loai}")
            continue
        
        # Tao khung voi 4 hàng (moi hàng 1 file)
        fig, axes = plt.subplots(len(cac_file), 2, figsize=(16, 3*len(cac_file)))
        
        # Neu chi co 1 file, axes khong phai la 2D array
        if len(cac_file) == 1:
            axes = axes.reshape(1, -1)
        
        for i, file_name in enumerate(cac_file):
            tin_hieu, nhan = du_lieu_file[file_name]
            
            # Cat audio ve 4 giay (thong nhat voi model)
            thoi_luong = 4
            so_mau_mong_muon = tan_so_mau * thoi_luong
            if len(tin_hieu) > so_mau_mong_muon:
                tin_hieu_cat = tin_hieu[:so_mau_mong_muon]
            else:
                tin_hieu_cat = np.pad(tin_hieu, (0, so_mau_mong_muon - len(tin_hieu)))
            
            # Waveform
            axes[i, 0].plot(tin_hieu_cat, color='steelblue', linewidth=0.8)
            axes[i, 0].set_title(f'{file_name[:40]}... - {ten_loai}', fontsize=11, fontweight='bold')
            axes[i, 0].set_ylabel('Bien do', fontsize=10)
            axes[i, 0].set_xlabel('Samples', fontsize=10)
            axes[i, 0].grid(True, alpha=0.3)
            
            # Mel Spectrogram (cat ve 4 giay)
            mel = librosa.feature.melspectrogram(y=tin_hieu_cat, sr=tan_so_mau, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            im = axes[i, 1].imshow(mel_db, aspect='auto', origin='lower', cmap='viridis')
            axes[i, 1].set_title(f'Mel Spectrogram - {ten_loai}', fontsize=11, fontweight='bold')
            axes[i, 1].set_ylabel('Mel bins', fontsize=10)
            axes[i, 1].set_xlabel('Time frames', fontsize=10)
            axes[i, 1].grid(True, alpha=0.3, color='white', linewidth=0.5)
            cbar = plt.colorbar(im, ax=axes[i, 1], label='dB')
            cbar.ax.tick_params(labelsize=9)
        
        plt.tight_layout()
        
        # Luu bieu do
        if thu_muc_luu:
            ten_file_luu = f'mau_audio_{ten_loai.lower().replace(" ", "_")}.png'
            duong_dan = thu_muc_luu / ten_file_luu
            plt.savefig(str(duong_dan), dpi=300, bbox_inches='tight')
            print(f"Da luu: {duong_dan}")
        
        plt.close()
    
    print(f"Hoan thanh hien thi mau audio cho 4 loai benh")

def ve_mel_spectrogram_truoc_sau_tien_xu_ly(du_lieu_file, tan_so_mau=16000, so_mau=4, thu_muc_luu=None):
    # Ve Mel Spectrogram TRUOC va SAU cac buoc tien xu ly
    print("\n" + "="*60)
    print("5. MEL SPECTROGRAM TRUOC/SAU TIEN XU LY")
    print("="*60)
    
    # Import cac ham tien xu ly
    from CNN_preprocessing import (
        giam_nhieu_nhe, loc_bang_thong, nang_cao_tan_so_cao, 
        chuan_hoa_audio, cat_khoang_lang
    )
    from config import (
        SU_DUNG_GIAM_NHIEU, SU_DUNG_LOC_BANG_THONG, SU_DUNG_PRE_EMPHASIS, 
        SU_DUNG_CHUAN_HOA, SU_DUNG_CAT_LANG
    )
    
    # Lay mau ngau nhien
    ten_file = list(du_lieu_file.keys())
    np.random.seed(42)
    mau_file = np.random.choice(ten_file, min(so_mau, len(ten_file)), replace=False)
    
    # Tao khung voi 4 hang (moi hang 1 file), 2 cot (mel truoc, mel sau)
    fig, axes = plt.subplots(len(mau_file), 2, figsize=(16, 5*len(mau_file)))
    
    # Neu chi co 1 file, axes khong phai la 2D array
    if len(mau_file) == 1:
        axes = axes.reshape(1, -1)
    
    for i, file_name in enumerate(mau_file):
        tin_hieu_goc, nhan = du_lieu_file[file_name]
        
        # Cat audio ve 4 giay
        thoi_luong = 4
        so_mau_mong_muon = tan_so_mau * thoi_luong
        if len(tin_hieu_goc) > so_mau_mong_muon:
            tin_hieu_goc = tin_hieu_goc[:so_mau_mong_muon]
        else:
            tin_hieu_goc = np.pad(tin_hieu_goc, (0, so_mau_mong_muon - len(tin_hieu_goc)))
        
        # ===== MEL SPECTROGRAM TRUOC TIEN XU LY =====
        mel_truoc = librosa.feature.melspectrogram(y=tin_hieu_goc, sr=tan_so_mau, n_mels=128)
        mel_db_truoc = librosa.power_to_db(mel_truoc, ref=np.max)
        
        im1 = axes[i, 0].imshow(mel_db_truoc, aspect='auto', origin='lower', cmap='viridis', vmin=-80, vmax=0)
        axes[i, 0].set_title(f'{file_name[:40]}...\nMel Spectrogram TRUOC tien xu ly', 
                            fontsize=12, fontweight='bold')
        axes[i, 0].set_ylabel('Mel bins', fontsize=11)
        axes[i, 0].set_xlabel('Time frames', fontsize=11)
        cbar1 = plt.colorbar(im1, ax=axes[i, 0], label='dB')
        cbar1.ax.tick_params(labelsize=10)
        
        # ===== TIEN XU LY =====
        tin_hieu_sau = tin_hieu_goc.copy()
        
        if SU_DUNG_GIAM_NHIEU:
            tin_hieu_sau = giam_nhieu_nhe(tin_hieu_sau, tan_so_mau)
        
        if SU_DUNG_LOC_BANG_THONG:
            tin_hieu_sau = loc_bang_thong(tin_hieu_sau, tan_so_mau)
        
        if SU_DUNG_PRE_EMPHASIS:
            tin_hieu_sau = nang_cao_tan_so_cao(tin_hieu_sau, he_so=1.2)
        
        if SU_DUNG_CHUAN_HOA:
            tin_hieu_sau = chuan_hoa_audio(tin_hieu_sau)
        
        if SU_DUNG_CAT_LANG:
            tin_hieu_sau = cat_khoang_lang(tin_hieu_sau, nguong_db=20)
            # Pad lai ve 4 giay neu can
            if len(tin_hieu_sau) < so_mau_mong_muon:
                tin_hieu_sau = np.pad(tin_hieu_sau, (0, so_mau_mong_muon - len(tin_hieu_sau)))
            elif len(tin_hieu_sau) > so_mau_mong_muon:
                tin_hieu_sau = tin_hieu_sau[:so_mau_mong_muon]
        
        # ===== MEL SPECTROGRAM SAU TIEN XU LY =====
        mel_sau = librosa.feature.melspectrogram(y=tin_hieu_sau, sr=tan_so_mau, n_mels=128)
        mel_db_sau = librosa.power_to_db(mel_sau, ref=np.max)
        
        im2 = axes[i, 1].imshow(mel_db_sau, aspect='auto', origin='lower', cmap='viridis', vmin=-80, vmax=0)
        axes[i, 1].set_title(f'{file_name[:40]}...\nMel Spectrogram SAU tien xu ly', 
                            fontsize=12, fontweight='bold')
        axes[i, 1].set_ylabel('Mel bins', fontsize=11)
        axes[i, 1].set_xlabel('Time frames', fontsize=11)
        cbar2 = plt.colorbar(im2, ax=axes[i, 1], label='dB')
        cbar2.ax.tick_params(labelsize=10)
        
        # In thong tin
        print(f"\nFile {i+1}: {file_name}")
        print(f"  Truoc tien xu ly - Min: {mel_db_truoc.min():.2f} dB, Max: {mel_db_truoc.max():.2f} dB, Mean: {mel_db_truoc.mean():.2f} dB")
        print(f"  Sau tien xu ly  - Min: {mel_db_sau.min():.2f} dB, Max: {mel_db_sau.max():.2f} dB, Mean: {mel_db_sau.mean():.2f} dB")
        print(f"  Cac buoc tien xu ly da ap dung:")
        print(f"    - Giam nhieu: {SU_DUNG_GIAM_NHIEU}")
        print(f"    - Loc bang thong: {SU_DUNG_LOC_BANG_THONG}")
        print(f"    - Pre-emphasis: {SU_DUNG_PRE_EMPHASIS}")
        print(f"    - Chuan hoa: {SU_DUNG_CHUAN_HOA}")
        print(f"    - Cat lang: {SU_DUNG_CAT_LANG}")
    
    plt.tight_layout()
    
    # Luu bieu do
    if thu_muc_luu:
        duong_dan = thu_muc_luu / 'mel_spectrogram_truoc_sau_tien_xu_ly.png'
        plt.savefig(str(duong_dan), dpi=300, bbox_inches='tight')
        print(f"\nDa luu: {duong_dan}")
    
    plt.close()

def chay_eda_day_du(du_lieu_file, thu_muc_audio='Audio Files', tan_so_mau=16000, duong_dan_luu=None, thong_ke_cat=None):
    # Chay toan bo phan tich EDA
    import json
    from pathlib import Path
    
    print("\n" + "="*60)
    print("BAT DAU PHAN TICH DU LIEU KHAM PHA (EDA)")
    print("="*60)
    
    # Neu khong co duong_dan_luu, su dung mac dinh
    if duong_dan_luu is None:
        duong_dan_luu = 'outputs/results/eda_results.json'
    
    # Tao thu muc neu chua co
    thu_muc_luu = Path(duong_dan_luu).parent
    thu_muc_luu.mkdir(parents=True, exist_ok=True)
    
    # 1. Tong quan dataset
    df = tong_quan_dataset(thu_muc_audio)
    
    # 2. Phan tich phan phoi chan doan
    phan_tich_phan_phoi_chan_doan(df, thu_muc_luu)
    
    # 3. Phan tich phan phoi nhan
    label_counts = phan_tich_phan_phoi_label(df, thu_muc_luu)
    
    # 4. Phan tich tuoi va gioi tinh
    phan_tich_tuoi_gioi_tinh(df, thu_muc_luu)
    
    # 5. Hien thi mau
    hien_thi_mau_audio(du_lieu_file, so_mau=4, tan_so_mau=tan_so_mau, thu_muc_luu=thu_muc_luu)
    
    # 6. Ve Mel Spectrogram truoc/sau tien xu ly
    ve_mel_spectrogram_truoc_sau_tien_xu_ly(du_lieu_file, tan_so_mau=tan_so_mau, so_mau=4, thu_muc_luu=thu_muc_luu)
    
    # 7. Luu ket qua EDA ra JSON
    print("\nDang luu ket qua EDA...")
    eda_results = {
        'tong_so_file': len(du_lieu_file),
        'label_counts': {
            'Binh thuong': int(label_counts.get('Binh thuong', 0)),
            'Bat thuong': int(label_counts.get('Bat thuong', 0))
        },
        'thong_ke_tuoi': {
            'trung_binh': float(df['tuoi'].mean()),
            'median': float(df['tuoi'].median()),
            'min': int(df['tuoi'].min()),
            'max': int(df['tuoi'].max()),
            'std': float(df['tuoi'].std())
        },
        'thong_ke_gioi_tinh': df['gioi_tinh'].value_counts().to_dict()
    }
    
    # Them thong ke cat file neu co
    if thong_ke_cat is not None:
        eda_results['thong_ke_cat_file'] = {
            'tong_so_file_goc': thong_ke_cat['tong_so_file_goc'],
            'tong_so_doan_cat': thong_ke_cat['tong_so_doan_cat'],
            'trung_binh_doan_tren_file': float(thong_ke_cat['tong_so_doan_cat'] / max(1, thong_ke_cat['tong_so_file_goc']))
        }
    
    # Luu JSON
    with open(duong_dan_luu, 'w') as f:
        json.dump(eda_results, f, indent=2)
    
    print(f"Da luu ket qua EDA: {duong_dan_luu}")
    
    print("\n" + "="*60)
    print("HOAN THANH PHAN TICH EDA")
    print("="*60)
    print(f"\nTat ca bieu do da duoc luu vao: {thu_muc_luu}")
    
    return label_counts

if __name__ == "__main__":
    print('EDA functions ready!')
