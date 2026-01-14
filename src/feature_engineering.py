import numpy as np
import librosa
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def chuyen_sang_mel(tin_hieu, tan_so_mau=16000, n_fft=2048, 
                    hop_length=512, n_mels=128, tang_cuong=False, muc_do=0,
                    trung_binh=None, do_lech=None, thoi_luong=4):
    # Chuyen audio thanh Mel Spectrogram da chuan hoa 
    #   Mel spectrogram da chuan hoa voi shape (n_mels, time_steps, 1)
    so_mau_mong_muon = tan_so_mau * thoi_luong
    
    # 1. Cat hoac pad ve do dai chuan
    if len(tin_hieu) > so_mau_mong_muon:
        tin_hieu = tin_hieu[:so_mau_mong_muon]
    else:
        tin_hieu = np.pad(tin_hieu, (0, so_mau_mong_muon - len(tin_hieu)))
    
    # 2. Tang cuong du lieu 
    if tang_cuong:
        # Time shift
        shift = np.random.randint(-tan_so_mau//4, tan_so_mau//4)
        tin_hieu = np.roll(tin_hieu, shift)
        
        # Add noise (muc do khac nhau)
        noise_level = [0.002, 0.004, 0.006][muc_do]
        noise = np.random.randn(len(tin_hieu)) * noise_level
        tin_hieu = tin_hieu + noise
        
        # Pitch shift (chi ap dung muc_do >= 1)
        if muc_do >= 1:
            n_steps = np.random.uniform(-2, 2)
            tin_hieu = librosa.effects.pitch_shift(tin_hieu, sr=tan_so_mau, n_steps=n_steps)
        
        # Time stretch (chi ap dung muc_do == 2)
        if muc_do == 2:
            rate = np.random.uniform(0.9, 1.1)
            tin_hieu = librosa.effects.time_stretch(tin_hieu, rate=rate)
            # Pad/cut lai sau time stretch
            if len(tin_hieu) > so_mau_mong_muon:
                tin_hieu = tin_hieu[:so_mau_mong_muon]
            else:
                tin_hieu = np.pad(tin_hieu, (0, so_mau_mong_muon - len(tin_hieu)))
    
    # 3. Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=tin_hieu,
        sr=tan_so_mau,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # 4. Chuan hoa voi mean/std tu tap train
    if trung_binh is not None and do_lech is not None:
        mel_chuan = (mel_db - trung_binh) / (do_lech + 1e-6)
    else:
        mel_chuan = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    
    return mel_chuan[..., np.newaxis], mel_db



if __name__ == "__main__":
    print('Feature engineering functions ready!')
