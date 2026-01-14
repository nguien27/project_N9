# Lung Sound Classification - Phân loại âm thanh phổi

Dự án phân loại âm thanh phổi (bình thường vs bất thường) sử dụng Deep Learning.

---

## 🔧 Cài đặt

### Yêu cầu
- Python 3.8+
- pip

### Bước 1: Tạo virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 2: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 3: Chuẩn bị dữ liệu
Đặt file audio (.wav) vào thư mục `data/Audio Files/`

---

##  Chạy project

### 1. Phân tích dữ liệu (EDA)
```bash
cd src
python eda.py
```
**Kết quả**: `outputs/results/eda_results.json`

### 2. Training CNN
```bash
cd src
python CNN_main.py
```
**Kết quả**:
- `models/lung_model_balanced.keras` - Mô hình
- `outputs/results/confusion_matrix.png` - Ma trận nhầm lẫn
- `outputs/results/training_history.png` - Lịch sử huấn luyện
- `outputs/results/evaluation_results.json` - Metrics

### 3. Training MobileNetV2
```bash
cd src
python Mobi_main.py
```
**Kết quả**: Tương tự CNN

### 4. Grad-CAM Visualization
```bash
cd src
python run_gradcam.py
```
**Mục đích**: Trực quan hóa vùng quan trọng trong Mel Spectrogram

---

##  Cấu hình

Chỉnh sửa `config.py` để thay đổi các tham số:

```python
# Audio config
TAN_SO_MAU = 16000          # Sample rate (Hz)
THOI_LUONG = 4              # Độ dài audio (giây)
N_FFT = 2048                # FFT size
DO_NHAY = 512               # Hop length
SO_MEL = 128                # Số mel bins

# Model config
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 5e-5
SEED = 42

# Data split
TY_LE_TRAIN = 0.8           # 80% train, 20% validation
```

---

##  Cấu trúc thư mục

```
project_N9/
├── README.md                    # Hướng dẫn cài đặt và chạy project
├── config.py                    # Cấu hình tổng thể
├── requirements.txt             # Danh sách thư viện cần cài
├── data/
│   └── Audio Files/             # Thư mục chứa dữ liệu audio
├── src/
│   ├── preprocessing.py         # Xử lý dữ liệu
│   ├── eda.py                   # Phân tích dữ liệu
│   ├── feature_engineering.py   # Tạo Mel Spectrogram
│   ├── evaluation.py            # Đánh giá mô hình
│   ├── CNN_main.py              # Training CNN
│   ├── CNN_NHNguyen.py          # Kiến trúc CNN
│   ├── Mobi_main.py             # Training MobileNetV2
│   ├── MobileNetV2_TMHung.py    # Kiến trúc MobileNetV2
│   ├── gradcam_feature_extraction.py  # Grad-CAM
│   └── run_gradcam.py           # Script chạy Grad-CAM
├── models/                      # Lưu mô hình đã train
└── outputs/
    └── results/                 # Kết quả (confusion matrix, history, JSON)
```

---

##  Troubleshooting

### ImportError: No module named 'librosa'
```bash
pip install librosa
```

### Out of Memory
- Giảm `BATCH_SIZE` trong config (từ 16 → 8)
- Giảm `EPOCHS` (từ 200 → 100)
- Sử dụng MobileNetV2 thay vì CNN

### Audio file not found
- Kiểm tra đường dẫn trong `config.py`
- Đảm bảo file audio có extension `.wav`

---

**Cập nhật lần cuối**: January 2026
