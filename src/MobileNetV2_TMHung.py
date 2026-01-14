import os
import sys
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# import config module
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BATCH_SIZE, EPOCHS, SEED

def thiet_lap_seed_lap_lai(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass
    
    print(f'SEED = {seed} (Reproducible mode)')

def chuyen_1_kenh_sang_3_kenh(X):
    print(f'Chuyển đổi từ shape {X.shape}', end=' ')
    
    # Lặp lại kênh đơn thành 3 kênh
    X_3_kenh = np.repeat(X, 3, axis=-1)
    
    print(f'sang shape {X_3_kenh.shape}')
    return X_3_kenh

def resize_de_phu_hop_mobilenet(X, kich_thuoc_muc_tieu=(128, 128)):
    if X.shape[1:3] == kich_thuoc_muc_tieu:
        return X
    
    print(f'Resize từ {X.shape[1:3]} sang {kich_thuoc_muc_tieu}...')
    
    # Sử dụng TensorFlow resize
    X_resized = tf.image.resize(X, kich_thuoc_muc_tieu, method='bilinear')
    X_resized = X_resized.numpy()
    
    return X_resized

def tao_mo_hinh_mobilenetv2(kich_thuoc_input, freeze_base=True, learning_rate=1e-4):
    print('\nTẠO MÔ HÌNH MOBILENETV2')
    print('='*50)
    
    # Load MobileNetV2 pretrained trên ImageNet
    base_model = MobileNetV2(
        input_shape=kich_thuoc_input,
        include_top=False,  # Không lấy classification head
        weights='imagenet'
    )
    
    # Freeze base model nếu cần (transfer learning phase 1)
    if freeze_base:
        base_model.trainable = False
        print('Base model: FROZEN (transfer learning phase 1)')
    else:
        # Fine-tune: unfreeze một số layers cuối
        base_model.trainable = True
        # Freeze các layer đầu (giữ low-level features)
        for layer in base_model.layers[:-30]:  # Freeze all except last 30 layers
            layer.trainable = False
        print('Base model: PARTIALLY UNFROZEN (fine-tuning phase 2)')
    
    # xây dựng model hoàn chỉnh
    model = models.Sequential([
        # Preprocessing (chuyển về scale [-1, 1])
        layers.Lambda(lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(x)),
        
        # Base model (MobileNetV2)
        base_model,
        
        # Custom classification head - Improved
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Build model trước khi đếm parameters
    model.build((None, *kich_thuoc_input))
    
    print(f'\nTổng số parameters: {model.count_params():,}')
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f'Trainable parameters: {trainable_params:,}')
    
    return model

def huan_luyen_mo_hinh_mobilenet(X_train, y_train, X_val, y_val, 
                                 duong_dan_luu='project_N9/models/lung_model_mobilenet.keras',
                                 two_phase=True):
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Bước 1: Chuyển 1 kênh sang 3 kênh
    X_train_3ch = chuyen_1_kenh_sang_3_kenh(X_train)
    X_val_3ch = chuyen_1_kenh_sang_3_kenh(X_val)
    
    # Bước 2: Resize nếu cần (MobileNet tốt nhất >= 96x96)
    kich_thuoc_muc_tieu = (128, 128)  # Balance giữa quality và speed
    if X_train_3ch.shape[1:3] != kich_thuoc_muc_tieu:
        X_train_3ch = resize_de_phu_hop_mobilenet(X_train_3ch, kich_thuoc_muc_tieu)
        X_val_3ch = resize_de_phu_hop_mobilenet(X_val_3ch, kich_thuoc_muc_tieu)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
    ]
    
    if two_phase:
        print('\n' + '='*60)
        print('PHASE 1: TRAIN CLASSIFICATION HEAD (BASE FROZEN)')
        print('='*60)
        
        # Phase 1: Freeze base, train head
        model = tao_mo_hinh_mobilenetv2(
            kich_thuoc_input=(*kich_thuoc_muc_tieu, 3),
            freeze_base=True,
            learning_rate=1e-3  # Learning rate cao hơn cho phase 1
        )
        
        history1 = model.fit(
            X_train_3ch, y_train,
            validation_data=(X_val_3ch, y_val),
            epochs=20,  # Ít epochs cho phase 1
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        print('\n' + '='*60)
        print('PHASE 2: FINE-TUNE (UNFREEZE BASE)')
        print('='*60)
        
        # Phase 2: Unfreeze và fine-tune
        # Tạo lại model với base unfrozen
        for layer in model.layers:
            if hasattr(layer, 'layers'):  # Base model
                layer.trainable = True
                for sublayer in layer.layers[:-30]:
                    sublayer.trainable = False
        
        # Compile lại với learning rate thấp hơn
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR thấp cho fine-tune
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            X_train_3ch, y_train,
            validation_data=(X_val_3ch, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Gộp 2 histories
        history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
        }
        
        # Convert sang object giống Keras History
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        history = CombinedHistory(history)
        
    else:
        # Single phase training - CHỈ TRAIN CLASSIFICATION HEAD
        print('\nSINGLE PHASE TRAINING - FREEZE BASE')
        print('='*60)
        
        model = tao_mo_hinh_mobilenetv2(
            kich_thuoc_input=(*kich_thuoc_muc_tieu, 3),
            freeze_base=True,
            learning_rate=1e-3
        )
        
        history = model.fit(
            X_train_3ch, y_train,
            validation_data=(X_val_3ch, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
    
    # lưu model
    model.save(duong_dan_luu)
    print(f'\nĐã lưu model tại: {duong_dan_luu}')
    
    return model, history

if __name__ == "__main__":
    print('MobileNetV2 model ready!')
    print('Sử dụng hàm huấn_luyện_mô_hình_mobilenet() để train')
