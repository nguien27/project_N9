import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def tao_mo_hinh_efficientnet_v2(kich_thuoc_input, freeze_base=True, learning_rate=1e-4):
    """
    Tao mo hinh EfficientNet-B0 voi 2 che do: Freeze hoac Fine-tune
    """
    # 1. Base model EfficientNetB0
    # Input truyen vao day phai la 3 kenh
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    if freeze_base:
        base_model.trainable = False
        print("Base model: FROZEN (Giai doan 1)")
    else:
        base_model.trainable = True
        # Giu lai cac feature co ban (layers dau), chi mo khoa 20 layers cuoi de fine-tune
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        print("Base model: PARTIALLY UNFROZEN (Giai doan 2)")

    # 2. Xay dung Classification Head
    model = models.Sequential([
        # Nhận đầu vào là Mel Spectrogram (1 kênh)
        layers.Input(shape=(kich_thuoc_input[0], kich_thuoc_input[1], 1)), 
        
        # Chuyển đổi từ 1 kênh sang 3 kênh (RGB)
        layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1)),
        
        # Resize về 224x224 ĐỂ KHỚP VỚI base_model ở trên
        layers.Resizing(224, 224),
        
        base_model,
        
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # Label Smoothing giup model ben bi hon voi nhieu am thanh
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def huan_luyen_efficientnet(X_train, y_train, X_val, y_val, duong_dan_luu):
    """
    Quy trinh huan luyen 2 giai doan giong file tham khao MobileNet
    """
    input_dim = (X_train.shape[1], X_train.shape[2], 3)
    
    # Callbacks chung
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]

    # --- PHASE 1: TRAIN TOP LAYERS ---
    print("\n>>> PHASE 1: HUAN LUYEN PHAN DAU (FREEZE BASE)")
    model = tao_mo_hinh_efficientnet_v2(input_dim, freeze_base=True, learning_rate=1e-3)
    
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15, # Giai doan 1 chay nhanh
        batch_size=16,
        callbacks=callbacks
    )

    # --- PHASE 2: FINE-TUNING ---
    print("\n>>> PHASE 2: FINE-TUNING (UNFREEZE PARTIAL BASE)")
    # Lay model hien tai va mo khoa layers
    model.get_layer(index=1).trainable = True # base_model nam o index 1
    # Cap nhat lai learning rate cuc thap de fine-tune
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )

    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50, 
        batch_size=16,
        callbacks=callbacks
    )

    model.save(duong_dan_luu)
    print(f"Da luu model tai: {duong_dan_luu}")
    return model, history2

if __name__ == "__main__":
    print('EfficientNet2 model ready!')
    print('Sử dụng hàm huan_luyen_efficientnet để train')