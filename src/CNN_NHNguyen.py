# Huan luyen mo hinh CNN cho phan loai am thanh phoi

import os
import sys
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED

def thiet_lap_seed_lap_lai(seed=42):
    # Thiet lap seed cho reproducibility
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

def tao_mo_hinh_cnn(kich_thuoc_input):
    # Tao mo hinh CNN cho phan loai am thanh phoi 
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), padding='same', activation='relu', 
                     kernel_regularizer=regularizers.l2(1e-4), 
                     input_shape=kich_thuoc_input),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.15),
        
        # Block 2
        layers.Conv2D(64, (3,3), padding='same', activation='relu', 
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.15),
        
        # Block 3
        layers.Conv2D(128, (3,3), padding='same', activation='relu', 
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile voi BCE + Label Smoothing
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    return model

def huan_luyen_mo_hinh(X_train, y_train, X_val, y_val, duong_dan_luu='project_N9/models/lung_model_balanced.keras'):
    # Huan luyen mo hinh
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Tao mo hinh
    model = tao_mo_hinh_cnn(X_train.shape[1:])
    model.summary()
    
    # Callbacks 
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
    ]
    
    # Huan luyen 
    print("\nBAT DAU HUAN LUYEN MO HINH")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Luu mo hinh
    model.save(duong_dan_luu)
    print(f'Da luu mo hinh: {duong_dan_luu}')
    
    return model, history

if __name__ == "__main__":
    print('CNN_NHNguyen functions ready!')
