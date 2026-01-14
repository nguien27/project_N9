import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report
)
import tensorflow as tf
import json
from pathlib import Path

def danh_gia_mo_hinh(model, X_test, y_test, nguong=0.5):
    # Danh gia mo hinh tren tap test - Chi luu metrics quan trong
    print("DANH GIA MO HINH")
    print("="*30)
    
    # Du doan
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > nguong).astype(int)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    report = classification_report(
        y_test, y_pred, 
        target_names=['Binh thuong (0)', 'Bat thuong (1)'],
        output_dict=True
    )
    print(classification_report(
        y_test, y_pred, 
        target_names=['Binh thuong (0)', 'Bat thuong (1)']
    ))
    
    # Tinh cac metrics - Su dung sklearn
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    print(f"\nTONG KET METRICS:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return results

def ve_confusion_matrix(cm, tieu_de="Confusion Matrix", save_path=None):
    # Ve confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Binh thuong', 'Bat thuong'], 
                yticklabels=['Binh thuong', 'Bat thuong'])
    plt.title(tieu_de)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Da luu: {save_path}")
    plt.show()

def ve_lich_su_huan_luyen(history, save_path=None):
    # Ve bieu do qua trinh training
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Da luu: {save_path}")
    plt.show()

def luu_ket_qua_json(results, duong_dan_luu='outputs/results/evaluation_results.json'):
    # Luu ket qua danh gia ra file JSON
    Path(duong_dan_luu).parent.mkdir(parents=True, exist_ok=True)
    
    with open(duong_dan_luu, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Da luu ket qua: {duong_dan_luu}")

if __name__ == "__main__":
    print('Evaluation functions ready!')
