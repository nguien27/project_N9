import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
from scipy.ndimage import zoom

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_PATH, TAN_SO_MAU, N_FFT, DO_NHAY, SO_MEL
from preprocessing import tien_xu_ly_audio_tho
from feature_engineering import chuyen_sang_mel

class GradCAMFeatureExtractor:
    # Trich xuat dac trung bang Grad-CAM
    
    def __init__(self, model_path):
        # Khoi tao Grad-CAM
        self.model = tf.keras.models.load_model(str(model_path))
        
        # Tim layer Conv cuoi cung TRUOC pooling/flatten
        self.last_conv_layer_name = None
        self.last_conv_layer_index = -1
        
        for i, layer in enumerate(reversed(self.model.layers)):
            if 'conv2d' in layer.name.lower():
                self.last_conv_layer_name = layer.name
                self.last_conv_layer_index = len(self.model.layers) - 1 - i
                break
        
        if self.last_conv_layer_name is None:
            raise ValueError("Khong tim thay Conv2D layer trong model")
           
    def compute_gradcam(self, mel_spectrogram, pred_index=None):
        # Tinh Grad-CAM heatmap bang cach goi truc tiep model.predict
        img_array = np.expand_dims(mel_spectrogram, axis=0).astype(np.float32)
        img_tensor = tf.convert_to_tensor(img_array)
        
        # Tao model intermediate: input -> last conv layer output
        last_conv_layer = self.model.get_layer(self.last_conv_layer_name)
        
        # Chay qua toan bo model va lay output cua conv layer
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            
            conv_outputs = None
            x = img_tensor
            for layer in self.model.layers:
                x = layer(x)
                if layer.name == self.last_conv_layer_name:
                    conv_outputs = x
            
            predictions = x
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Lay gradient cua output theo conv_outputs
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            print("  [WARN] Gradients are None")
            return None, predictions[0].numpy()
        
        # Tinh Grad-CAM: weighted average cua feature maps
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        
        # Weighted sum
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        # ReLU de chi giu positive values
        heatmap = tf.nn.relu(heatmap)
        
        # Normalize: [0, 1]
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        
        return heatmap.numpy(), predictions[0].numpy()
    
    def visualize_3_layers(self, mel_db_goc, heatmap, file_name="", save_path=None):
        # Ve 3 tang: Tang 1 - Mel Spectrogram (giong EDA), Tang 2 - Grad-CAM, Tang 3 - Overlay
        
        mel_db = mel_db_goc
        
        # Resize heatmap de match voi mel spectrogram
        h_ratio = mel_db.shape[0] / heatmap.shape[0]
        w_ratio = mel_db.shape[1] / heatmap.shape[1]
        heatmap_resized = zoom(heatmap, (h_ratio, w_ratio), order=1)
        
        # Normalize heatmap_resized ve [0, 1]
        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
        
        # Tao figure voi 3 subplot (1 hang, 3 cot)
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # TANG 1: Mel Spectrogram (giong EDA - voi grid)
        im1 = axes[0].imshow(
            mel_db,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            vmin=-80, vmax=0
        )
        axes[0].set_title(f'Tang 1: Mel Spectrogram', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Mel bins', fontsize=10)
        axes[0].set_xlabel('Time frames', fontsize=10)
        axes[0].grid(False)
        cbar1 = plt.colorbar(im1, ax=axes[0], label='dB')
        
        # TANG 2: Grad-CAM Dac trung
        im2 = axes[1].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='hot')
        axes[1].set_title(f'Tang 2: Grad-CAM Dac trung', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Mel bins', fontsize=10)
        axes[1].set_xlabel('Time frames', fontsize=10)
        cbar2 = plt.colorbar(im2, ax=axes[1], label='Importance')
        
        # TANG 3: Overlay (Mel + Grad-CAM)
        im3_1 = axes[2].imshow(mel_db, aspect='auto', origin='lower', cmap='viridis')
        im3_2 = axes[2].imshow(heatmap_resized, aspect='auto', origin='lower', cmap='hot', alpha=0.6)
        axes[2].set_title(f'Tang 3: Overlay (Mel + Grad-CAM)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Mel bins', fontsize=10)
        axes[2].set_xlabel('Time frames', fontsize=10)
        cbar3 = plt.colorbar(im3_2, ax=axes[2], label='Grad-CAM Importance')
        
        # Tieu de chung
        fig.suptitle(f'Grad-CAM 3 Tang - {file_name}', fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
            print(f"[OK] Saved: {save_path}")
        
        plt.close()
    
    def extract_features_from_file(self, audio_file, output_dir=None):
        # Trich xuat dac trung tu file audio
        try:
            # Load va xu ly audio
            tin_hieu, _ = librosa.load(audio_file, sr=TAN_SO_MAU)
            tin_hieu = tien_xu_ly_audio_tho(tin_hieu, TAN_SO_MAU)
            
            # Tao Mel Spectrogram - nhan ca 2 gia tri
            mel_chuan, mel_db_goc = chuyen_sang_mel(tin_hieu, TAN_SO_MAU, N_FFT, DO_NHAY, SO_MEL, tang_cuong=False)
            
            # Tinh Grad-CAM voi mel_chuan
            print(f"  [PROCESSING] Computing Grad-CAM...")
            heatmap, prediction = self.compute_gradcam(mel_chuan)
            print(f"  [PROCESSING] Grad-CAM computed, drawing visualization...")
            
            if heatmap is None:
                return None, prediction, mel_db_goc
            
            # Ve va luu - dung mel_db_goc de ve
            if output_dir:
                file_name = Path(audio_file).stem
                save_path = Path(output_dir) / f"gradcam_{file_name}.png"
                self.visualize_3_layers(mel_db_goc, heatmap, file_name, save_path)
            
            return heatmap, prediction, mel_db_goc
        
        except Exception as e:
            print(f"  [ERROR] Error in extract_features_from_file: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def extract_features_batch(self, audio_dir, output_dir, num_samples=10):
        # Trich xuat dac trung tu nhieu file audio
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Lay danh sach file
        audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        
        if not audio_files:
            print(f"[ERROR] No audio files found in {audio_dir}")
            return []
        
        # Lay mau ngau nhien
        np.random.seed(42)
        selected_files = np.random.choice(audio_files, min(num_samples, len(audio_files)), replace=False)
        
        print(f"\n{'='*60}")
        print(f"GRAD-CAM FEATURE EXTRACTION - 3 TANG")
        print(f"{'='*60}")
        print(f"Processing {len(selected_files)} audio files...")
        print(f"Audio directory: {audio_dir}")
        print(f"Output directory: {output_dir}")
        
        results = []
        success_count = 0
        
        for i, file_name in enumerate(selected_files, 1):
            try:
                file_path = os.path.join(audio_dir, file_name)
                print(f"\n[{i}/{len(selected_files)}] {file_name}")
                
                heatmap, prediction, mel = self.extract_features_from_file(file_path, output_dir)
                
                if heatmap is None:
                    print(f"  [ERROR] Failed to compute Grad-CAM")
                    continue
                
                # Luu thong tin
                results.append({
                    'file': file_name,
                    'prediction': float(prediction[0]),
                    'heatmap_shape': heatmap.shape,
                    'heatmap_max': float(np.max(heatmap)),
                    'heatmap_mean': float(np.mean(heatmap))
                })
                
                print(f"  [OK] Prediction: {prediction[0]:.4f}")
                print(f"  [OK] Heatmap max: {np.max(heatmap):.4f}, mean: {np.mean(heatmap):.4f}")
                success_count += 1
                
            except Exception as e:
                print(f"  [ERROR] {str(e)[:100]}")
        
        print(f"\n{'='*60}")
        print(f"COMPLETED - Successfully processed {success_count}/{len(selected_files)} files")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
        
        return results

def main():
    # Main function
    import argparse
    
    parser = argparse.ArgumentParser(description='Grad-CAM Feature Extraction')
    parser.add_argument('--model', type=str, default=str(MODEL_PATH), help='Model path')
    parser.add_argument('--audio-dir', type=str, default='project_N9/data/Audio Files', help='Audio directory')
    parser.add_argument('--output-dir', type=str, default='project_N9/outputs/gradcam', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples')
    
    args = parser.parse_args()
    
    # Kiem tra model
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        return
    
    # Kiem tra audio directory
    if not os.path.exists(args.audio_dir):
        print(f"[ERROR] Audio directory not found: {args.audio_dir}")
        return
    
    # Tao Grad-CAM extractor
    extractor = GradCAMFeatureExtractor(args.model)
    
    # Trich xuat dac trung
    results = extractor.extract_features_batch(args.audio_dir, args.output_dir, args.num_samples)

if __name__ == "__main__":
    main()
