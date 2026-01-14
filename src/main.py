import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description='Phan loai am thanh phoi')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mobilenet'],
                       help='Model de chay: cnn hoac mobilenet (mac dinh: cnn)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'predict'],
                       help='Mode: train, eval, hoac predict (mac dinh: train)')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"PHAN LOAI AM THANH PHOI - {args.model.upper()}")
    print("="*60)
    
    if args.model == 'cnn':
        print("\nChay CNN model...")
        from CNN_main import main as cnn_main
        cnn_main()
    elif args.model == 'mobilenet':
        print("\nChay MobileNetV2 model...")
        from Mobi_main import main as mobi_main
        mobi_main()

if __name__ == "__main__":
    main()
