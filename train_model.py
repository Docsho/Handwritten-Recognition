"""
train_model.py
Script to train the EMNIST handwriting recognition model
"""

from handwriting_recognition import HandwritingRecognizer
import sys
import os

def main():
    """Main training function"""
    
    # Check if dataset file exists
    dataset_path = 'emnist-byclass.mat'
    model_save_path = 'emnist_model.pkl'
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found!")
        print("\nPlease download the EMNIST ByClass dataset from:")
        print("https://www.nist.gov/itl/products-and-services/emnist-dataset")
        print("\nLook for 'emnist-byclass.mat' in the MATLAB format files.")
        sys.exit(1)
    
    print("="*60)
    print("EMNIST Handwriting Recognition - Training Script")
    print("="*60)
    print()
    
    # Initialize recognizer
    recognizer = HandwritingRecognizer()
    
    # Start training
    print(f"Dataset: {dataset_path}")
    print(f"Model will be saved to: {model_save_path}")
    print()
    print("Training will take approximately 30-60 minutes depending on your hardware.")
    print("The model will be trained on 62 classes (0-9, A-Z, a-z)")
    print()
    
    response = input("Do you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    print("\n" + "="*60)
    print("Starting training process...")
    print("="*60 + "\n")
    
    try:
        recognizer.train(dataset_path, model_save_path)
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        print(f"\nModel saved to: {model_save_path}")
        print("\nYou can now run the web application using:")
        print("  python app.py")
        print()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()