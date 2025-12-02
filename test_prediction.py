"""
test_prediction.py
Script to test predictions on individual images
"""

from handwriting_recognition import HandwritingRecognizer
from PIL import Image
import numpy as np
import sys
import os

def test_image(image_path, model_path='emnist_model.pkl'):
    """
    Test prediction on a single image
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
    """
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using: python train_model.py")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    recognizer = HandwritingRecognizer(model_path)
    print("Model loaded successfully!\n")
    
    # Load and preprocess image
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Show original size
    print(f"Original image size: {image.size}")
    
    # Resize to 28x28 if needed
    if image.size != (28, 28):
        print("Resizing to 28x28...")
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Check if inversion is needed
    mean_val = np.mean(img_array)
    print(f"Image mean pixel value: {mean_val:.2f}")
    
    if mean_val > 127:
        print("Inverting image (detected light background)...")
        img_array = 255 - img_array
    
    # Save preprocessed image for inspection
    preprocessed = Image.fromarray(img_array)
    preprocessed_path = 'preprocessed_' + os.path.basename(image_path)
    preprocessed.save(preprocessed_path)
    print(f"Preprocessed image saved to: {preprocessed_path}\n")
    
    # Make prediction
    print("Making prediction...")
    result = recognizer.predict(img_array)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nPredicted Character: {result['character']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nTop 5 Predictions:")
    print("-" * 40)
    for i, (char, conf) in enumerate(result['top5'], 1):
        print(f"{i}. '{char}' - {conf:.2%}")
    print("="*60 + "\n")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_prediction.py <image_path> [model_path]")
        print("\nExample:")
        print("  python test_prediction.py test_image.png")
        print("  python test_prediction.py test_image.png emnist_model.pkl")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'emnist_model.pkl'
    
    test_image(image_path, model_path)

if __name__ == "__main__":
    main()