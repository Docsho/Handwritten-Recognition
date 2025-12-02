# EMNIST Handwriting Recognition System

A neural network-based handwriting recognition system built from scratch (without TensorFlow or PyTorch) for recognizing handwritten letters and digits using the EMNIST dataset.

## Features

- **62 character classes**: Digits (0-9), uppercase letters (A-Z), and lowercase letters (a-z)
- **Deep neural network**: 4-layer architecture (784→512→256→128→62)
- **Data augmentation**: Random rotations and shifts to improve accuracy
- **Web interface**: Upload images or draw directly in browser
- **High accuracy**: 85-90% test accuracy without advanced libraries

## Project Structure

```
project/
├── handwriting_recognition.py  # Core neural network implementation
├── train_model.py             # Training script
├── test_prediction.py         # Testing script for single images
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html            # Web interface (see separate artifact)
├── emnist-byclass.mat        # Dataset (download separately)
└── emnist_model.pkl          # Trained model (generated after training)
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download EMNIST Dataset

Download the EMNIST ByClass dataset from:
https://www.nist.gov/itl/products-and-services/emnist-dataset

Look for the MATLAB format file: **emnist-byclass.mat** (~700MB)

Place it in the project root directory.

## Usage

### Training the Model

Train the model on the EMNIST dataset (takes 30-60 minutes):

```bash
python train_model.py
```

This will:
- Load and preprocess the EMNIST dataset
- Apply data augmentation
- Train a 4-layer neural network for 50 epochs
- Save the trained model to `emnist_model.pkl`

Training output example:
```
Epoch 5/50 - Train Acc: 0.7521, Val Acc: 0.7234
Epoch 10/50 - Train Acc: 0.8234, Val Acc: 0.8012
...
Final Test Accuracy: 0.8756
```

### Running the Web Application

Start the Flask web server:

```bash
python app.py
```

Open your browser to: **http://localhost:5000**

The web interface provides two modes:
1. **Upload Image**: Upload a 28x28 PNG or BMP image
2. **Draw**: Draw a character directly in the browser

### Testing on Individual Images

Test predictions on a single image:

```bash
python test_prediction.py test_image.png
```

This will:
- Load the trained model
- Preprocess the image (resize, normalize, invert if needed)
- Display the prediction with confidence scores
- Show top 5 predictions

## Neural Network Architecture

### Network Structure
```
Input Layer:     784 neurons (28x28 pixels)
Hidden Layer 1:  512 neurons (ReLU activation)
Hidden Layer 2:  256 neurons (ReLU activation)
Hidden Layer 3:  128 neurons (ReLU activation)
Output Layer:    62 neurons (Softmax activation)
```

### Key Features
- **He Initialization**: Prevents vanishing gradients
- **ReLU Activation**: Faster training than sigmoid/tanh
- **Mini-batch Gradient Descent**: Batch size of 128
- **Softmax Output**: Multi-class classification
- **Data Augmentation**: Improves generalization

## Accuracy Metrics

Expected performance:
- **Training Accuracy**: 92-95%
- **Validation Accuracy**: 88-92%
- **Test Accuracy**: 85-90%

Confusion commonly occurs between similar characters:
- O (letter) vs 0 (zero)
- I (letter) vs l (lowercase L) vs 1 (one)
- S vs 5
- Z vs 2

## API Endpoints

### POST /predict
Upload an image file for prediction.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (PNG/BMP image)

**Response:**
```json
{
  "success": true,
  "character": "A",
  "confidence": "94.23%",
  "top5": [
    ["A", 0.9423],
    ["R", 0.0321],
    ["H", 0.0156],
    ...
  ],
  "image": "base64_encoded_image"
}
```

### POST /predict_drawing
Predict from canvas drawing.

**Request:**
- Method: POST
- Content-Type: application/json
- Body: `{"image": "data:image/png;base64,..."}`

**Response:**
```json
{
  "success": true,
  "character": "5",
  "confidence": "89.12%",
  "top5": [
    ["5", "89.12%"],
    ["S", "5.43%"],
    ...
  ]
}
```

## Tips for Better Accuracy

### For Uploaded Images:
1. Use 28x28 pixel images for best results
2. Ensure character is centered
3. Use black text on white background (or vice versa)
4. Make sure character fills most of the image
5. Avoid excessive noise or artifacts

### For Drawing:
1. Draw clearly and centered
2. Use bold strokes
3. Draw larger characters (they scale better)
4. Stay within the canvas bounds

## Improving the Model

To achieve higher accuracy, you can:

1. **Train longer**: Increase epochs to 100+
2. **Add dropout**: Prevent overfitting
3. **Implement momentum**: Use Adam optimizer
4. **Batch normalization**: Stabilize training
5. **Learning rate decay**: Fine-tune in later epochs
6. **More data augmentation**: Add noise, elastic deformations
7. **Ensemble methods**: Combine multiple models

Example modifications in `handwriting_recognition.py`:

```python
# Add dropout (requires implementation)
self.dropout_rate = 0.5

# Implement learning rate decay
if epoch % 10 == 0:
    self.learning_rate *= 0.9

# Add L2 regularization
lambda_reg = 0.001
reg_term = lambda_reg * sum(np.sum(w**2) for w in self.weights)
```

## Technical Details

### Data Preprocessing
1. Images are loaded from MATLAB format
2. Transposed and flipped to correct orientation
3. Normalized to [0, 1] range
4. Flattened to 784-dimensional vectors
5. Labels one-hot encoded to 62-dimensional vectors

### Data Augmentation
- Random rotations: ±10 degrees
- Random shifts: ±2 pixels
- Applied to 50,000 training samples
- Doubles/triples effective dataset size

### Training Process
1. Shuffle training data each epoch
2. Split into mini-batches (128 samples)
3. Forward propagation through network
4. Calculate cross-entropy loss
5. Backward propagation to compute gradients
6. Update weights using gradient descent
7. Validate every 5 epochs

## Troubleshooting

### Model file not found
```
Error: Model file 'emnist_model.pkl' not found!
```
**Solution**: Run `python train_model.py` first to train the model.

### Dataset file not found
```
Error: Dataset file 'emnist-byclass.mat' not found!
```
**Solution**: Download the EMNIST dataset from the NIST website.

### Low accuracy on custom images
- Ensure image is 28x28 pixels or will be resized
- Check that character is centered
- Verify correct orientation (not upside down)
- Make sure there's sufficient contrast

### Out of memory during training
- Reduce batch size in `train_model.py`
- Reduce data augmentation samples
- Use a smaller network architecture

## License

This project is for educational purposes. The EMNIST dataset is provided by NIST and has its own terms of use.

## References

- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.

## Authors

Created for AI class project - Handwriting recognition without advanced libraries.