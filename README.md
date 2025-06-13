# Handwritten Digit Recognition using CNN (MNIST)

## Project Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model learns to recognize digits (0–9) from grayscale images and achieves high accuracy on the test set.

---

## Tools & Libraries
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- seaborn (for visualization)

---

## Dataset
- **MNIST dataset**: 70,000 grayscale images of handwritten digits (28x28 pixels).
- 60,000 images for training and 10,000 images for testing.
- Dataset is available directly through Keras datasets API.

---

## Project Roadmap

### 1. Data Loading & Preprocessing
- Load MNIST dataset from Keras.
- Normalize pixel values to [0, 1].
- Reshape data to fit CNN input shape `(28, 28, 1)`.

### 2. Exploratory Data Analysis (EDA)
- Visualize sample digits.
- Analyze dataset distribution.

### 3. Model Building (CNN)
- Define Sequential CNN model.
- Add convolutional, pooling, dropout, and dense layers.
- Use `ReLU` activations and `softmax` for output.

### 4. Model Compilation
- Compile model with `Adam` optimizer.
- Use `sparse_categorical_crossentropy` loss.
- Track accuracy metric.

### 5. Model Training
- Train model on training data with validation split.
- Visualize training and validation accuracy/loss.

### 6. Model Evaluation
- Evaluate model on test data.
- Generate classification report and confusion matrix.
- Visualize predictions vs true labels.

### 7. Model Saving & Loading
- Save the trained model to disk (`.h5` format).
- Load model for future inference or further training.
