# Efficient-Classification-of-Rice-Varieties-Using-Optimized-Convolutional-Neural-Networks


# Project Overview

This project presents a deep learning-based approach using an optimized Convolutional Neural Network (CNN) model implemented with PyTorch for classifying five rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag. The model achieves high accuracy while maintaining computational efficiency, making it a viable solution for agricultural automation.


# Features

1. High Accuracy: Achieved 99.81% validation accuracy with minimal validation loss (0.01%).

2. Optimized CNN Model: Unlike conventional models like MobileNetV2, VGG16, and EfficientNetB0, this approach significantly reduces computational complexity.

3. Lightweight & Efficient: Model size is only 30.76 MB, making it suitable for real-world applications.

4. Non-Destructive Evaluation: Uses image classification techniques to replace manual, time-consuming quality assessments.

5. Implementation with PyTorch: Ensures flexible model training and evaluation.

# Dataset

The dataset used consists of 75,000 images categorized into five rice varieties:

Training Set: 52,500 images

Validation & Test Set: 22,500 images

# Methodology

1. Data Preprocessing: Image resizing, normalization, and augmentation.

2. Model Architecture:

Convolutional Layers for feature extraction

Batch Normalization & Max Pooling for better generalization

Fully Connected Layers for final classification

Softmax Activation for multi-class classification

3. Training & Evaluation:

i. Optimizer: Adam (Adaptive Moment Estimation)

ii. Loss Function: Cross-Entropy Loss

iii. Epochs: 100

iv. Batch Size: 32

4. Performance Evaluation

The model was assessed using key metrics:

i. Accuracy: 99.81%

ii. Precision, Recall, F1-Score: High across all five classes

iii. Confusion Matrix: Demonstrated strong classification performance

# Hardware & Software Requirements

Hardware:

i. Processor: Intel Core i7-4790M (3.60GHz)

ii. RAM: 8GB

iii. GPU (Optional): Recommended for faster training

Software:

i. Python 3.9

ii. PyTorch

iii. OpenCV

iv. NumPy & Matplotlib

# Installation & Setup

i. Clone the repository:

git clone https://github.com/your-repo/rice-classification.git
cd rice-classification

ii. Install dependencies:

pip install -r requirements.txt

iii. Run the training script:

python train.py

iv. Perform classification on new images:

python classify.py --image path_to_image.jpg

# Applications

  i. Agriculture & Farming: Automated rice quality assessment.
  
  ii. Food Industry: Ensuring standardization of rice varieties.
  
  iii. Supply Chain & Commerce: Streamlining rice sorting processes.

# Future Enhancements

  i. Deployment on mobile and IoT devices for real-time classification.
  
  ii. Integration with cloud-based services for large-scale analysis.
  
  iii. Expansion to classify more rice varieties with extended datasets.

<!-- Contributors

Abhishek Kumar Pathak (IIT Indore)

Ankit Kumar Singh (Motihari College of Engineering)

Vimal Bhatia (IIT Indore)

Puneet Singh (IIT Indore) -->

# Acknowledgments

This research was supported by the Science and Engineering Research Board (SERB), Department of Science and Technology, Government of India (CRG/2021/001215).

# License

This project is licensed under the MIT License. See the LICENSE file for details.

