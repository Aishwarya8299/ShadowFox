Cat vs Dog Image Classification using CNN
📌 Project Overview
This project is a simple image classification system developed using a Convolutional Neural Network (CNN). The model is trained to classify images into two categories: Cat and Dog.

This project was completed as part of the ShadowFox Virtual Internship Program.

🎯 Objective
The main objective of this project is to build a practical image classification model using Deep Learning and CNN architecture. The model learns visual features from images and predicts whether an input image belongs to the Cat or Dog class.

📂 Dataset
The dataset contains a total of 1,000 images:

🐱 Cat Images: 500
🐶 Dog Images: 500
The dataset was divided into:

80% Training Data – 800 images
20% Validation Data – 200 images
🛠️ Technologies Used
Python
TensorFlow
Keras
Convolutional Neural Network (CNN)
NumPy
Matplotlib
Kaggle Notebook
🧠 CNN Architecture
The model consists of the following layers:

Input Layer
Convolutional Layer – 32 filters
Max Pooling Layer
Convolutional Layer – 64 filters
Max Pooling Layer
Convolutional Layer – 128 filters
Max Pooling Layer
Flatten Layer
Dense Layer – 128 neurons
Dropout Layer
Output Layer – Sigmoid Activation
⚙️ Data Preprocessing
The images were resized to 150 × 150 pixels and pixel values were normalized to a range between 0 and 1.

The dataset was loaded using TensorFlow's image_dataset_from_directory() method.

🚀 Model Training
The CNN model was trained using:

Optimizer: Adam
Loss Function: Binary Crossentropy
Evaluation Metric: Accuracy
Number of Epochs: 15
Batch Size: 32
📊 Results
The model achieved the following results:

Final Training Accuracy: 99.25%
Final Validation Accuracy: 97.50%
Best Validation Accuracy: 99.00%
The results demonstrate that the CNN model can effectively distinguish between cat and dog images.

📁 Project Files
dogs_vs_cats.ipynb – Contains the complete implementation, model training, and evaluation.
cats_dogs_cnn.keras – Saved trained CNN model (if available).
▶️ How to Run
Open the dogs_vs_cats.ipynb notebook.
Upload or connect the Cat vs Dog dataset.
Install the required Python libraries if necessary.
Run the notebook cells sequentially.
Train the CNN model.
Evaluate the model using the validation dataset.
📌 Conclusion
This project demonstrates the use of a Convolutional Neural Network for binary image classification. Through this project, I gained practical experience in data preprocessing, CNN architecture design, model training, validation, and performance evaluation using TensorFlow and Keras.
