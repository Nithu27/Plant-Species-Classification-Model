# Plant Identification using Machine Learning

This project is a plant identification system built using various machine learning algorithms to classify and identify plants from images. The system leverages both traditional and deep learning techniques to provide a comprehensive approach to plant recognition.

## Features

- **Convolutional Neural Network (CNN):** A deep learning model that excels at recognizing patterns in images, particularly effective for plant identification due to its ability to capture complex features and details.
  
- **K-Nearest Neighbors (KNN):** A simple, yet effective, algorithm that classifies plants based on the similarity to the nearest examples in the training dataset.
  
- **Random Forest:** An ensemble learning method that constructs multiple decision trees to improve accuracy and robustness in plant classification.
  
- **Multi-Layer Perceptron (MLP):** A type of artificial neural network that consists of multiple layers of nodes, used for nonlinear mapping between input images and output classifications.
  
- **Naive Bayes:** A probabilistic classifier based on applying Bayes' theorem, useful for handling datasets with a large number of features.

## How It Works

1. **Data Collection:** The system uses a dataset of plant images, pre-processed to ensure uniformity and quality.
  
2. **Training:** Each machine learning algorithm is trained on a subset of the data to learn distinguishing features of different plant species.
  
3. **Prediction:** Given a new image, the trained models predict the plant species by analyzing the image's features.
  
4. **Evaluation:** The models are evaluated based on accuracy, precision, recall, and F1-score to determine their effectiveness in real-world scenarios.

## Datasets

The project uses two main datasets: a **training dataset** and a **test dataset**.

- **Training Dataset:** This dataset consists of a large number of labeled plant images that the models use to learn. The images are categorized by plant species, and the dataset is divided into features and labels for supervised learning. The training process involves feeding these images into the machine learning models to adjust their parameters and improve accuracy.

- **Test Dataset:** After the models are trained, they are evaluated using the test dataset, which contains images that the models have never seen before. This dataset is used to assess the model's performance in terms of accuracy, precision, recall, and F1-score, ensuring that the model can generalize well to new, unseen data.

Both datasets should be properly formatted and pre-processed to ensure consistent input data, such as resizing images, normalizing pixel values, and augmenting the dataset to improve model robustness.

## Datasets 

- https://www.dropbox.com/scl/fi/wa8qsfreyigbx1xc79xmc/train.rar?rlkey=klciqss3adi3twvv6xibm3f2k&st=apltf9ki&dl=0
- https://www.dropbox.com/scl/fi/mphggjytbhekrfubu6h9r/test.zip?rlkey=7jmluxvbf4idr96ecadurp8qc&st=t00lvqhd&dl=0
