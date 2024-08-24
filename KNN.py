import os
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random

# Set the path to your dataset
train_data_dir = '/content/train'
test_data_dir = '/content/reduced'

# Parameters
image_size = (128, 128)
k_neighbors = 3  # Specify the number of neighbors (k)
noise_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.43,0.37,0.29,0.7]  # Adjust these noise rates

# Load and preprocess the training data
def load_images_from_directory(directory):
    images = []
    labels = []
    label_mapping = {}

    for label, class_name in enumerate(os.listdir(directory)):
        label_mapping[label] = class_name
        class_directory = os.path.join(directory, class_name)

        for filename in os.listdir(class_directory):
            if filename.endswith(".jpg"):
                img = tf.keras.preprocessing.image.load_img(
                    os.path.join(class_directory, filename),
                    target_size=image_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(label)

    return np.array(images), np.array(labels), label_mapping

X_test, y_test, _ = load_images_from_directory(test_data_dir)

# Reshape the test data
X_test = X_test.reshape(X_test.shape[0], -1)

# Loop through different noise rates
for noise_rate in noise_rates:
    X_train, y_train, label_mapping = load_images_from_directory(train_data_dir)

    # Introduce label noise
    num_noisy_labels = int(len(y_train) * noise_rate)
    noisy_indices = random.sample(range(len(y_train)), num_noisy_labels)
    for idx in noisy_indices:
        available_labels = [i for i in range(len(label_mapping)) if i != y_train[idx]]
        y_train[idx] = random.choice(available_labels)

    # Reshape the training data
    X_train = X_train.reshape(X_train.shape[0], -1)

    # Initialize and train the KNN model
    knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)

    # Fit the model
    knn_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Noise Rate: {noise_rate:.2f} - Test Accuracy: {accuracy * 100:.2f}%")
