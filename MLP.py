import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
import random

# Set the path to your dataset
train_data_dir = '/content/train'
test_data_dir = '/content/reduced'

# Parameters
image_size = (128, 128)
noise_rates = [0.0, 0.42, 0.5, 0.2, 0.37, 0.62, 0.3, 0.27, 0.4, 0.55]  # Adjust these noise rates
batch_size = 32
epochs = 25

# Load and preprocess the training and test data
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
X_test = X_test / 255.0  # Normalize pixel values to [0, 1]

# Loop through different noise rates
for noise_rate in noise_rates:
    X_train, y_train, label_mapping = load_images_from_directory(train_data_dir)

    # Introduce label noise
    num_noisy_labels = int(len(y_train) * noise_rate)
    noisy_indices = random.sample(range(len(y_train)), num_noisy_labels)
    for idx in noisy_indices:
        available_labels = [i for i in range(len(label_mapping)) if i != y_train[idx]]
        y_train[idx] = random.choice(available_labels)

    X_train = X_train / 255.0  # Normalize pixel values to [0, 1]

    # Create an MLP model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=image_size + (3,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(label_mapping), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Noise Rate: {noise_rate:.2f} - Test Accuracy: {test_accuracy * 100:.2f}%")
