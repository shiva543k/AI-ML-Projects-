import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images (scale pixel values between 0 and 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define class names
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# Display some sample images from the dataset
def show_sample_images():
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i])
        plt.title(class_names[y_train[i][0]])
        plt.axis('off')
    plt.show()

# Call function to display images
show_sample_images()

# Save the preprocessed dataset as NumPy arrays
import numpy as np
np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)

print("Dataset saved as NumPy arrays: x_train.npy, y_train.npy, x_test.npy, y_test.npy")
