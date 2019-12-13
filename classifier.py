from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os


# Create data sets
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Add class_names to identify labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Preprocess datasets. Scale values into 0 or zero instead
train_images = train_images / 255.0
test_images = test_images / 255.0

save_path = 'saved_model/my_model'

def get_model():
    model = None
    trained = False
    if os.path.exists(save_path):
        print("LOAD DATA")
        # Load model
        model = tf.keras.models.load_model(save_path)
        trained = True
    else:
        # Create a model with 3 layers: Flatten, and 2 Dense layers
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        # Compile the models with settings on the loss function, optimizer and metrics
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    return (model, trained)

def create_callback():
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    return cp_callback

# Train the model with callbacks
def train_model(model, cp_callback):
    model.fit(train_images,
              train_labels,
              epochs=10,
              validation_data=(test_images,test_labels),
              callbacks=[cp_callback])
    # Save trained model
    model.save(save_path)

# Test the model
def test_model(model):
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

# Plot predictions
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


def main():
    cp_callback = create_callback()
    (model, trained) = get_model()
    # Check if model already trained
    if not trained:
        train_model(model, cp_callback)
    test_model(model)


if __name__== "__main__":
  main()
