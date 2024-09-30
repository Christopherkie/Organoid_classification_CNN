
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def load_inception_v3(model_dir):
    """Loads the Inception v3 model."""
    model_path = tf.keras.utils.get_file(
        'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
        cache_subdir=model_dir,
        file_hash='bcbd6486424b2319ff4ef7d526e38f63'
    )
    model = tf.keras.applications.InceptionV3(include_top=False, weights=model_path)
    return model


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def load_inception_v3():
    """Loads the Inception v3 model pre-trained on ImageNet data."""
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    return model

def visualize_layer_filters(model, layer_name, max_filters=64):
    """Visualizes the filters of a specified layer from a model, handling layers without biases."""
    # Retrieve the layer and its weights (assuming it has filters)
    try:
        layer = model.get_layer(layer_name)
        weights = layer.get_weights()
        if len(weights) > 0:
            filters = weights[0]
        else:
            print(f"No weights found for layer {layer_name}")
            return

        # Normalize filter values to 0-1 so we can visualize them
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        # Plot the first few filters
        n_filters = min(max_filters, filters.shape[3])
        n_columns = 6  # How many columns of subplots in the figure
        n_rows = (n_filters + n_columns - 1) // n_columns  # Calculate rows needed
        plt.figure(figsize=(20, max(2 * n_rows, 2)))
        for i in range(n_filters):
            f = filters[:, :, :, i]
            for j in range(3):  # Assuming the input images are RGB
                ax = plt.subplot(n_rows, n_columns * 3, i * 3 + j + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(f[:, :, j], cmap='gray')
        plt.show()

    except ValueError as e:
        print(f"Error accessing weights for layer {layer_name}: {e}")

# Example usage
model = load_inception_v3()
visualize_layer_filters(model, 'conv2d')
