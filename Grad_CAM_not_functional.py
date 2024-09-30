# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from training_images_list import traininglist
from valid_images_list import validlist
from testing_images_list import testlist


import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
# from sklearn.metrics import confusion_matrix
from six.moves import urllib
# import tensorflow as tf
import tensorflow.compat.v1 as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
#import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
from six.moves import urllib

# Model parameters
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def load_image(image_path):
    """Load and preprocess an image."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = (image / 255.0) * 2.0 - 1.0  # Inception v3 specific preprocessing
    return image


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system."""
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images, testing_images, validation_images = [], [], []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            if base_name in testlist:
                testing_images.append(base_name)
            elif base_name in validlist:
                validation_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                                 filepath,
                                                 _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_inception_graph():
    """Creates a graph from the saved GraphDef file and returns a Graph object."""
    with tf.Graph().as_default() as graph:
        model_filename = os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def grad_cam(image_path, target_class_index, model, layer_name='mixed_10/tower_2/conv'):
    """Generates a Grad-CAM heatmap for the given image and model."""
    image_data = tf.io.read_file(image_path)
    img = load_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    with tf.Session(graph=model) as sess:
        # Get the model's output tensor and the last convolutional layer's output
        model_output = model.get_tensor_by_name('softmax:0')
        conv_layer = model.get_tensor_by_name('mixed_10/tower_1/mixed/conv_1/Conv2D:0')

        # Get the gradients of the output with respect to the last convolutional layer
        grads = tf.gradients(model_output[0, target_class_index], conv_layer)[0]

        # Compute the activations and gradients
        conv_layer_output, grads_val = sess.run(
            [conv_layer, grads],
            feed_dict={model.get_tensor_by_name(JPEG_DATA_TENSOR_NAME): image_data}
        )

        # Pool the gradients over the channel dimensions
        pooled_grads = np.mean(grads_val, axis=(0, 1, 2))

        # Weight the feature maps by the pooled gradients
        heatmap = np.dot(conv_layer_output[0], pooled_grads)

        # Apply ReLU to the heatmap
        heatmap = np.maximum(heatmap, 0)

        # Normalize the heatmap
        heatmap /= np.max(heatmap)

        # Resize the heatmap to match the original image dimensions
        original_img = tf.image.decode_jpeg(image_data, channels=3)
        original_img = tf.image.resize(original_img, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH])
        original_img = sess.run(original_img)

        # Resize the heatmap to match the original image dimensions
        heatmap = tf.image.resize(heatmap, [original_img.shape[0], original_img.shape[1]])
        heatmap = sess.run(heatmap)

        # Convert heatmap to color using a colormap (viridis)
        heatmap_color = plt.get_cmap('viridis')(heatmap)[:, :, :3]  # Keep RGB channels only
        heatmap_color = np.uint8(255 * heatmap_color)  # Convert to uint8

        # Overlay the heatmap on the original image
        overlayed_image = np.clip(original_img * 0.6 + heatmap_color * 0.4, 0, 255).astype(np.uint8)

        return overlayed_image


def main(_):
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir + datetime.now().strftime("%Y%m%d-%H%M")):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir + datetime.now().strftime("%Y%m%d-%H%M"))
    tf.gfile.MakeDirs(FLAGS.summaries_dir + datetime.now().strftime("%Y%m%d-%H%M"))

    # Set up the pre-trained graph.
    maybe_download_and_extract()
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = create_inception_graph()

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        print('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        print(
            'Only one valid folder of images found at ' + FLAGS.image_dir + ' - multiple classes are needed for classification.')
        return -1

    # Run Grad-CAM if specified
    if FLAGS.grad_cam:
        target_class_index = 0  # Specify the class index you want to visualize
        image_path = FLAGS.grad_cam_image  # Path to the input image for Grad-CAM
        overlayed_image = grad_cam(image_path, target_class_index, graph)

        # Plotting the overlayed image
        plt.figure(figsize=(10, 10))
        plt.imshow(overlayed_image)
        plt.axis('off')
        plt.title('Grad-CAM Overlay')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='', help='Path to folders of labeled images.')
    parser.add_argument('--output_graph', type=str, default='/tmp/output_graph.pb',
                        help='Where to save the trained graph.')
    parser.add_argument('--output_labels', type=str, default='/tmp/output_labels.txt',
                        help='Where to save the trained graph\'s labels.')
    parser.add_argument('--summaries_dir', type=str, default='/tmp/retrain_logs',
                        help='Where to save summary logs for TensorBoard.')
    parser.add_argument('--how_many_training_steps', type=int, default=4000,
                        help='How many training steps to run before ending.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='How large a learning rate to use when training.')
    parser.add_argument('--testing_percentage', type=int, default=10,
                        help='What percentage of images to use as a test set.')
    parser.add_argument('--validation_percentage', type=int, default=10,
                        help='What percentage of images to use as a validation set.')
    parser.add_argument('--eval_step_interval', type=int, default=10,
                        help='How often to evaluate the training results.')
    parser.add_argument('--train_batch_size', type=int, default=100, help='How many images to train on at a time.')
    parser.add_argument('--test_batch_size', type=int, default=-1, help='How many images to test on.')
    parser.add_argument('--validation_batch_size', type=int, default=100,
                        help='How many images to use in an evaluation batch.')
    parser.add_argument('--print_misclassified_test_images', default=False,
                        help='Whether to print out a list of all misclassified test images.', action='store_true')
    parser.add_argument('--model_dir', type=str, default='/tmp/imagenet', help='Path to classify_image_graph_def.pb.')
    parser.add_argument('--bottleneck_dir', type=str, default='/tmp/bottleneck',
                        help='Path to cache bottleneck layer values as files.')
    parser.add_argument('--final_tensor_name', type=str, default='final_result',
                        help='The name of the output classification layer in the retrained graph.')

    # Grad-CAM specific arguments
    parser.add_argument('--grad_cam', default=False, help='Whether to generate Grad-CAM visualization.',
                        action='store_true')
    parser.add_argument('--grad_cam_image', type=str, default='',
                        help='Path to the image for which to generate Grad-CAM.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
