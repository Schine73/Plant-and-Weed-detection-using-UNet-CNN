from genericpath import isfile
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.lines import Line2D
import pathlib



# ----- define model parameters and other variables
n_classes = 4

project_dir = str(pathlib.Path(__file__).parent.resolve())
img_dir = project_dir + "/test_images"
temp_dir = project_dir + "/temp"
model_file_path = project_dir + '/trained_models/2022-06-29_16-16-35_epochs-245_learnRate-0,0001_valLoss-0,006471.hdf5'
# -----


# ----- load trained model
def weighted_cross_entropy(beta):
    def loss(y_true, y_pred):
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = 1 - tf.cast(y_true, tf.float32)
        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b 
        return tf.reduce_mean(o)
    return loss

model = tf.keras.models.load_model(model_file_path, custom_objects={'loss': weighted_cross_entropy})
# -----

def get_image(fname):
    """returns image as 4-D Tensor of shape [batch, height, width, channels]"""
    img_str = tf.io.read_file(fname)
    image = tf.image.decode_jpeg(img_str, channels=3)
    image = tf.expand_dims(image, axis=0)
    return image

def predict_label(image):
    """takes images of any size and returns predicted labels in same size"""
    orig_size =[len(image[0]), len(image[0][0])]

    scaled_image = tf.image.resize(image, [256, 256])
    scaled_image = tf.cast(scaled_image, tf.uint8)
    scaled_image = tf.cast(scaled_image, tf.float32) *  (1 / 255.)

    label = model.predict(scaled_image)
    label = tf.image.resize(label, orig_size)
    label = np.argmax(label, axis=3)
    label = tf.cast(label, tf.uint8)
    return label

def get_pixel_percentage(label):
    """takes label and returns dictionary containing the
    classes within the label and thier respective percentage"""
    unique, counts = np.unique(label, return_counts=True)
    total = np.sum(counts)
    pixel_percentage = dict(zip(unique, counts))
    for i in range(n_classes):
        try:
            pixel_percentage[i] = pixel_percentage[i] / total
        except KeyError:
            pixel_percentage[i] = 0
    return pixel_percentage

def display_results(image, label, file_nr):
    """takes original image and label, overlays them in a plot
    and outputs the plot to a file ending in file_nr"""
    pixel_percentage = get_pixel_percentage(label)
    color_map = plt.get_cmap('plasma', n_classes)
    legend_elements = [Line2D([0], [0], marker='s', color=color_map(0),
                              label='Background: {:.2f}%'.format(pixel_percentage[0]*100), linewidth=0),
                    Line2D([0], [0], marker='s', color=color_map(1),
                           label='Weeds: {:.2f}%'.format(pixel_percentage[1]*100), linewidth=0),
                    Line2D([0], [0], marker='s', color=color_map(2),
                           label='Sugar Beet: {:.2f}%'.format(pixel_percentage[2]*100), linewidth=0),
                    Line2D([0], [0], marker='s', color=color_map(3),
                           label='Maize: {:.2f}%'.format(pixel_percentage[3]*100), linewidth=0)]
    plt.figure(figsize=(10,5))
    plt.imshow(image[0])
    plt.imshow(label[0], alpha=0.5, cmap=color_map, vmin=-0.5, vmax=3.5)
    plt.legend(handles=legend_elements, loc='upper right', fontsize='x-small')
    plt.axis("off")
    print("saving image nr. {} ...".format(file_nr))
    plt.savefig(temp_dir + "/full_image_test{}.png".format(file_nr), bbox_inches='tight', pad_inches = 0, dpi=300)


# ----- get image paths
filenames = []

for file in os.listdir(img_dir):
    if os.path.isfile(os.path.join(img_dir, file)):
        if os.path.splitext(file)[1] == ".jpg":
            filenames.append(os.path.join(img_dir, file))

print("Number of Images: {}".format(len(filenames)))             
# -----


# ----- predict labels for all images
for i in range(len(filenames)):
    image = get_image(filenames[i])
    label = predict_label(image)
    display_results(image, label, i + 1)
# -----