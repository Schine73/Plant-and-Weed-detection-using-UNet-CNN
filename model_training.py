from datetime import datetime
from time import strftime
from gc import callbacks
from genericpath import isfile
import os
import functools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
import pathlib

# ----- test that Tensorflow is correctly installd and GPU is available
print(f"Tensorflow version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
# -----



# ----- define model parameters and other variables
n_classes = 4               # nr of classes in images
img_shape = (256, 256, 3)   # define size of model input
batch_size = 3              # define batch size for datasets
epochs = 3                  # define max nr of epochs to run
learning_rate = 0.005       # define learning rate of optimizer
es_min_delta = 0            # define min absolute change in val_loss before early stopping
es_patience = 10            # define how many epochs to wait before early stopping

project_dir = str(pathlib.Path(__file__).parent.resolve())
img_dir = project_dir + "/training_images"  # training images and segmentation masks will be used from here
temp_dir = project_dir + "/temp"            # model files and training results will be saved to here
# -----



# ----- get image paths
x_filenames = []
y_filenames = []

for file in os.listdir(img_dir):
    if os.path.isfile(os.path.join(img_dir, file)):
        if os.path.splitext(file)[1] == ".jpg" or os.path.splitext(file)[1] == ".jpeg":
            # original images are jpg
            x_filenames.append(os.path.join(img_dir, file))
            # label images are .png and have same name as original images:
            y_filenames.append(os.path.join(img_dir, os.path.splitext(file)[0] + ".png"))

# split data into train- and testset
x_train_filenames, x_test_filenames, y_train_filenames, y_test_filenames = train_test_split(x_filenames, y_filenames, test_size=0.2, random_state=42)
# split testset into test- and validationset
x_val_filenames, x_test_filenames, y_val_filenames, y_test_filenames = train_test_split(x_test_filenames, y_test_filenames, test_size=0.2, random_state=42)

num_train_examples = len(x_train_filenames)
num_val_examples = len(x_val_filenames)
num_test_examples = len(x_test_filenames)

print("Number of training examples: {}".format(num_train_examples))
print("Number of validation examples: {}".format(num_val_examples))
print("Number of test examples: {}".format(num_test_examples))               
# -----



# ----- visualize example images
# select 5 random images from train dataset:
# r_choices = np.random.choice(num_train_examples, 5)

# plt.figure(figsize=(10, 15))
# for i in range(0, 5 * 2, 2):
#   img_num = r_choices[i // 2]
#   x_pathname = x_train_filenames[img_num]
#   y_pathname = y_train_filenames[img_num]

#   plt.subplot(5, 2, i + 1)
#   plt.imshow(mpimg.imread(x_pathname))
#   plt.title("Original Image")

#   example_labels = Image.open(y_pathname)
#   label_vals = np.unique(example_labels)

#   plt.subplot(5, 2, i + 2)
#   plt.imshow(example_labels)
#   plt.title("Original Mask")
# plt.suptitle("Original Images and thier Masks")
# plt.show()
# -----



# ----- define image processing functions
def _process_pathnames(fname, label_path): 
    img_str = tf.io.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels=3)

    label_img_str = tf.io.read_file(label_path)
    label_img = tf.image.decode_png(label_img_str, channels=1)
        
    label_img = tf.one_hot(label_img, depth=n_classes, on_value=1.0, off_value=0.0)
    label_img = tf.squeeze(label_img, axis=-2)
    return img, label_img

def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random.uniform([], 
                                                -width_shift_range * img_shape[1],
                                                width_shift_range * img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random.uniform([],
                                                -height_shift_range * img_shape[0],
                                                height_shift_range * img_shape[0])
        # Translate both 
        output_img = tfa.image.translate(output_img, [width_shift_range, height_shift_range])
        label_img = tfa.image.translate(label_img, [width_shift_range, height_shift_range])
    return output_img, label_img

def flip_img(horizontal_flip, tr_img, label_img):
    if horizontal_flip:
        flip_prob = tf.random.uniform([], 0.0, 1.0)
        tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                lambda: (tr_img, label_img))
    return tr_img, label_img

def _augment(img,
             label_img,
             resize=None,               # Resize the image to some size e.g. [256, 256]
             scale=1,                   # Scale image e.g. 1 / 255.
             hue_delta=0,               # Adjust the hue of an RGB image by random factor
             horizontal_flip=False,     # Random left right flip
             width_shift_range=0,       # Randomly translate the image horizontally
             height_shift_range=0):     # Randomly translate the image vertically 
    if resize is not None:
        # Resize both images
        label_img = tf.image.resize(label_img, resize)
        img = tf.image.resize(img, resize)
        
    if hue_delta:
        img = tf.image.random_hue(img, hue_delta)

    img, label_img = flip_img(horizontal_flip, img, label_img)
    img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
    label_img = tf.cast(label_img, tf.float32) * scale
    img = tf.cast(img, tf.float32) * scale 
    return img, label_img
# -----



# ----- get datasets
def get_baseline_dataset(filenames, 
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=5, 
                         batch_size=batch_size,
                         shuffle=True):           
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
        assert batch_size == 1, "Batching images must be of the same size"

    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)

    # It's necessary to repeat our data for all epochs 
    dataset = dataset.repeat().batch(batch_size)
    return dataset

# training dataset:
tr_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255.#,
    #'hue_delta': 0.1,
    #'horizontal_flip': True,
    #'width_shift_range': 0.1,
    #'height_shift_range': 0.1
}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)

# validation dataset:
val_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255.,
}
val_preprocessing_fn = functools.partial(_augment, **val_cfg)

# test dataset:
test_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255.,
}
test_preprocessing_fn = functools.partial(_augment, **test_cfg)

train_ds = get_baseline_dataset(x_train_filenames,
                                y_train_filenames,
                                preproc_fn=tr_preprocessing_fn,
                                batch_size=batch_size)

val_ds = get_baseline_dataset(x_val_filenames,
                                y_val_filenames, 
                                preproc_fn=val_preprocessing_fn,
                                batch_size=batch_size)

test_ds = get_baseline_dataset(x_test_filenames,
                                y_test_filenames, 
                                preproc_fn=test_preprocessing_fn,
                                batch_size=batch_size)
# -----



# ----- test image processing pipeline
# plt.figure(figsize=(10, 20))
# i = 0
# for element in train_ds:
#     if i >= 5: break;
#     batch_of_imgs, label = element
#     label = np.argmax(label[0], axis=2)
#     img = batch_of_imgs[0]
    
#     plt.subplot(5, 2, 2 * i + 1)
#     plt.imshow(img)
#     plt.title("Augmented Image")
    
#     plt.subplot(5, 2, 2 * i + 2)
#     plt.imshow(label)
#     plt.title("Augmented Mask")
#     i+=1
# plt.suptitle("Augmented Images and Masks")
# plt.show()
# -----



# ----- build model architecture
def conv_block(input_tensor, num_filters):
    encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('relu')(encoder)
    encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = tf.keras.layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = tf.keras.layers.BatchNormalization()(decoder)
    decoder = tf.keras.layers.Activation('relu')(decoder)
    decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = tf.keras.layers.BatchNormalization()(decoder)
    decoder = tf.keras.layers.Activation('relu')(decoder)
    decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = tf.keras.layers.BatchNormalization()(decoder)
    decoder = tf.keras.layers.Activation('relu')(decoder)
    return decoder


inputs = tf.keras.layers.Input(shape=img_shape)                 # 256
encoder0_pool, encoder0 = encoder_block(inputs, 32)             # 128
encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)      # 64
encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)     # 32
encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)     # 16
encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)     # 8

center = conv_block(encoder4_pool, 1024)                        # center

decoder4 = decoder_block(center, encoder4, 512)                 # 16
decoder3 = decoder_block(decoder4, encoder3, 256)               # 32
decoder2 = decoder_block(decoder3, encoder2, 128)               # 64
decoder1 = decoder_block(decoder2, encoder1, 64)                # 128
decoder0 = decoder_block(decoder1, encoder0, 32)                # 256

outputs = tf.keras.layers.Conv2D(n_classes, (1, 1))(decoder0)

model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

def weighted_cross_entropy(beta):
    def loss(y_true, y_pred):
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = 1 - tf.cast(y_true, tf.float32)
        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b 
        return tf.reduce_mean(o)
    return loss

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, loss=weighted_cross_entropy(beta=1), metrics=['accuracy'])
# model.summary()
# -----



# ----- train model
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=es_min_delta,
    patience=es_patience,
    mode='min',
    restore_best_weights=True
    )

history = model.fit(
    train_ds, 
    steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),
    epochs=epochs,
    validation_data=val_ds,
    validation_steps=int(np.ceil(num_val_examples / float(batch_size))),
    callbacks=[es]
    )

if es.stopped_epoch != 0: epochs = es.stopped_epoch + 1
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model.save(temp_dir + "/{}_epochs-{}_learnRate-{}_valLoss-{:.6f}".format(timestamp, epochs , learning_rate, np.min(history.history['val_loss'])).replace('.',',') + ".hdf5")
# -----



# ----- visualize training process
loss = history.history['loss']
val_loss = history.history['val_loss']

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='lower right')
plt.title('Training and Validation Loss')

plt.savefig(temp_dir + "/{}_trainingProcess".format(timestamp).replace('.',',') + ".png")
plt.show()
# -----



# ----- test model with test dataset
plt.figure(figsize=(10, 20))

i = 0

for element in test_ds:
    if i >= 5: break;
    batch_of_imgs, label = element
    label = np.argmax(label[0], axis=2)
    img = batch_of_imgs[0]
    predicted_label = model.predict(batch_of_imgs)[0]
    predicted_label = np.argmax(predicted_label, axis=2)
    
    plt.subplot(5, 3, 3 * i + 1)
    plt.imshow(img)
    plt.title("Input image")
    
    plt.subplot(5, 3, 3 * i + 2)
    plt.imshow(X=label, vmin=0, vmax=3)
    plt.title("Actual Mask")
    
    plt.subplot(5, 3, 3 * i + 3)
    plt.imshow(X=predicted_label, vmin=0, vmax=3)
    plt.title("Predicted Mask")
    
    i+=1

plt.suptitle("Examples of Input Image, Label, and Prediction")
plt.savefig(temp_dir + "/{}_testImages".format(timestamp).replace('.',',') + ".png")
plt.show()
# -----