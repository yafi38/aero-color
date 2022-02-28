import cv2
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Concatenate


bin_val = np.array([32, 80, 104, 116, 124, 132, 140, 152, 176, 224])
num_bins = bin_val.shape[0]


def scale_image(image, size):
    image_height, image_width = image.shape[:2]

    if image_height <= image_width:
        ratio = image_width / image_height
        h = size
        w = int(ratio * h)

        image = cv2.resize(image, (w, h))

    else:
        ratio = image_height / image_width
        w = size
        h = int(ratio * w)

        image = cv2.resize(image, (w, h))

    return image


def center_crop(image, size):
    image_height, image_width = image.shape[:2]

    if image_height <= image_width and abs(image_width - size) > 1:

        dx = int((image_width - size) / 2)
        image = image[:, dx:-dx, :]
    elif abs(image_height - size) > 1:
        dy = int((image_height - size) / 2)
        image = image[dy:-dy, :, :]

    image_height, image_width = image.shape[:2]
    if image_height is not size or image_width is not size:
        image = cv2.resize(image, (size, size))

    return image


def resize_image(image, size):
    image = scale_image(image, size)
    image = center_crop(image, size)
    return image


def conv_stack(input_layer, filters):
    conv1 = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch_norm1)

    conv2 = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(relu1)
    batch_norm2 = BatchNormalization()(conv2)
    relu2 = ReLU()(batch_norm2)

    return relu2


def encoder_block(input_layer, filters):
    conv = conv_stack(input_layer, filters)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv)

    return conv, max_pool


def decoder_block(input_layer, skip_layer, filters):
    up = Conv2DTranspose(filters, 2, strides=2, padding='same')(input_layer)
    conc = Concatenate()([up, skip_layer])
    dec = conv_stack(conc, filters)

    return dec


def get_model(size, init_filters):
    inputs = Input((size, size, 1))

    conv1, max_pool1 = encoder_block(inputs, init_filters)
    conv2, max_pool2 = encoder_block(max_pool1, init_filters * 2)
    conv3, max_pool3 = encoder_block(max_pool2, init_filters * 4)

    middle_block = conv_stack(max_pool3, init_filters * 8)

    decoder1 = decoder_block(middle_block, conv3, init_filters * 4)
    decoder2 = decoder_block(decoder1, conv2, init_filters * 2)
    decoder3 = decoder_block(decoder2, conv1, init_filters)

    # soft = Softmax(axis=1)(decoder3)
    output_a = Conv2D(num_bins, 1, padding='same', activation='softmax')(decoder3)
    # output_b = Conv2D(32, 1, padding='same', activation='softmax')(decoder3)

    model = Model(inputs, output_a)
    return model