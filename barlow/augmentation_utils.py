import tensorflow as tf
from tensorflow.keras import layers


# CROP_TO = 32


def random_resize_crop(image, scale=[0.75, 1.0], crop_size=32):
    if crop_size == 32:
        image_shape = 48
        image = tf.image.resize(image, (image_shape, image_shape))
    else:
        image_shape = int(1.5 * crop_size)
        image = tf.image.resize(image, (image_shape, image_shape))
    size = tf.random.uniform(
        shape=(1,),
        minval=scale[0] * image_shape,
        maxval=scale[1] * image_shape,
        dtype=tf.float32,
    )
    size = tf.cast(size, tf.int32)[0]
    crop = tf.image.random_crop(image, (size, size, 3))
    crop_resize = tf.image.resize(crop, (crop_size, crop_size))
    return crop_resize


def flip_random_crop(image, crop_to):
    image = tf.image.random_flip_left_right(image)
    image = random_resize_crop(image, crop_size=crop_to)
    return image


@tf.function
def float_parameter(level, maxval):
    return tf.cast(level * maxval / 10.0, tf.float32)


@tf.function
def sample_level(n):
    return tf.random.uniform(shape=[1], minval=0.1, maxval=n, dtype=tf.float32)


@tf.function
def solarize(image, level=6):
    threshold = float_parameter(sample_level(level), 1)
    return tf.where(image < threshold, image, 255 - image)


def color_jitter(x, strength=0.5):
    x = tf.image.random_brightness(x, max_delta=0.8 * strength)
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength
    )
    x = tf.image.random_hue(x, max_delta=0.2 * strength)
    x = tf.clip_by_value(x, 0, 255)
    return x


def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x


def custom_augment(image, crop_to):
    image = tf.cast(image, tf.float32)
    image = flip_random_crop(image, crop_to)
    image = random_apply(color_jitter, image, p=0.9)
    image = random_apply(color_drop, image, p=0.3)
    image = random_apply(solarize, image, p=0.3)
    return image


def get_tf_augment(crop_to):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom((0, -0.4)),
        layers.experimental.preprocessing.RandomContrast(0.3),
        layers.Resizing(crop_to, crop_to),
        # layers.Rescaling(1. / 255)
    ])
    return data_augmentation


def dermoscopic_augment(image, crop_to):
    image = tf.image.random_hue(image, 0.02)
    # image = tf.image.random_contrast(image, 0.7, 0.8)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    zoomed = tf.image.central_crop(image, central_fraction=0.5)

    return tf.image.resize(zoomed, (crop_to, crop_to))

