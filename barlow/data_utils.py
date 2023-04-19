from barlow.augmentation_utils import *
# import visualization.visual_augmentations as vis

AUTO = tf.data.AUTOTUNE
SEED = 42


#
# BATCH_SIZE = 512


def get_cfar_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(f"Total training examples: {len(x_train)}")
    print(f"Total test examples: {len(x_test)}")
    return (x_train, y_train), (x_test, y_test)


# def show_sample(ds, cnt):
#     samples = next(iter(ds))
#     samples = samples[:cnt]
#     samples = [vis.get_showable_img(img) for img in samples]
#     all = vis.concat_all(samples)
#     vis.cv2_imshow(all)


def prepare_data_loader(x_train, batch_size, augment_function):
    # ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_one = (
        x_train.shuffle(1024, seed=SEED)
            .map(augment_function, num_parallel_calls=AUTO)
            .batch(batch_size)
            .prefetch(AUTO)
    )

    # ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_two = (
        x_train.shuffle(1024, seed=SEED)
            .map(augment_function, num_parallel_calls=AUTO)
            .batch(batch_size)
            .prefetch(AUTO)
    )

    # show_sample(ssl_ds_one, 5)
    # show_sample(ssl_ds_two, 5)
    # We then zip both of these datasets.
    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))
    return ssl_ds


def prepare_supervised_data_loader(train_ds, test_ds, batch_size, crop_to):
    train_ds = (
        train_ds.shuffle(1024)
            .map(lambda x, y: (flip_random_crop(x, crop_to), y), num_parallel_calls=AUTO)
            .batch(batch_size)
            .prefetch(AUTO)
    )
    test_ds = test_ds.batch(batch_size).prefetch(AUTO)
    return train_ds, test_ds


def prepare_input_data(x_ds, crop_to, batch_size):
    return x_ds.map(lambda x: tf.image.resize(x, (crop_to, crop_to))).batch(batch_size)
