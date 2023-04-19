import tensorflow as tf
from prototype_network.ResNetAE.ResNetAE import ResNetVQVAE
import time
import tqdm
import os


##### Tensorflow Dataset ###############################################################################################
# load and process image
def parse_function(filename):
    # Read entire contents of image
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize image with padding to 256x256
    image = tf.image.resize_with_pad(image, 256, 256, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Convert image to grayscale
    image = tf.image.rgb_to_grayscale(image)

    return image


def get_dataset(train_path, val_path, test_path, batch_size):
    train_ds = tf.data.Dataset.from_tensor_slices((train_path))
    train_ds = train_ds.shuffle(len(train_ds))
    train_ds = train_ds.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices((val_path))
    val_ds = val_ds.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((test_path))
    test_ds = test_ds.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)

    return train_ds, test_ds, val_ds


def get_model(in_shape, num_latent, dim_latent):
    ##### Define Model, Optimizer and Loss #
    BETA = 0.25  # Weight for the commitment loss

    model = ResNetVQVAE(input_shape=in_shape,
                        vq_num_embeddings=num_latent,
                        vq_embedding_dim=dim_latent,
                        vq_commiment_cost=BETA)

    return model


def get_optimizer():
    VQVAE_LEARNING_RATE = 1e-4  # Learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=VQVAE_LEARNING_RATE)
    return optimizer


##### ResNetVQVAE Training Loop ########################################################################################
train_total_loss = tf.keras.metrics.Mean()
train_VecQuant_loss = tf.keras.metrics.Mean()
train_reconstruction_loss = tf.keras.metrics.Mean()

val_total_loss = tf.keras.metrics.Mean()
val_VecQuant_loss = tf.keras.metrics.Mean()
val_reconstruction_loss = tf.keras.metrics.Mean()
mse_loss = tf.keras.losses.MSE


@tf.function
def train_step(data, model, optimizer):
    with tf.GradientTape() as tape:
        vq_loss, data_recon, perplexity, encodings = model(data)
        recon_err = mse_loss(data_recon, data)
        loss = vq_loss + recon_err

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_total_loss(loss)
    train_reconstruction_loss(recon_err)
    train_VecQuant_loss(vq_loss)


@tf.function
def val_step(data, model):
    vq_loss, data_recon, perplexity, encodings = model(data)
    recon_err = mse_loss(data_recon, data)
    loss = vq_loss + recon_err

    val_total_loss(loss)
    val_reconstruction_loss(recon_err)
    val_VecQuant_loss(vq_loss)


def train_model(model, train_ds, val_ds, optimizer, epochs, save_path):
    for epoch in range(epochs):

        train_total_loss.reset_states()
        train_reconstruction_loss.reset_states()
        train_VecQuant_loss.reset_states()

        val_total_loss.reset_states()
        val_reconstruction_loss.reset_states()
        val_VecQuant_loss.reset_states()

        print('=' * 50, f'Training EPOCH {epoch}', '=' * 50)
        start = time.time()

        ## Train Step
        t = tqdm.tqdm(enumerate(train_ds), total=len(train_ds))
        for step, data in t:
            train_step(data, model, optimizer)
            t.set_description('>%d, t_loss=%.5f, recon_loss=%.5f, VecQuant_loss=%.5f' % (step,
                                                                                         train_total_loss.result(),
                                                                                         train_reconstruction_loss.result(),
                                                                                         train_VecQuant_loss.result()))

        ## Evaluation Step
        for step, data in tqdm.tqdm(enumerate(val_ds), total=len(val_ds)):
            val_step(data)

        print(f'Epoch {epoch} complete in: {time.time() - start:.5f}')
        print('t_loss={:.5f}, recon_loss={:.5f}, VecQuant_loss={:.5f}'.format(val_total_loss.result(),
                                                                              val_reconstruction_loss.result(),
                                                                              val_VecQuant_loss.result()))

        model.save_weights(os.path.join(save_path, f'model_{epoch}.h5'))


if __name__ == '__main__':
    in_shape, num_latent, dim_latent = (128, 128, 3), 512, 128
    train_path, val_path, test_path, batch_size = '', '', '', 16
    epochs, save_path = 100, ''

    optimizer = get_optimizer()
    model = get_model(in_shape, num_latent, dim_latent)
    train_ds, test_ds, val_ds = get_dataset(train_path, val_path, test_path, batch_size)
    train_model(model, train_ds, val_ds, optimizer, epochs, save_path)
