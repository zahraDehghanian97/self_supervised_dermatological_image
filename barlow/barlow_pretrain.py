import barlow.resnet20 as resnet20
import barlow.lr_scheduler as lr_scheduler

from barlow.data_utils import *
# from datasets.ham_dataset import *

AUTO = tf.data.AUTOTUNE


def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n - 1, n + 1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])


def normalize_repr(z):
    z_norm = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
    return z_norm


def compute_loss(z_a, z_b, lambd):
    loss, on_diag, off_diag = compute_loss_detail(z_a, z_b, lambd)
    return loss


def compute_loss_detail(z_a, z_b, lambd):
    # Get batch size and representation dimension.
    batch_size = tf.cast(tf.shape(z_a)[0], z_a.dtype)
    repr_dim = tf.shape(z_a)[1]

    # Normalize the representations along the batch dimension.
    z_a_norm = normalize_repr(z_a)
    z_b_norm = normalize_repr(z_b)

    # Cross-correlation matrix.
    c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / batch_size

    # Loss.
    on_diag = tf.linalg.diag_part(c) + (-1)
    on_diag = tf.reduce_sum(tf.pow(on_diag, 2))

    off_diag = off_diagonal(c)
    off_diag = tf.reduce_sum(tf.pow(off_diag, 2))

    loss = on_diag + (lambd * off_diag)
    return loss, on_diag, off_diag


class BarlowTwins(tf.keras.Model):
    def __init__(self, encoder, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.encoder = encoder
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z_a, z_b = self.encoder(ds_one, training=True), self.encoder(ds_two, training=True)
            loss = compute_loss(z_a, z_b, self.lambd)

        # Compute gradients and update the parameters.
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def get_lr(x_train, batch_size, epochs):
    STEPS_PER_EPOCH = len(x_train) // batch_size
    # TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
    WARMUP_EPOCHS = int(epochs * 0.1)
    WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

    lr_decayed_fn = lr_scheduler.WarmUpCosine(
        learning_rate_base=1e-3,
        total_steps=epochs * STEPS_PER_EPOCH,
        warmup_learning_rate=0.0,
        warmup_steps=WARMUP_STEPS
    )
    return lr_decayed_fn


def compile_barlow(encoder, opt):
    barlow_twins = BarlowTwins(encoder)
    barlow_twins.compile(optimizer=opt)
    return barlow_twins


def prepare_x(x_ds, bs, crop_to):
    return x_ds.map(lambda x: tf.image.resize(x, (crop_to, crop_to))).batch(bs)


if __name__ == '__main__':
    crop_to = 128
    batch_size = 16
    project_dim = 2048
    epochs = 5

    # (x_train, y_train), (x_test, y_test) = get_cfar_data()
    ds = HAMDataset(data_path='../../data/ISIC/ham10000/', label_filename='disease_labels.csv',
                    image_col='image', image_folder='resized256/', data_size=30)
    x_train, x_test = ds.get_x_train_test_ds()

    ssl_ds = prepare_data_loader(x_train, crop_to, batch_size)
    augment_function = lambda x: custom_augment(x, crop_to)
    val_ssl_ds = prepare_data_loader(x_test, batch_size, augment_function)
    lr_decayed_fn = get_lr(x_train, batch_size, epochs)

    resnet_enc = resnet20.get_network(crop_to, hidden_dim=project_dim, use_pred=False,
                                      return_before_head=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)
    model = compile_barlow(resnet_enc, optimizer)
    history = model.fit(ssl_ds, epochs=epochs)
