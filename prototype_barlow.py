import tensorflow as tf
from PrototypeDL.autoencoder_helpers import list_of_distances, list_of_norms
from barlow import barlow_pretrain


class PrototypeBarlow(tf.keras.Model):
    def __init__(self, auto_encoder, n_prototypes, project_dim, lambd=5e-3):
        super(PrototypeBarlow, self).__init__()
        self.auto_encoder = auto_encoder
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # the list prototype feature vectors
        self.prototype_feature_vectors = tf.Variable(tf.random.uniform(shape=[n_prototypes, project_dim],
                                                                       dtype=tf.float32),
                                                     name='prototype_feature_vectors')

    @property
    def metrics(self):
        return [self.loss_tracker]

    def calc_list_distance(self, a, b):
        distances = list_of_distances(a, b)
        return tf.identity(distances, name='prototype_distances')

    def calc_prototype_distance(self, z):
        return self.calc_list_distance(z, self.prototype_feature_vectors)

    def calc_feature_distance(self, z):
        return self.calc_list_distance(self.prototype_feature_vectors, z)

    def prototype_loss(self, z_a, z_b):
        prot_dist_a = self.calc_prototype_distance(z_a)
        prot_dist_b = self.calc_prototype_distance(z_b)

        feature_dist_a = self.calc_feature_distance(z_a)
        feature_dist_b = self.calc_feature_distance(z_b)

        error_1 = tf.reduce_mean(tf.reduce_min(feature_dist_a + feature_dist_b, axis=1), name='error_1')
        error_2 = tf.reduce_mean(tf.reduce_min(prot_dist_a + prot_dist_b, axis=1), name='error_2')

        return error_1 + error_2, prot_dist_a, prot_dist_b

    def vae_loss_embedding(self, x, z):
        decoded = self.auto_encoder.decode(z)
        return tf.reduce_mean(list_of_norms(decoded - x), name='ae_error')

    def vae_loss(self, ds1, z1, ds2, z2):
        vae_loss_a = self.vae_loss_embedding(ds1, z1)
        vae_loss_b = self.vae_loss_embedding(ds2, z2)
        return vae_loss_a + vae_loss_b

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z_a, z_b = self.auto_encoder.encode(ds_one), self.auto_encoder.encode(ds_two)

            prot_loss, prot_dist_a, prot_dist_b = self.prototype_loss(z_a, z_b)

            barlow_loss = barlow_pretrain.compute_loss(prot_dist_a, prot_dist_b, self.lambd)
            vae_loss = self.vae_loss(ds_one, z_a, ds_two, z_b)
            loss = prot_loss + vae_loss + barlow_loss

        # Compute gradients and update the parameters.
        auto_encoder_gradients = tape.gradient(loss, self.auto_encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(auto_encoder_gradients, self.auto_encoder.trainable_variables))

        # prot_gradients = tape.gradient(loss, self.prototype_feature_vectors)
        # self.optimizer.apply_gradients(zip(prot_gradients, self.prototype_feature_vectors))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


from ResNetAE import ResNetAE

if __name__ == '__main__':
    ct = 128
    auto_encoder = ResNetAE.ResNetAE(input_shape=(ct, ct, 3), n_levels=2,
                                     z_dim=2048, bottleneck_dim=128)
    prot_barlow = PrototypeBarlow(auto_encoder, n_prototypes=10, project_dim=2048)
    
