from ResNetAE.ResNetAE import *
##### Define Model, Optimizer and Loss #################################################################################
import tensorflow as tf


model = ResNetAE(input_shape=INPUT_SHAPE,
                    vq_num_embeddings=NUM_LATENT_K,
                    vq_embedding_dim=NUM_LATENT_D,
                    vq_commiment_cost=BETA)

optimizer = tf.keras.optimizers.Adam(learning_rate=VQVAE_LEARNING_RATE)
mse_loss = tf.keras.losses.MSE

train_total_loss = tf.keras.metrics.Mean()
train_VecQuant_loss = tf.keras.metrics.Mean()
train_reconstruction_loss = tf.keras.metrics.Mean()

val_total_loss = tf.keras.metrics.Mean()
val_VecQuant_loss = tf.keras.metrics.Mean()
val_reconstruction_loss = tf.keras.metrics.Mean()


##### ResNetVQVAE Training Loop ########################################################################################

@tf.function
def train_step(data):
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
def val_step(data):
    vq_loss, data_recon, perplexity, encodings = model(data)
    recon_err = mse_loss(data_recon, data)
    loss = vq_loss + recon_err

    val_total_loss(loss)
    val_reconstruction_loss(recon_err)
    val_VecQuant_loss(vq_loss)

for epoch in range(12):

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
        train_step(data)
        t.set_description('>%d, t_loss=%.5f, recon_loss=%.5f, VecQuant_loss=%.5f' % (step,
                                                                                     train_total_loss.result(),
                                                                                     train_reconstruction_loss.result(),
                                                                                     train_VecQuant_loss.result()))

    # ## Evaluation Step
    # for step, data in tqdm.tqdm(enumerate(val_ds), total=len(val_ds)):
    #     val_step(data)

    # print(f'Epoch {epoch} complete in: {time.time() - start:.5f}')
    # print('t_loss={:.5f}, recon_loss={:.5f}, VecQuant_loss={:.5f}'.format(val_total_loss.result(),
    #                                                                       val_reconstruction_loss.result(),
    #                                                                       val_VecQuant_loss.result()))

    # model.save_weights(os.path.join(SAVE_DIR, f'model_{epoch}.h5'))

