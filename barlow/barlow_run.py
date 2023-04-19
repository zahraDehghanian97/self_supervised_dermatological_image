from barlow.barlow_pretrain import *
from barlow.barlow_finetune import *
from utils.model_utils import *
from barlow import inception_v3
from segmentation import unet_model
from barlow import resnet20
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


def get_backbone(backbone_name, use_batchnorm=True,
                 use_dropout=False, dropout_rate=0.2,
                 crop_to=None, project_dim=None):
    backbone = None
    if backbone_name == 'resnet':
        resnet = resnet20.ResNet(use_batchnorm, use_dropout, dropout_rate)
        backbone = resnet.get_network(crop_to, hidden_dim=project_dim, use_pred=False,
                                      return_before_head=False)
    elif backbone_name == 'inception':
        backbone = inception_v3.get_network()

    elif backbone_name == 'unet':
        backbone = unet_model.get_unet_backbone((crop_to, crop_to))
    return backbone


def get_aug_function(aug_name, crop_to):
    if aug_name == 'tf':
        return get_tf_augment(crop_to)
    elif aug_name == 'original':
        return lambda x: custom_augment(x, crop_to)

    elif 'central-crop':
        return lambda img: dermoscopic_augment(img, crop_to)
    else:
        return 'no-augmentation function found'


class PretrainParams:
    def __init__(self, crop_to, batch_size, project_dim, checkpoints,
                 save_path, name, experiment_path='',
                 use_batchnorm=True, use_dropout=False, dropout_rate=0.2,
                 optimizer=tf.keras.optimizers.Adam(), backbone='resnet', aug_name='tf'):
        self.crop_to = crop_to
        self.batch_size = batch_size
        self.project_dim = project_dim
        self.checkpoints = checkpoints
        self.save_path = save_path + 'pretrain/'
        if len(experiment_path) > 0:
            self.save_path += experiment_path + '/'
        self.name = name

        self.backbone = backbone
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        if backbone == 'inception':
            self.crop_to = 299
            self.normalized = True
        else:
            self.normalized = False
        self.optimizer = optimizer
        self.aug_name = aug_name

        self.augment_function = get_aug_function(aug_name, crop_to)

    def get_summary(self):
        # summary = f'{self.name}_ct{self.crop_to}_bs{self.batch_size}_aug_{self.aug_name}'
        # if self.backbone == 'resnet':
        #     return summary
        # return summary + f'_{self.backbone}'
        return self.name

    def get_old_summary(self):
        return f'{self.name}_pretrain_projdim{self.project_dim}_bs{self.batch_size}_ct{self.crop_to}'

    def get_model_path(self):
        return self.save_path + self.get_summary()
        # if 'old' in self.save_path:
        #     return self.save_path + self.get_old_summary()
        # return self.save_path + self.get_summary()

    def get_report(self):
        return 'pretrain params: epochs {0}, bs {1}, image size {2}, project dim{3}' \
            .format(self.checkpoints[-1], self.batch_size, self.crop_to, self.project_dim)


class FineTuneParams:
    def __init__(self, checkpoints, batch_size,
                 pretrain_params: PretrainParams,
                 pretrain_epoch, save_path,
                 experiment_path='',
                 name='', loss=None):
        self.checkpoints = checkpoints
        self.batch_size = batch_size
        self.crop_to = pretrain_params.crop_to
        self.pretrain_params = pretrain_params
        self.pretrain_epoch = pretrain_epoch
        self.save_path = save_path + 'finetune/'

        if len(experiment_path) > 0:
            self.save_path += experiment_path + '/'

        if loss is None:
            self.loss = "binary_crossentropy"
            self.loss_name = 'normal'
        else:
            self.loss = loss
            self.loss_name = 'weighted'
        self.name = name
        self.experiment_path = experiment_path

    def get_summary(self):
        summary = self.pretrain_params.get_summary()
        if len(self.name) > 0:
            return f'{self.name}_{summary}'
        else:
            return summary

    def get_old_summary(self):

        return f'finetune_bs{self.batch_size}_ct{self.crop_to}_loss_{self.loss_name}'

    def get_model_path(self):
        return self.save_path + self.get_summary()


from barlow import eval_pretrain


def run_pretrain(ds, params: PretrainParams, debug=False):
    # es = EarlyStopping(monitor='loss', mode='min', verbose=1, min_delta=1)
    model_path = params.save_path + params.get_summary() + '/best_model'
    mc = ModelCheckpoint(model_path, monitor='loss', mode='min', save_best_only=True, verbose=1)
    backbone = get_backbone(params.backbone, params.use_batchnorm,
                            params.use_dropout, params.dropout_rate,
                            params.crop_to, params.project_dim)
    x_train, x_test = ds.get_x_train_test_ds()
    ssl_ds = prepare_data_loader(x_train, params.batch_size, params.augment_function)

    # lr_decayed_fn = get_lr(x_train, params.batch_size, params.checkpoints[-1])
    optimizer = params.optimizer  # .SGD(learning_rate=lr_decayed_fn, momentum=0.9)
    model = compile_barlow(backbone, optimizer)

    compile_function = lambda model: model.compile(optimizer=optimizer)
    train_model(model, ssl_ds, params.checkpoints, params.save_path,
                params.get_summary(), load_latest_model=True,
                debug=debug, compile_function=compile_function)

    val_ssl_ds = prepare_data_loader(x_test, params.batch_size, params.augment_function)
    eval_pretrain.model_val_loss(params.save_path + params.get_summary(), val_ssl_ds)
    return model.encoder


def run_fine_tune(ds, params: FineTuneParams, barlow_enc=None):
    print('running-finetune')
    outshape = ds.train_labels.shape[-1]
    train_ds, test_ds = ds.get_supervised_ds_sample()
    print(f'training on {len(train_ds)} data...')
    train_ds, test_ds = prepare_supervised_data_loader(train_ds, test_ds, params.batch_size, params.crop_to)

    if barlow_enc is None:
        print('loading pretrained-encoder')
        pretrain_path = params.pretrain_params.get_model_path()

        pretrain_path += f'/e{params.pretrain_epoch}'

        barlow_enc = tf.keras.models.load_model(pretrain_path)
    else:
        print('not loading backbone. using function inputs.')

    # cosine_lr = 0.01  # get_cosine_lr(params.checkpoints[-1], len(train_ds), params.batch_size)
    linear_model = get_classifier(barlow_enc, params.crop_to, outshape,
                                  params.pretrain_params.use_dropout)
    # Compile model and start training.
    linear_model.compile(
        loss=params.loss,
        metrics=get_metrics(),
        optimizer=tf.keras.optimizers.Adam()
    )

    train_model(linear_model, train_ds, params.checkpoints, params.save_path, params.get_summary(),
                test_ds)

    # test_acc = linear_model.evaluate(test_ds)
    # print(f'test acc {test_acc}')
    # _, test_acc = linear_model.evaluate(test_ds)
    # print("Test accuracy: {:.2f}%".format(test_acc * 100))

    # linear_model.save(params.get_model_path())
    # plt.plot(history.history['loss'])
    # plt.savefig('{0}figures/{1}.png'.format(params.pretrain_params.save_path, params.get_summary()))
    return linear_model
