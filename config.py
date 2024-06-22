from __future__ import print_function, division


from option.config import Config

# config file
config = Config({
    # device
    'gpu_id': "0",  # specify GPU number to use
    'num_workers': 0,

    # data
    "is_dicom": False,
    "model_classes": ['ankle', 'wrist', 'd_ankle'],
    "now_class": 0,  # database type
    'train_data': 'img/',
    'label': 'new_label.xlsx',
    'test_path': 'test.txt',
    'random_seed': 23,
    'K': 1,  # train/vaildation separation ratio
    'batch_size': 4,

    'output_size': (16, 1),
    'input_size': (1, 640, 640),

    'dropout': 0.1,  # dropout ratio
    'layer_norm_epsilon': 1e-12,

    # optimization & training parameters
    'n_epoch': 200,  # total training epochs
    'learning_rate': 1e-4,  # initial learning rate
    'weight_decay': 0.001,  # L2 regularization weight
    'momentum': 0.9,  # SGD momentum
    'T_max': 150,  # period (iteration) of cosine learning rate decay
    'eta_min': 1e-6,  # minimum learning rate
    'save_freq': 50,  # save checkpoint frequency (epoch)
    'val_freq': 1,  # validation frequency (epoch)

    # load & save checkpoint
    'snap_path': './weights',  # directory for saving checkpoint
    'checkpoint': None,  # load checkpoint
})


class MyConfig():
    def __init__(self):
        self.config = config
