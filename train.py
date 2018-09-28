from tools import *
from model_manager import *
from .chord_model import ChordModel


def train(dataset_name):
    dataset = dataset_from_name(dataset_name=dataset_name)

    rnn_type = 'LSTM'
    n_features = 15
    n_hidden = 100
    n_tokens = 12
    n_layers = 2

    print('Creating ChordModel...', end='', flush=True)
    chord_model = ChordModel(rnn_type=rnn_type,
                             n_features=n_features,
                             n_hidden=n_hidden,
                             n_tokens=n_tokens,
                             n_layers=n_layers
                             )
    print('Done.')

    batch_size = 128
    batches_per_epoch = 100
    print('Defining a model manager...')
    model_manager = ModelManager(model=chord_model,
                                 dataset=dataset,
                                 loss_name='BCE',
                                 optimizer_name='adam',
                                 lr=1e-3)

    # load saved version if any
    model_manager.load_if_saved()

    # training
    model_manager.train_model(batch_size=batch_size,
                              batches_per_epoch=batches_per_epoch,
                              num_epochs=1000,
                              )

    # solve the problem
    try:
        model_manager.solve_melodies()
    except:
        pass


if __name__ == '__main__':
    train('wikifonia-db_full')