import itertools
import os
import sys
from datetime import datetime
from itertools import islice
import torch.nn as nn
import collections
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.utils as utils
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Dataset
from tqdm import tqdm
from chord_model import ChordModel
from optimizers import optimizer_from_name
# from main_instance import CUDA_ENABLED, args
from loss import loss_from_name
from tools import *
from itertools import accumulate


CUDA_ENABLED = torch.cuda.is_available()


class ModelManager:

    def __init__(self, model: ChordModel, dataset, loss_name: str='cross_entropy', optimizer_name: str='adam', lr=1e-3):
        self.model = model
        if CUDA_ENABLED:
            print('Moving the model to GPU...', end='', flush=True)
            self.model.cuda()
            print('Done.')

        self.optimizer_name = optimizer_name
        self.optimizer = optimizer_from_name(optimizer_name, lr=lr)(self.model.parameters())
        self.lr = lr
        if self.optimizer_name == 'GD':
            self.lr = 40

        self.loss = loss_from_name(loss_name)
        self.dataset = dataset

        #TODO : implement note_look_up_table

    def load_if_saved(self):
        self.model.load()

    def save(self):
        self.model.save()

    def loss_and_acc_on_epoch(self, batch_size, batches_per_epoch, data_loader, train=True):
        generator = None
        mean_loss = 0.
        n_correct_chords = 0.

        hidden = self.model.init_hidden(batch_size)
        hidden = (wrap_cuda(hidden[0]), wrap_cuda(hidden[1]))

        for input_melody, chord_target in tqdm(islice(data_loader, batches_per_epoch)):
            input_melody = wrap_cuda(input_melody)
            input_melody = input_melody.transpose(0, 1)
            chord_target = wrap_cuda(chord_target)

            hidden = repackage_hidden(hidden)
            self.optimizer.zero_grad()
            output, hidden = self.model(input_melody, hidden)

            loss = self.mean_loss(output, chord_target)

            if train:
                loss.backward()
                utils.clip_grad_norm_(self.model.parameters(), 0.25)
                if self.optimizer_name != 'GD':
                    self.optimizer.step()
                else:
                    for p in self.model.parameters():
                        p.data.add_(-self.lr, p.grad.data)

            notes_accuracy, correct_chords = self.accuracy(output_chord=output, chord_target=chord_target)

            mean_loss += loss.mean()
            n_correct_chords += correct_chords

        return mean_loss / batches_per_epoch, n_correct_chords / batches_per_epoch

    def mean_loss(self, chord_target, chord_output):
        # TODO: check there is no problem with reduction='elementwise_mean'
        assert chord_output.size()[0] == chord_target.size()[0]
        return self.loss(chord_target, chord_output)

    def accuracy(self, output_chord, chord_target):
        # TODO : CAN BE LOGSOFTMAX
        # f = nn.LogSoftmax()
        f = nn.Softmax()
        log_probabilities = f(output_chord)
        decoded_chord = torch.argmax(log_probabilities, dim=0)
        correct_chords = (decoded_chord == chord_target).sum().item()
        mean_correct_chords = correct_chords/output_chord.size()[0]
        return correct_chords * 100

    def prepare_data(self, batch_size, test_batch_size, **kwargs):

        class Subset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __getitem__(self, index):
                return self.dataset[self.indices[index]]

            def __len__(self):
                return len(self.indices)

            def data_augmentation(self):
                c = 0
                l = len(self.dataset)
                print('Augmenting dataset...')
                for index in tqdm(self.indices):
                    for i in range(1, 12):
                        self.dataset.append(data_transpose(self.dataset[i][0], self.dataset[i][1], transposition=i))
                        c += 1
                assert c == 11 * len(self.indices)
                new_indices = wrap_cuda(torch.LongTensor(range(l, l + c)))
                self.indices = torch.cat((self.indices, new_indices))
                self.indices = wrap_cuda(self.indices)
                print('Done.')

        # TODO try and compare accuracy if we also augment validation
        def random_split(dataset, lengths):
            assert sum(lengths) == len(dataset)
            indices = torch.randperm(sum(lengths))
            indices = wrap_cuda(indices)
            list_ = [None] * 3
            offset = 0
            for i, length in enumerate(lengths):
                offset += length
                list_[i] = Subset(dataset, indices[offset - length : offset])
                if i == 0:
                    list_[i].data_augmentation()
            return list_[0], list_[1], list_[2]

        def pad_melody(melody, max_len):
            padded_melody = [None] * max_len
            padded_melody[:len(melody)] = melody
            if len(melody) < max_len:
                padded_melody[len(melody):] = ['PAD'] * (max_len - len(melody))
            return padded_melody


        # TODO CHECK CHORD_TO_BATCH

        def collate_batch(batch):
            max_len = 0
            for sequence in batch:
                if len(sequence[0]) > max_len:
                    max_len = len(sequence[0])
            data = [pad_melody(sequence[0], max_len) for sequence in batch]
            data = melody_to_batch(data, 15, self.note_look_up_table)
            target = [sequence[1] for sequence in batch]
            target = chord_to_batch_0(target, self.model.n_tokens)
            target = wrap_cuda(target.type('torch.FloatTensor'))
            return [data, target]

        num_melodies = len(self.dataset)
        a = int(85 * num_melodies / 100)
        b = int(10 * num_melodies / 100)
        c = num_melodies - (a + b)

        lengths = [a, b, c]
        train_dataset, validation_dataset, test_dataset = random_split(self.dataset, lengths)

        # DEBUG: Check that the pointer is the same
        assert train_dataset.dataset is validation_dataset.dataset
        assert validation_dataset.dataset is test_dataset.dataset

        # dataloaders
        # Todo fix problem with worker shutdown
        #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, collate_fn=collate_batch, **kwargs)
        #validation_loader = DataLoader(validation_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, num_workers=8, collate_fn=collate_batch, **kwargs)
        #test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True, num_workers=8, collate_fn=collate_batch, **kwargs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_batch, **kwargs)
        validation_loader = DataLoader(validation_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, collate_fn=collate_batch, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True, collate_fn=collate_batch, **kwargs)

        return train_loader, validation_loader, test_loader

    def train_model(self, batch_size, batches_per_epoch, num_epochs, plot=False, save_every=2):
        train_loader, validation_loader, test_loader = self.prepare_data(batch_size, test_batch_size=batch_size//2)
        best_loss = 1e3
        c = 0
        for epoch_index in range(num_epochs):
            self.model.train()
            mean_loss, chord_accuracy = self.loss_and_acc_on_epoch(batch_size=batch_size, batches_per_epoch=batches_per_epoch, data_loader=train_loader, train=True)

            self.model.eval()
            val_mean_loss, val_chord_accuracy = self.loss_and_acc_on_epoch(batch_size=batch_size//2, batches_per_epoch=batches_per_epoch, data_loader=validation_loader, train=False)

            print(
                f'Train Epoch: {epoch_index}/{num_epochs} \t'
                f'Loss: {mean_loss}\t'
                f'Chord prediction Accuracy: {chord_accuracy} %\t'
            )
            if self.optimizer_name == 'GD':
                print(f'Learning Rate: {self.lr}')
            print(
                f'Validation Loss: {val_mean_loss} \t'
                f'Validation Chord Accuracy: {val_chord_accuracy} %'
            )
            print('-' * 100)

            if val_mean_loss < best_loss:
                best_loss = val_mean_loss
                self.model.save()
                self.solve_melodies()
                if self.optimizer_name == 'GD':
                    c -= 2
            elif self.optimizer_name == 'GD':
                c += 1
                if c == 5:
                    self.lr = self.lr / 2
                    c = 0

        print('End of training.')

        print('Testing the model...')
        self.model.eval()
        test_mean_loss, test_chord_accuracy = self.loss_and_acc_on_epoch(batch_size=batch_size//2, batches_per_epoch=batches_per_epoch, data_loader=test_loader, train=False)
        print('*' * 100)
        print(
            f'Test Loss: {test_mean_loss}'
            f'Test Chord Accuracy: {test_chord_accuracy * 100} %'
        )

    def solve_melodies(self):
        sigm = nn.Sigmoid()
        IN_PATH = os.getcwd()
        OUT_PATH = os.getcwd() + '/evaluation/results.txt'
        with open(IN_PATH+'/evaluation/solve_melodies.txt', 'r') as f:
            file = f.read()
        melodies = file.split('\n')[:-1]
        for melody in melodies:
            l_ = []
            melody_list = melody.split('/')
            for elt in melody_list:
                l_.append((elt.split(' ')[0], elt.split(' ')[-1]))
            melody_tensor = convert_sequence(l_)
            hidden = self.model.init_hidden(1)
            output, _ = self.model(melody_tensor, hidden)
            predicted_chord = decode_chord_0(output)
            self.write_output_chord_to_file(OUT_PATH, melody, predicted_chord)
        with open(OUT_PATH, 'a') as f:
            f.write('=' * 10 + '\n')
            f.close()

    def write_output_chord_to_file(self, path, melody, predicted_chord):
        PATH = os.getcwd() + '/evaluation/results.txt'
        with open(PATH, 'a') as f:
            s = str(melody) + ': ' + one_hot_to_string(predicted_chord) + '\n'
            f.write(s)
            f.close()
