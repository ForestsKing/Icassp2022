import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from data.dataset import Dataset
from model.model import Model
from utils.earlystopping import EarlyStopping
from utils.score import get_score


class Exp:
    def __init__(self, iter, epochs, batch_size, patience, lr, rootpath, device, k, database):
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.rootpath = rootpath
        self.device = device
        self.iter = iter
        self.k = k
        self.database = database

        self.savepath = self.rootpath + 'checkpoint/checkpoint_iter' + str(self.iter) + '_fold' + str(self.k) + '_'

        self._get_data(self.database, shuffle=True)
        self._get_model()

        if not os.path.exists(self.rootpath + 'checkpoint/'):
            os.makedirs(self.rootpath + 'checkpoint/')

    def _get_data(self, database, shuffle=True):
        self.dataset = {
            'train': {'Root1': Dataset(database['train'][str(self.k)]['Root1']['data'],
                                       database['train'][str(self.k)]['Root1']['label']),
                      'Root2': Dataset(database['train'][str(self.k)]['Root2']['data'],
                                       database['train'][str(self.k)]['Root2']['label']),
                      'Root3': Dataset(database['train'][str(self.k)]['Root3']['data'],
                                       database['train'][str(self.k)]['Root3']['label'])},
            'valid': {'Root1': Dataset(database['valid'][str(self.k)]['Root1']['data'],
                                       database['valid'][str(self.k)]['Root1']['label']),
                      'Root2': Dataset(database['valid'][str(self.k)]['Root2']['data'],
                                       database['valid'][str(self.k)]['Root2']['label']),
                      'Root3': Dataset(database['valid'][str(self.k)]['Root3']['data'],
                                       database['valid'][str(self.k)]['Root3']['label'])},
            'test': {'Root1': Dataset(database['test']['Root1']['data'], database['test']['Root1']['label']),
                     'Root2': Dataset(database['test']['Root2']['data'], database['test']['Root2']['label']),
                     'Root3': Dataset(database['test']['Root3']['data'], database['test']['Root3']['label'])},
            'unlabel': {'Root1': Dataset(database['unlabel']['Root1']['data'], database['unlabel']['Root1']['label']),
                        'Root2': Dataset(database['unlabel']['Root2']['data'], database['unlabel']['Root2']['label']),
                        'Root3': Dataset(database['unlabel']['Root3']['data'], database['unlabel']['Root3']['label'])}
        }
        self.dataloader = {
            'train': {'Root1': DataLoader(self.dataset['train']['Root1'], batch_size=self.batch_size, shuffle=shuffle),
                      'Root2': DataLoader(self.dataset['train']['Root2'], batch_size=self.batch_size, shuffle=shuffle),
                      'Root3': DataLoader(self.dataset['train']['Root3'], batch_size=self.batch_size, shuffle=shuffle)},
            'valid': {'Root1': DataLoader(self.dataset['valid']['Root1'], batch_size=self.batch_size, shuffle=False),
                      'Root2': DataLoader(self.dataset['valid']['Root2'], batch_size=self.batch_size, shuffle=False),
                      'Root3': DataLoader(self.dataset['valid']['Root3'], batch_size=self.batch_size, shuffle=False)},
            'test': {'Root1': DataLoader(self.dataset['test']['Root1'], batch_size=self.batch_size, shuffle=False),
                     'Root2': DataLoader(self.dataset['test']['Root2'], batch_size=self.batch_size, shuffle=False),
                     'Root3': DataLoader(self.dataset['test']['Root3'], batch_size=self.batch_size, shuffle=False)},
            'unlabel': {
                'Root1': DataLoader(self.dataset['unlabel']['Root1'], batch_size=self.batch_size, shuffle=shuffle),
                'Root2': DataLoader(self.dataset['unlabel']['Root2'], batch_size=self.batch_size, shuffle=shuffle),
                'Root3': DataLoader(self.dataset['unlabel']['Root3'], batch_size=self.batch_size, shuffle=shuffle)}

        }

        print('train: {0}, valid: {1}, test: {2} unlabel: {3}'.format(len(self.dataset['train']['Root1']),
                                                                      len(self.dataset['valid']['Root1']),
                                                                      len(self.dataset['test']['Root1']),
                                                                      len(self.dataset['unlabel']['Root1'])))

    def _get_model(self):
        self.model = {
            'Root1': Model(d_input=14).to(self.device),
            'Root2': Model(d_input=14).to(self.device),
            'Root3': Model(d_input=14).to(self.device)
        }

        self.criterion = {
            'Root1': nn.CrossEntropyLoss(),
            'Root2': nn.CrossEntropyLoss(),
            'Root3': nn.CrossEntropyLoss()
        }

        self.preoptimizer = {
            'Root1': optim.Adam(self.model['Root1'].parameters(), lr=self.lr, weight_decay=1e-4),
            'Root2': optim.Adam(self.model['Root2'].parameters(), lr=self.lr, weight_decay=1e-4),
            'Root3': optim.Adam(self.model['Root3'].parameters(), lr=self.lr, weight_decay=1e-4)
        }

        self.optimizer = {
            'Root1': optim.Adam(self.model['Root1'].parameters(), lr=self.lr, weight_decay=1e-4),
            'Root2': optim.Adam(self.model['Root2'].parameters(), lr=self.lr, weight_decay=1e-4),
            'Root3': optim.Adam(self.model['Root3'].parameters(), lr=self.lr, weight_decay=1e-4)
        }

        self.earlystopping = {
            'Root1': EarlyStopping(patience=self.patience),
            'Root2': EarlyStopping(patience=self.patience),
            'Root3': EarlyStopping(patience=self.patience)
        }

        self.scheduler = {
            'Root1': LambdaLR(self.optimizer['Root1'], lr_lambda=lambda epoch: 0.5 ** ((epoch - 1) // 5)),
            'Root2': LambdaLR(self.optimizer['Root1'], lr_lambda=lambda epoch: 0.5 ** ((epoch - 1) // 5)),
            'Root3': LambdaLR(self.optimizer['Root1'], lr_lambda=lambda epoch: 0.5 ** ((epoch - 1) // 5))
        }

    def _process_one_batch(self, batch_x, batch_y, root):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.to(torch.int64).to(self.device)

        out = self.model[root](batch_x)
        loss = self.criterion[root](out, batch_y)
        return out, loss

    def pretrain(self):
        for root in self.model.keys():
            for e in range(self.epochs):
                self.model[root].train()
                train_loss = []
                for (batch_x, batch_y) in self.dataloader['unlabel'][root]:
                    self.preoptimizer[root].zero_grad()
                    out, loss = self._process_one_batch(batch_x, batch_y, root)
                    train_loss.append(loss.item())
                    loss.backward()
                    self.preoptimizer[root].step()
                self.model[root].eval()
                valid_loss = []
                for (batch_x, batch_y) in self.dataloader['valid'][root]:
                    out, loss = self._process_one_batch(batch_x, batch_y, root)
                    valid_loss.append(loss.item())
                train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
                if (e + 1) % 5 == 0:
                    print("Iter {0} Fold {1} {2} Epoch {3} || Train Loss: {4:.6f} Vali Loss: {5:.6f}".format(self.iter,
                                                                                                             self.k,
                                                                                                             root,
                                                                                                             e + 1,
                                                                                                             train_loss,
                                                                                                             valid_loss))

    def train(self):
        for root in self.model.keys():
            for e in range(self.epochs):
                self.model[root].train()

                train_loss = []
                for (batch_x, batch_y) in self.dataloader['train'][root]:
                    self.optimizer[root].zero_grad()
                    out, loss = self._process_one_batch(batch_x, batch_y, root)
                    train_loss.append(loss.item())
                    loss.backward()
                    self.optimizer[root].step()

                self.model[root].eval()
                valid_loss = []
                for (batch_x, batch_y) in self.dataloader['valid'][root]:
                    out, loss = self._process_one_batch(batch_x, batch_y, root)
                    valid_loss.append(loss.item())

                train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
                print("Iter {0} Fold {1} {2} Epoch {3} || Train Loss: {4:.6f} Vali Loss: {5:.6f}".format(self.iter,
                                                                                                         self.k, root,
                                                                                                         e + 1,
                                                                                                         train_loss,
                                                                                                         valid_loss))

                self.earlystopping[root](valid_loss, self.model[root], self.savepath + root + '.pkl')
                if self.earlystopping[root].early_stop:
                    print("Iter {0} Fold {1} {2} is Early stopping!".format(self.iter, self.k, root))
                    break
                self.scheduler[root].step()
            self.model[root].load_state_dict(torch.load(self.savepath + root + '.pkl'))

    def valid(self, load=False, riter=0):
        self._get_data(self.database, shuffle=False)
        result = {
            'train': {'Root1': {'out': [], 'loss': [], 'true': []},
                      'Root2': {'out': [], 'loss': [], 'true': []},
                      'Root3': {'out': [], 'loss': [], 'true': []}},
            'valid': {'Root1': {'out': [], 'loss': [], 'true': []},
                      'Root2': {'out': [], 'loss': [], 'true': []},
                      'Root3': {'out': [], 'loss': [], 'true': []}},
        }

        for root in self.model.keys():
            if load:
                readpath = self.rootpath + 'checkpoint/checkpoint_iter' + str(riter) + '_fold' + str(
                    self.k) + '_' + root + '.pkl'
                self.model[root].load_state_dict(torch.load(readpath))

            self.model[root].eval()

            for state in ['train', 'valid']:
                for (batch_x, batch_y) in self.dataloader[state][root]:
                    out, loss = self._process_one_batch(batch_x, batch_y, root)
                    result[state][root]['loss'].append(loss.item())
                    result[state][root]['true'].extend(batch_y.detach().cpu().numpy())
                    result[state][root]['out'].extend(out.detach().cpu().numpy().argmax(axis=1))
                result[state][root]['loss'] = np.average(result[state][root]['loss'])
                result[state][root]['true'] = np.array(result[state][root]['true'])
                result[state][root]['out'] = np.array(result[state][root]['out'])
                print(
                    "Iter {0} Fold {1} {2} {3} || loss: {4:.6f} accuracy: {5:.6f} precision: {6:.6f} recall: {7:.6f} f1: {8:.6f}".format(
                        self.iter, self.k,
                        root, state, result[state][root]['loss'],
                        accuracy_score(result[state][root]['true'], result[state][root]['out']),
                        precision_score(result[state][root]['true'], result[state][root]['out']),
                        recall_score(result[state][root]['true'], result[state][root]['out']),
                        f1_score(result[state][root]['true'], result[state][root]['out'])))

        for state in ['train', 'valid']:
            outs = np.hstack((result[state]['Root1']['out'].reshape(-1, 1),
                              result[state]['Root2']['out'].reshape(-1, 1),
                              result[state]['Root3']['out'].reshape(-1, 1)))
            trues = np.hstack((result[state]['Root1']['true'].reshape(-1, 1),
                               result[state]['Root2']['true'].reshape(-1, 1),
                               result[state]['Root3']['true'].reshape(-1, 1)))
            score = get_score(outs, trues)

            print('Iter {0} Fold {1} {2} score: {3:.6f}'.format(self.iter, self.k, state, score))

    def test(self, load=False, riter=0):
        result = {'Root1': [],
                  'Root2': [],
                  'Root3': []
                  }

        for root in self.model.keys():
            if load:
                readpath = self.rootpath + 'checkpoint/checkpoint_iter' + str(riter) + '_fold' + str(
                    self.k) + '_' + root + '.pkl'
                self.model[root].load_state_dict(torch.load(readpath))

            self.model[root].eval()
            for (batch_x, batch_y) in self.dataloader['test'][root]:
                out, loss = self._process_one_batch(batch_x, batch_y, root)
                result[root].extend(out.detach().cpu().numpy().argmax(axis=1))
            result[root] = np.array(result[root])

        outs = np.hstack((result['Root1'].reshape(-1, 1),
                          result['Root2'].reshape(-1, 1),
                          result['Root3'].reshape(-1, 1)))

        resultdf = pd.DataFrame(columns=['ID', 'Root1', 'Root2', 'Root3', 'Root4', 'Root5', 'Root6'])
        for i in range(len(outs)):
            value = [i]
            value.extend(list(outs[i]))
            value.extend([0, 0, 0])
            resultdf.loc[i, :] = value
        resultdf.to_csv('submit_iter' + str(self.iter) + '_fold' + str(self.k) + '.csv', index=False)
