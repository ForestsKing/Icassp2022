import numpy as np
import pandas as pd
import torch

from exp.exp import Exp
from utils.makedata import make_data
from utils.setseed import set_seed

if __name__ == "__main__":
    set_seed(42)

    batch_size = 32
    epochs = 100
    patience = 20
    iters = 1
    lr = 0.0001

    rootpath = './'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for iter in range(iters):
        database = make_data(generate=True)
        for k in range(5):
            exp = Exp(iter, epochs, batch_size, patience, lr, rootpath, device, k, database)
            exp.pretrain()
            exp.valid(load=False)
            exp.train()
            exp.valid(load=False)
            exp.test(load=False)
            print('=====================================================================')

    result = pd.DataFrame(columns=['ID', 'Root1', 'Root2', 'Root3', 'Root4', 'Root5', 'Root6'])
    df0 = pd.read_csv('./submit_iter0_fold0.csv')[['Root1', 'Root2', 'Root3', 'Root4', 'Root5', 'Root6']]
    df1 = pd.read_csv('./submit_iter0_fold1.csv')[['Root1', 'Root2', 'Root3', 'Root4', 'Root5', 'Root6']]
    df2 = pd.read_csv('./submit_iter0_fold2.csv')[['Root1', 'Root2', 'Root3', 'Root4', 'Root5', 'Root6']]
    df3 = pd.read_csv('./submit_iter0_fold3.csv')[['Root1', 'Root2', 'Root3', 'Root4', 'Root5', 'Root6']]
    df4 = pd.read_csv('./submit_iter0_fold4.csv')[['Root1', 'Root2', 'Root3', 'Root4', 'Root5', 'Root6']]

    for i in range(600):
        v0 = df0.iloc[i, :].values
        v1 = df1.iloc[i, :].values
        v2 = df2.iloc[i, :].values
        v3 = df3.iloc[i, :].values
        v4 = df4.iloc[i, :].values

        v = v0 + v1 + v2 + v3 + v4

        value = [i]
        value.extend((v > 0).astype(np.int64))
        result.loc[i, :] = value

    result.to_csv('submit.csv', index=False)
