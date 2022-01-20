import numpy as np
import pandas as pd


def get_mean_distance(points):
    data = [[0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31]]

    index = {}
    for i in range(len(data)):
        for j in range(len(data[0])):
            index[data[i][j]] = np.array([i, j])

    try:
        dis = []
        for i in range(len(points)):
            for j in range(len(points) - i - 1):
                dis.append(np.sqrt(np.sum((index[points[i]] - index[points[i + j + 1]]) ** 2)))
        return np.mean(dis), np.std(dis)
    except:
        return np.NaN, np.NaN


def k_fold(df, k=5):
    n = len(df) // k

    dfs = []
    for i in range(k):
        a = df.sample(n=n, replace=False)
        df = df[~df.index.isin(a.index)]
        dfs.append(a)
    return dfs


def fillna_with_mean(df):
    for column in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[column].mean()
        df[column] = df[column].fillna(mean_val)
    return df


def upsample(df, n=10):
    new = pd.DataFrame(columns=df.columns)

    for i in range(n):
        a = df[df['Root2'] == 1].sample(n=1, replace=True)
        if (a['Root2'] == 1).all() and (a['Root3'] == 1).all():
            b = df[(df['Root2'] == 1) & (df['Root3'] == 1)].sample(n=1, replace=True)
        else:
            b = df[(df['Root2'] == 1) & (df['Root3'] == 0)].sample(n=1, replace=True)
        c = np.random.rand()
        value = c * a.values + (1 - c) * b.values
        new.loc[i, :] = value
    return new
