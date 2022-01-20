import numpy as np

from utils.tools import get_mean_distance

feature20 = ['feature20_0', 'feature20_1', 'feature20_2', 'feature20_3', 'feature20_4', 'feature20_5', 'feature20_6', 'feature20_7']
feature28 = ['feature28_0', 'feature28_1', 'feature28_2', 'feature28_3', 'feature28_4', 'feature28_5', 'feature28_6', 'feature28_7']
feature36 = ['feature36_0', 'feature36_1', 'feature36_2', 'feature36_3', 'feature36_4', 'feature36_5', 'feature36_6', 'feature36_7']
feature61 = ['feature61_0', 'feature61_1', 'feature61_2', 'feature61_3', 'feature61_4', 'feature61_5', 'feature61_6', 'feature61_7']
feature69 = ['feature69_0', 'feature69_1', 'feature69_2', 'feature69_3', 'feature69_4', 'feature69_5', 'feature69_6', 'feature69_7']

def deal(df):
    df['feature60'] = df['feature60'].apply(lambda x: np.array(str(x).split(';')).astype(np.float64).mean())

    df['feature20_mean'] = df[feature20].apply(lambda x: get_mean_distance(x)[0], axis=1)
    df['feature28_mean'] = df[feature28].apply(lambda x: x.mean(), axis=1)
    df['feature36_mean'] = df[feature36].apply(lambda x: x.mean(), axis=1)
    df['feature61_mean'] = df[feature61].apply(lambda x: x.mean(), axis=1)
    df['feature69_mean'] = df[feature69].apply(lambda x: x.mean(), axis=1)

    df['feature20_std'] = df[feature20].apply(lambda x: get_mean_distance(x)[1], axis=1)
    df['feature28_std'] = df[feature28].apply(lambda x: x.std(), axis=1)
    df['feature36_std'] = df[feature36].apply(lambda x: x.std(), axis=1)
    df['feature61_std'] = df[feature61].apply(lambda x: x.std(), axis=1)
    df['feature69_std'] = df[feature69].apply(lambda x: x.std(), axis=1)
    return df