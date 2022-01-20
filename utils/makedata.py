import re
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

from utils.dfdeal import deal
from utils.tools import k_fold

warnings.filterwarnings("ignore")

featurebase = ['feature13', 'feature15', 'feature19', 'feature60',
               'feature20_mean', 'feature28_mean', 'feature36_mean', 'feature61_mean', 'feature69_mean',
               'feature20_std', 'feature28_std', 'feature36_std', 'feature61_std', 'feature69_std'
               ]

feature = {'Root1': featurebase,
           'Root2': featurebase,
           'Root3': featurebase}


def make_data(frac=0.7, generate=False):
    if generate:
        # get label
        df = pd.read_csv("./data/data/train/train_label.csv")
        label = pd.DataFrame(columns=['ID', 'Root1', 'Root2', 'Root3', 'Root4', 'Root5', 'Root6'])

        for i in range(len(df)):
            idx = re.findall("\d", df['root-cause(s)'].values[i])
            idx = np.array(idx).astype('int')
            label.loc[i, :] = np.hstack((df['sample_index'].values[i], np.sum(np.eye(6)[idx - 1], axis=0)))

        # get labeled data
        columns = ['ID']
        columns.extend(featurebase)

        data = pd.DataFrame(columns=columns)
        for i, idx in enumerate(label['ID'].values):
            df = pd.read_csv('./data/data/train/' + str(int(idx)) + '.csv')
            df = deal(df)
            df = df[featurebase]
            values = [idx]
            for col in featurebase:
                values.append(df[col].mean())
            data.loc[i, :] = values

        data = pd.merge(data, label, on='ID', how='left')

        # up sample
        # upsampledf = upsample(data, n=900)
        # data = pd.concat((data, upsampledf))
        # data = data.reset_index(drop=True)

        # get test data
        columns = ['ID']
        columns.extend(featurebase)

        test_data = pd.DataFrame(columns=columns)
        for i in range(600):
            df = pd.read_csv('./data/data/test/' + str(int(i)) + '.csv')
            df = deal(df)
            df = df[featurebase]
            values = [i]
            for col in featurebase:
                values.append(df[col].mean())
            test_data.loc[i, :] = values

        # get unlabeled data
        columns = ['ID']
        columns.extend(featurebase)
        columns.extend(['Root1', 'Root2', 'Root3', 'Root4', 'Root5', 'Root6'])

        unlabel_data = pd.DataFrame(columns=columns)
        i = 0
        for idx in range(2984):
            if idx not in list(label['ID'].values):
                df = pd.read_csv('./data/data/train/' + str(int(idx)) + '.csv')
                df = deal(df)
                df = df[featurebase]
                values = [idx]
                for col in featurebase:
                    values.append(df[col].mean())
                values.extend([-1, -1, -1, 0, 0, 0])
                unlabel_data.loc[i, :] = values
                i = i + 1

        # scaler
        scaler = StandardScaler()
        scaler.fit(data[data.columns[1:-6]])
        data[data.columns[1:-6]] = scaler.transform(data[data.columns[1:-6]])
        test_data[test_data.columns[1:]] = scaler.transform(test_data[test_data.columns[1:]])
        unlabel_data[unlabel_data.columns[1:-6]] = scaler.transform(unlabel_data[unlabel_data.columns[1:-6]])

        # fill nan
        data[data.columns[1:-6]] = data[data.columns[1:-6]].ffill(0)
        test_data[test_data.columns[1:]] = test_data[test_data.columns[1:]].ffill(0)
        unlabel_data[unlabel_data.columns[1:-6]] = unlabel_data[unlabel_data.columns[1:-6]].ffill(0)

        # label unlabel data
        for root in ['Root1', 'Root2', 'Root3']:
            X = np.vstack((data[feature[root]].values, unlabel_data[feature[root]].values))
            y = np.hstack((data[root].values, unlabel_data[root].values)).astype(np.float64)
            svc = SVC(probability=True, gamma="auto")
            self_training_model = SelfTrainingClassifier(svc)
            self_training_model.fit(X, y)
            pred = self_training_model.predict(unlabel_data[feature[root]].values)
            unlabel_data[root] = pred

        # save
        data.to_csv('./data/data/train_data.csv', index=False)
        test_data.to_csv('./data/data/test_data.csv', index=False)
        unlabel_data.to_csv('./data/data/unlabel_data.csv', index=False)
    else:
        data = pd.read_csv('./data/data/train_data.csv')
        test_data = pd.read_csv('./data/data/test_data.csv')
        unlabel_data = pd.read_csv('./data/data/unlabel_data.csv')

    datas = k_fold(data)

    database = {
        'train': {'0': {'Root1': {'data': pd.concat((datas[1], datas[2], datas[3], datas[4]))[feature['Root1']].values,
                                  'label': pd.concat((datas[1], datas[2], datas[3], datas[4]))['Root1'].values},
                        'Root2': {'data': pd.concat((datas[1], datas[2], datas[3], datas[4]))[feature['Root2']].values,
                                  'label': pd.concat((datas[1], datas[2], datas[3], datas[4]))['Root2'].values},
                        'Root3': {'data': pd.concat((datas[1], datas[2], datas[3], datas[4]))[feature['Root3']].values,
                                  'label': pd.concat((datas[1], datas[2], datas[3], datas[4]))['Root3'].values}},
                  '1': {'Root1': {'data': pd.concat((datas[0], datas[2], datas[3], datas[4]))[feature['Root1']].values,
                                  'label': pd.concat((datas[0], datas[2], datas[3], datas[4]))['Root1'].values},
                        'Root2': {'data': pd.concat((datas[0], datas[2], datas[3], datas[4]))[feature['Root2']].values,
                                  'label': pd.concat((datas[0], datas[2], datas[3], datas[4]))['Root2'].values},
                        'Root3': {'data': pd.concat((datas[0], datas[2], datas[3], datas[4]))[feature['Root3']].values,
                                  'label': pd.concat((datas[0], datas[2], datas[3], datas[4]))['Root3'].values}},
                  '2': {'Root1': {'data': pd.concat((datas[0], datas[1], datas[3], datas[4]))[feature['Root1']].values,
                                  'label': pd.concat((datas[0], datas[1], datas[3], datas[4]))['Root1'].values},
                        'Root2': {'data': pd.concat((datas[0], datas[1], datas[3], datas[4]))[feature['Root2']].values,
                                  'label': pd.concat((datas[0], datas[1], datas[3], datas[4]))['Root2'].values},
                        'Root3': {'data': pd.concat((datas[0], datas[1], datas[3], datas[4]))[feature['Root3']].values,
                                  'label': pd.concat((datas[0], datas[1], datas[3], datas[4]))['Root3'].values}},
                  '3': {'Root1': {'data': pd.concat((datas[0], datas[1], datas[2], datas[4]))[feature['Root1']].values,
                                  'label': pd.concat((datas[0], datas[1], datas[2], datas[4]))['Root1'].values},
                        'Root2': {'data': pd.concat((datas[0], datas[1], datas[2], datas[4]))[feature['Root2']].values,
                                  'label': pd.concat((datas[0], datas[1], datas[2], datas[4]))['Root2'].values},
                        'Root3': {'data': pd.concat((datas[0], datas[1], datas[2], datas[4]))[feature['Root3']].values,
                                  'label': pd.concat((datas[0], datas[1], datas[2], datas[4]))['Root3'].values}},
                  '4': {'Root1': {'data': pd.concat((datas[0], datas[1], datas[2], datas[3]))[feature['Root1']].values,
                                  'label': pd.concat((datas[0], datas[1], datas[2], datas[3]))['Root1'].values},
                        'Root2': {'data': pd.concat((datas[0], datas[1], datas[2], datas[3]))[feature['Root2']].values,
                                  'label': pd.concat((datas[0], datas[1], datas[2], datas[3]))['Root2'].values},
                        'Root3': {'data': pd.concat((datas[0], datas[1], datas[2], datas[3]))[feature['Root3']].values,
                                  'label': pd.concat((datas[0], datas[1], datas[2], datas[3]))['Root3'].values}},
                  },
        'valid': {'0': {'Root1': {'data': datas[0][feature['Root1']].values,
                                  'label': datas[0]['Root1'].values},
                        'Root2': {'data': datas[0][feature['Root2']].values,
                                  'label': datas[0]['Root2'].values},
                        'Root3': {'data': datas[0][feature['Root3']].values,
                                  'label': datas[0]['Root3'].values}},
                  '1': {'Root1': {'data': datas[1][feature['Root1']].values,
                                  'label': datas[1]['Root1'].values},
                        'Root2': {'data': datas[1][feature['Root2']].values,
                                  'label': datas[1]['Root2'].values},
                        'Root3': {'data': datas[1][feature['Root3']].values,
                                  'label': datas[2]['Root3'].values}},
                  '2': {'Root1': {'data': datas[2][feature['Root1']].values,
                                  'label': datas[2]['Root1'].values},
                        'Root2': {'data': datas[2][feature['Root2']].values,
                                  'label': datas[2]['Root2'].values},
                        'Root3': {'data': datas[2][feature['Root3']].values,
                                  'label': datas[2]['Root3'].values}},
                  '3': {'Root1': {'data': datas[3][feature['Root1']].values,
                                  'label': datas[3]['Root1'].values},
                        'Root2': {'data': datas[3][feature['Root2']].values,
                                  'label': datas[3]['Root2'].values},
                        'Root3': {'data': datas[3][feature['Root3']].values,
                                  'label': datas[3]['Root3'].values}},
                  '4': {'Root1': {'data': datas[4][feature['Root1']].values,
                                  'label': datas[4]['Root1'].values},
                        'Root2': {'data': datas[4][feature['Root2']].values,
                                  'label': datas[4]['Root2'].values},
                        'Root3': {'data': datas[4][feature['Root3']].values,
                                  'label': datas[4]['Root3'].values}},
                  },
        'test': {'Root1': {'data': test_data[feature['Root1']].values,
                           'label': np.zeros(len(test_data))},
                 'Root2': {'data': test_data[feature['Root2']].values,
                           'label': np.zeros(len(test_data))},
                 'Root3': {'data': test_data[feature['Root3']].values,
                           'label': np.zeros(len(test_data))}},
        'unlabel': {'Root1': {'data': unlabel_data[feature['Root1']].values,
                              'label': unlabel_data['Root1'].values},
                    'Root2': {'data': unlabel_data[feature['Root2']].values,
                              'label': unlabel_data['Root2'].values},
                    'Root3': {'data': unlabel_data[feature['Root3']].values,
                              'label': unlabel_data['Root3'].values}}
    }
    return database
