import pandas as pd
import numpy as np
from sklearn import preprocessing

def load_dataframe(dataset=None, info=False):

    if dataset in ['CricketX', 'UWaveGestureLibraryAll', 'InsectWingbeatSound']:

        if info: print(f'Dataset: {dataset}')

        train = np.loadtxt(f'./Datasets/{dataset}/{dataset}_TRAIN.tsv').astype(np.float32)
        test = np.loadtxt(f'./Datasets/{dataset}/{dataset}_TEST.tsv').astype(np.float32)

        X_train = train[:, 1:]
        y_train = np.squeeze(train[:, 0:1],axis=1).astype(np.int64)
        X_test = test[:, 1:]
        y_test= np.squeeze(test[:, 0:1],axis=1).astype(np.int64)

        encoder = preprocessing.LabelEncoder()
        encoder.fit(y_train)
        y_train = encoder.transform(y_train)
        encoder.fit(y_test)
        y_test = encoder.transform(y_test)

        attributes = [f'X{i}' for i in range(len(X_train[0]))]
   
        df_Xtrain = pd.DataFrame(X_train, columns=attributes)
        df_ytrain = pd.DataFrame(y_train, columns=['target'])
        df_train = pd.concat([df_Xtrain, df_ytrain], axis = 1)

        df_Xtest = pd.DataFrame(X_test, columns=attributes)
        df_ytest = pd.DataFrame(y_test, columns=['target'])
        df_test = pd.concat([df_Xtest, df_ytest], axis = 1)

        num_classes = len(set(df_ytrain['target']))

        return df_train, df_test, num_classes
    
    elif dataset in ['MFPT', 'XJTU']:
        x = np.load(f"./Datasets/{dataset}/{dataset}_data.npy")
        y = np.load(f"./Datasets/{dataset}/{dataset}_label.npy")

        encoder = preprocessing.LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)

        attributes = [f'X{i}' for i in range(len(x[0]))]

        df_X = pd.DataFrame(x, columns=attributes)
        df_y = pd.DataFrame(y, columns=['target'])
        df = pd.concat([df_X, df_y], axis=1)
        df = df.sample(frac=1).reset_index().drop('index', axis='columns')

        l_train = int(np.floor(len(df)*(1/2)))

        df_train = df.iloc[0:l_train, :].reset_index().drop('index', axis='columns')
        df_test = df.iloc[l_train:, :].reset_index().drop('index', axis='columns')

        num_classes = len(set(df_y['target']))

        return df_train, df_test, num_classes

    elif dataset == 'EpilepticSeizure':

        df = pd.read_csv('Datasets/EpilepticSeizure/data.csv').drop('X0', axis='columns').rename(columns={'y':'target'})
        df_X = df.iloc[:, 0:len(df.columns)-1].astype('float32')
        df_y = df.iloc[:, len(df.columns)-1:]
        df = pd.concat([df_X, df_y], axis=1)

        l_train = int(np.floor(len(df)*(1/2)))
        df_train = df.iloc[0:l_train, :].reset_index().drop('index', axis='columns').astype('float32')
        df_test = df.iloc[l_train:, :].reset_index().drop('index', axis='columns')

        num_classes = len(set(df['target']))

        return df_train, df_test, num_classes