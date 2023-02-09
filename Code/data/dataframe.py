import pandas as pd
import numpy as np
from sklearn import preprocessing

def load_dataframe(dataset = None, info=False):

    if info: print(f'Dataset: {dataset}')

    #load dataset
    train = np.loadtxt(f'./Datasets/{dataset}/{dataset}_TRAIN.tsv').astype(np.float32)
    test = np.loadtxt(f'./Datasets/{dataset}/{dataset}_TEST.tsv').astype(np.float32)

    #divide dataset in attributes and labels
    X_train = train[:, 1:] # each row, columns from 1 to n
    y_train = np.squeeze(train[:, 0:1],axis=1).astype(np.int64) # each row, columns 0
    X_test = test[:, 1:] # each row, columns from 1 to n
    y_test= np.squeeze(test[:, 0:1],axis=1).astype(np.int64) # each row, columns 0

    #encodes labels between 0 and n-1
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    encoder.fit(y_test)
    y_test = encoder.transform(y_test)

    #create a list of name for attributes
    attributes = [f'att{i}' for i in range(len(X_train[0]))]

    #create a pandas dataframe for train set    
    df_Xtrain = pd.DataFrame(X_train, columns=attributes)
    df_ytrain = pd.DataFrame(y_train, columns=['target'])
    df_train = pd.concat([df_Xtrain, df_ytrain], axis = 1)

    #create a pandas dataframe for test set
    df_Xtest = pd.DataFrame(X_test, columns=attributes)
    df_ytest = pd.DataFrame(y_test, columns=['target'])
    df_test = pd.concat([df_Xtest, df_ytest], axis = 1)

    num_classes = len(set(df_ytrain['target']))

    return df_train, df_test, num_classes