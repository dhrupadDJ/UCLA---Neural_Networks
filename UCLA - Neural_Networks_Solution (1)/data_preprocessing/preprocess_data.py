import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def convert_target(df):
    df['Admit_Chance'] = (df['Admit_Chance'] >= 0.8).astype(int)
    return df

def drop_unnecessary_columns(df, columns):
    df = df.drop(columns=columns, axis=1)
    return df

def create_dummies(df, columns):
    df = pd.get_dummies(df, columns=columns)
    return df

def split_data(df, target_column, test_size=0.2, random_state=123):
    x = df.drop(target_column, axis=1)
    y = df[target_column]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return xtrain, xtest, ytrain, ytest

def scale_data(xtrain, xtest):
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    Xtrain = scaler.transform(xtrain)
    Xtest = scaler.transform(xtest)
    return Xtrain, Xtest
