import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})
    X = df['Message']
    Y = df['Category']
    return X, Y