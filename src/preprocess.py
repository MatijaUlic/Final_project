import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, training=True):
    df = df.copy()
    drop_cols = ['id', 'Flight']
    df.drop(columns=[c for c in drop_cols if c in df], inplace=True, errors='ignore')
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    if training and 'Delay' in df.columns:
        return df.drop('Delay', axis=1), df['Delay']
    return df