import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, training=True):
    df = df.copy()

    # Drop unnecessary columns
    drop_cols = ['id', 'Flight']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Split into features and target
    if training and 'Delay' in df.columns:
        X = df.drop('Delay', axis=1)
        y = df['Delay']
        return X, y
    return df
