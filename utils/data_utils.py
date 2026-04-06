import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_dataset(root: str, name: str, label_col: str = 'Label'):
    splits = {}
    for split in ['train_set', 'crossval_set', 'test_set']:
        path = os.path.join(root, f'{split}.parquet')
        if not os.path.exists(path):
            raise FileNotFoundError(f'[load_dataset] Missing file: {path}')

        df = pd.read_parquet(path)
        df.columns = df.columns.str.strip()

        if label_col not in df.columns:
            match = [c for c in df.columns if c.lower() == label_col.lower()]
            if match:
                df.rename(columns={match[0]: label_col}, inplace=True)
            else:
                raise KeyError(f'Label column "{label_col}" not found in {path}.')

        splits[split] = df
        print(f'  {name}/{split}: {df.shape[0]:>10,} rows x {df.shape[1]} cols')

    train_df = pd.concat([splits['train_set'], splits['crossval_set']], ignore_index=True)
    test_df = splits['test_set']
    return train_df, test_df

def prepare_Xy(train_df, test_df, features, label_col='Label', shared_classes=None):
    if shared_classes is not None:
        train_df = train_df[train_df[label_col].isin(shared_classes)].copy()
        test_df  = test_df[test_df[label_col].isin(shared_classes)].copy()

    X_train_raw = train_df[features].values.astype(np.float64)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col].values)

    # Filter test set to only include labels seen during training
    mask = test_df[label_col].isin(le.classes_)
    if not mask.all():
        test_df = test_df[mask].copy()
    
    X_test_raw = test_df[features].values.astype(np.float64)
    X_test = scaler.transform(X_test_raw)
    y_test = le.transform(test_df[label_col].values)

    print(f'  Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows')
    print(f'  Classes ({len(le.classes_)}): {list(le.classes_)}')
    return X_train, X_test, y_train, y_test, le