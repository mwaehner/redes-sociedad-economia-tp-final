import numpy as np
import pandas as pd
import torch
import os

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def normalize_columns(df, cols):
    x = df[cols].values.astype(np.float64)
    standard_scaler = StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    df[cols] = x_scaled


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return res

class StudentsDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 output_col: str = "G3",
                 ):

        self.data_dir = data_dir
        self.output_col = output_col
        self.nominal_values = ["Mjob", "Fjob", "guardian", "reason"]
        self.df = self.get_df()
        self.process_dataframe_columns()
        self.train_indices = []


    def process_dataframe_columns(self):
        self.df[self.output_col] = self.df[self.output_col].apply(lambda x: x >= 10)
        self.df = self.df.drop(["G1", "G2"], axis=1)
        for column in self.df:
            if column == self.output_col:
                continue
            if column in self.nominal_values:  # Categoric data, perform one-hot encoding
                self.df = encode_and_bind(self.df, column)
            else:
                self.df[column] = self.df[column].astype('category')
                self.df[column] = self.df[column].cat.codes
        [normalize_columns(self.df, [col]) for col in self.df if col != self.output_col]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        attributes = self.df.iloc[idx].drop([self.output_col])
        label = self.df.iloc[idx][self.output_col]
        return {'image': torch.tensor(attributes).float(), 'label': torch.tensor(label).float()}

    def get_label(self, idx):
        return torch.tensor([self.df[self.output_col][idx]]).float()

    def compute_negative_to_positive_ratio(self):
        return len(self.df[self.df[self.output_col]==0])/len(self.df[self.df[self.output_col]==1])

    def get_df(self):
        df_path = (Path(__file__).parent / os.path.join(self.data_dir)).resolve()
        df = pd.read_csv(
            df_path,
            sep=",",
        ).dropna(subset=[self.output_col]).reset_index(drop=True)

        return df.reset_index(drop=True)
