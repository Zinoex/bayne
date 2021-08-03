import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils import data

import numpy as np
import pandas as pd

########################################################
# The dataset is a Kaggle weather history dataset where
# the goal is to predict the temperature given humidity
# and precipitation type.
#
# https://www.kaggle.com/budincsevity/szeged-weather
########################################################

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, 'data/weatherHistory.csv'))
df = df[['Temperature (C)', 'Humidity', 'Precip Type']]

# No precipitation is listed as nan. Replace with N/A string.
df['Precip Type'] = df['Precip Type'].replace(np.nan, 'N/A')
unique_precip_type = len(df['Precip Type'].unique())

# This is equal to the number of input features
total_columns = unique_precip_type + 1

df = pd.get_dummies(df).astype(np.float32)

y = df[['Temperature (C)']].values
X = df[['Humidity', 'Precip Type_N/A', 'Precip Type_rain', 'Precip Type_snow']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class WeatherHistoryDataset(data.Dataset):
    def __init__(self, train=True):
        self.X = X_train if train else X_test
        self.y = y_train if train else y_test

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]
