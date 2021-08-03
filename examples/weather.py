import os

import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils import data

import numpy as np
import pandas as pd
import seaborn as sns

########################################################
# The dataset is a Kaggle weather history dataset where
# the goal is to predict the temperature given apparent
# temperature. They are closely related, so should yield
# little uncertainty
#
# https://www.kaggle.com/budincsevity/szeged-weather
########################################################

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, 'data/weatherHistory.csv'))
df = df[['Temperature (C)', 'Apparent Temperature (C)']].astype(np.float32)

X = df[['Apparent Temperature (C)']].values
y = df[['Temperature (C)']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

weather_features = X.shape[1]


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
