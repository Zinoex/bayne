import os

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils import data

import numpy as np
import pandas as pd

########################################################
# The dataset is a Kaggle weather history dataset where
# the goal is to predict the temperature given apparent
# temperature. They are closely related, so should yield
# little uncertainty
#
# https://www.kaggle.com/budincsevity/szeged-weather
########################################################
from torch.utils.data import TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, 'data/weatherHistory.csv'))
df = df[['Temperature (C)', 'Apparent Temperature (C)']].astype(np.float32)

X = df[['Apparent Temperature (C)']].values
y = df[['Temperature (C)']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = torch.tensor(X_scaler.fit_transform(X_train))
X_test = torch.tensor(X_scaler.transform(X_test))
y_train = torch.tensor(y_scaler.fit_transform(y_train))
y_test = torch.tensor(y_scaler.transform(y_test))

weather_features = X.shape[1]


class WeatherHistoryDataset(TensorDataset):
    def __init__(self, train=True):
        if train:
            super(WeatherHistoryDataset, self).__init__(X_train, y_train)
        else:
            super(WeatherHistoryDataset, self).__init__(X_test, y_test)
