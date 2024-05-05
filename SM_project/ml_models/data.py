"""
This module is used to read the data and preprocess it.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


class Data:

    TARGET = "doctorco"
    DROP_COLS = ['constant', 'agesq', 'medicine']

    def __init__(self, data_path='data\\HealthCareAustralia.csv'):
        """read the data"""
        self.df = pd.read_csv(data_path, index_col=0)
        self._drop_cols(Data.DROP_COLS)
        self.x = self.df.drop([Data.TARGET], axis=1)
        self.y = self.df[Data.TARGET]
        self.shuffle()
        del self.df

    def _drop_cols(self, col_names):
        """drop columns"""
        for name in col_names:
            self.df = self.df.drop([name], axis=1)

    def keep_cols(self, col_names):
        """keep columns"""
        for name in self.x.columns:
            if name not in col_names:
                self.x = self.x.drop([name], axis=1)

    def remove_cols(self, col_names):
        """remove columns"""
        for name in col_names:
            self.x = self.x.drop([name], axis=1)

    def normalize_x(self):
        """normalize x"""
        for name in self.x.columns:
            self.x[name] /= self.x[name].max()

    def x_to_one_hot(self, drop_first=False):
        """convert x to one-hot encoding dummy"""
        for name in self.x.columns:
            n_values = len(set(self.x[name]))
            if n_values > 2:
                self.x = pd.get_dummies(self.x, columns=[name], drop_first=drop_first)
        self.x = self.x.astype(int)

    def y_to_one_hot(self, drop_first=False):
        """convert y to one-hot encoding dummy"""
        self.y = pd.get_dummies(self.y, drop_first=drop_first)
        self.y = self.y.astype(int)

    def shuffle(self, random_state=42):
        """change the order of the data (x and y)"""
        random_indices = self.x.sample(frac=1, random_state=random_state).index
        self.x = self.x.loc[random_indices]
        self.y = self.y.loc[random_indices]

    def get_data(self):
        """get the data"""
        return self.x, self.y

    def train_test_split(self, test_size=0.2, random_state=42):
        """split the data
        returns: x_train, x_test, y_train, y_test
        """
        return train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)

    def filter_rows(self, filter):
        """filter rows with boolean vector"""
        self.x = self.x[filter]
        self.y = self.y[filter]
