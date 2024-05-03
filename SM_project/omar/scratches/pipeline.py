import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Data:
    TARGET = "doctorco"
    DROP_COLS = ['constant', 'agesq', 'medicine',
                 #'age', 'income', 'levyplus',
                 #'freepera', 'nonpresc', 'chcond1',
                 #'freepoor'
                 ]

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
        """split the data"""
        return train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)


class Pipeline:

    def __init__(self, x, y, model):
        self.x = x
        self.y = y
        self.model = model
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.df_original = None
        self.train_indices = None
        self.test_indices = None

    def save_original(self):
        """save the original data"""
        self.df_original = self.x.copy()
        self.df_original['doctorco'] = self.y

    def preprocess_x(self):
        """convert to one-hot encoding dummy"""
        print('\n\n Preprocessing x \n')
        for name in self.x.columns:
            n_values = len(set(self.x[name]))
            print(f'{name:>15}  {n_values}')
            if n_values > 2:
                self.x = pd.get_dummies(self.x, columns=[name], drop_first=True)
        self.x = self.x.astype(int)

    def preprocess_y(self):
        """convert to one-hot encoding dummy"""
        self.y = (self.y > 0).astype(int)

    def train_test_split(self, train_size=0.8):
        """split the data into train and test sets"""
        random_indices = self.x.sample(frac=1).index
        n_train = int(train_size * len(random_indices))
        self.train_indices = random_indices[:n_train]
        self.test_indices = random_indices[n_train:]
        self.x_train = self.x.loc[self.train_indices]
        self.x_test = self.x.loc[self.test_indices]
        self.y_train = self.y.loc[self.train_indices]
        self.y_test = self.y.loc[self.test_indices]

    def train(self):
        """train the model"""
        print('\n\n Training \n')
        self.model.fit(self.x_train, self.y_train)

    def predictions(self):
        """predict and convert one-hot back to original"""
        self.y_pred = self.model.predict(self.x_test)

        # keep only test set values
        self.df_original['predicted'] = self.model.predict(self.x)
        self.df_original = self.df_original.loc[self.test_indices]

    def presentation(self, n_lines=200):
        """show examples of predicted vs actual"""
        text = self.df_original.head(n_lines).to_string().split('\n')
        print('\n' + text[0])
        for i in range(n_lines):
            if self.y_test.iloc[i] != self.y_pred[i]:
                # print mistakes in red
                print(f'\033[91m{text[i + 1]}\033[0m')
            else:
                if self.y_test.iloc[i] > 0:
                    # print correct predictions in green
                    print(f'\033[92m{text[i + 1]}\033[0m')
                else:
                    print(text[i + 1])

    def evaluation(self):
        """evaluate the model's performance"""
        print('\n\n Evaluaiton \n')

        # calculate rmse
        rmse = ((self.y_test - self.y_pred) ** 2).mean() ** 0.5
        print(f'      RMSE = {rmse:.2f}')

        y_pred_train = self.model.predict(self.x_train)
        rmse_train = ((self.y_train - y_pred_train) ** 2).mean() ** 0.5
        print(f'RMSE train = {rmse_train:.2f}')

        # calculate dummy rmse
        most_common_value = self.y_train.value_counts().idxmax()
        print(f' dummy_val = {most_common_value}')
        y_dummy = np.ones(self.y_test.shape) * most_common_value
        dummy_rmse = ((self.y_test - y_dummy) ** 2).mean() ** 0.5
        print(f'Dummy RMSE = {dummy_rmse:.2f}')

        # calculate usefulness
        usefulness = (dummy_rmse - rmse) / dummy_rmse
        print(f'Usefulness = {usefulness:.1%}')

    def run(self, train_size=0.8):
        """run the pipeline"""

        # save the original data
        self.save_original()

        # pre-proces x
        self.preprocess_x()

        # pre-proces y
        self.preprocess_y()

        # train-test split
        self.train_test_split(train_size)

        # train model
        self.train()

        # predict
        self.predictions()

        # present test set with predictions
        self.presentation()

        # evaluate performance
        self.evaluation()
