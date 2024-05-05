"""
Comments for the report

This method consists on using neural networks to learn a distance between two data points, based on the difference
between the values of the respective target features. During prediction, the new data point is compared to all the
training data points, and the k closest data points are used to calculate the average of the target features.


avg_rmse           0.197
avg_mabse          0.350
std_rmse           0.003
std_mabse          0.004
avg_dummy_rmse     0.193
avg_dummy_mabse    0.350
avg_training_time  84.118
std_training_time  15.504
"""
import numpy as np
from data import Data
from evaluation import evaluate, repeated_evaluation
from sklearn.neural_network import MLPRegressor
import pandas as pd
from time import time


class Model(dict):
    """Trained proximity knn model
    the model learns a distance between two data points
    """

    def fit(self, x_train, y_train, random_state=42, data_size=20_000, epsilon=0.1):
        print('\nBuilding dataset\n')
        self['x_train'] = x_train
        self['y_train_onehot'] = y_train
        y_train_values = np.argmax(y_train, axis=1)
        self['y_train_values'] = y_train_values

        # create an empty matrix with the same number of columns as x_train
        x_diff = np.zeros((0, x_train.shape[1]))
        y_diff = []

        for _ in range(data_size):
            if _ % 10_000 == 0:
                print(f'building dataset {_}/{data_size}')
            i = np.random.randint(0, x_train.shape[0])
            j = np.random.randint(0, x_train.shape[0])

            # todo dynamically adjust i, j
            if _ % 10 == 0:
                i = j

            x_i, x_j = x_train.iloc[i], x_train.iloc[j]
            x_abs_diff = np.abs(x_i - x_j)
            y_abs_diff = np.abs(y_train_values[i] - y_train_values[j])

            x_diff = np.vstack((x_diff, x_abs_diff))
            y_diff.append(y_abs_diff)

        y_diff = np.array(y_diff)
        y_diff = np.log(y_diff + epsilon)

        x_diff_train = pd.DataFrame(x_diff, columns=x_train.columns)
        y_diff_train = pd.DataFrame(y_diff, columns=['y_diff'])
        y_diff_train = np.ravel(y_diff_train)
        # y_diff_train = y_diff_train.clip(0, 1)  # todo remove?

        print(f'\nTraining model with {data_size} samples\n')
        # the model learns the distance between two data points
        self['model'] = MLPRegressor(hidden_layer_sizes=(200, 100),
                                     activation="relu",
                                     batch_size="auto",
                                     learning_rate="constant",
                                     max_iter=200,
                                     random_state=random_state,
                                     tol=1e-3,
                                     verbose=True
                                     )
        self['model'].fit(x_diff_train, y_diff_train)

    def predict(self, x_test, k=5):
        """
        check each row in x_test against each row in x_train, and
        return the average of y_train of the closest data points
        """
        y_pred = []
        y_train_onehot = self['y_train_onehot'].to_numpy()

        t1 = 0
        t2 = 0

        for i in range(x_test.shape[0]):

            if i % 100 == 0:
                print(f'predicting {i}/{x_test.shape[0]}')

            # compute the distance (y_proba) between x_test[i] and each x_train[j]
            t1 -= time()
            x_i = x_test.iloc[i]
            x_diff = np.abs(self['x_train'] - x_i)
            x_diff = pd.DataFrame(x_diff, columns=x_test.columns)

            y_proba = self['model'].predict(x_diff)

            t1 += time()

            # join the distance with the y_train_values, then sort by distance, take the k closest data points
            t2 -= time()
            v = list(zip(y_proba, y_train_onehot))
            v.sort(key=lambda x: x[0])
            v = v[:k]

            t2 += time()

            # take the average of the k closest data points
            new_y = np.mean([x[1] for x in v], axis=0)
            y_pred.append(new_y)

        t1 *= 1000 / x_test.shape[0]
        t2 *= 1000 / x_test.shape[0]
        print('\ntimes (ms)')
        print(f't1: {t1:.1f}, t2: {t2:.1f}\n')

        return np.array(y_pred)


def training(verbose=False, plot=False, title='Trained distance', k=5):
    data = Data()

    # keep only some columns
    data.keep_cols(['actdays', 'prescrib', 'hospadmi', 'illness', 'hscore'])

    # normalize data
    for name in data.x.columns:
        data.x[name] /= data.x[name].max()

    data.x_to_one_hot()
    data.y_to_one_hot()

    x_train, x_test, y_train, y_test = data.train_test_split()

    model = Model()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test, k=k)

    return evaluate(y_train, y_test, y_pred, verbose=verbose, plot=plot, title=title)


def final_evaluation():
    results = repeated_evaluation(training, n=5)
    for key, value in results.items():
        print(f'{key:18} {value:.3f}')


if __name__ == '__main__':
    # training(verbose=True, plot=True)
    final_evaluation()
