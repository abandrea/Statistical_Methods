"""
Comments for the report

The Poisson regression on its own fails to make useful predictions due to the unbalanced nature of the data.


Results

rmse               0.224
mabse              0.350
dummy_rmse         0.203
dummy_mabse        0.350
"""
import numpy as np
from data import Data
from evaluation import evaluate, repeated_evaluation
from sklearn.linear_model import PoissonRegressor


class Model(dict):
    """naive bayes"""
    def fit(self, x_train, y_train, random_state=42, verbose=True):
        y_train_values = np.argmax(y_train, axis=1)
        self['model'] = PoissonRegressor(max_iter=1000)
        self['model'].fit(x_train, y_train_values)

    def predict(self, x_test):
        y_pred = self['model'].predict(x_test)
        y_pred = np.clip(y_pred, 0, 8)

        # convert y_pred to one-hot
        y_pred = np.eye(9)[y_pred.astype(int)]

        return y_pred


def training(verbose=False, plot=False, random_state=None, title='Logistic Regression'):
    data = Data()

    data.keep_cols(['actdays', 'prescrib', 'hospadmi', 'illness', 'hscore', 'nondocco'])

    data.y = np.clip(data.y, 0, 8)  # todo fix limit issue

    # data.normalize_x()
    data.x_to_one_hot()
    data.y_to_one_hot()

    x_train, x_test, y_train, y_test = data.train_test_split(random_state=random_state)

    model = Model()
    model.fit(x_train, y_train, random_state=random_state, verbose=verbose)

    y_pred = model.predict(x_test)

    return evaluate(y_train, y_test, y_pred, verbose=verbose, plot=plot, title=title)


def final_evaluation():
    results = repeated_evaluation(training, n=20)
    for key, value in results.items():
        print(f'{key:18} {value:.3f}')


if __name__ == '__main__':
    training(verbose=True, plot=True, random_state=42)
    # final_evaluation()
