"""
Comments for the report

Logistic regression

Results

avg_rmse           0.181
avg_mabse          0.289
std_rmse           0.005
std_mabse          0.026
avg_dummy_rmse     0.195
avg_dummy_mabse    0.305
avg_training_time  0.172
std_training_time  0.018
"""
import numpy as np
from data import Data
from evaluation import evaluate, repeated_evaluation
from sklearn.linear_model import LogisticRegression


class Model(dict):
    """naive bayes"""
    def fit(self, x_train, y_train, random_state=42, verbose=True):
        y_train_values = np.argmax(y_train, axis=1)
        self['model'] = LogisticRegression(random_state=random_state, max_iter=1000)
        self['model'].fit(x_train, y_train_values)

    def predict(self, x_test):
        return self['model'].predict_proba(x_test)


def training(verbose=False, plot=False, random_state=None, title='NN'):
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
    # training(verbose=True, plot=True, random_state=42)
    final_evaluation()
