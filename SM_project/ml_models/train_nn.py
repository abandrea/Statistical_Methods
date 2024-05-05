"""
Comments for the report

Neural Networks are powerful models that can learn complex patterns in the data. The model is able to learn the
relationship between the input features and the output labels. Neural networks are computationally expensive to train
when compared to other models that we used.


Results

avg_rmse           0.182
avg_mabse          0.287
std_rmse           0.003
std_mabse          0.019
avg_dummy_rmse     0.195
avg_dummy_mabse    0.305
avg_training_time  44.66
std_training_time  9.656
"""
import numpy as np
from data import Data
from evaluation import evaluate, repeated_evaluation
from sklearn.neural_network import MLPClassifier


class Model(dict):
    """naive bayes"""
    def fit(self, x_train, y_train, random_state=42, verbose=True):
        y_train_values = np.argmax(y_train, axis=1)
        self['model'] = MLPClassifier(hidden_layer_sizes=(200, 100),
                                      activation="relu",
                                      solver="adam",
                                      alpha=0.001,
                                      batch_size="auto",
                                      learning_rate="constant",
                                      learning_rate_init=1e-5,
                                      max_iter=500,
                                      shuffle=True,
                                      random_state=random_state,
                                      tol=1e-4,
                                      verbose=verbose,
                                      warm_start=False,
                                      n_iter_no_change=20,
                                      beta_1=0.99,
                                      beta_2=0.9)
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
