"""
Comments for the report

This model makes prediction by averaging the values of the target variables that match the features of the input
data-point. Although it's a simple model, it can be a good baseline to compare with more complex models.
This model's weakness is that it can't predict unseen data-points, as it requires the exact same features to be present.
Therefore, it is unable to generalize, and it can handle only few features at the time.

avg_rmse           0.173
avg_mabse          0.288
std_rmse           0.004
std_mabse          0.021
avg_dummy_rmse     0.184
avg_dummy_mabse    0.296
avg_training_time  0.905
std_training_time  0.039
"""
import numpy as np
from data import Data
from evaluation import evaluate, repeated_evaluation


class Model(dict):

    def train(self, x_train, y_train, names=('actdays_0', 'actdays_14', 'prescrib_0', 'hospdays_0', 'hospadmi_0')):
        self['names'] = names
        self['y_size'] = y_train.shape[1]
        self['x_train'] = x_train
        self['y_train'] = y_train

    def predict(self, x_test):
        """
        The model outputs the average y_train of the x_train data-points
        with the correct values for the features
        """
        out = np.zeros((len(x_test), self['y_size']))
        names = self['names']
        for i, (index, x) in enumerate(x_test.iterrows()):
            indices = np.ones(len(self['x_train']), dtype=bool)
            for name in names:
                indices *= self['x_train'][name] == x[name]
            out[i] = self['y_train'][indices].mean(axis=0)
        return out


def training(verbose=False, plot=False, random_state=None, title='Lookup Model'):

    data = Data()
    data.x_to_one_hot()
    data.y_to_one_hot()

    x_train, x_test, y_train, y_test = data.train_test_split(random_state=random_state)

    model = Model()
    model.train(x_train, y_train)
    y_pred = model.predict(x_test)

    return evaluate(y_train, y_test, y_pred,
                    verbose=verbose, plot=plot, title=title)


def final_evaluation():
    results = repeated_evaluation(training, n=20)
    print()
    for key, value in results.items():
        print(f'{key:18} {value:.3f}')


if __name__ == '__main__':
    # training(verbose=True, plot=True, random_state=0)
    final_evaluation()
