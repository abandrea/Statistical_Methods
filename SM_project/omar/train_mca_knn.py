"""
Comments for the report

The MCA-KNN model is a combination of two methods: Multiple Correspondence Analysis (MCA) and K-Nearest Neighbors (KNN).
    The purpose of MCA is to reduce the dimensionality the data. It is analogous to PCA, but specifically designed
for categorical data. The output of MCA is a latent representation of the data, which is then used as input to the
KNN model. In this form, the data points belonging to different classes are expected to be more separable.
    The KNN model is a non-parametric method used for classification. It is based on the idea that similar data points
are likely to belong to the same class. The model is trained by storing the training data, and then it classifies new
data points based on their similarity to the training data. The model is simple and intuitive, but it can be sensitive
to the choice of the number of neighbors and the distance metric. A big number of neighbors "k" can lead to
underfitting, especially for categories with a low number of samples. On the other hand, a small number of neighbors
can lead to overfitting.


Results

n_neighbors=10, n_components=10
avg_rmse           0.190
avg_mabse          0.300
std_rmse           0.004
std_mabse          0.025
avg_dummy_rmse     0.194
avg_dummy_mabse    0.303
avg_training_time  0.126
std_training_time  0.026

# n_neighbors=5, n_components=10
avg_rmse           0.197
avg_mabse          0.299
std_rmse           0.005
std_mabse          0.024
avg_dummy_rmse     0.193
avg_dummy_mabse    0.298
avg_training_time  0.118
std_training_time  0.032

# n_neighbors=10, n_components=5
avg_rmse           0.189
avg_mabse          0.297
std_rmse           0.005
std_mabse          0.028
avg_dummy_rmse     0.194
avg_dummy_mabse    0.303
avg_training_time  0.102
std_training_time  0.038

# n_neighbors=10, n_components=5
avg_rmse           0.197
avg_mabse          0.306
std_rmse           0.006
std_mabse          0.025
avg_dummy_rmse     0.195
avg_dummy_mabse    0.302
avg_training_time  0.104
std_training_time  0.035
"""
import numpy as np
import pandas as pd
from data import Data
from prince import MCA
from sklearn.neighbors import KNeighborsClassifier
from evaluation import evaluate, repeated_evaluation


class Model(dict):
    """naive bayes"""
    def fit(self, x_train, y_train, n_neighbors=5, n_components=4, random_state=42, verbose=True):

        # learn a function to reduce dimensionality
        self['mca'] = MCA(n_components=n_components,
                          copy=True,
                          check_input=True,
                          engine='sklearn',
                          random_state=random_state,
                          one_hot=False
                          )

        self['mca'] = self['mca'].fit(x_train)
        reduced_x = self['mca'].row_coordinates(x_train).to_numpy()

        self['model'] = KNeighborsClassifier(n_neighbors=n_neighbors,
                                             weights="uniform",
                                             algorithm="auto",
                                             leaf_size=30,
                                             p=2,
                                             metric="minkowski")

        y_train_values = np.argmax(y_train, axis=1)

        self['model'].fit(reduced_x, y_train_values)

    def predict(self, x_test: pd.DataFrame):
        reduced_data = self['mca'].row_coordinates(x_test).to_numpy()
        y_pred = self['model'].predict_proba(reduced_data)
        return y_pred


def training(verbose=False, plot=False, random_state=None, title='MCA-KNN',
             n_neighbors=5, n_components=5, n_ones_min=5):
    data = Data()

    data.keep_cols(['actdays', 'prescrib', 'hospadmi', 'illness', 'nondocco'])

    data.y = np.clip(data.y, 0, 8)  # todo fix limit issue

    # data.normalize_x()
    data.x_to_one_hot()
    data.y_to_one_hot()

    # remove columns with not enough 1s
    remove_cols = data.x.columns[(data.x.sum() <= n_ones_min)]
    data.remove_cols(remove_cols)

    if verbose:
        print(f'Columns with less than {n_ones_min} ones removed:')
        print(remove_cols)

    x_train, x_test, y_train, y_test = data.train_test_split(random_state=random_state)

    model = Model()
    model.fit(x_train, y_train,
              n_neighbors=n_neighbors, n_components=n_components,
              random_state=random_state, verbose=verbose)

    y_pred = model.predict(x_test)

    return evaluate(y_train, y_test, y_pred, verbose=verbose, plot=plot, title=title)


def final_evaluation():
    results = repeated_evaluation(training, n=20)
    for key, value in results.items():
        print(f'{key:18} {value:.3f}')


if __name__ == '__main__':
    # training(verbose=True, plot=False, random_state=1)
    final_evaluation()
