"""
Comments for the report

Random forest is an ensemble method that aggregates the predictions of several individual decision trees.
Each tree is trained on a random subset of the data. Random forest uses a technique called bagging, that trains
independently multiple decision trees by randomly sampling the training data with replacement.


Results

avg_rmse           0.186
avg_mabse          0.298
std_rmse           0.003
std_mabse          0.019
avg_dummy_rmse     0.194
avg_dummy_mabse    0.303
avg_training_time  0.187
std_training_time  0.006
"""
import numpy as np
from data import Data
from evaluation import evaluate, repeated_evaluation
from sklearn.ensemble import RandomForestClassifier


def training(verbose=False, plot=False, random_state=None, title='Random Forest'):
    data = Data()

    data.y = np.clip(data.y, 0, 8)  # todo fix limit issue

    data.x_to_one_hot()
    data.y_to_one_hot()

    # keep only some columns
    names = ['actdays_0', 'actdays_14', 'actdays_6', 'age_0.22',
             'age_0.32', 'age_0.42', 'age_0.52', 'hospadmi_0',
             'hospadmi_1', 'hscore_0', 'hscore_8', 'income_0.06',
             'prescrib_0']
    data.x = data.x[names]

    x_train, x_test, y_train, y_test = data.train_test_split(random_state=random_state)

    y_train_values = np.argmax(y_train.values, axis=1)

    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train_values)

    y_pred = model.predict_proba(x_test)

    return evaluate(y_train, y_test, y_pred, verbose=verbose, plot=plot, title=title)


def final_evaluation():
    results = repeated_evaluation(training, n=20)
    for key, value in results.items():
        print(f'{key:18} {value:.3f}')


if __name__ == '__main__':
    # training(verbose=True, plot=True, random_state=42)
    final_evaluation()
