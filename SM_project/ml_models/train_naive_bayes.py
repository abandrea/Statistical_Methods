"""
Comments for the report

The Naive Bayes model is a simple probabilistic classifier based on applying Bayes' theorem with strong independence
assumptions. It is a popular method for text classification, but it can be used for any type of data. The model is
trained by estimating the probability of each class given the input data, and then it uses these probabilities to
predict the class of new data points. The Naive Bayes model is known for its simplicity and speed, and it is often used
as a baseline for comparison with more complex models. The model is based on the assumption that the features are
independent, which is often not true in practice. However, the model can still perform well in practice, especially when
the independence assumption is not too far from the truth.
    In our case, as shown by our analysis, feature independence is not a realistic assumption, but we are still
interested in the performance of the Naive Bayes model as a baseline for comparison with more complex models.


Results
avg_rmse           0.190
avg_mabse          0.292
std_rmse           0.004
std_mabse          0.026
avg_dummy_rmse     0.194
avg_dummy_mabse    0.303
avg_training_time  0.079
std_training_time  0.004

"""
import numpy as np
from data import Data
from evaluation import evaluate, repeated_evaluation
from sklearn.naive_bayes import MultinomialNB


def training(verbose=False, plot=False, random_state=None, title='Naive Bayes'):
    data = Data()

    data.y = np.clip(data.y, 0, 8)  # todo fix limit issue

    data.x_to_one_hot()
    data.y_to_one_hot()

    # keep only some columns
    names = ['actdays_0', 'actdays_10', 'actdays_14', 'age_0.27',
             'hscore_0', 'illness_0', 'income_0.55']

    data.keep_cols(names)

    x_train, x_test, y_train, y_test = data.train_test_split(random_state=random_state)

    y_train_values = np.argmax(y_train.values, axis=1)

    model = MultinomialNB()
    model.fit(x_train, y_train_values)
    y_pred = model.predict_proba(x_test)

    return evaluate(y_train, y_test, y_pred, verbose=verbose, plot=plot, title=title)


def final_evaluation():
    results = repeated_evaluation(training, n=20)
    print()
    for key, value in results.items():
        print(f'{key:18} {value:.3f}')


if __name__ == '__main__':
    # training()
    final_evaluation()
