import pandas as pd
from pipeline import Data, Pipeline
# from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


def example_data():
    pipeline = Data()
    x_train, x_test, y_train, y_test = pipeline.train_test_split()
    print()
    print(x_train.head(10).to_string())
    print()
    print(y_train.head(10).to_string())


def example_pipeline():
    x, y = Data().get_data()

    # model = MLPClassifier(hidden_layer_sizes=(100, 50),
    #                       max_iter=200,
    #                       random_state=0,
    #                       verbose=True,
    #                       activation='relu',
    #                       solver='adam',
    #                       learning_rate='adaptive',
    #                       learning_rate_init=0.01,
    #                       alpha=0.01,
    #                       tol=1e-3,
    #                       n_iter_no_change=10,
    #                       )

    model = GaussianNB()

    pipeline = Pipeline(x, y, model)
    pipeline.run()


def example():
    x, y = Data().get_data()

    # convert to one-hot encoding dummy
    for name in x.columns:
        n_values = len(set(x[name]))
        if n_values > 20:
            raise ValueError(f"Too many unique values for {name}: {n_values}")
        if n_values > 2:
            x = pd.get_dummies(x, columns=[name], drop_first=True)

    x = x.astype(int)

    y = pd.get_dummies(y, drop_first=False)
    y = y.astype(int)

    print(x)
    print(y)


if __name__ == '__main__':
    # example_data()
    example_pipeline()
    # example()
