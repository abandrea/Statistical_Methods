import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pipeline import Data


def logit(x):
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():

    # load date
    data = Data()
    x_train, x_test, y_train, y_test = data.train_test_split()

    probabilities = dict()
    counts = dict()

    v = y_train == 0
    x1 = x_train[v]
    x2 = x_train[v.__neg__()]

    for name in x1.columns:
        # stacked bars plot
        df = pd.DataFrame({'y=0': x1[name].value_counts(),
                           'y=1': x2[name].value_counts()})
        df = df.fillna(0)
        df['y=0'] += 1
        df['y=1'] += 1

        total = df.sum(axis=1)

        df = df.div(total, axis=0)

        x_labels = df.index
        y_sigmoid = df['y=0'].to_numpy()
        y_line = logit(y_sigmoid)

        # TODO uncertainty
        # todo y[0] is underestimated

        # fit a line on x_label and y_to_fit2
        if len(x_labels) == 2:
            degree = 1
        else:
            degree = 2
        z = np.polyfit(x_labels, y_line, degree)
        p = np.poly1d(z)

        # evaluate the line on x_labels
        y_sigmoid_fit = p(x_labels)
        y_sigmoid_fit = sigmoid(y_sigmoid_fit)

        probabilities[name] = {x_labels[i]: y_sigmoid[i] for i in range(len(x_labels))}

        # fig, ax = plt.subplots(1, 2)
        #
        # plt.sca(ax[0])
        # df.plot(kind='bar', stacked=True, ax=ax[0])
        # plt.title(name)
        #
        # plt.sca(ax[1])
        # plt.scatter(x_labels, y_sigmoid, color='blue')
        # plt.plot(x_labels, y_sigmoid_fit, color='red')
        # plt.title(name)
        #
        # plt.show()

    for key, value in probabilities.items():
        print(key)
        print(value)
        print()

    def model(x):
        p = np.array([probabilities[name][x[name]] for name in x.index])

        return 1 - np.average(p)

    # print(x_test.iloc[0])
    # print(y_test.iloc[0])
    # print(model(x_test.iloc[0]))

    # print x_test and y_test side by side
    print('Predictions')
    y_pred = x_test.apply(model, axis=1)
    df = x_test.copy()
    df['actual'] = y_test
    df['predicted'] = y_pred
    print(df.head(40).to_string())

    # rmse
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    y_dummy = y_train.mean()
    dummy_rmse = np.sqrt(np.mean((y_test - y_dummy) ** 2))
    usefulness = (dummy_rmse - rmse) / dummy_rmse

    print()
    print(f"    y_mean = {np.mean(y_pred)}")
    print(f"   y_dummy = {y_dummy:.2f}")
    print()
    print(f"      rmse = {rmse:.3f}")
    print(f"dummy_rmse = {dummy_rmse:.3f}")
    print(f"usefulness = {usefulness:.1%}")



if __name__ == '__main__':
    main()
