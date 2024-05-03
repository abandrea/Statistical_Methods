"""
Bayesian approach

The prior is the average of y_train
For each feature, and for each value of that feature, count
the number of times that value is associated with a positive outcome
We then use Bayes' theorem to compute the probability of a positive outcome
"""
from pipeline import Data
import numpy as np


def logit(x):
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_model(model, x_test, y_test, dummy_value):
    y_pred = x_test.apply(model, axis=1)

    rmse = ((y_test - y_pred) ** 2).mean() ** 0.5
    dummy_rmse = ((y_test - dummy_value) ** 2).mean() ** 0.5
    usefulness = (dummy_rmse - rmse) / dummy_rmse

    # print()
    # print(f'      RMSE = {rmse:.2f}')
    # print(f'dummy_RMSE = {dummy_rmse:.2f}')
    # print(f'Usefulness = {usefulness:.1%}')

    return usefulness


def bayesian_model(counts, prior):
    def wrap(x):
        """predict the probability of a positive outcome"""
        out = prior

        for name in counts:

            # positive and negative counts for that value
            value = x[name]
            n_pos, n_neg = counts[name][value]
            n_pos += 1
            n_neg += 1

            # sum positive and negative counts for all the values of the feature
            n_all_pos = sum([counts[name][v][0] for v in counts[name]])
            n_all_neg = sum([counts[name][v][1] for v in counts[name]])
            n_all_pos += len(counts[name])
            n_all_neg += len(counts[name])

            # probability of a positive outcome given that value
            p_ab = n_pos / n_all_pos

            # probability of that value
            p_b = (n_pos + n_neg) / (n_all_pos + n_all_neg)

            # update the prior
            out = out * p_ab / p_b

        if out > 1:
            raise ValueError('probability greater than 1')

        return out

    return wrap


def training(coeff):

    # load data
    data = Data()
    x_train, x_test, y_train, y_test = data.train_test_split()
    y_train = y_train > 0
    y_test = y_test > 0

    # for each feature, and for each value of that feature, count
    # the number of times that value is associated with a positive and negative outcome
    counts = dict()
    for name in x_train.columns:
        counts[name] = dict()
        for value in x_train[name].unique():
            n_pos = np.sum((x_train[name] == value) & y_train.__neg__())
            n_neg = np.sum((x_train[name] == value) & y_train)
            counts[name][value] = (n_pos, n_neg)

    # TODO: modify the count to generalize better

    # display counts
    # for name in counts:
    #     print(name)
    #     for value in counts[name]:
    #         print(f'    {value}: {counts[name][value]}')

    # define model
    prior = y_train.mean()

    def model(x):
        out = logit(prior)
        out += coeff[x['chcond1']]
        out += coeff[x['chcond2']+2]
        return sigmoid(out)

    u_train = test_model(model, x_train, y_train, dummy_value=y_train.mean())
    u_test = test_model(model, x_test, y_test, dummy_value=y_train.mean())

    return u_train, u_test


def main(std=.05):
    # chcond1
    # c0 = np.array([-0.148,0.198])  # 0.23% (0.32%)

    # chcond2
    # c0 = np.array([-0.127,0.745])  # 0.87% (1.02%)

    # chcond1, chcond2
    c0 = np.array([0.445,1.109,-0.907,0.288])  # 1.51% (1.84%)

    # actdays
    # c0 = np.array([-0.373,0.295,1.070,1.698,1.643,1.994,2.531,1.701,1.513,1.539,2.027,2.554,1.483,3.336,1.847])

    # hospadmi
    # c0 = np.array([-0.159,0.642,1.176,1.496,1.474,0.970])

    best_c = c0
    best_u_train, best_u_test = training(best_c)
    best_u_test = best_u_train
    v = ",".join([f'{i:.3f}' for i in best_c])
    print(f'\n{best_u_train:.2%} ({best_u_test:.2%}) [{v}]')

    while True:
        c = np.random.normal(best_c, std)
        u_train, u_test = training(c)
        if u_train > best_u_train:
            best_u_train = u_train
            best_u_test = u_test
            best_c = c
            v = ",".join([f'{i:.3f}' for i in best_c])
            print(f'\n{best_u_train:.2%} ({best_u_test:.2%}) [{v}]')


if __name__ == '__main__':
    main()
