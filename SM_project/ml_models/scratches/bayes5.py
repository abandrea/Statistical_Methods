#  AVERAGE PERFORMANCE 4.8% +- 3.6%   (30)
from pipeline import Data
import numpy as np
from copy import deepcopy
import pickle


def logit(x):
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Parameters(dict):

    def __iter__(self):
        for y_value in self.keys():
            for name in self[y_value]:
                if name == 'prior':
                    yield y_value, name, None, self[y_value][name]
                else:
                    for x_value in self[y_value][name]:
                        yield y_value, name, x_value, self[y_value][name][x_value]

    def n_parameters(self):
        """return the number of parameters"""
        return len(list(self.__iter__()))

    def set_parameters(self, values):
        """set the parameters to the values in the list"""
        for i, (y_value, name, x_value, weight) in enumerate(self.__iter__()):
            if x_value is None:
                self[y_value][name] = values[i]
            else:
                self[y_value][name][x_value] = values[i]

    def increase_parameters(self, values):
        """increase the parameters by the values in the list"""
        for i, (y_value, name, x_value, weight) in enumerate(self.__iter__()):
            if x_value is None:
                self[y_value][name] += values[i]
            else:
                self[y_value][name][x_value] += values[i]


class Model:

    def __init__(self, par):
        self.par = par

    def __call__(self, x):
        """predict the y value for the given x value"""
        for y_value in self.par.keys():
            p = self.par[y_value]['prior']
            for name in self.par[y_value]:
                if name != 'prior':
                    value = x[name]
                    if value in self.par[y_value][name]:
                        p += self.par[y_value][name][value]
            if p >= 0:
                return y_value
        return len(self.par.keys())


def eval_model(x_test, y_test, model):
    """return the rmse of the model on the test set"""
    y_pred = x_test.apply(model, axis=1)
    rmse = ((y_test - y_pred) ** 2).mean() ** 0.5
    return rmse


def train_model(x_train, y_train, n_epoch=260, std=0.1):
    """train the model and return the model as a function of x"""
    counts = dict()
    for name in x_train.columns:
        counts[name] = len(x_train[name].unique())
    counts['TARGET'] = len(y_train.unique())

    # print('\nUnique values')
    # for name in counts:
    #     print(name, counts[name])

    # initail parameters
    par = Parameters()
    for i in range(counts['TARGET']-1):
        par[i] = {'prior': 0.}
        for name in counts:
            if name != 'TARGET':
                par[i][name] = dict()
                for value in x_train[name].unique():
                    par[i][name][value] = 0.
    par.set_parameters(np.random.normal(0, std, par.n_parameters()))

    best_model = Model(par)
    new_model = Model(deepcopy(par))
    best_rmse = eval_model(x_train, y_train, best_model)
    print(f'\ninitial rmse: {best_rmse:.3f}')

    for i_epoch in range(n_epoch):

        delta = np.random.normal(0, std, par.n_parameters())
        new_model.par.increase_parameters(delta)
        new_rmse = eval_model(x_train, y_train, new_model)
        if new_rmse < best_rmse:
            best_model = deepcopy(new_model)
            best_rmse = new_rmse
            print(f'   best rmse: {best_rmse:.3f}    {(i_epoch+1)/n_epoch:.1%}')
        else:
            new_model = deepcopy(best_model)

    return best_model


def train_and_test():
    data = Data()
    x_train, x_test, y_train, y_test = data.train_test_split()

    model = train_model(x_train, y_train, n_epoch=300, std=0.1)

    y_pred_train = x_train.apply(model, axis=1)
    y_pred_test = x_test.apply(model, axis=1)

    print('\nExamples')
    df = x_test.copy()
    df['TARGET'] = y_test
    df['PREDICT'] = df.apply(model, axis=1)
    print(df.head(100).to_string())

    train_rmse = ((y_train - y_pred_train) ** 2).mean() ** 0.5
    test_rmse = ((y_test - y_pred_test) ** 2).mean() ** 0.5
    dummy_out = 0
    dummy_rmse = ((y_test - dummy_out) ** 2).mean() ** 0.5
    usefulness_test = (dummy_rmse - test_rmse) / dummy_rmse
    generalization = max(0, min(1, usefulness_test))

    print(f'\nTraining set performance')
    print(f'    dummy_RMSE = {dummy_rmse:6.2f}')
    print(f'    train_RMSE = {train_rmse:6.2f}')
    print(f'     test_RMSE = {test_rmse:6.2f}')
    print(f'   test_useful = {usefulness_test:6.1%}')
    print(f'generalization = {generalization:6.1%}')

    # save model
    with open(f'models\\model_par_{generalization*1000:.0f}.pkl', 'wb') as f:
        pickle.dump(model.par, f)

    return usefulness_test


def main():
    performance = []
    while True:
        usefulness = train_and_test()
        performance.append(usefulness)
        print(f'\n AVERAGE PERFORMANCE {np.mean(performance):.1%} +- {np.std(performance):.1%}   ({len(performance)})\n')


if __name__ == '__main__':
    main()
