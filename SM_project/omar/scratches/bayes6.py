from pipeline import Data
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle


def loss(y_true_distribution, y_pred_values) -> float:
    """return the loss of the model"""
    out = 0
    for i in range(len(y_true_distribution)):
        new = 0
        for j in range(len(y_true_distribution[i])):
            p = y_true_distribution[i][j]
            d = j - y_pred_values[i]

            # new += p * (d != 0)
            new += p * abs(d)
            # new += p * d**2

        out += new
    return out / len(y_true_distribution)


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
        out = np.zeros(len(self.par.keys()))
        for y_value in self.par.keys():
            p = self.par[y_value]['prior']
            for name in self.par[y_value]:
                if name != 'prior':
                    value = x[name]
                    if value in self.par[y_value][name]:
                        p += self.par[y_value][name][value]
            out[y_value] += p
        out = np.exp(out)
        out /= out.sum()
        return out


def eval_model(x_test, y_test, model):
    """return the loss of the model on the test set"""
    y_pred = np.array(list(x_test.apply(model, axis=1)))
    return loss(y_pred, y_test)


def train_model(x_train, y_train, shape, n_epoch, std=0.1):
    """train the model and return the model as a function of x"""

    # initail parameters
    par = Parameters()
    for i in range(shape['TARGET']):
        par[i] = {'prior': 0.}
        for name in shape:
            if name != 'TARGET':
                par[i][name] = dict()
                for value in x_train[name].unique():
                    par[i][name][value] = 0.
    par.set_parameters(np.random.normal(0, std, par.n_parameters()))

    best_model = Model(par)
    new_model = Model(deepcopy(par))
    best_loss = eval_model(x_train, y_train, best_model)
    print(f'\ninitial loss: {best_loss:.3f}')

    for i_epoch in range(n_epoch):

        delta = np.random.normal(0, std, par.n_parameters())

        if i_epoch % 2:
            new_model.par.increase_parameters(delta)
        else:
            new_model.par.set_parameters(delta)

        new_loss = eval_model(x_train, y_train, new_model)
        if new_loss < best_loss:
            best_model = deepcopy(new_model)
            best_loss = new_loss
            print(f'   best loss: {best_loss:.3f}    {(i_epoch+1)/n_epoch:.1%}')
        else:
            new_model = deepcopy(best_model)

    return best_model


def train_and_test(n_epoch=200, std=4.):
    """ train and test the model and return the usefulness of the model """
    data = Data()
    x_train, x_test, y_train, y_test = data.train_test_split()

    # names = list(x_train.columns)
    # names = ['illness', 'actdays', 'hscore', 'chcond1', 'chcond2', 'nondocco', 'hospadmi', 'hospdays', 'prescrib']
    names = ['actdays']
    # names = ['illness', 'actdays', 'hscore']
    # names = ['chcond1', 'chcond2', 'nondocco']
    # names = ['hospadmi', 'hospdays', 'prescrib']

    print(names)
    shape = dict()
    for name in names:
        shape[name] = len(x_train[name].unique())
    shape['TARGET'] = len(y_train.unique())

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    model = train_model(x_train, y_train, shape, n_epoch=n_epoch, std=std)

    y_pred_train = np.array(x_train.apply(model, axis=1))
    y_pred_test = np.array(list(x_test.apply(model, axis=1)))

    # print('\nExamples')
    # df = x_test.copy()
    # df['TARGET'] = y_test
    # text = df.head(30).to_string().split('\n')
    # print(text[0])
    # for i in range(1, len(text)):
    #     print(f'{text[i]} {np.round(y_pred_test[i-1], 2)}')

    train_loss = loss(y_pred_train, y_train)
    test_loss = loss(y_pred_test, y_test)

    dummy_out = np.zeros((len(y_test), shape['TARGET']), dtype=float)
    dummy_out[:, 0] = 1.
    dummy_loss = loss(dummy_out, y_test)

    usefulness_test = (dummy_loss - test_loss) / dummy_loss
    usefulness_test = max([0, min([1, usefulness_test])])

    print(f'\nTraining set performance')
    print(f'    dummy_loss = {dummy_loss:6.2f}')
    print(f'    train_loss = {train_loss:6.2f}')
    print(f'     test_loss = {test_loss:6.2f}')
    print()
    print(f'   test_useful = {usefulness_test:6.1%}')

    # save model
    if usefulness_test > 0:
        file_name = f'{usefulness_test*1000:.0f}'
        while len(file_name) < 3:
            file_name = '0' + file_name
        file_name += '_' + '_'.join(names)
        file_name = f'models_6\\model_par_{file_name}.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(model.par, f)

    # plot the predictions
    n_examples = 30
    n_figs = 5

    fig, axs = plt.subplots(1, n_figs)
    plt.suptitle('Predictions')

    for i in range(n_figs):
        plt.sca(axs[i])

        # plot the predictions
        plt.imshow(y_pred_test[n_examples*i:n_examples*(i+1)], aspect='auto', cmap='magma')
        plt.xticks(list(range(shape['TARGET'])), list(range(shape['TARGET'])))
        plt.clim(0, 1)
        if i+1==n_figs:
            plt.colorbar()

        # plot the true values
        plt.scatter(y_test[n_examples*i:n_examples*(i+1)], range(n_examples), c='w', s=20, edgecolors='k')
        plt.yticks([])

    plt.show()

    return usefulness_test, names


def main():
    performance = []
    while True:
        usefulness, names = train_and_test()
        performance.append(usefulness)
        mean = np.mean(performance)
        std = np.std(performance)
        print(f'\n AVERAGE PERFORMANCE {mean:.1%} +- {std:.1%}   ({len(performance)})\n')


if __name__ == '__main__':
    train_and_test()
    # main()
