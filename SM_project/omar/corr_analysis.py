"""
This program shows the most promising features
to predict whether the target variable y == n or y > n
for various values of n.
"""
import numpy as np
from data import Data


def test_corr(y_value=0, n_print=10):
    data = Data()
    data.x_to_one_hot()
    data.y_to_one_hot()
    x, y = data.get_data()

    total_length = len(x)

    indices_to_keep = np.sum(y.iloc[:, y_value:], axis=1) == 1
    x = x[indices_to_keep]
    y = y[indices_to_keep]

    target = y.iloc[:, y_value]   # y == 0
    values = []
    names = sorted(list(x.columns))

    # compute the correlation between each feature and the target variable
    for name in names:
        if np.sum(x[name]) > 0:
            corr = np.corrcoef(x[name], target)[0, 1]
            values.append((name, corr, (name, )))

    # compute the interactions between the features
    for i in range(len(names)):
        for j in range(i):
            name1 = names[j]
            name2 = names[i]
            new_x = x[name1] * x[name2]
            if sum(new_x) > 0:
                new_name = name1 + '_AND_' + name2
                corr = np.corrcoef(new_x, target)[0, 1]
                values.append((new_name, corr, (name1, name2)))

    # sort the values by the absolute value of the correlation
    values.sort(key=lambda x: abs(x[1]))
    values = values[-n_print:]
    print(f'\ncorr with y = {y_value} given y >= {y_value}   ({len(x)}/{total_length})\n')
    for name, corr, names in values:
        print(f'{corr:+6.2%}   {name:28} (' + ", ".join(names) + ')')

    best_names = set(name for name, _, _ in values)
    return best_names


def main():
    best_names = set()
    for i in range(7):
        s = test_corr(y_value=i)
        best_names = best_names.union(s)

    print('\n\n BEST NAMES\n')
    for name in sorted(list(best_names)):
        print(name)


if __name__ == '__main__':
    main()
