"""This program shows the mean of the target variables for each category of the features"""
import numpy as np
import matplotlib.pyplot as plt
from data import Data


data = Data()
data.x_to_one_hot()
data.y_to_one_hot()
x, y = data.get_data()


names = list(x.columns)
n_targets = len(y.columns)

print(names)
print(n_targets)

mat = np.zeros((len(names), n_targets))
for i in range(len(names)):
    for j in range(n_targets):
        mat[i, j] = np.mean(y[x[names[i]] == 1].iloc[:, j])

mat = mat.T

plt.title('Mean of target variables for each category of the features')
plt.imshow(mat, aspect=1.618, cmap='magma')
plt.colorbar()
x_ticks = names

for i in range(len(names)):
    end = x_ticks[i].split('_')[-1]
    if end.isnumeric():
        if end == '0':
            x_ticks[i] = x_ticks[i][:-2]
        else:
            x_ticks[i] = ''

plt.xticks(range(len(names)), x_ticks, rotation=90)
plt.yticks(range(n_targets), y.columns)
plt.ylabel('number of visits')
plt.show()
