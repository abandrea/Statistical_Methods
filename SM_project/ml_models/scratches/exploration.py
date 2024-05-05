import matplotlib.pyplot as plt
import numpy as np
from handler import Handler


np.random.seed(0)

handler = Handler()
handler.shuffle()
df = handler.get_df()

print(df.head(30).to_string())


# PLOT 1
for name in df.columns:
    print(name)
    x = df[name].to_numpy()
    y = df[Handler.TARGET].to_numpy()

    # sort x and y by x
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]

    x_avg = sorted(list(set(x_sorted)))
    y_avg = []
    y_std = []

    for i in range(len(x_avg)):
        mask = x_sorted == x_avg[i]
        y_avg.append(np.mean(y_sorted[mask]))
        y_std.append(np.std(y_sorted[mask]))

    plt.scatter(x_sorted, y_sorted, alpha=0.2)
    plt.plot(x_avg, y_avg, color='red')
    plt.fill_between(x_avg, np.array(y_avg) - np.array(y_std), np.array(y_avg) + np.array(y_std), color='red', alpha=0.2)
    plt.title(name + ' vs ' + Handler.TARGET)
    plt.xlabel(name)
    plt.ylabel(Handler.TARGET)
    plt.show()


# # PLOT 2
# x1 = df['illness'].to_numpy()
# x2 = df['income'].to_numpy()
#
# y = df['doctorco'].to_numpy()
# indices = y == 0
# x1_zero = x1[indices]
# x2_zero = x2[indices]
#
# n = 300
# indices = np.random.permutation(len(x1_zero))[:n]
# x1_zero = x1_zero[indices]
# x2_zero = x2_zero[indices]
# indices = np.random.permutation(len(x1))[:n]
# x1 = x1[indices]
# x2 = x2[indices]
#
#
# dx = 0.1
# x1_zero = np.float32(x1_zero) + np.random.normal(0, dx, len(x1_zero))
# x2_zero = np.float32(x2_zero) + np.random.normal(0, dx, len(x2_zero))
# x1 = np.float32(x1) + np.random.normal(0, dx, len(x1))
# x2 = np.float32(x2) + np.random.normal(0, dx, len(x2))
#
# plt.scatter(x1, x2, alpha=0.5, label='target=1', color='white', edgecolor='black')
# plt.scatter(x1_zero, x2_zero, alpha=0.5, label='target=0', color='black', edgecolor='black')
# plt.title('prescrib vs illness')
# plt.xlabel('illness')
# plt.ylabel('prescrib')
# plt.legend()
# plt.show()
