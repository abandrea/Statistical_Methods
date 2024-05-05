import numpy as np
import matplotlib.pyplot as plt
from handler import Handler


handler = Handler()
handler.shuffle()
df = handler.get_df()

# sort by doctorco
df = df.sort_values(by='doctorco', ascending=True)

# print all the lines in df, but highlight the ones with doctorco == 0
# x_limit = 2
# for i, line in enumerate(df.head(100).to_string().split('\n')):
#     if df.iloc[i-1]['doctorco'] <= x_limit:
#         print(line)
#     else:
#         print('\033[1;40m' + line + '\033[0m')

# print all different values for each column
# for name in df.columns:
#     print()
#     print(name)
#     v = sorted(list(set(df[name].to_numpy())))
#     print(v)

# print average value if doctorco >= x_limit and if doctorco < x_limit
names = list(df.columns)
names.remove(Handler.TARGET)

y_max = 9
mat = np.zeros((len(names), y_max))

for y_limit in range(y_max):
    # print(f'\ny_limit = {y_limit}')
    y = df[Handler.TARGET].to_numpy()
    values = dict()
    for i, name in enumerate(names):
        x = df[name].to_numpy()

        mask = y > y_limit
        b = np.mean(x[mask])
        b_var = np.var(x[mask])

        mask = y <= y_limit
        a = np.mean(x[mask])
        a_var = np.var(x[mask])

        c = min(a, b) / max(a, b)
        values[name] = c

        mat[i, y_limit] = 1 - c

    # sort values by ratio
    # for name in sorted(values, key=values.get, reverse=True):
    #     if values[name] < 0.4:
    #         print(f'{name:>15}  {values[name]:.1%}')

z_lim = 0.5
mat = np.clip(mat, z_lim, 1)
mat = (mat - z_lim) / (1 - z_lim)
mat = mat.T

plt.imshow(mat, cmap='hot')
plt.title('feature importance')
plt.colorbar()
plt.xticks(ticks=range(len(names)), labels=names, rotation=70)
plt.yticks(ticks=range(y_max), labels=[str(i) for i in range(y_max)])
plt.ylabel('doctorco')
plt.show()
