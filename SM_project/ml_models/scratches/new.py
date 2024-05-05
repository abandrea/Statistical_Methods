import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from pipeline import Data


def example(train_size=0.9):
    x, y = Data().get_data()

    df_original = x.copy()
    df_original['doctorco'] = y

    # pre-proces x
    for name in x.columns:
        n_values = len(set(x[name]))
        if n_values > 2:
            x = pd.get_dummies(x, columns=[name], drop_first=True)
    x = x.astype(int)

    # pre-proces y
    y = pd.get_dummies(y, drop_first=False)
    y = y.astype(int)

    # train-test split
    random_indices = x.sample(frac=1).index
    n_train = int(train_size * len(random_indices))
    train_indices = random_indices[:n_train]
    test_indices = random_indices[n_train:]

    x_train = x.loc[train_indices]
    x_test = x.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    # train
    model = MLPClassifier(hidden_layer_sizes=(200, 200),
                          max_iter=200,
                          random_state=0,
                          verbose=True,
                          activation='relu',
                          solver='adam',
                          learning_rate='adaptive',
                          learning_rate_init=0.01,
                          alpha=0.01,
                          tol=1e-3,
                          n_iter_no_change=20,
                          )

    model.fit(x_train, y_train)

    # predict and convert one-hot back to original
    y_pred = model.predict(x_test).argmax(axis=1)

    y_train = y_train.idxmax(axis=1)
    y_test = y_test.idxmax(axis=1)
    df_original['predicted'] = model.predict(x).argmax(axis=1)

    # show examples of predicted vs actual

    # keep only test set values
    df_original = df_original.loc[test_indices]
    print()
    print(df_original.head(100).to_string())


    # calculate rmse
    rmse = ((y_test - y_pred) ** 2).mean() ** 0.5
    print(f'      RMSE = {rmse:.2f}')

    y_pred_train = model.predict(x_train).argmax(axis=1)
    rmse_train = ((y_train - y_pred_train) ** 2).mean() ** 0.5
    print(f'RMSE train = {rmse_train:.2f}')


    # calculate dummy rmse
    y_dummy = np.zeros(y_test.shape)
    dummy_rmse = ((y_test - y_dummy) ** 2).mean() ** 0.5
    print(f'Dummy RMSE = {dummy_rmse:.2f}')

    # calculate usefulness
    usefulness = (dummy_rmse - rmse) / dummy_rmse
    print(f'Usefulness = {usefulness:.1%}')



if __name__ == '__main__':
    example()
