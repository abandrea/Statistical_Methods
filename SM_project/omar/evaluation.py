"""evaluation of the model's performance"""
import numpy as np
import matplotlib.pyplot as plt
from time import time


def rmse(y_test, y_pred):
    """mean squared error of the predicted probability distribution"""
    return np.mean((y_test - y_pred)**2) ** .5


def abs_error(y_test, y_pred):
    """mean absolute error of the predicted values"""
    return np.mean(np.abs(y_test - y_pred))


def evaluate(y_train, y_test, y_pred, verbose=True, plot=True,
             n_rows=30, n_cols=8, colorbar=True, title='Predictions', log=False, log_limit=6):
    """evaluate the model's performance"""

    if verbose:
        print('\n\n Evaluaiton \n')

    # argmax
    y_test_argmax = np.argmax(y_test, axis=1)
    y_pred_argmax = np.argmax(y_pred, axis=1)

    # dummy predictions
    y_dummy_argmax = np.zeros(len(y_test), dtype=int)  # todo most common

    dummy_pred = np.mean(y_train, axis=0).to_numpy()

    y_dummy_distrib = np.outer(np.ones(len(y_test)), dummy_pred)

    # performance
    assert y_test.shape == y_pred.shape, f'y_test and y_pred must have the same shape ({y_test.shape}!={y_pred.shape})'
    rmse_val = rmse(y_test, y_pred)
    mean_abs_error_val = abs_error(y_test_argmax, y_pred_argmax)

    # dummy performance
    dummy_rmse = rmse(y_test, y_dummy_distrib)
    dummy_abs_error = abs_error(y_test_argmax, y_dummy_argmax)

    # usefulness
    usefulness_rmse = 1 - (rmse_val / dummy_rmse)
    usefulness_abs_error = 1 - (mean_abs_error_val / dummy_abs_error)

    if verbose:
        print(f'                RMSE = {rmse_val:.3f}')
        print(f'      mean_abs_error = {mean_abs_error_val:.3f}')
        print()
        print(f'          dummy_rmse = {dummy_rmse:.3f}')
        print(f'         dummy_mabse = {dummy_abs_error:.3f}')
        print()
        print(f'     usefulness_rmse = {usefulness_rmse:.1%}')
        print(f'    usefulness_mabse = {usefulness_abs_error:.1%}')
        print()

    # plot the predictions
    if plot:

        shape = y_test.shape

        fig, axs = plt.subplots(1, n_cols)
        plt.suptitle(title)

        for i in range(n_cols):
            plt.sca(axs[i])

            # plot the predictions
            mat = y_pred[n_rows * i:n_rows * (i + 1)]
            if log:
                mat = np.log(mat + 10**(-log_limit))
            if log:
                cmap = 'viridis'
            else:
                cmap = 'magma'
            plt.imshow(mat, aspect='auto', cmap=cmap)
            x_ = [str(j) for j in range(shape[-1])]
            plt.xticks(list(range(shape[-1])), x_)
            if log:
                plt.clim(-log_limit, 0)
            else:
                plt.clim(0, 1)
            if colorbar:
                if i + 1 == n_cols:
                    plt.colorbar()

            # plot the true values
            _x = y_test_argmax[n_rows * i:n_rows * (i + 1)]
            _y = range(n_rows)
            plt.scatter(_x, _y, c='w', s=20, edgecolors='k')
            plt.yticks([])

        plt.show()

    return {'rmse': rmse_val, 'mabs_error': mean_abs_error_val,
            'dummy_rmse': dummy_rmse, 'dummy_mabs_error': dummy_abs_error,
            'usefulness_rmse': usefulness_rmse, 'usefulness_mabs_error': usefulness_abs_error}


def repeated_evaluation(training_method, n=10):
    """evaluate the model's performance multiple times"""
    rmse = []
    mabse = []
    dummy_rmse = []
    dummy_mabse = []
    training_times = []

    for i in range(n):
        print(f'{i + 1}/{n} ', end='\t')
        t = time()
        results = training_method()
        training_times.append(time() - t)
        print(results)
        rmse.append(results['rmse'])
        mabse.append(results['mabs_error'])
        dummy_rmse.append(results['dummy_rmse'])
        dummy_mabse.append(results['dummy_mabs_error'])

    return {'avg_rmse': np.mean(rmse),
            'avg_mabse': np.mean(mabse),
            'std_rmse': np.std(rmse),
            'std_mabse': np.std(mabse),
            'avg_dummy_rmse': np.mean(dummy_rmse),
            'avg_dummy_mabse': np.mean(dummy_mabse),
            'avg_training_time': np.mean(training_times),
            'std_training_time': np.std(training_times)}
