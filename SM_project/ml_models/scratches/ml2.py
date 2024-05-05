import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from handler import Handler
from scipy.stats import mode


class NormalizedModel:
    """
    normalize training set
    apply the same normalization to the test set
    """
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.parameters = dict()

        self.normalization_parameters()

    def normalization_parameters(self):
        """compute min-max normalization parameters"""
        # for name in self.x_train.columns:
        #     min_ = self.x_train[name].min()
        #     max_ = self.x_train[name].max()
        #     self.parameters[name] = (min_, max_)

    def normalize_x(self, df):
        # apply the same normalization to the test set
        df_copy = df.copy()
        for name in df_copy.columns:
            # min_, max_ = self.parameters[name]
            # df_copy[name] = (df_copy[name] - min_) / (max_ - min_)
            df_copy[name] = np.log2(1+df_copy[name])
        return df_copy

    def normalize_y(self, y):
        return y

    def denormalize_y(self, y):
        out = np.round(y)
        out = np.clip(out, 0, None)
        return np.int32(out)

    def train(self):
        x_train = self.normalize_x(self.x_train)
        y_train = self.normalize_y(self.y_train)
        self.model.fit(x_train, y_train)

    def predict(self, df_x):

        df = self.normalize_x(df_x)
        y_pred = self.model.predict(df)
        y_pred = self.denormalize_y(y_pred)
        return y_pred

def evaluate(x_test, y_test, y_pred):
    rmse = root_mean_squared_error(y_test, y_pred)
    most_common_class = mode(y_test)[0]
    y_dummy = np.zeros(len(y_test)) + most_common_class
    dummy_rmse = np.sqrt(np.mean((y_test - y_dummy) ** 2))
    usefulness = (dummy_rmse - rmse) / dummy_rmse

    print('\nPredicted vs actual:\n')
    df_demo = x_test.copy()
    df_demo['actual'] = y_test
    df_demo['predicted'] = y_pred
    print(df_demo.head(100).to_string())

    print()
    print(f"      rmse = {rmse:.3f}")
    print(f"dummy_rmse = {dummy_rmse:.3f}")
    print(f"usefulness = {usefulness:.1%}")

    plt.scatter(y_test, y_pred, alpha=0.2)
    plt.title('Predicted vs actual')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    max_y = max(y_test.max(), y_pred.max()) + 1
    plt.plot([0, max_y], [0, max_y], color='k', linestyle='--')
    plt.show()


def example():
    handler = Handler()
    # handler.df = handler.df[['illness', 'actdays', 'hscore',
    #                          'nondocco', 'hospadmi', 'medicine',
    #                          'prescrib', 'doctorco']]
    x_train, x_test, y_train, y_test = handler.train_test_split()

    # model
    # model = MLPRegressor(hidden_layer_sizes=(50,),
    #                      max_iter=500,
    #                      random_state=0,
    #                      verbose=True,
    #                      activation='relu',
    #                      solver='adam',
    #                      learning_rate='adaptive',
    #                      learning_rate_init=0.01,
    #                      alpha=0.01,
    #                      tol=1e-5,
    #                      n_iter_no_change=10,
    #                      )

    model = LinearRegression()

    # model = RandomForestRegressor(n_estimators=100,
    #                               random_state=0)

    normalized_model = NormalizedModel(model, x_train, y_train, x_test, y_test)
    normalized_model.train()
    y_pred = normalized_model.predict(x_test)

    # y_pred = np.int32((x_test['illness'].to_numpy() >= 3) *
    #                   (x_test['prescrib'].to_numpy() >= 2))

    evaluate(x_test, y_test, y_pred)


if __name__ == '__main__':
    example()
