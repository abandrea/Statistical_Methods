from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from handler import Handler


# get the data
handler = Handler()
x_train, x_test, y_train, y_test = handler.train_test_split()

y_train = np.log(1 + y_train)

# train a model
model = MLPRegressor(hidden_layer_sizes=(100, 20),
                     max_iter=500,
                     random_state=0,
                     verbose=True,
                     activation='relu',
                     solver='adam',
                     learning_rate='adaptive',
                     learning_rate_init=0.01,
                     alpha=0.01,
                     tol=1e-4,
                     n_iter_no_change=10,
                     )
model.fit(x_train, y_train)


# make predictions
y_pred = model.predict(x_test)
y_pred = np.exp(y_pred) - 1


# show examples of predicted vs actual
print('\nPredicted vs actual:\n')
df_demo = x_test.copy()
df_demo['actual'] = y_test
df_demo['predicted'] = y_pred
print(df_demo.head(40).to_string())


# evaluate the model
rmse = root_mean_squared_error(y_test, y_pred)
dummy_rmse = np.sqrt(np.mean((y_test - np.mean(y_train))**2))
usefulness = (dummy_rmse - rmse) / dummy_rmse

print()
print(f"      rmse = {rmse:.3f}")
print(f"dummy_rmse = {dummy_rmse:.3f}")
print(f"usefulness = {usefulness:.1%}")

plt.scatter(y_test, y_pred, alpha=0.2)
plt.title('Predicted vs actual')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.plot([0, 9], [0, 9], color='k', linestyle='--')
plt.show()
