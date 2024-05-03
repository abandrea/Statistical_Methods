from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import root_mean_squared_error
from handler import Handler
import numpy as np


# get the data
handler = Handler()
x_train, x_test, y_train, y_test = handler.train_test_split()

# remove rows where the target is zero
x_train = x_train[y_train > 0]
y_train = y_train[y_train > 0]
x_test = x_test[y_test > 0]
y_test = y_test[y_test > 0]

# train a model
model = PoissonRegressor()
model.fit(x_train, y_train)

# make predictions
y_pred = model.predict(x_test)

rmse = root_mean_squared_error(y_test, y_pred)
dummy_rmse = np.sqrt(np.mean((y_test - np.mean(y_train))**2))
rel_err = np.mean(np.abs(y_test - y_pred) / y_test)
usefulness = (dummy_rmse - rmse) / dummy_rmse
print()
print(f"      rmse = {rmse:.3f}")
print(f"dummy_rmse = {dummy_rmse:.3f}")
print(f"   rel_err = {rel_err:.1%}")
print(f"usefulness = {usefulness:.1%}")

# show examples of misclassified data
print('\nPredicted vs actual:\n')
df_demo = x_test.copy()
df_demo['actual'] = y_test
df_demo['predicted'] = y_pred
print(df_demo.head(15).to_string())