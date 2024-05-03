from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from handler import Handler


# get the data
handler = Handler()
handler.df['doctorco'] = handler.df['doctorco'] > 0
x_train, x_test, y_train, y_test = handler.train_test_split()

# train a model
model = LogisticRegression()
model.fit(x_train, y_train)

# make predictions
y_pred = model.predict(x_test)

# calculate accuracy
acc = accuracy_score(y_test, y_pred)
dummy_accuracy = max(y_test.mean(), 1 - y_test.mean())
f1_score = f1_score(y_test, y_pred)
usefulness = (acc - dummy_accuracy) / (1 - dummy_accuracy)
print()
print(f"      accuracy = {acc:.1%}")
print(f"dummy accuracy = {dummy_accuracy:.1%}")
print(f"      F1 score = {f1_score:.1%}")
print(f"    usefulness = {usefulness:.1%}")


# show examples of misclassified data
print('\nPredicted vs actual:\n')
df_demo = x_test.copy()
df_demo['actual'] = y_test
df_demo['predicted'] = y_pred
print(df_demo.head(15).to_string())
