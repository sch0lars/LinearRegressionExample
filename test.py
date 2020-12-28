import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import tensorflow

# The file with the student records.
file = 'student/student-mat.csv'

data = pd.read_csv(file, sep=';')
# Get the desired fields.
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

# Predict the final grade (G3)
predict = 'G3'

X = np.array(data.drop(predict, axis=1))
y = np.array(data[predict])

# Split the data up into testing and training data.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Create a linear regression model.
linear = linear_model.LinearRegression()

# Train the model.
linear.fit(X_train, y_train)
# Get the accuracy of the model.
acc = linear.score(X_test, y_test)

print(f'Accuracy: {acc*100:.2f}%')
print(f'Coefficients: {linear.coef_}')
print(f'Intercept: {linear.intercept_}')

# Test the predictions.
predictions = linear.predict(X_test)
print(f'{"Prediction":<15}{"X_test":<25}{"y_test"}')
for i in range(len(predictions)):
    print(f'{predictions[i]:<15.2f}{X_test[i].__str__():<25}{y_test[i]}')


