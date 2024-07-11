import pandas as pd
import numpy as np
#Here I check with these data only
data = {
    'Temperature': [23, 25, 28, 22, 26, 27, 29, 30, 21, 24],
    'Rainfall': [150, 200, 250, 100, 180, 190, 210, 220, 110, 160],
    'Soil_Quality': [6, 7, 8, 5, 7, 7.5, 8.5, 9, 4.5, 6.5],
    'Yield': [2000, 2200, 2500, 1800, 2100, 2300, 2400, 2600, 1700, 2000]
}
df = pd.DataFrame(data)
#Declared the climate of field
X = df[['Temperature', 'Rainfall', 'Soil_Quality']]
y = df['Yield']
#one column for intercept to "X"
X = np.hstack((np.ones((X.shape[0], 1)), X))
#Here the main work will occurs like tresting
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

X_train_T_X_train = np.dot(X_train.T, X_train)
X_train_T_y_train = np.dot(X_train.T, y_train)
theta = np.linalg.solve(X_train_T_X_train, X_train_T_y_train)
#In this step the decesion of crop will takes 
y_pred = np.dot(X_test, theta)
#finding which type of model
mse = np.mean((y_pred - y_test) ** 2)
r2 = 1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
#Showing the mean square ans scare
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

#Complie time exampe purpose
example = np.array([1, 25, 190, 7.5])  
predicted_yield = np.dot(example, theta)
print(f"Predicted Yield: {predicted_yield}")
