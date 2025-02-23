import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score

# Data
X = np.array([7.8, 7.7, 5.8, 7.0, 7.6, 7.0, 8.1, 7.8, 7.3, 7.5, 8.6, 6.5, 9.0, 6.7, 7.4, 6.7,
              7.3, 6.4, 7.0, 9.2, 7.4, 6.8, 6.6, 7.1, 6.4, 7.9, 6.9, 7.6, 7.1, 6.7, 7.5, 7.4,
              8.2, 7.3, 6.2, 6.6, 6.3, 8.0, 6.8, 5.3, 7.4, 6.8, 6.3, 6.9, 6.6, 7.3, 6.6, 7.6,
              7.7, 6.6, 9.1, 6.4, 7.0, 6.0, 7.8])
Y = np.array([32700.0, 1000.0, 1200.0, 3000.0, 3912.0, 10000.0, 2150.0, 2550.0, 60000.0, 12000.0,
              4800.0, 1100.0, 11168.0, 1243.0, 465.0, 158.0, 28.0, 60.0, 12225.0, 131.0,
              400.0, 8064.0, 300.0, 15900.0, 3000.0, 70000.0, 875.0, 2199.0, 10820.5, 2311.0,
              1578.0, 20000.0, 450.0, 3816.5, 2800.0, 1340.0, 14.0, 25000.0, 37500.0, 274.0,
              37500.0, 1384.0, 9748.0, 6434.0, 322.0, 2394.0, 4500.0, 17127.0, 20085.0,
              26271.0, 227898.0, 28903.0, 204000.0, 1163.0, 57350.0])

# Log transformation
Y_log = np.log(Y + 1)
x = X.reshape(-1, 1)
y = Y_log

# Model
model_tf = Sequential()
model_tf.add(Dense(1, input_dim=1, activation='linear'))
model_tf.compile(optimizer='sgd', loss='mse')
model_tf.fit(x, y, epochs=200, verbose=0)

# Predictions
y_pred_tf = model_tf.predict(x, verbose=0)

# Evaluation
mse_tf = mean_squared_error(y, y_pred_tf)
r2_tf = r2_score(y, y_pred_tf)
print(f"TensorFlow MSE: {mse_tf:.2f}, R²: {r2_tf:.2f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual Data (Log Transformed)', alpha=0.7)
plt.plot(x, y_pred_tf, color='red', label='TensorFlow Regression Line', linewidth=2)
plt.title('Log Transformed Data - Linear Regression (TensorFlow)')
plt.xlabel('Magnitude (X)')
plt.ylabel('Log(Deaths + 1) (Y)')
plt.legend()
plt.grid(True)
plt.show()
