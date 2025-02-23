import matplotlib.pyplot as plt

# Data
X = [7.8, 7.7, 5.8, 7.0, 7.6, 7.0, 8.1, 7.8, 7.3, 7.5, 8.6, 6.5, 9.0, 6.7, 7.4, 6.7,
     7.3, 6.4, 7.0, 9.2, 7.4, 6.8, 6.6, 7.1, 6.4, 7.9, 6.9, 7.6, 7.1, 6.7, 7.5, 7.4,
     8.2, 7.3, 6.2, 6.6, 6.3, 8.0, 6.8, 5.3, 7.4, 6.8, 6.3, 6.9, 6.6, 7.3, 6.6, 7.6,
     7.7, 6.6, 9.1, 6.4, 7.0, 6.0, 7.8]
Y = [32700.0, 1000.0, 1200.0, 3000.0, 3912.0, 10000.0, 2150.0, 2550.0, 60000.0, 12000.0,
     4800.0, 1100.0, 11168.0, 1243.0, 465.0, 158.0, 28.0, 60.0, 12225.0, 131.0,
     400.0, 8064.0, 300.0, 15900.0, 3000.0, 70000.0, 875.0, 2199.0, 10820.5, 2311.0,
     1578.0, 20000.0, 450.0, 3816.5, 2800.0, 1340.0, 14.0, 25000.0, 37500.0, 274.0,
     37500.0, 1384.0, 9748.0, 6434.0, 322.0, 2394.0, 4500.0, 17127.0, 20085.0,
     26271.0, 227898.0, 28903.0, 204000.0, 1163.0, 57350.0]

# Calculate mean
def calculate_mean(values):
    return sum(values) / len(values)

X_mean = calculate_mean(X)
Y_mean = calculate_mean(Y)

# Calculate slope (a) and intercept (b)
numerator = sum((X[i] - X_mean) * (Y[i] - Y_mean) for i in range(len(X)))
denominator = sum((X[i] - X_mean) ** 2 for i in range(len(X)))
a = numerator / denominator
b = Y_mean - a * X_mean

# Predicted values
Y_pred = [a * x + b for x in X]

# Print results
print("Slope (a):", a)
print("Intercept (b):", b)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.title('Simple Linear Regression (Pure Python)')
plt.xlabel('Magnitude')
plt.ylabel('Deaths')
plt.legend()
plt.grid()
plt.show()

# Predict for new value
new_x = 7.7
predicted_y = a * new_x + b
print(f"Predicted deaths for magnitude {new_x}: {predicted_y}")
