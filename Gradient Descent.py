import numpy as np
import matplotlib.pyplot as plt

epochs        = 3    # Numarul de treceri prin setul de date
lr            = 0.01  # Learning rate

# MSE - Mean Squarred Error
def loss_function(y_pred, y_true):
    number_of_examples = y_true.shape[0]
    return np.sum(np.square(y_true - y_pred)) / number_of_examples

# Setul de antrenare
X = np.array([2, 2.5, 3, 3.4, 4.2, 5.1, 5.8, 6, 6.5, 7, 7.3, 8, 8.6, 9, 10])
y = np.array([30, 34, 39, 46, 55, 61, 58, 66, 75, 80, 88, 91, 92, 94, 99])

n = X.shape[0]  # Numar de exemple din setul de antrenare
a = 0
b = 0

loss_function_values = []
for _ in range(epochs):
    y_pred = a*X + b
    loss_function_values.append(loss_function(y_pred, y))
    deriv_a = (-2 / n) * np.sum((y - y_pred) * X)  # Derivata partiala a functiei de cost in raport cu a
    deriv_b = (-2 / n) * np.sum((y - y_pred))  # Derivata partiala a functiei de cost in raport cu b
    a = a - lr * deriv_a
    b = b - lr * deriv_b

points = np.array([min(X), max(X)])
plt.figure(1)
plt.plot(list(range(epochs)), loss_function_values)
plt.title('Variația funcției de cost în funcție de numărul de epoci')
plt.xlabel('Număr de Epoci')
plt.ylabel('MSE')
plt.show()

plt.figure(2)
plt.scatter(X, y)
#fit function
f = lambda x: a*x + b


x = np.array([min(X), max(X)])

plt.plot(x, f(x), c="orange")
plt.xlabel('Număr ore dormite(var.independentă)')
plt.ylabel('Indice Performanță(var.dependentă)')
plt.show()

