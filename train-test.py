import pandas as pd
from regression import PolynomialRegression

data = pd.read_csv('data.csv')
data = data.dropna()

X = data['x']
y = data['y']

model = PolynomialRegression(degree=3)
model.fit(X, y)
model.plot(X, y)
print(model.predict([2]))