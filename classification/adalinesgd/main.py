import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from classification.helpers import plot_decision_regions

from adalinesgd import AdalineSGD

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None)

# desired classes
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# features values
X = df.iloc[0: 100, [0, 2]].values

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada_std = AdalineSGD(n_iter=15, eta=0.01, random_state=1).fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_std)

plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.savefig('decision_regions.png')
plt.close()

plt.plot(range(1, len(ada_std.cost_) + 1), ada_std.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.title('Adaline - Learning rate 0.01')
plt.savefig('sse_std.png')
plt.close()
