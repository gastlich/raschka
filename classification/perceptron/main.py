import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from classification.helpers import plot_decision_regions

from perceptron import Perceptron

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None)

# desired classes
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# features values
X = df.iloc[0: 100, [0, 2]].values

ppn = Perceptron()
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.savefig('misclassifications.png')
plt.close()

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.savefig('decision_regions.png')
plt.close()
