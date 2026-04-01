import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay
"""

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')

mushrooms_dataset_path = "C:\\Users\\conor\\Documents\\msiss\\information systems\\Group Project\\mushrooms.csv"
df = pd.read_csv(mushrooms_dataset_path)
print(df.iloc[:, 1:3])
print(df.iloc[:, 1:5].describe())
print(np.unique(df.columns))

df['class'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


'''
iris = load_iris()
iris
X = iris.data
y = iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

print('Iris dataset: 3 classes, 4 features, 150 instances')

print('Dataset shape:', X.shape)
print('Features:', iris.feature_names)
print('Classes:', iris.target_names)
print('Class distribution:', np.bincount(y))
'''
git config --global user.name "ConorMoneyBro"
git config --global user.email "bowec1@tcd.com"