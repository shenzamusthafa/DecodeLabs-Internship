
'''PROJECT 2 — DATA CLASSIFICATION USING AI
Dataset   : Iris Dataset
Algorithm : K-Nearest Neighbors (KNN)
Done by   : Shenza P M'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# Load Dataset

iris = load_iris()
X = iris.data
y = iris.target

# Sample Data

print("\nFirst 5 Samples:")
for i in range(5):
    print(X[i], "->", iris.target_names[y[i]])
print()

# EDA

print("EXPLORATORY DATA ANALYSIS")
print("Dataset Shape  :", X.shape)
print("Class Distribution:", np.bincount(y),
      "→", list(iris.target_names))

print("\nFeature Ranges:")
for i, name in enumerate(iris.feature_names):
    print(f"  {name:<25} "
          f"min={X[:,i].min():.1f}  "
          f"max={X[:,i].max():.1f}  "
          f"mean={X[:,i].mean():.2f}")

print("\nMissing Values :", np.isnan(X).sum(), "✓")
print()

# Class Distribution Graph

plt.figure(figsize=(6,4))
plt.hist(y)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

'''Split Dataset
stratify=y ensures both train and test sets keep the
same class ratio — important for reliable evaluation'''

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Train size :{len(X_train)} samples")
print(f"Test  size :{len(X_test)}  samples\n")

# Feature Scaling

scaler=StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN Classification

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("KNN Model Results")
print("Accuracy :", round(accuracy_score(y_test, y_pred_knn), 4))
print()

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred_knn)

print("Confusion Matrix:")
print(cm)
print()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix")
plt.show()

# Classification Report

print("Classification Report:")
print(classification_report(
    y_test,
    y_pred_knn,
    target_names=iris.target_names
))

# Finding Best K Value

k_values = []
accuracies = []

for k in range(1, 21):

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    k_values.append(k)
    accuracies.append(accuracy)

best_k = k_values[accuracies.index(max(accuracies))]
best_acc = max(accuracies)

print("PART 2—BEST K VALUE")

print("\nAccuracy for different K values:")

for k, acc in zip(k_values, accuracies):

    marker = " ← best" if k == best_k else ""

    print(f"K={k:2d}  |  Accuracy = {round(acc, 4)}{marker}")

print(f"\nBest K={best_k}  |  Accuracy = {best_acc:.4f}\n")

# Plot K Value vs Accuracy

plt.figure(figsize=(9,5))

plt.plot(
    k_values,
    accuracies,
    marker='o',
    color='#378ADD',
    linewidth=2,
    markersize=6
)

plt.axvline(
    best_k,
    color='#E24B4A',
    linestyle='--',
    linewidth=1.6,
    label=f'Best k = {best_k}'
)

plt.fill_between(
    k_values,
    accuracies,
    alpha=0.1,
    color='#378ADD'
)

plt.xlabel("K Value", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("K Value vs Accuracy")
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig("k_vs_accuracy.png", dpi=150)

plt.show()

print("✓ Plot saved → k_vs_accuracy.png\n")

# Compare with Decision Tree

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("Model Comparison")

print(f"  KNN Accuracy: {knn_accuracy:.4f}")
print(f"  Decision Tree Accuracy: {dt_accuracy:.4f}")

winner = "KNN" if knn_accuracy >= dt_accuracy else "Decision Tree"

print(f"Better model: {winner}\n")

print("Observation:")

if knn_accuracy > dt_accuracy:
    print("KNN performed slightly better on this dataset.\n")
elif dt_accuracy > knn_accuracy:
    print("Decision Tree performed slightly better on this dataset.\n")
else:
    print("Both models performed equally well on this dataset.\n")

# Predict New Flower

print("PART 4—PREDICT NEW FLOWER")

print("\nEnter flower measurements:\n")

sepal_length = float(input("Sepal Length (cm): "))
sepal_width = float(input("Sepal Width  (cm): "))
petal_length = float(input("Petal Length (cm): "))
petal_width = float(input("Petal Width  (cm): "))

sample = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

sample=scaler.transform(sample)

prediction=knn.predict(sample)

confidence=knn.predict_proba(sample).max() * 100

print(f"\nPredicted Species : {iris.target_names[prediction[0]]}")
print(f"Confidence: {confidence:.1f}%\n")

'''
Conclusion:

This project helped me understand how machine learning
classification works using the K-Nearest Neighbors algorithm.

I learned the importance of data preprocessing, especially
feature scaling, because KNN relies on distance calculations.

I also explored how changing K values affects model accuracy
and compared KNN with Decision Tree to understand how
different algorithms perform on the same dataset.

Overall, this project gave me practical experience with
training, evaluating and testing classification models.
'''