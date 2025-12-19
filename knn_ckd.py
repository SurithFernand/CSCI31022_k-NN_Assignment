# k-NN_ChronicKidneyDisease.py

import pandas as pd         # Load the CSV Dataset
import numpy as np          # Handle Missing Values
from sklearn.preprocessing import LabelEncoder      # Encode Categorical Values
from sklearn.preprocessing import StandardScaler        # Normalize Features
from sklearn.model_selection import train_test_split        # Split Dataset
from sklearn.neighbors import KNeighborsClassifier         # Train k-NN & Find Best k 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt         # Plot k vs Validation Accuracy
from sklearn.metrics import confusion_matrix, classification_report         # Final Evaluation on TEST Set


# Load dataset

df = pd.read_csv("chronic_kidney_disease.csv")

print(df.head())
print("Dataset shape:", df.shape)

print(df.columns.tolist())



# Handle Missing Values

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert numeric columns properly
numeric_cols = [
    'age','bp','bgr','bu','sc','sod','pot','hemo',
    'pcv','wbcc','rbcc'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
for col in df.columns:
    if df[col].dtype != object:
        # Numeric → Median
        df[col].fillna(df[col].median(), inplace=True)
    else:
        # Categorical → Mode
        df[col].fillna(df[col].mode()[0], inplace=True)

# print("Missing values handled using median & mode")
# print(df.head())




# Encode Categorical Values (Text → Numbers) , since k-NN cannot work with text.

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == object:
        df[col] = le.fit_transform(df[col])




# Separate Features and Class

X = df.drop("class", axis=1)
y = df["class"]




# Normalize Features

scaler = StandardScaler()
X = scaler.fit_transform(X)




# Split Dataset

# First split → 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

print("Training samples:", len(X_train))

# Second split → 20% validation, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))




# Train k-NN & Find Best k (VALIDATION SET)

k_values = range(1, 21)
val_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    val_accuracies.append(acc)
    print(f"k = {k}, Validation Accuracy = {acc:.4f}")




# Plot k vs Validation Accuracy

plt.plot(k_values, val_accuracies, marker='o')
plt.xlabel("k value")
plt.ylabel("Validation Accuracy")
plt.title("Effect of k on Validation Accuracy")
plt.grid(True)
plt.show()




# Train Final Model with Best k

best_k = k_values[val_accuracies.index(max(val_accuracies))]
print("Best k:", best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)




# Final Evaluation on TEST Set

y_test_pred = knn.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
