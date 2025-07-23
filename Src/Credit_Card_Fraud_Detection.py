import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/card_transdata.csv")
print("Dataset loaded successfully.")

print("Dataset shape:", df.shape)
print("First few rows of the dataset:")
print(df.head())
print("Missing values in each column:")
print(df.isnull().sum())

print("Column names in the dataset:")
print(df.columns)

target_column = 'fraud'  # Updated to match your dataset
if target_column in df.columns:
    print("Target column found.")
else:
    print(f"Error: The target column '{target_column}' does not exist in the dataset.")
    exit()
    
if 'distance_from_home' in df.columns:
    df['Normalized_distance_from_home'] = (df['distance_from_home'] - df['distance_from_home'].mean()) / df['distance_from_home'].std()
    df.drop(['distance_from_home'], axis=1, inplace=True)
    
X = df.drop([target_column], axis=1)  # Features
y = df[target_column]                  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Split data into training set ({X_train.shape[0]} samples) and testing set ({X_test.shape[0]} samples).")

model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best model found through GridSearchCV.")

best_model.fit(X_train, y_train)
print("Model trained successfully.")

y_pred = best_model.predict(X_test)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

confusion_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
