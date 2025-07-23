# 🛡️ Credit Card Fraud Detection

This project demonstrates a machine learning pipeline for detecting fraudulent credit card transactions using a **Random Forest Classifier**. The dataset is preprocessed, tuned with `GridSearchCV`, and evaluated with performance metrics including a confusion matrix and classification report.

---

## 📁 Dataset

- File: `data/card_transdata.csv`
- Target column: `fraud`
- Features include transaction distance, merchant location, and transaction time, among others.

---

## 🧰 Tools & Libraries Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🚀 Project Workflow

1. **Load Dataset**  
   Load and inspect the CSV dataset, check for missing values and relevant columns.

2. **Data Preprocessing**  
   - Normalize `distance_from_home`.
   - Drop the original column after normalization.

3. **Train-Test Split**  
   Split the data into 80% training and 20% testing using `train_test_split`.

4. **Model Building and Hyperparameter Tuning**  
   - Use `RandomForestClassifier`.
   - Perform grid search with cross-validation to find the best hyperparameters.

5. **Model Evaluation**  
   - Generate predictions on the test set.
   - Print classification report.
   - Plot confusion matrix.

---

## 🧪 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
