# Loan Prediction Streamlit App 🚀

This project is a **Streamlit-based Machine Learning App** for predicting whether a **loan application** will be **approved** or **denied** based on financial features.

It uses a trained **XGBoost model** (optimized on top 5 features) and is fully **Dockerized** for easy deployment.

---

## 📂 Project Structure

```text
Loan_Prediction_Project/
├── app/
│   └── app.py                      # Streamlit application
├── models/
│   └── xgb_model_5_features.pkl     # Trained XGBoost model
├── notebooks/
│   └── training_notebook.ipynb      # Data cleaning, feature selection, model training
├── Dockerfile                       # Dockerfile to containerize the app
├── requirements.txt                 # Python package dependencies
└── README.md                        # Project documentation
```

---

## ⚙️ Features

- Input financial data (Loan Amount, Annual Income, Loan-to-Value Ratio, Debt Ratio, Credit Score)
- Predict whether a loan application will be **Approved** or **Denied**
- Clean and modern web interface using **Streamlit**
- Dockerized for easy deployment

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/loan-prediction-project.git
cd loan-prediction-project
```

### 2. Install Dependencies (Local)

```bash
pip install -r requirements.txt
```

### 3. Run the App Locally

```bash
streamlit run app/app.py
```

Then open your browser and visit:

```
http://localhost:8501
```

---

## 🐳 Running with Docker

### Step 1: Build the Docker Image

```bash
docker build -t loan-prediction-app .
```

### Step 2: Run the Docker Container

```bash
docker run -p 8501:8501 loan-prediction-app
```

Then open your browser and visit:

```
http://localhost:8501
```

---

## 🎯 Model Details

- **Model**: XGBoost Classifier
- **Top 5 Features**:
  - Loan Amount
  - Annual Income
  - Loan-to-Value Ratio
  - Debt Ratio
  - Credit Score
- **Target**:
  - `1` = Denied
  - `0` = Approved

---

## 📈 Metrics Used

You can calculate the model's evaluation metrics using:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("R2 Score:", r2)
```

---

## 📊 Confusion Matrix Visualization

Plot the confusion matrix with:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Approved (0)", "Denied (1)"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()
```

---

## 📈 Future Improvements

- Improve hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Add user authentication (Login System)
- Save prediction history to a database
- Deploy the app on AWS / Azure / GCP

---

## 🤝 Contributions

- Fork this project
- Create a new branch
- Make your changes
- Submit a pull request 🚀

---

## 🧑‍💻 Author

- **Name**: [Your Name Here]
- **GitHub**: [Your GitHub Profile Link]

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

👉 **This is a complete, clean, GitHub-ready README file!**

