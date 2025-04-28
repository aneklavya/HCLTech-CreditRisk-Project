Loan Prediction Streamlit App 🚀
This project is a Streamlit-based Machine Learning App for predicting whether a loan application will be approved or denied based on a few financial features.

It uses a trained XGBoost model (optimized on the top 5 features) and is fully Dockerized for easy deployment.

📂 Project Structure
bash
Copy
Edit
Loan_Prediction_Project/
├── app/
│   ├── app.py                      # Streamlit application
├── models/
│   └── xgb_model_5_features.pkl     # Trained XGBoost model
├── notebooks/
│   └── training_notebook.ipynb      # Data cleaning, feature selection, model training
├── Dockerfile                       # Dockerfile to containerize the app
├── requirements.txt                 # Python package dependencies
└── README.md                        # Project documentation
⚙️ Features
Input financial data (Loan Amount, Annual Income, Loan-to-Value Ratio, Debt Ratio, Credit Score)

Predict if a loan application is Approved or Denied

Displays clean and clear results

Built with Streamlit for a modern web app interface

Packaged with Docker for easy deployment anywhere

🛠 Setup Instructions
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/loan-prediction-project.git
cd loan-prediction-project
2. Install dependencies (for local running)
bash
Copy
Edit
pip install -r requirements.txt
3. Run the app locally (without Docker)
bash
Copy
Edit
streamlit run app/app.py
Then open your browser and navigate to:

http://localhost:8501

🐳 Running with Docker
Step 1: Build the Docker image
bash
Copy
Edit
docker build -t loan-prediction-app .
Step 2: Run the Docker container
bash
Copy
Edit
docker run -p 8501:8501 loan-prediction-app
The app will be available at:

http://localhost:8501

🎯 Model Details
Model: XGBoost Classifier

Features Used:

Loan Amount

Annual Income

Loan-to-Value Ratio

Debt Ratio

Credit Score

Target Labels:

1 = Denied

0 = Approved

📈 Future Improvements
Improve hyperparameter tuning for even higher accuracy

Add login authentication

Deploy on cloud platforms (AWS / GCP / Azure)

Add database integration to save prediction history