import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Initialize HF API with token
api = HfApi(token=os.getenv("HF_TOKEN"))

# HuggingFace dataset path
DATASET_PATH = "hf://datasets/Parthipan00410/Bank-Customer-Churn-Dataset/bank_customer_churn.csv"

# Load dataset
bank_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Target column
target = "Exited"

# Numerical features
numeric_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

# Categorical features
categorical_features = ["Geography"]

# Feature matrix (X)
X = bank_dataset[numeric_features + categorical_features]

# Target variable (y)
y = bank_dataset[target]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Save split datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train-test files saved.")

# Upload each file to HuggingFace dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id="Parthipan00410/Bank-Customer-Churn-Dataset",  # ✅ FIXED
        repo_type="dataset",
    )
