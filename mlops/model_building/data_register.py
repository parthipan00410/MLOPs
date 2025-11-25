from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi
import os

repo_id = "Parthipan00410/Bank-Customer-Churn-Dataset"
repo_type = "dataset"

# Initialize API client with token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Step 2: Upload contents
api.upload_folder(
    folder_path="mlops/data",   # make sure this path is correct
    repo_id=repo_id,
    repo_type=repo_type,
)
