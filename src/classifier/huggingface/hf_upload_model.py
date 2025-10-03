from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

def upload_model_to_huggingface():
    HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
    HF_ENDPOINT = os.environ.get("HUGGINGFACE_ENDPOINT")
    REPO_ID = os.environ.get("HUGGINGFACE_REPO")
    MODEL_PATH = os.environ.get("HUGGINGFACE_MODEL_PATH")

    api = HfApi(
        endpoint=HF_ENDPOINT
        token=HF_TOKEN
    )
    api.upload_file(
        path_or_fileobj="classifier/models/supervised_hetero.pt",
        path_in_repo=MODEL_PATH,
        repo_id=REPO_ID,
        repo_type="model",
    )
