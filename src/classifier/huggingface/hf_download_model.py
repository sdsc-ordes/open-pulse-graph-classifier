from dotenv import load_dotenv
import os

load_dotenv()


def download_model_from_huggingface():
    REPO_ID = os.environ.get("HUGGINGFACE_REPO")
    MODEL_PATH = os.environ.get("HUGGINGFACE_MODEL")

    hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_PATH,
        local_dir="classifier/models/supervised_hetero.pt",
    )
