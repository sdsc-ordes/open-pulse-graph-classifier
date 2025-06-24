from jose import jwt
from dotenv import load_dotenv
import os
import requests

load_dotenv()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")

INFERENCE_URL = os.getenv("INFERENCE_URL")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

payload = {"service": "airflow", "exp": 9999999999}

token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

headers = {"Authorization": f"Bearer {token}"}

res = requests.get(INFERENCE_URL + f"/inference/{NEO4J_DATABASE}", headers=headers)
