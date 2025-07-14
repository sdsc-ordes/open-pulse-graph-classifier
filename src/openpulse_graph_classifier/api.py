from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from dotenv import load_dotenv
import torch
import os

from openpulse_graph_classifier.data_extraction import extract_data
from openpulse_graph_classifier.data_transformer import data_transformer
from openpulse_graph_classifier.inference import inference

app = FastAPI()

security = HTTPBearer()

load_dotenv()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")


def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if payload.get("service") != "airflow":
            raise HTTPException(status_code=403, detail="Invalid service identity")
        return payload
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid token")


@app.get("/test")
async def test():
    return {"message": "Test endpoint is working!"}


@app.get("/inference/{neo4j_database}")
async def do_inference(neo4j_database: str, token_data: dict = Depends(verify_jwt)):
    output = inference(neo4j_database)
    # should the predictions be returned in the response or should it be saved to NEO4J or something?
    return {"output": output}
