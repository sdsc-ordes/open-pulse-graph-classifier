from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from dotenv import load_dotenv
import torch
import os

from classifier.inference.inference import inference

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
    except JWTError as e:
        print("JWT decode error:", e)
        raise HTTPException(status_code=403, detail="Invalid token")


@app.get("/v1/test")
async def test():
    return {
        "data": {
            "type": "test",
            "id": "1",
            "attributes": {"message": "Test endpoint is working!"},
        }
    }


@app.get("/v1/inference/epfl/{neo4j_database}")
async def do_inference(neo4j_database: str, token_data: dict = Depends(verify_jwt)):
    inference(neo4j_database)
    return {
        "data": {
            "type": "inference",
            "id": neo4j_database,
            "attributes": {
                "results": "Predictions were uploaded to your Neo4J database under the node property 'predition'."
            },
        }
    }
