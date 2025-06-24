from fastapi import FastAPI
from pydantic import BaseModel
import torch

from data_extraction import extract_data
from data_transformer import data_transformer
from train_eval import evaluate

app = FastAPI()


@app.get("/inference/{neo4j_database}")
async def do_inference(neo4j_database: str):
    extracted_data = extract_data(neo4j_database)
    transformed_data = data_transformer(extracted_data)
    # need to add a step to format data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = torch.load("open-pulse-graph-classifier/models/supervised_hetero.pt")
    model = loaded_model.to(device)
    output = evaluate(transformed_data, device, model)
    # should the predictions be returned in the response or should it be saved to NEO4J or something?
    return {"output": output}
