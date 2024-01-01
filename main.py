# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/example", response_model=dict, response_model_exclude_none=True, json_schema_extra={"example": {"key": "value"}})
def read_root():
    return {"Hello": "World"}
