import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .manager import ModelManager
from .store import ModelStore


app = FastAPI(title="Multi-Model Serving Demo")


model_store = ModelStore("config/models.json")
model_manager = ModelManager(model_store)


class PredictionRequest(BaseModel):
    model_id: str
    input_data: str


@app.post("/predict")
async def predict(request: PredictionRequest):
    worker = model_manager.get_model_worker(request.model_id)

    if worker is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model {request.model_id} not found",
        )

    try:
        return await worker.predict(request.input_data)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.get("/models")
async def list_models():
    return {
        "available_models": model_store.list_models(),
        "loaded_models": model_manager.list_loaded_models(),
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)