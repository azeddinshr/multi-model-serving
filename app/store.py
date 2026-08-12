import json
from typing import Dict, Optional

from pydantic import BaseModel

class ModelMetadata(BaseModel):
    id: str
    name: str
    type: str
    framework: str
    version: str
    description: str

    tokenizer: str
    max_sequence_length: int = 128
    labels: list[str]

class ModelStore:
    def __init__(self, config_path: str):
        self.models: Dict[str, ModelMetadata] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        with open(config_path, "r") as f:
            config = json.load(f)

        for model in config["models"]:
            metadata = ModelMetadata(**model)
            self.models[metadata.id] = metadata

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        return self.models.get(model_id)

    def list_models(self) -> Dict[str, ModelMetadata]:
        return self.models