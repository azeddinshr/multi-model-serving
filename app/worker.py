from abc import ABC, abstractmethod
from typing import Any, Dict

import asyncio
import os
import requests
import numpy as np
import tritonclient.http as httpclient
from pytriton.client import AsyncioModelClient
from transformers import AutoTokenizer


class ModelWorker(ABC):
    """Base interface for models served through Triton."""

    def __init__(self, model_metadata):
        self.model_metadata = model_metadata
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    async def predict(self, input_data: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def unload(self):
        pass


class TritonWorker(ModelWorker):
    """Gateway worker for a model served by NVIDIA Triton."""

    def __init__(self, model_metadata):
        self.triton_url = os.getenv("TRITON_URL", "localhost:8000")
        self.tokenizer = None
        self.client = None

        # Used only for model lifecycle operations.
        self.lifecycle_client = httpclient.InferenceServerClient(
            url=self.triton_url
        )

        super().__init__(model_metadata)

    def _load_model(self):
        model_name = self.model_metadata.name

        # Ask Triton to load the model.
        load_url = (
            f"http://{self.triton_url}"
            f"/v2/repository/models/{model_name}/load"
        )

        response = requests.post(load_url)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to load Triton model '{model_name}': "
                f"{response.text}"
            )

        # Make sure Triton actually reports the model as ready.
        if not self.lifecycle_client.is_model_ready(model_name):
            raise RuntimeError(
                f"Triton model '{model_name}' is not ready"
            )

        # Tokenizer belongs to the gateway because Triton receives
        # already-tokenized tensors.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_metadata.tokenizer
        )

        # Persistent async client.
        # Reused for every inference request.
        self.client = AsyncioModelClient(
            self.triton_url,
            model_name,
        )

    async def predict(self, input_data: Any) -> Dict[str, Any]:
        if self.client is None or self.tokenizer is None:
            raise RuntimeError("Triton worker is not initialized")

        encoded = await asyncio.to_thread(
            self.tokenizer,
            input_data,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.model_metadata.max_sequence_length,
        )

        input_ids = encoded["input_ids"].astype(np.int64)[0]
        attention_mask = encoded["attention_mask"].astype(np.int64)[0]

        result = await self.client.infer_sample(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = result["logits"]

        # Convert logits → probabilities.
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp / np.sum(exp, axis=-1, keepdims=True)

        prediction_id = int(np.argmax(probabilities))

        labels = self.model_metadata.labels

        return {
            "label": labels[prediction_id],
            "probabilities": probabilities.tolist(),
            "logits": logits.tolist(),
        }

    def unload(self):
        model_name = self.model_metadata.name

        unload_url = (
            f"http://{self.triton_url}"
            f"/v2/repository/models/{model_name}/unload"
        )

        try:
            requests.post(unload_url, timeout=5)
        except requests.RequestException:
            pass