from typing import Optional

from app.store import ModelMetadata
from app.worker import ModelWorker, TritonWorker


class ModelEngine:
    """Creates and manages Triton-backed workers."""

    def __init__(self):
        self.workers: dict[str, ModelWorker] = {}

    def create_worker(self, model_metadata: ModelMetadata) -> ModelWorker:
        """Create a Triton worker for a model."""

        if model_metadata.id not in self.workers:
            if model_metadata.framework != "triton":
                raise ValueError(
                    f"Unsupported framework: {model_metadata.framework}"
                )

            self.workers[model_metadata.id] = TritonWorker(model_metadata)

        return self.workers[model_metadata.id]

    def delete_worker(self, model_id: str) -> None:
        """Unload and remove a worker."""

        worker = self.workers.pop(model_id, None)

        if worker is not None:
            worker.unload()