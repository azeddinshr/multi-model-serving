from collections import OrderedDict
from typing import Optional

from .engine import ModelEngine
from .store import ModelStore
from .worker import ModelWorker


class ModelManager:
    """Maintains an LRU cache of Triton model workers."""

    def __init__(
        self,
        model_store: ModelStore,
        max_models: int = 2,
    ):
        self.model_store = model_store
        self.max_models = max_models

        # Ordered from least recently used to most recently used.
        self.model_cache: OrderedDict[str, ModelWorker] = OrderedDict()

        self.model_engine = ModelEngine()

    def get_model_worker(
        self,
        model_id: str,
    ) -> Optional[ModelWorker]:
        """Return a cached worker or load the model through Triton."""

        if model_id in self.model_cache:
            # Cache hit: mark as most recently used.
            self.model_cache.move_to_end(model_id)
            return self.model_cache[model_id]

        model_metadata = self.model_store.get_model(model_id)

        if model_metadata is None:
            return None

        # Cache full: remove the least recently used model.
        if len(self.model_cache) >= self.max_models:
            lru_model_id, _ = self.model_cache.popitem(last=False)
            self.model_engine.delete_worker(lru_model_id)

        worker = self.model_engine.create_worker(model_metadata)
        self.model_cache[model_id] = worker

        return worker

    def list_loaded_models(self) -> dict[str, str]:
        """Return the models currently held in the LRU cache."""

        return {
            model_id: worker.model_metadata.name
            for model_id, worker in self.model_cache.items()
        }