import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import requests
import yaml
from src.utils.data_models import ModelProvider


class ModelRetriever:
    def __init__(
        self,
        config_path: Path = Path(__file__).resolve().parents[1]
        / "configs/available_models.yaml",
    ) -> None:
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file)
        self.providers: dict[str, ModelProvider] = {
            name: ModelProvider.model_validate(provider)
            for name, provider in config["model_provider"].items()
        }

    @staticmethod
    def get_ollama_models() -> list[str]:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=(3, 10))
            response.raise_for_status()
            payload: object = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Invalid model list")
            models = payload.get("models")
            if not isinstance(models, list):
                raise ValueError("Invalid model list")
            model_names: list[str] = []
            for model in models:
                if not isinstance(model, dict):
                    raise ValueError("Invalid model list")
                name = model.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("Invalid model list")
                model_names.append(name)
            return model_names
        except requests.RequestException as error:
            raise RuntimeError(
                "Cannot retrieve Ollama models. Check that Ollama is running."
            ) from error
        except (ValueError, KeyError, TypeError) as error:
            raise RuntimeError("Ollama returned an invalid model list.") from error

    def get_model_providers(self) -> list[str]:
        return list(self.providers)

    def get_models(self, model_provider: str) -> list[str]:
        if model_provider not in self.providers:
            raise KeyError(model_provider)

        provider = self.providers[model_provider]
        if provider.retriever_function:
            retriever = getattr(self, provider.retriever_function, None)
            if retriever is None:
                raise RuntimeError(
                    f"Unknown model retriever: {provider.retriever_function}"
                )
            if not callable(retriever):
                raise RuntimeError(
                    f"Unknown model retriever: {provider.retriever_function}"
                )
            return cast(Callable[[], list[str]], retriever)()
        return list(provider.available_models)
