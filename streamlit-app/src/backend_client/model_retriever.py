import requests

OPENAI_MODELS = (
    "gpt-6-astra",
    "gpt-6.1-sol",
    "gpt-6-luna",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
)


def get_ollama_models() -> list[str]:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=(3, 10))
        resp.raise_for_status()
        models = resp.json()["models"]
        if not isinstance(models, list) or any(
            not isinstance(model, dict)
            or not isinstance(model.get("name"), str)
            or not model["name"]
            for model in models
        ):
            raise ValueError("Invalid model list")
        return [model["name"] for model in models]
    except requests.RequestException as error:
        raise RuntimeError(
            "Cannot retrieve Ollama models. Check that Ollama is running."
        ) from error
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError("Ollama returned an invalid model list.") from error


def get_openai_models() -> list[str]:
    return list(OPENAI_MODELS)
