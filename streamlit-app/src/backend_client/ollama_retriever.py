import requests


def get_ollama_models() -> list[str]:
    resp = requests.get("http://localhost:11434/api/tags")
    resp.raise_for_status()
    return [model["name"] for model in resp.json()["models"]]
