"""
Tanek sloj nad modeli.
Nov model samo dodaš v MODEL_MAP, drugje ni treba nič spreminjat.
"""
import os
from abc import ABC, abstractmethod

from openai import OpenAI
import anthropic

MODEL_MAP: dict[str, tuple[str, str]] = {
    "gpt-4.1":          ("openai",     "gpt-4.1"),
    "gpt-4o-mini":      ("openai",     "gpt-4o-mini"),
    "claude-3.5-sonnet": ("anthropic", "claude-sonnet-4-6"),
}


class BaseModelClient(ABC):
    @abstractmethod
    def chat(self, system: str, user: str, temperature: float = 0) -> str:
        pass


class OpenAIClient(BaseModelClient):
    def __init__(self, model_id: str):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_id = model_id

    def chat(self, system: str, user: str, temperature: float = 0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content


class AnthropicClient(BaseModelClient):
    def __init__(self, model_id: str):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_id = model_id

    def chat(self, system: str, user: str, temperature: float = 0) -> str:
        resp = self.client.messages.create(
            model=self.model_id,
            max_tokens=4096,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text


def get_model_client(model_name: str) -> BaseModelClient:
    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Neznan model: {model_name!r}. Dostopni: {list(MODEL_MAP)}"
        )
    provider, model_id = MODEL_MAP[model_name]
    if provider == "openai":
        return OpenAIClient(model_id)
    return AnthropicClient(model_id)
