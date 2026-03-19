"""
src/diagnosis/llm_client.py
────────────────────────────
Model-agnostic LLM client wrapper.
Supports OpenAI GPT-4, Anthropic Claude, and Azure OpenAI.
Provides structured JSON output validation and retry logic.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class LLMClient:
    """
    Unified LLM client that abstracts over OpenAI, Anthropic, and Azure.
    All diagnosis calls expect structured JSON output validated against
    the DiagnosisOutput schema (see diagnoser.py).
    """

    def __init__(self, config: dict):
        self.config = config
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4-turbo")
        self.max_tokens = config.get("max_tokens", 1500)
        self.temperature = config.get("temperature", 0.1)
        self.timeout = config.get("timeout_seconds", 30)
        self._client = self._build_client(config)

    def _build_client(self, config: dict):
        provider = config.get("provider", "openai")
        if provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=config["openai"]["api_key"])
        elif provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=config["anthropic"]["api_key"])
        elif provider == "azure_openai":
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=config["azure_openai"]["api_key"],
                azure_endpoint=config["azure_openai"]["endpoint"],
                api_version=config["azure_openai"]["api_version"],
            )
        raise ValueError(f"Unsupported LLM provider: {provider}")

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a completion request and return the raw text response."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self._do_complete(
                    system_prompt,
                    user_message,
                    max_tokens or self.max_tokens,
                    temperature if temperature is not None else self.temperature,
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "LLM API error, retrying",
                    attempt=attempt + 1,
                    wait=wait,
                    error=str(e),
                )
                time.sleep(wait)

    def _do_complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if self.provider in ("openai", "azure_openai"):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self.timeout,
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.config["anthropic"].get("model", "claude-3-opus-20240229"),
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text.strip()

        raise ValueError(f"Unsupported provider: {self.provider}")

    def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        schema: dict | None = None,
    ) -> dict:
        """
        Complete and parse JSON response.
        Strips markdown code fences if present.
        """
        raw = self.complete(system_prompt, user_message)

        # Strip ```json ... ``` fences
        if "```" in raw:
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
            if match:
                raw = match.group(1)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse LLM JSON response",
                error=str(e),
                raw_preview=raw[:200],
            )
            raise ValueError(f"LLM returned invalid JSON: {e}") from e
