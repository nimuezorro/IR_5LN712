"""OpenAI-compatible LLM client."""

from __future__ import annotations

import logging
import time
from typing import Any

from pantrypal.utils.config import Settings

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI SDK wrapper for OpenAI-compatible chat APIs.

    The wrapper reads model, key, and base URL from ``Settings``. It contains no
    retrieval, memory, tool, or planning logic, so Berget.AI or another
    OpenAI-compatible endpoint can be swapped in through environment variables.
    """

    def __init__(self, settings: Settings, sdk_client: Any | None = None) -> None:
        """Create a client from runtime settings."""

        self.settings = settings
        self._sdk_client = sdk_client

    def is_configured(self) -> bool:
        """Return whether API credentials are available."""

        return bool(self.settings.openai_api_key)

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Generate a chat response from a list of OpenAI-style messages.

        Args:
            messages: Chat messages such as ``{"role": "user", "content": "..."}``.

        Returns:
            The assistant message content. If no API key is configured, returns a
            clear local placeholder used by the offline CLI demo.

        Raises:
            RuntimeError: If the SDK is missing, message input is invalid, or the
                configured provider fails after retries.
        """

        if not self.is_configured():
            return (
                "LLM is not configured. Set OPENAI_API_KEY, OPENAI_BASE_URL, "
                "and OPENAI_MODEL to enable synthesis."
            )
        normalized_messages = self._validate_messages(messages)
        client = self._client()
        attempts = max(self.settings.openai_max_retries, 0) + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                logger.info(
                    "Calling LLM model=%s base_url=%s attempt=%s/%s",
                    self.settings.openai_model,
                    self.settings.openai_base_url or "OpenAI default",
                    attempt,
                    attempts,
                )
                response = client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=normalized_messages,
                    timeout=self.settings.openai_timeout_seconds,
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("LLM returned an empty response.")
                return str(content)
            except Exception as error:  # pragma: no cover - exact SDK errors vary
                last_error = error
                logger.warning(
                    "LLM call failed on attempt %s/%s: %s",
                    attempt,
                    attempts,
                    error,
                )
                if attempt < attempts:
                    time.sleep(min(0.5 * attempt, 2.0))
        raise RuntimeError(self._error_message(last_error)) from last_error

    def _client(self) -> Any:
        """Return a cached OpenAI SDK client."""

        if self._sdk_client is not None:
            return self._sdk_client

        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "OpenAI Python SDK is not installed. Run `pip install -r requirements.txt`."
            ) from error

        self._sdk_client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=self.settings.openai_timeout_seconds,
            max_retries=0,
        )
        return self._sdk_client

    def _validate_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Validate and normalize OpenAI-style messages."""

        if not messages:
            raise RuntimeError("LLM chat requires at least one message.")
        normalized = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not role or content is None:
                raise RuntimeError("Each LLM message must contain 'role' and 'content'.")
            normalized.append({"role": str(role), "content": str(content)})
        return normalized

    def _error_message(self, error: Exception | None) -> str:
        """Build a useful provider-neutral error message."""

        detail = str(error) if error else "unknown error"
        return (
            "LLM request failed after retries. Check OPENAI_API_KEY, "
            "OPENAI_BASE_URL, OPENAI_MODEL, network access, and timeout settings. "
            f"Last error: {detail}"
        )
