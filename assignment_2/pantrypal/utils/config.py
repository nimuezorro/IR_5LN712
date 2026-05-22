"""Environment-based configuration for PantryPal."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str
    data_dir: Path
    memory_dir: Path
    openai_timeout_seconds: float = 30.0
    openai_max_retries: int = 2
    retrieval_confidence_threshold: float = 0.35
    web_search_provider: str = "mock"
    web_search_api_key: str | None = None
    web_search_endpoint: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""

        package_root = Path(__file__).resolve().parents[1]
        project_root = package_root.parent
        load_env_file(project_root / ".env")
        default_data_dir = project_root / "data"
        default_memory_dir = project_root / "data"
        
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_timeout_seconds=parse_env(
                "OPENAI_TIMEOUT_SECONDS",
                30.0,
                float,
            ),
            openai_max_retries=parse_env("OPENAI_MAX_RETRIES", 2, int),
            data_dir=env_path("PANTRYPAL_DATA_DIR", default_data_dir, project_root),
            memory_dir=env_path("PANTRYPAL_MEMORY_DIR", default_memory_dir, project_root),
            retrieval_confidence_threshold=parse_env(
                "PANTRYPAL_RETRIEVAL_CONFIDENCE_THRESHOLD",
                0.35,
                float,
            ),
            web_search_provider=os.getenv("PANTRYPAL_WEB_SEARCH_PROVIDER", "mock"),
            web_search_api_key=os.getenv("PANTRYPAL_WEB_SEARCH_API_KEY"),
            web_search_endpoint=os.getenv("PANTRYPAL_WEB_SEARCH_ENDPOINT"),
        )


def parse_env(name: str, default: T, parser: Callable[[str], T]) -> T:
    """Parse an environment variable with a useful configuration error."""

    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return parser(raw_value)
    except ValueError as error:
        raise ValueError(f"Invalid value for {name}: {raw_value!r}") from error


def env_path(name: str, default: Path, base_dir: Path) -> Path:
    """Read a path from the environment, resolving relative paths predictably."""

    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file without extra dependencies.

    Existing environment variables are not overwritten. Supported syntax is
    intentionally small: blank lines, comments, optional ``export``, and quoted
    or unquoted values.
    """

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse one .env line into a key-value pair."""

    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ").strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", maxsplit=1)
    key = key.strip()
    value = strip_inline_comment(value.strip())
    if not key:
        return None
    return key, unquote(value)


def strip_inline_comment(value: str) -> str:
    """Remove unquoted inline comments from a .env value."""

    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char
        if char == "#" and quote is None:
            return value[:index].strip()
    return value


def unquote(value: str) -> str:
    """Remove matching single or double quotes from a .env value."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
