"""Smoke tests for the PantryPal starter project."""

from __future__ import annotations

import unittest
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pantrypal.cli.main import build_agent
from pantrypal.llm.client import LLMClient
from pantrypal.memory.preferences_memory import PreferencesMemory
from pantrypal.memory.pantry_memory import PantryMemory
from pantrypal.memory.system import MemorySystem
from pantrypal.retrieval.recipe_retriever import RecipeRetriever
from pantrypal.retrieval.substitution_retriever import SubstitutionRetriever
from pantrypal.tools.pantry_tool import PantryTool
from pantrypal.tools.preferences_tool import PreferencesTool
from pantrypal.tools.substitution_tool import SubstitutionTool
from pantrypal.tools.web_search_tool import MockSearchProvider, WebSearchTool
from pantrypal.utils.config import Settings, env_path, load_env_file, parse_env_line

DATA_DIR = PROJECT_ROOT / "data"


class PantryPalSmokeTests(unittest.TestCase):
    """Smoke tests for the starter package."""

    def test_dotenv_loader_reads_values_without_overwriting_environment(self) -> None:
        """Local .env loading should support common syntax and preserve real env vars."""

        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "OPENAI_MODEL=from-env-file",
                        "OPENAI_BASE_URL='https://example.test/v1'",
                        "export PANTRYPAL_LOG_LEVEL=INFO # inline comment",
                    ]
                ),
                encoding="utf-8",
            )
            old_values = {
                key: os.environ.get(key)
                for key in ["OPENAI_MODEL", "OPENAI_BASE_URL", "PANTRYPAL_LOG_LEVEL"]
            }
            os.environ["OPENAI_MODEL"] = "already-set"
            os.environ.pop("OPENAI_BASE_URL", None)
            os.environ.pop("PANTRYPAL_LOG_LEVEL", None)
            try:
                load_env_file(env_path)

                self.assertEqual(os.environ["OPENAI_MODEL"], "already-set")
                self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://example.test/v1")
                self.assertEqual(os.environ["PANTRYPAL_LOG_LEVEL"], "INFO")
            finally:
                for key, value in old_values.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_dotenv_parser_ignores_comments_and_blank_lines(self) -> None:
        """The .env parser should ignore non-assignment lines."""

        self.assertIsNone(parse_env_line(""))
        self.assertIsNone(parse_env_line("# comment"))
        self.assertEqual(parse_env_line('OPENAI_MODEL="model-name"'), ("OPENAI_MODEL", "model-name"))

    def test_env_path_resolves_relative_paths_from_project_root(self) -> None:
        """Relative paths from .env should not depend on the current shell directory."""

        old_value = os.environ.get("PANTRYPAL_DATA_DIR")
        os.environ["PANTRYPAL_DATA_DIR"] = "data"
        try:
            self.assertEqual(env_path("PANTRYPAL_DATA_DIR", Path("/tmp/default"), PROJECT_ROOT), DATA_DIR)
        finally:
            if old_value is None:
                os.environ.pop("PANTRYPAL_DATA_DIR", None)
            else:
                os.environ["PANTRYPAL_DATA_DIR"] = old_value

    def test_pantry_memory_roundtrip(self) -> None:
        """Pantry memory should persist added ingredients."""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "pantry.json"
            memory = PantryMemory.load(path)
            memory.add(["Rice", " eggs "])

            loaded = PantryMemory.load(path)

        self.assertEqual(loaded.list_items(), ["eggs", "rice"])

    def test_pantry_tool_stores_natural_keep_statement(self) -> None:
        """Pantry tool should store items from a natural pantry statement."""

        with TemporaryDirectory() as directory:
            memory = PantryMemory.load(Path(directory) / "pantry_memory.json")
            tool = PantryTool(memory)

            result = tool.run({"query": "I usually keep chickpeas and feta."})

        self.assertIn("chickpeas", result["items"])
        self.assertIn("feta", result["items"])

    def test_preferences_memory_roundtrip(self) -> None:
        """Preferences memory should persist typed preference categories."""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "preferences_memory.json"
            memory = PreferencesMemory.load(path)
            memory.add_dietary_preferences(["Vegetarian"])
            memory.add_disliked_ingredients(["Mushrooms"])
            memory.add_favorite_cuisines(["Greek"])

            loaded = PreferencesMemory.load(path)

        self.assertEqual(loaded.list_dietary_preferences(), ["vegetarian"])
        self.assertEqual(loaded.list_disliked_ingredients(), ["mushrooms"])
        self.assertEqual(loaded.list_favorite_cuisines(), ["greek"])

    def test_preferences_tool_stores_dislike_statement(self) -> None:
        """Preferences tool should store disliked ingredients from natural text."""

        with TemporaryDirectory() as directory:
            memory = PreferencesMemory.load(Path(directory) / "preferences_memory.json")
            tool = PreferencesTool(memory)

            result = tool.run({"query": "I hate mushrooms."})

        self.assertIn("mushrooms", result["preferences"]["disliked_ingredients"])

    def test_memory_system_loads_expected_files(self) -> None:
        """Memory system should use the requested local JSON filenames."""

        with TemporaryDirectory() as directory:
            memory_dir = Path(directory)
            memory = MemorySystem.load(memory_dir)
            memory.pantry.add(["chickpeas"])
            memory.preferences.add_favorite_cuisines(["italian"])

            self.assertTrue((memory_dir / "pantry_memory.json").exists())
            self.assertTrue((memory_dir / "preferences_memory.json").exists())

    def test_recipe_retriever_finds_local_recipe(self) -> None:
        """Recipe retriever should find starter corpus entries."""

        retriever = RecipeRetriever(DATA_DIR / "recipes.json")
        response = retriever.search("rice eggs", pantry_ingredients=["rice", "eggs"])

        self.assertTrue(response.results)
        self.assertEqual(response.results[0].recipe.title, "Egg Fried Rice")
        self.assertIn("rice", response.results[0].matched_ingredients)
        self.assertGreater(response.confidence, 0)

    def test_recipe_retriever_filters_dietary_preferences(self) -> None:
        """Recipe retriever should support dietary preference filtering."""

        retriever = RecipeRetriever(DATA_DIR / "recipes.json")
        response = retriever.search("curry", dietary_preferences=["vegan"])

        self.assertTrue(response.results)
        self.assertTrue(
            all("vegan" in result.recipe.dietary_tags for result in response.results)
        )

    def test_recipe_retriever_supports_partial_ingredient_matches(self) -> None:
        """Recipe retriever should match partial ingredient names."""

        retriever = RecipeRetriever(DATA_DIR / "recipes.json")
        response = retriever.search("tomato")

        self.assertTrue(response.results)
        self.assertIn("canned tomatoes", response.results[0].matched_ingredients)

    def test_recipe_retriever_boosts_pantry_ingredients(self) -> None:
        """Recipe retriever should prefer recipes covered by pantry ingredients."""

        retriever = RecipeRetriever(DATA_DIR / "recipes.json")
        response = retriever.search(
            "what can I cook",
            pantry_ingredients=["rice", "eggs", "soy sauce"],
        )

        self.assertTrue(response.results)
        self.assertEqual(response.results[0].recipe.title, "Egg Fried Rice")
        self.assertIn("frozen peas", response.results[0].missing_ingredients)

    def test_substitution_retriever_ranks_available_substitute(self) -> None:
        """Substitution retriever should boost substitutes already in pantry."""

        retriever = SubstitutionRetriever(DATA_DIR / "substitutions.json")
        result = retriever.search(
            "heavy cream",
            available=["milk", "butter", "greek yogurt"],
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.best_substitute.name, "milk and butter")
        self.assertTrue(result.best_substitute.in_pantry)
        self.assertGreater(result.confidence, 0)

    def test_substitution_tool_returns_structured_evidence(self) -> None:
        """Substitution tool should return best, alternatives, explanation, and evidence."""

        retriever = SubstitutionRetriever(DATA_DIR / "substitutions.json")
        tool = SubstitutionTool(retriever)

        result = tool.run(
            {
                "missing": "heavy cream",
                "available": ["milk", "butter", "greek yogurt"],
            }
        )

        self.assertIn("best milk and butter", result["content"])
        self.assertGreater(result["confidence"], 0)
        first = result["results"][0]
        self.assertEqual(first["best_substitute"]["name"], "milk and butter")
        self.assertIn("evidence", first)

    def test_recipe_retriever_signals_web_fallback_for_sparse_results(self) -> None:
        """Recipe retriever should signal web fallback for weak local matches."""

        retriever = RecipeRetriever(DATA_DIR / "recipes.json", confidence_threshold=0.9)
        response = retriever.search("rare holiday dish")

        self.assertTrue(response.use_web_fallback)

    def test_agent_runs_without_llm_key(self) -> None:
        """Agent should return tool context when no LLM key is configured."""

        with TemporaryDirectory() as directory:
            settings = Settings(
                openai_api_key=None,
                openai_base_url=None,
                openai_model="test-model",
                data_dir=DATA_DIR,
                memory_dir=Path(directory),
                web_search_provider="mock",
            )
            agent = build_agent(settings)

            response = agent.handle("what can I cook with rice and eggs?")

        self.assertIn("Egg Fried Rice", response)
        self.assertIn("LLM is not configured", response)
        self.assertIn("Tool trace:", response)

    def test_agent_triggers_web_fallback_for_low_local_coverage(self) -> None:
        """Agent should call web fallback when local recipes are sparse."""

        with TemporaryDirectory() as directory:
            settings = Settings(
                openai_api_key=None,
                openai_base_url=None,
                openai_model="test-model",
                data_dir=DATA_DIR,
                memory_dir=Path(directory),
                web_search_provider="mock",
            )
            agent = build_agent(settings)

            response = agent.handle("what can I cook with plantains and cassava?")

        self.assertIn("web-search fallback", response)
        self.assertIn("web_search.run", response)

    def test_web_search_tool_returns_structured_mock_results(self) -> None:
        """Web search tool should return title, snippet, URL, and confidence."""

        tool = WebSearchTool(MockSearchProvider())
        result = tool.run(
            {
                "query": "heavy cream substitute",
                "search_type": "substitution",
                "limit": 2,
            }
        )

        self.assertEqual(result["provider"], "mock")
        self.assertEqual(result["search_type"], "substitution")
        self.assertGreater(result["confidence"], 0)
        self.assertIn("title", result["results"][0])
        self.assertIn("snippet", result["results"][0])
        self.assertIn("url", result["results"][0])

    def test_web_search_tool_supports_technique_lookup(self) -> None:
        """Web search tool should support cooking technique lookup."""

        tool = WebSearchTool(MockSearchProvider())
        result = tool.run({"query": "how to temper chocolate", "limit": 1})

        self.assertEqual(result["search_type"], "technique")
        self.assertEqual(result["results"][0]["type"], "technique")

    def test_agent_uses_web_for_unfamiliar_cuisine(self) -> None:
        """Agent should use web search when the requested cuisine is unfamiliar."""

        with TemporaryDirectory() as directory:
            settings = Settings(
                openai_api_key=None,
                openai_base_url=None,
                openai_model="test-model",
                data_dir=DATA_DIR,
                memory_dir=Path(directory),
                web_search_provider="mock",
            )
            agent = build_agent(settings)

            response = agent.handle("Suggest a Peruvian dinner recipe")

        self.assertIn("web_search.run", response)
        self.assertIn("Web Recipe:", response)

    def test_agent_routes_technique_lookup_to_web(self) -> None:
        """Agent should use web search for cooking technique lookup."""

        with TemporaryDirectory() as directory:
            settings = Settings(
                openai_api_key=None,
                openai_base_url=None,
                openai_model="test-model",
                data_dir=DATA_DIR,
                memory_dir=Path(directory),
                web_search_provider="mock",
            )
            agent = build_agent(settings)

            response = agent.handle("how to temper chocolate")

        self.assertIn("Cooking Technique:", response)
        self.assertIn("web_search.run", response)
        self.assertNotIn("substitution.run", response)

    def test_llm_client_returns_placeholder_without_key(self) -> None:
        """LLM client should stay usable in offline demo mode."""

        settings = Settings(
            openai_api_key=None,
            openai_base_url=None,
            openai_model="test-model",
            data_dir=DATA_DIR,
            memory_dir=DATA_DIR,
        )
        client = LLMClient(settings)

        response = client.chat([{"role": "user", "content": "hello"}])

        self.assertIn("LLM is not configured", response)

    def test_llm_client_uses_openai_style_messages(self) -> None:
        """LLM client should pass dict messages through to the SDK."""

        fake_client = FakeOpenAIClient(["hello from model"])
        settings = Settings(
            openai_api_key="key",
            openai_base_url="https://example.test/v1",
            openai_model="test-model",
            data_dir=DATA_DIR,
            memory_dir=DATA_DIR,
        )
        client = LLMClient(settings, sdk_client=fake_client)

        response = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(response, "hello from model")
        self.assertEqual(fake_client.calls[0]["model"], "test-model")
        self.assertEqual(fake_client.calls[0]["messages"][0]["role"], "user")
        self.assertEqual(fake_client.calls[0]["timeout"], 30.0)

    def test_llm_client_retries_failures(self) -> None:
        """LLM client should retry transient SDK failures."""

        fake_client = FakeOpenAIClient([RuntimeError("temporary"), "recovered"])
        settings = Settings(
            openai_api_key="key",
            openai_base_url=None,
            openai_model="test-model",
            data_dir=DATA_DIR,
            memory_dir=DATA_DIR,
            openai_max_retries=1,
        )
        client = LLMClient(settings, sdk_client=fake_client)

        with self.assertLogs("pantrypal.llm.client", level="WARNING"):
            response = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(response, "recovered")
        self.assertEqual(len(fake_client.calls), 2)

    def test_llm_client_reports_failure_after_retries(self) -> None:
        """LLM client should provide a useful final error message."""

        fake_client = FakeOpenAIClient([RuntimeError("still down")])
        settings = Settings(
            openai_api_key="key",
            openai_base_url=None,
            openai_model="test-model",
            data_dir=DATA_DIR,
            memory_dir=DATA_DIR,
            openai_max_retries=0,
        )
        client = LLMClient(settings, sdk_client=fake_client)

        with self.assertLogs("pantrypal.llm.client", level="WARNING"):
            with self.assertRaisesRegex(RuntimeError, "LLM request failed after retries"):
                client.chat([{"role": "user", "content": "hello"}])


class FakeMessage:
    """Minimal fake OpenAI SDK message."""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    """Minimal fake OpenAI SDK choice."""

    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    """Minimal fake OpenAI SDK response."""

    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    """Fake completions endpoint that can fail then recover."""

    def __init__(self, parent: "FakeOpenAIClient") -> None:
        self.parent = parent

    def create(self, **kwargs: object) -> FakeResponse:
        self.parent.calls.append(kwargs)
        next_response = self.parent.responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return FakeResponse(str(next_response))


class FakeChat:
    """Fake chat endpoint."""

    def __init__(self, parent: "FakeOpenAIClient") -> None:
        self.completions = FakeCompletions(parent)


class FakeOpenAIClient:
    """Fake OpenAI client used by LLM wrapper tests."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []
        self.chat = FakeChat(self)


if __name__ == "__main__":
    unittest.main()
