"""Interactive CLI for PantryPal."""

from __future__ import annotations

from pantrypal.agent.loop import AgentLoop
from pantrypal.llm.client import LLMClient
from pantrypal.memory.system import MemorySystem
from pantrypal.retrieval.recipe_retriever import RecipeRetriever
from pantrypal.retrieval.substitution_retriever import SubstitutionRetriever
from pantrypal.tools.pantry_tool import PantryTool
from pantrypal.tools.preferences_tool import PreferencesTool
from pantrypal.tools.recipe_search_tool import RecipeSearchTool
from pantrypal.tools.substitution_tool import SubstitutionTool
from pantrypal.tools.web_search_tool import WebSearchTool, build_search_provider
from pantrypal.utils.config import Settings
from pantrypal.utils.logging import configure_logging


def build_agent(settings: Settings | None = None) -> AgentLoop:
    """Build a fully wired PantryPal agent."""

    resolved_settings = settings or Settings.from_env()
    memory = MemorySystem.load(resolved_settings.memory_dir)
    recipe_retriever = RecipeRetriever(
        resolved_settings.data_dir / "recipes.json",
        confidence_threshold=resolved_settings.retrieval_confidence_threshold,
    )
    substitution_retriever = SubstitutionRetriever(
        resolved_settings.data_dir / "substitutions.json"
    )
    search_provider = build_search_provider(
        provider_name=resolved_settings.web_search_provider,
        api_key=resolved_settings.web_search_api_key,
        endpoint=resolved_settings.web_search_endpoint,
    )
    tools = [
        PantryTool(memory.pantry),
        PreferencesTool(memory.preferences),
        RecipeSearchTool(recipe_retriever),
        SubstitutionTool(substitution_retriever),
        WebSearchTool(search_provider),
    ]
    return AgentLoop(tools=tools, llm_client=LLMClient(resolved_settings))


def main() -> None:
    """Run the interactive PantryPal chat loop."""

    configure_logging()
    agent = build_agent()
    print("PantryPal CLI. Type 'exit' or 'quit' to leave.")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        print(f"pantrypal> {agent.handle(user_input)}")


if __name__ == "__main__":
    main()
