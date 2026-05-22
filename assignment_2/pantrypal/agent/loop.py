"""ReAct-style agent loop for PantryPal."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pantrypal.agent.planner import Planner
from pantrypal.agent.prompts import SYSTEM_PROMPT, build_final_prompt, build_tool_context
from pantrypal.llm.client import LLMClient
from pantrypal.tools.base import Tool, ToolInput, ToolOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceStep:
    """One Thought -> Action -> Observation step in the agent loop."""

    thought: str
    action: str
    observation: str

    def format(self) -> str:
        """Format the step for prompts or debugging."""

        return (
            f"Thought: {self.thought}\n"
            f"Action: {self.action}\n"
            f"Observation: {self.observation}"
        )


class AgentLoop:
    """Coordinates planning, retrieval, tool tracing, and answer synthesis."""

    low_confidence_threshold = 0.35
    min_relevant_recipes = 3

    def __init__(self, tools: list[Tool], llm_client: LLMClient) -> None:
        """Create an agent loop with available tools and an LLM client."""

        self.tools = {tool.name: tool for tool in tools}
        self.llm_client = llm_client
        self.planner = Planner()

    def handle(self, user_message: str) -> str:
        """Handle a single user message with a lightweight ReAct loop."""

        plan = self.planner.plan(user_message)
        logger.info("Agent plan: %s", plan.reason)

        if plan.tool_names in (["pantry"], ["preferences"]):
            result = self._run_tool(plan.tool_names[0], {"query": user_message})
            return str(result.get("content", "Memory tool unavailable."))

        trace: list[TraceStep] = []
        outputs: dict[str, ToolOutput] = {}
        for tool_name in plan.tool_names:
            output = self._run_tool(tool_name, self._build_tool_input(tool_name, user_message, outputs))
            outputs[tool_name] = output
            trace.append(
                TraceStep(
                    thought=self._thought_for_tool(tool_name),
                    action=f"{tool_name}.run",
                    observation=str(output.get("content", "")),
                )
            )

        if self._should_trigger_web(outputs.get("recipe_search")) and "web_search" not in outputs:
            output = self._run_tool(
                "web_search",
                self._build_tool_input("web_search", user_message, outputs),
            )
            outputs["web_search"] = output
            trace.append(
                TraceStep(
                    thought="Local recipe retrieval has low coverage, so web fallback is needed.",
                    action="web_search.run",
                    observation=str(output.get("content", "")),
                )
            )

        context = self._format_context(outputs)
        trace_text = "\n\n".join(step.format() for step in trace)
        final_prompt = build_final_prompt(user_message, trace_text, context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": final_prompt},
        ]
        response = self.llm_client.chat(messages)
        if self.llm_client.is_configured():
            return response
        return self._fallback_answer(user_message, outputs, trace, response)

    def _run_tool(self, tool_name: str, input_data: ToolInput) -> ToolOutput:
        """Run a tool with logging, tracing, and graceful error handling."""

        tool = self.tools.get(tool_name)
        if tool is None:
            logger.warning("Requested unavailable tool: %s", tool_name)
            return {
                "tool_name": tool_name,
                "content": f"Tool '{tool_name}' is unavailable.",
                "error": "tool_not_found",
            }
        logger.info("Action: %s.run input=%s", tool_name, input_data)
        try:
            output = tool.run(input_data)
        except Exception as error:  # pragma: no cover - defensive boundary
            logger.exception("Tool failed: %s", tool_name)
            return {
                "tool_name": tool_name,
                "content": f"{tool_name} failed gracefully: {error}",
                "error": str(error),
            }
        logger.info("Observation from %s: %s", tool_name, output.get("content", ""))
        return output

    def _build_tool_input(
        self,
        tool_name: str,
        user_message: str,
        outputs: dict[str, ToolOutput],
    ) -> ToolInput:
        """Build structured input for a tool from prior observations."""

        input_data: ToolInput = {"query": user_message, "limit": 5}
        if tool_name == "recipe_search":
            input_data["pantry_items"] = outputs.get("pantry", {}).get("items", [])
            preferences = outputs.get("preferences", {}).get("preferences", {})
            if isinstance(preferences, dict):
                input_data["dietary_preferences"] = preferences.get("dietary_preferences", [])
        if tool_name == "substitution":
            input_data["pantry_items"] = outputs.get("pantry", {}).get("items", [])
        if tool_name == "web_search":
            input_data["search_type"] = self._infer_web_search_type(user_message)
        return input_data

    def _infer_web_search_type(self, user_message: str) -> str:
        """Infer which web retrieval mode is useful for a query."""

        lower = user_message.lower()
        if any(word in lower for word in ["substitute", "replacement", "instead of"]):
            return "substitution"
        if any(word in lower for word in ["technique", "how do i", "how to", "method"]):
            return "technique"
        return "recipe"

    def _should_trigger_web(self, recipe_output: ToolOutput | None) -> bool:
        """Return whether web fallback should be called for recipe coverage."""

        if recipe_output is None:
            return True
        if "use_web_fallback" in recipe_output:
            return bool(recipe_output["use_web_fallback"])
        results = recipe_output.get("results", [])
        confidence = float(recipe_output.get("confidence", 0.0))
        return (
            not isinstance(results, list)
            or len(results) < self.min_relevant_recipes
            or confidence < self.low_confidence_threshold
        )

    def _format_context(self, outputs: dict[str, ToolOutput]) -> str:
        """Format all tool outputs as retrieval context."""

        return build_tool_context(
            [
                f"[{name}]\n{output.get('content', '')}"
                for name, output in outputs.items()
                if output.get("content")
            ]
        )

    def _fallback_answer(
        self,
        user_message: str,
        outputs: dict[str, ToolOutput],
        trace: list[TraceStep],
        llm_placeholder: str,
    ) -> str:
        """Build a grounded answer when no LLM credentials are configured."""

        local_recipes = self._list_recipe_titles(outputs.get("recipe_search"))
        web_recipes = self._list_recipe_titles(outputs.get("web_search"))
        substitutions = self._list_substitutions(outputs.get("substitution"))
        pantry = outputs.get("pantry", {}).get("items", [])
        preferences = outputs.get("preferences", {}).get("preferences", {})

        lines = [
            "Final Answer:",
            f"I retrieved context for: {user_message}",
        ]
        if pantry:
            lines.append(f"Pantry memory: {', '.join(str(item) for item in pantry)}.")
        if isinstance(preferences, dict) and any(preferences.values()):
            lines.append(f"Preferences memory: {self._format_preferences_inline(preferences)}.")
        if local_recipes:
            lines.append(f"Local recipe matches: {', '.join(local_recipes)}.")
        else:
            lines.append("Local recipe matches: none.")
        if web_recipes:
            lines.append(f"Web recipe matches: {', '.join(web_recipes)}.")
        elif "web_search" in outputs:
            lines.append("Web search fallback was triggered, but no web provider is configured.")
        if substitutions:
            lines.append(f"Retrieved substitutions: {'; '.join(substitutions)}.")
        elif "substitution" in outputs:
            lines.append("Retrieved substitutions: none.")
        lines.extend(
            [
                "",
                "Sources used:",
                self._source_summary(outputs),
                "",
                "Tool trace:",
                "\n\n".join(step.format() for step in trace),
                "",
                llm_placeholder,
            ]
        )
        return "\n".join(lines)

    def _list_recipe_titles(self, output: ToolOutput | None) -> list[str]:
        """Extract recipe titles from a recipe-like tool output."""

        if output is None:
            return []
        results = output.get("results", [])
        if not isinstance(results, list):
            return []
        return [
            str(item.get("title"))
            for item in results
            if isinstance(item, dict) and item.get("title")
        ]

    def _list_substitutions(self, output: ToolOutput | None) -> list[str]:
        """Extract human-readable substitution summaries."""

        if output is None:
            return []
        results = output.get("results", [])
        if not isinstance(results, list):
            return []
        summaries = []
        for item in results:
            if not isinstance(item, dict):
                continue
            ingredient = item.get("ingredient")
            best = item.get("best_substitute")
            alternatives = item.get("alternative_substitutes", [])
            if ingredient and isinstance(best, dict):
                names = [str(best.get("name"))]
                if isinstance(alternatives, list):
                    names.extend(
                        str(candidate.get("name"))
                        for candidate in alternatives
                        if isinstance(candidate, dict) and candidate.get("name")
                    )
                summaries.append(f"{ingredient} -> {', '.join(names)}")
        return summaries

    def _source_summary(self, outputs: dict[str, ToolOutput]) -> str:
        """Describe which retrieval sources informed the answer."""

        labels = {
            "pantry": "pantry memory",
            "preferences": "preferences memory",
            "recipe_search": "local recipe retrieval",
            "substitution": "local substitution retrieval",
            "web_search": "web-search fallback",
        }
        used = [labels[name] for name in outputs if name in labels]
        return ", ".join(used) if used else "No retrieval sources were available."

    def _format_preferences_inline(self, preferences: dict[str, object]) -> str:
        """Format stored preferences for the no-LLM fallback answer."""

        chunks = []
        labels = {
            "dietary_preferences": "dietary",
            "disliked_ingredients": "dislikes",
            "favorite_cuisines": "favorite cuisines",
        }
        for key, label in labels.items():
            values = preferences.get(key, [])
            if isinstance(values, list) and values:
                chunks.append(f"{label}: {', '.join(str(value) for value in values)}")
        return "; ".join(chunks)

    def _thought_for_tool(self, tool_name: str) -> str:
        """Return the trace thought associated with a tool."""

        thoughts: dict[str, str] = {
            "pantry": "Check pantry memory before recommending food.",
            "preferences": "Check dietary preferences and tastes before recommending food.",
            "recipe_search": "Retrieve local recipes relevant to the question.",
            "substitution": "Look for ingredient substitutions that may help.",
            "web_search": "Use web fallback when local retrieval is insufficient.",
        }
        return thoughts.get(tool_name, "Run the selected tool.")
