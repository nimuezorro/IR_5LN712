"""Prompt templates for PantryPal."""

SYSTEM_PROMPT = """You are PantryPal, a retrieval-focused cooking assistant.
Use pantry memory, local recipe results, substitution results, and web fallback
evidence before answering. Give a practical cooking answer grounded only in the
provided context. Explain which sources were used, distinguish local recipes
from web-search results, and mention retrieved substitutions. Avoid inventing
unavailable facts."""


def build_tool_context(tool_outputs: list[str]) -> str:
    """Join tool outputs into a compact context block for the LLM."""

    if not tool_outputs:
        return "No tool context available."
    return "\n\n".join(tool_outputs)


def build_final_prompt(question: str, trace: str, context: str) -> str:
    """Build the final grounded synthesis prompt."""

    return (
        f"User question:\n{question}\n\n"
        f"ReAct trace:\n{trace}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Write the final answer. Include a short 'Sources used' note."
    )
