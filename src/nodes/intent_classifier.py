"""Intent classification node."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from utils.nodes import LLM, load_prompt, render_prompt
from utils.llm import extract_text_from_response
from state import SQLAgentState


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "intent_classifier.j2"


class IntentClassifierNode:
    """Classify user intention as query or modification."""

    def __init__(
        self,
        model: LLM,
        prompt_template: str | None = None,
    ) -> None:
        """Initialize the intent classifier node."""
        self.model = model
        self.prompt_template = prompt_template or load_prompt(
            PROMPT_PATH,
            (
                "You classify user intention as either 'query' or 'modification'.\n"
                "Return JSON with key `intent` (string) that is either 'query' or 'modification'.\n"
                "A query is a request to retrieve data (SELECT statements).\n"
                "A modification is a request to change data (INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, etc).\n\n"
                "Question:\n{{ question }}\n"
            ),
        )

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """
        Classify the user's intention from the question in the state.

        Args:
            state: A dictionary containing the current state in the graph,
            expected to have a "question" key.

        Returns:
            A dictionary containing the classified intent or an error message.
        """
        question = state["question"]

        prompt = render_prompt(
            self.prompt_template,
            question=question,
        )
        response_text = extract_text_from_response(self.model.invoke(prompt))

        try:
            decision = json.loads(response_text.strip())
            intent = str(decision.get("intent") or "").strip().lower()

            if intent == "query":
                return {
                    "intent": intent,
                    "intent_error": None,
                }
            elif intent == "modification":
                message = "Modifications are not allowed. Please ask questions to retrieve information."
                return {
                    "intent": intent,
                    "intent_error": message,
                    "execution_error": message,
                    "analysis": message,
                    "metadata": {"intent_failed": True},
                }
            else:
                message = "Could not classify the intention. Please ask questions to retrieve information."
                return {
                    "intent": None,
                    "intent_error": message,
                    "execution_error": message,
                    "analysis": message,
                    "metadata": {"intent_failed": True},
                }

        except json.JSONDecodeError:
            message = "Could not classify the intention. Please ask questions to retrieve information."
            return {
                "intent": None,
                "intent_error": message,
                "execution_error": message,
                "analysis": message,
                "metadata": {"intent_failed": True},
            }
