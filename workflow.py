"""LangGraph workflow for classification, retrieval, and answer generation."""

from __future__ import annotations

import os
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph

from .classifier import Category, classify_query
from .rag import format_context, format_sources, retrieve
from .settings import SETTINGS


class AssistantState(TypedDict, total=False):
    query: str
    programme: str
    history_text: str
    category: Category
    documents: list
    context: str
    answer: str
    sources: list[str]


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful College AI Assistant in a fictional classroom demo. "
            "For college policies, dates, amounts, attendance rules, credits, and course "
            "requirements, use only the retrieved context. If the context does not contain "
            "the answer, say that clearly and direct the student to the appropriate college "
            "office. Personalize the explanation for the selected programme, but never "
            "invent a student's marks, attendance, balance, or personal record. Use short "
            "paragraphs and simple language."
        ),
        (
            "human",
            "Selected programme: {programme}\n"
            "Recent conversation:\n{history_text}\n\n"
            "Student question: {query}\n\n"
            "Retrieved context:\n{context}",
        ),
    ]
)


GENERAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a friendly College AI Assistant. Answer general educational questions "
            "briefly. Do not invent institution-specific dates, fees, policies, contacts, or "
            "student records. If the question needs official college information, say which "
            "office or approved document the student should check."
        ),
        (
            "human",
            "Selected programme: {programme}\n"
            "Recent conversation:\n{history_text}\n\n"
            "Student question: {query}",
        ),
    ]
)


def _message_text(content: object) -> str:
    """Handle either plain text or structured content returned by a chat model."""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


class CollegeAssistant:
    """Own the Mistral model and a compiled LangGraph application."""

    def __init__(self) -> None:
        if not os.getenv("MISTRAL_API_KEY"):
            raise RuntimeError("MISTRAL_API_KEY is not set in the environment or .env file")
        self.llm = ChatMistralAI(
            model=SETTINGS.mistral_model,
            temperature=0.1,
            max_retries=2,
            max_tokens=450,
        )
        self.graph = self._build_graph()

    def _classify_node(self, state: AssistantState) -> dict:
        return {"category": classify_query(state["query"])}

    @staticmethod
    def _route_after_classification(
        state: AssistantState,
    ) -> Literal["retrieve", "general_answer"]:
        return "general_answer" if state["category"] == "general" else "retrieve"

    @staticmethod
    def _retrieve_node(state: AssistantState) -> dict:
        documents = retrieve(state["category"], state["query"], state["programme"])
        return {
            "documents": documents,
            "context": format_context(documents),
            "sources": format_sources(documents),
        }

    def _rag_answer_node(self, state: AssistantState) -> dict:
        messages = RAG_PROMPT.format_messages(
            programme=state["programme"],
            history_text=state["history_text"],
            query=state["query"],
            context=state["context"],
        )
        response = self.llm.invoke(messages)
        return {"answer": _message_text(response.content)}

    def _general_answer_node(self, state: AssistantState) -> dict:
        messages = GENERAL_PROMPT.format_messages(
            programme=state["programme"],
            history_text=state["history_text"],
            query=state["query"],
        )
        response = self.llm.invoke(messages)
        return {"answer": _message_text(response.content), "sources": []}

    def _build_graph(self):
        builder = StateGraph(AssistantState)
        builder.add_node("classify", self._classify_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("rag_answer", self._rag_answer_node)
        builder.add_node("general_answer", self._general_answer_node)

        builder.add_edge(START, "classify")
        builder.add_conditional_edges("classify", self._route_after_classification)
        builder.add_edge("retrieve", "rag_answer")
        builder.add_edge("rag_answer", END)
        builder.add_edge("general_answer", END)
        return builder.compile()

    def ask(self, query: str, programme: str, history_text: str = "") -> dict:
        result = self.graph.invoke(
            {
                "query": query,
                "programme": programme,
                "history_text": history_text or "No earlier messages.",
                "sources": [],
            }
        )
        return {
            "answer": result["answer"],
            "category": result["category"],
            "sources": result.get("sources", []),
        }

