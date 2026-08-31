"""Gradio entry point for the College AI Assistant."""

from __future__ import annotations

from functools import lru_cache

import gradio as gr

from src.workflow import CollegeAssistant


PROGRAMMES = ["B.Tech CSE", "B.Sc Data Science", "BBA"]


@lru_cache(maxsize=1)
def get_assistant() -> CollegeAssistant:
    """Create the assistant only when the first question is submitted."""

    return CollegeAssistant()


def history_to_text(history: list | None, limit: int = 4) -> str:
    """Convert recent Gradio history into short plain text for the prompt."""

    if not history:
        return "No earlier messages."

    lines: list[str] = []
    for item in history[-limit:]:
        if isinstance(item, dict):
            role = str(item.get("role", "message")).title()
            content = item.get("content", "")
            if isinstance(content, str):
                lines.append(f"{role}: {content}")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            lines.append(f"Student: {item[0]}")
            lines.append(f"Assistant: {item[1]}")
    return "\n".join(lines) or "No earlier text messages."


def chat(message: str, history: list | None, programme: str) -> str:
    """Run one student question through the LangGraph workflow."""

    if not message or not message.strip():
        return "Please enter a question."

    try:
        result = get_assistant().ask(
            query=message.strip(),
            programme=programme or PROGRAMMES[0],
            history_text=history_to_text(history),
        )
    except Exception as exc:  # Friendly boundary for a classroom UI.
        return (
            "I could not start the assistant. Check that `.env` contains a valid "
            "`MISTRAL_API_KEY` and that `python build_indexes.py` completed.\n\n"
            f"Technical detail: `{type(exc).__name__}: {exc}`"
        )

    source_text = ", ".join(result["sources"]) if result["sources"] else "No PDF retrieval (general route)"
    return (
        f"{result['answer']}\n\n"
        "---\n"
        f"**Route:** {result['category'].title()}  \n"
        f"**Retrieved sources:** {source_text}"
    )


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        "# College AI Assistant\n"
        "Ask about academics, attendance, examinations, course requirements, or fees. "
        "This classroom demo uses fictional Northstar College documents."
    )
    programme_input = gr.Dropdown(
        PROGRAMMES,
        value=PROGRAMMES[0],
        label="Student programme",
        info="The answer uses this value for programme-specific guidance.",
    )
    chatbot = gr.Chatbot(
        height=470,
        placeholder="Try: What attendance do I need for the end-semester exam?",
    )
    gr.ChatInterface(
        fn=chat,
        chatbot=chatbot,
        additional_inputs=[programme_input],
        textbox=gr.Textbox(placeholder="Type a college-related question...", scale=7),
        examples=[
            ["What is the minimum attendance required for the end-semester exam?", "B.Tech CSE"],
            ["How many credits are required for my degree?", "B.Sc Data Science"],
            ["What is my semester tuition and when is it due?", "BBA"],
            ["Explain what a prerequisite course means.", "B.Tech CSE"],
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(show_error=True)

