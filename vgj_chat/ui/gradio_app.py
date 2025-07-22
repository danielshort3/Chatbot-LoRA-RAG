from __future__ import annotations

import gradio as gr

from ..models.rag import _ensure_boot, answer_stream


def user_submit(msg: str, hist: list[dict[str, str]]):
    hist.append({"role": "user", "content": msg})
    return "", hist


page_title = "Unofficial Visit\u00a0Grand\u00a0Junction\u00a0Demo – not endorsed by VGJ"


def build_demo() -> gr.Blocks:
    _ensure_boot()
    with gr.Blocks(theme=gr.themes.Soft(), title=page_title) as demo:
        gr.Markdown(
            (
                "## \ud83d\udcac Unofficial Visit\u202fGrand\u202fJunction\u202fDemo Chatbot\n"
                "<small>Portfolio prototype, **not** endorsed by Visit\u202fGrand\u202fJunction. "
                "Content sourced from public VGJ blogs under a fair-use rationale.</small>"
            )
        )

        chat_state = gr.State([])

        chatbox = gr.Chatbot(height=450, type="messages", label="Conversation")
        textbox = gr.Textbox(
            placeholder="Ask about Grand Junction…",
            show_label=False,
            container=False,
        )

        textbox.submit(
            user_submit,
            inputs=[textbox, chat_state],
            outputs=[textbox, chat_state],
        ).then(
            answer_stream,
            inputs=[chat_state],
            outputs=[chatbox, chat_state],
        )
    return demo


demo = build_demo()
