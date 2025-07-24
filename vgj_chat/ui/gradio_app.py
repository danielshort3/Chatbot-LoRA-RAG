from __future__ import annotations

import asyncio

import gradio as gr

from .. import run_baseline, run_enhanced
from ..config import CFG
from ..models.rag import _ensure_boot, answer_stream


def user_submit(msg: str, hist: list[dict[str, str]]):
    hist.append({"role": "user", "content": msg})
    return "", hist


page_title = "Unofficial Visit Grand Junction Demo - not endorsed by VGJ"


def build_demo() -> gr.Blocks:
    _ensure_boot()
    with gr.Blocks(theme=gr.themes.Soft(), title=page_title) as demo:
        gr.Markdown(
            (
                "## Unofficial Visit Grand Junction Demo Chatbot\n"
                "<small>Portfolio prototype, **not** endorsed by Visit Grand Junction. "
                "Content sourced from public VGJ blogs and listings under a fair-use rationale.</small>"
            )
        )

        if CFG.compare_mode:
            enhanced_state = gr.State([])
            baseline_state = gr.State([])

            with gr.Row():
                baseline_box = gr.Chatbot(
                    type="messages",
                    label="Baseline (raw)",
                )
                enhanced_box = gr.Chatbot(
                    type="messages",
                    label="Enhanced (LoRA + FAISS)",
                )

            textbox = gr.Textbox(
                placeholder="Ask about Grand Junction...",
                show_label=False,
                container=False,
            )


            def submit(msg: str, hist_e: list, hist_b: list):
                hist_e.append({"role": "user", "content": msg})
                hist_b.append({"role": "user", "content": msg})
                return "", hist_e, hist_b

            async def respond_enhanced(hist: list):
                answer = await asyncio.to_thread(run_enhanced, hist[-1]["content"])
                hist.append({"role": "assistant", "content": answer})
                return hist, hist

            async def respond_baseline(hist: list):
                answer = await asyncio.to_thread(run_baseline, hist[-1]["content"])
                hist.append({"role": "assistant", "content": answer})
                return hist, hist

            event = textbox.submit(
                submit,
                inputs=[textbox, enhanced_state, baseline_state],
                outputs=[textbox, enhanced_state, baseline_state],
            )

            event.then(
                respond_enhanced,
                inputs=[enhanced_state],
                outputs=[enhanced_box, enhanced_state],
            )

            event.then(
                respond_baseline,
                inputs=[baseline_state],
                outputs=[baseline_box, baseline_state],
            )

        else:
            chat_state = gr.State([])

            chatbox = gr.Chatbot(type="messages", label="Conversation")
            textbox = gr.Textbox(
                placeholder="Ask about Grand Junction...",
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
