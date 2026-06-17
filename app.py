from __future__ import annotations

import chainlit as cl

from src.langgraph_pipeline import LangGraphPipeline


pipeline: LangGraphPipeline | None = None


def get_pipeline() -> LangGraphPipeline:
    global pipeline
    if pipeline is None:
        pipeline = LangGraphPipeline()
    return pipeline


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Warfarin side effects",
            message="What are the side effects of Warfarin?",
        ),
        cl.Starter(
            label="Drug interaction",
            message="Can I take ibuprofen with warfarin?",
        ),
        cl.Starter(
            label="Diabetes symptoms",
            message="What are symptoms of type 2 diabetes?",
        ),
        cl.Starter(
            label="Pregnancy and medicine",
            message="Can pregnant women take Acetaminophen?",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    stats = get_pipeline().rag_pipeline.vector_store.get_stats()
    await cl.Message(
        content=(
            "Medical Assistant ready.\n\n"
            f"Dataset: {stats['vi_count']} Medical Cases\n"
            "Search: Hybrid vector + BM25\n"
            "Safety: emergency, scope, evidence, citations"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    result = get_pipeline().process_query(message.content)

    if result.get("type") in {"emergency", "out_of_scope", "insufficient_evidence"}:
        await cl.Message(content=result["message"]).send()
        return

    elements = []
    for source in result.get("sources", []):
        elements.append(
            cl.Text(
                name=f"[{source['index']}] {source.get('title') or 'Source'}",
                content=(
                    f"Source: {source.get('source', '')}\n"
                    f"URL: {source.get('url', '')}\n"
                    f"Section: {source.get('section', '')}\n"
                    f"Score: {source.get('score', 0):.4f}"
                ),
                display="side",
            )
        )

    risk = result.get("risk_level", "medium")
    confidence = result.get("confidence", "unknown")
    content = (
        f"Risk: {risk} | Evidence confidence: {confidence}\n\n"
        f"{result.get('answer', '')}\n\n"
        f"---\n{result.get('disclaimer', '')}"
    )
    await cl.Message(content=content, elements=elements).send()
