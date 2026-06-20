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
            label="Tác dụng phụ của Warfarin",
            message="Tác dụng phụ của Warfarin là gì?",
        ),
        cl.Starter(
            label="Tương tác thuốc",
            message="Tôi có thể uống ibuprofen cùng với warfarin không?",
        ),
        cl.Starter(
            label="Triệu chứng tiểu đường",
            message="Triệu chứng của bệnh tiểu đường tuýp 2 là gì?",
        ),
        cl.Starter(
            label="Thai kỳ và thuốc",
            message="Phụ nữ mang thai có thể uống Acetaminophen không?",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    stats = get_pipeline().rag_pipeline.vector_store.get_stats()
    await cl.Message(
        content=(
            "Trợ lý Y tế đã sẵn sàng.\n\n"
            f"Dữ liệu: {stats['vi_count']} tài liệu y khoa\n"
            "Tìm kiếm: Kết hợp Vector + BM25\n"
            "An toàn: phát hiện khẩn cấp, phạm vi, trích dẫn"
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
                name=f"[{source['index']}] {source.get('title') or 'Nguồn'}",
                content=(
                    f"Nguồn: {source.get('source', '')}\n"
                    f"URL: {source.get('url', '')}\n"
                    f"Mục: {source.get('section', '')}\n"
                    f"Điểm: {source.get('score', 0):.4f}"
                ),
                display="side",
            )
        )

    risk = result.get("risk_level", "medium")
    confidence = result.get("confidence", "unknown")
    route = result.get("route", "rag")
    category = result.get("category", "unknown")
    route_label = "🔍 RAG Pipeline" if route != "general_qa" else "🤖 Local LLM"
    content = (
        f"📋 Danh mục: {category} | Rủi ro: {risk} | Độ tự tin: {confidence}\n"
        f"🔀 Luồng: {route_label}\n\n"
        f"{result.get('answer', '')}\n\n"
        f"---\n{result.get('disclaimer', '')}"
    )
    await cl.Message(content=content, elements=elements).send()
