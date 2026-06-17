from src.query_router import QueryClassification
from src.response_generator import ResponseGenerator


def test_find_sentence_with_terms_uses_english_warning_for_english_terms():
    warning = ResponseGenerator()._find_sentence_with_terms(
        "Warfarin may cause gangrene or skin necrosis.",
        ["necrosis", "gangrene", "purple", "dark"],
    )

    assert warning.startswith("Warfarin may cause necrosis")


def test_extract_interaction_answer_uses_interaction_warning():
    generator = ResponseGenerator()
    classification = QueryClassification(
        intent="interaction_check",
        category="drug_interaction",
        entities=["ibuprofen", "warfarin"],
        risk_level="critical",
        confidence=0.88,
        requires_rag=True,
        language="en",
    )
    chunks = [
        {
            "id": "warfarin-1",
            "content": (
                "Warfarin comes as a tablet to take by mouth. "
                "* NSAIDs such as ibuprofen may interact with warfarin. "
                "Do not start ibuprofen while taking warfarin without discussing "
                "it with your healthcare provider."
            ),
            "metadata": {
                "title": "Warfarin drug information",
                "source": "MedlinePlus",
                "category": "drug_safety",
            },
        }
    ]

    answer, _ = generator._generate_extractive(
        "Can I take ibuprofen with warfarin?", chunks, classification
    )

    assert "ibuprofen may interact with warfarin" in answer
    assert "without discussing" in answer
    assert "comes as a tablet" not in answer
