from src.evidence_grader import EvidenceGrader
from src.topic_relevance import should_route_pregnancy_to_drug_safety


def test_pregnancy_avoidance_rejects_lactation_only_evidence():
    grader = EvidenceGrader(use_llm=False)
    question = "phụ nữ có thai không nên uống thuốc gì"
    chunks = [
        {
            "content": (
                "Nồng độ thuốc trong sữa mẹ được đo sau sinh. Trẻ bú mẹ nhận "
                "một lượng rất nhỏ qua sữa, và nghiên cứu theo dõi việc cho con bú."
            ),
            "metadata": {
                "source": "NCBI Bookshelf LactMed",
                "title": "Insulin pregnancy and lactation safety",
                "section": "Drug Levels",
                "category": "pregnancy",
            },
            "score": 0.9,
        }
    ]

    result = grader.grade(question, chunks)

    assert result.relevant_chunks == []
    assert result.needs_crawl is True


def test_pregnancy_avoidance_accepts_direct_drug_warning():
    grader = EvidenceGrader(use_llm=False)
    question = "phụ nữ có thai không nên uống thuốc gì"
    chunks = [
        {
            "content": (
                "Ibuprofen có thể gây hại cho thai nhi và gây biến chứng khi dùng "
                "từ tuần thứ 20 trở đi trong thai kỳ. Không dùng ibuprofen sau "
                "tuần thứ 20 của thai kỳ trừ khi bác sĩ chỉ định."
            ),
            "metadata": {
                "source": "MedlinePlus Drug Information",
                "title": "Ibuprofen drug information",
                "section": "Before taking ibuprofen",
                "category": "drug_safety",
            },
            "score": 0.9,
        }
    ]

    result = grader.grade(question, chunks)

    assert len(result.relevant_chunks) == 1
    assert result.relevant_chunks[0]["relevance_score"] >= 0.5


def test_pregnancy_avoidance_rejects_unrelated_avoidance_sentence():
    grader = EvidenceGrader(use_llm=False)
    question = "phụ nữ có thai không nên uống thuốc gì"
    chunks = [
        {
            "content": (
                "Không nên bắt đầu dùng bất kỳ thuốc nào này trong khi dùng phenytoin "
                "mà không tham khảo ý kiến chuyên gia y tế. Thông báo với bác sĩ nếu "
                "bạn đang mang thai, dự định mang thai hoặc đang cho con bú."
            ),
            "metadata": {
                "source": "MedlinePlus Drug Information",
                "title": "Phenytoin drug information",
                "section": "Before taking phenytoin",
                "category": "drug_safety",
            },
            "score": 0.9,
        }
    ]

    result = grader.grade(question, chunks)

    assert result.relevant_chunks == []


def test_specific_drug_question_rejects_other_drug_pregnancy_warning():
    grader = EvidenceGrader(use_llm=False)
    question = "Can pregnant women take acetaminophen?"
    chunks = [
        {
            "content": "Phenytoin may harm the fetus if used during pregnancy.",
            "metadata": {
                "entity": "Phenytoin",
                "title": "Phenytoin drug information",
                "category": "drug_safety",
            },
            "score": 0.9,
        },
        {
            "content": (
                "Tell your doctor if you are pregnant, plan to become pregnant, "
                "or are breastfeeding. If you become pregnant while taking "
                "acetaminophen, call your doctor."
            ),
            "metadata": {
                "entity": "Acetaminophen",
                "title": "Acetaminophen drug information",
                "category": "drug_safety",
            },
            "score": 0.6,
        },
    ]

    result = grader.grade(question, chunks)

    assert len(result.relevant_chunks) == 1
    assert result.relevant_chunks[0]["metadata"]["entity"] == "Acetaminophen"


def test_interaction_question_prefers_interaction_evidence():
    grader = EvidenceGrader(use_llm=False)
    question = "Can I take ibuprofen with warfarin?"
    chunks = [
        {
            "content": "Warfarin comes as a tablet to take by mouth once a day.",
            "metadata": {"entity": "Warfarin", "category": "drug_safety"},
            "score": 0.9,
        },
        {
            "content": (
                "NSAIDs such as ibuprofen may interact with warfarin. Do not start "
                "ibuprofen while taking warfarin without discussing it with your healthcare provider."
            ),
            "metadata": {"entity": "Warfarin", "category": "drug_safety"},
            "score": 0.8,
        },
    ]

    result = grader.grade(question, chunks)

    assert len(result.relevant_chunks) == 1
    assert "ibuprofen" in result.relevant_chunks[0]["content"].lower()
    assert result.relevant_chunks[0]["relevance_score"] >= 0.5


def test_pregnancy_medication_query_routes_to_drug_safety():
    assert should_route_pregnancy_to_drug_safety(
        "phụ nữ có thai không nên uống thuốc gì"
    )
    assert not should_route_pregnancy_to_drug_safety(
        "phụ nữ cho con bú dùng thuốc gì an toàn"
    )
