from config import CATEGORIES_PATH
from src.safety_guard import SafetyGuard


def test_emergency_detection_vi():
    sg = SafetyGuard(CATEGORIES_PATH)
    emergencies = [
        "toi uong qua lieu paracetamol",
        "dau nguc du doi",
        "kho tho khong tho duoc",
        "muon tu tu",
        "co giat bat tinh",
    ]
    for query in emergencies:
        assert sg.is_emergency(query), f"Failed to detect: {query}"


def test_emergency_detection_en():
    sg = SafetyGuard(CATEGORIES_PATH)
    emergencies = [
        "severe chest pain",
        "I took too many pills",
        "difficulty breathing",
        "suicidal thoughts",
    ]
    for query in emergencies:
        assert sg.is_emergency(query), f"Failed to detect: {query}"


def test_non_emergency():
    sg = SafetyGuard(CATEGORIES_PATH)
    normal = [
        "tac dung phu cua warfarin",
        "what is diabetes",
        "thuoc metformin",
    ]
    for query in normal:
        assert not sg.is_emergency(query), f"False positive: {query}"


def test_scope_classification():
    sg = SafetyGuard(CATEGORIES_PATH)
    assert sg.is_medical_scope("thuoc warfarin")[0] is True
    assert sg.is_medical_scope("thoi tiet hom nay")[0] is False
    assert sg.is_medical_scope("side effects of aspirin")[0] is True
