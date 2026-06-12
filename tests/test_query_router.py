from src.query_router import QueryRouter


def test_router_drug_interaction():
    router = QueryRouter()
    result = router.classify("Can I take ibuprofen with warfarin?", "en")
    assert result.category == "drug_interaction"
    assert result.requires_rag is True
    assert "warfarin" in result.entities


def test_router_out_of_scope():
    router = QueryRouter()
    result = router.classify("How to cook spaghetti?", "en")
    assert result.category == "out_of_scope"
    assert result.requires_rag is False


def test_router_pregnancy():
    router = QueryRouter()
    result = router.classify("Can pregnant women take acetaminophen?", "en")
    assert result.category == "pregnancy"
    assert result.risk_level == "critical"
