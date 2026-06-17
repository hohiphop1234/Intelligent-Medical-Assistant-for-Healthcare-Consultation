from src import data_cleaner as data_cleaner_module
from src.data_cleaner import DataCleaner


def test_resolve_raw_dir_accepts_downloads_data_layout(tmp_path, monkeypatch):
    source_base = tmp_path / "data"
    rag_processed = source_base / "rag_processed"
    rag_processed.mkdir(parents=True)
    (rag_processed / "rag_chunks.jsonl").write_text("{}\n", encoding="utf-8")
    (rag_processed / "clean_rag_chunks_vi.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        data_cleaner_module, "RAW_DATA_FALLBACK_DIRS", [str(source_base)]
    )

    resolved = DataCleaner()._resolve_raw_dir(str(tmp_path / "missing"))

    assert resolved == rag_processed
