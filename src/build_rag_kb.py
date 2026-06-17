from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "rag_raw"
PROCESSED_DIR = ROOT / "data" / "rag_processed"
CRAWL4AI_HOME = PROCESSED_DIR / ".crawl4ai_home"
USER_AGENT = "VietnameseMedicalChatbotAcademicRAG/0.1 (+educational non-commercial crawl)"
REQUEST_TIMEOUT = 45
RATE_LIMIT_SECONDS = 1.2
MIN_GOOD_WORDS = 220

os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", str(CRAWL4AI_HOME))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class Target:
    topic_group: str
    entity: str
    entity_type: str
    document_type: str
    source: str
    url: str
    title: str
    required_sections: tuple[str, ...] = ()
    supported_risk_categories: tuple[str, ...] = ()
    patient_population: str = "general"
    dynamic: str = ""


@dataclass
class CrawlRecord:
    target: Target
    final_url: str = ""
    status: str = "failed"
    extraction_method: str = ""
    quality: str = "poor"
    words: int = 0
    chunks: int = 0
    error: str = ""
    raw_path: str = ""


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")[:90] or "document"


def clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\[(Skip to main content|Skip navigation|.*?Facebook.*?|.*?Twitter.*?)\]\([^)]+\)", "", text, flags=re.I)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines: list[str] = []
    seen_lines: set[str] = set()
    boilerplate = re.compile(
        r"(cookie|privacy|terms of use|subscribe|newsletter|share this page|"
        r"related links|advertisement|print|email this page|follow us|"
        r"site map|accessibility|language assistance|external links)",
        re.I,
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if boilerplate.search(line) and len(line.split()) < 18:
            continue
        normalized = re.sub(r"\s+", " ", line).lower()
        if normalized in seen_lines and len(normalized) < 120:
            continue
        seen_lines.add(normalized)
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def section_coverage(text: str, required: tuple[str, ...]) -> float:
    if not required:
        return 1.0
    lowered = text.lower()
    hits = sum(1 for section in required if section.lower() in lowered)
    return hits / len(required)


def is_good_extraction(text: str, target: Target) -> tuple[bool, str]:
    words = count_words(text)
    coverage = section_coverage(text, target.required_sections)
    nav_ratio = len(re.findall(r"\b(menu|subscribe|cookie|footer|login|advertisement)\b", text, re.I)) / max(words, 1)
    if words < MIN_GOOD_WORDS:
        return False, f"too_short:{words}"
    if coverage < 0.25 and target.required_sections:
        return False, f"missing_sections:{coverage:.2f}"
    if nav_ratio > 0.02:
        return False, f"boilerplate_ratio:{nav_ratio:.3f}"
    return True, "good"


def robots_allowed(url: str) -> tuple[bool, str]:
    parsed = urllib.parse.urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
        return parser.can_fetch(USER_AGENT, url), robots_url
    except Exception as exc:
        return True, f"{robots_url} (robots read failed: {exc})"


def jina_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return f"https://r.jina.ai/http://{parsed.netloc}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")


def fetch_jina(url: str) -> str:
    response = requests.get(
        jina_url(url),
        headers={"User-Agent": USER_AGENT, "Accept": "text/markdown"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


def source_name(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    if "dailymed" in host:
        return "DailyMed"
    if "medlineplus" in host:
        return "MedlinePlus"
    if "ncbi.nlm.nih.gov" in host:
        return "NCBI Bookshelf"
    if "nhs.uk" in host:
        return "NHS"
    if "cdc.gov" in host:
        return "CDC"
    if "niddk.nih.gov" in host:
        return "NIDDK"
    return host


def discover_dailymed_url(drug: str) -> str | None:
    api = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
    response = requests.get(api, params={"drug_name": drug, "pagesize": 10}, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json().get("data", [])
    if not data:
        return None
    preferred = next((item for item in data if item.get("setid")), data[0])
    setid = preferred.get("setid")
    return f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}" if setid else None


def discover_lactmed_url(drug: str) -> str | None:
    api = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "books", "term": f"{drug} LactMed", "retmode": "json", "retmax": 5}
    response = requests.get(api, params=params, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    ids = response.json().get("esearchresult", {}).get("idlist", [])
    for book_id in ids:
        return f"https://www.ncbi.nlm.nih.gov/books/{book_id}/"
    return None


async def crawl4ai_fetch(url: str) -> str:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, word_count_threshold=20)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        markdown = getattr(result, "markdown", "") or ""
        if hasattr(markdown, "raw_markdown"):
            markdown = markdown.raw_markdown
        if not markdown:
            markdown = getattr(result, "cleaned_html", "") or ""
        return str(markdown)


def split_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_header = "overview"
    current_lines: list[str] = []
    for line in markdown.splitlines():
        header = re.match(r"^(#{1,4})\s+(.+)$", line)
        if header:
            if current_lines:
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = header.group(2).strip()[:120]
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))
    return [(name, body) for name, body in sections if count_words(body) >= 40]


def chunk_text(markdown: str, target: Target, doc_hash: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    section_blocks = split_sections(markdown)
    if not section_blocks:
        section_blocks = [("overview", markdown)]

    token_stream: list[tuple[str, str]] = []
    for section, body in section_blocks:
        for token in re.findall(r"\S+", body):
            token_stream.append((section, token))

    chunk_index = 0
    start = 0
    target_size = 680
    overlap = 150
    while start < len(token_stream):
        end = min(start + target_size, len(token_stream))
        if end - start < 120 and chunks:
            break
        window = token_stream[start:end]
        content = " ".join(token for _, token in window).strip()
        if count_words(content) >= 70:
            normalized = re.sub(r"\W+", " ", content.lower()).strip()
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                section = Counter(section for section, _ in window).most_common(1)[0][0]
                chunk_index += 1
                chunks.append(
                    {
                        "id": f"{slugify(target.topic_group)}__{slugify(target.entity)}__{doc_hash[:10]}__{chunk_index:03d}",
                        "source": target.source,
                        "url": target.url,
                        "title": target.title,
                        "section": section,
                        "entity": target.entity,
                        "entity_type": target.entity_type,
                        "document_type": target.document_type,
                        "topic_group": target.topic_group,
                        "supported_risk_categories": list(target.supported_risk_categories),
                        "patient_population": target.patient_population,
                        "content": content,
                    }
                )
        if end == len(token_stream):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_targets() -> list[Target]:
    drug_sections = ("warnings", "contraindications", "interactions", "pregnancy", "pediatric use", "overdose")
    disease_sections = ("overview", "symptoms", "causes", "treatment", "prevention", "when to seek medical care")
    pregnancy_sections = ("pregnancy safety", "breastfeeding safety", "fetal risk", "neonatal risk", "monitoring")
    triage_sections = ("emergency symptoms", "red flags", "overdose symptoms", "antidote", "when to seek emergency care")

    drug_ids = {
        "Warfarin": "a682277",
        "Metformin": "a696005",
        "Insulin": "a682611",
        "Acetaminophen": "a681004",
        "Aspirin": "a682878",
        "Digoxin": "a682301",
        "Phenytoin": "a682022",
        "Rifampin": "a682403",
        "Ibuprofen": "a682159",
        "Amoxicillin": "a685001",
    }
    lactmed_ids = {
        "Warfarin": "NBK501137",
        "Metformin": "NBK501020",
        "Insulin": "NBK500991",
        "Acetaminophen": "NBK501194",
        "Aspirin": "NBK501196",
        "Digoxin": "NBK501845",
        "Phenytoin": "NBK501273",
        "Rifampin": "NBK501348",
        "Ibuprofen": "NBK500986",
        "Amoxicillin": "NBK500887",
    }
    disease_urls = {
        "Chronic Kidney Disease": "https://medlineplus.gov/chronickidneydisease.html",
        "Liver Disease": "https://medlineplus.gov/liverdiseases.html",
        "Diabetes": "https://medlineplus.gov/diabetes.html",
        "Hypertension": "https://medlineplus.gov/highbloodpressure.html",
        "Heart Failure": "https://medlineplus.gov/heartfailure.html",
        "Asthma": "https://medlineplus.gov/asthma.html",
        "Epilepsy": "https://medlineplus.gov/epilepsy.html",
        "Arrhythmia": "https://medlineplus.gov/arrhythmia.html",
        "Anemia": "https://medlineplus.gov/anemia.html",
        "Peptic Ulcer Disease": "https://medlineplus.gov/pepticulcer.html",
    }
    pop_urls = {
        "Pediatric medication safety": ("population", "https://medlineplus.gov/medicinesandchildren.html", "pediatric"),
        "Neonatal medication safety": ("population", "https://www.nhs.uk/conditions/baby/health/medicines-for-babies-and-children/", "neonatal/pediatric"),
        "Renal impairment medication safety": ("population", "https://www.niddk.nih.gov/health-information/kidney-disease/keeping-kidneys-safe", "renal impairment"),
        "Hepatic impairment medication safety": ("population", "https://www.niddk.nih.gov/health-information/liver-disease", "hepatic impairment"),
        "Elderly medication safety": ("population", "https://medlineplus.gov/olderadultmedicines.html", "older adult"),
        "Pediatric asthma": ("disease", "https://medlineplus.gov/asthmainchildren.html", "children"),
        "Pediatric fever": ("symptom", "https://medlineplus.gov/ency/article/003090.htm", "children"),
        "Pediatric seizure": ("symptom", "https://medlineplus.gov/ency/article/000995.htm", "children"),
        "CKD medication safety": ("population", "https://www.niddk.nih.gov/health-information/kidney-disease/kidney-failure/choosing-treatment", "renal impairment"),
        "Liver disease medication safety": ("population", "https://www.nhs.uk/conditions/cirrhosis/treatment/", "hepatic impairment"),
    }
    triage_urls = {
        "Acetaminophen overdose": "https://medlineplus.gov/ency/article/002598.htm",
        "Iron overdose": "https://medlineplus.gov/ency/article/002659.htm",
        "Insulin overdose": "https://medlineplus.gov/ency/article/002690.htm",
        "Warfarin bleeding": "",
        "Digoxin toxicity": "https://medlineplus.gov/ency/article/002581.htm",
        "Aspirin toxicity": "https://medlineplus.gov/ency/article/002542.htm",
        "Seizure emergency": "https://www.nhs.uk/conditions/what-to-do-if-someone-has-a-seizure-fit/",
        "Severe bleeding": "https://medlineplus.gov/ency/article/000045.htm",
        "Shortness of breath": "https://medlineplus.gov/ency/article/003075.htm",
        "Confusion / delirium": "https://medlineplus.gov/ency/article/003205.htm",
    }

    targets: list[Target] = []
    for drug, med_id in drug_ids.items():
        targets.append(
            Target(
                topic_group="Drug Safety",
                entity=drug,
                entity_type="drug",
                document_type="drug_information",
                source="MedlinePlus Drug Information",
                url=f"https://medlineplus.gov/druginfo/meds/{med_id}.html",
                title=f"{drug} drug information",
                required_sections=drug_sections,
                supported_risk_categories=("warnings", "contraindications", "interactions", "pregnancy", "pediatric", "overdose"),
            )
        )
    for disease, url in disease_urls.items():
        targets.append(
            Target(
                topic_group="Disease Knowledge",
                entity=disease,
                entity_type="disease",
                document_type="disease_overview",
                source="MedlinePlus",
                url=url,
                title=f"{disease} overview",
                required_sections=disease_sections,
                supported_risk_categories=("symptoms", "causes", "treatment", "prevention", "triage"),
            )
        )
    for drug in drug_ids:
        targets.append(
            Target(
                topic_group="Pregnancy & Lactation",
                entity=drug,
                entity_type="drug",
                document_type="pregnancy_lactation_reference",
                source="NCBI Bookshelf LactMed",
                url=f"https://www.ncbi.nlm.nih.gov/books/{lactmed_ids[drug]}/",
                title=f"{drug} pregnancy and lactation safety",
                required_sections=("summary of use during lactation", "drug levels", "effects in breastfed infants"),
                supported_risk_categories=("pregnancy", "breastfeeding", "fetal_risk", "neonatal_risk", "monitoring"),
                patient_population="pregnancy/lactation",
            )
        )
    for entity, (etype, url, population) in pop_urls.items():
        targets.append(
            Target(
                topic_group="Pediatric & Special Populations",
                entity=entity,
                entity_type=etype,
                document_type="special_population_reference",
                source=source_name(url),
                url=url,
                title=entity,
                supported_risk_categories=("pediatric", "neonatal", "renal_impairment", "hepatic_impairment", "geriatric"),
                patient_population=population,
            )
        )
    for entity, url in triage_urls.items():
        targets.append(
            Target(
                topic_group="Overdose & Triage",
                entity=entity,
                entity_type="symptom",
                document_type="overdose_toxicology" if "overdose" in entity.lower() or "toxicity" in entity.lower() else "symptom_triage",
                source=source_name(url),
                url=url,
                title=entity,
                required_sections=triage_sections,
                supported_risk_categories=("overdose", "emergency_red_flags", "triage", "antidote"),
                dynamic="dailymed" if entity == "Warfarin bleeding" else "",
            )
        )
    return targets


def resolve_dynamic_targets(targets: list[Target]) -> tuple[list[Target], list[str]]:
    resolved: list[Target] = []
    failures: list[str] = []
    for target in targets:
        if target.dynamic == "lactmed":
            try:
                url = discover_lactmed_url(target.entity)
                time.sleep(RATE_LIMIT_SECONDS)
            except Exception as exc:
                failures.append(f"{target.entity}: LactMed discovery failed: {exc}")
                url = None
            if url:
                resolved.append(Target(**{**target.__dict__, "url": url}))
            else:
                fallback_url = discover_dailymed_url(target.entity)
                if fallback_url:
                    resolved.append(Target(**{**target.__dict__, "url": fallback_url, "source": "DailyMed", "document_type": "drug_label_population_sections"}))
                else:
                    failures.append(f"{target.entity}: no LactMed or DailyMed fallback URL")
        elif target.dynamic == "dailymed":
            try:
                drug = target.entity.split()[0]
                url = discover_dailymed_url(drug)
                time.sleep(RATE_LIMIT_SECONDS)
            except Exception as exc:
                failures.append(f"{target.entity}: DailyMed discovery failed: {exc}")
                url = None
            if url:
                resolved.append(Target(**{**target.__dict__, "url": url, "source": "DailyMed"}))
            else:
                failures.append(f"{target.entity}: no DailyMed URL")
        else:
            resolved.append(target)
    return resolved, failures


async def process_target(target: Target) -> tuple[CrawlRecord, list[dict[str, Any]]]:
    record = CrawlRecord(target=target, final_url=target.url)
    allowed, robots = robots_allowed(target.url)
    if not allowed:
        record.error = f"robots_disallowed via {robots}"
        return record, []

    raw = ""
    try:
        raw = await crawl4ai_fetch(target.url)
        raw = clean_markdown(raw)
        good, reason = is_good_extraction(raw, target)
        if good:
            record.status = "success"
            record.extraction_method = "crawl4ai"
            record.quality = "good"
        else:
            record.error = f"crawl4ai poor extraction: {reason}"
    except Exception as exc:
        record.error = f"crawl4ai failed: {exc}"

    if record.status != "success":
        try:
            raw = clean_markdown(fetch_jina(target.url))
            good, reason = is_good_extraction(raw, target)
            record.status = "success" if good or count_words(raw) >= 140 else "partial"
            record.extraction_method = "jina_reader"
            record.quality = "good" if good else f"fallback_partial:{reason}"
        except Exception as exc:
            record.error = f"{record.error}; jina failed: {exc}"
            return record, []

    doc_hash = hashlib.sha256(f"{target.url}\n{raw}".encode("utf-8")).hexdigest()
    raw_name = f"{slugify(target.topic_group)}__{slugify(target.entity)}__{doc_hash[:10]}.md"
    raw_path = RAW_DIR / raw_name
    frontmatter = {
        "source": target.source,
        "url": target.url,
        "title": target.title,
        "entity": target.entity,
        "entity_type": target.entity_type,
        "document_type": target.document_type,
        "topic_group": target.topic_group,
        "patient_population": target.patient_population,
        "extraction_method": record.extraction_method,
    }
    raw_path.write_text("---\n" + json.dumps(frontmatter, ensure_ascii=False, indent=2) + "\n---\n\n" + raw + "\n", encoding="utf-8")
    chunks = chunk_text(raw, target, doc_hash)
    record.words = count_words(raw)
    record.chunks = len(chunks)
    record.raw_path = str(raw_path.relative_to(ROOT))
    return record, chunks


def dedupe_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        normalized = re.sub(r"\W+", " ", chunk["content"].lower()).strip()
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(chunk)
    return unique, len(chunks) - len(unique)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(records: list[CrawlRecord], chunks: list[dict[str, Any]], discovery_failures: list[str], duplicate_chunks: int) -> None:
    docs = [record for record in records if record.status in {"success", "partial"}]
    failed = [record for record in records if record.status == "failed"]
    per_group = Counter(record.target.topic_group for record in docs)
    per_source = Counter(record.target.source for record in docs)
    inventory = "\n".join(f"- {record.target.topic_group} | {record.target.source} | {record.target.entity}: {record.target.url}" for record in docs)
    failed_lines = "\n".join(f"- {record.target.entity}: {record.error} ({record.target.url})" for record in failed) or "- None"
    duplicate_urls = len(records) - len({record.target.url for record in records if record.target.url})
    report = f"""# RAG Crawl Report

Academic project: Intelligent Medical Assistant for Healthcare Consultation

This crawl collected trusted-source documents only, using Crawl4AI first and Jina Reader as fallback when extraction failed or quality was poor. It prepared markdown and JSONL chunks for RAG ingestion only. No embeddings, vector database, FAISS, ChromaDB, or model training were created.

## Final Report

- Total documents collected: {len(docs)}
- Total chunks: {len(chunks)}
- Duplicate URLs removed: {duplicate_urls}
- Duplicate chunks removed: {duplicate_chunks}

## Documents Per Topic Group

{chr(10).join(f'- {group}: {count}' for group, count in sorted(per_group.items()))}

## Documents Per Source

{chr(10).join(f'- {source}: {count}' for source, count in sorted(per_source.items()))}

## Failed URLs

{failed_lines}

## Discovery Notes

{chr(10).join(f'- {item}' for item in discovery_failures) if discovery_failures else '- None'}

## Final URL Inventory

{inventory}

## Output Files

- Raw markdown documents: `data/rag_raw/`
- Processed chunks: `data/rag_processed/rag_chunks.jsonl`
- Crawl logs: `data/rag_processed/crawl_log.csv`
- Source statistics: `data/rag_processed/source_statistics.csv`
"""
    (ROOT / "README_RAG_CRAWL.md").write_text(report, encoding="utf-8")


async def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CRAWL4AI_HOME.mkdir(parents=True, exist_ok=True)
    for old_raw in RAW_DIR.glob("*.md"):
        old_raw.unlink()

    targets, discovery_failures = resolve_dynamic_targets(build_targets())
    records: list[CrawlRecord] = []
    all_chunks: list[dict[str, Any]] = []

    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {target.topic_group} - {target.entity} - {target.url}", flush=True)
        record, chunks = await process_target(target)
        records.append(record)
        all_chunks.extend(chunks)
        print(f"  -> {record.status} via {record.extraction_method or 'none'}; words={record.words}; chunks={len(chunks)}; {record.error}", flush=True)
        time.sleep(RATE_LIMIT_SECONDS)

    all_chunks, duplicate_chunks = dedupe_chunks(all_chunks)
    with (PROCESSED_DIR / "rag_chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    log_rows = [
        {
            "topic_group": r.target.topic_group,
            "entity": r.target.entity,
            "source": r.target.source,
            "url": r.target.url,
            "status": r.status,
            "extraction_method": r.extraction_method,
            "quality": r.quality,
            "words": r.words,
            "chunks": r.chunks,
            "raw_path": r.raw_path,
            "error": r.error,
        }
        for r in records
    ]
    write_csv(
        PROCESSED_DIR / "crawl_log.csv",
        log_rows,
        ["topic_group", "entity", "source", "url", "status", "extraction_method", "quality", "words", "chunks", "raw_path", "error"],
    )

    stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"documents": 0, "chunks": 0, "words": 0})
    for r in records:
        if r.status in {"success", "partial"}:
            key = (r.target.source, r.target.topic_group)
            stats[key]["documents"] += 1
            stats[key]["chunks"] += r.chunks
            stats[key]["words"] += r.words
    stat_rows = [
        {"source": source, "topic_group": topic_group, **values}
        for (source, topic_group), values in sorted(stats.items())
    ]
    write_csv(PROCESSED_DIR / "source_statistics.csv", stat_rows, ["source", "topic_group", "documents", "chunks", "words"])
    write_report(records, all_chunks, discovery_failures, duplicate_chunks)


if __name__ == "__main__":
    asyncio.run(main())
