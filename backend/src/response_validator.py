from __future__ import annotations

import re
from typing import Any

from src.safety_guard import get_disclaimer
from src.utils import normalize_for_match


class ResponseValidator:
    """Post-generation safety checks."""

    PROHIBITED_PATTERNS = [
        r"you should take \d+ ?mg",
        r"i diagnose you with",
        r"i prescribe",
        r"this will cure",
        r"toi chan doan",
        r"ban nen uong \d+",
        r"thuoc nay se chua",
    ]

    def validate(self, response: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
        answer = response.get("answer", "")
        issues: list[str] = []

        cited_numbers = [int(num) for num in re.findall(r"\[(\d+)\]", answer)]
        for number in cited_numbers:
            if number > len(chunks):
                issues.append(f"Citation [{number}] references a missing source")

        normalized_answer = normalize_for_match(answer)
        for pattern in self.PROHIBITED_PATTERNS:
            if re.search(pattern, normalized_answer, flags=re.IGNORECASE):
                issues.append(f"Prohibited pattern found: {pattern}")

        category = response.get("category")
        exit_type = response.get("exit_type")
        if len(answer) > 100 and not cited_numbers and category not in ("faq", "out_of_scope") and exit_type != "insufficient_evidence":
            issues.append("Answer contains medical claims but no citations")

        language = response.get("language", "vi")
        risk_level = response.get("risk_level", "medium")
        response["disclaimer"] = get_disclaimer(risk_level)
        response["validation_issues"] = issues
        response["is_valid"] = len(issues) == 0

        if any("Prohibited pattern" in issue for issue in issues):
            response["answer"] = (
                "Toi khong the dua ra chan doan, don thuoc, hoac lieu dung ca nhan. "
                "Vui long trao doi voi bac si hoac duoc si."
                if language == "vi"
                else "I cannot provide diagnosis, prescriptions, or personal dosage instructions. "
                "Please consult a doctor or pharmacist."
            )
        return response
