# Part 3: Safety & Guardrails (~2h)

> Phụ thuộc: Part 1 (categories.json)
> Có thể làm song song với Part 4

---

## 3.1 Emergency Detection

#### [NEW] `src/safety_guard.py`

**Đây là layer QUAN TRỌNG NHẤT** — chạy trước mọi thứ, không qua RAG, không qua LLM.

```python
import re

EMERGENCY_PATTERNS = {
    "en": [
        r"\bchest\s+pain\b", r"\bcan'?t\s+breathe\b", r"\bdifficulty\s+breathing\b",
        r"\bstroke\b", r"\bheart\s+attack\b", r"\boverdose[d]?\b", r"\btook\s+too\s+many\b",
        r"\bsuicid(?:e|al)\b", r"\bwant\s+to\s+die\b", r"\bself[- ]?harm\b",
        r"\bsevere\s+bleeding\b", r"\bunconscious\b", r"\bseizure\b",
        r"\ballergic\s+reaction\b.*\bswelling\b", r"\banaphyla\b",
        r"\bpoisoning\b", r"\bnot\s+responsive\b",
    ],
    "vi": [
        r"\bđau\s+ngực\b", r"\bkhó\s+thở\b", r"\bkhông\s+thở\s+được\b",
        r"\bđột\s+quỵ\b", r"\bnhồi\s+máu\b", r"\bquá\s+liều\b", r"\buống\s+quá\s+nhiều\b",
        r"\btự\s+tử\b", r"\bmuốn\s+chết\b", r"\btự\s+gây\s+thương\s+tích\b",
        r"\bchảy\s+máu\s+nhiều\b", r"\bbất\s+tỉnh\b", r"\bco\s+giật\b",
        r"\bdị\s+ứng\b.*\bsưng\b", r"\bsốc\s+phản\s+vệ\b",
        r"\bngộ\s+độc\b", r"\bhôn\s+mê\b",
    ]
}
```

### Emergency Response

```python
EMERGENCY_RESPONSES = {
    "vi": """🚨 **CẢNH BÁO KHẨN CẤP**

Triệu chứng bạn mô tả có thể là tình huống cấp cứu y tế.

→ Hãy **GỌI CẤP CỨU (115)** hoặc đến cơ sở y tế gần nhất **NGAY LẬP TỨC**.
→ Nếu ở Mỹ: gọi **911**

⚠️ Tôi không thể thay thế bác sĩ trong tình huống khẩn cấp.
Vui lòng tìm kiếm sự trợ giúp y tế trực tiếp ngay.""",

    "en": """🚨 **EMERGENCY WARNING**

The symptoms you describe may indicate a medical emergency.

→ **Call emergency services (911)** or go to the nearest hospital **IMMEDIATELY**.
→ In Vietnam: call **115**

⚠️ I cannot replace a doctor in emergency situations.
Please seek immediate medical attention."""
}
```

---

## 3.2 Scope Classification

Kiểm tra câu hỏi có thuộc scope y tế hay không:

```python
class SafetyGuard:
    def __init__(self, categories_path: str):
        self.categories = load_categories(categories_path)
        self.emergency_patterns = compile_patterns(EMERGENCY_PATTERNS)
    
    # === LAYER 1: Emergency Check (regex, instant) ===
    def is_emergency(self, query: str) -> bool:
        query_lower = query.lower()
        for lang_patterns in self.emergency_patterns.values():
            for pattern in lang_patterns:
                if pattern.search(query_lower):
                    return True
        return False
    
    def emergency_response(self, query: str, language: str) -> dict:
        return {
            "type": "emergency",
            "message": EMERGENCY_RESPONSES[language],
            "risk_level": "critical",
            "requires_human": True,
        }
    
    # === LAYER 2: Scope Check (keyword-based, fast) ===
    def is_medical_scope(self, query: str) -> tuple[bool, float]:
        """Quick keyword check against category definitions"""
        query_lower = query.lower()
        max_score = 0.0
        for cat in self.categories:
            keywords = cat["keywords_en"] + cat["keywords_vi"]
            matches = sum(1 for kw in keywords if kw in query_lower)
            score = matches / len(keywords) if keywords else 0
            max_score = max(max_score, score)
        return max_score > 0.1, max_score
    
    def out_of_scope_response(self, query: str, language: str) -> dict:
        messages = {
            "vi": """Câu hỏi này nằm ngoài phạm vi kiến thức y tế mà tôi được đào tạo.

Tôi có thể hỗ trợ về:
• 💊 Thuốc và tác dụng phụ
• ⚠️ Tương tác thuốc
• 🏥 Thông tin bệnh lý
• 🤰 Thai kỳ và thuốc
• 👶 Nhi khoa
• 👴 Người cao tuổi

Vui lòng đặt câu hỏi trong phạm vi trên, hoặc tham khảo ý kiến chuyên gia.""",
            "en": """This question is outside my medical knowledge scope.

I can help with:
• 💊 Drug information and side effects
• ⚠️ Drug interactions
• 🏥 Disease & condition info
• 🤰 Pregnancy & medication
• 👶 Pediatric care
• 👴 Elderly care

Please ask within these topics, or consult a healthcare professional."""
        }
        return {
            "type": "out_of_scope",
            "message": messages[language],
            "risk_level": "none",
        }
    
    # === LAYER 3: Insufficient Evidence ===
    def insufficient_evidence_response(self, query: str, language: str) -> dict:
        messages = {
            "vi": """Tôi không tìm được đủ thông tin đáng tin cậy để trả lời câu hỏi này một cách chính xác.

Để đảm bảo an toàn, vui lòng:
• Tham khảo ý kiến bác sĩ hoặc dược sĩ
• Tra cứu trên MedlinePlus (medlineplus.gov)
• Gọi đường dây tư vấn y tế""",
            "en": """I couldn't find sufficient reliable information to answer this question accurately.

For your safety, please:
• Consult a doctor or pharmacist
• Check MedlinePlus (medlineplus.gov)
• Call a medical helpline"""
        }
        return {
            "type": "insufficient_evidence",
            "message": messages[language],
            "risk_level": "medium",
        }
```

---

## 3.3 Mandatory Disclaimers

**Risk-based disclaimer levels:**

```python
DISCLAIMERS = {
    "low": {
        "vi": "ℹ️ *Thông tin tham khảo. Tham khảo bác sĩ nếu cần.*",
        "en": "ℹ️ *For informational purposes. Consult a doctor if needed.*",
    },
    "medium": {
        "vi": "⚕️ *Thông tin chỉ mang tính tham khảo, không thay thế tư vấn y tế chuyên nghiệp. Vui lòng tham khảo ý kiến bác sĩ.*",
        "en": "⚕️ *This information is for reference only and does not replace professional medical advice. Please consult a doctor.*",
    },
    "high": {
        "vi": "⚠️ *QUAN TRỌNG: Thông tin này chỉ mang tính tham khảo. KHÔNG tự ý thay đổi liều thuốc hoặc phác đồ điều trị. Hãy tham khảo bác sĩ hoặc dược sĩ trước khi thực hiện bất kỳ thay đổi nào.*",
        "en": "⚠️ *IMPORTANT: This is for reference only. Do NOT change dosage or treatment on your own. Consult your doctor or pharmacist before making any changes.*",
    },
    "critical": {
        "vi": "🚨 *CẢNH BÁO: Đây là thông tin nhạy cảm về nhóm đối tượng đặc biệt. BẮT BUỘC phải tham khảo bác sĩ chuyên khoa trước khi áp dụng. Thông tin này KHÔNG thay thế khám và tư vấn trực tiếp.*",
        "en": "🚨 *WARNING: This involves sensitive information for special populations. You MUST consult a specialist before acting on this. This does NOT replace in-person consultation.*",
    }
}

def get_disclaimer(risk_level: str, language: str) -> str:
    return DISCLAIMERS.get(risk_level, DISCLAIMERS["medium"])[language]
```

---

## Verification cho Part 3

```bash
# Test emergency detection
python -c "
from src.safety_guard import SafetyGuard
sg = SafetyGuard('data/categories.json')

# Should be True (emergencies)
assert sg.is_emergency('tôi uống quá liều paracetamol')
assert sg.is_emergency('I have severe chest pain')
assert sg.is_emergency('muốn tự tử')

# Should be False (normal)
assert not sg.is_emergency('tác dụng phụ của warfarin')
assert not sg.is_emergency('what is diabetes')

# Scope check
assert sg.is_medical_scope('thuốc warfarin')[0] == True
assert sg.is_medical_scope('thời tiết hôm nay')[0] == False

print('All safety tests passed ✅')
"
```

**Xong Part 3 → chuyển sang [Part 4](file:///C:/Users/votru/.gemini/antigravity/brain/72dfbf26-8c80-4ebc-85dd-2bf3db7b1dca/plan_part4_intelligence.md)**
