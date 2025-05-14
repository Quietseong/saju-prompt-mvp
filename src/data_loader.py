import json
from typing import List, Dict

def load_personas(path: str = "data/sample_personas.json") -> List[Dict]:
    """샘플 페르소나 데이터를 로드합니다."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)