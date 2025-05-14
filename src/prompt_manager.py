import os
import json
from datetime import datetime
from typing import Dict, Optional

class PromptTemplate:
    def __init__(self, template_id: str, version: int, template_text: str, description: str = ""):
        self.template_id = template_id
        self.version = version
        self.template_text = template_text
        self.description = description
        self.created_at = datetime.now().isoformat()

    def format(self, **kwargs) -> str:
        """프롬프트에 변수를 삽입하여 완성된 텍스트를 반환합니다."""
        return self.template_text.format(**kwargs)

    def to_dict(self) -> Dict:
        return {
            "template_id": self.template_id,
            "version": self.version,
            "template_text": self.template_text,
            "description": self.description,
            "created_at": self.created_at
        }

class PromptManager:
    def __init__(self, templates_dir: str = "prompts"):
        self.templates_dir = templates_dir
        self.templates = {}
        self._load_templates()

    def _load_templates(self):
        """prompts 폴더에서 모든 템플릿 파일을 로드합니다."""
        os.makedirs(self.templates_dir, exist_ok=True)
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.templates_dir, filename), "r", encoding="utf-8") as f:
                    template_data = json.load(f)
                    template = PromptTemplate(
                        template_data["template_id"],
                        template_data["version"],
                        template_data["template_text"],
                        template_data.get("description", "")
                    )
                    key = f"{template.template_id}_v{template.version}"
                    self.templates[key] = template

    def get_template(self, template_id: str, version: Optional[int] = None) -> PromptTemplate:
        """특정 ID와 버전의 템플릿을 반환합니다. version이 None이면 최신 버전 반환."""
        if version is None:
            # 최신 버전 찾기
            versions = [t for t in self.templates.keys() if t.startswith(f"{template_id}_v")]
            if not versions:
                raise ValueError(f"No templates found for ID: {template_id}")
            key = max(versions, key=lambda x: int(x.split('_v')[1]))
            return self.templates[key]
        else:
            key = f"{template_id}_v{version}"
            if key not in self.templates:
                raise ValueError(f"Template not found: {key}")
            return self.templates[key]

    def save_template(self, template: PromptTemplate):
        """새 템플릿을 파일로 저장합니다."""
        key = f"{template.template_id}_v{template.version}"
        self.templates[key] = template
        filename = f"{template.template_id}_v{template.version}.json"
        with open(os.path.join(self.templates_dir, filename), "w", encoding="utf-8") as f:
            json.dump(template.to_dict(), f, indent=2, ensure_ascii=False)