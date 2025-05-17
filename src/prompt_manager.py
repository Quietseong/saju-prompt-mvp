import os
import json
import re
from datetime import datetime
from typing import Dict, Optional, List, Any, Set

class PromptTemplate:
    def __init__(self, template_id: str, version: int, template_text: str, description: str = "", 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        프롬프트 템플릿 객체를 초기화합니다.
        
        Args:
            template_id: 템플릿 고유 식별자
            version: 템플릿 버전
            template_text: 템플릿 텍스트 (포맷 문자열)
            description: 템플릿 설명
            metadata: 추가 메타데이터 (언어, 문서 타입 등)
        """
        self.template_id = template_id
        self.version = version
        self.template_text = template_text
        self.description = description
        self.created_at = datetime.now().isoformat()
        self.metadata = metadata or {}
        
        # 템플릿 변수 캐싱
        self._variables = self._extract_variables()

    def format(self, **kwargs) -> str:
        """
        프롬프트에 변수를 삽입하여 완성된 텍스트를 반환합니다.
        
        Args:
            **kwargs: 템플릿에 삽입할 변수와 값
            
        Returns:
            완성된 프롬프트 텍스트
            
        Raises:
            KeyError: 필요한 변수가 제공되지 않았을 때
        """
        # 필요한 모든 변수가 제공되었는지 확인
        missing_vars = [var for var in self._variables if var not in kwargs]
        if missing_vars:
            raise KeyError(f"Missing required variables: {', '.join(missing_vars)}")
            
        return self.template_text.format(**kwargs)
    
    def _extract_variables(self) -> Set[str]:
        """
        템플릿 텍스트에서 모든 변수 이름을 추출합니다.
        
        Returns:
            템플릿에 사용된 변수 이름 집합
        """
        pattern = r'\{([^{}]*)\}'
        return set(re.findall(pattern, self.template_text))
    
    def get_variables(self) -> Set[str]:
        """
        템플릿에 필요한 모든 변수 이름을 반환합니다.
        
        Returns:
            템플릿에 사용된 변수 이름 집합
        """
        return self._variables
        
    def validate(self, values: Dict[str, Any]) -> bool:
        """
        제공된 값들이 템플릿의 모든 변수를 충족하는지 검증합니다.
        
        Args:
            values: 검증할 변수 이름과 값
            
        Returns:
            모든 필수 변수가 제공되었으면 True
            
        Raises:
            ValueError: 필수 변수가 누락되었을 때
        """
        missing_vars = [var for var in self._variables if var not in values]
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        템플릿을 딕셔너리로 변환합니다.
        
        Returns:
            템플릿 속성을 포함한 딕셔너리
        """
        return {
            "template_id": self.template_id,
            "version": self.version,
            "template_text": self.template_text,
            "description": self.description,
            "created_at": self.created_at,
            "metadata": self.metadata
        }

class PromptManager:
    def __init__(self, templates_dir: str = "prompts"):
        """
        프롬프트 매니저를 초기화합니다.
        
        Args:
            templates_dir: 템플릿 파일이 저장된 디렉토리 경로
        """
        self.templates_dir = templates_dir
        self.templates = {}
        self._load_templates()

    def _load_templates(self):
        """prompts 폴더에서 모든 템플릿 파일을 로드합니다."""
        os.makedirs(self.templates_dir, exist_ok=True)
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.templates_dir, filename), "r", encoding="utf-8") as f:
                    try:
                        template_data = json.load(f)
                        template = PromptTemplate(
                            template_data["template_id"],
                            template_data["version"],
                            template_data["template_text"],
                            template_data.get("description", ""),
                            template_data.get("metadata", {})
                        )
                        key = f"{template.template_id}_v{template.version}"
                        self.templates[key] = template
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error loading template from {filename}: {e}")

    def get_template(self, template_id: str, version: Optional[int] = None) -> PromptTemplate:
        """
        특정 ID와 버전의 템플릿을 반환합니다. version이 None이면 최신 버전 반환.
        
        Args:
            template_id: 템플릿 ID
            version: 템플릿 버전 (None이면 최신 버전)
            
        Returns:
            템플릿 객체
            
        Raises:
            ValueError: 템플릿을 찾을 수 없을 때
        """
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
    
    def list_templates(self, filter_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        사용 가능한 모든 템플릿 목록을 반환합니다.
        
        Args:
            filter_id: 특정 ID의 템플릿만 필터링 (옵션)
            
        Returns:
            템플릿 정보 목록 (딕셔너리)
        """
        result = []
        for key, template in self.templates.items():
            if filter_id is None or key.startswith(f"{filter_id}_v"):
                result.append({
                    "key": key,
                    "id": template.template_id,
                    "version": template.version,
                    "description": template.description,
                    "created_at": template.created_at,
                    "metadata": template.metadata,
                    "variables": list(template.get_variables())
                })
        return result

    def save_template(self, template: PromptTemplate):
        """
        새 템플릿을 파일로 저장합니다.
        
        Args:
            template: 저장할 템플릿 객체
        """
        key = f"{template.template_id}_v{template.version}"
        self.templates[key] = template
        filename = f"{template.template_id}_v{template.version}.json"
        with open(os.path.join(self.templates_dir, filename), "w", encoding="utf-8") as f:
            json.dump(template.to_dict(), f, indent=2, ensure_ascii=False)
    
    def update_template(self, template_id: str, version: int, 
                        new_text: Optional[str] = None, 
                        new_description: Optional[str] = None,
                        new_metadata: Optional[Dict[str, Any]] = None) -> PromptTemplate:
        """
        기존 템플릿을 새 버전으로 업데이트합니다.
        
        Args:
            template_id: 업데이트할 템플릿 ID
            version: 업데이트할 템플릿 버전
            new_text: 새 템플릿 텍스트 (None이면 기존 텍스트 유지)
            new_description: 새 설명 (None이면 기존 설명 유지)
            new_metadata: 새 메타데이터 (None이면 기존 메타데이터 유지)
            
        Returns:
            업데이트된 템플릿 객체
            
        Raises:
            ValueError: 템플릿을 찾을 수 없을 때
        """
        # 기존 템플릿 찾기
        key = f"{template_id}_v{version}"
        if key not in self.templates:
            raise ValueError(f"Template not found: {key}")
        
        old_template = self.templates[key]
        
        # 새 버전 생성
        new_version = version + 1
        new_template = PromptTemplate(
            template_id=template_id,
            version=new_version,
            template_text=new_text if new_text is not None else old_template.template_text,
            description=new_description if new_description is not None else old_template.description,
            metadata=new_metadata if new_metadata is not None else old_template.metadata.copy()
        )
        
        # 저장
        self.save_template(new_template)
        return new_template
    
    def create_template(self, template_id: str, template_text: str, 
                        description: str = "", metadata: Optional[Dict[str, Any]] = None) -> PromptTemplate:
        """
        새 템플릿을 생성합니다. 이미 존재하는 ID인 경우 새 버전을 생성합니다.
        
        Args:
            template_id: 템플릿 ID
            template_text: 템플릿 텍스트
            description: 템플릿 설명
            metadata: 메타데이터
            
        Returns:
            생성된 템플릿 객체
        """
        # 기존 버전 확인
        versions = [int(t.split('_v')[1]) for t in self.templates.keys() 
                    if t.startswith(f"{template_id}_v")]
        
        if versions:
            version = max(versions) + 1
        else:
            version = 1
            
        template = PromptTemplate(
            template_id=template_id,
            version=version,
            template_text=template_text,
            description=description,
            metadata=metadata
        )
        
        self.save_template(template)
        return template
    
    def get_template_by_metadata(self, key: str, value: Any) -> List[PromptTemplate]:
        """
        메타데이터 값으로 템플릿을 검색합니다.
        
        Args:
            key: 검색할 메타데이터 키
            value: 찾을 값
            
        Returns:
            조건에 맞는 템플릿 목록
        """
        results = []
        for template in self.templates.values():
            if key in template.metadata and template.metadata[key] == value:
                results.append(template)
        return results
        
    def generate_prompt(self, template_key: str, persona_data) -> str:
        """
        템플릿과 페르소나 데이터를 사용하여 완성된 프롬프트를 생성합니다.
        
        Args:
            template_key: 템플릿 키 (예: "summary_v1_ko")
            persona_data: 페르소나 데이터 (PersonaData 객체)
            
        Returns:
            완성된 프롬프트 문자열
            
        Raises:
            ValueError: 템플릿을 찾을 수 없거나 데이터 형식이 맞지 않을 때
        """
        # 템플릿 키 파싱
        parts = template_key.split('_')
        if len(parts) < 2:
            raise ValueError(f"Invalid template key format: {template_key}")
            
        # 언어 코드가 있는 경우 (예: summary_v1_ko)
        if len(parts) >= 3 and parts[-1] in ['ko', 'en']:
            template_id = '_'.join(parts[:-1])
            language = parts[-1]
        else:
            template_id = template_key
            language = 'en'  # 기본값
            
        try:
            # 템플릿 가져오기
            template = self.get_template(template_id)
            
            # 페르소나 데이터 준비
            birth_info = persona_data.birth_info
            
            # 간지/사주 데이터는 나중에 추가할 수 있음
            # 여기서는 기본 정보만 포맷
            
            format_data = {
                "name": persona_data.name,
                "birth_year": birth_info.year,
                "birth_month": birth_info.month,
                "birth_day": birth_info.day,
                "birth_hour": birth_info.hour,
                "birth_minute": birth_info.minute,
                "gender": persona_data.gender,
                "language": language
            }
            
            # 추가 정보가 있는 경우 포함
            if persona_data.location:
                format_data["location"] = persona_data.location
                
            if persona_data.additional_info:
                for key, value in persona_data.additional_info.items():
                    format_data[key] = value
            
            # 프롬프트 생성
            return template.format(**format_data)
            
        except Exception as e:
            raise ValueError(f"Error generating prompt: {str(e)}")