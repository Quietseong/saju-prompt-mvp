# tests/test_prompt_manager.py

import os
import json
import tempfile
import pytest
from datetime import datetime
from src.prompt_manager import PromptTemplate, PromptManager

# 테스트용 템플릿 데이터
TEMPLATE_DATA = {
    "template_id": "test_template",
    "version": 1,
    "description": "테스트를 위한 템플릿",
    "template_text": "hello, {name}! your birthday is {birth_date} and gender is {gender}.",
    "metadata": {
        "language": "en",
        "type": "test"
    }
}

@pytest.fixture
def temp_dir():
    """테스트용 임시 디렉토리를 생성합니다."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def sample_template():
    """샘플 템플릿 객체를 생성합니다."""
    return PromptTemplate(
        template_id=TEMPLATE_DATA["template_id"],
        version=TEMPLATE_DATA["version"],
        template_text=TEMPLATE_DATA["template_text"],
        description=TEMPLATE_DATA["description"],
        metadata=TEMPLATE_DATA["metadata"]
    )

@pytest.fixture
def prompt_manager(temp_dir, sample_template):
    """테스트용 프롬프트 매니저를 생성합니다."""
    manager = PromptManager(templates_dir=temp_dir)
    manager.save_template(sample_template)
    return manager

def test_prompt_template_init():
    """PromptTemplate 초기화 테스트"""
    template = PromptTemplate(
        template_id="test",
        version=1,
        template_text="Hello, {name}!",
        description="Test template",
        metadata={"language": "en"}
    )
    
    assert template.template_id == "test"
    assert template.version == 1
    assert template.template_text == "Hello, {name}!"
    assert template.description == "Test template"
    assert template.metadata == {"language": "en"}
    assert "_variables" in template.__dict__
    assert "name" in template._variables

def test_extract_variables():
    """변수 추출 테스트"""
    template = PromptTemplate(
        template_id="test",
        version=1,
        template_text="Hello, {name}! Your age is {age} and your city is {city}."
    )
    
    variables = template.get_variables()
    assert len(variables) == 3
    assert "name" in variables
    assert "age" in variables
    assert "city" in variables

def test_format_template():
    """템플릿 포맷팅 테스트"""
    template = PromptTemplate(
        template_id="test",
        version=1,
        template_text="Hello, {name}! You are {age} years old."
    )
    
    result = template.format(name="John", age=30)
    assert result == "Hello, John! You are 30 years old."

def test_format_template_missing_variable():
    """누락된 변수가 있을 때 템플릿 포맷팅 테스트"""
    template = PromptTemplate(
        template_id="test",
        version=1,
        template_text="Hello, {name}! You are {age} years old."
    )
    
    with pytest.raises(KeyError) as excinfo:
        template.format(name="John")
    
    assert "Missing required variables: age" in str(excinfo.value)

def test_validate_template_values():
    """템플릿 값 검증 테스트"""
    template = PromptTemplate(
        template_id="test",
        version=1,
        template_text="Hello, {name}! Your age is {age}."
    )
    
    # 모든 변수가 있는 경우
    assert template.validate({"name": "John", "age": 30}) is True
    
    # 누락된 변수가 있는 경우
    with pytest.raises(ValueError) as excinfo:
        template.validate({"name": "John"})
    
    assert "Missing required variables: age" in str(excinfo.value)

def test_prompt_manager_load_templates(temp_dir):
    """템플릿 로드 테스트"""
    # 테스트 템플릿 파일 생성
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, "test_v1.json"), "w", encoding="utf-8") as f:
        json.dump(TEMPLATE_DATA, f)
    
    # 템플릿 로드
    manager = PromptManager(templates_dir=temp_dir)
    
    # 확인
    assert len(manager.templates) == 1
    assert "test_template_v1" in manager.templates
    assert manager.templates["test_template_v1"].template_id == "test_template"
    assert manager.templates["test_template_v1"].version == 1

def test_get_template(prompt_manager):
    """템플릿 가져오기 테스트"""
    template = prompt_manager.get_template("test_template")
    assert template.template_id == "test_template"
    assert template.version == 1
    
    # 존재하지 않는 템플릿
    with pytest.raises(ValueError):
        prompt_manager.get_template("nonexistent")

def test_get_template_by_version(prompt_manager, sample_template):
    """버전으로 템플릿 가져오기 테스트"""
    # 버전 2 추가
    new_template = PromptTemplate(
        template_id="test_template",
        version=2,
        template_text="새로운 템플릿 텍스트 {name}",
        description="버전 2 템플릿",
        metadata={"language": "en", "type": "updated"}
    )
    prompt_manager.save_template(new_template)
    
    # 특정 버전 가져오기
    v1 = prompt_manager.get_template("test_template", version=1)
    assert v1.version == 1
    
    # 최신 버전 가져오기
    latest = prompt_manager.get_template("test_template")
    assert latest.version == 2

def test_list_templates(prompt_manager, sample_template):
    """템플릿 목록 조회 테스트"""
    # 추가 템플릿 저장
    another_template = PromptTemplate(
        template_id="another",
        version=1,
        template_text="다른 템플릿 {var}",
        metadata={"language": "en"}
    )
    prompt_manager.save_template(another_template)
    
    # 모든 템플릿 목록
    all_templates = prompt_manager.list_templates()
    assert len(all_templates) == 2
    
    # 특정 ID로 필터링
    filtered = prompt_manager.list_templates(filter_id="test_template")
    assert len(filtered) == 1
    assert filtered[0]["id"] == "test_template"

def test_update_template(prompt_manager):
    """템플릿 업데이트 테스트"""
    # 업데이트
    updated = prompt_manager.update_template(
        template_id="test_template",
        version=1,
        new_text="업데이트된 템플릿 텍스트 {name}, {new_var}",
        new_description="업데이트된 설명",
        new_metadata={"language": "en", "updated": True}
    )
    
    # 확인
    assert updated.version == 2
    assert updated.template_text == "업데이트된 템플릿 텍스트 {name}, {new_var}"
    assert updated.description == "업데이트된 설명"
    assert updated.metadata.get("updated") is True
    
    # 변수 확인
    variables = updated.get_variables()
    assert "name" in variables
    assert "new_var" in variables

def test_create_template(prompt_manager):
    """새 템플릿 생성 테스트"""
    # 새 템플릿 생성
    new_template = prompt_manager.create_template(
        template_id="brand_new",
        template_text="완전히 새로운 템플릿 {var1}, {var2}",
        description="새 템플릿 설명",
        metadata={"language": "en", "type": "new"}
    )
    
    # 확인
    assert new_template.template_id == "brand_new"
    assert new_template.version == 1
    assert "var1" in new_template.get_variables()
    assert "var2" in new_template.get_variables()
    
    # 기존 ID로 생성 (버전 증가)
    another_version = prompt_manager.create_template(
        template_id="brand_new",
        template_text="또 다른 버전 {var3}",
        description="버전 2",
        metadata={"language": "en", "type": "newer"}
    )
    
    assert another_version.template_id == "brand_new"
    assert another_version.version == 2

def test_get_template_by_metadata(prompt_manager):
    """메타데이터로 템플릿 찾기 테스트"""
    # 추가 템플릿들 저장
    prompt_manager.create_template(
        template_id="meta_test1",
        template_text="메타데이터 테스트 1 {var}",
        metadata={"category": "A", "language": "en"}
    )
    
    prompt_manager.create_template(
        template_id="meta_test2",
        template_text="메타데이터 테스트 2 {var}",
        metadata={"category": "A", "language": "en"}
    )
    
    prompt_manager.create_template(
        template_id="meta_test3",
        template_text="메타데이터 테스트 3 {var}",
        metadata={"category": "B", "language": "en"}
    )
    
    # 메타데이터로 조회
    results = prompt_manager.get_template_by_metadata("category", "A")
    assert len(results) == 2
    assert all(t.metadata.get("category") == "A" for t in results)
    
    # 다른 메타데이터로 조회
    results = prompt_manager.get_template_by_metadata("language", "en")
    assert len(results) == 3  # 원래 템플릿 + 추가된 2개
    assert all(t.metadata.get("language") == "en" for t in results)