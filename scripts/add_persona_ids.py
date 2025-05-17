#!/usr/bin/env python
"""
샘플 페르소나 데이터에 고유 ID를 추가하는 스크립트

sample_personas.json 파일을 읽어 각 페르소나에 고유 ID를 추가합니다.
"""

import os
import json
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리 설정
project_root = Path(__file__).parent.parent
sample_file = project_root / "data" / "sample_personas.json"
backup_file = project_root / "data" / "sample_personas_backup.json"

def add_ids_to_personas():
    """페르소나 데이터에 ID를 추가합니다."""
    
    print(f"샘플 페르소나 파일 '{sample_file}' 처리 중...")
    
    # 파일이 존재하는지 확인
    if not sample_file.exists():
        print(f"오류: {sample_file} 파일이 존재하지 않습니다.")
        return False
    
    # 백업 파일 생성
    try:
        import shutil
        shutil.copy2(sample_file, backup_file)
        print(f"백업 파일이 생성되었습니다: {backup_file}")
    except Exception as e:
        print(f"백업 파일 생성 실패: {e}")
        return False
    
    # 데이터 로드
    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"파일 로드 실패: {e}")
        return False
    
    # 데이터 구조 확인 및 처리
    if isinstance(data, list):
        personas = data
        container = None
    elif isinstance(data, dict) and "personas" in data:
        personas = data["personas"]
        container = data
    else:
        print(f"지원되지 않는 데이터 형식: {type(data)}")
        return False
    
    # ID 추가
    modified = False
    persona_count = len(personas)
    print(f"총 {persona_count}개의 페르소나 데이터가 발견되었습니다.")
    
    for i, persona in enumerate(personas):
        if "id" not in persona:
            persona["id"] = i + 1
            modified = True
        elif not isinstance(persona["id"], (int, str)):
            persona["id"] = i + 1
            modified = True
    
    if not modified:
        print("모든 페르소나에 이미 ID가 있습니다. 변경 사항 없음.")
        return True
    
    # 변경된 데이터 저장
    try:
        with open(sample_file, "w", encoding="utf-8") as f:
            if container:
                json.dump(container, f, ensure_ascii=False, indent=2)
            else:
                json.dump(personas, f, ensure_ascii=False, indent=2)
        print(f"ID가 추가된 페르소나 데이터가 저장되었습니다: {sample_file}")
        return True
    except Exception as e:
        print(f"파일 저장 실패: {e}")
        print(f"백업 파일 {backup_file}에서 복원할 수 있습니다.")
        return False

if __name__ == "__main__":
    success = add_ids_to_personas()
    sys.exit(0 if success else 1) 