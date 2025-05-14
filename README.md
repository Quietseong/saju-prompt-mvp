[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Quietseong/saju-prompt-mvp)

# Overall System
![image](https://github.com/user-attachments/assets/fd1af049-a307-4e30-9ea9-6e6631281417)

# 간단 사용방법
1. notebook>saju_pe_test.ipynb 실행
2. 필수 패키지 재설치 (!pip install ...)
3. 허깅페이스 토큰 입력(Meta-Llama-3-8B-Instruct -> huggingface gated model token)
4. 계속 실행 후 그라디오 데모 사용

# Demo(Gradio)
https://github.com/user-attachments/assets/9bc369cc-c4b3-4e48-9fba-de88c830cd5e

--------
# Product Requirements Document (PRD)

## Product Name

"SajuMate Prompt MVP" – Saju Fortune Prompt Engineering MVP

## Overview

SajuMate Prompt MVP는 Saju(사주) 운세 해석을 위한 프롬프트 엔지니어링 실험 및 검증을 목적으로 하는 최소 기능 제품(MVP)입니다. 프론트엔드/백엔드 구현 없이, 입력값(생년월일, 시간, 장소, 성별)과 프롬프트 템플릿, GPT API 연동, 결과 분석 및 반복 개선에 집중합니다.

## Objective

- Saju 운세 해석에 적합한 프롬프트 구조를 설계하고, 다양한 입력값에 대해 일관적이고 신뢰할 수 있는 결과를 생성하는지 검증
- 향후 모바일 앱/웹 연동을 위한 프롬프트 및 입력/출력 포맷 표준화

## Target Users

- 프롬프트 엔지니어, ML/AI 개발자, SajuMate 서비스 기획자
- SajuMate의 핵심 가치(운세 해석)를 빠르게 검증하고자 하는 내부 팀

## Platform

- 로컬/클라우드 환경에서 실행 가능한 Python 스크립트 기반 실험
- OpenAI GPT API (gpt-4, gpt-4-turbo)

## Key Features

1. **입력값 구조 및 검증**
   - 생년월일(YYYY-MM-DD), 시간(HH:MM, AM/PM 또는 24시간), 장소(타임존 변환용), 성별
   - 입력값 예시 데이터셋(3~5개 페르소나)

2. **프롬프트 템플릿 설계 및 버전 관리**
   - 요약 카드용 프롬프트(성격, 사랑, 직업, 건강 점수)
   - 상세 해석용 프롬프트(성격 설명, 올해 운세)
   - 피드백 요청 프롬프트(Thumbs up/down)
   - 프롬프트 버전별 관리 및 실험

3. **실험 자동화 및 결과 기록**
   - 입력값/프롬프트 조합별 GPT API 호출 자동화
   - 결과(입력, 프롬프트, 응답, 피드백) 기록 (CSV/JSON/MLflow 등)

4. **결과 분석 및 프롬프트 개선**
   - 응답 품질 평가(점수화, 자연스러움, 일관성 등)
   - 개선 포인트 도출 및 반복 실험

5. **문서화 및 확장성 고려**
   - 프롬프트 설계 의도, 실험 결과, 개선 내역 문서화
   - 향후 프론트/백엔드 연동, 다국어, 추가 기능 확장 고려

## Functional Requirements

- 입력값 유효성 검사 및 예시 데이터 생성
- 프롬프트 템플릿 파일/모듈화 및 버전 관리
- GPT API 연동 및 비동기 실험 자동화
- 결과 기록 및 분석(간단한 통계/시각화 포함)
- 반복 개선을 위한 실험 로그 관리

## Non-Functional Requirements

- 실험 1회당 2초 이내 응답(비동기 처리)
- 실험 결과 재현성 확보(입력/프롬프트/응답 로그)
- 개인정보 미수집, GDPR 준수

## MVP Scope

- 입력값 구조 및 예시 데이터셋
- 프롬프트 템플릿 2종(요약/상세)
- GPT API 연동 및 실험 자동화 스크립트
- 결과 기록 및 간단한 분석
- 반복 개선 프로세스 문서화

## Success Metrics

- 프롬프트별 응답 품질(정성/정량 평가)
- 실험 반복을 통한 프롬프트 개선 횟수
- 향후 서비스 연동을 위한 입력/출력 포맷 표준화 여부

## Timeline
- 시작: 05.13(화)
- ~~요구사항/입력값/프롬프트 설계: ~ 05.15(수)~~
- ~~실험 자동화/결과 기록: ~ 05.17(금)~~ -> 05.14(수)
- 결과 분석/프롬프트 개선: ~ 05.19(일)
- 문서화 및 확장성 검토: ~ 05.20(월)
- 총 1주일 내 MVP 완성
